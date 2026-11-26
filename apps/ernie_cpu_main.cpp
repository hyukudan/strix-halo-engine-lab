#include "core/bf16_reference.hpp"
#include "core/sha256.hpp"
#include "core/streamed_linear_reference.hpp"
#include "moe_core/reference.hpp"
#include "strix/ernie/calibration.hpp"
#include "strix/ernie/checkpoint.hpp"
#include "strix/ernie/forward.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/ernie_chat.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
namespace fs = std::filesystem;
namespace ernie = strix::ernie;
namespace core = strix::core;
using strix::serve::Json;
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMiB = std::size_t{1} << 20;
static_assert(std::endian::native == std::endian::little, "CPU trace files use little endian");

struct Options {
    unsigned steps = 32, threads = 1, cache_mib = 0;
    std::optional<unsigned> prompt;
    std::string activation = "soft-clip";
    fs::path capture, packed, replay;
    bool bf16 = false, fixed_gain = false;
};

unsigned number(std::string_view text) {
    unsigned value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned integer");
    return value;
}

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 4; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--bf16-projections") { opts.bf16 = true; continue; }
        if (arg == "--multiply-fixed-gain") { opts.fixed_gain = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing option value");
        if (arg == "--generate") opts.steps = number(argv[i]);
        else if (arg == "--threads") opts.threads = number(argv[i]);
        else if (arg == "--cache-mib") opts.cache_mib = number(argv[i]);
        else if (arg == "--prompt-index") opts.prompt = number(argv[i]);
        else if (arg == "--activation") opts.activation = argv[i];
        else if (arg == "--capture") opts.capture = argv[i];
        else if (arg == "--pack") opts.packed = argv[i];
        else if (arg == "--replay") opts.replay = argv[i];
        else throw std::invalid_argument("unrecognized option " + std::string(arg));
    }
    if (!opts.steps || opts.steps > 4096 || !opts.threads || opts.threads > 16 || opts.cache_mib > 32768)
        throw std::invalid_argument("expected 1-4096 output tokens, 1-16 threads and 0-32768 MiB of decoded cache");
    if (!opts.capture.empty() && (!opts.packed.empty() || !opts.replay.empty()))
        throw std::invalid_argument("capture requires a free-running source-checkpoint run");
    return opts;
}

ernie::ReferenceActivation activation(std::string_view name) {
    using enum ernie::CandidateActivation;
    if (name == "soft-clip") return ernie::candidate_activation(soft_clip);
    if (name == "hard-clip") return ernie::candidate_activation(hard_clip);
    if (name == "swiglu") return ernie::candidate_activation(swiglu);
    throw std::invalid_argument("unknown activation candidate");
}

std::ofstream output_file(const fs::path& path) {
    std::ofstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(path, std::ios::binary);
    return file;
}

template <class T>
void write_row(std::ostream& file, std::span<const T> values) {
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
}

std::ofstream trace_file(const fs::path& path, std::string_view header) {
    auto file = output_file(path);
    file << header << '\n' << std::setprecision(9);
    return file;
}

std::vector<std::uint32_t> top_logits(std::span<const float> logits) {
    if (logits.size() < 20 || !std::ranges::all_of(logits, [](float z) { return std::isfinite(z); }))
        throw std::runtime_error("nonfinite logits or invalid vocabulary");
    std::vector<std::uint32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0U);
    std::partial_sort(ids.begin(), ids.begin() + 20, ids.end(), [&](auto a, auto b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    ids.resize(20);
    return ids;
}

float rms(std::span<const float> row) {
    double squares = 0;
    for (float v : row) squares += double{v} * v;
    return static_cast<float>(std::sqrt(squares / static_cast<double>(row.size())));
}

struct RunFiles {
    std::ofstream scores, generated, hidden, routes;

    explicit RunFiles(const fs::path& folder)
        : scores(output_file(folder / "raw-logits.f32")), generated(output_file(folder / "generated.u32")),
          hidden(trace_file(folder / "hidden.tsv", "position\tlayer\tattention_rms\trms\tattention_sha256\tsha256")),
          routes(trace_file(folder / "routes.tsv", "position\tlayer\texpert\tweight")) {}

    void layer(std::size_t position, unsigned layer_id, const ernie::LayerResult& result) {
        const auto attention = std::span(result.attention_streams);
        const auto streams = std::span(result.streams);
        const float a = rms(attention), h = rms(streams);
        if (!std::isfinite(a) || !std::isfinite(h)) throw std::runtime_error("nonfinite layer hidden RMS");
        hidden << position << '\t' << layer_id << '\t' << a << '\t' << h << '\t'
               << core::sha256_hex(attention.data(), attention.size_bytes()) << '\t'
               << core::sha256_hex(streams.data(), streams.size_bytes()) << '\n';
        for (std::size_t slot = 0; slot < result.experts.size(); ++slot)
            routes << position << '\t' << layer_id << '\t' << result.experts[slot] << '\t'
                   << result.expert_weights[slot] << '\n';
    }
};

Json load_json(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open prompt JSON");
    return strix::serve::parse_json_unique(std::string(std::istreambuf_iterator<char>(file), {}));
}

std::ifstream input_file(const fs::path& path) {
    std::ifstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(path, std::ios::binary);
    return file;
}

struct Replay {
    std::ifstream ids, logits;
    std::ofstream comparison;
    std::vector<float> row;
    bool matched = true;
    double kl = 0.0;
    unsigned agreements = 0;

    Replay(const fs::path& previous, const fs::path& folder, std::span<const std::uint32_t> prompt, unsigned vocab)
        : ids(input_file(previous / "generated.u32")), logits(input_file(previous / "raw-logits.f32")),
          comparison(trace_file(folder / "comparison.tsv", "step\tkl\treference_id\tcandidate_id\tsame_bits")),
          row(vocab) {
        auto original = input_file(previous / "prompt.u32");
        std::vector<std::uint32_t> original_ids(prompt.size());
        original.read(reinterpret_cast<char*>(original_ids.data()), static_cast<std::streamsize>(prompt.size_bytes()));
        if (!std::ranges::equal(original_ids, prompt) ||
            fs::file_size(previous / "prompt.u32") != prompt.size_bytes())
            throw std::invalid_argument("replay prompt differs from source");
    }

    std::uint32_t next(unsigned step, std::span<const float> current) {
        std::uint32_t id{};
        ids.read(reinterpret_cast<char*>(&id), sizeof(id));
        logits.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(current.size_bytes()));
        const auto cmp = ernie::compare_logits(row, current);
        if (cmp.reference_top1 != id) throw std::invalid_argument("replay source was not greedy");
        const auto same = std::ranges::equal(row, current, [](float a, float b) {
            return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        });
        comparison << step << '\t' << cmp.kl_reference_to_candidate << '\t' << id << '\t'
                   << cmp.candidate_top1 << '\t' << same << '\n';
        matched &= same;
        agreements += cmp.top1_agrees() ? 1U : 0U;
        kl += cmp.kl_reference_to_candidate;
        return id;
    }
};

void run_prompt(const Json& request, unsigned index, const strix::serve::BpeTokenizer& tokenizer,
                const Options& opts, const fs::path& folder, const ernie::ReferenceRead& read,
                const ernie::ReferenceProject& project, ernie::ExpertCalibration* calibration) {
    const auto* messages = request.find("messages");
    if (!messages) throw std::invalid_argument("prompt requires messages");
    strix::serve::ChatTemplateInput chat{};
    chat.messages = *messages;
    if (const auto* tools = request.find("tools")) chat.tools = *tools;
    chat.enable_thinking = false;
    const auto prompt = strix::serve::render_ernie_chat(chat);
    const auto ids = prompt.encode(tokenizer);
    if (ids.empty() || ids.size() > 4096 || opts.steps > 4096 - ids.size())
        throw std::invalid_argument("CPU runner requires prompt plus generation to fit context 4096");
    fs::create_directory(folder);
    output_file(folder / "request.json") << request.dump() << '\n';
    output_file(folder / "prompt.txt") << prompt.text();
    auto prompt_out = output_file(folder / "prompt.u32");
    write_row(prompt_out, std::span(ids));
    prompt_out.close();
    std::cerr << "prompt " << index << " prefill " << ids.size() << " rows\n";

    auto shape = ernie::checkpoint_reference_shape();
    if (!opts.fixed_gain) shape.route.scale = 1.0F;
    ernie::ExpertObserver expert_observer;
    std::size_t window = 0;
    if (calibration) {
        window = calibration->add_window(ids);
        expert_observer = [&](auto layer, auto expert, auto value, bool down) {
            calibration->observe(window, layer, expert, down ? strix::moe::ExpertProjection::down :
                                 strix::moe::ExpertProjection::gate_up, value);
        };
    }
    ernie::CpuReference reference(shape, read, activation(opts.activation), project, expert_observer);
    RunFiles files(folder);
    const auto observer = [&](auto pos, auto layer, const auto& result) {
        files.layer(pos, layer, result);
        if (index == 0 && pos + 1 == ids.size())
            std::cerr << "prefill layer " << layer << '\n';
    };
    std::vector<std::uint32_t> generated;
    generated.reserve(opts.steps);
    std::optional<Replay> replay;
    if (!opts.replay.empty())
        replay.emplace(opts.replay / folder.filename(), folder, ids, shape.vocab);
    const auto started = Clock::now();
    auto logits = reference.append(ids, observer);
    const auto prefill = Clock::now();
    for (unsigned step = 0; step < opts.steps; ++step) {
        const auto top = top_logits(logits);
        if (step == 0) {
            auto first = trace_file(folder / "first-token-top20.tsv", "id\traw_logit");
            for (auto id : top) first << id << '\t' << logits[id] << '\n';
        }
        generated.push_back(replay ? replay->next(step, logits) : top.front());
        write_row(files.generated, std::span<const std::uint32_t>(generated).last(1));
        write_row(files.scores, std::span<const float>(logits));
        files.generated.flush();
        files.scores.flush();
        output_file(folder / "partial-response.txt") << tokenizer.decode(generated, false);
        std::cerr << "prompt " << index << " generated " << generated.size() << " id " << generated.back() << '\n';
        if (std::ranges::find(ernie::data::tokens::generation_eos_ids, generated.back()) !=
            std::end(ernie::data::tokens::generation_eos_ids) || step + 1 == opts.steps) break;
        logits = reference.append(std::span<const std::uint32_t>(generated).last(1), observer);
    }
    if (calibration)
        calibration->extend_window(window, std::span<const std::uint32_t>(generated).first(generated.size() - 1));
    const auto end = Clock::now();
    const auto prefill_seconds = std::chrono::duration<double>(prefill - started).count();
    const auto after_first = std::chrono::duration<double>(end - prefill).count();
    const auto rate = generated.size() > 1 ? static_cast<double>(generated.size() - 1) / after_first : 0.0;
    const auto response = tokenizer.decode(generated, false);
    output_file(folder / "response.txt") << response;
    auto receipt = Json(Json::Object{
        {"identity", "candidate-raw-head"}, {"activation", opts.activation},
        {"bf16_projections", opts.bf16}, {"multiply_fixed_gain", opts.fixed_gain},
        {"pack", !opts.packed.empty()}, {"teacher_forced", replay.has_value()},
        {"threads", static_cast<std::uint64_t>(opts.threads)}, {"cache_mib", static_cast<std::uint64_t>(opts.cache_mib)},
        {"prompt_tokens", static_cast<std::uint64_t>(ids.size())},
        {"generated_tokens", static_cast<std::uint64_t>(generated.size())},
        {"prefill_seconds", prefill_seconds}, {"decode_tokens_per_second", rate}});
    if (replay) {
        const auto count = static_cast<double>(generated.size());
        receipt.set("KL", replay->kl / count);
        receipt.set("top1_agreement", replay->agreements / count);
        receipt.set("bit_equal", replay->matched);
    }
    output_file(folder / "run.json") << receipt.dump() << '\n';
    std::cout << "prompt " << index << " prefill " << prefill_seconds << " s, decode " << rate << " tokens/s\n"
              << response << '\n';
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 4) {
        std::cerr << "usage: strix-ernie-cpu CHECKPOINT PROMPTS.json NEW_TRACE_DIR"
                     " [--generate N] [--prompt-index N] [--activation soft-clip|hard-clip|swiglu]"
                     " [--bf16-projections] [--multiply-fixed-gain] [--threads N] [--cache-mib N]"
                     " [--capture MODEL_DESCRIPTION] [--pack SIDECAR] [--replay TRACE_DIR]\n";
        return 2;
    }
    const fs::path checkpoint = argv[1], folder = argv[3];
    const auto opts = parse_options(argc, argv);
    (void)activation(opts.activation);
    const auto prompts = load_json(argv[2]).as_array();
    if (prompts.empty() || (opts.prompt && *opts.prompt >= prompts.size()))
        throw std::invalid_argument("missing or invalid prompt index");
    if (!fs::create_directory(folder)) throw std::invalid_argument("trace directory must be new");

    const auto tokenizer = strix::serve::load_ernie_tokenizer((checkpoint / "tokenizer.json").string());
    ernie::Inventory inventory;
    const auto inspected = ernie::inspect_weights(checkpoint, inventory);
    if (!inspected) throw std::runtime_error(inspected.detail);
    if (!inventory.has_manifest || !inventory.missing_shards.empty() || inventory.missing_known_tensors != 0)
        throw std::runtime_error("checkpoint inventory is incomplete");
    ernie::CheckpointWeights weights(std::move(inventory));
    std::optional<ernie::ExpertCalibration> calibration;
    if (!opts.capture.empty()) calibration.emplace();
    std::optional<ernie::PackedExpertOverrides> packed;
    if (!opts.packed.empty()) packed.emplace(checkpoint, opts.packed);
    core::HostLaneConfig lane_config;
    lane_config.critical_threads = 0;
    lane_config.latency_threads = 0;
    lane_config.background_threads = opts.threads > 1 ? opts.threads : 0;
    core::HostLane lane(lane_config);
    const ernie::CheckpointWeights::RowExecutor execute = [&](auto size, const auto& range) {
        core::reference_row_ranges(lane, opts.threads, size, range);
    };
    const ernie::ReferenceRead read = [&](auto name, auto row, auto offset) {
        weights.read(name, row, offset, execute);
    };
    core::ReferenceMatrixCache cache(kMiB * opts.cache_mib);
    const ernie::ReferenceProject project = [&](const auto& name, auto in, auto out, unsigned groups) {
        if (packed && groups == 1 && packed->project(name, in, out)) {
            // The same projection-store policy is applied below.
        } else if (name == "lm_head.weight") {
            constexpr std::size_t kRows = 4096;
            std::vector<float> block(kRows * in.size());
            for (std::size_t first = 0; first < out.size(); first += kRows) {
                const auto count = std::min(kRows, out.size() - first);
                const auto matrix = std::span(block).first(count * in.size());
                read(name, matrix, first * in.size());
                core::parallel_linear_reference(lane, opts.threads, matrix, in, out.subspan(first, count));
            }
        } else {
            const auto matrix = cache.get(name, out.size() * (in.size() / groups), read);
            core::parallel_linear_reference(lane, opts.threads, matrix, in, out, groups);
        }
        if (opts.bf16 && name.find(".linear_o_group_proj") == std::string::npos) core::round_bf16_reference(out);
    };
    const unsigned first = opts.prompt.value_or(0);
    const unsigned last = opts.prompt ? first + 1 : static_cast<unsigned>(prompts.size());
    for (unsigned index = first; index < last; ++index)
        run_prompt(prompts[index], index, *tokenizer, opts, folder / ("prompt-" + std::to_string(index)),
                   read, project, calibration ? &*calibration : nullptr);
    if (calibration) {
        const auto dir = folder / "calibration";
        calibration->write(dir, opts.capture);
        constexpr unsigned selected_layers[] = {1, 7, 14};
        calibration->write_slice_plan(dir / "sp2.tsv", selected_layers, 2, "sp2");
        calibration->write_slice_plan(dir / "sp3.tsv", selected_layers, 2, "sp3");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
