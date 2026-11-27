#include "core/bf16_reference.hpp"
#include "core/multimax_reference.hpp"
#include "core/sha256.hpp"
#include "core/streamed_linear_reference.hpp"
#include "cpu/core/ordered_linear_reference.hpp"
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
#include <map>
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
    unsigned min_new_tokens = 0;
    unsigned start_layer = 0, end_layer = ernie::kLayerCount, limit = 128;
    std::optional<unsigned> prompt;
    std::string activation = "soft-clip";
    std::string multimax = "identity";
    std::vector<std::string> variants;
    ernie::HostContracts contracts;
    ernie::CheckpointReadPolicy decode;
    fs::path capture, packed, replay, resume;
    bool bf16 = false, fixed_gain = false, oracle = false, bos = false, save_layers = false, profile = false;
    bool simd = false, parallel_mla = false, dump_logits = false;
};

void set_variant(Options& opts, std::string_view name) {
    auto& c = opts.contracts;
    using ernie::RouterContract;
    using ernie::ResidualContract;
    using ernie::VhaContract;
    using ernie::KeyNormContract;
    if (name == "router-unit") c.router = RouterContract::unit_gain;
    else if (name == "router-gain-before-norm") c.router = RouterContract::gain_before_normalize;
    else if (name == "router-summed-scores") c.router = RouterContract::summed_scores;
    else if (name == "norm-routed-output") c.norm_routed_output = true;
    else if (name == "checkpoint-attention") c.checkpoint_attention = true;
    else if (name == "checkpoint-hc") c.checkpoint_hc = true;
    else if (name == "checkpoint-stores") c.checkpoint_stores = true;
    else if (name == "router-swap") c.router = RouterContract::swapped_planes;
    else if (name == "router-one-plane") c.router = RouterContract::single_plane;
    else if (name == "mla-global-kv") c.global_kv_split = true;
    else if (name == "mla-gate-first") c.gate_before_vha = true;
    else if (name == "vha-swap") c.vha = VhaContract::swapped;
    else if (name == "vha-off") c.vha = VhaContract::off;
    else if (name == "vha-no-residual") c.vha = VhaContract::non_residual;
    else if (name == "mhc-transpose") c.residual = ResidualContract::transpose;
    else if (name == "mhc-half-post") c.residual = ResidualContract::half_post;
    else if (name == "mhc-interleave") c.residual = ResidualContract::feature_interleave;
    else if (name == "mhc-bf16") c.residual = ResidualContract::bf16;
    else if (name == "hca-v-no-head-norm") c.key_norm = KeyNormContract::value_before_head_norm;
    else if (name == "kv-no-head-norm") c.key_norm = KeyNormContract::weighted_only;
    else if (name == "qk-no-head-norm") c.key_norm = KeyNormContract::none;
    else if (name == "rope-position-one") c.position_offset = 1;
    else if (name == "rope-base-160k") c.rope_base = 160000;
    else if (name == "hca-no-inverse-rope") c.no_hca_inverse_rope = true;
    else if (name == "rms-eps-1e-5") c.rms_epsilon = 1e-5F;
    else if (name == "mxfp4-nibble-swap") opts.decode.high_nibble_first = true;
    else if (name == "mxfp4-transposed-experts") opts.decode.transpose_experts = true;
    else if (name == "fp8-transposed-scales") opts.decode.transpose_fp8_scales = true;
    else throw std::invalid_argument("unknown host contract " + std::string(name));
    opts.variants.emplace_back(name);
}

unsigned number(std::string_view text) {
    unsigned value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned integer");
    return value;
}

Options parse_options(int argc, char** argv) {
    Options opts;
    enum class Contract { checkpoint, recovered, candidate };
    auto contract = Contract::checkpoint;
    std::vector<std::string_view> overrides;
    std::optional<std::string_view> multimax_override;
    for (int i = 4; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--oracle") { opts.oracle = true; continue; }
        if (arg == "--ordered-simd") { opts.simd = true; continue; }
        if (arg == "--table-mxfp4") { opts.decode.table_mxfp4 = true; continue; }
        if (arg == "--parallel-mla") { opts.parallel_mla = true; continue; }
        if (arg == "--recovered-contract") { contract = Contract::recovered; continue; }
        if (arg == "--candidate-contract") { contract = Contract::candidate; continue; }
        if (arg == "--checkpoint-contract") { contract = Contract::checkpoint; continue; }
        if (arg == "--bos") { opts.bos = true; continue; }
        if (arg == "--save-layers") { opts.save_layers = true; continue; }
        if (arg == "--dump-logits") { opts.dump_logits = true; continue; }
        if (arg == "--profile-host") { opts.profile = true; continue; }
        if (arg == "--bf16-projections") { opts.bf16 = true; continue; }
        if (arg == "--multiply-fixed-gain") { opts.fixed_gain = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing option value");
        if (arg == "--generate") opts.steps = number(argv[i]);
        else if (arg == "--min-new-tokens") opts.min_new_tokens = number(argv[i]);
        else if (arg == "--threads") opts.threads = number(argv[i]);
        else if (arg == "--cache-mib") opts.cache_mib = number(argv[i]);
        else if (arg == "--prompt-index") opts.prompt = number(argv[i]);
        else if (arg == "--activation") opts.activation = argv[i];
        else if (arg == "--variant") overrides.emplace_back(argv[i]);
        else if (arg == "--multimax") multimax_override = argv[i];
        else if (arg == "--capture") opts.capture = argv[i];
        else if (arg == "--pack") opts.packed = argv[i];
        else if (arg == "--replay") opts.replay = argv[i];
        else if (arg == "--resume") opts.resume = argv[i];
        else if (arg == "--start-layer") opts.start_layer = number(argv[i]);
        else if (arg == "--end-layer") opts.end_layer = number(argv[i]);
        else if (arg == "--limit") opts.limit = number(argv[i]);
        else throw std::invalid_argument("unrecognized option " + std::string(arg));
    }
    if (contract == Contract::checkpoint) {
        set_variant(opts, "router-summed-scores");
        set_variant(opts, "norm-routed-output");
        set_variant(opts, "kv-no-head-norm");
        set_variant(opts, "vha-swap");
        set_variant(opts, "checkpoint-attention");
        set_variant(opts, "checkpoint-hc");
        set_variant(opts, "checkpoint-stores");
        opts.multimax = "checkpoint";
    } else if (contract == Contract::recovered) {
        set_variant(opts, "router-gain-before-norm");
        set_variant(opts, "kv-no-head-norm");
        set_variant(opts, "vha-swap");
    }
    for (const auto variant : overrides) set_variant(opts, variant);
    if (multimax_override) opts.multimax = *multimax_override;
    if (opts.multimax != "checkpoint" && opts.multimax != "identity" && opts.multimax != "hinge" &&
        opts.multimax != "signed-hinge" && opts.multimax != "softplus")
        throw std::invalid_argument("unknown vocabulary activation");
    if (!opts.steps || opts.steps > 4096 || !opts.threads || opts.threads > 16 || opts.cache_mib > 32768)
        throw std::invalid_argument("expected 1-4096 output tokens, 1-16 threads and 0-32768 MiB of decoded cache");
    if (opts.min_new_tokens > opts.steps || (opts.min_new_tokens && (opts.oracle || !opts.replay.empty())))
        throw std::invalid_argument("minimum new tokens require a free-running generation with a sufficient limit");
    if (!opts.capture.empty() && (!opts.packed.empty() || !opts.replay.empty()))
        throw std::invalid_argument("capture requires a free-running source-checkpoint run");
    if (opts.oracle && (opts.prompt || !opts.capture.empty() || !opts.replay.empty() ||
        opts.limit < 2 || opts.limit > 128 || opts.cache_mib > 8192 ||
        opts.start_layer > opts.end_layer || opts.end_layer > ernie::kLayerCount ||
        (opts.start_layer != 0 && opts.resume.empty())))
        throw std::invalid_argument("invalid oracle options or missing layer cache");
    if (!opts.oracle && (opts.save_layers || opts.dump_logits || !opts.resume.empty() || opts.start_layer != 0 ||
        opts.end_layer != ernie::kLayerCount || opts.limit != 128))
        throw std::invalid_argument("layer and text options require --oracle");
    return opts;
}

Json::Array variant_names(const Options& opts) {
    Json::Array names;
    for (const auto& variant : opts.variants) names.emplace_back(variant);
    return names;
}

void transform_logits(std::span<float> logits, std::string_view mode, const ernie::ReferenceRead& read) {
    if (mode == "identity") return;
    float ranges[4], strengths[4];
    read("model.lm_head.multimax_ranges", ranges, 0);
    read("model.lm_head.multimax_ts", strengths, 0);
    if (mode == "checkpoint") {
        core::multimax_f32_reference(logits, ranges, strengths);
    } else if (mode == "hinge" || mode == "signed-hinge") {
        for (auto& z : logits) {
            const auto x = mode == "signed-hinge" ? std::abs(z) : z;
            float y = x;
            for (std::size_t i = 0; i < std::size(ranges); ++i)
                y += std::expm1(strengths[i]) * std::max(0.0F, x - ranges[i]);
            z = mode == "signed-hinge" ? std::copysign(y, z) : y;
        }
    } else if (mode == "softplus") {
        for (auto& z : logits) {
            float delta = 0;
            for (std::size_t i = 0; i < std::size(ranges); ++i) {
                const float t = z - ranges[i];
                delta += std::expm1(strengths[i]) * (std::max(t, 0.0F) + std::log1p(std::exp(-std::abs(t))));
            }
            z += delta;
        }
    } else {
        throw std::invalid_argument("unknown MultiMax candidate");
    }
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

bool is_eos(std::uint32_t token) {
    return std::ranges::find(ernie::data::tokens::generation_eos_ids, token) !=
           std::end(ernie::data::tokens::generation_eos_ids);
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

fs::path layer_file(const fs::path& folder, unsigned layer) {
    return folder / ("layer-" + std::to_string(layer) + ".f32");
}

void run_oracle(std::vector<std::uint32_t> ids, const Options& opts, const fs::path& folder,
                const ernie::ReferenceRead& read, const ernie::ReferenceProject& project,
                core::HostLane& lane, strix::cpu::OrderedFma kernel,
                const ernie::AttentionHeadExecutor& execute_heads) {
    if (ids.size() < 2) throw std::invalid_argument("oracle needs at least two input tokens");
    auto shape = ernie::checkpoint_reference_shape();
    if (!opts.fixed_gain) shape.route.scale = 1.0F;
    shape.contracts = opts.contracts;
    auto token_out = output_file(folder / "prompt.u32");
    write_row(token_out, std::span<const std::uint32_t>(ids));
    token_out.close();
    const auto inputs = std::span<const std::uint32_t>(ids).first(ids.size() - 1);
    const auto targets = std::span<const std::uint32_t>(ids).subspan(1);
    const auto width = std::size_t{shape.hidden} * ernie::data::extras::hc_mult;
    ernie::CpuReference ref(shape, read, activation(opts.activation), project, {}, execute_heads);
    const auto started = Clock::now();
    ernie::ReferenceRows rows;
    if (opts.resume.empty()) {
        rows = ref.embed_rows(inputs);
    } else {
        const auto token_path = opts.resume / "prompt.u32";
        if (fs::file_size(token_path) != ids.size() * sizeof(ids[0]))
            throw std::invalid_argument("oracle layer cache has a different token count");
        auto saved = input_file(token_path);
        std::vector<std::uint32_t> original(ids.size());
        saved.read(reinterpret_cast<char*>(original.data()), static_cast<std::streamsize>(ids.size() * sizeof(ids[0])));
        if (original != ids) throw std::invalid_argument("oracle layer cache tokens differ");
        const auto path = layer_file(opts.resume, opts.start_layer);
        const auto row_bytes = width * sizeof(float);
        if (fs::file_size(path) != inputs.size() * row_bytes)
            throw std::invalid_argument("oracle layer cache has an invalid size");
        auto boundary = input_file(path);
        rows.assign(inputs.size(), std::vector<float>(width));
        for (auto& row : rows)
            boundary.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row_bytes));
    }
    auto scales = trace_file(folder / "rms.tsv", "boundary\tposition\trms");
    auto timing = trace_file(folder / "layers.tsv", "boundary\tseconds\tmean_rms");
    auto previous = Clock::now();
    rows = ref.prefill(std::move(rows), opts.start_layer, opts.end_layer, [&](unsigned layer, const auto& streams) {
        const auto now = Clock::now();
        const double seconds = std::chrono::duration<double>(now - previous).count();
        double total = 0;
        for (std::size_t row = 0; row < streams.size(); ++row) {
            const auto scale = rms(streams[row]);
            if (!std::isfinite(scale)) throw std::runtime_error("oracle: nonfinite residual RMS");
            scales << layer << '\t' << row << '\t' << scale << '\n';
            total += scale;
        }
        timing << layer << '\t' << seconds << '\t' << total / static_cast<double>(streams.size()) << '\n';
        if (opts.save_layers) {
            auto file = output_file(layer_file(folder, layer));
            for (const auto& row : streams) write_row(file, std::span(row));
        }
        timing.flush();
        std::cerr << "oracle boundary " << layer << " in " << seconds << " s\n";
        previous = Clock::now();
    });
    const auto prefill = Clock::now();
    for (auto& row : rows) row = ref.head_input(row);
    // Read each head block once for all positions. Preserve each dot's
    // accumulation order and stream the decoded weights instead of caching it.
    constexpr std::size_t kRows = 1024;
    std::vector<float> block(kRows * shape.hidden);
    std::vector<std::vector<float>> logits(rows.size(), std::vector<float>(shape.vocab));
    std::optional<strix::cpu::OrderedLinearBatch> batch;
    if (opts.simd) batch.emplace(rows, kernel);
    for (std::size_t first = 0; first < shape.vocab; first += kRows) {
        const auto count = std::min(kRows, shape.vocab - first);
        const auto matrix = std::span(block).first(count * shape.hidden);
        read("lm_head.weight", matrix, first * shape.hidden);
        core::reference_row_ranges(lane, opts.threads, count, [&](auto begin, auto end) {
            for (auto col = begin; col < end; ++col) {
                const auto weight = matrix.subspan(col * shape.hidden, shape.hidden);
                if (batch) {
                    batch->column(weight, logits, first + col);
                } else {
                    for (std::size_t row = 0; row < rows.size(); ++row)
                        logits[row][first + col] = core::dot_f32(weight, rows[row]);
                }
            }
        });
    }
    if (opts.bf16 || opts.contracts.checkpoint_stores)
        for (auto& row : logits) core::round_bf16_reference(row);
    if (opts.dump_logits) {
        auto dump = output_file(folder / "raw-logits.f32");
        for (const auto& row : logits) write_row(dump, std::span(row));
    }
    std::map<std::uint32_t, std::size_t> frequencies;
    for (auto id : targets) ++frequencies[id];
    auto scores = trace_file(folder / "scores.tsv", "position\tinput_id\ttarget_id\ttop_id\tnll\thit");
    double loss = 0, unigram = 0;
    unsigned hits = 0;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        transform_logits(logits[row], opts.multimax, read);
        const auto score = ernie::score_next_token(logits[row], targets[row]);
        const bool hit = score.best_id == targets[row];
        scores << row << '\t' << inputs[row] << '\t' << targets[row] << '\t' << score.best_id << '\t'
               << score.nll << '\t' << hit << '\n';
        loss += score.nll;
        hits += hit ? 1U : 0U;
        unigram += std::log(static_cast<double>(targets.size()) / static_cast<double>(frequencies.at(targets[row])));
    }
    const auto count = static_cast<double>(targets.size());
    const auto end = Clock::now();
    const auto prefill_seconds = std::chrono::duration<double>(prefill - started).count();
    const auto head_seconds = std::chrono::duration<double>(end - prefill).count();
    const auto receipt = Json(Json::Object{
        {"activation", opts.activation}, {"multiply_fixed_gain", opts.fixed_gain}, {"bf16_projections", opts.bf16},
        {"variants", variant_names(opts)}, {"multimax", opts.multimax},
        {"bos", opts.bos}, {"first_layer", std::uint64_t{opts.start_layer}}, {"end_layer", std::uint64_t{opts.end_layer}},
        {"cache_prefix", opts.resume.generic_string()}, {"threads", std::uint64_t{opts.threads}},
        {"cache_mib", std::uint64_t{opts.cache_mib}}, {"positions", std::uint64_t{targets.size()}},
        {"linear_isa", strix::cpu::isa_name(kernel.isa)}, {"parallel_mla", opts.parallel_mla},
        {"table_mxfp4", opts.decode.table_mxfp4},
        {"mean_nll", loss / count}, {"perplexity", std::exp(loss / count)}, {"top1_hit_rate", hits / count},
        {"unigram_nll", unigram / count}, {"uniform_nll", std::log(static_cast<double>(shape.vocab))},
        {"prefill_seconds", prefill_seconds}, {"head_seconds", head_seconds}});
    output_file(folder / "run.json") << receipt.dump() << '\n';
    std::cout << receipt.dump() << '\n';
}

struct ProfileBucket {
    std::size_t calls = 0, elements = 0;
    double seconds = 0;

    void record(Clock::time_point start, std::size_t size) {
        ++calls;
        elements += size;
        seconds += std::chrono::duration<double>(Clock::now() - start).count();
    }
};

std::string_view profile_group(std::string_view name) {
    if (name == "lm_head.weight") return "head";
    if (name.find(".experts.") != name.npos) return "routed";
    if (name.find(".self_attn.") != name.npos) return "attention";
    if (name.find("_hyper_connection.") != name.npos) return "mHC";
    return "other";
}

struct Replay {
    std::ifstream ids, logits;
    std::ofstream comparison;
    std::vector<float> row;
    bool matched = true;
    double kl = 0.0;
    unsigned agreements = 0;

    Replay(const fs::path& previous, const fs::path& folder, std::span<const std::uint32_t> prompt, unsigned vocab)
        : ids(input_file(previous / "generated.u32")),
          logits(input_file(fs::exists(previous / "selection-logits.f32") ? previous / "selection-logits.f32"
                                                                           : previous / "raw-logits.f32")),
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
                const ernie::ReferenceProject& project, ernie::ExpertCalibration* calibration,
                const ernie::AttentionHeadExecutor& execute_heads) {
    const auto* messages = request.find("messages");
    if (!messages) throw std::invalid_argument("prompt requires messages");
    strix::serve::ChatTemplateInput chat{};
    chat.messages = *messages;
    if (const auto* tools = request.find("tools")) chat.tools = *tools;
    chat.enable_thinking = false;
    const auto prompt = strix::serve::render_ernie_chat(chat);
    auto ids = prompt.encode(tokenizer);
    constexpr std::string_view bos_marker = "<|ERNIE_BEGIN|>";
    if (opts.bos) ids.insert(ids.begin(), tokenizer.find(bos_marker));
    if (ids.empty() || ids.size() > 4096 || opts.steps > 4096 - ids.size())
        throw std::invalid_argument("CPU runner requires prompt plus generation to fit context 4096");
    fs::create_directory(folder);
    output_file(folder / "request.json") << request.dump() << '\n';
    output_file(folder / "prompt.txt") << (opts.bos ? bos_marker : std::string_view{}) << prompt.text();
    auto prompt_out = output_file(folder / "prompt.u32");
    write_row(prompt_out, std::span<const std::uint32_t>(ids));
    prompt_out.close();
    std::cerr << "prompt " << index << " prefill " << ids.size() << " rows\n";

    auto shape = ernie::checkpoint_reference_shape();
    if (!opts.fixed_gain) shape.route.scale = 1.0F;
    shape.contracts = opts.contracts;
    ernie::ExpertObserver expert_observer;
    std::size_t window = 0;
    if (calibration) {
        window = calibration->add_window(ids);
        expert_observer = [&](auto layer, auto expert, auto value, bool down) {
            calibration->observe(window, layer, expert, down ? strix::moe::ExpertProjection::down :
                                 strix::moe::ExpertProjection::gate_up, value);
        };
    }
    ernie::CpuReference reference(shape, read, activation(opts.activation), project, expert_observer, execute_heads,
                                  ids.size() + opts.steps - 1);
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
    std::optional<std::ofstream> selection_scores;
    if (opts.multimax != "identity") selection_scores.emplace(output_file(folder / "selection-logits.f32"));
    for (unsigned step = 0; step < opts.steps; ++step) {
        write_row(files.scores, std::span<const float>(logits));
        transform_logits(logits, opts.multimax, read);
        if (selection_scores) {
            write_row(*selection_scores, std::span<const float>(logits));
            selection_scores->flush();
        }
        const auto top = top_logits(logits);
        if (step == 0) {
            const auto label = opts.multimax == "identity" ? "id\traw_logit" : "id\tselection_logit";
            auto first = trace_file(folder / "first-token-top20.tsv", label);
            for (auto id : top) first << id << '\t' << logits[id] << '\n';
        }
        auto choice = top.front();
        if (step < opts.min_new_tokens) {
            const auto next = std::ranges::find_if_not(top, is_eos);
            if (next == top.end()) throw std::logic_error("top candidates contain only stop tokens");
            choice = *next;
        }
        generated.push_back(replay ? replay->next(step, logits) : choice);
        write_row(files.generated, std::span<const std::uint32_t>(generated).last(1));
        files.generated.flush();
        files.scores.flush();
        output_file(folder / "partial-response.txt") << tokenizer.decode(generated, false);
        std::cerr << "prompt " << index << " generated " << generated.size() << " id " << generated.back() << '\n';
        if (is_eos(generated.back()) || step + 1 == opts.steps) break;
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
        {"identity", opts.multimax == "checkpoint" ? "checkpoint-host" :
                     (opts.multimax == "identity" ? "candidate-raw-head" : "candidate-transformed-head")},
        {"activation", opts.activation}, {"multimax", opts.multimax},
        {"bos", opts.bos}, {"minimum_new_tokens", std::uint64_t{opts.min_new_tokens}},
        {"bf16_projections", opts.bf16}, {"multiply_fixed_gain", opts.fixed_gain},
        {"variants", variant_names(opts)},
        {"pack", !opts.packed.empty()}, {"teacher_forced", replay.has_value()},
        {"ordered_simd", opts.simd}, {"table_mxfp4", opts.decode.table_mxfp4}, {"parallel_mla", opts.parallel_mla},
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
                     " [--generate N] [--min-new-tokens N] [--bos] [--prompt-index N] [--activation soft-clip|hard-clip|swiglu]"
                     " [--bf16-projections] [--multiply-fixed-gain] [--checkpoint-contract|--recovered-contract|--candidate-contract] [--threads N] [--cache-mib N]"
                     " [--capture MODEL_DESCRIPTION] [--pack SIDECAR] [--replay TRACE_DIR]\n"
                     "       strix-ernie-cpu CHECKPOINT TEXT_FILE NEW_TRACE_DIR --oracle"
                     " [--bos] [--limit N] [--save-layers] [--dump-logits] [--resume TRACE_DIR]"
                     " [--start-layer N] [--end-layer N] [--threads N] [--cache-mib N] [--profile-host]"
                     " [--ordered-simd] [--table-mxfp4] [--parallel-mla] [--checkpoint-contract|--recovered-contract|--candidate-contract]"
                     " [--variant NAME] [--multimax checkpoint|identity|hinge|signed-hinge|softplus]\n";
        return 2;
    }
    const fs::path checkpoint = argv[1], folder = argv[3];
    const auto opts = parse_options(argc, argv);
    (void)activation(opts.activation);
    if (!fs::create_directory(folder)) throw std::invalid_argument("trace directory must be new");

    const auto tokenizer = strix::serve::load_ernie_tokenizer((checkpoint / "tokenizer.json").string());
    ernie::Inventory inventory;
    const auto inspected = ernie::inspect_weights(checkpoint, inventory);
    if (!inspected) throw std::runtime_error(inspected.detail);
    if (!inventory.has_manifest || !inventory.missing_shards.empty() || inventory.missing_known_tensors != 0)
        throw std::runtime_error("checkpoint inventory is incomplete");
    ernie::CheckpointWeights weights(std::move(inventory), opts.decode);
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
    std::map<std::pair<std::string_view, std::string_view>, ProfileBucket> profile;
    const ernie::ReferenceRead load = [&](auto name, auto row, auto offset) {
        const auto started = opts.profile ? Clock::now() : Clock::time_point{};
        weights.read(name, row, offset, execute);
        if (opts.profile) profile[{"read+decode", profile_group(name)}].record(started, row.size());
    };
    core::ReferenceMatrixCache cache(kMiB * opts.cache_mib);
    const ernie::ReferenceRead read = [&](auto name, auto row, auto offset) {
        if (opts.parallel_mla && opts.cache_mib && offset == 0 && name.ends_with(".kv_b_proj.weight"))
            std::ranges::copy(cache.get(std::string(name), row.size(), load), row.begin());
        else load(name, row, offset);
    };
    ernie::AttentionHeadExecutor execute_heads;
    if (opts.parallel_mla) execute_heads = [&](auto heads, const auto& work) {
        core::reference_row_ranges(lane, opts.threads, heads, work);
    };
    const auto kernel = strix::cpu::ordered_fma_kernel(opts.simd);
    const auto linear = [&](std::string_view name, auto matrix, auto in, auto out, unsigned groups = 1) {
        const auto started = opts.profile ? Clock::now() : Clock::time_point{};
        if (opts.simd && kernel.isa != strix::cpu::Isa::scalar) {
            strix::cpu::parallel_ordered_matvec(kernel, lane, opts.threads, matrix, in, out, groups);
        } else {
            core::parallel_linear_reference(lane, opts.threads, matrix, in, out, groups);
        }
        if (opts.profile) profile[{"dot", profile_group(name)}].record(started, matrix.size());
    };
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
                linear(name, matrix, in, out.subspan(first, count));
            }
        } else {
            const auto matrix = cache.get(name, out.size() * (in.size() / groups), load);
            linear(name, matrix, in, out, groups);
        }
        const bool raw_router = name.ends_with(".gate.weight") || name.ends_with(".gate.weight_1");
        if ((opts.contracts.checkpoint_stores && !raw_router) ||
            (opts.bf16 && name.find(".linear_o_group_proj") == std::string::npos))
            core::round_bf16_reference(out);
    };
    if (opts.oracle) {
        if (fs::file_size(argv[2]) > kMiB) throw std::invalid_argument("oracle text exceeds 1 MiB");
        auto file = input_file(argv[2]);
        auto ids = tokenizer->encode_ordinary(std::string(std::istreambuf_iterator<char>(file), {}));
        if (opts.bos) ids.insert(ids.begin(), tokenizer->find("<|ERNIE_BEGIN|>"));
        if (ids.size() > opts.limit) ids.resize(opts.limit);
        run_oracle(std::move(ids), opts, folder, read, project, lane, kernel, execute_heads);
    } else {
        const auto prompts = load_json(argv[2]).as_array();
        if (prompts.empty() || (opts.prompt && *opts.prompt >= prompts.size()))
            throw std::invalid_argument("missing or invalid prompt index");
        const unsigned first = opts.prompt.value_or(0);
        const unsigned last = opts.prompt ? first + 1 : static_cast<unsigned>(prompts.size());
        for (unsigned index = first; index < last; ++index)
            run_prompt(prompts[index], index, *tokenizer, opts, folder / ("prompt-" + std::to_string(index)),
                       read, project, calibration ? &*calibration : nullptr, execute_heads);
        if (calibration) {
            const auto dir = folder / "calibration";
            calibration->write(dir, opts.capture);
            constexpr unsigned selected_layers[] = {1, 7, 14};
            calibration->write_slice_plan(dir / "sp2.tsv", selected_layers, 2, "sp2");
            calibration->write_slice_plan(dir / "sp3.tsv", selected_layers, 2, "sp3");
        }
    }
    if (opts.profile) {
        auto file = trace_file(folder / "profile.tsv", "stage\tgroup\tcalls\telements\tseconds");
        for (const auto& [key, value] : profile)
            file << key.first << '\t' << key.second << '\t' << value.calls << '\t' << value.elements << '\t'
                 << value.seconds << '\n';
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
