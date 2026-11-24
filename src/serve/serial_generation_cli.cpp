#include "strix/serve/serial_generation_cli.hpp"

#include "core/dense_linear.hpp"
#include "strix/serial_request.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/media.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/session/cli_output.hpp"
#include "strix/spec/greedy_continuation.hpp"
#include "strix/spec/verify_cost_table.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace strix::serve {
namespace {

constexpr auto speculation_modes = "off|auto|verify|lookup";

std::string_view speculation_name(CliSpeculation mode) {
    switch (mode) {
        case CliSpeculation::off: return "off";
        case CliSpeculation::automatic: return "auto";
        case CliSpeculation::verify: return "verify";
        case CliSpeculation::lookup: return "lookup";
    }
    throw std::logic_error("unknown speculation mode");
}

using Clock = session::CliClock;
namespace fs = std::filesystem;

// The persisted verify walls of this run's speculator (strix/spec/verify_cost_table.hpp), or nothing: no device
// identity from the decoder, the `verify` diagnostic (equal prices), no table directory (STRIX_COST_TABLE=0, baseline,
// no cache directory) or no executable identity. The model identity is the checkpoint's path, size and write time.
std::optional<CostTableCache> open_verify_cost_table(const SerialGenerationFamily& family, const std::string& checkpoint,
                                                     const SerialGenerationDecoder& decoder, CliSpeculation mode,
                                                     bool sampled, std::uint32_t max_rows, std::uint32_t prompt_size) {
    auto device = decoder.verify_cost_device();
    if (device.empty() || mode == CliSpeculation::verify) return std::nullopt;
    const auto directory = cost_table_directory_from_env();
    auto binary = cost_table_executable_identity();
    if (!directory || binary.empty()) return std::nullopt;
    std::error_code ec;
    auto model = fs::weakly_canonical(fs::path(checkpoint), ec).string();
    if (ec) model = checkpoint;
    if (const auto size = fs::file_size(fs::path(checkpoint), ec); !ec) model += ';' + std::to_string(size);
    if (const auto written = fs::last_write_time(fs::path(checkpoint), ec); !ec)
        model += ';' + std::to_string(written.time_since_epoch().count());
    auto policy = cost_table_policy_identity("serial-verify-cost-v1;mode=" + std::string(speculation_name(mode)) +
                                             ";sampled=" + (sampled ? "1" : "0") + ";rows=" + std::to_string(max_rows));
    std::optional<CostTableCache> table;
    table.emplace(*directory,
                  CostTableKey{std::string(family.executable), std::move(binary), std::move(device),
                               cost_table_hash_hex(model), std::move(policy), cost_table_context_bucket(prompt_size)},
                  spec::verify_cost_table_options());
    (void)table->load();
    return table;
}

template <class Number>
Number parse_number(std::string_view value, const char* error) {
    Number result{};
    const auto [end, code] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (code != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument(error);
    return result;
}

void protect_asset(const char* output, const fs::path& asset, const char* error) {
    if (output && fs::exists(output) && fs::equivalent(asset, output)) throw std::invalid_argument(error);
}

std::string read_text(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open text input: " + std::string(path));
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("could not read text input: " + std::string(path));
    return text;
}

}  // namespace

// --messages with image parts: the shared media layer normalizes them (data URLs / base64 only: the CLI has no
// fetcher), the family renders one placeholder per part, its adapter encodes each image within the context left
// by the text, and the placeholders expand to the virtual ids of the rows (session::expand_embed_rows).
struct MediaPrompt {
    std::vector<std::uint32_t> tokens;
    std::shared_ptr<const session::EmbedTable> table;  // null: the body holds no image
};

MediaPrompt render_media_prompt(const SerialGenerationFamily& family, const std::string& checkpoint,
                                const BpeTokenizer& tokenizer, std::string_view body_text,
                                std::uint32_t reserve_tokens, std::unique_ptr<SerialMediaInput>& media) {
    MediaPrompt out;
    auto body = parse_json(body_text);
    if (body.is_array()) {  // a bare messages array is the body {"messages": [...]}
        auto wrapped = Json::object();
        wrapped.set("messages", std::move(body));
        body = std::move(wrapped);
    }
    auto normalized = media::normalize(body, media::Api::chat_completions);
    if (normalized.items.empty()) {
        out.tokens = tokenizer.encode(family.render_messages(body_text), true);
        return out;
    }
    media = family.media(checkpoint);
    if (!media) throw std::runtime_error("the family opened no media input");
    auto& adapter = media->adapter();
    const auto placeholder_ids = tokenizer.encode(adapter.placeholder(), true);
    if (placeholder_ids.size() != 1U || session::is_embed_id(placeholder_ids[0]))
        throw std::invalid_argument("media placeholder must tokenize to one id");
    const auto placeholder = placeholder_ids[0];
    const auto tokens = tokenizer.encode(family.render_media_messages(normalized.body.dump()), true);
    const auto count = normalized.items.size();
    if (static_cast<std::size_t>(std::count(tokens.begin(), tokens.end(), placeholder)) != count)
        throw std::invalid_argument("the rendered prompt does not hold one placeholder per image");
    const std::uint64_t text = tokens.size() - count + reserve_tokens;
    std::uint64_t room = family.max_context > text ? family.max_context - text : 0U;
    std::vector<std::vector<std::uint32_t>> layouts;
    std::vector<session::EmbedLease> leases;
    for (const auto& item : normalized.items) {
        const auto budget = static_cast<std::uint32_t>(
            std::min<std::uint64_t>({adapter.max_rows(), room, session::kEmbedMaxRows}));
        if (budget == 0U) throw std::invalid_argument("image rows do not fit the context");
        auto span = adapter.encode(item, budget);
        if (!span.lease || span.layout.empty() || span.layout.size() > room)
            throw std::invalid_argument("image rows do not fit the context");
        room -= span.layout.size();
        layouts.push_back(std::move(span.layout));
        leases.push_back(std::move(span.lease));
    }
    auto expanded = session::expand_embed_rows(tokens, placeholder, layouts);
    out.tokens = std::move(expanded.tokens);
    out.table = std::make_shared<const session::EmbedTable>(std::move(leases));
    return out;
}

SerialGenerationOptions parse_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily& family,
                                                    const SerialCliOption& extra) {
    SerialGenerationOptions options;
    bool grouped_decode = false;
    options.prefill_chunk = family.default_prefill_chunk;
    if (family.supports_speculation) options.speculation = family.default_speculation;
    std::optional<float> temperature;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const auto value = [&]() {
            if (++i == argc) throw std::invalid_argument("missing option value");
            return argv[i];
        };
        if (argument == "--generate") {
            options.generate = core::parse_u32(value());
            options.generate_requested = true;
        }
        else if (argument == "--logits") options.logits = value();
        else if (family.supports_speculation && argument == "--speculation") {
            const std::string_view mode(value());
            if (mode == "off") options.speculation = CliSpeculation::off;
            else if (mode == "auto") options.speculation = CliSpeculation::automatic;
            else if (mode == "verify") options.speculation = CliSpeculation::verify;
            else if (mode == "lookup") options.speculation = CliSpeculation::lookup;
            else throw std::invalid_argument("--speculation must be off, auto, verify or lookup");
        }
        else if (argument == "--temperature" || argument == "--top-k" || argument == "--top-p" ||
                 argument == "--min-p" || argument == "--seed") {
            options.sampling_requested = true;
            if (argument == "--temperature") temperature = parse_number<float>(value(), "invalid temperature");
            else if (argument == "--top-k") options.sampling.top_k = core::parse_u32(value());
            else if (argument == "--top-p") options.sampling.top_p = parse_number<float>(value(), "invalid top-p");
            else if (argument == "--min-p") options.sampling.min_p = parse_number<float>(value(), "invalid min-p");
            else options.sampling.seed = parse_number<std::uint64_t>(value(), "invalid seed");
        }
        else if (family.load_tokenizer && argument == "--prompt") options.text = value();
        else if (family.render_messages && argument == "--messages") options.messages = value();
        else if (family.load_tokenizer && argument == "--tokenize") options.tokenize_file = value();
        else if (family.load_tokenizer && argument == "--tokenizer") options.tokenizer = value();
        else if (family.supports_packed_decode && argument == "--packed-decode") options.packed_decode = true;
        else if (family.supports_packed_decode && argument == "--grouped-decode") grouped_decode = true;
        else if (family.default_prefill_chunk && argument == "--prefill-chunk") {
            options.prefill_chunk = core::parse_u32(value());
            options.prefill_chunk_explicit = true;
            if (!options.prefill_chunk || options.prefill_chunk > family.max_context)
                throw std::invalid_argument("prefill chunk size must be within the context");
        } else if (argument.starts_with("--")) {
            if (!extra || !extra(argument, value))
                throw std::invalid_argument("unknown option " + std::string(argument));
        } else {
            options.prompt.push_back(core::parse_u32(argument));
        }
    }
    const unsigned inputs = static_cast<unsigned>(!options.prompt.empty()) +
        static_cast<unsigned>(options.text != nullptr) + static_cast<unsigned>(options.messages != nullptr) +
        static_cast<unsigned>(options.tokenize_file != nullptr);
    if (inputs > 1U) throw std::invalid_argument("choose one of token ids, --prompt, --messages or --tokenize");
    if (options.tokenize_file && (options.generate_requested || options.logits || options.sampling_requested))
        throw std::invalid_argument("--tokenize takes no --generate, --logits or sampling options");
    if (options.tokenizer && !options.text && !options.messages && !options.tokenize_file)
        throw std::invalid_argument("--tokenizer requires --prompt, --messages or --tokenize");
    if (options.packed_decode && grouped_decode)
        throw std::invalid_argument("--packed-decode and --grouped-decode are exclusive");
    options.packed_decode = options.packed_decode || (family.packed_decode_default && !grouped_decode);
    const bool filtered = options.sampling.top_k || options.sampling.top_p < 1.0F || options.sampling.min_p > 0.0F;
    options.sampling.temperature = temperature.value_or(filtered ? 1.0F : 0.0F);
    validate_token_sampling_policy(options.sampling);
    if ((options.speculation == CliSpeculation::verify || options.speculation == CliSpeculation::lookup) &&
        token_sampling_enabled(options.sampling) && !family.sampled_speculation)
        throw std::invalid_argument("--speculation " + std::string(speculation_name(options.speculation)) +
                                    " requires greedy decoding");
    return options;
}

void run_serial_generation(const std::string& checkpoint, SerialGenerationOptions options,
                           const SerialGenerationFamily& family, std::ostream& out,
                           const std::function<void()>& on_loaded) {
    protect_asset(options.logits, checkpoint, "logits output must differ from the checkpoint");
    std::shared_ptr<const BpeTokenizer> tokenizer;
    std::unique_ptr<SerialMediaInput> media;            // outlives the prefill: it owns the interned rows' adapter
    std::shared_ptr<const session::EmbedTable> embeds;  // rows of the virtual ids of an image prompt
    if (options.text || options.messages || options.tokenize_file) {
        const fs::path asset = options.tokenizer ? fs::path(options.tokenizer) :
            fs::path(checkpoint).parent_path() / "tokenizer.json";
        protect_asset(options.logits, asset, "logits output must differ from the tokenizer");
        tokenizer = family.load_tokenizer(asset.string());
        if (options.tokenize_file) {
            for (const auto id : tokenizer->encode(read_text(options.tokenize_file), false)) out << id << '\n';
            if (!out) throw std::runtime_error("could not write token IDs");
            return;
        }
        if (options.messages) {
            protect_asset(options.logits, options.messages, "logits output must differ from the messages file");
            const auto body = read_text(options.messages);
            if (family.media && family.render_media_messages) {
                auto prompt = render_media_prompt(family, checkpoint, *tokenizer, body, options.generate, media);
                options.prompt = std::move(prompt.tokens);
                embeds = std::move(prompt.table);
            } else {
                options.prompt = tokenizer->encode(family.render_messages(body), true);
            }
        } else {
            options.prompt = tokenizer->encode(options.text, false);
        }
    }
    if (options.prompt.empty() || options.prompt.size() > family.max_context ||
        options.generate > family.max_context - options.prompt.size())
        throw std::invalid_argument("prompt and generation must fit the context");
    if (!embeds) {
        core::validate_token_ids(options.prompt.data(), options.prompt.size(), family.input_vocabulary);
    } else {
        for (const auto id : options.prompt)
            if (!session::is_embed_id(id)) core::validate_token_ids(&id, 1U, family.input_vocabulary);
    }
    // Drafters read prompt ids; a prompt with virtual ids runs serial decode (same ids, no drafting).
    if (embeds) options.speculation = CliSpeculation::off;
    const auto prompt_size = static_cast<std::uint32_t>(options.prompt.size());
    const auto positions = prompt_size + options.generate;
    const bool greedy = !token_sampling_enabled(options.sampling);
    const bool rows_decidable = greedy || family.sampled_speculation;
    const auto verify_rows = options.speculation != CliSpeculation::off && rows_decidable && options.generate > 1U
        ? options.generate - 1U : 0U;
    const auto load_start = Clock::now();
    auto decoder = load_with_prefill_admission(
        options.prefill_chunk_explicit ? std::optional{options.prefill_chunk} : std::nullopt,
        family.default_prefill_chunk, family.prefill_chunk_fallback, prompt_size, std::cerr, [&](std::uint32_t rows) {
        return family.load({checkpoint, positions, rows, options.packed_decode, verify_rows,
                            verify_rows ? options.speculation : CliSpeculation::off,
                            prefill_tail_rows(prompt_size, rows, family.prefill_tail_divisor)});
    });
    if (family.report_load_timing) session::write_load_timing(out, load_start);
    if (on_loaded) on_loaded();

    std::optional<TokenSampler> sampler;
    const TokenSamplerOptions sampler_options{};
    if (!greedy) sampler.emplace(options.sampling, sampler_options);
    std::vector<float> logits((options.logits || sampler) ? family.output_vocabulary : 0U);
    const auto prefill_start = Clock::now();
    auto next = embeds ? decoder->prefill_embedded(options.prompt, embeds, logits) : decoder->prefill(options.prompt, logits);
    if (sampler) next = sampler->decide(logits, prompt_size, nullptr).token;
    const auto prefill_ms = session::cli_milliseconds(Clock::now() - prefill_start);
    if (options.logits) session::write_logits(options.logits, logits);
    if (!sampler) logits.clear();

    out << "next_token " << next << '\n';
    if (options.generate) out << "generated" << std::flush;
    const SerialTokenContinuationRequest request{
        next, prompt_size, family.output_vocabulary, positions, options.generate, family.eos};
    Clock::duration emit_time{};
    const auto emit = [&](std::uint32_t token) {
        const auto start = Clock::now();
        out << ' ' << token << std::flush;
        emit_time += Clock::now() - start;
    };
    const auto max_rows = verify_rows > 1U
        ? std::min(verify_rows, decoder->max_verify_rows(options.speculation)) : 0U;
    auto* verify = max_rows > 1U ? decoder->verify_target() : nullptr;
    // A sampled request verifies rows with the serial sampler's draw at each position (same policy and request
    // identity as the serial path below), so the ids equal serial sampled decode.
    const auto row_rule = greedy ? spec::RowRule{} : spec::RowRule(options.sampling, sampler_options.request_identity);
    if (verify && !greedy && !verify->supports_row_rule(true)) {
        if (options.speculation != CliSpeculation::automatic)
            throw std::invalid_argument("this target cannot verify sampled rows");
        verify = nullptr;
    }
    std::optional<spec::BlockDrafter> drafter;
    // Opened (file read) before the decode clock; stored after it.
    auto cost_table = verify ? open_verify_cost_table(family, checkpoint, *decoder, options.speculation, !greedy,
                                                      max_rows, prompt_size)
                             : std::nullopt;
    // An untimed bucket seeds the curve shape from the nearest timed bucket of the same key (read only).
    const auto cost_prior = cost_table ? spec::open_nearest_verify_cost_prior(*cost_table, max_rows) : std::nullopt;
    std::uint32_t cost_seeded = 0, cost_recorded = 0, chain_k = 0;
    bool cost_stored = false, prices_frozen = false;
    SerialGreedyRequestResult generated;
    const auto decode_start = Clock::now();
    if (verify) {
        spec::BlockConfig config;
        config.hybrid = false;  // The optional chain owns a complete, independent block (an engine may opt in).
        if (options.speculation != CliSpeculation::verify) decoder->tune_speculation(config);
        if (config.arbiter.windows.empty() || max_rows > config.arbiter.windows.back())
            throw std::invalid_argument("target exceeds the supported verify row count");
        config.max_block = max_rows;
        config.force_lookup = options.speculation == CliSpeculation::lookup;
        auto* chain = options.speculation == CliSpeculation::automatic ? decoder->chain_drafter(next, prompt_size) : nullptr;
        if (!chain) config.hybrid = false;
        chain_k = chain && config.chain ? std::min(config.chain_max_k, max_rows > 1U ? max_rows - 1U : 0U) : 0U;
        // Frozen prices (strix/spec/verify_cost_table.hpp): a table that prices every window of this run (and a chain
        // depth) is the whole cost model, so the row choices are a function of the table and the acceptance history,
        // not of this run's host clock. Otherwise auto learns verify + commit and draft walls in this run. The
        // diagnostic keeps its original equal prices.
        prices_frozen = cost_table && spec::table_prices_run(*cost_table, config.arbiter.windows, max_rows, chain_k);
        config.fixed_costs = options.speculation == CliSpeculation::verify || prices_frozen;
        config.frozen_prices = prices_frozen;
        // Learned walls: the first pass of a width can carry its one-time work (graph capture), never a price.
        if (!config.fixed_costs) config.arbiter.cold_walls = 1;
        if (config.fixed_costs) spec::set_context_cost(config.arbiter, "off");
        // Frozen prices also price the exploration (ArbiterConfig::priced_exploration: probes never buy a window that
        // cannot pay and never move the arm hysteresis) and stop a drafter whose realized cycles lose at those prices
        // (ArbiterConfig::cycle_half_life_tokens).
        if (prices_frozen) spec::set_frozen_price_rules(config.arbiter);
        drafter.emplace(config, chain);
        // Before seeding: switching the drafter to sampled rows resets its learned observations.
        drafter->set_row_rule(row_rule);
        if (options.speculation == CliSpeculation::verify)
            for (std::uint32_t row = 1; row <= max_rows; ++row) drafter->seed_verify_ms(row, 1.0);
        else
            decoder->seed_speculation(*drafter);
        // Walls measured by earlier runs win over the engine's declared seeds (applied after them).
        if (cost_table) {
            const auto& seeds = cost_prior ? *cost_prior : *cost_table;
            cost_seeded = spec::seed_verify_costs(seeds, *drafter, max_rows);
            if (chain_k > 0U) (void)spec::seed_draft_costs(seeds, *drafter, chain_k);
        }
        generated =spec::execute_block_token_continuation(request, options.prompt, *drafter, *verify, max_rows, emit,
                                                           row_rule);
    } else {
        generated = execute_serial_token_continuation(request, [&](std::uint32_t input, std::uint64_t position) {
            const auto selected = decoder->decode(input, logits);
            return sampler ? sampler->decide(logits, position + 1U, nullptr).token : selected;
        }, emit);
    }
    const auto decode_ms = session::cli_milliseconds(Clock::now() - decode_start - emit_time);
    const auto decode_tok_s = generated.steps_executed && decode_ms > 0.0
        ? 1000.0 * static_cast<double>(generated.steps_executed) / decode_ms : 0.0;
    if (cost_table && drafter) {
        cost_recorded = spec::record_verify_costs(*cost_table, drafter->stats());
        if (chain_k > 0U) cost_recorded += spec::record_draft_costs(*cost_table, *drafter, chain_k);
        cost_stored = cost_table->store();
    }
    if (options.generate) out << '\n';
    out << "timing prompt_tokens " << prompt_size << " prefill_ms " << prefill_ms
        << " decode_tokens " << generated.steps_executed << " decode_ms "
        << (generated.steps_executed ? decode_ms : 0.0) << " decode_tok_s " << decode_tok_s << '\n';
    if (options.speculation != CliSpeculation::off) {
        const spec::SpecStats empty;
        const auto& stats = drafter ? drafter->stats() : empty;
        const auto sum = [](const auto& counts) { return std::accumulate(counts.begin(), counts.end(), std::uint64_t{}); };
        const auto accepted = sum(stats.source_accepted);
        const auto drafted = sum(stats.source_proposed);
        const auto draft_passes = stats.passes - stats.source_passes[static_cast<std::size_t>(spec::Source::serial)];
        out << "speculation mode " << speculation_name(options.speculation)
            << " acceptance " << accepted << '/' << drafted << " draft_passes " << draft_passes
            << " mean_accepted_length "
            << (draft_passes ? static_cast<double>(accepted) / static_cast<double>(draft_passes) : 0.0) << '\n';
        if (drafter) out << stats.line() << '\n';
        if (cost_table)
            out << "speculation cost-table " << cost_table_load_name(cost_table->load_status()) << " seeded "
                << cost_seeded << " recorded " << cost_recorded << " stored " << (cost_stored ? 1 : 0) << " file "
                << cost_table->file().filename().string() << " prices " << (prices_frozen ? "frozen" : "learned")
                << (cost_prior ? " prior-bucket " + std::to_string(cost_prior->key().context_bucket) : std::string{})
                << '\n';
    }
    if (tokenizer && !generated.generated_tokens.empty())
        out << "text " << tokenizer->decode(generated.generated_tokens, true) << '\n';
}

std::uint32_t SerialGenerationDecoder::prefill_embedded(std::span<const std::uint32_t> ids,
                                                       const std::shared_ptr<const session::EmbedTable>& table,
                                                       std::span<float> logits) {
    if (table) throw std::invalid_argument("this family takes no provided embedding rows");
    return prefill(ids, logits);
}

int run_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily& family,
                              std::ostream& out, std::ostream& error) {
    if (argc < 3) {
        error << "usage: " << family.executable << ' ' << family.checkpoint_argument;
        if (family.load_tokenizer) error << " (TOKEN [TOKEN ...] | --prompt TEXT [--tokenizer FILE])";
        else error << " TOKEN [TOKEN ...]";
        if (family.default_prefill_chunk) error << " [--prefill-chunk N]";
        if (family.supports_packed_decode) error << " [--packed-decode]";
        if (family.supports_speculation) error << " [--speculation " << speculation_modes << "]";
        error << " [--generate N] [--logits FILE] [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n";
        return 2;
    }
    try {
        run_serial_generation(argv[1], parse_serial_generation_cli(argc, argv, family), family, out);
    } catch (const std::exception& failure) {
        error << family.diagnostic_name << ": " << failure.what() << '\n';
        return 1;
    }
    return 0;
}

}  // namespace strix::serve
