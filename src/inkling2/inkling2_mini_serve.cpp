#include "inkling2_mini_serve.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/inkling2_chat.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::inkling2 {
namespace {

std::uint64_t number(std::string_view name, std::string_view input, std::uint64_t max) {
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), parsed);
    if (error != std::errc{} || end != input.data() + input.size() || parsed == 0 || parsed > max)
        throw std::invalid_argument("invalid " + std::string(name));
    return parsed;
}

struct ResolvedService {
    MiniServeOptions options;
    std::string_view backend;
};

serve::CohortConfig scheduling(const MiniServeOptions& opt, bool hip, std::string_view device_class);

ResolvedService resolve(const serve::ServiceSelection& selection, const MiniServeBackends& backends) {
    ResolvedService resolved;
    auto& out = resolved.options;
    std::string identity_name;  // --served-identity (empty = the data's default)
    const auto& args = selection.arguments;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view name = args[i];
        if (name == "--dspark") throw std::invalid_argument("Inkling serving does not support --dspark");
        const auto next = [&]() -> const std::string& {
            if (++i == args.size()) throw std::invalid_argument("missing value for " + std::string(name));
            return args[i];
        };
        if (parse_mini_weight_option(out.weight_options, name, next, true)) continue;
        const auto& value = next();
        if (name == "--checkpoint") out.checkpoint = value;
        else if (name == "--tokenizer") out.tokenizer = value;
        else if (name == "--backend") {
            if (value == "hip") resolved.backend = "hip";
            else if (value == "cuda") resolved.backend = "cuda";
            else throw std::invalid_argument("Inkling --backend requires hip or cuda");
        } else if (name == "--devices") out.devices_path = value;
        else if (name == "--device-id") out.device_id = value;
        else if (name == "--device-profile") out.device_profile = value;
        else if (name == "--context") out.context = static_cast<unsigned>(number(name, value, kMaxContext));
        else if (name == "--prefill-rows" || name == "--prefill-chunk")  // one spelling with the CLI; the old one kept
            out.prefill_rows = out.prefill_rows_requested = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--served-identity") identity_name = value;
        else if (name == "--prefill-rows-decoding")
            out.prefill_rows_decoding = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--prefill-gap-ms" || name == "--prompt-share") {
            // the whole value is a number (1000ms is refused), finite and >= 0
            double v = -1.0;
            const std::string text(value);
            std::size_t used = 0;
            try { v = std::stod(text, &used); } catch (const std::exception&) { used = 0; }
            if (text.empty() || used != text.size() || !std::isfinite(v) || v < 0.0)
                throw std::invalid_argument("Inkling " + std::string(name) + " requires a number >= 0, got '" + text + "'");
            (name == "--prompt-share" ? out.prompt_share : out.prefill_gap_ms) = v;
        }
        else if (name == "--slots") out.slots = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--kv-bytes") out.kv_bytes = number(name, value, UINT64_MAX);
        else if (name == "--speculation") {
            if (value != "off") throw std::invalid_argument("Inkling serving requires --speculation off");
        } else throw std::invalid_argument("unknown Inkling argument " + std::string(name));
    }
    const auto resolution = resolve_mini_profile(out.weight_options, true);
    out.tier = resolution.profile;
    out.moe_w4a4 = mini_moe_w4a4_class(resolution);
    if (out.checkpoint.empty()) throw std::invalid_argument("Inkling requires --checkpoint FILE");
    if (out.tokenizer.empty())
        out.tokenizer = (std::filesystem::path(out.checkpoint).parent_path() / "tokenizer.json").string();
    if (out.context < 3U) throw std::invalid_argument("Inkling context must hold a prompt and a reply");
    if (resolved.backend.empty()) resolved.backend = backends.hip ? "hip" : "cuda";
    // The served data (models/inkling2-mini.toml [serve_prefill], mini_serve_prefill): the chunk (HIP: the CLI's,
    // CUDA: the incumbent 128 rows), the CLI's tail rule and STRIX_INKLING_PREFILL_TAIL=0 off switch, the pieces.
    const bool hip = resolved.backend == "hip";
    // The served identity (one registry, kMiniServedIdentities): served-incumbent keeps main's served bits at any
    // chunk that is a multiple of main's served chunk (form_rows: the requested --prefill-rows, else 128, clipped to
    // the context as main did), with no tail absorption; served-cli-prefill takes the CLI's chunks and tail rule.
    out.served_identity = mini_resolve_served_identity(identity_name);
    out.prefill_rows = mini_resolve_serve_chunk(out.prefill_rows, out.context, hip);
    if (out.served_identity == MiniServedIdentity::incumbent) {
        out.form_rows = mini_serve_slot_cap(out.prefill_rows_requested, out.context);  // main's served chunk
        out.prefill_rows = std::max(out.form_rows, out.prefill_rows / out.form_rows * out.form_rows);
        out.prefill_tail_divisor = 0U;
    } else {
        out.prefill_tail_divisor = mini_prefill_schedule_from_env(hip).tail_divisor;
    }
    // The scheduler's own checks (serve::validate_prompt_pieces) with the decoder's declared cut rule, before any device
    // work: inspection refuses what the launch would refuse. The device class's gap (unset here) is data-validated.
    try {
        const auto rule = mini_serve_piece_rule(hip, out.served_identity, out.form_rows, out.prefill_rows);
        serve::validate_prompt_pieces(scheduling(out, hip, {}), rule, out.prefill_rows);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Inkling served prefill (--prefill-rows, --prefill-rows-decoding, "
                                                "--prefill-gap-ms, --prompt-share): ") + e.what());
    }
    if (!(resolved.backend == "hip" ? backends.hip : backends.cuda))
        throw std::invalid_argument("Inkling backend " + std::string(resolved.backend) + " is not built");
    return resolved;
}

serve::CohortConfig scheduling(const MiniServeOptions& opt, bool hip, std::string_view device_class) {
    const auto data = mini_serve_prefill(hip, device_class);
    serve::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    config.prompt_slice = opt.prefill_rows;
    // Regimes: alone, a prompt runs whole chunks; beside a decoding lane, bounded pieces of the same chunks, sized by
    // the explicit gap target or the device class's (none without pieces).
    // (the piece rows: an explicit --prefill-rows-decoding, else the device class's data row, else the backend's)
    config.prompt_slice_decoding = mini_resolve_decoding_rows(opt.prefill_rows_decoding, opt.prefill_rows, hip, device_class);
    config.prompt_gap_ms = opt.prefill_gap_ms >= 0.0 ? opt.prefill_gap_ms
                                                     : (config.prompt_slice_decoding != 0U ? data.gap_ms : 0.0);
    config.prompt_pass_fixed_ms = config.prompt_gap_ms > 0.0 ? data.pass_fixed_ms : 0.0;
    config.prompt_share = opt.prompt_share >= 0.0 ? opt.prompt_share : data.decode_share;  // the class's data
    return config;
}

class MiniService final : public serve::SessionService {
public:
    MiniService(const std::string& model, const MiniServeOptions& opt, const MiniServeBackend& backend,
                const MiniServeTokenizer& tokenizer, MiniChat chat)
        : tokenizer_(tokenizer(opt.tokenizer)), device_(backend(opt)),
          queue_(std::max(64U, device_.backend->lanes())),
          cohort_(queue_, *device_.backend, scheduling(opt, device_.hip, mini_device_class(device_.device_identity)),
                  [min_prompt = mini_served_identity(opt.served_identity).emits_first ? 1U : 2U](
                      const serve::GenerationJob& job) {
              if (job.copy_rows) throw std::invalid_argument("Inkling serving does not support copy_rows");
              if (job.window_rows) throw std::invalid_argument("Inkling serving does not support window_rows");
              // served-cli-prefill prefills the whole prompt (one token is a prompt); served-incumbent decodes the final
              // prompt token, so its prompt has at least two (main's rule)
              if (job.prompt.size() < min_prompt ||
                  std::ranges::any_of(job.prompt, [](auto id) { return id >= kVocabulary; }))
                  throw std::invalid_argument("invalid Inkling prompt");
          }),
          api_(api_config(model, opt.tier, std::move(chat)), tokenizer_, queue_, device_.backend->max_positions(),
               device_.backend->supports_final_logits()) {
        // The resolved served prefill and its data row: receipts state the schedule and the decode route that ran.
        const auto device_class = mini_device_class(device_.device_identity);
        const auto data = mini_serve_prefill(device_.hip, device_class);
        const auto config = scheduling(opt, device_.hip, device_class);
        const auto& identity = mini_served_identity(opt.served_identity);
        std::cerr << "Inkling served identity " << identity.name << " reference " << identity.reference
                  << " form_rows " << opt.form_rows << " served prefill chunk " << config.prompt_slice << " tail_divisor " << opt.prefill_tail_divisor
                  << " pieces " << config.prompt_slice_decoding << " gap_ms " << config.prompt_gap_ms
                  << " pass_fixed_ms " << config.prompt_pass_fixed_ms << " share " << config.prompt_share
                  << " device_class " << (device_class.empty() ? std::string_view("-") : device_class)
                  << " moe_decode grouped (data: " << data.receipt << ")\n";
    }

    serve::SessionRuntime& runtime() override { return cohort_.runtime(); }
    void configure_listener(serve::HttpServerConfig& config) const override {
        config.max_connections = std::max(config.max_connections, 2U * device_.backend->lanes());
    }
    void listening(const serve::HttpServerConfig& config, std::uint16_t port) override {
        std::cerr << "Inkling listening " << config.bind_address << ':' << port << '\n';
    }
    void handle(const serve::HttpRequest& request, serve::HttpResponseWriter& writer) override {
        api_.handle(request, writer);
    }

private:
    serve::ApiConfig api_config(const std::string& model, profile::Name tier, MiniChat chat) {
        serve::ApiConfig config;
        cohort_.configure_api(config);
        config.capabilities_extra = [cohort = std::move(config.capabilities_extra), tier,
                                     weights = device_.weights](serve::Json& value) {
            cohort(value);
            value.set("profile", serve::Json(mini_profile_name(tier)));
            value.set("weight_load", serve::Json(serve::Json::Object{
                {"storage", serve::Json(weights.storage.name)},
                {"revision", serve::Json(static_cast<std::uint64_t>(weights.storage.revision))},
                {"status", serve::Json(profile::status_name(weights.status))},
                {"reason", serve::Json(weights.reason)},
            }));
            value.set("speculation", serve::Json("off"));
        };
        config.model_id = model;
        config.default_thinking = false;
        config.default_max_tokens = 256;
        config.stop_tokens = {kMiniEosToken};
        config.responses = false;
        config.ollama_api = false;
        config.prepare_chat = [tokenizer = tokenizer_, render = std::move(chat)](const serve::Json& body) {
            for (const char* field : {"dspark", "speculation"}) {
                if (const auto* value = body.find(field); value && !value->is_null())
                    throw serve::ChatTemplateError(std::string("Inkling serving does not support ") + field);
            }
            if (const auto* tools = body.find("tools"); tools && tools->size())
                throw serve::ChatTemplateError("Inkling serving supports text messages");
            const auto* messages = body.find("messages");
            if (!messages || !messages->is_array() || messages->as_array().empty())
                throw serve::ChatTemplateError("messages must be a nonempty array");
            for (const auto& message : messages->as_array()) {
                const auto* role = message.find("role");
                const auto* calls = message.find("tool_calls");
                if ((role && role->is_string() && role->as_string() == "tool") || (calls && calls->size()))
                    throw serve::ChatTemplateError("Inkling serving supports text messages");
            }
            const auto* effort = body.find("reasoning_effort");
            if (effort && !effort->is_null() && (!effort->is_string() || effort->as_string() != "none"))
                throw serve::ChatTemplateError("Inkling serving supports reasoning_effort none");
            serve::ApiChatPrompt prompt;
            prompt.tokens = tokenizer->encode(render(*messages, {}, serve::Json("none")), true);
            return prompt;
        };
        config.output_parser = [](const std::string& text, bool, const serve::Json&) {
            serve::AssistantOutput out;
            out.content = text;
            return out;
        };
        return config;
    }

    std::shared_ptr<const serve::BpeTokenizer> tokenizer_;
    MiniServeCohort device_;
    serve::GenerationQueue queue_;
    serve::CohortSession cohort_;
    serve::ApiService api_;
};

} // namespace

void install_mini_service(serve::ServiceFactories& factories, MiniServeBackends backends) {
    install_mini_service(factories, std::move(backends), [](const std::string& path) {
        return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
    }, [](const serve::Json& messages, const serve::Json& tools, const serve::Json& reasoning_effort) {
        return serve::render_inkling2_chat(messages, tools, reasoning_effort);  // HTTP: text only, image parts refused
    });
}

void install_mini_service(serve::ServiceFactories& factories, MiniServeBackends backends,
                          MiniServeTokenizer tokenizer, MiniChat chat) {
    const bool serves_cuda = static_cast<bool>(backends.cuda);
    serve::ServiceFactory factory{
        [backends](const auto& selection) { (void)resolve(selection, backends); },
        [backends](const auto& selection, std::ostream& out) {
            const auto resolved = resolve(selection, backends);
            const auto& opt = resolved.options;
            out << "backend " << resolved.backend << " context " << opt.context << " prefill_rows " << opt.prefill_rows
                << " prefill_tail_divisor " << opt.prefill_tail_divisor
                << " prefill_rows_decoding ";
            // unset: the device class's data, resolved at launch (models/inkling2-mini.toml [serve_prefill]); a backend
            // whose row runs no pieces at this chunk prints 0
            const bool pieces = mini_resolve_decoding_rows(opt.prefill_rows_decoding, opt.prefill_rows,
                                                           resolved.backend == "hip") != 0U;
            if (opt.prefill_rows_decoding != 0U) out << opt.prefill_rows_decoding;
            else if (!pieces) out << 0;
            else out << "device-class";
            out << " prefill_gap_ms ";
            if (opt.prefill_gap_ms >= 0.0) out << opt.prefill_gap_ms;
            else if (!pieces) out << 0;
            else out << "device-class";
            out << " prompt_share ";
            if (opt.prompt_share >= 0.0) out << opt.prompt_share;
            else out << "device-class";
            const auto& identity = mini_served_identity(opt.served_identity);
            out << " served_identity " << identity.name << " reference " << identity.reference << " form_rows "
                << opt.form_rows
                << " slot_cap " << opt.slots << " kv_budget_bytes " << opt.kv_bytes
                << " profile " << mini_profile_name(opt.tier) << " speculation off";
            if (kMiniWeightLoadDecisions.empty() || opt.tier == profile::Name::standard || opt.weight_options.disable_q8) {
                const auto backend = resolved.backend == "hip" ? weight::Backend::hip : weight::Backend::cuda;
                const weight::WeightLoadScope scope{backend, {}, {}, {}, {}, {}};
                const auto plan = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope);
                out << " weight_storage " << plan.storage.name << '@' << plan.storage.revision
                    << " weight_status " << profile::status_name(plan.status);
            } else {
                out << " weight_status pending-device";
            }
            out << '\n';
        },
        [backends = std::move(backends), tokenizer = std::move(tokenizer), chat = std::move(chat)]
            (const auto& selection) {
            const auto resolved = resolve(selection, backends);
            const auto& backend = resolved.backend == "hip" ? backends.hip : backends.cuda;
            return std::make_unique<MiniService>(selection.model.name, resolved.options, backend, tokenizer, chat);
        }
    };
    factory.serves_cuda = serves_cuda;  // the CUDA backend runs in process, not through the shell launcher
    factories.install("inkling2", std::move(factory));
}

} // namespace strix::inkling2
