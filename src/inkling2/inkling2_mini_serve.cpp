#include "inkling2_mini_serve.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/inkling2_chat.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>
#include <ostream>
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

ResolvedService resolve(const serve::ServiceSelection& selection, const MiniServeBackends& backends) {
    ResolvedService resolved;
    auto& out = resolved.options;
    const auto& args = selection.arguments;
    std::optional<unsigned> explicit_rows;  // --prefill-rows
    std::optional<bool> explicit_speculation;  // --speculation off|auto
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
        else if (name == "--prefill-rows") explicit_rows = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--slots") out.slots = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--kv-bytes") out.kv_bytes = number(name, value, UINT64_MAX);
        else if (name == "--speculation") {
            if (value != "off" && value != "auto") throw std::invalid_argument("Inkling --speculation must be off or auto");
            explicit_speculation = value == "auto";
        } else throw std::invalid_argument("unknown Inkling argument " + std::string(name));
    }
    const auto resolution = resolve_mini_profile(out.weight_options, true);
    out.tier = resolution.profile;
    out.weight_options.lowbit_turbo = mini_lowbit_turbo(resolution);
    out.prepared_round_fold = mini_prepared_round_fold(resolution);
    out.experts_nvfp4 = mini_experts_nvfp4(resolution);
    out.combine_image = mini_combine_image(resolution);
    out.served_route_requested = mini_served_cli_route(resolution);
    out.attn_projected_bias = mini_attn_projected_bias(resolution);
    out.screened_head = mini_screened_head(resolution);
    // One speculation row for the command line and the server (an explicit --speculation wins); its verify windows
    // take the verify rows body.
    out.speculation_requested = explicit_speculation.value_or(mini_speculation(resolution));
    out.verify_rows = mini_speculation_verify_rows(mini_verify_rows(resolution), out.speculation_requested);
    out.controller_one_plane = mini_controller_one_plane(resolution);
    if (out.checkpoint.empty()) throw std::invalid_argument("Inkling requires --checkpoint FILE");
    if (out.tokenizer.empty())
        out.tokenizer = (std::filesystem::path(out.checkpoint).parent_path() / "tokenizer.json").string();
    if (out.context < 3U) throw std::invalid_argument("Inkling context must hold a prompt and a reply");
    if (resolved.backend.empty()) resolved.backend = backends.hip ? "hip" : "cuda";
    if (!(resolved.backend == "hip" ? backends.hip : backends.cuda))
        throw std::invalid_argument("Inkling backend " + std::string(resolved.backend) + " is not built");
    // Served verify windows are admitted on HIP (the verify rows body and per-slot tapes); a CUDA server declines.
    out.speculation = out.speculation_requested && resolved.backend == "hip";
    // The W4A4 class is admitted for the resolved backend (HIP refuses a selected class instead of running BF16).
    out.moe_w4a4 = admit_mini_moe_w4a4(resolution, resolved.backend);
    // Backend admission of the served route lever (kMiniServedCliRoute): HIP runs the command line's route, a CUDA
    // server declines it (requested stays in the receipt, mini_served_route_startup_field).
    out.served_cli_route = mini_served_prefill_emits_first(resolved.backend == "hip", out.served_route_requested);
    // The prompt chunk: --prefill-rows, else the route's (the incumbent 128-row held-back chunks without the lever,
    // the command line's chunk and tail rule with it), clamped to the context. Lanes: mini_serve_lane_rows.
    const auto schedule = mini_served_prefill_schedule_from_env(true, out.served_cli_route, out.tier);
    out.prefill_rows = std::min(explicit_rows.value_or(schedule.chunk), out.context);
    if (out.served_cli_route) {
        if (out.prefill_rows == 0U) throw std::invalid_argument("--prefill-rows must be positive");
        // An explicit chunk without --slots also caps the lanes (the clamped chunk), as on the incumbent route.
        if (explicit_rows) out.lane_rows = out.prefill_rows;
        out.prefill_tail = mini_prefill_tail_capacity(out.prefill_rows, schedule.tail_divisor, out.context);
    }
    return resolved;
}

serve::CohortConfig scheduling(const MiniServeOptions& opt, const serve::CohortBackend& backend) {
    serve::CohortConfig config;
    // Admitted speculation: each greedy lane drafts from its own context (the shared cohort drafter) and verifies the
    // window on its slot; draft rows by live lanes are the model's data (drafts only while one lane is live).
    config.context_draft = opt.speculation;
    config.max_draft = opt.speculation ? kMiniMaxVerifyRows - 1U : 0U;
    if (opt.speculation) config.max_draft_by_live = mini_served_draft_by_live();
    // A prompt runs as the command line's calls (whole chunks, the final one with its tail); beside a decoding lane
    // the cohort cuts pieces of kMiniServeDecodingRows inside those chunks where the decoder's piece rule allows (HIP),
    // so the bits never depend on other lanes (the backend's prompt_piece_rule; CUDA: none, whole chunks).
    config.prompt_slice = opt.prefill_rows;
    config.prompt_slice_decoding = opt.served_cli_route && backend.prompt_piece_rule().cuttable() &&
                                   opt.prefill_rows > kMiniServeDecodingRows ? kMiniServeDecodingRows : 0U;
    return config;
}

class MiniService final : public serve::SessionService {
public:
    MiniService(const std::string& model, const MiniServeOptions& opt, const MiniServeBackend& backend,
                const MiniServeTokenizer& tokenizer, MiniChat chat)
        : tokenizer_(tokenizer(opt.tokenizer)), device_(backend(opt)),
          queue_(std::max(64U, device_.backend->lanes())),
          cohort_(queue_, *device_.backend, scheduling(opt, *device_.backend), [](const serve::GenerationJob& job) {
              if (job.copy_rows) throw std::invalid_argument("Inkling serving does not support copy_rows");
              if (job.window_rows) throw std::invalid_argument("Inkling serving does not support window_rows");
              if (job.prompt.size() < 2 ||
                  std::ranges::any_of(job.prompt, [](auto id) { return id >= kVocabulary; }))
                  throw std::invalid_argument("invalid Inkling prompt");
          }),
          api_(api_config(model, opt, std::move(chat)), tokenizer_, queue_, device_.backend->max_positions(),
               device_.backend->supports_final_logits()) {}

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
    // The command line's projected-bias observation, once the owner has stopped: what the served requests submitted.
    void stopped() override {
        if (device_.projected_bias_dispatches)
            std::cerr << mini_projected_bias_dispatches_line(device_.projected_bias_dispatches()) << std::endl;
    }

private:
    serve::ApiConfig api_config(const std::string& model, const MiniServeOptions& opt, MiniChat chat) {
        serve::ApiConfig config;
        cohort_.configure_api(config);
        config.capabilities_extra = [cohort = std::move(config.capabilities_extra), tier = opt.tier,
                                     requested = opt.served_route_requested, admitted = opt.served_cli_route,
                                     weights = device_.weights,
                                     speculation = std::string(mini_served_speculation_name(opt))](serve::Json& value) {
            cohort(value);
            value.set("profile", serve::Json(mini_profile_name(tier)));
            // The served route lever, only when requested (absent: the incumbent route, as before the lever): what was
            // requested, what this backend admitted, and the qualification of the admitted route.
            if (requested)
                value.set("served_route", serve::Json(serve::Json::Object{
                    {"requested", serve::Json(true)},
                    {"admitted", serve::Json(admitted)},
                    {"qualification", serve::Json(admitted ? "unqualified" : "n/a")},
                }));
            value.set("weight_load", serve::Json(serve::Json::Object{
                {"storage", serve::Json(weights.storage.name)},
                {"revision", serve::Json(static_cast<std::uint64_t>(weights.storage.revision))},
                {"head_storage", serve::Json(weights.head_storage.name)},
                {"status", serve::Json(profile::status_name(weights.status))},
                {"reason", serve::Json(weights.reason)},
            }));
            value.set("speculation", serve::Json(speculation));
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
                << " served_route " << mini_served_route_startup_field(opt)
                << " slot_cap " << opt.slots << " kv_budget_bytes " << opt.kv_bytes
                << " profile " << mini_profile_name(opt.tier) << " speculation " << mini_served_speculation_name(opt);
            const auto backend = resolved.backend == "hip" ? weight::Backend::hip : weight::Backend::cuda;
            // Without a device the plan is exact unless a full row (gate + gain) of this backend could select for this
            // instance's extent there (the same extent make_mini_device_cohort resolves with).
            const auto of_backend = [backend](const weight::WeightLoadDecision& row) { return row.scope.backend == backend; };
            const weight::WeightLoadExtent extent{opt.context, opt.slots};
            const bool device_dependent = opt.tier != profile::Name::standard && !opt.weight_options.disable_q8 &&
                ((opt.weight_options.qualify != nullptr && !opt.weight_options.qualify_q8) ||
                 mini_may_load_lowbit_row(opt.tier, opt.weight_options, backend, extent) ||
                 std::ranges::any_of(kMiniWeightLoadDecisions, [&](const weight::WeightLoadDecision& row) {
                     return of_backend(row) && weight::weight_load_gain_selects(row, extent);
                 }));
            if (!device_dependent) {
                const weight::WeightLoadScope scope{backend, {}, {}, {}, {}, {}};
                const auto plan = resolve_mini_resident_weight_load(opt.tier, opt.weight_options, scope, extent);
                out << " weight_storage " << plan.storage.name << '@' << plan.storage.revision
                    << (plan.head_storage != plan.storage ? " weight_head_storage " : "")
                    << (plan.head_storage != plan.storage ? plan.head_storage.name : "")
                    << " weight_status " << profile::status_name(plan.status);
            } else {
                out << " weight_status pending-device";
            }
            if (std::ranges::any_of(kMiniWeightLoadDecisions, of_backend))
                out << " weight_rows " << weight::registered_regime_rows(kMiniWeightLoadPolicy, backend,
                                                                         kMiniWeightLoadRegimes, kMiniWeightLoadDecisions);
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
