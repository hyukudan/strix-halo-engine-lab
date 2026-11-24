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
    bool packed_flag = false;   // --packed-decode / --grouped-decode, as on the command line
    bool grouped_flag = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view name = args[i];
        if (name == "--dspark") throw std::invalid_argument("Inkling serving does not support --dspark");
        if (name == "--packed-decode" || name == "--grouped-decode") {
            (name == "--packed-decode" ? packed_flag : grouped_flag) = true;
            continue;
        }
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
        else if (name == "--prefill-rows") {
            out.prefill_rows = static_cast<unsigned>(number(name, value, 65535U));
            if (out.prefill_rows == 0U) throw std::invalid_argument("--prefill-rows must be positive");
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
    out.packed_decode = mini_packed_decode(packed_flag, grouped_flag);
    out.combine_image = mini_combine_image(resolution);
    if (out.checkpoint.empty()) throw std::invalid_argument("Inkling requires --checkpoint FILE");
    if (out.tokenizer.empty())
        out.tokenizer = (std::filesystem::path(out.checkpoint).parent_path() / "tokenizer.json").string();
    if (out.context < 3U) throw std::invalid_argument("Inkling context must hold a prompt and a reply");
    if (resolved.backend.empty()) resolved.backend = backends.hip ? "hip" : "cuda";
    if (!(resolved.backend == "hip" ? backends.hip : backends.cuda))
        throw std::invalid_argument("Inkling backend " + std::string(resolved.backend) + " is not built");
    // The command line's prefill schedule for this backend (one resolver): the same chunk and tail rule.
    const auto schedule = mini_prefill_schedule_from_env(resolved.backend == "hip");
    if (out.prefill_rows == 0U) out.prefill_rows = schedule.chunk;
    out.prefill_rows = std::min(out.prefill_rows, out.context);
    out.prefill_tail = mini_prefill_tail_capacity(out.prefill_rows, schedule.tail_divisor, out.context);
    return resolved;
}

serve::CohortConfig scheduling(const MiniServeOptions& opt, const serve::CohortBackend& backend) {
    serve::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    // A prompt runs as the command line's calls (whole chunks, the final one with its tail); beside a decoding lane
    // the cohort cuts pieces of kMiniServeDecodingRows inside those chunks where the decoder's piece rule allows (HIP),
    // so the bits never depend on other lanes (the backend's prompt_piece_rule; CUDA: none, whole chunks).
    config.prompt_slice = opt.prefill_rows;
    config.prompt_slice_decoding = backend.prompt_piece_rule().cuttable() && opt.prefill_rows > kMiniServeDecodingRows ? kMiniServeDecodingRows : 0U;
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
          api_(api_config(model, opt.tier, std::move(chat)), tokenizer_, queue_, device_.backend->max_positions(),
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
                << " prefill_tail " << opt.prefill_tail << " first_id prefill"
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
