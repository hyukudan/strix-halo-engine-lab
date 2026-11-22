#include "inkling2_mini_serve.hpp"
#include "inkling2_mini_config.hpp"

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/inkling2_chat.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
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
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view name = args[i];
        if (++i == args.size()) throw std::invalid_argument("missing value for " + std::string(name));
        const auto& value = args[i];
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
        else if (name == "--prefill-rows") out.prefill_rows = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--slots") out.slots = static_cast<unsigned>(number(name, value, 65535U));
        else if (name == "--kv-bytes") out.kv_bytes = number(name, value, UINT64_MAX);
        else throw std::invalid_argument("unknown Inkling argument " + std::string(name));
    }
    if (out.checkpoint.empty()) throw std::invalid_argument("Inkling requires --checkpoint FILE");
    if (out.tokenizer.empty())
        out.tokenizer = (std::filesystem::path(out.checkpoint).parent_path() / "tokenizer.json").string();
    if (out.context < 3U) throw std::invalid_argument("Inkling context must hold a prompt and a reply");
    out.prefill_rows = std::min(out.prefill_rows, out.context);
    if (resolved.backend.empty()) resolved.backend = backends.hip ? "hip" : "cuda";
    if (!(resolved.backend == "hip" ? backends.hip : backends.cuda))
        throw std::invalid_argument("Inkling backend " + std::string(resolved.backend) + " is not built");
    return resolved;
}

serve::CohortConfig scheduling(const MiniServeOptions& opt) {
    serve::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    config.prompt_slice = opt.prefill_rows;
    return config;
}

class MiniService final : public serve::SessionService {
public:
    MiniService(const std::string& model, const MiniServeOptions& opt, const MiniServeBackend& backend,
                const MiniServeTokenizer& tokenizer, MiniChat chat)
        : tokenizer_(tokenizer(opt.tokenizer)), backend_(backend(opt)),
          queue_(std::max(64U, backend_->lanes())),
          cohort_(queue_, *backend_, scheduling(opt), [](const serve::GenerationJob& job) {
              if (job.prompt.size() < 2 || job.copy_rows || job.window_rows ||
                  std::ranges::any_of(job.prompt, [](auto id) { return id >= kVocabulary; }))
                  throw std::invalid_argument("invalid Inkling prompt or decode policy");
          }),
          api_(api_config(model, std::move(chat)), tokenizer_, queue_, backend_->max_positions(), false) {}

    serve::SessionRuntime& runtime() override { return cohort_.runtime(); }
    void configure_listener(serve::HttpServerConfig& config) const override {
        config.max_connections = std::max(config.max_connections, 2U * backend_->lanes());
    }
    void handle(const serve::HttpRequest& request, serve::HttpResponseWriter& writer) override {
        api_.handle(request, writer);
    }

private:
    serve::ApiConfig api_config(const std::string& model, MiniChat chat) {
        serve::ApiConfig config;
        cohort_.configure_api(config);
        config.model_id = model;
        config.default_thinking = false;
        config.default_max_tokens = 256;
        config.stop_tokens = {kMiniEosToken};
        config.responses = false;
        config.ollama_api = false;
        config.prepare_chat = [tokenizer = tokenizer_, render = std::move(chat)](const serve::Json& body) {
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
    std::unique_ptr<serve::CohortBackend> backend_;
    serve::GenerationQueue queue_;
    serve::CohortSession cohort_;
    serve::ApiService api_;
};

} // namespace

void install_mini_service(serve::ServiceFactories& factories, MiniServeBackends backends) {
    install_mini_service(factories, std::move(backends), [](const std::string& path) {
        return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
    }, serve::render_inkling2_chat);
}

void install_mini_service(serve::ServiceFactories& factories, MiniServeBackends backends,
                          MiniServeTokenizer tokenizer, MiniChat chat) {
    factories.install("inkling2", {
        [backends](const auto& selection) { (void)resolve(selection, backends); },
        [backends](const auto& selection, std::ostream& out) {
            const auto resolved = resolve(selection, backends);
            const auto& opt = resolved.options;
            out << "backend " << resolved.backend << " context " << opt.context << " prefill_rows " << opt.prefill_rows
                << " slot_cap " << opt.slots << " kv_budget_bytes " << opt.kv_bytes << '\n';
        },
        [backends = std::move(backends), tokenizer = std::move(tokenizer), chat = std::move(chat)]
            (const auto& selection) {
            const auto resolved = resolve(selection, backends);
            const auto& backend = resolved.backend == "hip" ? backends.hip : backends.cuda;
            return std::make_unique<MiniService>(selection.model.name, resolved.options, backend, tokenizer, chat);
        }
    });
}

} // namespace strix::inkling2
