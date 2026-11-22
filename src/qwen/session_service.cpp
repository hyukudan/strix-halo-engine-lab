#include "qwen/session_service.hpp"

#include "hip_decode_options.hpp"
#include "strix/device/gate.hpp"
#include "strix/device/topology.hpp"
#include "strix/qwen_model_description.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/request_fields.hpp"

#include <algorithm>
#include <ostream>

namespace strix::qwen {
namespace {
struct Prepared final : serve::ServicePreparation {
    std::vector<std::string> arguments;
    serve::Json inspection;
    std::string config_output;
    bool inspect_native = false;
};

const Prepared& prepared(const serve::ServiceSelection& selection) {
    const auto result = std::dynamic_pointer_cast<const Prepared>(selection.preparation);
    if (!result) throw std::invalid_argument("Qwen service has not passed validation");
    return *result;
}
} // namespace

serve::Json inspect_http_options(const decode_cli::Options& options) {
    auto out = serve::Json::object();
    out.set("phase", serve::Json("arguments"));
    out.set("model", serve::Json(options.http_model_id));
    out.set("context", serve::Json(options.context_tokens));
    out.set("baseline", serve::Json(options.baseline));
    out.set("batch_cap", serve::Json(static_cast<std::uint64_t>(options.http_batch_cap)));
    out.set("batch_explicit", serve::Json(options.http_batch_explicit));
    out.set("slot_admission", serve::Json(options.http_slot_reason));
    const auto sampling = decode_cli::resolve_http_sampling(options, decode_cli::ordinary_http_batch_slots(options) != 0);
    out.set("sampling_policy", serve::Json(std::string(sampling.policy)));
    if (options.http_default_temperature) out.set("default_temperature", serve::Json(*options.http_default_temperature));
    out.set("logprobs_requested", serve::Json(options.http_logprobs));
    out.set("structured_output_requested", serve::Json(options.http_constraints));
    out.set("session_resume_requested", serve::Json(options.session_management_requested));
    out.set("capabilities_endpoint", serve::Json(options.http_capabilities));
    out.set("max_tokens_default", serve::Json(static_cast<std::uint64_t>(options.http_default_max_tokens)));
    out.set("model_and_device_admission", serve::Json("pending startup"));
    return out;
}

void bind_http_session_support(serve::ApiConfig& config, bool enabled) {
    config.sessions_enabled = enabled;
    config.capabilities_extra = [extra = std::move(config.capabilities_extra), enabled](serve::Json& out) {
        if (extra) extra(out);
        auto cancellation = serve::Json::object();
        cancellation.set("client_disconnect", serve::Json(true));
        out.set("cancellation", std::move(cancellation));
        out.set("session_resume", serve::Json(enabled));
    };
    if (!enabled) config.validate_request = [](const serve::HttpRequest&, const serve::Json& body) {
        if (const auto* session = body.find("session"); session && !session->is_null())
            throw serve::RequestFieldError("session", "named session resume is disabled for this Qwen instance");
    };
}

void bind_http_sampled_fields(serve::ApiConfig& config) {
    config.prepare_job = [](const serve::HttpRequest&, const serve::Json& body, serve::GenerationJob& job) {
        if (job.sampling.temperature == 0.0) return;
        const auto refuse = [](bool requested, const char* field) {
            if (requested) throw serve::RequestFieldError(field, std::string(field) + " is unavailable with sampling");
        };
        refuse(job.ngram_draft.value_or(false), "ngram_draft");
        refuse(job.ngram_peers.value_or(false), "ngram_peers");
        refuse(job.tool_draft.value_or(false), "tool_draft");
        for (const auto* field : {"copy_rows", "window_rows"}) {
            const auto* value = body.find(field);
            refuse(value && !value->is_null(), field);
        }
    };
}

void install_session_service(serve::ServiceFactories& factories, NativeHttp launch) {
    if (!launch) throw std::invalid_argument("Qwen service requires a native startup binding");
    serve::ServiceFactory factory;
    factory.validate = [](const serve::ServiceSelection& selection) {
        qwen_description::assert_matches(ModelContract::qwen38_flash_next(), selection.description);
        auto result = std::make_shared<Prepared>();
        result->arguments = {"strix-serve"};
        result->arguments.insert(result->arguments.end(), selection.arguments.begin(), selection.arguments.end());
        if (std::find(selection.arguments.begin(), selection.arguments.end(), "--http-serve") == selection.arguments.end())
            result->arguments.emplace_back("--http-serve");
        auto arguments = result->arguments;
        std::vector<char*> argv;
        for (auto& arg : arguments) argv.push_back(arg.data());
        int argc = static_cast<int>(argv.size());
        argv.push_back(nullptr);
        (void)device::take_command_line(argc, argv.data());
        (void)device::take_topology_command_line(argc, argv.data());
        const auto config = serve::config_file::expand(argc, argv.data(), decode_cli::hip_serve_config_options());
        if (config.print) {
            result->config_output = config.effective;
        } else {
            if (argc < 3 || std::string_view(argv[1]).empty() || std::string_view(argv[1]).starts_with("--"))
                throw std::invalid_argument("Qwen serving requires MODEL.gguf as its first family argument");
            const auto options = decode_cli::parse_options(argc, argv.data(), true, {}, decode_cli::DecodeAssets::defer);
            result->inspection = inspect_http_options(options);
            result->inspect_native = options.print_effective_config || options.deploy_check ||
                options.print_model_graph || options.print_plan || options.print_cache_family;
        }
        selection.preparation = std::move(result);
    };
    factory.inspect = [](const serve::ServiceSelection& selection, std::ostream& out) {
        out << prepared(selection).inspection.dump() << '\n';
    };
    factory.early_exit = [](const serve::ServiceSelection& selection, std::ostream& out) -> std::optional<int> {
        const auto& config = prepared(selection).config_output;
        if (config.empty()) return {};
        out << config;
        return 0;
    };
    factory.create_scoped = [launch = std::move(launch)](const serve::ServiceSelection& selection,
            const serve::ServiceFactory::ServeScoped& run_service) {
        const auto& plan = prepared(selection);
        bool entered = false;
        const int status = launch(plan.arguments, [&](serve::SessionService& service) {
            if (entered) throw std::logic_error("Qwen service constructed more than one HTTP owner");
            entered = true;
            run_service(service);
        });
        if (status == 0 && !entered && !plan.inspect_native)
            throw std::runtime_error("Qwen startup did not construct an HTTP owner");
        return status;
    };
    factories.install("qwen", std::move(factory));
}
} // namespace strix::qwen
