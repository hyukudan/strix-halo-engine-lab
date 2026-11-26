#include "qwen/session_service.hpp"

#include "hip_decode_options.hpp"
#include "session_replay/legacy_reference.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
namespace cli = strix::decode_cli;
namespace qwen = strix::qwen;
using namespace session_fixture;

struct Invocation {
    int code;
    std::string out, error;
};

Invocation invoke(std::vector<std::string> args, const sv::ServiceFactories& factories,
                  const sv::ServeShell& shell = {}) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    std::ostringstream out, error;
    const auto result = sv::run(static_cast<int>(argv.size()), argv.data(), factories, out, error, shell);
    return {result, out.str(), error.str()};
}

cli::Options legacy_options(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return cli::parse_options(static_cast<int>(argv.size()), argv.data(), true, {}, cli::DecodeAssets::defer);
}

void check_http_capabilities() {
    const auto tokenizer = sv::BpeTokenizer::from_parts({"A", "<|endoftext|>"}, {1, 3}, {}, "qwen35");
    for (const bool enabled : {false, true}) {
        sv::GenerationQueue queue;
        sv::ApiConfig config;
        config.engine_logprobs = enabled;
        config.engine_constraints = enabled;
        config.sampling_policy = enabled ? "sample" : "reject";
        qwen::bind_http_session_support(config, enabled);
        sv::ApiService api(config, tokenizer, queue, 128, enabled);
        sv::HttpRequest request;
        request.method = "GET";
        request.path = "/v1/capabilities";
        Writer capabilities;
        api.handle(request, capabilities);
        CHECK(capabilities.status == 200);
        const auto document = sv::parse_json(capabilities.body);
        CHECK(document.find("stream")->find("sse")->as_bool());
        CHECK(document.find("cancellation")->find("client_disconnect")->as_bool());
        CHECK(document.find("sampling")->find("sample")->as_bool() == enabled);
        CHECK(document.find("logprobs")->as_bool() == enabled);
        CHECK(document.find("json_schema")->as_bool() == enabled);
        CHECK(document.find("function_tools")->as_bool());
        CHECK(document.find("session_resume")->as_bool() == enabled);

        if (!enabled) {
            request.method = "POST";
            request.path = "/v1/chat/completions";
            request.body = R"({"session":"parked"})";
            Writer response;
            api.handle(request, response);
            CHECK(response.status == 400);
            CHECK(response.body.find("named session resume is disabled") != std::string::npos);
            CHECK(queue.snapshot().queued == 0);
        }
    }
}

class NativeFixture final : public sv::SessionService {
public:
    explicit NativeFixture(bool& stopped) : stopped_(stopped) {}
    sv::SessionRuntime& runtime() override { return runtime_; }
    void stopped() override { stopped_ = true; }
    void handle(const sv::HttpRequest&, sv::HttpResponseWriter& writer) override {
        writer.send(200, "text/plain", "native fixture", {});
    }
private:
    bool& stopped_;
    FakeEngine engine_{1};
    sv::GenerationQueue queue_;
    sv::SessionRuntime runtime_{queue_, engine_, engine_.store, {}, runtime_hooks(engine_)};
};

void check_factory(const std::string& root) {
    sv::ServiceFactories production;
    sv::install_service_factories(production);
#ifdef STRIX_EXPECT_QWEN_SERVICE
    CHECK(production.find("qwen"));
#else
    CHECK(!production.find("qwen"));
#endif
    CHECK(!production.find("qwen35"));

    sv::ServiceFactories factories;
    unsigned launched = 0;
    bool stopped = false;
    std::string launched_options;
    const auto launch = [&](const std::vector<std::string>& args, const sv::ServiceFactory::ServeScoped& serve) {
        ++launched;
        CHECK(args.back() == "--http-serve");
        launched_options = qwen::inspect_http_options(legacy_options(args)).dump();
        NativeFixture service(stopped);
        serve(service);
        CHECK(stopped);
        return 0;
    };
    qwen::install_session_service(factories, launch);
    CHECK(factories.find("qwen"));
    bool duplicate = false;
    try { qwen::install_session_service(factories, launch); }
    catch (const std::invalid_argument& error) { duplicate = std::string_view(error.what()) == "duplicate service factory"; }
    CHECK(duplicate);

    const std::vector<std::string> select{"strix-serve", "--registry-root", root, "--model", "qwen", "--port", "0"};
    const auto bundle = (std::filesystem::path(root) / "recipes/qwen-distill-g-yarn.mtp-bundle.json").string();
    std::vector<std::string> family{"fixture-missing.gguf"};
    for (unsigned id = 1; id <= 17; ++id) family.push_back(std::to_string(id));
    family.insert(family.end(), {"--generate", "8", "--context", "8192", "--mtp-bundle", bundle});
    CHECK(!std::filesystem::exists(family.front()));

    const std::vector<std::vector<std::string>> cases{
        {},
        {"--http-batch", "0", "--http-sampling", "reject", "--http-logprobs", "off"},
        {"--http-batch", "2", "--http-sampling", "sample",
         "--http-constraints", "off", "--http-default-max-tokens", "24"},
        {"--http-batch", "2", "--http-slot-build", "admission"}};
    for (const auto& extra : cases) {
        auto legacy = family;
        legacy.insert(legacy.begin(), "strix-hip-decode-token");
        legacy.emplace_back("--http-serve");
        legacy.insert(legacy.end(), extra.begin(), extra.end());
        const auto resolved = legacy_options(legacy);
        CHECK(resolved.http_build_slots_at_admission ==
              (std::find(extra.begin(), extra.end(), "admission") != extra.end()));
        const auto expected = qwen::inspect_http_options(resolved);
        auto inspect = select;
        inspect.insert(inspect.end(), {"--family-inspect", "--"});
        inspect.insert(inspect.end(), family.begin(), family.end());
        inspect.insert(inspect.end(), extra.begin(), extra.end());
        const auto result = invoke(inspect, factories);
        if (result.code != 0) throw std::runtime_error(result.error);
        CHECK(result.out == expected.dump() + "\n");
        CHECK(result.out.find("pending startup") != std::string::npos);
    }
    CHECK(launched == 0);

    auto start = select;
    start.push_back("--");
    start.insert(start.end(), family.begin(), family.end());
    auto unsupported = start;
    unsupported.insert(unsupported.end(), {"--http-batch", "0", "--http-sampling", "sample"});
    const auto refused = invoke(unsupported, factories);
    CHECK(refused.code == 2 && refused.error.find("sample runs on the batched route") != std::string::npos);
    CHECK(launched == 0);

    sv::ServeShell shell;
    shell.serve = [&](const sv::HttpServerConfig& config, sv::HttpHandler handler, sv::SessionRuntime& runtime) {
        CHECK(config.port == 0 && !stopped);
        CHECK(runtime.capabilities().max_slots == 1);
        Writer writer;
        handler({}, writer);
        CHECK(writer.status == 200 && writer.body == "native fixture");
    };
    const auto started = invoke(start, factories, shell);
    if (started.code != 0) throw std::runtime_error(started.error);
    CHECK(launched == 1 && stopped);
    auto legacy = family;
    legacy.insert(legacy.begin(), "strix-hip-decode-token");
    legacy.push_back("--http-serve");
    CHECK(launched_options == qwen::inspect_http_options(legacy_options(legacy)).dump());
}
// The front as the user runs it (`strix-serve --model qwen3.8-flash-next --profile fast --asset MODEL.gguf ...`): the
// expanded native argv resolves the registry drafter beside the model and keeps MTP on with --http-batch 8.
// A header-only GGUF v3 (no tensors) declaring the drafter-target identity keys discovery reads.
void write_target_gguf(const std::filesystem::path& path, std::uint32_t layers) {
    std::ofstream f(path, std::ios::binary);
    const auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); };
    const auto u64 = [&](std::uint64_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); };
    const auto str = [&](const std::string& v) { u64(v.size()); f.write(v.data(), static_cast<std::streamsize>(v.size())); };
    f.write("GGUF", 4);
    u32(3);
    u64(0);
    u64(3);
    str("general.architecture");
    u32(8);
    str("qwen4exp");
    str("qwen4exp.block_count");
    u32(4);
    u32(layers);
    str("qwen4exp.embedding_length");
    u32(4);
    u32(2560);
    const auto end = static_cast<std::uint64_t>(f.tellp());
    for (auto pad = (32U - end % 32U) % 32U; pad != 0; --pad) f.put(char{0});
}
void check_discovered_drafter(const std::string& root) {
    const auto dir = std::filesystem::temp_directory_path() / "strix-serve-discovered-drafter";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "MTP");
    const auto model = dir / "model.gguf";
    write_target_gguf(model, 48);
    const auto sidecar = dir / "MTP" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
    { std::ofstream file(sidecar, std::ios::binary); file << "x"; }
    std::filesystem::resize_file(sidecar, 2786568256ULL);
    const auto ids = dir / "cal.ids";
    { std::ofstream file(ids); for (unsigned id = 1; id <= 64; ++id) file << id << " "; }
    sv::ServiceFactories factories;
    std::vector<std::string> expanded;
    bool stopped = false;
    const auto launch = [&](const std::vector<std::string>& args, const sv::ServiceFactory::ServeScoped& serve) {
        expanded = args;
        NativeFixture service(stopped);
        serve(service);
        return 0;
    };
    qwen::install_session_service(factories, launch);
    const std::vector<std::string> start{"strix-serve", "--registry-root", root, "--model", "qwen3.8-flash-next",
        "--profile", "fast", "--asset", model.string(), "--port", "0", "--", "--prompt-ids", ids.string(),
        "--generate", "64", "--http-batch", "8"};
    sv::ServeShell shell;
    shell.serve = [&](const sv::HttpServerConfig&, sv::HttpHandler, sv::SessionRuntime&) {};
    const auto started = invoke(start, factories, shell);
    std::cerr << "expanded argv:";
    for (const auto& a : expanded) std::cerr << " " << a;
    std::cerr << "\nrc=" << started.code << " " << started.error << "\n";
    CHECK(!expanded.empty());
    const auto options = legacy_options(expanded);
    CHECK(options.http_serve && options.mtp.enabled && options.mtp_native_bundle &&
          options.supplied_options.contains("--http-batch"));
    CHECK(options.mtp.sidecar == sidecar.string());
    // A model of another identity (layer count) never binds the draft placed beside it.
    write_target_gguf(model, 64);
    expanded.clear();
    (void)invoke(start, factories, shell);
    CHECK(expanded.empty() || legacy_options(expanded).mtp.sidecar != sidecar.string());
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}
// The documented serving quick start (docs/models/qwen3.8/README.md): calibration ids, a bundle and a long
// context, with no --mtp pins, no golden flags and no --known-prefix-* flags, resolves the tiled MTP route.
void check_quick_start(const std::string& root) {
    const auto dir = std::filesystem::temp_directory_path() / "strix-serve-quick-start";
    std::filesystem::create_directories(dir);
    const auto ids = dir / "calibration.ids";
    {
        std::ofstream file(ids);
        for (unsigned id = 1; id <= 17; ++id) file << id << (id == 17 ? "\n" : " ");
    }
    const auto bundle = (std::filesystem::path(root) / "recipes/qwen-distill-g-yarn.mtp-bundle.json").string();
    for (const char* context : {"8192", "36864"}) {
        const auto options = legacy_options({"strix-hip-decode-token", "fixture-missing.gguf", "--prompt-ids", ids.string(),
            "--generate", "64", "--context", context, "--mtp-bundle", bundle, "--http-serve"});
        CHECK(options.http_serve && options.mtp.enabled && options.mtp_native_bundle && options.mtp.golden.empty() &&
              !options.known_prefix_window);
    }
    // The bare command (no calibration prompt) is refused: the requirement is inherent.
    bool refused = false;
    try { (void)legacy_options({"strix-hip-decode-token", "fixture-missing.gguf", "--mtp-bundle", bundle, "--http-serve"}); }
    catch (const std::exception&) { refused = true; }
    CHECK(refused);
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}
} // namespace

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        check_factory(argv[1]);
        check_http_capabilities();
        check_discovered_drafter(argv[1]);
        check_quick_start(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
