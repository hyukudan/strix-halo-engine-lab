// The example family of docs/adding-an-engine.md serves a completion and a chat
// request through the shared serve shell with the CPU reference backend.
#include "../examples/skeleton_engine/skeleton_engine.hpp"
#include "strix/profile/model_preset.hpp"
#include "strix/serve/native_handoff.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace {
namespace sv = strix::serve;

void check(bool ok, const char* what) {
    if (ok) return;
    std::cerr << "FAILED: " << what << "\n";
    std::exit(1);
}
#define CHECK(x) check((x), #x)

struct Writer final : sv::HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int s, std::string_view, std::string_view b,
              const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int s, std::string_view,
                      const std::vector<std::pair<std::string, std::string>>&) override { status = s; return true; }
    bool write(std::string_view b) override { body += b; return true; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};

Writer post(const sv::HttpHandler& handler, const std::string& path, const std::string& body) {
    sv::HttpRequest request;
    request.method = "POST";
    request.path = path;
    request.target = path;
    request.body = body;
    Writer writer;
    handler(request, writer);
    return writer;
}
} // namespace

// Embedding rows through the engine contract: a bound table resolves virtual
// ids, everything else is refused before a write.
void embed_rows() {
    namespace ss = strix::session;
    ss::CpuBackend backend;
    skeleton::Engine engine(backend);
    CHECK(engine.caps().embed_rows);
    ss::EmbedInterner interner;
    ss::EmbedRows rows;
    rows.digest[0] = 7;
    rows.rows = 3;
    rows.width = 2;
    rows.data.assign(6, 0x3f80);
    const auto lease = interner.intern(std::move(rows));
    auto table = std::make_shared<const ss::EmbedTable>(std::vector<ss::EmbedLease>{lease});
    std::vector<std::uint32_t> prompt{'a', 'b'};
    for (const auto id : lease.ids()) prompt.push_back(id);
    prompt.push_back('c');

    auto lane = engine.store.acquire();
    const auto refused = [&](std::span<const std::uint32_t> ids) {
        try {
            engine.prefill(ids, lane);
        } catch (const std::invalid_argument&) {
            return engine.store.consumed(lane) == 0;
        }
        return false;
    };
    CHECK(refused(prompt)); // no table bound
    engine.bind_embeds(lane, table);
    const std::vector<std::uint32_t> outside{'a', ss::embed_id(lease.handle(), 3)};
    CHECK(refused(outside)); // row outside the span
    const std::vector<std::uint32_t> unknown{'a', ss::embed_id(lease.handle() + 1, 0)};
    CHECK(refused(unknown)); // handle outside the request
    engine.prefill(prompt, lane);
    CHECK(engine.store.consumed(lane) == prompt.size());
    engine.store.release(lane);
    CHECK(interner.status().live == 1); // lease + table still hold it
}

void shell_config_tests() {
    namespace fs = std::filesystem;
    const auto dir = fs::current_path() / "serve-shell-config-fixture";
    fs::create_directories(dir / "models");
    std::ofstream(dir / "models/registry.toml") << "schema = \"strix-registry/v1\"\n[models.fixture]\n"
        "name = \"fixture\"\nfamily = \"fixture\"\ndescription = \"models/m.toml\"\n";
    std::ofstream(dir / "models/m.toml") << "schema = \"strix-model/v1\"\nname = \"fixture\"\n";
    const auto config = dir / "serve.toml";
    std::ofstream(config) << "registry_root = \".\"\nregistry = \"models/registry.toml\"\n"
        "description = \"models/m.toml\"\nport = 9000\n";
#if !defined(_WIN32)
    ::chmod(config.c_str(), 0600);
#endif
    const auto invoke = [&](std::vector<std::string> args, bool print) {
        std::vector<char*> pointers;
        for (auto& a : args) pointers.push_back(a.data());
        std::ostringstream out, error;
        const sv::ServiceFactories factories;
        const int result = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, error);
        if (result) std::cerr << error.str();
        CHECK(result == 0);
        if (print) {
            CHECK(out.str().find("port = \"9000\"") != std::string::npos);
            CHECK(out.str().find("1234") == std::string::npos);
        } else CHECK(out.str().find("fixture\tfixture\tfixture\tmodels/m.toml") != std::string::npos);
    };
    invoke({"serve", "--serve-config", config.string(), "--inspect"}, false);
    invoke({"serve", "--serve-config", fs::relative(config).string(), "--inspect"}, false);
    invoke({"serve", "--serve-config", config.string(), "--print-serve-config", "--", "--port", "1234",
        "--serve-config", "missing.toml", "--print-serve-config"}, true);
    fs::remove_all(dir);
}

// strix-serve --device-count N --print-effective-config prints the shared resolver's config for the same query, and its
// multi_device member is what the replica router admits (one resolver for CLI, server and router).
void multi_device_parity() {
    namespace fs = std::filesystem;
    namespace md = strix::profile::multi_device;
    const auto dir = fs::current_path() / "serve-multi-device-fixture";
    fs::create_directories(dir / "models");
    std::ofstream(dir / "models/registry.toml") << "schema = \"strix-registry/v1\"\n[models.fixture]\n"
        "name = \"fixture\"\nfamily = \"fixture\"\ndescription = \"models/m.toml\"\n";
    const std::string text = "schema = \"strix-model/v1\"\nname = \"fixture\"\n[multi_device]\nversion = 1\n"
        "default = \"replica\"\n[multi_device.forms.replica]\nranks = 2\n";
    std::ofstream(dir / "models/m.toml") << text;
    const auto description = strix::model_description::parse(text);
    for (std::uint32_t devices : {1U, 2U}) {
        std::vector<std::string> args{"serve", "--registry-root", dir.string(), "--model", "fixture", "--device-count",
                                      std::to_string(devices), "--print-effective-config"};
        std::vector<char*> pointers;
        for (auto& a : args) pointers.push_back(a.data());
        std::ostringstream out, error;
        const sv::ServiceFactories factories;
        const int result = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, error);
        if (result) std::cerr << error.str();
        CHECK(result == 0);
        md::Query query;
        query.devices = devices;
        const auto expected = strix::profile::resolve_model_preset(description, {}, "hip", query).config.dump();
        CHECK(out.str() == expected + "\n");
        const auto form = std::string("\"form\":\"") + (devices == 1 ? "single" : "replica") + "\"";
        CHECK(out.str().find(form) != std::string::npos);
        if (devices == 2) {
            const auto router = strix::profile::resolve_replica_router(description, {}, 2);
            CHECK(out.str().find(router.config.find("multi_device")->dump()) != std::string::npos);
        } else {
            bool refused = false;
            try { (void)strix::profile::resolve_replica_router(description, {}, 1); }
            catch (const std::invalid_argument&) { refused = true; }
            CHECK(refused);
        }
    }
    // The shell's count is --device-count: `--devices FILE` (an engine's device file, e.g. Inkling's) still reaches
    // the family arguments unchanged, as before the multi-device options existed.
    {
        std::vector<std::string> seen;
        sv::ServiceFactories family;
        sv::ServiceFactory fixture;
        fixture.validate = [&](const sv::ServiceSelection& selection) { seen = selection.arguments; };
        fixture.inspect = [](const sv::ServiceSelection&, std::ostream&) {};
        fixture.create = [](const sv::ServiceSelection&) -> std::unique_ptr<sv::SessionService> { return nullptr; };
        fixture.early_exit = [](const sv::ServiceSelection&, std::ostream&) -> std::optional<int> { return 0; };
        family.install("fixture", std::move(fixture));
        std::vector<std::string> args{"serve", "--registry-root", dir.string(), "--model", "fixture", "--devices",
                                      "devices.toml"};
        std::vector<char*> pointers;
        for (auto& a : args) pointers.push_back(a.data());
        std::ostringstream out, error;
        const int result = sv::run(static_cast<int>(pointers.size()), pointers.data(), family, out, error);
        if (result) std::cerr << error.str();
        CHECK(result == 0);
        CHECK((seen == std::vector<std::string>{"--devices", "devices.toml"}));
        // This process serves one worker: --device-count 2 resolves the model's replica default, which only
        // strix-router executes, so the launch is refused before the family sees it (inspection above reports it).
        seen.clear();
        std::vector<std::string> two{"serve", "--registry-root", dir.string(), "--model", "fixture", "--device-count", "2"};
        std::vector<char*> two_pointers;
        for (auto& a : two) two_pointers.push_back(a.data());
        std::ostringstream two_out, two_error;
        CHECK(sv::run(static_cast<int>(two_pointers.size()), two_pointers.data(), family, two_out, two_error) == 2);
        CHECK(two_error.str().find("runs behind strix-router") != std::string::npos);
        CHECK(seen.empty());
    }
    fs::remove_all(dir);
}

// strix-serve's native handoff (no in-process adapter, or an external CUDA launcher) forwards the whole multi-device
// query, and a launch is admitted before delegation only for the form one engine executes. Backend arguments are
// given so the print-only inspection path cannot return before the handoff.
void native_handoff() {
    namespace fs = std::filesystem;
    const auto dir = fs::current_path() / "serve-native-handoff-fixture";
    fs::create_directories(dir / "models");
    std::ofstream(dir / "models/registry.toml") << "schema = \"strix-registry/v1\"\n[models.fixture]\n"
        "name = \"fixture\"\nfamily = \"fixture\"\ndescription = \"models/m.toml\"\n";
    std::ofstream(dir / "models/m.toml") << "schema = \"strix-model/v1\"\nname = \"fixture\"\n[multi_device]\nversion = 1\n"
        "[multi_device.forms.replica]\nranks = 2\n[multi_device.forms.layers]\nranks = 2\n";
    const auto invoke = [&](std::vector<std::string> extra, std::optional<strix::cli::Request>& handed, std::string& error) {
        std::vector<std::string> args{"serve", "--registry-root", dir.string(), "--model", "fixture", "--asset", "w.gguf"};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char*> pointers;
        for (auto& a : args) pointers.push_back(a.data());
        handed.reset();
        sv::ServeShell shell;
        shell.launch = [&](const sv::ServiceSelection& selection, const sv::ServeShell::NativeRequest& request,
                           std::ostream&) {
            handed = sv::native_serve_request(selection, request);
            return 0;
        };
        std::ostringstream out, err;
        const sv::ServiceFactories none;   // no in-process adapter: every preset launch is a native handoff
        const int result = sv::run(static_cast<int>(pointers.size()), pointers.data(), none, out, err, shell);
        error = err.str();
        return result;
    };
    std::optional<strix::cli::Request> handed;
    std::string error;
    // Explicit single on two devices reaches the backend launcher with both inputs.
    CHECK(invoke({"--device-count", "2", "--placement", "single", "--bind", "127.0.0.2"}, handed, error) == 0);
    CHECK(handed && handed->verb == strix::cli::Verb::serve && handed->model == "fixture" && handed->asset == "w.gguf");
    CHECK(handed->device_count == 2 && handed->placement == "single" && handed->host == "127.0.0.2");
    // A count alone (the model's default is single) is forwarded too, with no placement.
    CHECK(invoke({"--device-count", "2"}, handed, error) == 0);
    CHECK(handed && handed->device_count == 2 && handed->placement.empty());
    // A declared but unexecuted form and a standalone replica are refused before delegation.
    CHECK(invoke({"--device-count", "2", "--placement", "layers"}, handed, error) == 2);
    CHECK(!handed && error.find("'layers' (explicit: --placement) is not executed") != std::string::npos);
    CHECK(invoke({"--device-count", "2", "--placement", "replica"}, handed, error) == 2);
    CHECK(!handed && error.find("runs behind strix-router") != std::string::npos);
    // Inspection is forwarded as inspection: the backend launcher prints the same decision and executes nothing.
    CHECK(invoke({"--device-count", "2", "--placement", "layers", "--print-effective-config"}, handed, error) == 0);
    CHECK(handed && handed->print_effective_config && handed->placement == "layers" && handed->device_count == 2);
    fs::remove_all(dir);
}

// Review r9 M1: a placement query alone does not apply a model preset. A family-native --profile after `--` keeps its
// passthrough and is the tier the placement is admitted at, in the in-process branch and in the external handoff.
void native_tier_placement() {
    namespace fs = std::filesystem;
    const auto dir = fs::current_path() / "serve-native-tier-fixture";
    fs::create_directories(dir / "models");
    std::ofstream(dir / "models/registry.toml") << "schema = \"strix-registry/v1\"\n[models.fixture]\n"
        "name = \"fixture\"\nfamily = \"fixture\"\ndescription = \"models/m.toml\"\n";
    // A narrow tensor wire: explicit tensor is refused at EXACT ("tier is exact") and reaches the executor check at FAST.
    std::ofstream(dir / "models/m.toml") << "schema = \"strix-model/v1\"\nname = \"fixture\"\n[multi_device]\nversion = 1\n"
        "[multi_device.forms.tensor]\nwire = \"fp16\"\n";
    std::vector<std::string> seen;
    bool launched = false;
    sv::ServiceFactories family;
    sv::ServiceFactory fixture;
    fixture.validate = [&](const sv::ServiceSelection& selection) { seen = selection.arguments; };
    fixture.inspect = [](const sv::ServiceSelection&, std::ostream&) {};
    fixture.create = [](const sv::ServiceSelection&) -> std::unique_ptr<sv::SessionService> { return nullptr; };
    fixture.early_exit = [](const sv::ServiceSelection&, std::ostream&) -> std::optional<int> { return 0; };
    family.install("fixture", std::move(fixture));
    const sv::ServiceFactories none;
    const auto invoke = [&](const sv::ServiceFactories& factories, std::vector<std::string> extra, std::string& error) {
        std::vector<std::string> args{"serve", "--registry-root", dir.string(), "--model", "fixture"};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char*> pointers;
        for (auto& a : args) pointers.push_back(a.data());
        sv::ServeShell shell;
        shell.launch = [&](const sv::ServiceSelection&, const sv::ServeShell::NativeRequest&, std::ostream&) {
            launched = true;
            return 0;
        };
        seen.clear();
        launched = false;
        std::ostringstream out, err;
        const int result = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, err, shell);
        error = err.str();
        return result;
    };
    std::string error;
    // In-process: the native --profile passes through untouched (no preset injection) under an explicit single.
    CHECK(invoke(family, {"--placement", "single", "--", "--profile", "fast"}, error) == 0);
    CHECK((seen == std::vector<std::string>{"--profile", "fast"}));
    CHECK(invoke(family, {"--placement", "single", "--", "--profile=fast"}, error) == 0);
    CHECK((seen == std::vector<std::string>{"--profile=fast"}));
    // The placement is admitted at the effective (native) tier: FAST passes the tier rule and meets the executor check,
    // while without a native tier the standard (EXACT) rule refuses the narrow wire.
    CHECK(invoke(family, {"--device-count", "2", "--placement", "tensor", "--", "--profile", "fast"}, error) == 2);
    CHECK(error.find("is not executed by any engine yet") != std::string::npos &&
          error.find("tier is exact") == std::string::npos);
    CHECK(invoke(family, {"--device-count", "2", "--placement", "tensor"}, error) == 2);
    CHECK(error.find("tier is exact") != std::string::npos);
    // External handoff: the same effective tier decides the admission made before delegation.
    CHECK(invoke(none, {"--asset", "w.gguf", "--placement", "single", "--", "--profile", "fast"}, error) == 0 && launched);
    CHECK(invoke(none, {"--asset", "w.gguf", "--device-count", "2", "--placement", "tensor", "--", "--profile", "fast"},
                 error) == 2);
    CHECK(!launched && error.find("is not executed by any engine yet") != std::string::npos &&
          error.find("tier is exact") == std::string::npos);
    CHECK(invoke(none, {"--asset", "w.gguf", "--device-count", "2", "--placement", "tensor"}, error) == 2);
    CHECK(!launched && error.find("tier is exact") != std::string::npos);
    fs::remove_all(dir);
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    shell_config_tests();
    multi_device_parity();
    native_handoff();
    native_tier_placement();
    embed_rows();
    sv::ServiceFactories factories;
    skeleton::install_session_service(factories);

    // The shell seam replaces only listen/accept: selection, validation,
    // construction and cleanup are the production path of strix::serve::run().
    bool served = false;
    sv::ServeShell shell;
    shell.serve = [&](const sv::HttpServerConfig&, sv::HttpHandler handler, sv::SessionRuntime& runtime) {
        std::thread owner([&] { runtime.run(); });
        const auto completion = post(handler, "/v1/completions",
            R"({"model":"skeleton","prompt":"hi","max_tokens":4})");
        const auto chat = post(handler, "/v1/chat/completions",
            R"({"model":"skeleton","messages":[{"role":"user","content":"hi"}],"max_tokens":4})");
        runtime.shutdown();
        owner.join();
        if (completion.status != 200 || chat.status != 200)
            std::cerr << completion.status << " " << completion.body << "\n" << chat.status << " " << chat.body << "\n";
        CHECK(completion.status == 200);
        // Two prompt rows: the counter continues with c, d, e, f.
        CHECK(completion.body.find("cdef") != std::string::npos);
        CHECK(chat.status == 200);
        CHECK(chat.body.find("cdef") != std::string::npos);
        served = true;
    };

    std::vector<std::string> arguments{"serve", "--registry-root", argv[1], "--registry", "models/registry.toml",
                                       "--model", "skeleton", "--port", "0"};
    std::vector<char*> pointers;
    for (auto& a : arguments) pointers.push_back(a.data());
    std::ostringstream out, error;
    const auto code = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, error, shell);
    if (code != 0) std::cerr << "run: " << code << " " << error.str() << "\n";
    CHECK(code == 0);
    CHECK(served);

    auto inspect = arguments;
    inspect.resize(7);
    inspect.push_back("--family-inspect");
    pointers.clear();
    for (auto& a : inspect) pointers.push_back(a.data());
    std::ostringstream inspected, inspect_error;
    CHECK(sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, inspected, inspect_error) == 0);
    CHECK(inspected.str().find("cpu backend") != std::string::npos);
    std::cout << "skeleton engine: ok\n";
    return 0;
}
