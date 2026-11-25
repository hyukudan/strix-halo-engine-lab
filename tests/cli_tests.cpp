#include "strix/cli/cli.hpp"
#include "strix/deploy/route.hpp"
#include "strix/serve/json.hpp"
#include "../src/cli/detail.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {
using namespace strix::cli;
using strix::serve::parse_json;
namespace fs = std::filesystem;
namespace support = strix::support;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void refuses(F&& fn, std::string_view reason) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(reason) != std::string_view::npos, error.what());
        return;
    }
    throw std::runtime_error("expected refusal");
}

class Probes final : public DoctorProbes {
public:
    Platform host = Platform::linux_host;
    bool nodes_open = false, cuda_loaded = false;
    std::vector<std::string> hip_archs{"gfx1151"};

private:
    Platform platform() const override { return host; }
    Check os() const override { return {"os", Status::ok, "test Linux"}; }
    Check cpu() const override { return {"cpu", Status::ok, "test \"CPU\""}; }
    Check ram() const override { return {"ram", Status::warn, "unavailable"}; }
    Check runtime(std::string_view name) const override {
        return {std::string(name), name == "hip" || (name == "cuda" && cuda_loaded) ?
                Status::ok : Status::warn, "load probe"};
    }
    std::vector<std::string> architectures(std::string_view backend) const override {
        if (backend == "hip") return hip_archs;
        if (backend == "cuda") return {"120"};
        return {"x86_64"};
    }
    Check node(std::string_view name) const override {
        return {std::string(name), nodes_open ? Status::ok : Status::fail, nodes_open ? "open" : "access denied"};
    }
    Check component(std::string_view name) const override {
        return {std::string(name), Status::skip, "no receipt\nfound"};
    }
};

void parsing() {
    require(parse({"--version"}).verb == Verb::version, "version alias");
    require(parse({"version"}).verb == Verb::version, "version command");
    const auto request = parse({"serve", "fixture-qwen", "--host", "::1", "--port", "9000", "--",
                                "model path.gguf", "--json"});
    require(request.host == "::1" && request.port == "9000" && !request.json &&
            request.args == std::vector<std::string>{"model path.gguf", "--json"}, "passthrough boundary");
    require(parse({"plan", "fixture-qwen", "--context", "2048", "--json"}).context == 2048, "context");
    refuses([] { (void)parse({"plan", "fixture-qwen", "--context", "2048"}); }, "--json");
    refuses([] { (void)parse({"plan", "fixture-qwen", "--context", "4096extra", "--json"}); }, "integer");
    refuses([] { (void)parse({"serve", "fixture-qwen", "--port", "65536"}); }, "--port");
    refuses([] { (void)parse({"run", "fixture-qwen", "--generate", "1"}); }, "unexpected");
    refuses([] { (void)parse({"inspect", "--json"}); }, "requires MODEL");
    refuses([] { (void)parse({"support", "--state", "unknown"}); }, "unknown support state");
    refuses([] { (void)parse({"support", "--backend"}); }, "requires a value");
    refuses([] { (void)parse({"support", "--family", "qwen", "--family", "glm5"}); }, "unexpected");
}

void inspection(const fs::path& data, const Probes& probes, const ExecutableExists& exists) {
    std::ostringstream out;
    const auto invoke = [&](const std::vector<std::string>& args, int status = 0) {
        out.str("");
        require(dispatch(parse(args), "test bin", data, probes, exists, out) == status, "dispatch status");
        return out.str();
    };
    const auto result = parse_json(invoke({"models", "list", "--json"}));
    require(result.size() == 2 && result.as_array()[0].find("name")->as_string() == "fixture-pair", "registry fixture");
    const auto inspected = parse_json(invoke({"inspect", "fixture_qwen", "--json"}));
    require(inspected.find("metadata")->find("dims")->find("layers")->as_int() == 2, "describe fixture");
#ifdef STRIX_CLI_QWEN_DESCRIPTION
    strix::model_registry::Entry canonical;
    canonical.name = "qwen3.8-flash-next";
    canonical.family = "qwen";
    const auto incomplete = strix::model_description::parse("name = \"qwen3.8-flash-next\"");
    refuses([&] { validate_description(canonical, incomplete); }, "Qwen model description refused");
#endif
    const auto planned = parse_json(invoke({"plan", "fixture-qwen", "--context", "2048", "--devices",
        (data / "devices.toml").string(), "--json"}, 2));
    require(planned.find("kind")->as_string() == "plan" && !planned.find("measurement")->as_bool() &&
            planned.find("status")->as_string() == "refused" && planned.find("device_budgets")->size() == 1,
            "plan reports budgets and preserves unknown admission");
    require(invoke({"update"}, 2) == "not available in this release\n", "update stub");
    require(invoke({"rollback"}, 2) == "not available in this release\n", "rollback stub");

    const auto doctor_result = parse_json(invoke({"doctor", "--json"}));
    const auto& checks = doctor_result.find("checks")->as_array();
    require(checks.size() == 15 && checks[1].find("detail")->as_string() == "test \"CPU\"", "doctor schema");
    for (const auto& check : checks)
        require(check.size() == 3 && check.find("name")->is_string() && check.find("status")->is_string() &&
                check.find("detail")->is_string(), "doctor check schema");
    require(checks[3].find("status")->as_string() == "fail" &&
            checks.back().find("detail")->as_string().find("requires /dev/kfd and /dev/dri") != std::string::npos &&
            checks.back().find("detail")->as_string().find("recommended package component") == std::string::npos,
            "doctor reports blocked recommendation");

    const auto dry = invoke({"serve", "fixture-qwen", "--host", "::1", "--port", "9000", "--dry-run",
        "--", "model path.gguf", "--http-batch", "0"});
#if defined(_WIN32)
    const std::string expected = "\"model path.gguf\" \"--http-serve\" \"--http-bind\" \"::1\" \"--http-port\" \"9000\" \"--http-batch\" \"0\"";
#else
    const std::string expected = "'model path.gguf' '--http-serve' '--http-bind' '::1' '--http-port' '9000' '--http-batch' '0'";
#endif
    require(dry.find("strix-hip-decode-token") != std::string::npos && dry.ends_with(expected + "\n"), "serve dry run");
}

void support_policy(const fs::path& data, const ExecutableExists& exists) {
    const auto matrix = support::load(data / "support/matrix.toml");
    Probes probes;
    probes.nodes_open = true;
    std::ostringstream out;
    const auto invoke = [&](const std::vector<std::string>& args) {
        out.str("");
        require(dispatch(parse(args), "test bin", data, probes, exists, out) == 0, "support dispatch");
        return out.str();
    };
    const auto rows = support::query(matrix, {"qwen", "hip", support::State::implemented});
    require(rows.size() == 1 && rows[0]->arch == "gfx1200", "fixture query");
    const std::vector<std::string> args{"support", "--family", "qwen", "--backend", "hip", "--state", "implemented"};
    require(invoke(args) == support::markdown(rows, matrix.consumers), "shared Markdown rendering");
    auto json_args = args;
    json_args.push_back("--json");
    require(invoke(json_args) == support::json(rows, matrix.consumers), "shared JSON rendering");

    const auto linux_report = invoke({"doctor"});
    require(linux_report.find("[packaged]; recommended package component hip") != std::string::npos &&
            linux_report.find("[connected]") != std::string::npos &&
            linux_report.find("gfx1200") == std::string::npos && linux_report.find("/windows/") == std::string::npos,
            "only matching packaged rows recommended");
    probes.hip_archs = {"gfx1200", "gfx9999"};
    const auto lower_report = invoke({"doctor"});
    require(lower_report.find("gfx1200/linux/cli-generate (fixture-gguf) [implemented]") != std::string::npos &&
            lower_report.find("gfx9999/linux: no support matrix row") != std::string::npos &&
            lower_report.find("recommended package component") == std::string::npos, "low and unknown architectures");
    probes.hip_archs.clear();
    require(invoke({"doctor"}).find("hip: architecture unavailable") != std::string::npos, "unknown architecture");
    probes.host = Platform::windows_host;
    probes.cuda_loaded = true;
    require(invoke({"doctor"}).find("cuda/120/windows/serve (fixture-gguf) [implemented]") != std::string::npos,
            "Windows backend matched by architecture");
    refuses([&] { (void)invoke({"serve", "fixture-qwen"}); },
            "qwen/cuda/120/windows/serve (fixture-gguf) [implemented] is below connected; Fixture server is pending.");
    refuses([&] { (void)invoke({"run", "fixture-qwen", "--dry-run"}); },
            "no support matrix row for qwen/hip/windows/cli-generate");
}

void launch(const fs::path& data, const ExecutableExists& exists) {
    auto entries = strix::model_registry::load((data / "models/registry.toml").string());
    auto matrix = support::load(data / "support/matrix.toml");
    const fs::path bin = "test bin";
    Probes probes;
    const auto resolve_args = [&](const std::vector<std::string>& args) {
        return resolve(parse(args), entries, matrix, probes, bin, exists);
    };
    const auto qwen = resolve_args({"run", "assets/fixture-qwen.gguf", "--dry-run", "--", "1", "--generate", "256"});
    require(qwen.executable.parent_path() == bin && qwen.executable.stem() == "strix-hip-decode-token", "Qwen run route");
    require(qwen.args == std::vector<std::string>{"assets/fixture-qwen.gguf", "1", "--generate", "256"}, "run passthrough");
    // A row for a different GPU must not authorize launch even when it is packaged.
    auto cross_arch = matrix;
    for (auto& row : cross_arch.rows) {
        if (row.family == "qwen" && row.backend == "hip" && row.function == "cli-generate")
            row.state = row.arch == "gfx1200" ? support::State::packaged : support::State::declared;
    }
    const auto request = parse({"run", "fixture-qwen", "--dry-run", "--", "model.gguf"});
    refuses([&] { (void)resolve(request, entries, cross_arch, probes, bin, exists); },
            "gfx1151/linux/cli-generate");
    Probes gfx1200;
    gfx1200.hip_archs = {"gfx1200"};
    require(resolve(request, entries, cross_arch, gfx1200, bin, exists).executable.stem() ==
            "strix-hip-decode-token", "matching gfx1200 can launch");
    Probes unknown;
    unknown.hip_archs.clear();
    refuses([&] { (void)resolve(request, entries, cross_arch, unknown, bin, exists); },
            "no support matrix row for qwen/hip/linux/cli-generate");
    const auto pair = resolve_args({"run", "fixture_pair", "--", "pair.gguf", "--rank", "1", "--transport", "usb4auto"});
    require(pair.executable.stem() == "strix-deepseek41-serve" && pair.args[0] == "pair.gguf", "pair production route");
    const auto follower = resolve_args({"serve", "fixture_pair", "--", "pair.gguf", "--rank", "1", "--transport", "usb4auto"});
    require(follower.args == pair.args, "follower uses the pair command");
    const auto leader = resolve_args({"serve", "fixture_pair", "--", "pair.gguf", "--rank", "0", "--transport", "usb4auto"});
    require(leader.args[1] == "--http-port", "leader starts HTTP");
    refuses([&] { (void)resolve_args({"run", "fixture-pair", "--", "pair.gguf"}); }, "--rank");
    refuses([&] { (void)resolve_args({"run", "fixture-pair", "--", "pair.gguf", "--rank", "1"}); }, "--transport");
    refuses([&] { (void)resolve_args({"run", "fixture-pair", "--", "pair.gguf", "--rank", "2"}); }, "--rank");
    refuses([&] { (void)resolve_args({"run", "fixture-qwen"}); }, "asset path");
    refuses([&] { (void)resolve_args({"run", "unknown.gguf"}); }, "unknown model");
    refuses([&] { (void)resolve(parse({"run", "fixture-qwen"}), entries, matrix, probes, bin,
                                [](const auto&) { return false; }); }, "no matching executable");
    matrix.rows[0].state = support::State::declared;
    refuses([&] { (void)resolve_args({"run", "fixture-qwen"}); },
            "qwen/hip/gfx1151/linux/cli-generate (fixture-gguf) [declared] is below connected");
    matrix.rows[0].state = support::State::packaged;
    entries.back().apps.clear();
    refuses([&] { (void)resolve_args({"run", "fixture-qwen"}); }, "no registered run route");
}

void model_presets(const fs::path& data, const ExecutableExists& exists) {
    namespace profile = strix::profile;
    using profile::Name;
    const auto entries = strix::model_registry::load((data / "models/registry.toml").string());
    const auto matrix = support::load(data / "support/matrix.toml");
    Probes probes;
    for (const auto& entry : entries) {
        const auto description = strix::model_description::load((data / entry.description).string());
        require(description.find("presets") != nullptr, "every registered model declares presets");
        for (const auto* backend : {"hip", "cuda"}) {
            const auto exact = profile::resolve_model_preset(description, Name::standard, backend);
            const auto implicit = profile::resolve_model_preset(description, {}, backend);
            const auto fast = profile::resolve_model_preset(description, Name::fast, backend);
            const auto turbo = profile::resolve_model_preset(description, Name::turbo, backend);
            const auto* declared_default = description.find("presets.default");
            require(declared_default != nullptr, "every model declares an implicit preset");
            const auto* declared_name = std::get_if<std::string>(&declared_default->data);
            require(declared_name != nullptr, "implicit preset must be a string");
            require(implicit.requested == profile::parse_tier(*declared_name),
                    "implicit preset must agree with its declared tier");
            if (entry.family == "inkling2")
                require(implicit.requested == Name::standard && implicit.levers == Name::standard,
                        "Inkling Mini implicit profile must remain EXACT");
            require(exact.levers == Name::standard && fast.levers == Name::fast && fast.requested == Name::fast,
                    "explicit exact and FAST select their native columns");
            const bool dense_hip = entry.family == "qwen35" && std::string_view(backend) == "hip";
            require(turbo.requested == Name::turbo && turbo.levers == (dense_hip ? Name::turbo : fast.levers),
                    "Qwen dense HIP receives TURBO and resolves the FAST fallback in the native profile");
            require(exact.kv != "fp8" && fast.kv != "fp8", "FP8 KV shadow is TURBO only");
            require((turbo.kv == "fp8") == (entry.family == "qwen35" && std::string_view(backend) == "cuda"),
                    "only the CUDA dense route has an FP8 KV shadow");
            require(fast.config.find("weights")->find("format")->is_object() && fast.config.find("drafter"),
                    "host configuration reports weight roles and drafter");
        }
    }

    const auto dense = strix::model_description::load((data / "models/qwen3.8-27b.toml").string());
    probes.host = Platform::windows_host;
    auto req = parse({"run", "--model", "qwen3.8-27b", "--backend", "cuda", "--profile=turbo",
                      "--asset", "model.gguf", "--print-effective-config", "--", "--n-gen", "32"});
    auto command = resolve(req, entries, matrix, probes, "bin", exists, &dense);
    require(command.executable.stem() == "strix-cuda-qwen35-decode", "launches the qualified backend route");
    require(command.args == std::vector<std::string>{"--model", "model.gguf", "--n-gen", "32", "--profile",
        "fast", "--kv", "fp8", "--print-effective-config"}, "TURBO and native config arguments");
    {
        // --device-count / --placement are resolver inputs: the engine arguments do not change, and the effective config
        // is the shared resolver's for the same query (strix-serve builds the same query from the same options).
        auto placed = parse({"run", "--model", "qwen3.8-27b", "--backend", "cuda", "--asset", "model.gguf",
                             "--device-count", "2", "--placement", "single", "--print-effective-config", "--", "--n-gen", "32"});
        require(placed.device_count == 2 && placed.placement == "single", "run parses --device-count and --placement");
        const auto placed_command = resolve(placed, entries, matrix, probes, "bin", exists, &dense);
        require(placed_command.args == std::vector<std::string>{"--model", "model.gguf", "--n-gen", "32", "--profile",
            "fast", "--kv", "native", "--print-effective-config"}, "placement options never reach the engine");
        strix::profile::multi_device::Query query;
        query.devices = 2;
        query.placement = strix::profile::multi_device::Form::single;
        require(placed_command.preset->config.dump() ==
                profile::resolve_model_preset(dense, {}, "cuda", query).config.dump(),
                "CLI effective config is the shared resolver's");
        require(placed_command.preset->config.find("multi_device")->find("rule")->as_string() == "explicit",
                "explicit placement is reported");
        placed.placement = "replica";
        refuses([&] { (void)resolve(placed, entries, matrix, probes, "bin", exists, &dense); }, "replica is not declared");
        refuses([] { (void)parse({"run", "qwen3.8-27b", "--placement", "mirror"}); }, "--placement requires");
        refuses([] { (void)parse({"run", "qwen3.8-27b", "--device-count", "0"}); }, "--device-count");
        // Inspection vs execution: with a model whose two-device default is replica, --dry-run reports the command,
        // while a real launch is refused before anything executes (one engine process cannot run replicas).
        std::stringstream text;
        text << std::ifstream(data / "models/qwen3.8-27b.toml").rdbuf()
             << "\n[multi_device]\nversion = 1\ndefault = \"replica\"\n[multi_device.forms.replica]\nranks = 2\n";
        const auto replicated = strix::model_description::parse(text.str());
        auto replica_run = parse({"run", "--model", "qwen3.8-27b", "--backend", "cuda", "--asset", "model.gguf",
                                  "--device-count", "2", "--dry-run"});
        std::ostringstream shown;
        require(strix::cli::launch(replica_run, entries, &replicated, "bin", data, probes, exists, shown) == 0 &&
                shown.str().find("strix-cuda-qwen35-decode") != std::string::npos, "--dry-run reports a replica launch");
        replica_run.dry_run = false;
        refuses([&] { std::ostringstream sink;
                      (void)strix::cli::launch(replica_run, entries, &replicated, "bin", data, probes, exists, sink); },
                "runs behind strix-router");

        // Review r5: the description reaches placement validation through dispatch even when the engine arguments carry
        // a native --profile (no preset injection), and a dry-run states the placement and its executor.
        const auto through_dispatch = [&](const fs::path& root, std::vector<std::string> args) {
            std::ostringstream sink;
            const auto status = strix::cli::dispatch(parse(args), "bin", root, probes, exists, sink);
            return std::pair{status, sink.str()};
        };
        const std::vector<std::string> native{"run", "qwen3.8-27b", "--backend", "cuda", "--asset", "model.gguf"};
        const auto with = [&](std::vector<std::string> extra, std::vector<std::string> tail) {
            auto args = native;
            args.insert(args.end(), extra.begin(), extra.end());
            args.emplace_back("--");
            args.insert(args.end(), tail.begin(), tail.end());
            return args;
        };
        const auto count = [](const std::string& haystack, std::string_view needle) {
            std::size_t n = 0;
            for (auto at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
            return n;
        };
        refuses([&] { (void)through_dispatch(data, with({"--device-count", "2", "--placement", "replica"},
                                                        {"--profile", "fast", "--n-gen", "32"})); },
                "replica is not declared");
        refuses([&] { (void)through_dispatch(data, with({"--device-count", "2", "--placement", "replica", "--dry-run"},
                                                        {"--profile=fast"})); }, "replica is not declared");
        const auto passthrough = through_dispatch(data, with({"--dry-run"}, {"--profile", "fast", "--n-gen", "32"}));
        require(passthrough.first == 0 && count(passthrough.second, "--profile") == 1 &&
                passthrough.second.find("# placement: single") != std::string::npos &&
                passthrough.second.find("executor=engine") != std::string::npos,
                "a native --profile keeps its passthrough and the dry-run states the placement");
        const auto root = fs::temp_directory_path() / "strix-cli-tests-replica-data";
        fs::remove_all(root);
        fs::create_directories(root / "models");
        fs::create_directories(root / "support");
        fs::copy_file(data / "support/matrix.toml", root / "support/matrix.toml");
        for (const auto& item : fs::directory_iterator(data / "models"))
            if (item.path().filename() != "qwen3.8-27b.toml") fs::copy_file(item.path(), root / "models" / item.path().filename());
        std::ofstream(root / "models/qwen3.8-27b.toml") << text.str();
        refuses([&] { (void)through_dispatch(root, with({"--device-count", "2"}, {"--profile", "fast", "--n-gen", "32"})); },
                "runs behind strix-router");
        const auto shown_replica = through_dispatch(root, with({"--device-count", "2", "--dry-run"},
                                                               {"--profile", "fast", "--n-gen", "32"}));
        require(shown_replica.first == 0 && shown_replica.second.find("# placement: replica") == 0 &&
                shown_replica.second.find("executor=router") != std::string::npos &&
                shown_replica.second.find("worker command") != std::string::npos &&
                shown_replica.second.find("strix-cuda-qwen35-decode") != std::string::npos,
                "a replica dry-run names the placement, its executor and labels the worker command");
        fs::remove_all(root);
    }
    req.profile = Name::fast;
    req.args = {"--kv=fp8"};
    refuses([&] { (void)resolve(req, entries, matrix, probes, "bin", exists, &dense); }, "requires the turbo preset");
    req.args = {"--profile", "exact"};
    refuses([&] { (void)resolve(req, entries, matrix, probes, "bin", exists, &dense); }, "before engine arguments");
    refuses([] { (void)parse({"run", "qwen3.8-27b", "--profile", "quality"}); }, "requires exact, fast or turbo");

    const auto invalid = strix::model_description::parse(
        "[presets]\ndefault = 'fast'\n[presets.fast]\nkv = 'fp8'\n");
    refuses([&] { (void)profile::resolve_model_preset(invalid, {}, "cuda"); }, "higher tier");

    const auto base = fs::current_path() / "model-preset-test";
    fs::create_directories(base / "launch");
    {
        std::ofstream config(base / "launch/qwen3.8-27b.toml");
        config << "[run]\nasset = 'model path.gguf'\nbackend = 'cuda'\nargs = ['--n-gen', '32']\n";
    }
    const auto local = profile::load_model_launch(base, "qwen3.8-27b", "run");
    require(local.asset == "model path.gguf" && local.backend == "cuda" &&
            local.args == std::vector<std::string>{"--n-gen", "32"}, "local asset and arguments");
    require(profile::load_model_launch(base, "qwen3.8-27b", "serve").asset.empty(),
            "CLI settings are separate from serve");
    fs::remove(base / "launch/qwen3.8-27b.toml");
    fs::remove(base / "launch");
    fs::remove(base);

    profile::Lever candidate{.name = "candidate", .env = {}, .option = {}, .value = {"", "1", "", ""}, .builtin = "0",
        .kind = profile::Kind::bit_changing, .ready = {true, false, true, false},
        .runnable = {true, true, true, false}, .target = {}, .off = "0", .source = {},
        .provisional = {false, true, false, false}, .identity = {{{"0", "v1"}, {"1", "v1"}}}};
    profile::Inputs inputs;
    inputs.profile = Name::fast;
    const profile::LadderRow timing{candidate.name, candidate.value[1], Name::fast, profile::Phase::prefill,
                                    "fixture", 8, 8, 1, 1, 1.02, 0.001, "fixture timing", false, {}, {},
                                    "ladder-arm/v4;fixture-exec;candidate=0{v1};candidate=candidate=1{v1};tier=fast"};  // measured over the builtin (LadderArms::over)
    const profile::Engine engine{"preset-test", "", std::span(&candidate, 1), false, {}, {}, {}, std::span(&timing, 1)};
    inputs.regime = {profile::Phase::prefill, 8, 1, "fixture", "fixture-exec"};
    require(profile::resolve(engine, inputs).find("candidate")->status == profile::Status::provisional,
            "legacy native profiles retain experimental admission");
    const bool preset_before_scope = profile::model_preset();
    {
        profile::PresetScope scope;
        require(profile::model_preset(), "preset scope enables shared preset environment");
        require(profile::resolve(engine, inputs).find("candidate")->status == profile::Status::pending,
                "model preset waits for the FAST gate");
        candidate.ready[1] = true;
        candidate.provisional[1] = false;
        require(profile::resolve(engine, inputs).find("candidate")->status == profile::Status::set,
                "model preset admits the qualified FAST row");
    }
    require(profile::model_preset() == preset_before_scope,
            "preset scope restores the original process environment");
}
void deployment(const fs::path& fixture, const Probes& probes, const ExecutableExists& exists) {
    const auto asset = fs::current_path() / "cli-fixture" / "minimax-m3.1-flash";
    const auto file = asset / "model.safetensors";
    fs::create_directories(asset);
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code ec;
            fs::remove(path, ec);
            fs::remove(path.parent_path(), ec);
            fs::remove(path.parent_path().parent_path(), ec);
        }
    } cleanup{file};
    const std::string header = R"({"embed_tokens.weight":{"dtype":"BF16","shape":[1,6144],"data_offsets":[0,12288]},"lm_head.trellis":{"dtype":"I16","shape":[1,1,16],"data_offsets":[12288,12320]}})";
    {
        std::ofstream stream(file, std::ios::binary);
        const auto size = static_cast<std::uint64_t>(header.size());
        for (unsigned i = 0; i < 8; ++i) stream.put(static_cast<char>((size >> (8U * i)) & 255U));
        stream << header;
        stream.seekp(static_cast<std::streamoff>(8U + size + 12319U));
        stream.put('\0');
        require(static_cast<bool>(stream), "write plan fixture");
    }
    const auto plan = parse({"plan", asset.string(), "--context", "2048", "--devices",
                             (fixture / "devices.toml").string(), "--json"});
    std::ostringstream out;
    require(dispatch(plan, "test bin", STRIX_CLI_BUILD_DATA, probes, exists, out) == 0, "MiniMax asset plan");
    const auto result = parse_json(out.str());
    require(result.find("status")->as_string() == "planned" && !result.find("measurement")->as_bool() &&
            result.find("route")->find("chosen")->as_string().starts_with("pair:") &&
            result.find("deployment")->find("ranks")->size() == 2 &&
            result.find("route")->find("route_id")->as_string() ==
                result.find("deployment")->find("route")->find("route_id")->as_string(),
            "CLI plan and native pair admission report the same selected route");

    const auto* binding = deployment_binding("minimax");
    const auto footprint = binding->footprints(asset);
    const auto provider = binding->route(*footprint);
    strix::deploy::Spec spec;
    spec.link = "usb4auto";
    spec.context = 2048;
    std::vector<strix::deploy::DeviceFacts> observations;
    for (const auto& rank : result.find("deployment")->find("ranks")->as_array()) {
        strix::deploy::DeviceFacts fact;
        fact.mem_total_bytes = static_cast<std::uint64_t>(rank.find("total_bytes")->as_int());
        fact.mem_free_bytes = static_cast<std::uint64_t>(rank.find("free_bytes")->as_int());
        fact.source = "fixture observation";
        observations.push_back(fact);
    }
    const auto native = strix::deploy::resolve_native_route(spec, observations, *provider);
    require(!native.refused && parse_json(native.route_json).find("route_id")->as_string() ==
        result.find("route")->find("route_id")->as_string(), "CLI route equals provider resolution with the same budgets");
}
} // namespace

int main() {
    try {
        const Probes probes;
        const ExecutableExists exists = [](const auto&) { return true; };
        parsing();
        inspection(STRIX_CLI_FIXTURE, probes, exists);
        support_policy(STRIX_CLI_FIXTURE, exists);
        launch(STRIX_CLI_FIXTURE, exists);
        deployment(STRIX_CLI_FIXTURE, probes, exists);
        model_presets(STRIX_CLI_BUILD_DATA, [](const auto&) { return true; });
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
