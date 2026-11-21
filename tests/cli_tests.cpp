#include "strix/cli/cli.hpp"
#include "strix/deploy/route.hpp"
#include "strix/serve/json.hpp"
#include "../src/cli/detail.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

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
    const auto resolve_args = [&](const std::vector<std::string>& args) {
        return resolve(parse(args), entries, matrix, Platform::linux_host, bin, exists);
    };
    const auto qwen = resolve_args({"run", "assets/fixture-qwen.gguf", "--dry-run", "--", "1", "--generate", "256"});
    require(qwen.executable.parent_path() == bin && qwen.executable.stem() == "strix-hip-decode-token", "Qwen run route");
    require(qwen.args == std::vector<std::string>{"assets/fixture-qwen.gguf", "1", "--generate", "256"}, "run passthrough");
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
    refuses([&] { (void)resolve(parse({"run", "fixture-qwen"}), entries, matrix, Platform::linux_host, bin,
                                [](const auto&) { return false; }); }, "no matching executable");
    matrix.rows[0].state = support::State::declared;
    refuses([&] { (void)resolve_args({"run", "fixture-qwen"}); },
            "qwen/hip/gfx1200/linux/cli-generate (fixture-gguf) [implemented] is below connected");
    matrix.rows[0].state = support::State::packaged;
    entries.back().apps.clear();
    refuses([&] { (void)resolve_args({"run", "fixture-qwen"}); }, "no registered run route");
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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
