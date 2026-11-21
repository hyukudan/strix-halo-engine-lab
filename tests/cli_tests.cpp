#include "strix/cli/cli.hpp"
#include "strix/serve/json.hpp"

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
    Platform platform() const override { return Platform::linux_host; }
    Check os() const override { return {"os", Status::ok, "test Linux"}; }
    Check cpu() const override { return {"cpu", Status::ok, "test \"CPU\""}; }
    Check ram() const override { return {"ram", Status::warn, "unavailable"}; }
    Check runtime(std::string_view name) const override {
        return {std::string(name), name == "hip" ? Status::ok : Status::warn, "load probe"};
    }
    Check node(std::string_view name) const override { return {std::string(name), Status::fail, "access denied"}; }
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
            checks.back().find("detail")->as_string().find("strix-hip") != std::string::npos, "doctor status and recommendation");

    const auto dry = invoke({"serve", "fixture-qwen", "--host", "::1", "--port", "9000", "--dry-run",
        "--", "model path.gguf", "--http-batch", "0"});
#if defined(_WIN32)
    const std::string expected = "\"model path.gguf\" \"--http-serve\" \"--http-bind\" \"::1\" \"--http-port\" \"9000\" \"--http-batch\" \"0\"";
#else
    const std::string expected = "'model path.gguf' '--http-serve' '--http-bind' '::1' '--http-port' '9000' '--http-batch' '0'";
#endif
    require(dry.find("strix-hip-decode-token") != std::string::npos && dry.ends_with(expected + "\n"), "serve dry run");
}

void launch(const fs::path& data, const ExecutableExists& exists) {
    auto entries = strix::model_registry::load((data / "models/registry.toml").string());
    const fs::path bin = "test bin";
    const auto resolve_args = [&](const std::vector<std::string>& args) { return resolve(parse(args), entries, bin, exists); };
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
    refuses([&] { (void)resolve(parse({"run", "fixture-qwen"}), entries, bin, [](const auto&) { return false; }); },
            "no matching executable");
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
            result.find("route")->find("chosen")->is_null() && result.find("deployment")->find("ranks")->size() == 2,
            "declared pair memory can fit while the pending layer-cut stays unselected");
}
} // namespace

int main() {
    try {
        const Probes probes;
        const ExecutableExists exists = [](const auto&) { return true; };
        parsing();
        inspection(STRIX_CLI_FIXTURE, probes, exists);
        launch(STRIX_CLI_FIXTURE, exists);
        deployment(STRIX_CLI_FIXTURE, probes, exists);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
