// Hy4-flash serve binding (host only, no device): the strix-serve shell selects the hy4_flash factory (--list-models,
// --inspect, --family-inspect, --print-effective-config, create refused without a device binding); ONE RESOLVER parity
// (the CLI's parse of an argv and the serve shell's family arguments give byte-identical effective configs); the request
// path through the shared ApiService (reasoning_effort no_think turns thinking off for the output parse, prompt bytes equal
// the CLI render); a tool-call round trip through the bound renderer and parser; GGUF family admission.
#include "hy4/hy4_speculation.hpp"
#include "hy4/serve/session_service.hpp"
#include "strix/hy4/profile_options.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/model_family_registry.hpp"
#include "strix/serve/engine.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>

namespace sv = strix::serve;
namespace h = strix::hy4::serve;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

struct Invocation { int code; std::string out, error; };
Invocation invoke(std::vector<std::string> args, const sv::ServiceFactories& factories) {
    std::vector<char*> pointers;
    for (auto& s : args) pointers.push_back(s.data());
    std::ostringstream out, error;
    const int code = sv::run(static_cast<int>(pointers.size()), pointers.data(), factories, out, error);
    return {code, out.str(), error.str()};
}

void write(const std::filesystem::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary);
    f << text;
    check(bool(f), "write " + p.string());
}

// One id per byte: the decoded prompt is the rendered text byte for byte.
struct ByteTokenizer final : sv::ApiTokenizer {
    std::vector<std::uint32_t> encode(std::string_view text, bool) const override {
        std::vector<std::uint32_t> ids;
        for (const char c : text) ids.push_back(static_cast<unsigned char>(c));
        return ids;
    }
    std::string decode(std::span<const std::uint32_t> ids, bool = false) const override {
        std::string text;
        for (const auto id : ids) text.push_back(static_cast<char>(id));
        return text;
    }
    std::string token_bytes(std::uint32_t id) const override { return std::string(1, static_cast<char>(id)); }
    std::size_t vocab_size() const override { return 256; }
    std::uint32_t find(std::string_view) const override { return 0; }
};

const char* kTools =
    R"([{"type":"function","function":{"name":"get_weather","description":"Weather of a city","parameters":{"type":"object",)"
    R"("properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}])";

void option_tests() {
    const std::vector<std::string> args{"--gguf", "p.gguf", "--ctx", "8192", "--max-tokens", "512", "--model-id", "hy"};
    const auto o = h::parse_options(args);
    check(o.gguf == "p.gguf" && o.ctx == 8192U && o.ctx_given && o.max_tokens == 512U && o.model_id == "hy", "options parsed");
    const auto refused = [](std::vector<std::string> a) {
        try { (void)h::parse_options(a); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(refused({"--bogus"}), "unknown option refused");
    check(refused({"--ctx"}), "missing value refused");
    check(refused({"--max-tokens", "0"}), "zero max tokens refused");
    check(refused({"--ctx", "-1"}), "negative ctx refused");
    // The DFly2 arm under the shared selector is opt-in: no option, no second drafter (a shard beside the pack is not
    // bound by discovery); --dfly FILE | beside names the request.
    check(o.dfly.empty(), "no DFly2 arm without --dfly");
    check(h::parse_options(std::vector<std::string>{"--dfly", "beside"}).dfly == "beside", "--dfly beside parsed");
    check(h::parse_options(std::vector<std::string>{"--dfly", "d.safetensors"}).dfly == "d.safetensors", "--dfly FILE parsed");
    check(refused({"--dfly"}), "--dfly without a value refused");
    check(refused({"--dfly", ""}), "--dfly with an empty value refused");
    check(refused({"--ctx", "12k"}), "malformed number refused");
}

void shell_tests(const std::filesystem::path& root) {
    std::filesystem::create_directories(root / "models");
    write(root / "models" / "registry.toml",
          "schema = \"strix-registry/v1\"\n\n[models.hy4_flash]\nname = \"hy4-flash\"\ndescription = \"models/hy4-flash.toml\"\n"
          "family = \"hy4_flash\"\ncmake = \"\"\nprofile_tag = \"hy4\"\nallow_pending_env = \"\"\ndefaults = []\napps = []\nadapter = []\n");
    write(root / "models" / "hy4-flash.toml", "name = \"hy4-flash\"\nschema = \"strix-model/v1\"\n");
    sv::ServiceFactories factories;
    h::install_session_service(factories);
    const std::string r = root.string();

    const auto list = invoke({"serve", "--registry-root", r, "--list-models"}, factories);
    check(list.code == 0 && list.out.find("hy4_flash\thy4-flash\thy4_flash\tbuilt") != std::string::npos, "list-models: " + list.out + list.error);
    const auto inspect = invoke({"serve", "--registry-root", r, "--model", "hy4_flash", "--inspect"}, factories);
    check(inspect.code == 0 && inspect.out == "hy4_flash\thy4-flash\thy4_flash\tmodels/hy4-flash.toml\tbuilt\n", "inspect: " + inspect.out + inspect.error);
    const auto family = invoke({"serve", "--registry-root", r, "--model", "hy4_flash", "--family-inspect", "--", "--max-tokens", "512"}, factories);
    check(family.code == 0 && family.out.find("models/hy4-flash.toml") != std::string::npos &&
              family.out.find("\"max_tokens\":{\"value\":512,\"origin\":\"cli\"}") != std::string::npos &&
              family.out.find("\"serve_only\"") != std::string::npos,
          "family-inspect: " + family.out + family.error);
    const auto help = invoke({"serve", "--registry-root", r, "--model", "hy4_flash", "--", "--help"}, factories);
    check(help.code == 0 && help.out == h::usage(), "family --help");
    const auto bad = invoke({"serve", "--registry-root", r, "--model", "hy4_flash", "--", "--bogus"}, factories);
    check(bad.code == 2 && bad.error.find("unknown hy4_flash option: --bogus") != std::string::npos, "bad option: " + bad.error);
    const auto nopack = invoke({"serve", "--registry-root", r, "--model", "hy4_flash"}, factories);
    check(nopack.code == 2 && nopack.error.find("requires --gguf") != std::string::npos, "serve without pack: " + nopack.error);
    const auto missing = invoke({"serve", "--registry-root", r, "--model", "hy4_flash", "--", "--gguf", (root / "absent.gguf").string()}, factories);
    check(missing.code == 2, "absent pack refused before device work");

    // ONE RESOLVER: the CLI's parse of the argv and the serve shell's family arguments end with the same bytes.
    for (const auto& args : std::vector<std::vector<std::string>>{
             {}, {"--ctx", "8192", "--max-tokens", "512", "--model-id", "hy"}, {"--model-id", "hy", "--max-tokens", "512", "--ctx", "8192"}}) {
        const auto o = h::parse_options(args);
        const auto cli = h::effective_config(o, h::read_pack_facts(o)).dump() + "\n";
        std::vector<std::string> argv{"serve", "--registry-root", r, "--model", "hy4_flash", "--"};
        argv.insert(argv.end(), args.begin(), args.end());
        argv.emplace_back("--print-effective-config");
        const auto served = invoke(argv, factories);
        check(served.code == 0 && served.out == cli, "CLI = serve effective config: " + served.out + " vs " + cli);
    }
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        namespace hp = strix::hy4::profile;
        const std::vector<std::string> switches{"--profile", tier, "--prompt-rows", "2048", "--prompt-rows=0", "--head-q8=0"};
        auto rank_args = switches;
        strix::profile::Inputs rank;
        const auto native_overrides = strix::hy4::take_profile_options(rank_args, rank);
        check(rank_args.empty() && native_overrides.prompt_rows == 0U && native_overrides.head_q8 == false,
              "native parser preserves explicit off and the last prompt width");
        const auto native = hp::resolve({hp::Device::unknown, hp::Topology::one_node, 8192}, rank);
        std::vector<std::string> argv{"serve", "--registry-root", r, "--model", "hy4_flash", "--", "--ctx", "8192",
                                      "--print-effective-config"};
        argv.insert(argv.end(), switches.begin(), switches.end());
        const auto served = invoke(argv, factories);
        check(served.code == 0, "serve profile inspection: " + served.out + served.error);
        if (served.code == 0) {
            const auto config = sv::parse_json(served.out);
            const auto* resolved = config.find("profile");
            check(resolved && resolved->dump() == sv::parse_json(native->resolution.json()).dump(),
                  "native rank and serve resolve the same tier and equals-spelled overrides without a device");
        }
    }
    for (const auto& switches : std::vector<std::vector<std::string>>{
             {"--head-q8", "10"}, {"--prompt-rows", "2048x"}, {"--prompt-rows", "4294967296"}}) {
        std::vector<std::string> argv{"serve", "--registry-root", r, "--model", "hy4_flash", "--", "--ctx", "8192",
                                      "--print-effective-config"};
        argv.insert(argv.end(), switches.begin(), switches.end());
        const auto served = invoke(argv, factories);
        auto rank_args = switches;
        strix::profile::Inputs rank;
        std::string rank_error;
        try { (void)strix::hy4::take_profile_options(rank_args, rank); }
        catch (const std::invalid_argument& error) { rank_error = error.what(); }
        check(served.code == 2 && !rank_error.empty() && served.error.find(rank_error) != std::string::npos,
              "native rank and serve refuse malformed explicit profile overrides with the same diagnostic");
    }
    // Exact speculative decode: one declaration, the shared default resolver decides for every tier.
    {
        strix::hy4::SpeculationRun run;
        run.block_in_pack = true;
        check(strix::hy4::speculation(run).enabled && strix::hy4::speculation(run).source == "default", "Hy4 MTP is the default with the block");
        run.whole_model = false;
        check(!strix::hy4::speculation(run).enabled && strix::hy4::speculation(run).source == "unavailable", "Hy4 layer split has no block");
        run.whole_model = true;
        run.block_in_pack = false;
        check(!strix::hy4::speculation(run).enabled, "Hy4 pack without the block decodes plain");
        run.block_in_pack = true;
        run.no_mtp = true;
        check(strix::hy4::speculation(run).source == "cli-opt-out", "Hy4 --no-mtp");
        run.no_mtp = false;
        run.environment = "0";
        check(strix::hy4::speculation(run).source == "env-opt-out", "Hy4 STRIX_MTP=0");
        run.environment = nullptr;
        run.per_step_route = true;
        check(!strix::hy4::speculation(run).enabled && strix::hy4::speculation(run).source == "explicit-route",
                "Hy4 per-step logits keep the plain steps");
        run.explicit_mtp = true;
        check(strix::hy4::speculation(run).enabled && strix::hy4::speculation(run).source == "explicit", "Hy4 --mtp wins over the per-step route");
    }
    {  // the decode route: the native rank's shared speculation decision (pack facts: the block ships in the pack)
        h::PackFacts with_block;
        with_block.next_token_block = true;
        for (const bool no_mtp : {false, true}) {
            const auto o = h::parse_options(no_mtp ? std::vector<std::string>{"--no-mtp"} : std::vector<std::string>{});
            const auto config = h::effective_config(o, with_block);
            const auto* spec = config.find("speculation");
            strix::hy4::SpeculationRun run;
            run.block_in_pack = true;
            run.no_mtp = no_mtp;
            run.environment = std::getenv(strix::session::speculation_environment);
            const auto native = strix::hy4::speculation(run);
            check(spec && spec->find("enabled") && spec->find("source") &&
                      spec->find("enabled")->dump() == sv::Json(native.enabled).dump() &&
                      spec->find("source")->dump() == sv::Json(std::string(native.source)).dump(),
                  "serve reports the native speculation decision");
        }
        const auto plain = h::effective_config(h::parse_options(std::vector<std::string>{}), {});
        const auto* none = plain.find("speculation");
        check(none && none->find("source") && none->find("source")->dump() == sv::Json("unavailable").dump(),
              "no pack, no block: plain decode");
    }
    const auto a = h::effective_config(h::parse_options(std::vector<std::string>{"--ctx", "8192", "--max-tokens", "512"}), {}).dump();
    const auto b = h::effective_config(h::parse_options(std::vector<std::string>{"--max-tokens", "512", "--ctx", "8192"}), {}).dump();
    check(a == b, "argument order does not change the effective config");
}

void request_tests() {
    const auto o = h::parse_options(std::vector<std::string>{"--max-tokens", "77"});
    sv::ApiConfig config;
    h::bind_api(config, o, h::read_pack_facts(o));
    check(config.default_max_tokens == 77U && config.model_id == "hy4-flash" && config.sampling_policy == "greedy", "bound defaults");
    check(config.stop_tokens == std::vector<std::uint32_t>{120025U} && config.tool_call_marker == "<tool_call", "bound stop/marker");
    check(config.thinking_off_efforts == std::vector<std::string>{"no_think"}, "no_think closes the reasoning block");

    sv::GenerationQueue queue;
    const auto tokenizer = std::make_shared<ByteTokenizer>();
    sv::ApiService api(config, sv::ApiTokenizerBinding(tokenizer), queue, 8192, false);
    const auto prompt = [&](const std::string& body) { return api.prepare_chat_prompt(sv::parse_json(body)); };
    const std::string user = R"("messages":[{"role":"user","content":"hi"}])";
    const auto render = [](std::optional<std::string> effort, std::optional<bool> thinking) {
        sv::ChatTemplateInput in;
        in.messages = sv::parse_json(R"([{"role":"user","content":"hi"}])");
        in.reasoning_effort = std::move(effort);
        in.enable_thinking = thinking;
        return sv::render_hy4_chat(in);
    };
    {
        const auto p = prompt("{" + user + R"(,"reasoning_effort":"no_think"})");
        check(!p.thinking, "no_think: thinking off for the parse");
        check(tokenizer->decode(p.tokens) == render("no_think", false), "no_think prompt = CLI render");
    }
    {
        const auto p = prompt("{" + user + R"(,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"no_think"}})");
        check(!p.thinking, "no_think wins over enable_thinking true (the template closes the block)");
    }
    {
        const auto p = prompt("{" + user + R"(,"reasoning_effort":"max"})");
        check(p.thinking && tokenizer->decode(p.tokens) == render("max", std::nullopt), "max: thinking on, prompt = CLI render");
    }
    {
        const auto p = prompt("{" + user + "}");
        check(p.thinking && tokenizer->decode(p.tokens) == render(std::nullopt, std::nullopt), "default high: prompt = CLI render");
        check(tokenizer->decode(p.tokens).find("reasoning_effort:high") != std::string::npos, "default effort high rendered");
    }
    {
        const auto p = prompt("{" + user + R"(,"reasoning_effort":"none"})");
        check(!p.thinking && tokenizer->decode(p.tokens) == render(std::nullopt, false), "none = no_think");
    }
    {  // Other families leave thinking_off_efforts empty: the shared layer keeps its previous behaviour.
        sv::ApiConfig plain = config;
        plain.thinking_off_efforts.clear();
        sv::GenerationQueue q2;
        sv::ApiService other(plain, sv::ApiTokenizerBinding(tokenizer), q2, 8192, false);
        check(other.prepare_chat_prompt(sv::parse_json("{" + user + R"(,"reasoning_effort":"no_think"})")).thinking,
              "empty thinking_off_efforts: unchanged behaviour");
    }
    {  // Tool-call round trip: an assistant call rendered by the bound renderer parses back through the bound parser.
        sv::ChatTemplateInput in;
        in.tools = sv::parse_json(kTools);
        in.add_generation_prompt = false;
        in.messages = sv::parse_json(
            R"([{"role":"user","content":"Weather in Paris?"},{"role":"assistant","reasoning_content":"need the tool","content":"Checking.",)"
            R"("tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"Paris","days":2}}}]}])");
        const auto text = config.chat_renderer(in);
        const std::string mid = "<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>", end = "<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>",
                          open = "<think:opensource>";
        const auto at = text.rfind("assistant" + mid);
        const auto stop = text.rfind(end);
        check(at != std::string::npos && stop != std::string::npos && stop > at, "assistant turn rendered");
        if (at != std::string::npos && stop != std::string::npos && stop > at) {
            auto generated = text.substr(at + 9 + mid.size(), stop - at - 9 - mid.size());
            check(generated.rfind(open, 0) == 0, "assistant turn opens the reasoning block");
            generated.erase(0, open.size());
            const auto out = config.output_parser(generated, true, in.tools);
            check(out.reasoning == "need the tool" && out.content == "Checking.", "round trip reasoning/content: " + out.reasoning + "|" + out.content);
            check(out.tool_calls.size() == 1U, "round trip one call");
            if (out.tool_calls.size() == 1U) {
                const auto& call = out.tool_calls.as_array()[0];
                check(call.find("name")->as_string() == "get_weather", "round trip name");
                const auto* args = call.find("arguments");
                check(args && args->find("city")->as_string() == "Paris" && args->find("days")->as_double() == 2.0, "round trip arguments");
            }
        }
    }
}

void placement_tests(const std::filesystem::path& root) {
    namespace md = strix::profile::multi_device;
    const auto model = strix::model_description::load((root / "models/hy4-flash.toml").string());
    const auto policy = md::parse_policy(model);
    strix::device::Topology devices;
    devices.nodes.resize(2);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        auto& node = devices.nodes[rank];
        node.id = node.host = "fake" + std::to_string(rank);
        node.rank = rank;
        node.arch = "gfx1151";
        node.memory_gbytes_ps = 200;
    }
    devices.links.push_back({"link", 0, 1, "usb4", 4.0, 30.0, 0.0, "usb4", true, true});
    md::BandReceipt receipt{"fake-identity", "hy4-flash", "hy4_tp_mtp_decode", strix::device::topology_id(devices),
        "usb4auto/channel", "test-layout", "gfx1151", 2048, 2048, 128, 512, 300, 300, 1.5, 1.3, 0.02};
    const auto query = strix::hy4::pair_query(devices, strix::profile::Name::fast, 277, 2048, 300, "usb4auto",
                                             "gfx1151", "test-layout", &receipt);
    const auto verify = [&](const md::Query& q, md::Form want) {
        const auto native = strix::hy4::resolve_pair_placement(model, q);
        const auto served = h::placement_config(model, q);
        check(native.form == want && native.json(policy, q).dump() == served.dump(), "CLI and serve placement parity");
    };
    verify(query, md::Form::tensor);
    check(!md::select(policy, query).ladder_reference.empty(), "TP has a bound execution key");
    auto changed = query;
    changed.phase = md::Phase::prefill;
    verify(changed, md::Form::layers);
    changed = query;
    changed.tier = strix::profile::Name::standard;
    verify(changed, md::Form::layers);
    changed = query;
    changed.tier = strix::profile::Name::turbo;
    verify(changed, md::Form::tensor);
    changed = query;
    changed.execution.receipt = nullptr;
    verify(changed, md::Form::layers);
    changed = query;
    changed.execution.pair = "different-pair";
    verify(changed, md::Form::layers);
    changed = query;
    changed.execution.link = "usb4auto/staged";
    verify(changed, md::Form::layers);
    changed = query;
    changed.execution.layout = "different-layout";
    verify(changed, md::Form::layers);
    changed = query;
    changed.execution.context = 4096;
    verify(changed, md::Form::layers);
    changed = query;
    changed.prompt_rows = 513;
    verify(changed, md::Form::layers);
    changed = query;
    changed.execution.decode_tokens = 16;
    verify(changed, md::Form::layers);
    changed = query;
    changed.devices = 1;
    verify(changed, md::Form::single);
    receipt.speedup = 1.01;
    verify(query, md::Form::layers);
    receipt.speedup = 1.5;
    receipt.single_speedup = 0.95;
    verify(query, md::Form::layers);
}

void admission_tests() {
    h::register_family();
    h::register_family();  // idempotent
    const auto* family = strix::find_model_family(h::kArchitecture);
    check(family && family->name == h::kFamily, "hy_v4_flash admitted by hy4_flash");
    if (family) {
        bool refused = false;
        try { family->validate(strix::gguf::FileInfo{}); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "metadata without the hy_v4_flash keys refused");
        check(family->reader_limits().max_retained_array_elements > 0U, "reader keeps the per-layer arrays the resolver reads");
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto root = std::filesystem::temp_directory_path() / "strix-hy4-serve-tests";
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        check(argc == 2, "source fixture root passed");
        if (argc == 2) placement_tests(argv[1]);
        option_tests();
        shell_tests(root);
        request_tests();
        admission_tests();
        std::filesystem::remove_all(root, ignored);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception " << e.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "hy4_serve_tests PASS\n";
    return 0;
}
