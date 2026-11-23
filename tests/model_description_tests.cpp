#include "strix/qwen_model_description.hpp"
#include "../apps/qwen_model_description_hook.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace md = strix::model_description;
namespace qd = strix::qwen_description;
namespace {
int checks = 0, failures = 0;
void expect(bool ok, const std::string& message) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class F>
std::string refuses(F action) {
    try { action(); }
    catch (const std::invalid_argument& e) { return e.what(); }
    return {};
}
std::string replace(std::string text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos == std::string::npos) throw std::runtime_error("missing fixture text: " + from);
    text.replace(pos, from.size(), to);
    return text;
}
template<class T>
void le(std::string& bytes, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t i = 0; i < sizeof(T); ++i) bytes += static_cast<char>((value >> (8 * i)) & 255U);
}
void text(std::string& bytes, const std::string& value) {
    le<std::uint64_t>(bytes, value.size()); bytes += value;
}
struct Entry { std::string key; std::uint32_t type; std::string bytes; };
Entry integer(const std::string& key, std::uint64_t value) {
    std::string bytes; le(bytes, value); return {key, 10, bytes};
}
Entry string(const std::string& key, const std::string& value) {
    std::string bytes; text(bytes, value); return {key, 8, bytes};
}
Entry float64(const std::string& key, double value) {
    std::string bytes; le(bytes, std::bit_cast<std::uint64_t>(value)); return {key, 12, bytes};
}
Entry integers(const std::string& key, std::initializer_list<std::uint32_t> values) {
    std::string bytes; le<std::uint32_t>(bytes, 4); le<std::uint64_t>(bytes, values.size());
    for (auto value : values) le(bytes, value);
    return {key, 9, bytes};
}
strix::gguf::FileInfo gguf(const std::vector<Entry>& entries, bool retain = true) {
    std::string bytes = "GGUF";
    le<std::uint32_t>(bytes, 3); le<std::uint64_t>(bytes, 0); le<std::uint64_t>(bytes, entries.size());
    for (const auto& entry : entries) { text(bytes, entry.key); le(bytes, entry.type); bytes += entry.bytes; }
    while (bytes.size() % 32) bytes += '\0';
    std::istringstream input(bytes);
    strix::gguf::Limits limits;
    if (retain) limits.max_retained_array_elements = 4;
    return strix::gguf::read(input, bytes.size(), limits);
}
void parser_tests() {
    const auto positive = md::parse(R"toml(
schema = "strix-model/v1" # comment
name = 'a # literal'
[dims]
hidden = +2560
[extras.child]
enabled = false
value = -1.25e+2
items = [1, # array comment
  "two", true, { width = 4, nested = { kind = "gdn" } }, ]
escaped = "a\n\t\"\\"
)toml");
    expect(std::get<std::int64_t>(positive.find("dims.hidden")->data) == 2560, "decimal integer/table");
    expect(std::get<std::string>(positive.find("name")->data) == "a # literal", "literal string and comments");
    expect(std::get<double>(positive.find("extras.child.value")->data) == -125, "float exponent");
    expect(!std::get<bool>(positive.find("extras.child.enabled")->data), "boolean");
    expect(std::get<md::Value::Array>(positive.find("extras.child.items")->data).size() == 4, "multiline mixed array/inline tables");
    expect(std::get<std::string>(positive.find("extras.child.escaped")->data) == "a\n\t\"\\", "string escapes");
    expect(!positive.find("dims.hidden.missing") && !positive.find("absent"), "absent and scalar-parent lookup");
    expect(std::get<std::int64_t>(md::parse("a.b = 1\na.c = 2\n").find("a.c")->data) == 2, "sibling dotted assignments");
    expect(md::parse("[a.b]\nx=1\n[a]\ny=2\n").find("a.y") != nullptr, "implicit super-table declaration");
    expect(md::parse("a=1\r\nb=2\r\n").find("b") != nullptr, "CRLF input");
    const std::vector<std::string> bad = {
        "a", "=1", "a=", "a=1 junk", "a=1\nb", "a=1\ra=2", "[a", "[a] junk", "[]", "[[a]]",
        "a=\"unterminated", "a='unterminated", "a=\"line\nbreak\"", "a=\"\\q\"", "a=\"\\u0041\"",
        "a=\"\"\"multiline\"\"\"", "a=True", "a=truth", "a=nan", "a=inf", "a=-inf", "a=01", "a=-01",
        "a=0x10", "a=1_000", "a=1.", "a=.5", "a=1e", "a=1e+", "a=1e9999", "a=1e-9999",
        "a=9223372036854775808", "a=-9223372036854775809", "a=2026-01-01", "a=[1 2]", "a=[1,,2]",
        "a=[1", "a={x=1,}", "a={x=1 y=2}", "a={x=1\ny=2}", "a={x=1", "a=1\na=2",
        "a=1\na=1", "[a]\nx=1\n[a]\ny=2", "a.b=1\n[a]\nx=2", "a=1\na.b=2",
        "a.b=1\na=2", "a={x=1,x=2}", "a={x=1}\na.y=2", "a={x=1}\n[a]\ny=2",
        "a={b={x=1}}\na.b.y=2", "a={b={x=1}, b.y=2}", "'quoted'=1", "a..b=1"
    };
    for (const auto& input : bad)
        expect(!refuses([&] { (void)md::parse(input); }).empty(), "strict parser refusal: " + input);
    const auto location = refuses([] { (void)md::parse("a=1\na=2"); });
    expect(location.find("line 2 column") != std::string::npos, "parser location diagnostic");
    const std::string nested = "a=" + std::string(34, '[') + "0" + std::string(34, ']');
    expect(!refuses([&] { (void)md::parse(nested); }).empty(), "nesting bound");
    expect(std::get<std::int64_t>(md::parse("a=-9223372036854775808").find("a")->data) == INT64_MIN, "minimum int64");
    expect(std::get<std::string>(md::parse("a='\xc3\xb1\xf0\x9f\x98\x80'").find("a")->data).size() == 6, "UTF-8 strings");
    for (const auto& bytes : {std::string("\xc0\x80"), std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"),
             std::string("\xc3"), std::string("\xc3\x20"), std::string(1, '\0'), std::string(1, '\x7f')})
        expect(!refuses([&] { (void)md::parse("# " + bytes); }).empty(), "invalid UTF-8/control byte in comment");
}
void contract_tests(const std::string& source) {
    const auto baked = strix::ModelContract::qwen38_flash_next();
    const auto description = md::parse(source);
    qd::assert_matches(baked, description);
    const auto model = qd::contract_from_description(description);
    expect(model.validate().empty() && model.hidden_size == 2560 && model.num_experts == 512, "positive model construction");
    // Mutation coverage must name every execution contract leaf, including
    // values that remain semantically valid after the change.
    std::vector<std::string> leaves;
    const auto collect = [&](const auto& self, const md::Value::Table& table, const std::string& prefix) -> void {
        for (const auto& [key, value] : table) {
            if (prefix.empty() && md::is_launch_metadata(key)) continue;
            const auto path = prefix + key;
            if (const auto* child = std::get_if<md::Value::Table>(&value.data)) self(self, *child, path + ".");
            else leaves.push_back(path);
        }
    };
    collect(collect, description.fields, "");
    auto all_wrong = description;
    for (const auto& path : leaves) {
        auto* table = &all_wrong.fields;
        std::size_t begin = 0, dot = path.find('.');
        while (dot != std::string::npos) {
            table = &std::get<md::Value::Table>(table->at(path.substr(begin, dot - begin)).data);
            begin = dot + 1; dot = path.find('.', begin);
        }
        auto& value = table->at(path.substr(begin));
        if (auto* integer_value = std::get_if<std::int64_t>(&value.data)) ++*integer_value;
        else if (auto* float_value = std::get_if<double>(&value.data)) *float_value += 1;
        else if (auto* bool_value = std::get_if<bool>(&value.data)) *bool_value = !*bool_value;
        else if (auto* string_value = std::get_if<std::string>(&value.data)) *string_value += "-wrong";
        else if (auto* array_value = std::get_if<md::Value::Array>(&value.data)) array_value->clear();
    }
    const auto all_errors = refuses([&] { qd::assert_matches(baked, all_wrong); });
    for (const auto& path : leaves)
        expect(all_errors.find(path + ": expected ") != std::string::npos, "exhaustive aggregated contract mismatch: " + path);
    const auto graph = qd::graph_template(description);
    std::size_t gdn = 0, qsa = 0, ple = 0;
    for (const auto& layer : graph.layers) {
        if (layer.attention == qd::AttentionKind::gdn) ++gdn; else ++qsa;
        if (layer.ple) ++ple;
        expect(layer.index < 48 && layer.ple == (layer.index == 1), "explicit layer/PLE template");
        expect(layer.state_effects.size() == (layer.ple ? 3U : 2U), "declared state effects for each layer");
        expect(layer.state_effects[0] == (layer.attention == qd::AttentionKind::gdn ? qd::StateEffect::gdn_recurrence : qd::StateEffect::qsa_kv),
            "typed recurrent versus KV state effect");
    }
    expect(gdn == 36 && qsa == 12 && ple == 1, "typed graph counts");
    const std::vector<std::pair<std::string, std::string>> changes = {
        {"schema = \"strix-model/v1\"", "schema = \"strix-model/v2\""},
        {"schema = \"strix-model/v1\"", "schema = 1"},
        {"schema = \"strix-model/v1\"", ""},
        {"hidden = 2560", "hidden = \"2560\""}, {"hidden = 2560", "hidden = 2560.0"},
        {"hidden = 2560", "hidden = true"}, {"hidden = 2560", "hidden = [2560]"},
        {"hidden = 2560", "hidden = {value=2560}"}, {"hidden = 2560", ""},
        {"hidden = 2560", "hidden = -1"}, {"hidden = 2560", "hidden = 0"},
        {"n_kv_heads = 2", "n_kv_heads = 0"}, {"n_heads = 24", "n_heads = 23"},
        {"norm_topk_probability = true", "norm_topk_probability = 1"},
        {"norm_topk_probability = true", "norm_topk_probability = false"},
        {"theta = 10000000.0", "theta = \"10000000\""},
        {"theta = 10000000.0", "theta = 1e300"},
        {"rms_epsilon = 1e-6", "rms_epsilon = 0.0"},
        {"experts = 512", "experts = 8"}, {"sections = [11, 11, 10]", "sections = [10, 11, 11]"},
        {"sections = [11, 11, 10]", "sections = [11.0, 11, 10]"},
        {"pattern = [\"gdn\", \"gdn\", \"gdn\", \"qsa\"]", "pattern = [\"qsa\", \"gdn\", \"gdn\", \"gdn\"]"},
        {"attention_kv = \"bf16\"", "attention_kv = \"int8_token\""},
        {"attention_kv = \"bf16\"", "attention_kv = \"unknown\""},
        {"expert_down = [\"q5_1\", \"q8_0\"]", "expert_down = [\"q4_k\"]"},
        {"original_max = 262144", "original_max = 131072"},
        {"[dims]", "[dims]\nmisspelled = 2560"},
        {"[dims]", "[unknown]\n[ dims ]"}
    };
    for (const auto& [before, after] : changes) {
        const auto changed = md::parse(replace(source, before, after));
        const auto error = refuses([&] { qd::assert_matches(baked, changed); });
        expect(error.find("expected ") != std::string::npos && error.find("actual ") != std::string::npos,
            "contract/schema/type refusal: " + after);
    }
    auto changed = replace(source, "hidden = 2560", "hidden = 1280");
    changed = replace(changed, "experts = 512", "experts = 256");
    changed = replace(changed, "top_k = 10", "top_k = 5");
    const auto errors = refuses([&] { qd::assert_matches(baked, md::parse(changed)); });
    for (const auto* field : {"dims.hidden: expected 2560, actual 1280", "moe.experts: expected 512, actual 256", "moe.top_k: expected 10, actual 5"})
        expect(errors.find(field) != std::string::npos, std::string("aggregated mismatch ") + field);
    expect(!refuses([&] { (void)qd::contract_from_description(md::parse(replace(source, "n_kv_heads = 2", "n_kv_heads = 0"))); }).empty(), "construction semantic check (zero divisor)");
    expect(!refuses([&] { (void)qd::contract_from_description(md::parse(replace(source, "n_heads = 24", "n_heads = 23"))); }).empty(), "construction semantic head divisibility");
    expect(!refuses([&] { (void)qd::contract_from_description(md::parse(replace(source, "experts = 512", "experts = 8"))); }).empty(), "construction semantic top-k");
    expect(!refuses([&] { (void)qd::graph_template(md::parse(changed)); }).empty(), "graph refuses different compiled template");
    auto exact = baked; exact.hidden_size = 1280; exact.num_experts = 256; exact.num_experts_per_token = 5;
    qd::assert_matches(exact, md::parse(changed));
    expect(true, "assert_matches uses supplied baked contract");
}
void gguf_tests() {
    const auto baked = strix::ModelContract::qwen38_flash_next();
    const std::vector<Entry> correct = {
        string("general.architecture", "qwen4exp"), integer("qwen4exp.embedding_length", 2560),
        integer("qwen4exp.block_count", 48), integer("qwen4exp.attention.head_count", 24),
        integer("qwen4exp.attention.head_count_kv", 2), integer("qwen4exp.attention.key_length", 256),
        integer("qwen4exp.expert_count", 512), integer("qwen4exp.expert_used_count", 10),
        integer("qwen4exp.expert_feed_forward_length", 640), integer("qwen4exp.ssm.group_count", 16),
        integer("qwen4exp.ssm.state_size", 128), integer("qwen4exp.ssm.time_step_rank", 48),
        integer("qwen4exp.ssm.inner_size", 6144), integer("qwen4exp.indexer.head_count", 4),
        integer("qwen4exp.rope.dimension_count", 64), float64("qwen4exp.rope.freq_base", 10000000)
    };
    expect(qd::assert_gguf_matches(baked, gguf(correct)).size() == correct.size(), "synthetic GGUF positive/all keys checked");
    expect(qd::assert_gguf_matches(baked, gguf({integer("unrelated.key", 1)})).empty(), "absent recognized keys are not invented");
    for (std::size_t i = 0; i < correct.size(); ++i) {
        auto wrong = correct;
        wrong[i] = integer(correct[i].key, 1);
        const auto error = refuses([&] { (void)qd::assert_gguf_matches(baked, gguf(wrong)); });
        expect(error.find(correct[i].key + ": expected ") != std::string::npos, "synthetic GGUF refuses " + correct[i].key);
    }
    auto wrong = correct;
    wrong[1] = string(wrong[1].key, "2560");
    wrong[2] = float64(wrong[2].key, 48.0);
    const auto errors = refuses([&] { (void)qd::assert_gguf_matches(baked, gguf(wrong)); });
    expect(errors.find("qwen4exp.embedding_length: expected 2560") != std::string::npos &&
        errors.find("qwen4exp.block_count: expected 48") != std::string::npos, "GGUF multiple dimension/type mismatches");
    expect(!refuses([&] { (void)qd::assert_gguf_matches(baked, gguf({integers("tokenizer.ggml.tokens", {1,2})})); }).empty(), "GGUF tokenizer element type/length refusal");
    auto duplicate = correct; duplicate.push_back(correct[1]);
    bool refused = false;
    try { (void)gguf(duplicate); } catch (const std::runtime_error&) { refused = true; }
    expect(refused, "existing GGUF reader rejects duplicate metadata");
}
struct Environment {
    const char* name;
    std::optional<std::string> previous;
    explicit Environment(const char* key) : name(key) {
        if (const char* value = std::getenv(name)) previous = value;
        set(nullptr);
    }
    void set(const char* value) const {
#ifdef _WIN32
        if (_putenv_s(name, value ? value : "") != 0) throw std::runtime_error("cannot set test environment");
#else
        if (value ? setenv(name, value, 1) != 0 : unsetenv(name) != 0)
            throw std::runtime_error("cannot set test environment");
#endif
    }
    ~Environment() { set(previous ? previous->c_str() : nullptr); }
};
struct TemporaryDescription {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("strix-qwen-description-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDescription() {
        if (!std::filesystem::create_directory(directory)) throw std::runtime_error("cannot create test directory");
    }
    void write(const std::string& source) const {
        std::ofstream output(directory / "model.toml", std::ios::binary);
        output << source;
        if (!output) throw std::runtime_error("cannot write test description");
    }
    ~TemporaryDescription() { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
};
std::string hook_refusal(const strix::ModelContract& baked, const strix::gguf::FileInfo& info) {
    try { qd::refuse_unless_described(baked, info); }
    catch (const std::runtime_error& error) {
        const std::string message = error.what();
        expect(message.starts_with("model description mismatch: "), "hook runtime_error prefix");
        return message;
    }
    return {};
}
void hook_tests(const std::string& source) {
    Environment check("STRIX_QWEN_MODEL_DESCRIPTION_CHECK"), description("STRIX_QWEN_MODEL_DESCRIPTION");
    const auto baked = strix::ModelContract::qwen38_flash_next();
    const auto matching = gguf({string("general.architecture", "qwen4exp"),
        integer("qwen4exp.embedding_length", 2560), integer("qwen4exp.block_count", 48)});
    const auto mismatching = gguf({integer("qwen4exp.embedding_length", 1280), integer("qwen4exp.block_count", 24)});
    expect(hook_refusal(baked, matching).empty(), "default hook accepts nonempty checked-key vector");
    expect(hook_refusal(baked, gguf({integer("unrelated.key", 1)})).empty(), "partial metadata does not refuse");
    const auto error = hook_refusal(baked, mismatching);
    expect(error == "model description mismatch: qwen4exp.embedding_length: expected 2560 (integer), actual 1280 (GGUF type 10); "
        "qwen4exp.block_count: expected 48 (integer), actual 24 (GGUF type 10)", "hook joins all GGUF mismatches");
    for (const char* enabled : {"1", "00"}) {
        check.set(enabled);
        expect(!hook_refusal(baked, mismatching).empty(), "only literal zero disables the hook");
    }
    check.set("0");
    expect(hook_refusal(baked, mismatching).empty(), "CHECK=0 skips mismatching GGUF");
    check.set(nullptr);
    auto runtime = baked;
    runtime.attention_kv_storage = strix::AttentionKvStorage::int8_token;
    expect(hook_refusal(runtime, matching).empty(), "GGUF check permits runtime KV storage override");
    auto different_payload = matching;
    different_payload.file_size += 4096;
    different_payload.tensors.push_back({"uninspected", {1}, 0, 0, 0, 4, 4});
    expect(hook_refusal(baked, different_payload).empty(), "same header/KVs pass without inspecting tensors or payload");

    TemporaryDescription fixture;
    const auto path = (fixture.directory / "model.toml").string();
    description.set(path.c_str());
    fixture.write(source);
    expect(hook_refusal(baked, matching).empty(), "hook loads matching TOML");
    expect(hook_refusal(runtime, matching).find("storage.attention_kv") != std::string::npos,
        "explicit TOML must describe supplied runtime KV storage");
    fixture.write(replace(source, "attention_kv = \"bf16\"", "attention_kv = \"int8_token\""));
    expect(hook_refusal(runtime, matching).empty(), "explicit TOML can describe runtime KV override");
    fixture.write(replace(source, "hidden = 2560", "hidden = 1280"));
    expect(hook_refusal(baked, matching).find("dims.hidden: expected 2560, actual 1280") != std::string::npos,
        "hook refuses mismatching TOML");
    check.set("0");
    expect(hook_refusal(baked, mismatching).empty(), "CHECK=0 skips both GGUF and TOML mismatches");
    fixture.write("schema = [");
    check.set(nullptr);
    expect(!hook_refusal(baked, matching).empty(), "hook wraps TOML parse failure");
    std::filesystem::remove(fixture.directory / "model.toml");
    expect(!hook_refusal(baked, matching).empty(), "hook wraps TOML load failure");
    check.set("0");
    expect(hook_refusal(baked, mismatching).empty(), "CHECK=0 avoids even loading missing TOML");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "expected model description fixture path\n"; return 2; }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open test fixture");
        const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        parser_tests(); contract_tests(source); gguf_tests(); hook_tests(source);
        expect(qd::contract_from_file(argv[1]).hidden_layers == 48, "load/contract_from_file round trip");
        bool io_refused = false;
        try { (void)md::load(std::string(argv[1]) + ".missing"); } catch (const std::runtime_error&) { io_refused = true; }
        expect(io_refused, "load refuses missing file");
    } catch (const std::exception& e) { ++failures; std::cerr << "unexpected exception: " << e.what() << '\n'; }
    std::cout << checks - failures << '/' << checks << " checks passed; " << failures << " failures\n";
    return failures ? 1 : 0;
}
