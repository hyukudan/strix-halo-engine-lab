#include "strix/deepseek41_model_description.hpp"

// Reuse the existing independent tiny GGUF writer and released-config fixture.
// No second copy of model dimensions or Engram contract constants.
#define main deepseek41_existing_model_tests_main
#include "deepseek41_model_tests.cpp"
#undef main

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

namespace {
int checks = 0;
void check(bool ok, const std::string& message) { ++checks; expect(ok, message); }

template <typename E = std::invalid_argument, typename F> void refuses(F action, const std::vector<std::string>& needles) {
    ++checks;
    try {
        action();
        expect(false, "expected refusal");
    } catch (const E& error) {
        for (const auto& needle : needles)
            expect(std::string(error.what()).find(needle) != std::string::npos,
                   "missing diagnostic '" + needle + "': " + error.what());
    }
}
namespace md = strix::model_description;
void leaves(const md::Value::Table& table, const std::string& prefix,
            std::map<std::string, md::Value>& result) {
    for (const auto& [key, value] : table) {
        const auto path = prefix.empty() ? key : prefix + '.' + key;
        if (const auto* nested = std::get_if<md::Value::Table>(&value.data)) leaves(*nested, path, result);
        else result.emplace(path, value);
    }
}
std::map<std::string, md::Value> leaves(const md::ModelDescription& description) {
    std::map<std::string, md::Value> result;
    leaves(description.fields, "", result);
    return result;
}
md::Value::Table& parent(md::ModelDescription& description, std::string_view path) {
    auto* table = &description.fields;
    while (path.find('.') != std::string_view::npos) {
        const auto dot = path.find('.');
        table = &std::get<md::Value::Table>(table->at(std::string(path.substr(0, dot))).data);
        path.remove_prefix(dot + 1);
    }
    return *table;
}
md::Value& field(md::ModelDescription& description, const std::string& path) {
    return parent(description, path)[path.substr(path.rfind('.') + 1)];
}
void erase(md::ModelDescription& description, const std::string& path) {
    parent(description, path).erase(path.substr(path.rfind('.') + 1));
}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read test file");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::deepseek41;
    namespace md = strix::model_description;
    if (argc == 3 && (std::string(argv[1]) == "--write-fixture-json" || std::string(argv[1]) == "--write-fixture-gguf")) {
        std::ofstream out(argv[2], std::ios::binary);
        const auto bytes = std::string(argv[1]) == "--write-fixture-json" ? small_config() : small_target().bytes();
        if (!out || !out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) return 1;
        out.close();
        return out ? 0 : 1;
    }
    if (argc < 2 || argc > 3) throw std::invalid_argument("expected shipped TOML and optional exported fixture TOML");
    const auto released = parse_config_json(real_config_json);
    const auto generated = describe_model_toml(released);
    check(read_text(argv[1]).starts_with(generated), "shipped execution contract equals exporter byte for byte");
    const auto roundtrip = config_from_model_description(md::load(argv[1]), released);
    check(model_description_config_json(roundtrip) == model_description_config_json(released), "all released Config fields round trip");
    check(expected_target_tensors(roundtrip).size() == expected_target_tensors(released).size(), "released inventory count preserved");

    const auto original = parse_config_json(small_config());
    auto description = md::parse(describe_model_toml(original));
    if (argc == 3) description = md::load(argv[2]);
    const auto converted = config_from_model_description(description, original);
    auto fixture = small_target();
    fixture.set("deepseek41.config", model_description_config_json(converted));
    const auto validated = validate_target(fixture.info());
    check(model_description_config_json(validated.config) == model_description_config_json(original), "exported fixture -> Config -> validate_target PASS");
    auto reference_fixture = small_target();
    reference_fixture.set("deepseek41.config", model_description_config_json(original));
    check(validated.manifest.layout_identity == validate_target(reference_fixture.info()).manifest.layout_identity,
          "round trip preserves complete synthetic manifest identity with identical JSON encoding");

    // Each exported field is mandatory, including every dimension, type and array.
    for (const auto& [key, value] : leaves(description)) {
        (void)value;
        auto wrong = description;
        erase(wrong, key);
        refuses([&] { (void)config_from_model_description(wrong, original); }, {key, "expected", "actual <missing>"});
        wrong = description;
        if (std::holds_alternative<std::string>(value.data)) field(wrong, key).data = false;
        else field(wrong, key).data = std::string("wrong-type");
        refuses([&] { (void)config_from_model_description(wrong, original); }, {key, "expected", "actual"});
        wrong = description;
        auto& data = field(wrong, key).data;
        if (auto* n = std::get_if<std::int64_t>(&data)) ++*n;
        else if (auto* number = std::get_if<double>(&data)) *number += 1.0;
        else if (auto* b = std::get_if<bool>(&data)) *b = !*b;
        else if (auto* text = std::get_if<std::string>(&data)) *text += "-changed";
        else std::get<md::Value::Array>(data).push_back(md::Value{std::int64_t{999}});
        refuses([&] { (void)config_from_model_description(wrong, original); }, {key, "expected", "actual"});
    }
    auto wrong = description;
    field(wrong, "dims.hidden").data = std::int64_t{512};
    field(wrong, "dims.layers").data = std::int64_t{6};
    field(wrong, "dims.n_heads").data = std::string("four");
    field(wrong, "moe.norm_topk_prob").data = std::int64_t{1};
    field(wrong, "attention.rope_theta").data = std::int64_t{10000};
    erase(wrong, "dims.max_context");
    field(wrong, "unknown").data = true;
    field(wrong, "schema").data = std::string("strix-model/v2");
    refuses([&] { (void)config_from_model_description(wrong, original); }, {
        "dims.hidden: expected 256 (integer), actual 512 (integer)",
        "dims.layers: expected 4 (integer), actual 6 (integer)",
        "dims.n_heads: expected 4 (integer), actual \"four\" (string)",
        "moe.norm_topk_prob: expected true (boolean), actual 1 (integer)",
        "attention.rope_theta: expected 10000.0 (float), actual 10000 (integer)",
        "dims.max_context", "<missing>", "unknown: expected <absent>, actual true", "schema", "strix-model/v1", "strix-model/v2"});
    // This is also checked through actual text parsing, without mutable test objects.
    auto perturbed = describe_model_toml(original);
    perturbed.replace(perturbed.find("hidden = 256"), std::string("hidden = 256").size(), "hidden = 512");
    perturbed.replace(perturbed.find("layers = 4"), std::string("layers = 4").size(), "layers = \"four\"");
    refuses([&] { (void)config_from_model_description(md::parse(perturbed), original); }, {"dims.hidden", "dims.layers", "actual 512", "actual \"four\""});

    for (const std::string text : {
        "x = 1\nx = 2\n", "[dims]\nx = 1\n[dims]\ny = 2\n", "x = [1,,2]\n",
        "x = [1 2]\n", "x = [1,\n", "x = \"unfinished\n", "x = \"bad\\q\"\n", "[dims\nx = 1\n",
        "x = 01\n", "x = 1.\n", "x = .5\n", "x = 1e\n", "x = 1e999\n", "x = nan\n", "x = inf\n",
        "x = 9223372036854775808\n", "x = true trailing\n", "x = {a = 1, a = 2}\n", "[[dims]]\nx = 1\n",
        "x = 1\n[x]\ny = 2\n", "[x.y]\na = 1\n[x]\ny = 2\n", "x.y = 1\nx = 2\n",
        "x = {a = 1,}\n", "x = 0x10\n", "x = truefalse\n", "x = 1\ry = 2\n"})
        refuses([&] { (void)md::parse(text); }, {"line"});
    const auto subset = md::parse("# comment\n[a.b]\nx = [1, # array comment\n2,]\ny = \"quote\\\" and # text\"\nz = -1.5e+2\nb = false\n");
    check(subset.find("a.b.x") && *subset.find("a.b.x") == md::Value{md::Value::Array{md::Value{std::int64_t{1}}, md::Value{std::int64_t{2}}}}, "dotted table and multiline array subset");
    check(subset.find("a.b.z") && *subset.find("a.b.z") == md::Value{-150.0}, "finite decimal float subset");
    refuses<std::runtime_error>([] { (void)md::load("missing-deepseek41-description-test.toml"); }, {"cannot open"});

    // Existing semantic refusals stay authoritative for programmatically supplied Configs.
    auto invalid = original;
    invalid.num_key_value_heads = 2;
    refuses([&] { (void)describe_model_toml(invalid); }, {"attention geometry"});
    invalid = original;
    invalid.kv_source_layer_ids.push_back(999);
    refuses([&] { (void)describe_model_toml(invalid); }, {"kv source layer out of range"});
    invalid = original;
    invalid.rope_theta = std::numeric_limits<double>::infinity();
    refuses([&] { (void)describe_model_toml(invalid); }, {"nonfinite"});
    expect(checks == 231, "original adapted suite keeps 231 checks");
    std::cout << "deepseek41 reconciled baseline: " << checks << " checks, " << failures << " failures\n";
    // Supported canonical forms stay parser-positive; DS shape/type policy is separate.
    const auto supported = md::parse("x = {a = 'literal', b = [1, \"two\", 2.0, true], c = {}}\n");
    check(supported.find("x.a") && *supported.find("x.a") == md::Value{std::string("literal")}, "literal and inline table supported");
    check(supported.find("x.b") && std::get<md::Value::Array>(supported.find("x.b")->data).size() == 4, "mixed typed array supported by canonical reader");
    check(supported.find("x.c") && std::get<md::Value::Table>(supported.find("x.c")->data).empty(), "empty inline table preserved");
    check(supported.find("x.a.child") == nullptr && supported.find("x.absent") == nullptr, "find refuses missing and non-table paths");
    auto equivalent = describe_model_toml(original);
    const auto section = equivalent.find("[attention.rope_scaling]\n");
    const auto section_end = equivalent.find("\n\n", section);
    auto inline_body = equivalent.substr(section + std::string("[attention.rope_scaling]\n").size(),
        section_end - section - std::string("[attention.rope_scaling]\n").size());
    for (std::size_t at = 0; (at = inline_body.find('\n', at)) != std::string::npos; at += 2)
        inline_body.replace(at, 1, ", ");
    equivalent.replace(section, section_end - section, "rope_scaling = {" + inline_body + '}');
    const auto schema_begin = equivalent.find("schema = \"strix-model/v1\"");
    equivalent.replace(schema_begin, std::string("schema = \"strix-model/v1\"").size(), "schema = 'strix-model/v1'");
    const auto array_begin = equivalent.find("compress_ratios = [") + std::string("compress_ratios = [").size();
    equivalent.insert(array_begin, "\n# supported multiline array\n");
    check(model_description_config_json(config_from_model_description(md::parse(equivalent), original)) ==
        model_description_config_json(original), "nested inline/literal/multiline forms preserve all Config fields");
    for (const std::string suffix : {"\n[unknown]\n", "\n[attention.unknown]\n", "\n[attention.unknown.nested]\n"}) {
        refuses([&] { (void)config_from_model_description(md::parse(describe_model_toml(original) + suffix), original); },
            {suffix.find("nested") == std::string::npos ? "unknown: expected <absent>, actual {}" : "attention.unknown.nested: expected <absent>, actual {}"});
    }
    wrong = description;
    field(wrong, "dims.hidden").data = md::Value::Table{};
    refuses([&] { (void)config_from_model_description(wrong, original); }, {"dims.hidden", "actual {} (table)"});
    wrong = description;
    field(wrong, "attention.rope_scaling").data = md::Value::Table{};
    refuses([&] { (void)config_from_model_description(wrong, original); },
        {"attention.rope_scaling.factor", "attention.rope_scaling.type", "actual <missing>"});
    wrong = description;
    field(wrong, "dims.hidden").data = md::Value::Array{md::Value{std::int64_t{1}}, md::Value{std::string("two")}};
    refuses([&] { (void)config_from_model_description(wrong, original); }, {"dims.hidden", "actual [1, \"two\"] (array)"});
    wrong = description;
    field(wrong, "attention").data = true;
    refuses([&] { (void)config_from_model_description(wrong, original); },
        {"attention: expected <table> (table), actual true (boolean)", "attention.rope_theta", "attention.rope_scaling.factor"});
    for (const std::string text : {"x = 'unterminated\n", "x = {a.b = 1, a.b = 2}\n", "x = {a = 1}\nx.b = 2\n",
        "x = {a = 1}\n[x]\nb = 2\n", "x = [1, {a = 2}\n"})
        refuses([&] { (void)md::parse(text); }, {"line", "column"});

    wrong = description;
    field(wrong, "dims.hidden") = *md::parse("x = \"\\b\\f\"\n").find("x");
    field(wrong, "dims.layers").data = std::int64_t{6};
    refuses([&] { (void)config_from_model_description(wrong, original); },
        {"dims.hidden: expected 256 (integer), actual \"\\b\\f\" (string)",
         "dims.layers: expected 4 (integer), actual 6 (integer)"});
    std::cout << "deepseek41 model description: " << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return 1;
}
