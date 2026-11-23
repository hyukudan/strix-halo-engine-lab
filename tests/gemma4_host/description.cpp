#include "test.hpp"
#include "gemma4/model/description.hpp"

#include <fstream>
#include <filesystem>
#include <iterator>
#include <type_traits>

namespace md = strix::model_description;
namespace g4 = strix::gemma4;
namespace {
unsigned cases = 0;
template<class Exception = std::runtime_error, class F> void rejected(F action, const std::string& field) {
    bool refused = false;
    try { action(); }
    catch (const Exception& e) {
        const std::string message = e.what();
        CHECK(message.find(field) != message.npos);
        if constexpr (std::is_same_v<Exception, std::invalid_argument>) {
            CHECK(message.starts_with("model description line "));
            CHECK(message.find(" column ") != message.npos);
        }
        refused = true;
    }
    CHECK(refused);
    ++cases;
}
void validate(const md::ModelDescription& d) { (void)g4::validate_model_description(d); }
using Fields = md::Value::Table;
void flatten(const Fields& table, const std::string& prefix, Fields& result) {
    for (const auto& [key, value] : table) {
        if (const auto* nested = std::get_if<Fields>(&value.data)) flatten(*nested, prefix + key + ".", result);
        else result.emplace(prefix + key, value);
    }
}
// Mutations belong to this test; the shared reader exposes only const find().
Fields& parent(md::ModelDescription& d, std::string_view path) {
    auto* table = &d.fields;
    for (auto dot = path.find('.'); dot != path.npos; dot = path.find('.')) {
        table = &std::get<Fields>(table->at(std::string(path.substr(0, dot))).data);
        path.remove_prefix(dot + 1);
    }
    return *table;
}
std::string leaf(std::string_view path) { return std::string(path.substr(path.rfind('.') + 1)); }
md::Value& at(md::ModelDescription& d, const std::string& path) { return parent(d, path).at(leaf(path)); }
void erase(md::ModelDescription& d, const std::string& path) { parent(d, path).erase(leaf(path)); }
}
int main() {
    const auto perturbed_path = std::filesystem::current_path() / "g4-description-negative.toml";
    bool owns_perturbed = false;
    try {
        const auto exported = g4::export_model_description();
        const auto description = md::parse(exported);
        Fields fields;
        flatten(description.fields, "", fields);
        CHECK(fields.size() == 52);
        const auto graph = g4::validate_model_description(description);
        CHECK(graph.config.hidden == 2560 && graph.config.layers == 42 && graph.config.inter == 10240);
        CHECK(graph.config.vocab == 262144 && graph.config.heads == 8 && graph.config.kv_heads == 2);
        CHECK(graph.config.ple_dim == 256 && graph.config.window == 512 && graph.config.max_positions == 65536);
        CHECK(graph.config.eps == g4::norm_eps && graph.attention_chunk_rows == 64);
        CHECK(graph.embedding_multiplier == g4::embedding_scale && graph.ple_embedding_multiplier == g4::ple_embedding_scale);
        CHECK(graph.ple_model_multiplier == g4::model_ple_scale && graph.ple_combine_multiplier == g4::ple_combine_scale);
        CHECK(graph.taps == std::vector<std::uint32_t>({5, 11, 17, 23, 35}));
        CHECK(graph.config.layer.size() == 42);
        for (unsigned l = 0; l < 42; ++l) {
            const bool global = l % 6 == 5;
            CHECK(graph.config.layer[l].attention == (global ? g4::AttentionKind::global : g4::AttentionKind::sliding));
            CHECK(graph.config.layer[l].head_dim == (global ? 512U : 256U));
            CHECK(graph.config.layer[l].producer == (l < 24 ? l : (global ? 23U : 22U)));
        }
        ++cases;

        // Exercise actual file -> reader -> adapter refusal, in addition to the
        // exhaustive typed-field mutations below.
        CHECK(!std::filesystem::exists(perturbed_path));
        for (const auto& [from, to] : std::initializer_list<std::pair<std::string, std::string>>{
                {"hidden = 2560", "hidden = 2561"}, {"layers = 42", "layers = \"42\""},
                {"head_dim = 256\n", ""}, {"strix-model/v1", "strix-model/v2"},
                {"hidden = 2560", "hidden = 2560\nhidden = 2560"}, {"window = 512", "window = [512"}}) {
            auto text = exported;
            const auto pos = text.find(from); CHECK(pos != text.npos);
            text.replace(pos, from.size(), to);
            { std::ofstream out(perturbed_path, std::ios::binary); CHECK(bool(out)); owns_perturbed = true; out << text; CHECK(bool(out)); }
            if (from == "hidden = 2560" && to.find('\n') != to.npos)
                rejected<std::invalid_argument>([&] { (void)md::load(perturbed_path.string()); }, "duplicate key");
            else if (from == "window = 512")
                rejected<std::invalid_argument>([&] { (void)md::load(perturbed_path.string()); }, "");
            else
                rejected([&] { validate(md::load(perturbed_path.string())); }, "");
        }
        std::filesystem::remove(perturbed_path); owns_perturbed = false;
        std::ifstream file(G4_DESCRIPTION_FILE, std::ios::binary);
        CHECK(bool(file));
        const std::string checked_in{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        CHECK(checked_in.starts_with(exported));
        CHECK(checked_in.find('\r') == checked_in.npos);
        validate(md::load(G4_DESCRIPTION_FILE));
        ++cases;

        // Independently perturb, omit and change the type of EVERY contract field.
        for (const auto& [field, value] : fields) {
            auto changed = description;
            erase(changed, field);
            rejected([&] { validate(changed); }, field + ": expected");
            auto wrong = value;
            if (auto* integer = std::get_if<std::int64_t>(&wrong.data)) ++*integer;
            else if (auto* real = std::get_if<double>(&wrong.data)) *real += 1.0;
            else if (auto* boolean = std::get_if<bool>(&wrong.data)) *boolean = !*boolean;
            else if (auto* text = std::get_if<std::string>(&wrong.data)) *text += "-wrong";
            else std::get<md::Value::Array>(wrong.data).pop_back();
            changed = description; at(changed, field) = wrong;
            rejected([&] { validate(changed); }, field + ": expected");
            at(changed, field) = std::holds_alternative<bool>(value.data) ? md::Value{std::string("true")} : md::Value{false};
            rejected([&] { validate(changed); }, "actual");
        }
        auto changed = description;
        at(changed, "dims.hidden") = md::Value{std::int64_t(1)};
        at(changed, "dims.layers") = md::Value{std::int64_t(2)};
        erase(changed, "dims.vocab");
        parent(changed, "dims.unrecognised").emplace("unrecognised", md::Value{true});
        try { validate(changed); CHECK(false); }
        catch (const std::runtime_error& e) {
            const std::string s = e.what();
            CHECK(s.find("dims.hidden: expected integer 2560, actual integer 1") != s.npos);
            CHECK(s.find("dims.layers: expected integer 42, actual integer 2") != s.npos);
            CHECK(s.find("dims.vocab: expected integer 262144, actual <missing>") != s.npos);
            CHECK(s.find("dims.unrecognised: expected <absent>, actual boolean true") != s.npos);
        }
        ++cases;
        for (const auto* field : {"attention.layer_types", "attention.head_dims", "attention.kv_producers", "extras.tap_layers"}) {
            changed = description;
            auto& array = std::get<md::Value::Array>(at(changed, field).data);
            array.front() = md::Value{std::int64_t(-1)};
            rejected([&] { validate(changed); }, field);
        }
        rejected([&] { validate(md::parse(exported + "\n[unknown]\n")); }, "unknown");
        rejected([&] { validate(md::parse(exported + "\n[unknown.nested]\nx = 1\n")); }, "unknown.nested.x");
        rejected([&] { validate(md::parse(exported + "\n[attention.unknown]\n")); }, "attention.unknown");
        changed = description;
        at(changed, "dims") = md::Value{Fields{}};
        rejected([&] { validate(changed); }, "dims: expected nonempty table, actual table {}");
        at(changed, "dims") = md::Value{false};
        rejected([&] { validate(changed); }, "dims.hidden: expected integer 2560, actual <missing>");
        changed = description;
        at(changed, "dims.hidden") = md::Value{Fields{}};
        rejected([&] { validate(changed); }, "dims.hidden: expected integer 2560, actual table {}");
        at(changed, "dims.hidden") = md::Value{Fields{{"extra", md::Value{true}}}};
        rejected([&] { validate(changed); }, "dims.hidden.extra: expected <absent>, actual boolean true");
        rejected([] { (void)md::load("missing-description-file.toml"); }, "cannot open");
        for (const auto* text : {"", "schema = 1", "schema = true", "schema = \"strix-model/v2\""})
            rejected([&] { validate(md::parse(text)); }, "schema: expected");
        const std::string schema = "schema = \"strix-model/v1\"\n";
        for (const auto* bad : {
            "x = 1\nx = 2", "schema = \"strix-model/v1\"", "[a]\nx = 1\n[a]\nx = 2",
            "a = 1\n[a]\nx = 2", "a = 1\n[a.b]\nx = 2", "a.b = 1\na = 2", "a = 1\na.b = 2",
            "[a]\nx = 1\nx.y = 2", "a.b = 1\n[a]\nc = 2", "[a]\n[a.b]\n[a]\n",
            "x = 1 2", "x = 01", "x = 1.", "x = .1", "x = 1e", "x = 1e+", "x = --1", "x = +",
            "x = 1_000", "x = 0x10", "x = 9223372036854775808", "x = -9223372036854775809",
            "x = 1e999", "x = 1e-999", "x = nan", "x = inf", "x = False", "x = null",
            "x = [1,,2]", "x = [1 2]", "x = [1", "x = [1]]",
            "x = { a = 1, }", "x = { a = 1, a = 2 }", "x = { a = 1 }\nx.b = 2",
            "x = { a = 1 }\n[x]\nb = 2", "x = 'unfinished", "x = [\n1,\n",
            "x = \"bad\\q\"", "x = \"unfinished", "x = \"a\"junk",
            "x = \"\x01\"", "x =", "= 1", "a..b = 1", "a. = 1", "[[a]]", "[a]junk", "[a", "x 1",
            "\"key\" = 1", "x = \"\"\"multiline\"\"\"", "x = '''multiline'''"})
            rejected<std::invalid_argument>([&] { (void)md::parse(schema + bad); }, "model description line ");
        const auto positive = md::parse(schema + "# comment\nx = \"a#b\\\"c\\\\d\\n\\r\\t\" # comment\n"
            "y = [+1, -2, 0,]\nz = [1.0, 2e0]\nb = [true, false]\ne = []\n[t]\na.b = 2\n");
        CHECK(std::get<std::string>(positive.find("x")->data) == "a#b\"c\\d\n\r\t");
        CHECK(std::get<std::int64_t>(positive.find("t.a.b")->data) == 2);
        CHECK(std::get<md::Value::Array>(positive.find("y")->data).size() == 3);
        ++cases;
        const auto nested = md::parse("x = { a = 1, b = 'literal\\path', c = { ok = true } }\n"
            "mixed = [1, true, [2], { k = 'v' }]\ny = [\n# comment\n1,\n2,\n]\n"
            "[a.b]\n[a]\nb.c = 2\n");
        CHECK(std::get<std::int64_t>(nested.find("x.a")->data) == 1);
        CHECK(std::get<std::string>(nested.find("x.b")->data) == "literal\\path");
        CHECK(std::get<bool>(nested.find("x.c.ok")->data));
        CHECK(std::get<md::Value::Array>(nested.find("mixed")->data).size() == 4);
        CHECK(std::get<md::Value::Array>(nested.find("y")->data).size() == 2);
        CHECK(std::get<std::int64_t>(nested.find("a.b.c")->data) == 2);
        CHECK(nested.find("x.a.missing") == nullptr && nested.find("missing") == nullptr);
        CHECK(nested.fields == md::parse("x = {a=1,b='literal\\path',c={ok=true}}\n"
            "mixed=[1,true,[2],{k='v'}]\ny=[1,2]\na.b.c=2").fields);
        ++cases;
        // Supported TOML spelling variants must survive the strict adapter too.
        auto inline_dims = exported;
        const auto dims_begin = inline_dims.find("\n[dims]\n");
        const auto dims_end = inline_dims.find("\n[extras]", dims_begin);
        CHECK(dims_begin != inline_dims.npos && dims_end != inline_dims.npos);
        auto entries = inline_dims.substr(dims_begin + 8, dims_end - dims_begin - 8);
        for (auto& c : entries) if (c == '\n') c = ',';
        if (!entries.empty() && entries.back() == ',') entries.pop_back();
        inline_dims.erase(dims_begin, dims_end - dims_begin);
        inline_dims.insert(inline_dims.find("\n["), "\ndims = {" + entries + "}\n");
        validate(md::parse(inline_dims));
        ++cases;
        auto literal = exported;
        const std::string name = "\"nemotron-3.5-bolt-e4b\"";
        literal.replace(literal.find(name), name.size(), "'nemotron-3.5-bolt-e4b'");
        const auto taps = literal.find("tap_layers = [");
        CHECK(taps != literal.npos);
        literal.insert(taps + 14, "\n# multiline array\n");
        validate(md::parse(literal));
        ++cases;
        std::cout << "description: " << cases << " cases passed; " << fields.size()
                  << " strict fields; exported file/typed schedule/negative cases\n";
        return 0;
    } catch (const std::exception& e) {
        if (owns_perturbed) std::filesystem::remove(perturbed_path);
        std::cerr << e.what() << '\n'; return 1;
    }
}
