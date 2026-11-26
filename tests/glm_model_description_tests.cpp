// Host-only syntax, explicit GLM contract and deployment refusal tests. No HIP,
// model payload, peer, numerical execution or printed identity is synthesized.
#include "glm/load/glm_model_description.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace md = strix::model_description;
namespace gm = strix::glm::model;
using strix::serve::Json;
namespace {
int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); } } while (false)
bool has(const std::vector<std::string>& errors, const std::string& part) {
    for (const auto& e : errors) if (e.find(part) != std::string::npos) return true;
    return false;
}
template<class Exception = std::runtime_error>
bool refuses(const std::function<void()>& f, const std::string& part = {}) {
    try { f(); } catch (const Exception& e) { return std::string(e.what()).find(part) != std::string::npos; }
    return false;
}
// Dotted leaf views and mutations are test conveniences, not generic reader API.
void leaves(const md::Value::Table& table, md::Value::Table& out, const std::string& prefix = {}) {
    for (const auto& [key, value] : table) {
        if (prefix.empty() && md::is_launch_metadata(key)) continue;
        const auto field = prefix.empty() ? key : prefix + "." + key;
        if (field == "pair" || field == "moe.route") continue;  // bound by typed code from the generated constants
        if (const auto* child = std::get_if<md::Value::Table>(&value.data)) leaves(*child, out, field);
        else out.emplace(field, value);
    }
}
void set(md::ModelDescription& d, std::string_view field, std::optional<md::Value> value) {
    auto* owner = &d.fields;
    while (true) {
        const auto dot = field.find('.');
        if (dot == std::string_view::npos) {
            if (value) owner->insert_or_assign(std::string(field), *value);
            else owner->erase(std::string(field));
            return;
        }
        owner = &std::get<md::Value::Table>(owner->at(std::string(field.substr(0, dot))).data);
        field.remove_prefix(dot + 1);
    }
}
// Independent source-config fixture, including the complete layer declarations.
Json correct_config() {
    auto j = strix::serve::parse_json_unique(R"({
      "hidden_size":4096,"vocab_size":154880,"num_hidden_layers":45,"hc_mult":4,
      "q_lora_rank":1536,"kv_lora_rank":512,"num_attention_heads":64,
      "qk_nope_head_dim":256,"qk_rope_head_dim":0,"v_head_dim":256,
      "index_n_heads":32,"index_head_dim":128,"index_kpool":4,"index_topk":2048,
      "n_routed_experts":288,"moe_intermediate_size":2048,"intermediate_size":12288,
      "num_nextn_predict_layers":1,"num_experts_per_tok":8,"first_k_dense_replace":3,
      "routed_scaling_factor":2.5,"rms_norm_eps":1e-5,
      "linear_attn_config":{"num_heads":64,"head_dim":128,"short_conv_kernel_size":4,"gate_lower_bound":-5.0}
    })");
    auto types = Json::array(), mlps = Json::array(), kda = Json::array();
    for (int l = 0; l < 45; ++l) {
        types.push(l % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention");
        mlps.push(l < 3 ? "dense" : "sparse");
        if (l % 4 != 3) kda.push(l);
    }
    auto linear = *j.find("linear_attn_config");
    linear.set("kda_layers", kda);
    j.set("linear_attn_config", linear);
    j.set("layer_types", types); j.set("mlp_layer_types", mlps);
    return j;
}
void syntax() {
    const std::string schema = "schema = \"strix-model/v1\"\n";
    const auto d = md::parse(schema + "# comment\n[t]\nword=\"a#b\\n\\\"\\\\\" # tail\nlist=[1, true, \"x\", [2,],]\nnum=+1.25e2\nint=-4\n");
    CHECK(std::get<std::string>(d.find("t.word")->data) == "a#b\n\"\\");
    CHECK(std::get<md::Value::Array>(d.find("t.list")->data).size() == 4);
    CHECK(std::get<double>(d.find("t.num")->data) == 125.0);
    CHECK(std::get<std::int64_t>(d.find("t.int")->data) == -4);
    CHECK(d.find("absent") == nullptr);
    CHECK(d.find("t.int.child") == nullptr);
    CHECK(d.find("t") && std::holds_alternative<md::Value::Table>(d.find("t")->data));
    static_assert(noexcept(d.find("t")));
    const auto supported = md::parse("a = { word = 'literal\\text', nested = { n = 2 }, list = [\n"
        "# multiline array\n1,\ntrue,\n'x',\n[2,],\n] }\ntab = \"a\tb\"\n");
    CHECK(std::get<std::string>(supported.find("a.word")->data) == "literal\\text");
    CHECK(std::get<std::string>(supported.find("tab")->data) == "a\tb");
    CHECK(std::get<std::int64_t>(supported.find("a.nested.n")->data) == 2);
    CHECK(std::get<md::Value::Array>(supported.find("a.list")->data).size() == 4);
    const auto equivalent = md::parse("tab='a\tb'\n[a]\nword='literal\\text'\n"
        "list=[1,true,'x',[2]]\n[a.nested]\nn=2\n");
    CHECK(supported.fields == equivalent.fields);
    auto changed = equivalent;
    set(changed, "a.nested.n", md::Value{std::int64_t{3}});
    CHECK(supported.fields != changed.fields);
    for (const std::string s : {
        "a=1\na=2", "[t]\na=1\n[t]\nb=2", "a=1\n[a]\nb=2", "a.b=1\na=2",
        "[a.b]\nx=1\n[a]\nb=2", "[a]\nx=1\n[a.x]\ny=2", "a=[1 2]", "a=[1,,2]",
        "a=[1", "a=\"abc", "a=\"\\q\"", "a=1 garbage", "a=01", "a=1.0.0", "a=nan", "a=inf",
        "a=1e9999", "a=9223372036854775808", "a=-9223372036854775809", "a=trueish",
        "a=", "a", "a..b=1", "[a] garbage", "[[a]]", "a=0x10",
        "a=1_000", "a=1.", "a=.1", "a=1e", "a=--2", "a=\"x\"oops",
        "a={x=1,x=2}", "a={x=1,}", "a={x=1}\na.y=2", "a={x=1}\n[a]\ny=2",
        "a='unterminated", "a=[\n1\n2\n]", "a=\"\"\"unsupported\"\"\""})
        CHECK(refuses<std::invalid_argument>([&] { (void)md::parse(schema + s); }, "model description line"));
    for (const std::string s : {"", "schema=1", "schema=true", "schema=\"strix-model/v2\"", "schema=[\"strix-model/v1\"]"})
        CHECK(refuses([&] { gm::validate_description(md::parse(s)); }, "schema: expected"));
    CHECK(refuses<std::invalid_argument>([&] { (void)md::parse(schema + "schema=\"strix-model/v1\""); }, "schema"));
    CHECK(refuses<std::invalid_argument>([&] { (void)md::parse(schema + "a=\"bad" + char(1) + "byte\""); }, "control byte"));
    CHECK(refuses<std::invalid_argument>([&] { (void)md::parse(schema + "a=" + std::string(34, '[') + "1" + std::string(34, ']')); }, "nesting"));
    CHECK(std::get<std::int64_t>(md::parse(schema + "n=9223372036854775807").find("n")->data) == INT64_MAX);
    CHECK(std::get<std::int64_t>(md::parse(schema + "n=-9223372036854775808").find("n")->data) == INT64_MIN);
}
void description(const md::ModelDescription& good) {
    CHECK(gm::description_mismatches(good).empty());
    gm::validate_description(good);
    // Every declared contract field must be checked independently, with both
    // a wrong value and a wrong type, and all omissions must name the field.
    md::Value::Table fields;
    leaves(good.fields, fields);
    for (const auto& [field, value] : fields) {
        auto bad = good;
        set(bad, field, std::nullopt);
        auto errors = gm::description_mismatches(bad);
        CHECK(errors.size() == 1 && has(errors, field + ": expected") && has(errors, "actual <missing>"));
        if (std::holds_alternative<std::int64_t>(value.data)) set(bad, field, md::Value{std::int64_t{-1}});
        else if (std::holds_alternative<double>(value.data)) set(bad, field, md::Value{-123.5});
        else set(bad, field, md::Value{std::string("wrong")});
        errors = gm::description_mismatches(bad);
        CHECK(errors.size() == 1 && has(errors, field) && has(errors, "expected") && has(errors, "actual"));
        set(bad, field, md::Value{true});
        CHECK(has(gm::description_mismatches(bad), field + ": expected"));
        set(bad, field, md::Value{md::Value::Array{}});
        CHECK(has(gm::description_mismatches(bad), "actual array"));
        set(bad, field, md::Value{md::Value::Table{}});
        CHECK(has(gm::description_mismatches(bad), field + ": expected") &&
            has(gm::description_mismatches(bad), "actual table"));
    }
    auto bad = good;
    set(bad, "dims.hidden", md::Value{std::int64_t{8192}});
    set(bad, "moe.experts", md::Value{std::int64_t{256}});
    set(bad, "dims.typo", md::Value{std::int64_t{1}});
    const auto errors = gm::description_mismatches(bad);
    CHECK(errors.size() == 3);
    CHECK(has(errors, "dims.hidden: expected integer 4096, actual integer 8192"));
    CHECK(has(errors, "moe.experts: expected integer 288, actual integer 256"));
    CHECK(has(errors, "dims.typo: expected supported GLM field"));
    CHECK(refuses([&] { gm::validate_description(bad); }, "moe.experts"));
    bad = good; set(bad, "dims.hidden", md::Value{4096.0});
    CHECK(has(gm::description_mismatches(bad), "actual float 4096"));
    bad = good; set(bad, "arithmetic.clamp", md::Value{std::int64_t{10}});
    CHECK(has(gm::description_mismatches(bad), "arithmetic.clamp: expected float 10, actual integer 10"));
    for (const char* field : {"unknown", "dims.typo", "extras.mtp.typo"}) {
        bad = good; set(bad, field, md::Value{md::Value::Table{}});
        CHECK(has(gm::description_mismatches(bad), std::string(field) + ": expected nonempty supported GLM table, actual empty table"));
    }
    bad = good; set(bad, "dims", md::Value{md::Value::Table{}});
    CHECK(has(gm::description_mismatches(bad), "dims: expected nonempty supported GLM table"));
    CHECK(has(gm::description_mismatches(bad), "dims.hidden: expected integer 4096, actual <missing>"));
    bad = good; set(bad, "dims", md::Value{true});
    CHECK(has(gm::description_mismatches(bad), "dims: expected supported GLM field, actual boolean true"));
    CHECK(has(gm::description_mismatches(bad), "dims.hidden: expected integer 4096, actual <missing>"));
    const auto unknown = md::parse("unknown={nested={leaf=1}, empty={}}\n");
    CHECK(has(gm::description_mismatches(unknown), "unknown.nested.leaf: expected supported GLM field"));
    CHECK(has(gm::description_mismatches(unknown), "unknown.empty: expected nonempty supported GLM table"));
}
void config() {
    const auto good = correct_config();
    CHECK(gm::config_mismatches(good.dump()).empty());
    gm::validate_config(good.dump());
    auto nested = Json::object(); nested.set("text_config", good);
    CHECK(gm::config_mismatches(nested.dump()).empty());
    for (const auto& [field, value] : good.as_object()) {
        if (!value.is_number()) continue;
        for (const auto& wrong : {Json(-1), Json(true), Json("4096"), Json(), Json(1.25)}) {
            auto bad = strix::serve::parse_json_unique(good.dump());
            bad.set(field, wrong);
            const auto errors = gm::config_mismatches(bad.dump());
            CHECK(errors.size() == 1 && has(errors, field + ": expected") && has(errors, "actual"));
        }
    }
    // Required dimensional fields cannot disappear, including nested KDA fields.
    for (const auto& [field, value] : good.as_object()) {
        (void)value;
        if (field == "routed_scaling_factor" || field == "rms_norm_eps" || field == "num_experts_per_tok" || field == "first_k_dense_replace") continue;
        auto missing = Json::object();
        for (const auto& [f, v] : good.as_object()) if (f != field) missing.set(f, v);
        CHECK(has(gm::config_mismatches(missing.dump()), field + ": expected"));
    }
    auto absent_optional = Json::object();
    for (const auto& [f, v] : good.as_object())
        if (f != "routed_scaling_factor" && f != "rms_norm_eps" && f != "num_experts_per_tok" && f != "first_k_dense_replace") absent_optional.set(f, v);
    CHECK(gm::config_mismatches(absent_optional.dump()).empty());
    for (const auto& [field, value] : good.find("linear_attn_config")->as_object()) {
        (void)value;
        auto bad = strix::serve::parse_json_unique(good.dump());
        auto linear = strix::serve::parse_json_unique(bad.find("linear_attn_config")->dump());
        linear.set(field, true); bad.set("linear_attn_config", linear);
        CHECK(has(gm::config_mismatches(bad.dump()), "linear_attn_config." + field + ": expected"));
    }
    auto bad = strix::serve::parse_json_unique(good.dump());
    bad.set("hidden_size", 8192); bad.set("n_routed_experts", 256);
    CHECK(gm::config_mismatches(bad.dump()).size() == 2);
    CHECK(refuses([&] { gm::validate_config(bad.dump()); }, "hidden_size: expected integer 4096, actual 8192"));
    CHECK(refuses([&] { gm::validate_config(bad.dump()); }, "n_routed_experts: expected integer 288, actual 256"));
    for (const char* field : {"layer_types", "mlp_layer_types"}) {
        bad = strix::serve::parse_json_unique(good.dump());
        auto layers = strix::serve::parse_json_unique(bad.find(field)->dump());
        layers.as_array()[3] = "wrong"; layers.as_array()[44] = true;
        layers.push("extra"); bad.set(field, layers);
        const auto errors = gm::config_mismatches(bad.dump());
        CHECK(errors.size() == 3 && has(errors, std::string(field) + "[3]") && has(errors, std::string(field) + "[44]"));
    }
    CHECK(!gm::config_mismatches("[]").empty());
    CHECK(!gm::config_mismatches("{\"text_config\":null}").empty());
    for (const std::string s : {"{", "{} trailing", "{\"a\":1,\"a\":2}", "{\"a\":{\"b\":1,\"b\":2}}", "{\"hidden_size\":NaN}"})
        CHECK(refuses<strix::serve::JsonError>([&] { gm::validate_config(s); }));
    bad = strix::serve::parse_json_unique(good.dump()); bad.set("hidden_size", Json::number_text("4096.0"));
    CHECK(has(gm::config_mismatches(bad.dump()), "hidden_size"));
    bad.set("hidden_size", Json::number_text("999999999999999999999999999999"));
    CHECK(has(gm::config_mismatches(bad.dump()), "hidden_size"));
}
void write(const std::filesystem::path& p, const std::string& s) {
    std::ofstream file(p, std::ios::binary); file << s;
    if (!file) throw std::runtime_error("test fixture write failed");
}
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
void deployment(const std::string& toml) {
    const auto root = std::filesystem::current_path() /
        ("strix-glm-description-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(root));
    const auto config_path = root / "config.json", desc_path = root / "glm-5.3-flash.toml";
    struct Cleanup {
        std::filesystem::path a, b, dir;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(a, ec); std::filesystem::remove(b, ec); std::filesystem::remove(dir, ec); }
    } cleanup{config_path, desc_path, root};
    // Admission keeps a legacy directory (no config.json) loadable; the tool form still requires the file.
    gm::validate_model_directory(root.string()); CHECK(true);
    CHECK(refuses([&] { gm::validate_model_directory_complete(root.string()); }, "cannot open config.json"));
    // A partial config.json: present keys are checked, absent ones are not required at admission.
    {
        Json partial = Json::object();
        partial.set("hidden_size", 4096);
        write(config_path, partial.dump());
        gm::validate_model_directory(root.string()); CHECK(true);
        CHECK(refuses([&] { gm::validate_model_directory_complete(root.string()); }, "vocab_size"));
        partial.set("vocab_size", 123);
        write(config_path, partial.dump());
        CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "vocab_size: expected integer 154880, actual 123"));
        // STRIX_GLM_MODEL_DESCRIPTION_CHECK=0 restores the previous admission: no file is read or checked.
        set_env("STRIX_GLM_MODEL_DESCRIPTION_CHECK", "0");
        CHECK(!gm::description_check_enabled());
        gm::validate_model_directory(root.string()); CHECK(true);
        set_env("STRIX_GLM_MODEL_DESCRIPTION_CHECK", nullptr);
        CHECK(gm::description_check_enabled());
        std::filesystem::remove(config_path);
    }
    write(config_path, correct_config().dump());
    gm::validate_model_directory(root.string()); CHECK(true);  // optional description absent
    write(desc_path, toml);
    gm::validate_model_directory(root.string()); CHECK(true);  // exact description present
    CHECK(md::load(desc_path.string()).fields == md::parse(toml).fields);
    write(desc_path, "schema=\"unterminated");
    CHECK(refuses<std::invalid_argument>([&] { (void)md::load(desc_path.string()); }, "model description line"));
    CHECK(refuses<std::invalid_argument>([&] { gm::validate_model_directory(root.string()); }, "model description line"));
    write(desc_path, "schema=\"strix-model/v1\"\n");
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "dims.hidden"));
    write(desc_path, "schema=\"wrong\"\n");
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "schema"));
    write(desc_path, toml);
    auto bad = correct_config(); bad.set("vocab_size", 123);
    write(config_path, bad.dump());
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "vocab_size: expected integer 154880, actual 123"));
    write(desc_path, "schema=\"strix-model/v1\"\n");
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "dims.hidden: expected integer 4096, actual <missing>"));
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "vocab_size: expected integer 154880, actual 123"));
    std::filesystem::remove(desc_path);
    CHECK(refuses([&] { gm::validate_model_directory(root.string()); }, "vocab_size"));
    CHECK(refuses([&] { (void)md::load(desc_path.string()); }, "cannot open"));
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: glm-model-description-tests MODEL.toml");
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) throw std::runtime_error("cannot read test description");
        const std::string toml((std::istreambuf_iterator<char>(file)), {});
        syntax(); description(md::load(argv[1])); config(); deployment(toml);
    } catch (const std::exception& e) { ++failures; std::fprintf(stderr, "exception: %s\n", e.what()); }
    std::printf("glm model description: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
