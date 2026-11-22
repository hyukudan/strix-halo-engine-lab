#include "strix/minimax/model_description.hpp"
#include "strix/minimax/model_assets.hpp"
#include "strix/serve/json.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace md = strix::model_description;
namespace mm = strix::minimax::description;
namespace fs = std::filesystem;
using J = strix::serve::Json;
unsigned passed = 0;
void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
    ++passed;
}
template<class Exception = std::runtime_error>
void refuses(const std::function<void()>& action, const std::string& field) {
    try { action(); }
    catch (const Exception& e) {
        require(std::string(e.what()).find(field) != std::string::npos, "missing diagnostic field " + field + ": " + e.what());
        return;
    }
    throw std::runtime_error("accepted invalid " + field);
}
J tensor(const std::string& dtype, const std::vector<std::uint64_t>& shape, std::uint64_t width) {
    J dims = J::array();
    std::uint64_t bytes = width;
    for (auto n : shape) { dims.push(J(n)); bytes *= n; }
    return J(J::Object{{"dtype", J(dtype)}, {"shape", dims}, {"data_offsets", J(J::Array{J(0), J(bytes)})}});
}
void accepted(const std::string& name, const std::vector<std::uint64_t>& shape, const std::string& dtype, std::uint64_t width) {
    const auto value = tensor(dtype, shape, width);
    const auto parsed = mm::header(J(J::Object{{name, value}}).dump());
    require(parsed.size() == 1 && parsed[0].name == name && parsed[0].shape == shape && parsed[0].dtype == dtype,
            "valid header " + name);
    auto wrong = shape;
    ++wrong[0];
    refuses([&] { mm::header(J(J::Object{{name, tensor(dtype, wrong, width)}}).dump()); }, name + ".shape");
    auto bad_type = value;
    bad_type.set("dtype", "BOOL");
    refuses([&] { mm::header(J(J::Object{{name, bad_type}}).dump()); }, name + ".dtype");
}
void linear(const std::string& role, std::uint64_t in, std::uint64_t out) {
    accepted(role + ".suh", {in}, "F16", 2);
    accepted(role + ".svh", {out}, "F16", 2);
    accepted(role + ".trellis", {in / 16, out / 16, 64}, "I16", 2);
    const auto name = role + ".trellis";
    for (const auto& dims : std::vector<std::vector<std::uint64_t>>{{in / 16, out / 16 + 1, 64}, {in / 16, out / 16, 63}, {in / 16, out / 16}})
        refuses([&] { mm::header(J(J::Object{{name, tensor("I16", dims, 2)}}).dump()); }, name + ".shape");
}
void parser_tests() {
    const auto m = md::parse("schema=\"strix-model/v1\"\n# comment\n[a]\ntext=\"a#b\\n\\\"\" # tail\ni=+12\nf=-1.25e+2\nb=true\narr=[1,\"two\",[false],]\n");
    require(std::get<std::int64_t>(m.find("a.i")->data) == 12, "integer parse");
    require(std::get<double>(m.find("a.f")->data) == -125, "float parse");
    require(std::get<std::string>(m.find("a.text")->data) == "a#b\n\"", "string parse");
    require(std::get<md::Value::Array>(m.find("a.arr")->data).size() == 3, "array parse");
    require(m.find("missing") == nullptr, "missing lookup");
    const auto tables = md::parse("a={b=1, c='literal'}\narr=[\n1, # comment\n2,\n]\n");
    require(std::get<std::int64_t>(tables.find("a.b")->data) == 1, "inline table");
    require(std::get<std::string>(tables.find("a.c")->data) == "literal", "literal string");
    require(std::get<md::Value::Array>(tables.find("arr")->data).size() == 2, "multiline array");
    require(md::parse("").fields.empty(), "generic reader has no model semantics");
    for (const char* bad : {"", "schema=1", "schema=\"strix-model/v2\"", "schema=true"})
        refuses([&] { mm::validate(md::parse(bad)); }, "schema");
    for (const char* bad : {"x=1\nx=2", "[x]\na=1\n[x]\nb=2", "x=1\n[x]\na=2", "x.a=1\nx=2", "[x.a]\nb=1\n[x]\na=2",
                            "x=", "x=01", "x=1.", "x=.1", "x=1e", "x=1e+", "x=nan", "x=inf", "x=1e999", "x=9223372036854775808",
                            "x=0x10", "x=1_000", "x=tru", "x=\"bad\\q\"", "x=\"end", "x=[1", "x=[,1]", "x=[1 2]", "x=1 junk", "x={a=1,a=2}", "x={a=1,}",
                            "[[x]]", "[x] junk", "[x", "x..a=1", ".x=1", "x.=1", "\"x\"=1", "x=\"a\"\nx.a=2"})
        refuses<std::invalid_argument>([&] { (void)md::parse(std::string("schema=\"strix-model/v1\"\n") + bad); }, "model description");
}
std::map<std::string, md::Value> flatten(const md::ModelDescription& model) {
    std::map<std::string, md::Value> result;
    const auto visit = [&](const auto& self, const md::Value::Table& table, const std::string& prefix) -> void {
        for (const auto& [key, value] : table) {
            if (prefix.empty() && md::is_launch_metadata(key)) continue;
            const auto path = prefix + key;
            if (path == "moe.route") continue;  // bound by typed code from the generated constants, not by this contract
            if (const auto* child = std::get_if<md::Value::Table>(&value.data)) self(self, *child, path + '.');
            else result.emplace(path, value);
        }
    };
    visit(visit, model.fields, "");
    return result;
}
md::Value::Table& parent(md::ModelDescription& model, std::string& field) {
    auto* table = &model.fields;
    for (;;) {
        const auto dot = field.find('.');
        if (dot == field.npos) return *table;
        const auto key = field.substr(0, dot);
        auto [entry, added] = table->try_emplace(key, md::Value{md::Value::Table{}});
        (void)added;
        table = &std::get<md::Value::Table>(entry->second.data);
        field.erase(0, dot + 1);
    }
}
md::Value& field(md::ModelDescription& model, std::string name) { auto& table = parent(model, name); return table[name]; }
void erase(md::ModelDescription& model, std::string name) { auto& table = parent(model, name); table.erase(name); }
void contract_tests(const std::string& path) {
    const auto model = mm::load(path);
    require(!model.fields.empty(), "load shipped contract");
    for (const auto& [field, value] : flatten(model)) {
        auto bad = model;
        erase(bad, field);
        refuses([&] { mm::validate(bad); }, field);
        bad = model;
        ::field(bad, field) = {std::string("wrong type/value")};
        refuses([&] { mm::validate(bad); }, field);
        if (const auto* n = std::get_if<std::int64_t>(&value.data)) {
            ::field(bad, field) = {*n + 1};
            refuses([&] { mm::validate(bad); }, "expected integer");
            ::field(bad, field) = {static_cast<double>(*n)};
            refuses([&] { mm::validate(bad); }, field);
        }
        if (const auto* n = std::get_if<double>(&value.data)) {
            ::field(bad, field) = {*n + 1.0};
            refuses([&] { mm::validate(bad); }, field);
            ::field(bad, field) = {std::int64_t{7}};
            refuses([&] { mm::validate(bad); }, field);
        }
    }
    auto bad = model;
    field(bad, "dims.hidden") = {std::int64_t{1}};
    field(bad, "moe.experts") = {std::int64_t{2}};
    try { mm::validate(bad); throw std::logic_error("accepted two wrong dimensions"); }
    catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        require(msg.find("dims.hidden: expected integer 6144, actual 1") != msg.npos &&
                msg.find("moe.experts: expected integer 128, actual 2") != msg.npos, "aggregate contract diagnostics");
    }
    bad = model;
    field(bad, "identities.exact") = {std::string("untrusted")};
    refuses([&] { mm::validate(bad); }, "identities.exact");
    bad = model;
    field(bad, "unknown") = {md::Value::Table{}};
    refuses([&] { mm::validate(bad); }, "unknown");
    bad = model;
    field(bad, "dims.hidden") = {md::Value::Table{}};
    refuses([&] { mm::validate(bad); }, "dims.hidden");
    refuses([&] { (void)md::load(path + ".missing"); }, "cannot open");
    refuses([&] { mm::load(path + ".missing"); }, "cannot open");
}
void header_tests() {
    const std::string p = "language_model.model.";
    accepted(p + "embed_tokens.weight", {200064, 6144}, "BF16", 2);
    accepted(p + "norm.weight", {6144}, "F32", 4);
    linear("language_model.lm_head", 6144, 200064);
    for (const unsigned l : {0U, 2U, 3U, 59U}) {
        const auto layer = p + "layers." + std::to_string(l) + '.';
        for (const auto* role : {"input_layernorm.weight", "post_attention_layernorm.weight"})
            accepted(layer + role, {6144}, "F32", 4);
        const auto a = layer + "self_attn.";
        for (const auto* role : {"q_norm.weight", "k_norm.weight", "index_q_norm.weight", "index_k_norm.weight"})
            accepted(a + role, {128}, "F16", 2);
        accepted(a + "index_q_proj.weight", {512, 6144}, "F16", 2);
        accepted(a + "index_k_proj.weight", {128, 6144}, "F16", 2);
        linear(a + "q_proj", 6144, 8192); linear(a + "k_proj", 6144, 512);
        linear(a + "v_proj", 6144, 512); linear(a + "o_proj", 8192, 6144);
        if (l < 3) {
            linear(layer + "mlp.gate_proj", 6144, 12288); linear(layer + "mlp.up_proj", 6144, 12288);
            linear(layer + "mlp.down_proj", 12288, 6144);
        } else {
            const auto b = layer + "block_sparse_moe.";
            accepted(b + "gate.weight", {128, 6144}, "F32", 4);
            accepted(b + "e_score_correction_bias", {128}, "F32", 4);
            for (const char* e : {"0", "127"}) {
                const auto pre = b + "experts." + e + '.';
                linear(pre + "w1", 6144, 3072); linear(pre + "w3", 6144, 3072); linear(pre + "w2", 3072, 6144);
            }
            linear(b + "shared_experts.gate_proj", 6144, 3072); linear(b + "shared_experts.up_proj", 6144, 3072);
            linear(b + "shared_experts.down_proj", 3072, 6144);
        }
    }
    const auto name = p + "layers.0.input_layernorm.weight";
    for (const auto* text : {"[]", "{", "{\"a\":{},\"a\":{}}", "{\"a\":{\"dtype\":1,\"shape\":[],\"data_offsets\":[0,0]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,0]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,4]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[true],\"data_offsets\":[0,4]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,4]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}",
                             "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,3]}}"})
        refuses([&] { mm::header(text); }, "");
    auto root = J(J::Object{{name, tensor("F32", {1}, 4)}, {p + "norm.weight", tensor("F32", {2}, 4)}});
    try { mm::header(root.dump()); throw std::logic_error("accepted wrong header"); }
    catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        require(msg.find(name + ".shape: expected [6144], actual [1]") != msg.npos &&
                msg.find(p + "norm.weight.shape: expected [6144], actual [2]") != msg.npos, "aggregate header diagnostics");
    }
    for (const auto* role : {"layers.60.input_layernorm.weight", "layers.3.block_sparse_moe.experts.128.w1.suh",
                             "layers.0.block_sparse_moe.gate.weight", "layers.3.mlp.gate_proj.suh"})
        refuses([&] { mm::header(J(J::Object{{p + role, tensor("F16", {6144}, 2)}}).dump()); }, p + role);
    const auto partial = mm::header("{\"draft-only\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}");
    require(partial.size() == 1, "partial draft pack accepted");
    // semantic=false skips only the role/dimension assertions; structural checks stay.
    const auto wrong_role = J(J::Object{{p + "layers.60.input_layernorm.weight", tensor("F16", {6144}, 2)}}).dump();
    require(mm::header(wrong_role, false).size() == 1, "OFF accepts a role with other dimensions");
    refuses([&] { mm::header(R"({"first":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"second":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}})", false); }, "second.data_offsets");
}
void pack_header_tests() {
    const std::string one = R"({"tensor":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})";
    refuses([&] { mm::header_payload_bytes(20, 12); }, "header length");
    refuses([&] { mm::header_payload_bytes(0, 7); }, "header length");
    refuses([&] { mm::header_payload_bytes(std::numeric_limits<std::uint64_t>::max(), 8); }, "header length");
    refuses([&] { mm::header(R"({"tensor":{"dtype":"F16","shape":[2],"data_offsets":[0,4])"); }, "header");
    refuses([&] { mm::header(R"({"first":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"second":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}})"); }, "second.data_offsets");
    mm::PackHeaders bounds;
    refuses([&] { bounds.add(one, mm::header_payload_bytes(one.size(), 8 + one.size() + 3)); }, "tensor.data_offsets");
    require(bounds.add(one, 4).size() == 1, "refused shard does not poison catalog");
    mm::PackHeaders shards;
    require(shards.add(one, 4).size() == 1, "first shard accepted");
    refuses([&] { shards.add(one, 4); }, "tensor");
    // Adjacent spans and empty spans are legal. JSON/name order is deliberately
    // different from offset order; header validation must not reorder entries.
    const auto adjacent = mm::header(R"({"z":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"a":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"empty":{"dtype":"F16","shape":[0],"data_offsets":[2,2]}})");
    require(adjacent.size() == 3 && adjacent[0].name == "z" && adjacent[0].begin == 0, "adjacent and empty offsets preserve JSON order");
    // DraftLinears::prefix and dimensions in draft_lin.hip and
    // minimax_dspark.hpp: five blocks, seven linears each, plus optional fc.
    // Only header strings are allocated; no tensor payload or device is needed.
    J root = J::object();
    std::uint64_t next = 0;
    const auto add = [&](const std::string& name, const std::vector<std::uint64_t>& shape, const std::string& dtype) {
        auto value = tensor(dtype, shape, 2);
        const auto bytes = static_cast<std::uint64_t>(value.find("data_offsets")->as_array()[1].as_int());
        value.set("data_offsets", J(J::Array{J(next), J(next + bytes)}));
        root.set(name, value);
        next += bytes;
    };
    const auto draft_linear = [&](const std::string& name, std::uint64_t in, std::uint64_t out) {
        add(name + ".suh", {in}, "F16");
        add(name + ".svh", {out}, "F16");
        add(name + ".trellis", {in / 16, out / 16, 64}, "I16");
    };
    for (unsigned layer = 0; layer < 5; ++layer) {
        const auto p = "language_model.model.dspark.layers." + std::to_string(layer) + ".decoder_layer.";
        draft_linear(p + "self_attn.q_proj", 6144, 8192);
        draft_linear(p + "self_attn.k_proj", 6144, 512);
        draft_linear(p + "self_attn.v_proj", 6144, 512);
        draft_linear(p + "self_attn.o_proj", 8192, 6144);
        draft_linear(p + "mlp.gate_proj", 6144, 12288);
        draft_linear(p + "mlp.up_proj", 6144, 12288);
        draft_linear(p + "mlp.down_proj", 12288, 6144);
    }
    draft_linear("language_model.model.dspark.fc", 30720, 6144);
    const auto json = root.dump();
    mm::PackHeaders draft;
    const auto parsed = draft.add(json, mm::header_payload_bytes(json.size(), 8 + json.size() + next));
    require(parsed.size() == 108, "all DSpark drafter linear headers accepted");
    for (const auto& t : parsed) {
        require(t.begin < t.end && t.end <= next, "drafter offsets retained: " + t.name);
        require(t.dtype == (t.name.ends_with(".trellis") ? "I16" : "F16"), "drafter dtype retained: " + t.name);
        const auto& dims = root.find(t.name)->find("shape")->as_array();
        require(t.shape.size() == dims.size(), "drafter rank retained: " + t.name);
        for (std::size_t i = 0; i < dims.size(); ++i)
            require(t.shape[i] == static_cast<std::uint64_t>(dims[i].as_int()), "drafter shape retained: " + t.name);
    }
}
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
struct Temp {
    fs::path path = fs::current_path() / ("strix-minimax-description-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { fs::create_directory(path); }
    ~Temp() { std::error_code ec; fs::remove_all(path, ec); }
};
void index_tests(const std::string& model) {
    Temp temp;
    const auto index = temp.path / "model.safetensors.index.json";
    const auto contract = temp.path / "minimax-m3.1-flash.toml";
    auto write = [&](const std::string& text) { std::ofstream(index, std::ios::binary) << text; };
    write("{\"weight_map\":{\"a\":\"part.safetensors\"}}");
    const auto before = strix::minimax::assets::pack_files(temp.path);
    require(before.size() == 1 && before[0].filename() == "part.safetensors", "legacy index without TOML");
    fs::copy_file(model, contract);
    require(strix::minimax::assets::pack_files(temp.path) == before, "correct optional TOML does not change files");
    std::ofstream(contract, std::ios::binary | std::ios::trunc) << "schema=\"strix-model/v1\"\n[dims]\nhidden=1\n";
    refuses([&] { strix::minimax::assets::pack_files(temp.path); }, "dims.hidden");
    // STRIX_MM_MODEL_DESCRIPTION_CHECK=0 restores the previous admission: the sidecar is neither opened nor interpreted.
    set_env("STRIX_MM_MODEL_DESCRIPTION_CHECK", "0");
    require(!mm::check_enabled(), "switch reads OFF");
    require(strix::minimax::assets::pack_files(temp.path) == before, "OFF ignores a wrong sidecar");
    set_env("STRIX_MM_MODEL_DESCRIPTION_CHECK", nullptr);
    require(mm::check_enabled(), "switch unset reads ON");
    refuses([&] { strix::minimax::assets::pack_files(temp.path); }, "dims.hidden");
    std::ofstream(contract, std::ios::binary | std::ios::trunc) << "x=[1 2]\n";
    refuses<std::invalid_argument>([&] { (void)md::load(contract.string()); }, "model description");
    refuses<std::invalid_argument>([&] { mm::load(contract.string()); }, "model description");
    fs::remove(contract);
    write("{\"weight_map\":{\"a\":\"x\",\"a\":\"y\"}}");
    refuses([&] { strix::minimax::assets::pack_files(temp.path); }, "a");
    write("{\"weight_map\":{\"a\":1}}");
    refuses([&] { strix::minimax::assets::pack_files(temp.path); }, "weight_map.a");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected model TOML fixture path");
        parser_tests(); contract_tests(argv[1]); header_tests(); pack_header_tests(); index_tests(argv[1]);
        std::cout << "PASS " << passed << " CPU model description checks\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL after " << passed << ": " << e.what() << '\n'; return 1; }
}
