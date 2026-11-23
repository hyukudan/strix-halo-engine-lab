#include "constants_profiles.hpp"

#include "strix/engine_profile.hpp"
#include "strix/weight/types.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace strix::model {
namespace {

using model_description::Value;

[[noreturn]] void bad_field(std::string_view field, std::string_view reason) {
    throw std::invalid_argument(std::string(field) + ": " + std::string(reason));
}

template<class T>
const T& expect(const Value& value, std::string_view field, std::string_view type) {
    if (const auto* item = std::get_if<T>(&value.data)) return *item;
    bad_field(field, "expected " + std::string(type));
}

bool one_of(std::string_view name, std::initializer_list<std::string_view> names) {
    return std::ranges::find(names, name) != names.end();
}

void known_fields(const Value::Table& table, const std::string& path,
                  std::initializer_list<std::string_view> keys) {
    for (const auto& [key, value] : table) {
        if (!one_of(key, keys)) bad_field(path + "." + key, "unknown field");
    }
}

std::int64_t count(const Value& value, const std::string& field, std::int64_t minimum = 1) {
    const auto number = expect<std::int64_t>(value, field, "an integer");
    if (number < minimum || number > std::numeric_limits<std::uint32_t>::max())
        bad_field(field, "integer out of range");
    return number;
}

const Value& required(const Value::Table& table, std::string_view key, const std::string& path) {
    const auto item = table.find(key);
    if (item == table.end()) bad_field(path + "." + std::string(key), "missing field");
    return item->second;
}

void validate_format(const Value& value, const std::string& field) {
    const auto& name = expect<std::string>(value, field, "a weight format name");
    for (const auto& format : weight::builtin_registry().formats) {
        const auto id = format.id.name;
        if (name == id || name == format.wire_schema || name == weight::serialize_id(format.id)) return;
        // The engine descriptions use the short storage names for these namespaces.
        for (const std::string_view prefix : {"plain.", "gguf.", "image.", "nvidia."})
            if (id.starts_with(prefix) && name == id.substr(prefix.size())) return;
        if (name == "mxfp8" && id == "block.mxfp8.e4m3-e8m0.g32") return;
    }
    bad_field(field, "unknown weight format '" + name + "'");
}

void validate_drafter(const Value::Table& table) {
    const std::string path = "drafter";
    known_fields(table, path, {"kind", "block", "taps", "layers", "windows", "rank", "head_screen", "weights",
                               "max_window", "rollback"});
    const auto& kind = expect<std::string>(required(table, "kind", path), "drafter.kind", "a drafter kind");
    if (!one_of(kind, {"chain", "block", "lookup", "table", "model"}))
        bad_field("drafter.kind", "unknown drafter kind '" + kind + "'");
    if (kind == "block") (void)required(table, "block", path);
    for (const auto& [key, value] : table) {
        const auto field = path + "." + key;
        if (one_of(key, {"block", "layers", "rank", "max_window"})) {
            (void)count(value, field);
        } else if (key == "taps" || key == "windows") {
            const auto& items = expect<Value::Array>(value, field, "an array");
            const bool named = key == "taps" && !items.empty() && std::holds_alternative<std::string>(items.front().data);
            for (const auto& item : items) {
                if (named) {
                    if (expect<std::string>(item, field, "a tap name").empty()) bad_field(field, "empty tap name");
                } else (void)count(item, field, 0);
            }
        } else if (key == "weights") {
            validate_format(value, field);
        } else if (key == "rollback") {
            const auto& mode = expect<std::string>(value, field, "a rollback mode");
            if (!one_of(mode, {"position_rings", "replay"})) bad_field(field, "unknown rollback mode '" + mode + "'");
        } else if (key == "head_screen") {
            if (expect<std::string>(value, field, "a head screen name").empty()) bad_field(field, "empty head screen name");
        }
    }
}

void validate_weight_roles(const Value::Table& table) {
    const std::string path = "weights.format";
    known_fields(table, path, {"attention", "head", "dense", "experts", "drafter"});
    for (const auto& [role, value] : table) {
        const auto field = path + "." + role;
        if (const auto* formats = std::get_if<Value::Array>(&value.data)) {
            if (formats->empty()) bad_field(field, "empty format list");
            for (const auto& format : *formats) validate_format(format, field);
        } else {
            validate_format(value, field);
        }
    }
}

void validate_fusion(const Value::Table& table) {
    const std::string path = "fusion";
    known_fields(table, path, {"groups"});
    const auto& groups = expect<Value::Array>(required(table, "groups", path), "fusion.groups", "an array");
    // Names for the shared projection groups and the prepared-GQA epilogue.
    constexpr std::array<std::string_view, 3> registry{"qkv", "gate_up", "conv4.clip.kv_append"};
    std::set<std::string_view> seen;
    for (const auto& value : groups) {
        const auto& group = expect<std::string>(value, "fusion.groups", "a fusion group name");
        if (std::ranges::find(registry, group) == registry.end())
            bad_field("fusion.groups", "unknown fusion group '" + group + "'");
        if (!seen.insert(group).second) bad_field("fusion.groups", "duplicate fusion group '" + group + "'");
    }
}

void validate_schedule(const Value::Table& table, const model_description::ModelDescription& description) {
    const std::string path = "schedule";
    known_fields(table, path, {"order", "core_first", "core_last", "core_passes", "residual_block_layers"});
    if (const auto order = table.find("order"); order != table.end())
        (void)expect<std::string>(order->second, "schedule.order", "a layer order expression");
    if (const auto block = table.find("residual_block_layers"); block != table.end())
        (void)count(block->second, "schedule.residual_block_layers");
    if (!table.contains("core_first") && !table.contains("core_last") && !table.contains("core_passes")) return;
    const auto* layers = description.find("dims.layers");
    if (!layers) bad_field("dims.layers", "required with loop dimensions");
    const auto physical = count(*layers, "dims.layers");
    const auto first = count(required(table, "core_first", path), "schedule.core_first", 0);
    const auto last = count(required(table, "core_last", path), "schedule.core_last");
    const auto passes = count(required(table, "core_passes", path), "schedule.core_passes");
    if (last <= first || last > physical) bad_field("schedule.core_last", "invalid exclusive loop bound");
    const auto visits = static_cast<std::uint64_t>(physical) +
        static_cast<std::uint64_t>(last - first) * static_cast<std::uint64_t>(passes - 1);
    if (visits > std::numeric_limits<std::uint32_t>::max()) bad_field("schedule.core_passes", "too many logical visits");
}

void project_profiles(Value& value, std::string_view model, std::span<const std::string_view> lever_names) {
    (void)expect<Value::Table>(value, "profile", "a table");
    auto& table = std::get<Value::Table>(value.data);
    const auto known_lever = [&](std::string_view key) {
        return std::ranges::find(lever_names, key) != lever_names.end();
    };
    for (const auto& [section, cells] : table) {
        const auto path = "profile." + section;
        // profile.fast / profile.turbo: tier columns keyed by lever. Any other section is one lever's row
        // (profile.<lever>: tier cells and the lever's own string attributes such as identity), emitted as is.
        const bool tier = one_of(section, {"fast", "turbo"});
        if (!tier && !known_lever(section)) bad_field(path, "unknown profile lever for " + std::string(model));
        for (const auto& [key, cell] : expect<Value::Table>(cells, path, "a table")) {
            const auto field = path + "." + key;
            if (tier && !known_lever(key)) bad_field(field, "unknown profile lever for " + std::string(model));
            (void)expect<std::string>(cell, field, "a profile cell string");
        }
    }
    const auto fast = table.find("fast");
    if (fast == table.end()) return;
    auto& turbo = std::get<Value::Table>(table.try_emplace("turbo", Value{Value::Table{}}).first->second.data);
    constexpr auto fast_column = static_cast<unsigned>(profile::Name::fast);
    constexpr auto turbo_column = static_cast<unsigned>(profile::Name::turbo);
    for (const auto& [key, cell] : std::get<Value::Table>(fast->second.data)) {
        const auto override = turbo.find(key);
        profile::Lever row{};
        row.value[fast_column] = std::get<std::string>(cell.data);
        if (override != turbo.end()) row.value[turbo_column] = std::get<std::string>(override->second.data);
        if (profile::value_column(row, profile::Name::turbo) == fast_column) turbo.insert_or_assign(key, cell);
    }
}

void validate_and_project(model_description::ModelDescription& description, std::string_view model,
                          std::span<const std::string_view> lever_names) {
    if (const auto* drafter = description.find("drafter"))
        validate_drafter(expect<Value::Table>(*drafter, "drafter", "a table"));
    if (const auto* weights = description.find("weights"))
        (void)expect<Value::Table>(*weights, "weights", "a table");
    if (const auto* formats = description.find("weights.format"))
        validate_weight_roles(expect<Value::Table>(*formats, "weights.format", "a table"));
    if (const auto* fusion = description.find("fusion"))
        validate_fusion(expect<Value::Table>(*fusion, "fusion", "a table"));
    if (const auto* schedule = description.find("schedule"))
        validate_schedule(expect<Value::Table>(*schedule, "schedule", "a table"), description);
    if (auto found = description.fields.find("profile"); found != description.fields.end())
        project_profiles(found->second, model, lever_names);
}

constexpr std::string_view kReserved[] = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr",
    "constinit", "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete",
    "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for",
    "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
    // common predefined macros that would silently rewrite a name
    "major", "minor", "linux", "unix", "i386", "NULL", "EOF", "assert", "errno", "stdin", "stdout", "stderr"};

std::string cxx_name(std::string_view key) {
    std::string out;
    out.reserve(key.size() + 2);
    for (const char c : key) {
        const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        out.push_back(word ? c : '_');
    }
    if (out.empty() || (out.front() >= '0' && out.front() <= '9')) out.insert(out.begin(), 'k');
    for (const auto reserved : kReserved)
        if (out == reserved) return out + '_';
    return out;
}

std::string integer_literal(std::int64_t v) {
    if (v == std::numeric_limits<std::int64_t>::min()) return "(-9223372036854775807 - 1)";
    return std::to_string(v);
}

std::string real_literal(double v) {
    std::array<char, 64> hex{};
    std::array<char, 64> dec{};
    const auto h = std::to_chars(hex.data(), hex.data() + hex.size(), v, std::chars_format::hex);
    const auto d = std::to_chars(dec.data(), dec.data() + dec.size(), v);
    if (h.ec != std::errc{} || d.ec != std::errc{}) throw std::runtime_error("cannot format a model real");
    std::string_view hv(hex.data(), static_cast<std::size_t>(h.ptr - hex.data()));
    std::string out;
    if (!hv.empty() && hv.front() == '-') { out = "-"; hv.remove_prefix(1); }
    out += "0x";
    out += hv;
    out += " /* ";
    out.append(dec.data(), d.ptr);
    out += " */";
    return out;
}

std::string string_literal(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        const auto u = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (u < 0x20U || u == 0x7fU) {
            std::array<char, 8> esc{};
            std::snprintf(esc.data(), esc.size(), "\\%03o", static_cast<unsigned>(u));
            out += esc.data();
        } else out.push_back(c);
    }
    return out + "\"";
}

// Element type and literal of a scalar; empty type for tables / arrays.
std::pair<std::string_view, std::string> scalar(const Value& v) {
    if (const auto* i = std::get_if<std::int64_t>(&v.data)) return {"std::int64_t", integer_literal(*i)};
    if (const auto* r = std::get_if<double>(&v.data)) return {"double", real_literal(*r)};
    if (const auto* b = std::get_if<bool>(&v.data)) return {"bool", *b ? "true" : "false"};
    if (const auto* s = std::get_if<std::string>(&v.data)) return {"std::string_view", string_literal(*s)};
    return {{}, {}};
}

void emit_table(std::string& out, const Value::Table& table, const std::string& indent) {
    for (const auto& [key, value] : table) {
        const auto name = cxx_name(key);
        if (const auto* sub = std::get_if<Value::Table>(&value.data)) {
            out += indent + "namespace " + name + " {\n";
            emit_table(out, *sub, indent);
            out += indent + "} // namespace " + name + "\n";
            continue;
        }
        if (const auto* array = std::get_if<Value::Array>(&value.data)) {
            if (array->empty()) { out += indent + "// " + key + ": empty array (no constant)\n"; continue; }
            std::string_view type;
            std::vector<std::string> items;
            bool uniform = true;
            for (const auto& item : *array) {
                auto [t, literal] = scalar(item);
                if (t.empty() || (!type.empty() && t != type)) { uniform = false; break; }
                type = t;
                items.push_back(std::move(literal));
            }
            if (!uniform) { out += indent + "// " + key + ": array of tables or mixed array (no constant)\n"; continue; }
            std::string line = indent + "inline constexpr " + std::string(type) + " " + name + "[] = {";
            std::string body;
            std::size_t column = line.size();
            for (std::size_t i = 0; i < items.size(); ++i) {
                std::string piece = items[i] + (i + 1 < items.size() ? ", " : "");
                if (column + piece.size() > 116) { body += "\n" + indent + "    "; column = indent.size() + 4; }
                body += piece;
                column += piece.size();
            }
            out += line + body + "};\n";
            continue;
        }
        const auto [type, literal] = scalar(value);
        out += indent + "inline constexpr " + std::string(type) + " " + name + " = " + literal + ";\n";
    }
}

} // namespace

std::string identifier(std::string_view model_name) { return cxx_name(model_name); }

std::string emit_constants(const model_description::ModelDescription& description,
                           std::string_view source_path) {
    return emit_constants(description, source_path, {});
}

std::string emit_constants(const model_description::ModelDescription& description,
                           std::string_view source_path, std::span<const std::string_view> lever_names) {
    const auto* name = description.find("name");
    const auto* text = name ? std::get_if<std::string>(&name->data) : nullptr;
    if (!text || text->empty()) throw std::invalid_argument(std::string(source_path) + ": name is missing");
    const auto ident = identifier(*text);
    std::string out;
    const auto source = std::string(source_path);
    auto projected = description;
    try {
        validate_and_project(projected, *text, lever_names);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(source + ": " + error.what());
    }
    out += "// Generated by strix-model-gen from " + source + ". Do not edit: edit the TOML, then run\n";
    out += "// strix-model-gen --write " + source + " include/strix/model_data/" + ident + ".hpp " + source + "\n";
    out += "// strix-model-constants-tests fails when this header and the TOML disagree. Engines bind these values\n";
    out += "// with strix/model/bind.hpp.\n";
    out += "#pragma once\n\n#include <cstdint>\n#include <string_view>\n\n";
    out += "namespace strix::model_data::" + ident + " {\n\n";
    emit_table(out, projected.fields, "");
    out += "\n} // namespace strix::model_data::" + ident + "\n";
    return out;
}

} // namespace strix::model
