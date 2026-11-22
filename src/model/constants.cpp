#include "strix/model/constants.hpp"

#include <array>
#include <charconv>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace strix::model {
namespace {

using model_description::Value;

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
    const auto* name = description.find("name");
    const auto* text = name ? std::get_if<std::string>(&name->data) : nullptr;
    if (!text || text->empty()) throw std::invalid_argument(std::string(source_path) + ": name is missing");
    const auto ident = identifier(*text);
    std::string out;
    const auto source = std::string(source_path);
    out += "// Generated by strix-model-gen from " + source + ". Do not edit: edit the TOML, then run\n";
    out += "// strix-model-gen --write " + source + " include/strix/model_data/" + ident + ".hpp " + source + "\n";
    out += "// strix-model-constants-tests fails when this header and the TOML disagree. Engines bind these values\n";
    out += "// with strix/model/bind.hpp.\n";
    out += "#pragma once\n\n#include <cstdint>\n#include <string_view>\n\n";
    out += "namespace strix::model_data::" + ident + " {\n\n";
    emit_table(out, description.fields, "");
    out += "\n} // namespace strix::model_data::" + ident + "\n";
    return out;
}

} // namespace strix::model
