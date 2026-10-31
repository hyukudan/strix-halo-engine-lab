#pragma once
// Host-only JSON factored from glm_pack.cpp (see its provenance notice), shared with metadata-only probes.
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace strix::glm::engine::detail {
struct Json;
using JsonObj = std::map<std::string, Json>;
using JsonArr = std::vector<Json>;
struct Json {
    std::variant<std::nullptr_t, double, std::string, JsonArr, JsonObj> v;
    const JsonObj& obj() const { return std::get<JsonObj>(v); }
    const JsonArr& arr() const { return std::get<JsonArr>(v); }
    const std::string& str() const { return std::get<std::string>(v); }
    double num() const { return std::get<double>(v); }
};



inline Json parse_json(const char* begin, const char* end) {
    struct Parser {
        const char* p;
        const char* end;
        void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
        bool take(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
        void need(char c) { if (!take(c)) throw std::runtime_error("json: delimiter"); }
        std::string string() {
            need('"');
            std::string s;
            while (p < end) {
                char c = *p++;
                if (c == '"') return s;
                if (c == '\\') {
                    if (p == end) break;
                    c = *p++;
                    if (c == 'n') c = '\n';
                    else if (c == 'r') c = '\r';
                    else if (c == 't') c = '\t';
                    else if (c == 'b') c = '\b';
                    else if (c == 'f') c = '\f';
                    else if (c == 'u') {
                        auto hex4 = [&]() {
                            unsigned value = 0;
                            for (unsigned i = 0; i < 4; ++i) {
                                if (p == end) throw std::runtime_error("json: unicode");
                                const char h = *p++;
                                const unsigned digit = h >= '0' && h <= '9' ? static_cast<unsigned>(h - '0') :
                                    h >= 'a' && h <= 'f' ? static_cast<unsigned>(h - 'a' + 10) :
                                    h >= 'A' && h <= 'F' ? static_cast<unsigned>(h - 'A' + 10) : 16U;
                                if (digit == 16) throw std::runtime_error("json: unicode");
                                value = value * 16 + digit;
                            }
                            return value;
                        };
                        unsigned u = hex4();
                        if (u >= 0xd800 && u <= 0xdbff) {
                            if (end - p < 2 || p[0] != '\\' || p[1] != 'u') throw std::runtime_error("json: surrogate");
                            p += 2;
                            const unsigned low = hex4();
                            if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("json: surrogate");
                            u = 0x10000 + (u - 0xd800) * 1024 + low - 0xdc00;
                        } else if (u >= 0xdc00 && u <= 0xdfff) throw std::runtime_error("json: surrogate");
                        if (u >= 0x10000) s += static_cast<char>(0xf0 | (u >> 18));
                        if (u >= 0x800) s += static_cast<char>((u >= 0x10000 ? 0x80 : 0xe0) | ((u >> 12) & 63));
                        if (u >= 0x80) s += static_cast<char>((u >= 0x800 ? 0x80 : 0xc0) | ((u >> 6) & 63));
                        s += static_cast<char>((u >= 0x80 ? 0x80 : 0) | (u & (u >= 0x80 ? 63 : 127)));
                        continue;
                    }
                    else if (c != '"' && c != '\\' && c != '/') throw std::runtime_error("json: escape");
                } else if (static_cast<unsigned char>(c) < 32) throw std::runtime_error("json: control character");
                s += c;
            }
            throw std::runtime_error("json: unterminated string");
        }
        Json parse(unsigned depth = 0) {
            if (depth > 64) throw std::runtime_error("json: nesting");
            ws();
            if (p == end) throw std::runtime_error("json: eof");
            if (take('{')) {
                JsonObj o;
                if (take('}')) return {o};
                do {
                    auto key = string();
                    need(':');
                    if (!o.emplace(std::move(key), parse(depth + 1)).second) throw std::runtime_error("json: duplicate key");
                    if (take('}')) return {o};
                } while (take(','));
                throw std::runtime_error("json: object");
            }
            if (take('[')) {
                JsonArr a;
                if (take(']')) return {a};
                do { a.push_back(parse(depth + 1)); if (take(']')) return {a}; } while (take(','));
                throw std::runtime_error("json: array");
            }
            if (*p == '"') return {string()};
            for (const auto& word : {std::string("null"), std::string("true"), std::string("false")}) {
                if (static_cast<std::size_t>(end - p) >= word.size() && std::string(p, word.size()) == word) {
                    p += word.size();
                    if (word == "null") return {nullptr};
                    return {word == "true" ? 1.0 : 0.0};
                }
            }
            const char* start = p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) ++p;
            const std::string number(start, p);
            char* tail = nullptr;
            const double v = std::strtod(number.c_str(), &tail);
            if (number.empty() || tail != number.c_str() + number.size()) throw std::runtime_error("json: number");
            return {v};
        }
    } parser{begin, end};
    auto out = parser.parse();
    parser.ws();
    if (parser.p != end) throw std::runtime_error("json: trailing input");
    return out;
}
}  // namespace strix::glm::engine::detail
