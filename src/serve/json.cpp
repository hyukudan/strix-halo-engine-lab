#include "strix/serve/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>

namespace strix::serve {

Json::Json(std::int64_t value) : kind_(Kind::number), text_(std::to_string(value)) {}
Json::Json(std::uint64_t value) : kind_(Kind::number), text_(std::to_string(value)) {}
Json::Json(double value) : kind_(Kind::number) {
    if (!std::isfinite(value)) throw JsonError("JSON numbers must be finite");
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    text_.assign(buffer, result.ptr);
}
Json Json::number_text(std::string text) {
    Json out;
    out.kind_ = Kind::number;
    out.text_ = std::move(text);
    return out;
}
bool Json::as_bool() const {
    if (kind_ != Kind::boolean) throw JsonError("JSON value is not a boolean");
    return boolean_;
}
double Json::as_double() const {
    if (kind_ != Kind::number) throw JsonError("JSON value is not a number");
    double value = 0.0;
    const auto result = std::from_chars(text_.data(), text_.data() + text_.size(), value);
    if (result.ec != std::errc{}) throw JsonError("JSON number is out of range");
    return value;
}
std::int64_t Json::as_int() const {
    if (kind_ != Kind::number) throw JsonError("JSON value is not a number");
    std::int64_t value = 0;
    const auto result = std::from_chars(text_.data(), text_.data() + text_.size(), value);
    if (result.ec == std::errc{} && result.ptr == text_.data() + text_.size()) return value;
    const auto real = as_double();
    if (real != std::floor(real) || std::fabs(real) > 9.007199254740992e15) throw JsonError("JSON number is not an integer");
    return static_cast<std::int64_t>(real);
}
const std::string& Json::as_string() const {
    if (kind_ != Kind::string) throw JsonError("JSON value is not a string");
    return text_;
}
const std::string& Json::number_source() const {
    if (kind_ != Kind::number) throw JsonError("JSON value is not a number");
    return text_;
}
const Json::Array& Json::as_array() const {
    if (kind_ != Kind::array) throw JsonError("JSON value is not an array");
    return *array_;
}
Json::Array& Json::as_array() {
    if (kind_ != Kind::array) throw JsonError("JSON value is not an array");
    return *array_;
}
const Json::Object& Json::as_object() const {
    if (kind_ != Kind::object) throw JsonError("JSON value is not an object");
    return *object_;
}
Json::Object& Json::as_object() {
    if (kind_ != Kind::object) throw JsonError("JSON value is not an object");
    return *object_;
}
const Json* Json::find(std::string_view key) const {
    if (kind_ != Kind::object) return nullptr;
    for (const auto& member : *object_) if (member.first == key) return &member.second;
    return nullptr;
}
Json& Json::set(std::string key, Json value) {
    if (kind_ == Kind::null) { kind_ = Kind::object; object_ = std::make_shared<Object>(); }
    auto& members = as_object();
    for (auto& member : members) if (member.first == key) { member.second = std::move(value); return member.second; }
    members.emplace_back(std::move(key), std::move(value));
    return members.back().second;
}
void Json::push(Json value) {
    if (kind_ == Kind::null) { kind_ = Kind::array; array_ = std::make_shared<Array>(); }
    as_array().push_back(std::move(value));
}
std::size_t Json::size() const {
    if (kind_ == Kind::array) return array_->size();
    if (kind_ == Kind::object) return object_->size();
    return 0;
}

void append_json_string(std::string& out, std::string_view value) {
    out.push_back('"');
    for (const char c : value) {
        const auto u = static_cast<unsigned char>(c);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (u < 0x20U) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", u);
                out += buffer;
            } else out.push_back(c);
        }
    }
    out.push_back('"');
}

void Json::write(std::string& out, bool spaced) const {
    switch (kind_) {
    case Kind::null: out += "null"; return;
    case Kind::boolean: out += boolean_ ? "true" : "false"; return;
    case Kind::number: out += text_; return;
    case Kind::string: append_json_string(out, text_); return;
    case Kind::array: {
        out.push_back('[');
        bool first = true;
        for (const auto& item : *array_) {
            if (!first) out += spaced ? ", " : ",";
            first = false;
            item.write(out, spaced);
        }
        out.push_back(']');
        return;
    }
    case Kind::object: {
        out.push_back('{');
        bool first = true;
        for (const auto& [key, item] : *object_) {
            if (!first) out += spaced ? ", " : ",";
            first = false;
            append_json_string(out, key);
            out += spaced ? ": " : ":";
            item.write(out, spaced);
        }
        out.push_back('}');
        return;
    }
    }
}
std::string Json::dump() const { std::string out; write(out, false); return out; }
std::string Json::dump_spaced() const { std::string out; write(out, true); return out; }

namespace {
class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}
    Json document() {
        auto value = parse(0);
        skip();
        if (at_ != text_.size()) fail("trailing characters after the JSON value");
        return value;
    }
private:
    [[noreturn]] void fail(const char* what) const {
        throw JsonError(std::string("JSON parse error at byte ") + std::to_string(at_) + ": " + what);
    }
    void skip() {
        while (at_ < text_.size() && (text_[at_] == ' ' || text_[at_] == '\t' || text_[at_] == '\n' || text_[at_] == '\r')) ++at_;
    }
    bool literal(std::string_view word) {
        if (text_.substr(at_, word.size()) != word) return false;
        at_ += word.size();
        return true;
    }
    Json parse(unsigned depth) {
        if (depth > 128U) fail("nesting is too deep");
        skip();
        if (at_ >= text_.size()) fail("unexpected end of input");
        const char c = text_[at_];
        if (c == '{') {
            ++at_;
            Json::Object members;
            skip();
            if (at_ < text_.size() && text_[at_] == '}') { ++at_; return Json(std::move(members)); }
            for (;;) {
                skip();
                if (at_ >= text_.size() || text_[at_] != '"') fail("expected an object key");
                auto key = string();
                skip();
                if (at_ >= text_.size() || text_[at_] != ':') fail("expected ':'");
                ++at_;
                auto value = parse(depth + 1U);
                bool replaced = false;
                for (auto& member : members) if (member.first == key) { member.second = std::move(value); replaced = true; break; }
                if (!replaced) members.emplace_back(std::move(key), std::move(value));
                skip();
                if (at_ < text_.size() && text_[at_] == ',') { ++at_; continue; }
                if (at_ < text_.size() && text_[at_] == '}') { ++at_; return Json(std::move(members)); }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++at_;
            Json::Array items;
            skip();
            if (at_ < text_.size() && text_[at_] == ']') { ++at_; return Json(std::move(items)); }
            for (;;) {
                items.push_back(parse(depth + 1U));
                skip();
                if (at_ < text_.size() && text_[at_] == ',') { ++at_; continue; }
                if (at_ < text_.size() && text_[at_] == ']') { ++at_; return Json(std::move(items)); }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') return Json(string());
        if (literal("true")) return Json(true);
        if (literal("false")) return Json(false);
        if (literal("null")) return Json();
        return number();
    }
    Json number() {
        const auto begin = at_;
        if (at_ < text_.size() && text_[at_] == '-') ++at_;
        const auto digits = [&] { const auto start = at_; while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') ++at_; return at_ - start; };
        if (at_ < text_.size() && text_[at_] == '0') ++at_;
        else if (!digits()) fail("invalid number");
        if (at_ < text_.size() && text_[at_] == '.') { ++at_; if (!digits()) fail("invalid fraction"); }
        if (at_ < text_.size() && (text_[at_] == 'e' || text_[at_] == 'E')) {
            ++at_;
            if (at_ < text_.size() && (text_[at_] == '+' || text_[at_] == '-')) ++at_;
            if (!digits()) fail("invalid exponent");
        }
        return Json::number_text(std::string(text_.substr(begin, at_ - begin)));
    }
    unsigned hex4() {
        if (at_ + 4U > text_.size()) fail("truncated \\u escape");
        unsigned value = 0;
        for (unsigned i = 0; i < 4U; ++i) {
            const char h = text_[at_++];
            value <<= 4U;
            if (h >= '0' && h <= '9') value |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') value |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') value |= static_cast<unsigned>(h - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return value;
    }
    static void utf8(std::string& out, std::uint32_t cp) {
        if (cp < 0x80U) out.push_back(static_cast<char>(cp));
        else if (cp < 0x800U) { out.push_back(static_cast<char>(0xC0U | (cp >> 6U))); out.push_back(static_cast<char>(0x80U | (cp & 0x3FU))); }
        else if (cp < 0x10000U) {
            out.push_back(static_cast<char>(0xE0U | (cp >> 12U))); out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else {
            out.push_back(static_cast<char>(0xF0U | (cp >> 18U))); out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU))); out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }
    std::string string() {
        ++at_; // opening quote
        std::string out;
        for (;;) {
            if (at_ >= text_.size()) fail("unterminated string");
            const char c = text_[at_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20U) fail("control character in string");
            if (c != '\\') { out.push_back(c); continue; }
            if (at_ >= text_.size()) fail("unterminated escape");
            const char e = text_[at_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t cp = hex4();
                if (cp >= 0xD800U && cp <= 0xDBFFU) {
                    if (at_ + 2U <= text_.size() && text_[at_] == '\\' && text_[at_ + 1U] == 'u') {
                        at_ += 2U;
                        const auto low = hex4();
                        if (low < 0xDC00U || low > 0xDFFFU) fail("invalid surrogate pair");
                        cp = 0x10000U + ((cp - 0xD800U) << 10U) + (low - 0xDC00U);
                    } else cp = 0xFFFDU;
                } else if (cp >= 0xDC00U && cp <= 0xDFFFU) cp = 0xFFFDU;
                utf8(out, cp);
                break;
            }
            default: fail("invalid escape");
            }
        }
    }
    std::string_view text_;
    std::size_t at_ = 0;
};
}  // namespace

Json parse_json(std::string_view text) { return Parser(text).document(); }

}  // namespace strix::serve
