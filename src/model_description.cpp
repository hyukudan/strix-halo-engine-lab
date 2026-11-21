#include "strix/model_description.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <deque>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace strix::model_description {
namespace {

class Parser {
public:
    Parser(std::string_view input, TableSyntax syntax) : input_(input), syntax_(syntax) {}

    ModelDescription run() {
        validate_text();
        ModelDescription result;
        Value::Table* current = &result.fields;
        while (true) {
            whitespace(true);
            if (end()) {
                for (auto& [key, rows] : array_tables_) {
                    auto& array = std::get<Value::Array>(result.fields.at(key).data);
                    array.assign(std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
                }
                return result;
            }
            if (take('[')) {
                const bool array = take('[');
                if (array && syntax_ != TableSyntax::root_array_tables)
                    fail("array tables are outside the supported subset");
                const auto path = key_path();
                expect(']');
                if (array) {
                    if (!take(']') || path.size() != 1) fail("expected root array table");
                    current = &append_table(result.fields, path.front());
                } else {
                    current = &table(result.fields, path, true);
                }
            } else {
                const auto path = key_path();
                expect('=');
                insert(*current, path, value(0));
            }
            whitespace(false);
            if (!end() && !take('\n')) fail("expected end of line");
        }
    }

private:
    std::string_view input_;
    TableSyntax syntax_;
    std::size_t position_ = 0;
    std::set<const Value::Table*> declared_, sealed_;
    // Row addresses stay stable while the parser tracks declared and sealed tables.
    std::map<std::string, std::deque<Value>> array_tables_;

    Value::Table& append_table(Value::Table& root, const std::string& key) {
        auto [rows, added] = array_tables_.try_emplace(key);
        if (added) {
            // Built in place: moving a temporary Value trips a GCC 15 maybe-uninitialized false positive.
            auto [slot, inserted] = root.try_emplace(key);
            if (!inserted) fail("array table conflicts with key: " + key);
            slot->second.data.emplace<Value::Array>();
        }
        auto& row = rows->second.emplace_back();
        return row.data.emplace<Value::Table>();
    }

    void validate_text() {
        // TOML is UTF-8. Reject invalid byte sequences even inside comments.
        for (position_ = 0; position_ < input_.size(); ++position_) {
            const auto c = static_cast<unsigned char>(input_[position_]);
            if ((c < 0x20U && c != '\t' && c != '\n' && c != '\r') || c == 0x7fU)
                fail("control byte in input");
            if (c < 0x80U) continue;
            unsigned following = 0;
            std::uint32_t codepoint = 0, minimum = 0;
            if (c >= 0xc2U && c <= 0xdfU) { following = 1; codepoint = c & 0x1fU; minimum = 0x80U; }
            else if (c >= 0xe0U && c <= 0xefU) { following = 2; codepoint = c & 0x0fU; minimum = 0x800U; }
            else if (c >= 0xf0U && c <= 0xf4U) { following = 3; codepoint = c & 7U; minimum = 0x10000U; }
            else fail("invalid UTF-8 lead byte");
            for (unsigned i = 0; i < following; ++i) {
                if (position_ + 1 >= input_.size()) fail("truncated UTF-8 sequence");
                const auto next = static_cast<unsigned char>(input_[++position_]);
                if ((next & 0xc0U) != 0x80U) fail("invalid UTF-8 continuation");
                codepoint = (codepoint << 6U) | (next & 0x3fU);
            }
            if (codepoint < minimum || codepoint > 0x10ffffU || (codepoint >= 0xd800U && codepoint <= 0xdfffU))
                fail("invalid UTF-8 codepoint");
        }
        position_ = 0;
    }

    bool end() const { return position_ == input_.size(); }
    char peek() const { return end() ? '\0' : input_[position_]; }
    bool take(char c) {
        if (!end() && peek() == c) { ++position_; return true; }
        return false;
    }
    [[noreturn]] void fail(const std::string& message) const {
        const auto prefix = input_.substr(0, position_);
        const auto line = 1 + std::count(prefix.begin(), prefix.end(), '\n');
        const auto last = prefix.rfind('\n');
        const auto column = position_ - (last == std::string_view::npos ? 0 : last + 1) + 1;
        throw std::invalid_argument("model description line " + std::to_string(line) +
            " column " + std::to_string(column) + ": " + message);
    }
    void whitespace(bool multiline) {
        while (!end()) {
            if (peek() == ' ' || peek() == '\t' || (multiline && peek() == '\n')) ++position_;
            else if (peek() == '\r' && position_ + 1 < input_.size() && input_[position_ + 1] == '\n') {
                ++position_; // CRLF is accepted, bare CR is not.
            } else if (peek() == '#') {
                while (!end() && peek() != '\n' && peek() != '\r') ++position_;
            } else break;
        }
    }
    void expect(char c) {
        whitespace(false);
        if (!take(c)) fail(std::string("expected '") + c + "'");
    }
    static bool key_byte(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-';
    }
    std::vector<std::string> key_path() {
        std::vector<std::string> path;
        do {
            whitespace(false);
            const auto begin = position_;
            while (!end() && key_byte(peek())) ++position_;
            if (position_ == begin) fail("expected bare key");
            path.emplace_back(input_.substr(begin, position_ - begin));
            whitespace(false);
        } while (take('.'));
        return path;
    }
    Value::Table& table(Value::Table& root, const std::vector<std::string>& path, bool declaration) {
        auto* current = &root;
        for (const auto& part : path) {
            if (sealed_.contains(current)) fail("cannot extend inline table");
            auto [entry, added] = current->try_emplace(part, Value{Value::Table{}});
            (void)added;
            current = std::get_if<Value::Table>(&entry->second.data);
            if (!current) fail("table conflicts with value: " + part);
        }
        if (sealed_.contains(current)) fail("cannot redeclare inline table");
        if (declaration && !declared_.insert(current).second) fail("duplicate table");
        return *current;
    }
    void insert(Value::Table& root, const std::vector<std::string>& path, Value item) {
        const std::vector<std::string> parents(path.begin(), path.end() - 1);
        auto& owner = table(root, parents, false);
        if (sealed_.contains(&owner)) fail("cannot extend inline table");
        const auto [entry, added] = owner.emplace(path.back(), std::move(item));
        if (!added) fail("duplicate key: " + path.back());
        if (auto* child = std::get_if<Value::Table>(&entry->second.data)) seal(*child);
        // Dotted assignments define their parents, which cannot later be reopened.
        auto* current = &root;
        for (const auto& part : parents) {
            current = &std::get<Value::Table>(current->at(part).data);
            declared_.insert(current);
        }
    }
    void seal(Value::Table& item) {
        sealed_.insert(&item);
        for (auto& [key, child] : item) {
            (void)key;
            if (auto* nested = std::get_if<Value::Table>(&child.data)) seal(*nested);
        }
    }
    std::string string() {
        const char quote = peek();
        ++position_;
        std::string result;
        while (!end()) {
            const char c = input_[position_++];
            if (c == quote) return result;
            if (static_cast<unsigned char>(c) < 0x20U && c != '\t') fail("control byte in string");
            if (c == '\x7f') fail("control byte in string");
            if (c != '\\' || quote == '\'') { result += c; continue; }
            if (end()) fail("unterminated escape");
            switch (input_[position_++]) {
            case 'b': result += '\b'; break;
            case 't': result += '\t'; break;
            case 'n': result += '\n'; break;
            case 'f': result += '\f'; break;
            case 'r': result += '\r'; break;
            case '"': result += '"'; break;
            case '\\': result += '\\'; break;
            default: fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }
    Value value(unsigned depth) {
        if (depth > 32) fail("value nesting exceeds 32");
        whitespace(false);
        if (peek() == '"' || peek() == '\'') return Value{string()};
        if (take('[')) {
            Value::Array items;
            whitespace(true);
            if (take(']')) return Value{std::move(items)};
            while (true) {
                items.push_back(value(depth + 1));
                whitespace(true);
                if (take(']')) break;
                expect(',');
                whitespace(true);
                if (take(']')) break;
            }
            return Value{std::move(items)};
        }
        if (take('{')) {
            Value::Table items;
            whitespace(false);
            if (!take('}')) while (true) {
                const auto path = key_path();
                expect('=');
                insert(items, path, value(depth + 1));
                whitespace(false);
                if (take('}')) break;
                expect(',');
                if (peek() == '}') fail("trailing comma in inline table");
            }
            return Value{std::move(items)};
        }
        const auto begin = position_;
        while (!end() && peek() != ' ' && peek() != '\t' && peek() != '\r' && peek() != '\n' &&
               peek() != '#' && peek() != ',' && peek() != ']' && peek() != '}') ++position_;
        const auto token = input_.substr(begin, position_ - begin);
        if (token == "true") return Value{true};
        if (token == "false") return Value{false};
        if (token.empty()) fail("missing value");
        // Decimal grammar is checked before conversion; no partial parses, NaN,
        // infinity, dates, base prefixes, underscores or leading-zero integers.
        auto numeric = token;
        if (numeric.front() == '+' || numeric.front() == '-') numeric.remove_prefix(1);
        const auto digits = [](char c) { return c >= '0' && c <= '9'; };
        std::size_t n = 0;
        while (n < numeric.size() && digits(numeric[n])) ++n;
        if (!n || (n > 1 && numeric.front() == '0')) fail("invalid decimal value");
        bool floating = false;
        if (n < numeric.size() && numeric[n] == '.') {
            floating = true;
            const auto first = ++n;
            while (n < numeric.size() && digits(numeric[n])) ++n;
            if (n == first) fail("missing fractional digits");
        }
        if (n < numeric.size() && (numeric[n] == 'e' || numeric[n] == 'E')) {
            floating = true;
            ++n;
            if (n < numeric.size() && (numeric[n] == '+' || numeric[n] == '-')) ++n;
            const auto first = n;
            while (n < numeric.size() && digits(numeric[n])) ++n;
            if (n == first) fail("missing exponent digits");
        }
        if (n != numeric.size()) fail("unsupported or malformed value");
        auto converted = token;
        if (converted.front() == '+') converted.remove_prefix(1);
        if (floating) {
            double number = 0;
            const auto r = std::from_chars(converted.data(), converted.data() + converted.size(), number);
            if (r.ec != std::errc{} || r.ptr != converted.data() + converted.size() || !std::isfinite(number))
                fail("floating value out of range");
            return Value{number};
        }
        std::int64_t number = 0;
        const auto r = std::from_chars(converted.data(), converted.data() + converted.size(), number);
        if (r.ec != std::errc{} || r.ptr != converted.data() + converted.size()) fail("integer out of range");
        return Value{number};
    }
};

} // namespace

const Value* ModelDescription::find(std::string_view path) const noexcept {
    const auto* current = &fields;
    while (true) {
        const auto dot = path.find('.');
        const auto entry = current->find(path.substr(0, dot));
        if (entry == current->end()) return nullptr;
        if (dot == std::string_view::npos) return &entry->second;
        current = std::get_if<Value::Table>(&entry->second.data);
        if (!current) return nullptr;
        path.remove_prefix(dot + 1);
    }
}

ModelDescription parse(std::string_view text, TableSyntax syntax) { return Parser(text, syntax).run(); }

ModelDescription load(const std::string& path, TableSyntax syntax) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open model description: " + path);
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("cannot read model description: " + path);
    return parse(text, syntax);
}

} // namespace strix::model_description
