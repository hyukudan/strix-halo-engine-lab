#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

// Small JSON value for the HTTP API: objects keep their key order (requests are echoed into the
// chat template in the order the client wrote them), numbers keep their source text so that they
// are re-emitted unchanged.
class Json {
public:
    enum class Kind : std::uint8_t { null, boolean, number, string, array, object };
    using Array = std::vector<Json>;
    using Member = std::pair<std::string, Json>;
    using Object = std::vector<Member>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool value) : kind_(Kind::boolean), boolean_(value) {}
    Json(int value) : Json(static_cast<std::int64_t>(value)) {}
    Json(std::int64_t value);
    Json(std::uint64_t value);
    Json(double value);
    Json(const char* value) : kind_(Kind::string), text_(value) {}
    Json(std::string value) : kind_(Kind::string), text_(std::move(value)) {}
    Json(std::string_view value) : kind_(Kind::string), text_(value) {}
    Json(Array value) : kind_(Kind::array), array_(std::make_shared<Array>(std::move(value))) {}
    Json(Object value) : kind_(Kind::object), object_(std::make_shared<Object>(std::move(value))) {}
    static Json number_text(std::string text);
    static Json array() { return Json(Array{}); }
    static Json object() { return Json(Object{}); }

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_null() const noexcept { return kind_ == Kind::null; }
    [[nodiscard]] bool is_bool() const noexcept { return kind_ == Kind::boolean; }
    [[nodiscard]] bool is_number() const noexcept { return kind_ == Kind::number; }
    [[nodiscard]] bool is_string() const noexcept { return kind_ == Kind::string; }
    [[nodiscard]] bool is_array() const noexcept { return kind_ == Kind::array; }
    [[nodiscard]] bool is_object() const noexcept { return kind_ == Kind::object; }

    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] std::int64_t as_int() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const std::string& number_source() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] Array& as_array();
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] Object& as_object();

    // Object access: nullptr when absent or not an object.
    [[nodiscard]] const Json* find(std::string_view key) const;
    // Appends or replaces a member, keeping the first position of an existing key.
    Json& set(std::string key, Json value);
    void push(Json value);
    [[nodiscard]] std::size_t size() const;

    // Compact form: {"a":1,"b":[1,2]} without spaces, non-ASCII kept as UTF-8.
    [[nodiscard]] std::string dump() const;
    // The template filter form: {"a": 1, "b": [1, 2]} (", " and ": " separators), non-ASCII kept.
    [[nodiscard]] std::string dump_spaced() const;

private:
    void write(std::string& out, bool spaced) const;
    Kind kind_ = Kind::null;
    bool boolean_ = false;
    std::string text_; // string value or number source text
    std::shared_ptr<Array> array_;
    std::shared_ptr<Object> object_;
};

class JsonError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Strict RFC 8259 parser (UTF-8 input, surrogate pairs combined); depth limited to 128.
[[nodiscard]] Json parse_json(std::string_view text);
void append_json_string(std::string& out, std::string_view value);

}  // namespace strix::serve
