#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace strix::model_description {

// Dependency-free TOML subset: bare/dotted keys, tables, inline tables, arrays,
// single-line basic/literal strings, decimal int64/double and booleans. Unsupported
// syntax is an error. Schema, required fields and model semantics belong to adapters.
struct Value {
    using Array = std::vector<Value>;
    using Table = std::map<std::string, Value, std::less<>>;
    using Data = std::variant<std::int64_t, double, bool, std::string, Array, Table>;
    Data data;
    friend bool operator==(const Value&, const Value&) = default;
};

struct ModelDescription {
    Value::Table fields;
    // Bare-key dotted path; nullptr for an absent path or a non-table parent.
    [[nodiscard]] const Value* find(std::string_view path) const noexcept;
};

// Model launch metadata is consumed by the launcher. Family validators still
// validate every field of their compiled tensor and arithmetic contracts.
[[nodiscard]] constexpr bool is_launch_metadata(std::string_view root_key) noexcept {
    return root_key == "presets" || root_key == "weights" || root_key == "drafter";
}

// Throws std::invalid_argument (with line/column) for malformed, duplicate or
// unsupported input. load additionally throws std::runtime_error for I/O errors.
// Root array tables opt in to repeated [[name]] headers with fields relative to each row.
enum class TableSyntax { standard, root_array_tables };
[[nodiscard]] ModelDescription parse(std::string_view text, TableSyntax syntax = TableSyntax::standard);
[[nodiscard]] ModelDescription load(const std::string& path, TableSyntax syntax = TableSyntax::standard);

} // namespace strix::model_description
