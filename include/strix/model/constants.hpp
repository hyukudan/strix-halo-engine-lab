#pragma once

// Model constants derived from the model TOML (host only, no SDK types).
//
// `models/<id>.toml` is the single source of a model's dimensions and numeric constants. Compile-time consumers
// (kernel shapes, device numerics, static plans) read the generated header include/strix/model_data/<ident>.hpp and
// bind their own typed names with strix/model/bind.hpp; host code reads the TOML through the shared readers
// (strix/model_description.hpp, or strix/graph/model_decl.hpp for strix-model/v2 executable declarations).
// The generated headers are committed; strix-model-constants-tests regenerates every registry model and fails on
// any byte of drift, so a TOML edit without regeneration (or a hand edit of a header) cannot merge.

#include "strix/model_description.hpp"

#include <string>
#include <string_view>

namespace strix::model {

// "nemotron-3.5-bolt-e4b" -> "nemotron_3_5_bolt_e4b" (the C++ namespace of a model's constants).
[[nodiscard]] std::string identifier(std::string_view model_name);

// The generated header: every scalar and every homogeneous scalar array of the description becomes an
// `inline constexpr` in namespace strix::model_data::<identifier(name)>[::<table>...]. Integers are std::int64_t,
// reals are double written as exact hexadecimal literals, booleans bool, strings std::string_view. An array of
// tables whose rows have the same keys and one scalar type per key is emitted structure-of-arrays:
// `namespace <key> { std::size_t count; <type> <field>[count]...; }`. Other arrays of tables and mixed arrays are
// skipped with a comment. `source_path` is written into the banner only.
[[nodiscard]] std::string emit_constants(const model_description::ModelDescription& description,
                                         std::string_view source_path);

} // namespace strix::model
