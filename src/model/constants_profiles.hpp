#pragma once

#include "strix/model/constants.hpp"

#include <span>

namespace strix::model {

// The caller supplies the accepted profile lever names for this description.
// With an empty list, any key in profile.fast or profile.turbo is unknown.
[[nodiscard]] std::string emit_constants(const model_description::ModelDescription& description,
                                         std::string_view source_path,
                                         std::span<const std::string_view> lever_names);

} // namespace strix::model
