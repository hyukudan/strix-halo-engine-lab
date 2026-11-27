#pragma once

#include <cstddef>
#include <functional>

namespace strix::attn {

using ReferenceHeadRange = std::function<void(std::size_t, std::size_t)>;
using ReferenceHeadExecutor = std::function<void(std::size_t, const ReferenceHeadRange&)>;

} // namespace strix::attn
