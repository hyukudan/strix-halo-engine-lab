#pragma once

#include "strix/execution_runtime.hpp"

#include <cstddef>
#include <string_view>

namespace strix {

[[nodiscard]] ExecutionBufferAddress resolve_dispatch_read(
    const ExecutionDispatchContext& context,
    std::size_t index);

[[nodiscard]] ExecutionBufferAddress resolve_dispatch_write(
    const ExecutionDispatchContext& context,
    std::size_t index);

[[nodiscard]] SessionStateAddress resolve_dispatch_state(
    const ExecutionDispatchContext& context,
    SessionStateComponent component);

[[nodiscard]] const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::uint32_t tensor_index);

[[nodiscard]] const RuntimeTensorBinding& resolve_dispatch_tensor(
    const ExecutionDispatchContext& context,
    std::string_view tensor_name);

}  // namespace strix
