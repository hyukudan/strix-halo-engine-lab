#pragma once

#include "core/gpu_port.hpp"

namespace strix::metal {

std::unique_ptr<core::DenseReference> open_reference(core::gpu_port::Capabilities* = nullptr);

} // namespace strix::metal
