// Forwarding header: the resident expanded-operand cache planner (pure host code) moved to the shared weight layer,
// strix/weight/operand_cache_plan.hpp (namespace strix::weight::operand_cache). Kept until the CUDA dense core includes
// the new path.
#pragma once

#include "strix/weight/operand_cache_plan.hpp"

namespace strix::cuda_core {
namespace operand_cache = ::strix::weight::operand_cache;
}  // namespace strix::cuda_core
