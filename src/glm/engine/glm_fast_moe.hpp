#pragma once
// GLM engine: MoE seam on the decode kernels (src/glm/moe, review finding; bitwise vs the
// strix_gemv chain + Tier A). TP2 rank slices only (I = 1024); default for world 2, STRIX_GLM_MOE=ref = DefaultMoe.
// Experts and the shared expert are loaded raw-only (file layout in the engine arena) when this seam is active.
#include "strix/glm/engine.hpp"

#include <memory>

namespace strix::glm::engine {

std::shared_ptr<MoePath> fast_moe();

}  // namespace strix::glm::engine
