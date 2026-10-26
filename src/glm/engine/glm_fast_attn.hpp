#pragma once
// GLM engine: attention seam on the fast decode kernels (glm_fast_attn.hip). The engine's default since the attention
// integration (exact: same bits as DefaultAttn); STRIX_GLM_ATTN=ref selects DefaultAttn.
#include "strix/glm/engine.hpp"

#include <cstdint>
#include <memory>

namespace strix::glm::engine {

std::shared_ptr<AttnPath> fast_attention(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity);

}  // namespace strix::glm::engine
