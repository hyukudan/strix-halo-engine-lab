// Forwarding header: the row-group union moved to the vendor-free shared attention core, strix/attn_rowgroup_union.hpp
// (namespace strix::attn::rowgroup). The CUDA row-group kernels (attention_rowgroup.cuh) live in this namespace and use
// the same functions through these declarations; kept until they name the shared namespace.
#pragma once

#include "strix/attn_rowgroup_union.hpp"

namespace strix::cuda::core::rowgroup {
using ::strix::attn::rowgroup::ascending;
using ::strix::attn::rowgroup::contains;
using ::strix::attn::rowgroup::lower_bound;
using ::strix::attn::rowgroup::owned;
using ::strix::attn::rowgroup::rank_and_mask;
}  // namespace strix::cuda::core::rowgroup
