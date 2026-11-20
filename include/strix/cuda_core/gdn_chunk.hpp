// Forwarding header: the chunked GDN prefill contract (gdn-chunk16-f32-v1, vendor-free) moved to strix/gdn_chunk.hpp
// (namespace strix::gdn_chunk). Kept until the CUDA core includes the new path.
#pragma once

#include "strix/gdn_chunk.hpp"

namespace strix::cuda_core {
namespace gdn_chunk = ::strix::gdn_chunk;
}  // namespace strix::cuda_core
