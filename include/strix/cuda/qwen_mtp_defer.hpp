// Forwarding header: the deferred prompt MTP rows policy moved to the shared speculative layer,
// strix/spec/mtp_defer.hpp (namespace strix::spec::mtp_defer). Kept until the CUDA Qwen driver includes the new path.
#pragma once

#include "strix/spec/mtp_defer.hpp"

namespace strix::cuda {
namespace qwen_mtp_defer = ::strix::spec::mtp_defer;
}  // namespace strix::cuda
