#include "backend.hpp"

namespace strix::metal {

#if defined(STRIX_HAVE_METAL)
std::unique_ptr<core::DenseReference> open_native(core::gpu_port::Capabilities*);
#endif

std::unique_ptr<core::DenseReference> open_reference([[maybe_unused]] core::gpu_port::Capabilities* caps) {
#if defined(STRIX_HAVE_METAL)
    return open_native(caps);
#else
    core::gpu_port::refuse("Metal", "native backend unavailable in this build");
#endif
}

} // namespace strix::metal
