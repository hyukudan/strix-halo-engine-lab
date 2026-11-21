#include "backend.hpp"

namespace strix::vulkan {

#if defined(STRIX_HAVE_VULKAN)
std::unique_ptr<core::DenseReference> open_native(core::gpu_port::Capabilities*);
#endif

std::unique_ptr<core::DenseReference> open_reference([[maybe_unused]] core::gpu_port::Capabilities* caps) {
#if defined(STRIX_HAVE_VULKAN)
    return open_native(caps);
#else
    core::gpu_port::refuse("Vulkan", "native backend/loader unavailable in this build");
#endif
}

} // namespace strix::vulkan
