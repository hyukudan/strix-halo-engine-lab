#pragma once

namespace strix::cuda::arch {

// Architecture-specific CUDA images have exact compute-capability matching.
struct NativeNvfp4Images {
    bool sm120 = false;
    bool sm121 = false;

    [[nodiscard]] constexpr bool supports(int major, int minor) const noexcept {
        return major == 12 && ((minor == 0 && sm120) || (minor == 1 && sm121));
    }
};

}  // namespace strix::cuda::arch
