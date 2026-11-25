#include "strix/hip/legacy_launch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>

namespace {
using hipStream_t = void*;

struct dim3 { unsigned x; };

unsigned launches = 0;

}  // namespace

namespace strix::core::controller_rms_rows::detail {
template <unsigned Units, unsigned Threads, class Args>
void submit(dim3, hipStream_t, const Args&) {
    ++launches;
}
}  // namespace strix::core::controller_rms_rows::detail

#include "controller_rms_launch_host.inc"

int main() {
    namespace crr = strix::core::controller_rms_rows;
    constexpr unsigned rows = 32U, width = 272U, rank = 16U;
    alignas(16) static float latent[rows * rank], current[rows * width], message[rows * width], output[rows * width];
    alignas(16) static std::uint16_t forget[width * rank], gain[width * rank], scale[width];
    const crr::Args args{latent, forget, gain, current, message, scale, output, rows, width, rank, 1e-6F};
    if (!crr::admits(args, 16U)) {
        std::puts("FAIL: fixture shape must pass argument admission");
        return 1;
    }

    hip_archguard_test::active_device = 2;  // gfx1201, wave32
    bool refused = false;
    try {
        (void)crr::enqueue(args, nullptr);
    } catch (const std::runtime_error& error) {
        refused = std::string_view(error.what()).find("HIP launch refused on gfx1201: requires RDNA3/RDNA3.5") == 0U;
    }
    if (!refused || launches != 0U) {
        std::printf("FAIL: gfx1201 refusal=%d submissions=%u (expected refusal with zero submissions)\n", refused,
                    launches);
        return 1;
    }

    hip_archguard_test::active_device = 0;  // gfx1151, wave32
    if (crr::enqueue(args, nullptr) != hipSuccess || launches != 1U) {
        std::puts("FAIL: gfx1151 should submit exactly one kernel for the same shape");
        return 1;
    }
    std::puts("PASS: gfx1201 refused before submission; gfx1151 submitted once");
    return 0;
}
