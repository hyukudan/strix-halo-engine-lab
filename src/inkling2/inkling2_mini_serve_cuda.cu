#include "inkling2_mini_serve_device.hpp"
#include "strix/device/cuda_enumerator.hpp"

namespace strix::inkling2 {

MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions& opt) {
    // Copy the budget out: binding a reference to a call that takes a temporary request trips
    // GCC -Wdangling-reference, although bind_cuda returns process-lifetime state.
    const auto budget = device::bind_cuda(mini_serve_device_request(opt), opt.devices_path).budget;
    // Stay within one shared dense decode window for the entire cohort.
    return make_mini_device_cohort(opt, budget, "CUDA", cuda_core::kMaxRows);
}

} // namespace strix::inkling2
