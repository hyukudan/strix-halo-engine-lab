#include "inkling2_mini_serve_device.hpp"
#include "strix/device/cuda_enumerator.hpp"

namespace strix::inkling2 {

std::unique_ptr<serve::CohortBackend> make_mini_cuda_cohort(const MiniServeOptions& opt) {
    const auto& bound = device::bind_cuda(mini_serve_device_request(opt), opt.devices_path);
    // Stay within one shared dense decode window for the entire cohort.
    return make_mini_device_cohort(opt, bound.budget, "CUDA", cuda_core::kMaxRows);
}

} // namespace strix::inkling2
