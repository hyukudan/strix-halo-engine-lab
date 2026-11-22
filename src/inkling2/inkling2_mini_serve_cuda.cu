#include "inkling2_mini_serve_device.hpp"
#include "strix/device/cuda_enumerator.hpp"

namespace strix::inkling2 {

MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions& opt) {
    // bind_cuda returns a reference to the process-owned binding.  Materialize
    // the budget rather than retain a reference through the temporary request:
    // strict GCC 15 / nvcc rejects the latter under -Wdangling-reference.
    const auto budget = device::bind_cuda(mini_serve_device_request(opt), opt.devices_path).budget;
    // Stay within one shared dense decode window for the entire cohort.
    return make_mini_device_cohort(opt, budget, "CUDA", cuda_core::kMaxRows);
}

} // namespace strix::inkling2
