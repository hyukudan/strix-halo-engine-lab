#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2_mini_serve_device.hpp"
#include "strix/device/hip_gate.hpp"

namespace strix::inkling2 {

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions& opt) {
    const auto& bound = device::bind_hip(mini_serve_device_request(opt), opt.devices_path);
    return make_mini_device_cohort(opt, bound.budget, "HIP");
}

} // namespace strix::inkling2
