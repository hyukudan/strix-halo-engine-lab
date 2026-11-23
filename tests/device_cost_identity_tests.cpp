// Cost-table device key (strix/device/cost_identity.hpp): the HIP text is byte-identical to the key the Qwen MTP
// cost tables stored before the helper moved to the shared device home, and every timing-relevant fact changes it.
#include "strix/device/cost_identity.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The former private formatter (src/hip_mtp_window_verifier.hip), kept here as the compatibility oracle.
std::string legacy_hip_key(const char* arch, const char* name, int units, std::size_t memory, int clock,
                           int runtime, int driver) {
    return std::string(arch) + '|' + name + "|cu" + std::to_string(units) + "|mem" + std::to_string(memory) +
           "|clk" + std::to_string(clock) + "|hip" + std::to_string(runtime) + "|drv" + std::to_string(driver);
}

}  // namespace

int main() {
    using strix::device::CostDeviceFacts;
    using strix::device::cost_device_identity;

    const CostDeviceFacts hip{"gfx1151", "AMD Radeon Graphics", 40, 137438953472ULL, 2900000, "hip", 70253211, 70253211};
    const auto key = cost_device_identity(hip);
    expect(key == legacy_hip_key("gfx1151", "AMD Radeon Graphics", 40, std::size_t{137438953472ULL}, 2900000, 70253211,
                                 70253211),
           "HIP key equals the stored Qwen cost-table key");
    expect(key == "gfx1151|AMD Radeon Graphics|cu40|mem137438953472|clk2900000|hip70253211|drv70253211",
           "HIP key text");

    const CostDeviceFacts cuda{"sm_120", "Workstation GPU", 188, 102000000000ULL, 2617000, "cuda", 13000, 13000};
    expect(cost_device_identity(cuda) == "sm_120|Workstation GPU|cu188|mem102000000000|clk2617000|cuda13000|drv13000",
           "CUDA key text");

    // Each fact that changes timing changes the key.
    auto changed = hip;
    changed.arch = "gfx1100";
    expect(cost_device_identity(changed) != key, "arch is keyed");
    changed = hip;
    changed.name = "Other GPU";
    expect(cost_device_identity(changed) != key, "device name is keyed");
    changed = hip;
    changed.units = 20;
    expect(cost_device_identity(changed) != key, "units are keyed");
    changed = hip;
    changed.memory = 1;
    expect(cost_device_identity(changed) != key, "memory is keyed");
    changed = hip;
    changed.clock_khz = 1;
    expect(cost_device_identity(changed) != key, "clock is keyed");
    changed = hip;
    changed.runtime = 1;
    expect(cost_device_identity(changed) != key, "runtime is keyed");
    changed = hip;
    changed.driver = 1;
    expect(cost_device_identity(changed) != key, "driver is keyed");
    changed = hip;
    changed.runtime_tag = "cuda";
    expect(cost_device_identity(changed) != key, "backend is keyed");

    if (failures != 0) return EXIT_FAILURE;
    std::puts("device cost identity: PASS");
    return EXIT_SUCCESS;
}
