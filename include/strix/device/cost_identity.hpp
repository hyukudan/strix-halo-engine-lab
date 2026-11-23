#pragma once

#include <string>

// Device key of a persisted cost table (speculation verify walls, MTP cost tables). Walls measured on one
// device and runtime are reused only by the same device and runtime, so the key names what changes the
// timing: architecture, product name, compute units, memory size, clock, runtime and driver versions.
// Backend-neutral: the HIP and CUDA queries (hip_cost_identity.hpp, cuda_cost_identity.hpp) only fill the facts.
namespace strix::device {

struct CostDeviceFacts {
    std::string arch;         // HIP gcnArchName, CUDA "sm_<major><minor>"
    std::string name;         // product name
    long long units = 0;      // compute units / streaming multiprocessors
    unsigned long long memory = 0;  // device memory, bytes
    long long clock_khz = 0;  // peak core clock
    std::string runtime_tag;  // "hip" / "cuda"
    long long runtime = 0;    // runtime version
    long long driver = 0;     // driver version
};

// `arch|name|cu<units>|mem<bytes>|clk<kHz>|<runtime_tag><runtime>|drv<driver>`. The HIP text is the key the
// Qwen MTP cost tables already store, so existing tables stay valid.
[[nodiscard]] inline std::string cost_device_identity(const CostDeviceFacts& facts) {
    return facts.arch + '|' + facts.name + "|cu" + std::to_string(facts.units) + "|mem" +
           std::to_string(facts.memory) + "|clk" + std::to_string(facts.clock_khz) + '|' + facts.runtime_tag +
           std::to_string(facts.runtime) + "|drv" + std::to_string(facts.driver);
}

}  // namespace strix::device
