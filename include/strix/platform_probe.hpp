#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace strix {

struct GpuNodeInfo {
    std::string node;
    std::optional<std::uint64_t> gfx_target_version;
    std::string gfx_arch;
    std::optional<std::uint64_t> simd_count;
    std::optional<std::uint64_t> simd_per_cu;
    std::optional<std::uint64_t> compute_units;
};

struct Usb4Topology {
    std::vector<std::string> domains;
    std::vector<std::string> host_routers;
    std::vector<std::string> stream_devices;
    bool stream_driver_present = false;
};

struct PlatformInfo {
    bool linux = false;
    std::string kernel_release;
    std::uint64_t mem_total_bytes = 0;
    std::optional<std::string> rocm_version;
    bool hipcc_present = false;
    std::vector<GpuNodeInfo> gpu_nodes;
    Usb4Topology usb4;
};

[[nodiscard]] PlatformInfo probe_platform(
    const std::filesystem::path& root = std::filesystem::path{"/"});

}  // namespace strix
