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

struct Usb4StreamInfo {
    std::string service;
    std::string name;
    std::optional<std::uint64_t> index;
    std::optional<bool> busy_poll;
    std::optional<std::uint64_t> ring_size;
    std::optional<std::uint64_t> throttling_ns;
    std::optional<std::int64_t> in_hopid;
    std::optional<std::int64_t> out_hopid;
    std::optional<std::string> device_path;
    std::optional<bool> device_is_character;
    std::optional<std::string> device_sysfs_route;
    bool read_write_candidate = false;
};

struct Usb4Topology {
    std::vector<std::string> domains;
    std::vector<std::string> host_routers;
    std::vector<std::string> stream_devices;
    std::vector<Usb4StreamInfo> configured_streams;
    bool stream_driver_present = false;
    bool dma_buf_import_gate_present = false;
    std::optional<bool> dma_buf_import_gate_enabled;
    std::optional<std::string> stream_driver_version;
    std::optional<std::string> stream_driver_srcversion;
    // No supported native capability-query implementation exists yet. Static
    // module/gate/device observations never establish an ABI or GPU proof.
    bool native_capability_query_available = false;
};

struct PlatformInfo {
    bool is_linux = false;
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
