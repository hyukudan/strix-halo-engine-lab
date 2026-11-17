#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strix::deploy {

// Device-usable bytes, not installed system RAM. Missing observations stay unknown.
struct DeviceFacts {
    std::string gfx_target;
    std::uint32_t cu_count = 0;
    std::uint64_t mem_total_bytes = 0, mem_free_bytes = 0;
    bool unified = false;
    std::string source = "unavailable";
    std::uint64_t gfx_target_version = 0;
    std::optional<std::uint64_t> gtt_total_bytes, gtt_used_bytes;
    std::optional<std::uint64_t> vram_total_bytes, vram_used_bytes;
    std::string drm_device; // Local observation path; deliberately omitted from plan JSON.
};

// root is a filesystem prefix containing sys/class/..., not the sys directory.
// A DRM device override preserves callers' explicit selection (including fixtures).
[[nodiscard]] std::vector<DeviceFacts> probe_devices(const std::string& sysfs_root = "",
                                                     const std::string& drm_device = "");
// A declared device budget (strix/device/budget.hpp) over observed facts: total = the budget, free = what the budget
// leaves after what is already in use. 0 or a budget at/above the device leaves the facts unchanged.
void apply_budget(DeviceFacts& facts, std::uint64_t device_bytes);
// Refuses ambiguous selection by returning unknown facts. Never silently selects GPU 0. A single device carries the
// bound device's declared budget (strix::device::process_budget(), set by the start gate).
[[nodiscard]] DeviceFacts probe_device(const std::string& sysfs_root = "",
                                      const std::string& drm_device = "");
// Existing GLM card-first selection, centralized here for its legacy telemetry contract.
[[nodiscard]] std::string first_drm_device(const std::string& sysfs_root = "");

} // namespace strix::deploy
