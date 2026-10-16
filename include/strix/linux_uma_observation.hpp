#pragma once
#include "strix/uma_owner_inventory.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace strix {
enum class LinuxUmaReadStatus : std::uint8_t { unavailable, read, malformed, oversized, unsupported };
struct LinuxUmaObservation {
    UmaMemoryObservation memory;
    bool supported = false;
    LinuxUmaReadStatus meminfo_status = LinuxUmaReadStatus::unavailable;
    LinuxUmaReadStatus process_status = LinuxUmaReadStatus::unavailable;
    std::uint64_t begin_steady_ns = 0, end_steady_ns = 0;
};
// Pure bounded parser for source fixtures/collectors. Values are reported kernel
// counters in bytes, not PFN residency, free-engine budget or proof of physical
// fit. id must identify a real read/fixture observation; no caller-value fallback.
[[nodiscard]] LinuxUmaObservation parse_linux_uma_observation(std::string_view meminfo,
    std::string_view process_status,std::uint64_t observation_identity);
// Setup-only, opt-in. Reads at most64KiB each from proc_root/meminfo and
// proc_root/self/status. Unsupported/unreadable fields remain unknown. The two
// files are sampled over an explicit interval, not an atomic system snapshot.
[[nodiscard]] LinuxUmaObservation observe_linux_uma_memory(
    const std::filesystem::path& proc_root = "/proc");
[[nodiscard]] std::string linux_uma_observation_json(const LinuxUmaObservation&);
} // namespace strix
