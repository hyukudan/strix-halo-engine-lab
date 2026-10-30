#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::deepseek41 {

// Host policy only: resolving this bundle never changes the numerical identity.
struct ServeCachePolicy {
    std::uint64_t prefix_cache_mb = 0, prefix_reserve_mb = 4096, prefix_disk_mb = 0;
    std::uint32_t prefix_block = 256, prefix_inrun_reach = 512;
    std::string prefix_dir, cache_profile = "agent";
    bool prefix_before_tail = false, prefix_session = false, prefix_inrun = false;
    bool set_cache_mb = false, set_reserve = false, set_disk_mb = false, set_dir = false;
    bool set_inrun = false, set_reach = false, set_session = false;
    bool no_cache = false, no_disk = false, no_before_tail = false, no_session = false, no_inrun = false;

    static bool takes_value(std::string_view s) {
        return s == "--prefix-cache-mb" || s == "--prefix-reserve-mb" || s == "--prefix-cache-disk-mb" ||
               s == "--prefix-block" || s == "--prefix-inrun-reach" || s == "--prefix-cache-dir" || s == "--cache-profile";
    }
    bool parse(std::string_view s, std::string_view value = {}) {
        const auto number = [&] (std::uint64_t max) {
            if (value.empty()) throw std::invalid_argument(std::string(s) + " needs a value");
            std::uint64_t n = 0;
            const auto r = std::from_chars(value.data(), value.data() + value.size(), n);
            if (value.empty() || r.ec != std::errc{} || r.ptr != value.data() + value.size() || n > max)
                throw std::invalid_argument(std::string(s) + " needs an unsigned integer in range");
            return n;
        };
        constexpr auto max_mb = std::numeric_limits<std::uint64_t>::max() >> 20U;
        constexpr auto max_u32 = std::numeric_limits<std::uint32_t>::max();
        if (s == "--prefix-cache-mb") {
            prefix_cache_mb = number(max_mb); set_cache_mb = true;
            if (prefix_cache_mb == 0U) no_cache = true;
        } else if (s == "--prefix-reserve-mb") prefix_reserve_mb = number(max_mb), set_reserve = true;
        else if (s == "--prefix-cache-disk-mb") prefix_disk_mb = number(max_mb), set_disk_mb = true;
        else if (s == "--prefix-block") prefix_block = static_cast<std::uint32_t>(number(max_u32));
        else if (s == "--prefix-inrun-reach") prefix_inrun_reach = static_cast<std::uint32_t>(number(max_u32)), set_reach = true;
        else if (s == "--prefix-cache-dir") prefix_dir = value, set_dir = true;
        else if (s == "--cache-profile") {
            if (value != "agent" && value != "off") throw std::invalid_argument("--cache-profile is agent or off");
            cache_profile = value;
        } else if (s == "--prefix-before-tail") prefix_before_tail = true;
        else if (s == "--prefix-session-snapshots") prefix_session = true, set_session = true;
        else if (s == "--prefix-inrun-snapshots") prefix_inrun = true, set_inrun = true;
        else if (s == "--no-prefix-cache") no_cache = true;
        else if (s == "--no-prefix-disk") no_disk = true;
        else if (s == "--no-prefix-before-tail") no_before_tail = true;
        else if (s == "--no-prefix-session-snapshots") no_session = true;
        else if (s == "--no-prefix-inrun-snapshots") no_inrun = true;
        else return false;
        return true;
    }
    void resolve(bool device, bool hetero, bool dspark, std::string_view env_dir = {}, std::string_view home = {}) {
        if (cache_profile == "agent" && device && !hetero && !no_cache) {
            if (!set_cache_mb) prefix_cache_mb = 512;
            if (prefix_cache_mb != 0U) {
                if (!set_reserve) prefix_reserve_mb = 8192;
                if (!set_inrun) prefix_inrun = true;
                if (!set_reach) prefix_inrun_reach = 0;
                if (!set_session && !dspark) prefix_session = true;
            }
        }
        // A numeric disk limit is an explicit opt-in using the historical path resolution.
        if (set_disk_mb && !set_dir && !no_disk && !no_cache) {
            if (!env_dir.empty()) prefix_dir = env_dir;
            else if (!home.empty()) prefix_dir = std::string(home) + "/.cache/strix/ds41-prefix";
        }
        if (!prefix_dir.empty() && !set_disk_mb) prefix_disk_mb = 4096;
        if (no_cache) {
            prefix_cache_mb = prefix_disk_mb = 0;
            prefix_dir.clear();
            prefix_before_tail = prefix_session = prefix_inrun = false;
        }
        if (no_disk) prefix_dir.clear(), prefix_disk_mb = 0;
        if (no_before_tail) prefix_before_tail = false;
        if (no_session) prefix_session = false;
        if (no_inrun) prefix_inrun = false;
    }
};

// Subtraction avoids overflow; unknown availability fails closed even with a zero reserve.
inline bool prefix_memory_admitted(std::uint64_t available_mb, std::uint64_t reserve_mb, std::uint64_t predicted_bytes) {
    const auto mb = predicted_bytes / (1U << 20U) + (predicted_bytes % (1U << 20U) != 0U ? 1U : 0U);
    return available_mb != 0U && available_mb >= reserve_mb && available_mb - reserve_mb >= mb;
}

}  // namespace strix::deepseek41
