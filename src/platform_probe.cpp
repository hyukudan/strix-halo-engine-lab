#include "strix/platform_probe.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace strix {
namespace {

std::filesystem::path rooted(const std::filesystem::path& root,
                             std::string_view relative) {
    return root / std::filesystem::path(relative);
}

std::string trim(std::string value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    std::size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }
    return value.substr(first);
}

std::optional<std::string> read_first_line(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    if (!input || !std::getline(input, line)) {
        return std::nullopt;
    }
    line = trim(std::move(line));
    if (line.empty()) {
        return std::nullopt;
    }
    return line;
}

std::optional<std::int64_t> parse_i64(std::string_view text) noexcept {
    std::int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::uint64_t> parse_u64(std::string_view text) noexcept {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::uint64_t mem_total(const std::filesystem::path& root) {
    std::ifstream input(rooted(root, "proc/meminfo"));
    std::string key;
    std::string value_text;
    std::string unit;
    while (input >> key >> value_text >> unit) {
        if (key != "MemTotal:") {
            continue;
        }
        const auto value = parse_u64(value_text);
        if (!value.has_value() || unit != "kB" ||
            *value > std::numeric_limits<std::uint64_t>::max() / 1024ULL) {
            return 0;
        }
        return *value * 1024ULL;
    }
    return 0;
}

std::map<std::string, std::uint64_t> read_properties(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::map<std::string, std::uint64_t> result;
    std::string key;
    std::string value;
    while (input >> key >> value) {
        if (const auto parsed = parse_u64(value); parsed.has_value()) {
            result[key] = *parsed;
        }
    }
    return result;
}

std::optional<std::uint64_t> property(const std::map<std::string, std::uint64_t>& values,
                                      std::string_view key) {
    const auto found = values.find(std::string(key));
    if (found == values.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::string gfx_arch(std::uint64_t target) {
    const auto major = target / 10'000ULL;
    const auto minor = (target / 100ULL) % 100ULL;
    const auto stepping = target % 100ULL;
    return "gfx" + std::to_string(major) + std::to_string(minor) +
           std::to_string(stepping);
}

bool all_digits(std::string_view value) noexcept {
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](char character) {
               return character >= '0' && character <= '9';
           });
}

bool is_host_router(std::string_view name) noexcept {
    const auto dash = name.find('-');
    return dash != std::string_view::npos &&
           name.find('-', dash + 1) == std::string_view::npos &&
           name.substr(dash + 1) == "0" && all_digits(name.substr(0, dash));
}

std::vector<std::string> entries_matching(const std::filesystem::path& path,
                                          bool (*predicate)(std::string_view)) {
    std::vector<std::string> result;
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(path, error), end;
         iterator != end && !error; iterator.increment(error)) {
        const auto name = iterator->path().filename().string();
        if (predicate(name)) {
            result.push_back(name);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool is_domain(std::string_view name) noexcept {
    return name.size() > 6 && name.substr(0, 6) == "domain" &&
           all_digits(name.substr(6));
}

bool is_tbstream(std::string_view name) noexcept {
    return name.size() > 8 && name.substr(0, 8) == "tbstream" &&
           all_digits(name.substr(8));
}

std::optional<std::string> rocm_version(const std::filesystem::path& root) {
    const std::filesystem::path candidates[] = {
        rooted(root, "opt/rocm/.info/version"),
        rooted(root, "opt/rocm/.info/version-dev"),
        rooted(root, "opt/rocm/share/rocm/.info/version"),
    };
    for (const auto& candidate : candidates) {
        if (const auto version = read_first_line(candidate); version.has_value()) {
            return version;
        }
    }
    return std::nullopt;
}


std::vector<Usb4StreamInfo> configured_streams(const std::filesystem::path& root) {
    std::vector<Usb4StreamInfo> streams;
    const auto base = rooted(root, "sys/kernel/config/thunderbolt/stream");
    std::error_code error;
    for (std::filesystem::directory_iterator service_it(base, error), end;
         service_it != end && !error; service_it.increment(error)) {
        if (!service_it->is_directory(error)) {
            if (error) {
                error.clear();
            }
            continue;
        }
        const auto service = service_it->path().filename().string();
        std::error_code child_error;
        for (std::filesystem::directory_iterator stream_it(service_it->path(), child_error), child_end;
             stream_it != child_end && !child_error; stream_it.increment(child_error)) {
            if (!stream_it->is_directory(child_error)) {
                if (child_error) {
                    child_error.clear();
                }
                continue;
            }

            Usb4StreamInfo info;
            info.service = service;
            info.name = stream_it->path().filename().string();

            if (const auto value = read_first_line(stream_it->path() / "index"); value.has_value()) {
                info.index = parse_u64(*value);
            }
            if (const auto value = read_first_line(stream_it->path() / "busy_poll"); value.has_value()) {
                if (*value == "0") info.busy_poll = false;
                if (*value == "1") info.busy_poll = true;
            }
            if (const auto value = read_first_line(stream_it->path() / "ring_size"); value.has_value()) {
                info.ring_size = parse_u64(*value);
            }
            if (const auto value = read_first_line(stream_it->path() / "throttling"); value.has_value()) {
                info.throttling_ns = parse_u64(*value);
            }
            if (const auto value = read_first_line(stream_it->path() / "in_hopid"); value.has_value()) {
                info.in_hopid = parse_i64(*value);
            }
            if (const auto value = read_first_line(stream_it->path() / "out_hopid"); value.has_value()) {
                info.out_hopid = parse_i64(*value);
            }
            streams.push_back(std::move(info));
        }
    }
    std::sort(streams.begin(), streams.end(), [](const auto& left, const auto& right) {
        if (left.service != right.service) return left.service < right.service;
        return left.name < right.name;
    });
    return streams;
}

std::vector<GpuNodeInfo> gpu_nodes(const std::filesystem::path& root) {
    std::vector<GpuNodeInfo> result;
    const auto nodes = rooted(root, "sys/class/kfd/kfd/topology/nodes");
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(nodes, error), end;
         iterator != end && !error; iterator.increment(error)) {
        const bool directory = iterator->is_directory(error);
        if (error) {
            error.clear();
            continue;
        }
        if (!directory) {
            continue;
        }
        const auto props = read_properties(iterator->path() / "properties");
        const auto target = property(props, "gfx_target_version");
        if (!target.has_value() || *target == 0) {
            continue;
        }

        GpuNodeInfo node;
        node.node = iterator->path().filename().string();
        node.gfx_target_version = target;
        node.gfx_arch = gfx_arch(*target);
        node.simd_count = property(props, "simd_count");
        node.simd_per_cu = property(props, "simd_per_cu");
        if (node.simd_count.has_value() && node.simd_per_cu.has_value() &&
            *node.simd_per_cu != 0 && *node.simd_count % *node.simd_per_cu == 0) {
            node.compute_units = *node.simd_count / *node.simd_per_cu;
        }
        result.push_back(std::move(node));
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.node < right.node;
    });
    return result;
}

}  // namespace

PlatformInfo probe_platform(const std::filesystem::path& root) {
    PlatformInfo info;
#if defined(__linux__)
    info.linux = true;
    if (root == std::filesystem::path{"/"}) {
        utsname name{};
        if (uname(&name) == 0) {
            info.kernel_release = name.release;
        }
    }
#endif
    info.mem_total_bytes = mem_total(root);
    info.rocm_version = rocm_version(root);
    info.hipcc_present = std::filesystem::exists(rooted(root, "opt/rocm/bin/hipcc"));
    info.gpu_nodes = gpu_nodes(root);

    const auto thunderbolt = rooted(root, "sys/bus/thunderbolt/devices");
    info.usb4.domains = entries_matching(thunderbolt, is_domain);
    info.usb4.host_routers = entries_matching(thunderbolt, is_host_router);
    info.usb4.stream_devices = entries_matching(rooted(root, "dev"), is_tbstream);
    info.usb4.configured_streams = configured_streams(root);
    info.usb4.stream_driver_present =
        std::filesystem::exists(rooted(root, "sys/module/thunderbolt_stream")) ||
        std::filesystem::exists(rooted(root, "sys/bus/thunderbolt/drivers/thunderbolt_stream"));

    return info;
}

}  // namespace strix
