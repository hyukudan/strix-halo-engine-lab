#include "strix/platform_probe.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& value) {
    std::string result;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(character));
        } else if (character >= 0x20U && character < 0x80U) {
            result.push_back(static_cast<char>(character));
        } else {
            std::ostringstream escaped;
            escaped << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
            result += escaped.str();
        }
    }
    return result;
}

void print_string_array(const std::vector<std::string>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << '"' << json_escape(values[index]) << '"';
    }
    std::cout << ']';
}

}  // namespace

int main() {
    const auto info = strix::probe_platform();

    std::cout << "{\n"
              << "  \"linux\": " << (info.is_linux ? "true" : "false") << ",\n"
              << "  \"kernel_release\": \"" << json_escape(info.kernel_release) << "\",\n"
              << "  \"mem_total_bytes\": " << info.mem_total_bytes << ",\n"
              << "  \"rocm_version\": ";
    if (info.rocm_version.has_value()) {
        std::cout << '"' << json_escape(*info.rocm_version) << '"';
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
              << "  \"hipcc_present\": " << (info.hipcc_present ? "true" : "false") << ",\n"
              << "  \"gpu_nodes\": [\n";

    for (std::size_t index = 0; index < info.gpu_nodes.size(); ++index) {
        const auto& gpu = info.gpu_nodes[index];
        std::cout << "    {\"node\":\"" << json_escape(gpu.node)
                  << "\",\"gfx_target_version\":";
        if (gpu.gfx_target_version.has_value()) {
            std::cout << *gpu.gfx_target_version;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"gfx_arch\":\"" << json_escape(gpu.gfx_arch)
                  << "\",\"simd_count\":";
        if (gpu.simd_count.has_value()) {
            std::cout << *gpu.simd_count;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"simd_per_cu\":";
        if (gpu.simd_per_cu.has_value()) {
            std::cout << *gpu.simd_per_cu;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"compute_units\":";
        if (gpu.compute_units.has_value()) {
            std::cout << *gpu.compute_units;
        } else {
            std::cout << "null";
        }
        std::cout << '}' << (index + 1 == info.gpu_nodes.size() ? '\n' : ',') << '\n';
    }

    std::cout << "  ],\n"
              << "  \"usb4\": {\n"
              << "    \"domains\": ";
    print_string_array(info.usb4.domains);
    std::cout << ",\n    \"host_routers\": ";
    print_string_array(info.usb4.host_routers);
    std::cout << ",\n    \"stream_devices\": ";
    print_string_array(info.usb4.stream_devices);
    std::cout << ",\n    \"configured_streams\": [\n";
    for (std::size_t index = 0; index < info.usb4.configured_streams.size(); ++index) {
        const auto& stream = info.usb4.configured_streams[index];
        std::cout << "      {\"service\":\"" << json_escape(stream.service)
                  << "\",\"name\":\"" << json_escape(stream.name) << "\",\"index\":";
        if (stream.index.has_value()) std::cout << *stream.index; else std::cout << "null";
        std::cout << ",\"busy_poll\":";
        if (stream.busy_poll.has_value()) std::cout << (*stream.busy_poll ? "true" : "false"); else std::cout << "null";
        std::cout << ",\"ring_size\":";
        if (stream.ring_size.has_value()) std::cout << *stream.ring_size; else std::cout << "null";
        std::cout << ",\"throttling_ns\":";
        if (stream.throttling_ns.has_value()) std::cout << *stream.throttling_ns; else std::cout << "null";
        std::cout << ",\"in_hopid\":";
        if (stream.in_hopid.has_value()) std::cout << *stream.in_hopid; else std::cout << "null";
        std::cout << ",\"out_hopid\":";
        if (stream.out_hopid.has_value()) std::cout << *stream.out_hopid; else std::cout << "null";
        std::cout << '}' << (index + 1 == info.usb4.configured_streams.size() ? '\n' : ',') << '\n';
    }
    std::cout << "    ]";
    std::cout << ",\n    \"stream_driver_present\": "
              << (info.usb4.stream_driver_present ? "true" : "false")
              << ",\n    \"dma_buf_import_gate_present\": "
              << (info.usb4.dma_buf_import_gate_present ? "true" : "false")
              << ",\n    \"dma_buf_import_gate_enabled\": ";
    if (info.usb4.dma_buf_import_gate_enabled.has_value()) {
        std::cout << (*info.usb4.dma_buf_import_gate_enabled ? "true" : "false");
    } else {
        std::cout << "null";
    }
    std::cout << "\n  }\n}\n";
    return 0;
}
