#include "strix/device_profile.hpp"
#include "strix/hip/arch.hpp"

#include <cstdlib>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>

#if defined(STRIX_DEVICE_PROFILE_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace strix {

DeviceIsaCapabilities device_isa_capabilities(std::string_view arch) {
    const auto family = hip::arch::family_from_name(arch);
    const bool rdna35 = family == hip::arch::Family::Rdna35;
    const bool legacy = rdna35 || family == hip::arch::Family::Rdna3;
    return {rdna35, legacy, legacy, legacy};
}

DeviceProfile probe_device_profile() {
#if defined(STRIX_DEVICE_PROFILE_HIP)
    const auto check = [](hipError_t status, const char* operation) {
        if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    };
    DeviceProfile profile;
    check(hipGetDevice(&profile.device_index), "device profile: current device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, profile.device_index), "device profile: properties");
    std::size_t free_bytes = 0, total_bytes = 0;
    check(hipMemGetInfo(&free_bytes, &total_bytes), "device profile: memory");
    profile.arch_name = properties.gcnArchName;
    profile.cu_count = properties.multiProcessorCount;
    profile.wave_size = properties.warpSize;
    profile.lds_bytes = static_cast<std::uint64_t>(properties.sharedMemPerBlock);
    profile.total_memory_bytes = static_cast<std::uint64_t>(total_bytes);
    profile.free_memory_bytes = static_cast<std::uint64_t>(free_bytes);
    profile.isa = device_isa_capabilities(profile.arch_name);
    return profile;
#else
    throw std::runtime_error("device profile: HIP support is not built");
#endif
}

std::string format_device_profile(const DeviceProfile& profile) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << "{\"schema\":\"strix-device-profile-v1\",\"arch_name\":\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const char ch : profile.arch_name) {
        const auto byte = static_cast<unsigned char>(ch);
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (byte < 0x20U || byte >= 0x7fU) out << "\\u00" << hex[byte >> 4U] << hex[byte & 15U];
        else out << ch;
    }
    out << "\",\"device_index\":" << profile.device_index
        << ",\"cu_count\":" << profile.cu_count << ",\"wave_size\":" << profile.wave_size
        << ",\"lds_bytes\":" << profile.lds_bytes
        << ",\"total_memory_bytes\":" << profile.total_memory_bytes
        << ",\"free_memory_bytes\":" << profile.free_memory_bytes
        << ",\"capability_source\":\"architecture-family-isa\",\"capabilities\":{\"rdna35_wmma\":" << profile.isa.rdna35_wmma
        << ",\"wmma_bf16_w32\":" << profile.isa.wmma_bf16_w32
        << ",\"dpp_row\":" << profile.isa.dpp_row << ",\"sudot4\":" << profile.isa.sudot4
        << "},\"bandwidth\":null,\"qualification\":\"not-assessed\"}\n";
    return out.str();
}

bool print_device_profile_if_requested(int argc, const char* const* argv) {
    bool explicit_request = false, effective_config = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        explicit_request = explicit_request || argument == "--print-device-profile";
        effective_config = effective_config || argument == "--print-effective-config";
    }
#if defined(_MSC_VER)
    char* environment_copy = nullptr;
    std::size_t environment_size = 0;
    if (_dupenv_s(&environment_copy, &environment_size, "STRIX_PRINT_DEVICE_PROFILE") != 0)
        throw std::runtime_error("device profile: cannot read environment");
    const std::unique_ptr<char, decltype(&std::free)> environment_owner(environment_copy, &std::free);
    const char* env = environment_owner.get();
#else
    const char* env = std::getenv("STRIX_PRINT_DEVICE_PROFILE");
#endif
    const bool requested = explicit_request || (env != nullptr && std::string_view(env) == "1");
    if (effective_config) {
        if (explicit_request) throw std::invalid_argument("--print-device-profile cannot be combined with --print-effective-config");
        return false;
    }
    if (!requested) return false;
    std::cout << format_device_profile(probe_device_profile());
    return true;
}

} // namespace strix
