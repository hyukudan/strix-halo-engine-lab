#pragma once

#include "strix/model_content_hash.hpp"
#include "strix/pair/handshake.hpp"

#include <hip/hip_runtime_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strix::pair {

[[nodiscard]] inline const ModelContentIdentity& executable_identity() {
    static const auto sha = [] {
#ifdef _WIN32
        std::wstring path(32768U, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (n == 0 || n >= path.size()) throw ProtocolError("pair hello: cannot locate the running executable");
        path.resize(n);
        return sha256_file_content(path, FileHashPolicy::full);
#else
        return sha256_file_content("/proc/self/exe", FileHashPolicy::full);
#endif
    }();
    return sha;
}

[[nodiscard]] inline PairHello native_hello(std::uint32_t protocol, std::uint32_t rank) {
    PairHello result;
    result.protocol = protocol;
    result.rank = rank;
    result.executable_sha256 = executable_identity().bytes;
    int device = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess)
        throw ProtocolError("pair hello: cannot query the GPU architecture");
    set_gpu_arch(result, prop.gcnArchName);
    return result;
}

}  // namespace strix::pair
