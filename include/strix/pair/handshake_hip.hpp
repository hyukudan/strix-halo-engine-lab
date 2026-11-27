#pragma once

#include "strix/pair/executable_identity.hpp"
#include "strix/pair/handshake.hpp"

#include <hip/hip_runtime_api.h>

namespace strix::pair {

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
