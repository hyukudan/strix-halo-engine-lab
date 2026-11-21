#pragma once

enum hipError_t { hipSuccess, hipErrorInvalidDevice };

struct hipDeviceProp_t {
    char gcnArchName[32];
    int warpSize;
};

namespace hip_archguard_test {
inline int active_device = 0;
inline unsigned property_reads = 0;
inline constexpr hipDeviceProp_t devices[] = {
    {"gfx1151:xnack-", 32}, {"gfx1200", 32}, {"gfx1201", 32},
    {"gfx942:sramecc+", 64}, {"gfx950", 64}, {"gfx11510", 32},
    {"gfx1151", 64},
};
}

inline hipError_t hipGetDevice(int* device) {
    *device = hip_archguard_test::active_device;
    return *device < 0 ? hipErrorInvalidDevice : hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* properties, int device) {
    ++hip_archguard_test::property_reads;
    *properties = hip_archguard_test::devices[device];
    return hipSuccess;
}

inline const char* hipGetErrorString(hipError_t) {
    return "invalid device";
}
