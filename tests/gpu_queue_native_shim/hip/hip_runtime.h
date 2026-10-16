#pragma once
// Extend the existing sequential HIP test double. Never native HIP evidence.
#include "../../hip_visibility_shim/hip/hip_runtime.h"
inline constexpr unsigned hipStreamNonBlocking=1;
inline hipError_t hipStreamCreateWithFlags(hipStream_t* stream,unsigned flags){if(flags!=hipStreamNonBlocking)return hipErrorUnknown;*stream=new FakeHipStream;return hipSuccess;}
inline hipError_t hipStreamGetId(hipStream_t stream,unsigned long long* id){if(!stream)return hipErrorUnknown;*id=reinterpret_cast<std::uintptr_t>(stream);return hipSuccess;}
inline hipError_t hipStreamDestroy(hipStream_t stream){if(!stream)return hipErrorUnknown;delete stream;return hipSuccess;}

inline constexpr unsigned hipEventReleaseToSystem = 0x80000000U;
