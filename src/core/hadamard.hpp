#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
namespace strix::core {
// Preserve an observable RN32 boundary before a later RN16 conversion. Without
// this register boundary LLVM selects mixed FP16-output arithmetic, which differs
// on double-rounding ties even with -ffp-contract=off. No memory/scratch access.
__device__ __forceinline__ float materialize_fp32(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(value));
#endif
    return value;
}
// The default observer of hadamard128: no per-value check and an RN32 boundary on the scaled result.
struct HadamardUnobserved {
    static constexpr bool boundary = true;
    __device__ __forceinline__ void step(float) const {}
};
// Four contiguous wave-sized slices give each lane indices lane+32*j.
// Strides 1..16 communicate across lanes; strides 32,64 remain in registers.
// The operation order matches the ascending-stride FP32 reference exactly.
// `lane` is the caller's wave lane (threadIdx.x & 31 unless the kernel remaps lanes).
// Observe (a trait; the values never depend on it): observe.step(v) runs after every value the transform
// produces (each butterfly output, then each scaled value), in production order; Observe::boundary keeps the
// RN32 boundary on the scaled values (a caller whose next RN16 conversion materializes itself may drop it).
template <class Observe>
__device__ __forceinline__ void hadamard128(float (&values)[4], unsigned lane, const Observe& observe) {
    #pragma unroll
    for(int stride=1;stride<32;stride*=2) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float peer=__shfl_xor(values[j],stride,32);
            values[j]=(lane&static_cast<unsigned>(stride))?peer-values[j]:values[j]+peer;
            observe.step(values[j]);
        }
    }
    #pragma unroll
    for(int stride=1;stride<4;stride*=2) {
        #pragma unroll
        for(int j=0;j<4;++j) if(!(j&stride)) {
            float left=values[j],right=values[j+stride];
            values[j]=left+right; values[j+stride]=left-right;
            observe.step(values[j]); observe.step(values[j+stride]);
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        if constexpr (Observe::boundary) values[j]=materialize_fp32(values[j]*0x1.6a09e6p-4f);
        else values[j]=values[j]*0x1.6a09e6p-4f;
        observe.step(values[j]);
    }
}
__device__ __forceinline__ void hadamard128(float (&values)[4], unsigned lane) {
    hadamard128(values, lane, HadamardUnobserved{});
}
__device__ __forceinline__ void hadamard128(float (&values)[4]) {
    hadamard128(values, threadIdx.x & 31U);
}

}
