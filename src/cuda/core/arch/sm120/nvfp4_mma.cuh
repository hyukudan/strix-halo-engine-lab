#pragma once

#include <cuda_runtime.h>

namespace strix::cuda::core::nvfp4::mma_detail {

__device__ __forceinline__ void m16n8k64(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ == 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    asm volatile("trap;");
#endif
}

}  // namespace strix::cuda::core::nvfp4::mma_detail
