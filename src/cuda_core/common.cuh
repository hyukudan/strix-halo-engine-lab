#pragma once
// Private helpers shared by the cuda_core kernels. Build flags: --fmad=false --prec-div=true --prec-sqrt=true --ftz=false.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::cuda_core {

// Programmatic dependent launch (PDL). A kernel launched with launch_pdl may START before its predecessor in the stream/graph has
// finished (once every predecessor block has called pdl_trigger or exited); it MUST call pdl_wait() before reading anything the
// predecessor wrote or writing anything the predecessor reads. Work before pdl_wait may only touch constant data (weights, committed
// state that nothing in the same graph writes). pdl_wait returns after the predecessor grid completed and its writes are visible.
// Depth rule: call pdl_trigger AFTER pdl_wait. Then a successor can launch only once its predecessor's predecessor completed,
// so at most one kernel runs ahead (pre-wait, constants only). A trigger before the wait lets a whole chain of small grids be
// launched while an earlier kernel still streams a buffer that a later kernel rewrites and a still later one reads: the NVFP4
// chain (nvfp4_test test_pdl_chain, ks 1, rows 16) then differed from serial order. Kernels still triggering at entry
// (rope, norms, attention, gdn, dense, dense_mma) follow this rule as their owners move them.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// L2 prefetch of one 128-byte line (no register, no ordering against pdl_wait): the way a kernel streams its constant
// weights before pdl_wait.
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

inline bool& pdl_enabled() {
    static bool on = true;
    return on;
}

template <class... KArgs, class... Args>
cudaError_t launch_pdl_smem(void (*kernel)(KArgs...), dim3 grid, dim3 block, std::size_t smem, cudaStream_t stream,
                            Args... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pdl_enabled() ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

template <class... KArgs, class... Args>
cudaError_t launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
    return launch_pdl_smem(kernel, grid, block, 0, stream, args...);
}

// Single points for transcendentals (a HIP twin swaps these for its own bit-matched versions).
__device__ __forceinline__ float cc_expf(float x) { return expf(x); }
__device__ __forceinline__ float cc_log1pf(float x) { return log1pf(x); }

__device__ __forceinline__ float cc_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, cc_expf(-x))); }
__device__ __forceinline__ float cc_sigmoid(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, cc_expf(-x))); }

__device__ __forceinline__ float bf16_lo(std::uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(std::uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// 32-lane xor butterfly (offsets 16,8,4,2,1): every lane ends with the same bits.
__device__ __forceinline__ float warp_sum_xor(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Block of exactly 256 threads: warp butterfly, then the 8 warp sums added in order 0..7. smem: 8 floats.
__device__ __forceinline__ float block_sum256(float v, float* smem) {
    v = warp_sum_xor(v);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) smem[w] = v;
    __syncthreads();
    float s = smem[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) s = __fadd_rn(s, smem[i]);
    __syncthreads();
    return s;
}

}  // namespace strix::cuda_core
