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
// Why that is enough at any chain depth (transitivity): if every PDL kernel reaches pdl_wait in at least one block of every
// launch (an early `return` before it is allowed only for blocks that cannot be all of the grid), then "K_n completed" implies
// "K_n-1 completed", so when K_n passes its wait every earlier kernel has completed and is visible. Only pre-wait code ever
// overlaps earlier kernels, and it touches constants only. The two conditions per kernel are the whole correctness contract;
// pdl_trigger placement does not affect correctness. Trigger placement is a resource choice: after the wait, at most one
// kernel sits in its pre-wait region (bounds the weight prefetch held in L2: fp4_gemv); at entry, a chain of small grids can
// all sit pre-wait at once. The chain-level form of this rule (which pairs overlap, demotion of a violating launch) is the
// backend-neutral include/strix/launch_overlap.hpp; the probe harness tests/cuda_core/pdl_harness.cuh measures each fact on
// the device (seeded hazards caught whenever the successor really started early, %globaltimer witness).
// Enforced by type where bound: a kernel takes its non-constant buffers as After<T>; their pointer is only reachable with the
// PdlWaited token that pdl_wait() returns, so no access to them can be written before the wait (bound: fp4_gemv,
// fp4_reduce, norm_kernel). Constant data (weights, gains, committed state) stays a plain pointer.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
class PdlWaited;
__device__ __forceinline__ PdlWaited pdl_wait();
class PdlWaited {
    __device__ PdlWaited() {}
    friend __device__ PdlWaited pdl_wait();
};
__device__ __forceinline__ PdlWaited pdl_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    return PdlWaited();
}
template <class T>
class After {
  public:
    After() = default;
    __host__ __device__ After(T* p) : p_(p) {}  // NOLINT(google-explicit-constructor): host fills jobs / launch args
    __device__ __forceinline__ T* operator()(const PdlWaited&) const { return p_; }

  private:
    T* p_ = nullptr;
};

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

// launch_pdl with a thread block cluster of `cluster_x` blocks along x (compute capability 9.0+; gridDim.x must be a
// multiple of cluster_x and cluster_x <= 8, the portable cluster size), `smem` bytes of dynamic shared memory per block.
template <class... KArgs, class... Args>
cudaError_t launch_pdl_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, unsigned cluster_x, std::size_t smem,
                               cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = cluster_x;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pdl_enabled() ? 2 : 1;
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

// One float copied global -> shared without a register (sm_80+ cp.async; the copy is issued, not waited). A row staged
// with these costs one memory round trip however many values each thread copies.
__device__ __forceinline__ void cp_async_f32(float* shared, const float* global) {
    const auto address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(address), "l"(global) : "memory");
}
// Waits for every cp.async this thread issued (pair with a block barrier before other threads read the values).
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// 32-lane xor butterfly (offsets 16,8,4,2,1): every lane ends with the same bits.
__device__ __forceinline__ float warp_sum_xor(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Block of exactly 256 threads: warp butterfly, then the 8 warp sums added in order 0..7. smem: 8 floats.
template <bool SwapTail = false>
__device__ __forceinline__ float block_sum256(float v, float* smem) {
    v = warp_sum_xor(v);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) smem[w] = v;
    __syncthreads();
    float s = smem[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) s = __fadd_rn(s, smem[SwapTail && i >= 6 ? 13 - i : i]);
    __syncthreads();
    return s;
}

}  // namespace strix::cuda_core
