// Enqueue side of the NVFP4 tile GEMM (contract: strix/cuda/core/nvfp4_tile.hpp; kernels: arch/sm120/nvfp4_tile.cuh).
// Built for sm_120a only (block-scaled FP4 mma, cvt e2m1x2); another device gets cudaErrorNotSupported, never a trap.
#include "strix/cuda/core/nvfp4_tile_cuda.hpp"

#include "arch/sm120/nvfp4_tile.cuh"

namespace strix::cuda::core::nvfp4::tile {
namespace {
cudaError_t capability() {
    int device = 0, major = 0, minor = 0;
    if (auto e = cudaGetDevice(&device); e != cudaSuccess) return e;
    if (auto e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); e != cudaSuccess) return e;
    if (auto e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); e != cudaSuccess) return e;
    return major == 12 && minor == 0 ? cudaSuccess : cudaErrorNotSupported;
}
}  // namespace

bool supported() { return capability() == cudaSuccess; }

cudaError_t repack_enqueue(const std::uint8_t* records, std::uint32_t out, std::uint32_t in, std::uint8_t* tile, cudaStream_t stream) {
    if (!records || !tile || out == 0 || out % 16 || in == 0 || in % kK) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    const std::size_t bytes = w_tile_bytes(out, in);
    sm120::repack_kernel<<<static_cast<unsigned>((bytes + 255) / 256), 256, 0, stream>>>(records, bytes, in, tile);
    return cudaGetLastError();
}

cudaError_t quantize_enqueue(const float* x, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile, float* rs, cudaStream_t stream) {
    if (!x || !x_tile || !rs || rows == 0 || in == 0 || in % kK || reinterpret_cast<std::uintptr_t>(x) % 16) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    sm120::quantize_kernel<<<rows_padded(rows), 256, 0, stream>>>(x, rows, in, x_tile, rs);
    return cudaGetLastError();
}

cudaError_t gemm_enqueue(const GemmArgs& a, cudaStream_t stream) {
    if (!valid(a)) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    // CUDA applies the attribute to the current device; idempotent, no process-global flag.
    if (auto e = cudaFuncSetAttribute(sm120::gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm120::kSmem));
        e != cudaSuccess)
        return e;
    const std::uint32_t per_cta = a.epi == Epi::glu ? kCtaOut / 2 : kCtaOut;  // GLU: 64 channels = 128 gate/up rows
    const std::uint64_t tiles = static_cast<std::uint64_t>((a.out + per_cta - 1) / per_cta) * (a.rows_pad / kCtaRows);
    if (tiles > 0x7fffffffULL) return cudaErrorInvalidValue;
    sm120::gemm_kernel<<<static_cast<unsigned>(tiles), sm120::kThreads, sm120::kSmem, stream>>>(a);
    return cudaGetLastError();
}
}  // namespace strix::cuda::core::nvfp4::tile
