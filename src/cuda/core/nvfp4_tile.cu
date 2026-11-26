// Enqueue side of the NVFP4 tile GEMM (contract: strix/cuda/core/nvfp4_tile.hpp; kernels: arch/sm120/nvfp4_tile.cuh).
// Block-scaled FP4 MMA and E2M1 conversion require a matching native image.
#include "strix/cuda/core/nvfp4_tile_cuda.hpp"

#include "arch/sm120/nvfp4_tile.cuh"
#include "../arch/native_nvfp4.hpp"

namespace strix::cuda::core::nvfp4::tile {
namespace {
cudaError_t capability() { return arch::native_nvfp4_status(); }
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
cudaError_t quantize_enqueue(const std::uint16_t* x, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile, float* rs,
                             cudaStream_t stream) {
    if (!x || !x_tile || !rs || rows == 0 || in == 0 || in % kK || reinterpret_cast<std::uintptr_t>(x) % 16) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    sm120::quantize_kernel<<<rows_padded(rows), 256, 0, stream>>>(x, rows, in, x_tile, rs);
    return cudaGetLastError();
}

namespace {
// One launch of the folded quantiser with row normaliser nm; NREG bucket = the smallest of 8/16/24/32 holding in / 256 elements per
// thread (padding entries are 0 and add +0 / sit outside every head: any bucket >= the need gives the same bits).
template <class Norm>
cudaError_t norm_quantize_launch(const float* x, const Norm& nm, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile, float* rs,
                                 cudaStream_t stream) {
    if (!norm_quantize_fits(in)) return cudaErrorInvalidValue;
    if (!x || !x_tile || !rs || rows == 0 || reinterpret_cast<std::uintptr_t>(x) % 16) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    const unsigned grid = rows_padded(rows);
    const std::size_t smem = static_cast<std::size_t>(in) * sizeof(float);
    const std::uint32_t nreg = (in + 255) / 256;
    if (nreg <= 8) sm120::norm_quantize_kernel<8, Norm><<<grid, 256, smem, stream>>>(x, nm, rows, in, x_tile, rs);
    else if (nreg <= 16) sm120::norm_quantize_kernel<16, Norm><<<grid, 256, smem, stream>>>(x, nm, rows, in, x_tile, rs);
    else if (nreg <= 24) sm120::norm_quantize_kernel<24, Norm><<<grid, 256, smem, stream>>>(x, nm, rows, in, x_tile, rs);
    else sm120::norm_quantize_kernel<32, Norm><<<grid, 256, smem, stream>>>(x, nm, rows, in, x_tile, rs);
    return cudaGetLastError();
}
}  // namespace

cudaError_t norm_quantize_enqueue(const float* x, const float* gain, float eps, std::uint32_t rows, std::uint32_t in, std::uint8_t* x_tile,
                                  float* rs, cudaStream_t stream) {
    return norm_quantize_launch(x, sm120::RmsRowNorm{gain, eps}, rows, in, x_tile, rs, stream);
}

cudaError_t gated_norm_quantize_enqueue(const float* x, const float* z, const float* gain, float eps, std::uint32_t rows,
                                        std::uint32_t in, std::uint8_t* x_tile, float* rs, cudaStream_t stream) {
    if (!z || !gain || in % kGatedNormHead) return cudaErrorInvalidValue;
    return norm_quantize_launch(x, sm120::GatedHeadNorm{z, gain, eps}, rows, in, x_tile, rs, stream);
}

cudaError_t gemm_enqueue(const GemmArgs& a, cudaStream_t stream) {
    if (!valid(a)) return cudaErrorInvalidValue;
    if (auto e = capability(); e != cudaSuccess) return e;
    // Feed trait: same tiles, same stage bytes, same MMA order (bitwise); only the stage copy and its synchronisation differ.
    // H16 trait (BF16 hidden rows) is a compile-time instantiation: the FP32 kernels carry no H16 code.
    const bool bulk = a.feed == Feed::bulk, h16 = a.y16 != nullptr;
    const auto kernel = bulk ? (h16 ? sm120::gemm_kernel<Feed::bulk, true> : sm120::gemm_kernel<Feed::bulk, false>)
                             : (h16 ? sm120::gemm_kernel<Feed::lsu, true> : sm120::gemm_kernel<Feed::lsu, false>);
    using Lsu = tall::Dense4<false, false>;
    using Bulk = tall::Dense4<true, false>;
    const unsigned threads = bulk ? Bulk::tile.threads : Lsu::tile.threads;
    // CUDA applies the attribute to the current device; idempotent, no process-global flag.
    if (auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm120::kSmem)); e != cudaSuccess)
        return e;
    const std::uint32_t per_cta = a.epi == Epi::glu ? Lsu::glu_outputs : Lsu::tile.outputs;
    const std::uint64_t tiles = static_cast<std::uint64_t>((a.out + per_cta - 1) / per_cta) * (a.rows_pad / Lsu::tile.rows);
    if (tiles > 0x7fffffffULL) return cudaErrorInvalidValue;
    kernel<<<static_cast<unsigned>(tiles), threads, sm120::kSmem, stream>>>(a);
    return cudaGetLastError();
}
}  // namespace strix::cuda::core::nvfp4::tile
