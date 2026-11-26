#include "cuda/core/grouped_w4a4.hpp"
#include "strix/cuda/arch/native_images.hpp"
#include "strix/cuda/core/nvfp4_tile_cuda.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/nvfp4.hpp"

#include <cstdio>

namespace {
using strix::cuda::arch::NativeNvfp4Images;
namespace cc = strix::cuda_core;
namespace tile = strix::cuda::core::nvfp4::tile;
namespace grouped = strix::cuda::core::grouped_w4a4;

static_assert(!NativeNvfp4Images{}.supports(12, 0));
static_assert(NativeNvfp4Images{true, false}.supports(12, 0));
static_assert(!NativeNvfp4Images{true, false}.supports(12, 1));
static_assert(NativeNvfp4Images{false, true}.supports(12, 1));
static_assert(!NativeNvfp4Images{false, true}.supports(12, 0));
static_assert(!NativeNvfp4Images{true, true}.supports(8, 9));
static_assert(!NativeNvfp4Images{true, true}.supports(12, 2));
static_assert(!NativeNvfp4Images{true, true}.supports(13, 0));
}  // namespace

int main() {
    // These addresses are never dereferenced. The portable build must refuse
    // an otherwise admitted operation before trying to access a CUDA device.
    alignas(16) float input[128]{};
    alignas(16) std::uint8_t bytes[16]{};
    float scale = 0.0F;
    cc::DenseWeight w{};
    w.format = cc::WeightFormat::nvfp4;
    w.data = bytes;
    w.in = 128;
    w.out = 16;
    if (cc::nvfp4_supported() || tile::supported() ||
        cc::nvfp4_quantize_f32(input, w.out, w.in, bytes, &scale, nullptr) != cudaErrorNotSupported ||
        cc::nvfp4_dense_rows(w, input, input, 1, nullptr, nullptr) != cudaErrorNotSupported ||
        cc::nvfp4_expand_f16(w, bytes, nullptr) != cudaErrorNotSupported ||
        cc::nvfp4_expand_e4m3(w, bytes, nullptr) != cudaErrorNotSupported ||
        tile::quantize_enqueue(input, 1, w.in, bytes, &scale, nullptr) != cudaErrorNotSupported ||
        grouped::launch({}, nullptr) != cudaErrorNotSupported) {
        std::fputs("portable CUDA build admitted a native NVFP4 operation\n", stderr);
        return 1;
    }
    return 0;
}
