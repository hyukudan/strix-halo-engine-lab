#include "strix/hip_repacked_hyper_projection_tests.hpp"
#include "strix/hc_repacked_down_layout.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {

// Diagnostic-only device module. Legacy device translation units remain literal
// copies of the qualified baseline; no production catalog selects this module.
typedef float F32x4 __attribute__((ext_vector_type(4)));
typedef unsigned U32x4 __attribute__((ext_vector_type(4)));

template<unsigned N, class F>
__device__ __forceinline__ void static_for(F&& operation) {
    [&]<unsigned... I>(std::integer_sequence<unsigned, I...>) {
        (operation.template operator()<I>(), ...);
    }(std::make_integer_sequence<unsigned, N>{});
}

template<class T, unsigned N> struct RegisterRing { T head; RegisterRing<T, N-1> tail; };
template<class T> struct RegisterRing<T, 0> {};
template<unsigned I, class T, unsigned N>
__device__ __forceinline__ T& slot(RegisterRing<T, N>& ring) {
    static_assert(I < N);
    if constexpr(I == 0) return ring.head;
    else return slot<I-1>(ring.tail);
}

struct PairSlot { RegisterRing<unsigned, 16> words; unsigned scales; };
struct TileSlot { F32x4 s0, s1, s2, s3; };

__device__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__device__ __forceinline__ void load_pair(PairSlot& part, const U32x4* codes,
    const unsigned* scales, unsigned pair, unsigned rank_pad, unsigned output) {
    const auto base = static_cast<std::uint64_t>(pair) * 4U * rank_pad + output;
    static_for<4>([&]<unsigned H>() {
        const U32x4 loaded = codes[base + H * rank_pad];
        static_for<4>([&]<unsigned J>() {
            slot<H * 4U + J>(part.words) = loaded[J];
        });
    });
    part.scales = scales[static_cast<std::uint64_t>(pair) * rank_pad + output];
}

__device__ __forceinline__ void load_tile(TileSlot& part, const float* inject,
    unsigned width, unsigned tile, unsigned lane) {
    const auto feature = static_cast<std::uint64_t>(tile) * 128U + lane * 4U;
    part.s0 = *reinterpret_cast<const F32x4*>(inject + feature);
    part.s1 = *reinterpret_cast<const F32x4*>(inject + width + feature);
    part.s2 = *reinterpret_cast<const F32x4*>(inject + 2U * width + feature);
    part.s3 = *reinterpret_cast<const F32x4*>(inject + 3U * width + feature);
}

template<bool LdsInput>
__device__ __forceinline__ float input_value(const float* input, const float* staged, unsigned feature) {
    if constexpr(LdsInput) return staged[feature];
    else return input[feature];
}

// The host launches exactly one wave32 per block. Release/acquire fences make
// the preceding LDS reads/stores visible across that wave's rendezvous, without
// imposing workgroup scope on unrelated, read-only global prefetches. This
// contract also requires wave32 in the compiled kernel descriptor.
__device__ __forceinline__ void wave_exchange() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template<bool LdsInput>
__device__ __forceinline__ void consume_pair(float& sum, const PairSlot& part,
    unsigned pair, const float* input, const float* staged) {
    static_for<2>([&]<unsigned E>() {
        const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(part.scales >> (16U * E))));
        static_for<2>([&]<unsigned H>() {
            static_for<16>([&]<unsigned J>() {
                const auto word = slot<(2U * E + H) * 4U + J / 4U>(part.words);
                const auto code = static_cast<std::int8_t>(static_cast<std::uint8_t>(word >> (8U * (J % 4U))));
                const float coefficient = scale * static_cast<float>(code);
                sum += input_value<LdsInput>(input, staged, (pair * 2U + E) * 32U + H * 16U + J) * coefficient;
            });
            if constexpr(LdsInput) __builtin_amdgcn_sched_barrier(0);
        });
    });
}

template<unsigned K, unsigned D, bool LdsInput>
__device__ __forceinline__ void projection_roles(const float* input, const float* staged,
    const U32x4* codes, const unsigned* scales, const float* inject, unsigned width,
    unsigned streams, unsigned rank, unsigned rank_pad, float* low, float* injection,
    unsigned low_blocks, float (*tile)[132]) {
    const unsigned lane = threadIdx.x;
    const std::uint64_t row = blockIdx.y;
    float sum = 0.0F;
    if(blockIdx.x < low_blocks) {
        const unsigned output = blockIdx.x * 32U + lane, pairs = width / 64U;
        RegisterRing<PairSlot, K> ring;
        static_for<K>([&]<unsigned I>() {
            if(I < pairs) load_pair(slot<I>(ring), codes, scales, I, rank_pad, output);
        });
        for(unsigned first = 0; first < pairs; first += K) {
            static_for<K>([&]<unsigned I>() {
                const unsigned pair = first + I;
                if(pair < pairs) {
                    consume_pair<LdsInput>(sum, slot<I>(ring), pair, input, staged);
                    if(pair + K < pairs) load_pair(slot<I>(ring), codes, scales, pair + K, rank_pad, output);
                }
            });
        }
        if(output < rank) {
            const float scaled = sum / static_cast<float>(streams);
            low[row * rank + output] = scaled * sigmoid(scaled);
        }
    } else {
        const unsigned tiles = width / 128U;
        RegisterRing<TileSlot, D> ring;
        static_for<D>([&]<unsigned I>() {
            if(I < tiles) load_tile(slot<I>(ring), inject, width, I, lane);
        });
        for(unsigned first = 0; first < tiles; first += D) {
            static_for<D>([&]<unsigned I>() {
                const unsigned current = first + I;
                if(current < tiles) {
                    wave_exchange();
                    *reinterpret_cast<F32x4*>(&tile[0][lane * 4U]) = slot<I>(ring).s0;
                    *reinterpret_cast<F32x4*>(&tile[1][lane * 4U]) = slot<I>(ring).s1;
                    *reinterpret_cast<F32x4*>(&tile[2][lane * 4U]) = slot<I>(ring).s2;
                    *reinterpret_cast<F32x4*>(&tile[3][lane * 4U]) = slot<I>(ring).s3;
                    wave_exchange();
                    if(current + D < tiles) load_tile(slot<I>(ring), inject, width, current + D, lane);
                    if(lane < streams) {
#pragma unroll
                        for(unsigned feature = 0; feature < 128U; ++feature)
                            sum += input_value<LdsInput>(input, staged, current * 128U + feature) * tile[lane][feature];
                    }
                }
            });
        }
        if(lane < streams) {
            const float scaled = sum / static_cast<float>(streams);
            injection[row * streams + lane] = 2.0F * sigmoid(scaled);
        }
    }
}

template<unsigned K, unsigned D, bool LdsInput>
__global__ __launch_bounds__(32) void word_repacked_hyper_projection_kernel(
    const float* __restrict__ normalized, const U32x4* __restrict__ codes,
    const unsigned* __restrict__ scales, const float* __restrict__ inject,
    unsigned width, unsigned streams, unsigned rank, unsigned rank_pad,
    float* __restrict__ low, float* __restrict__ injection, unsigned low_blocks) {
    __shared__ __align__(16) float tile[4][132];
    const auto* input = normalized + static_cast<std::uint64_t>(blockIdx.y) * width;
    if constexpr(LdsInput) {
        __shared__ __align__(16) float staged[10240];
        for(unsigned chunk = 0; chunk < width; chunk += 2048U) {
            RegisterRing<F32x4, 16> parts;
            static_for<16>([&]<unsigned U>() {
                if(chunk + U * 128U < width)
                    slot<U>(parts) = *reinterpret_cast<const F32x4*>(input + chunk + U * 128U + threadIdx.x * 4U);
            });
            static_for<16>([&]<unsigned U>() {
                if(chunk + U * 128U < width)
                    *reinterpret_cast<F32x4*>(staged + chunk + U * 128U + threadIdx.x * 4U) = slot<U>(parts);
            });
        }
        __syncthreads();
        projection_roles<K, D, true>(input, staged, codes, scales, inject, width, streams, rank, rank_pad, low, injection, low_blocks, tile);
    } else {
        projection_roles<K, D, false>(input, nullptr, codes, scales, inject, width, streams, rank, rank_pad, low, injection, low_blocks, tile);
    }
}

void check_hip(hipError_t error, const char* operation) {
    if(error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
} // namespace

void hc_repacked_projection_diagnostic::enqueue(
    hc_projection_diagnostic::Shape shape,
    hc_projection_diagnostic::Span normalized,
    hc_projection_diagnostic::Span codes,
    hc_projection_diagnostic::Span scales,
    std::uint32_t rank_pad,
    hc_projection_diagnostic::Span inject,
    hc_projection_diagnostic::Span low,
    hc_projection_diagnostic::Span injection,
    Schedule schedule,
    hipStream_t stream) {
    using Span = hc_projection_diagnostic::Span;
    if (shape.width < 256U || shape.width > 10240U ||
        shape.width % 256U != 0U || shape.streams != 4U ||
        shape.rank == 0U || shape.rank > 512U ||
        shape.rows == 0U || shape.rows > 65535U ||
        rank_pad != ((shape.rank + 31U) / 32U) * 32U ||
        (schedule != Schedule::w1_scalar_k8_d8 &&
         schedule != Schedule::w2_lds_k8_d8 &&
         schedule != Schedule::w3_scalar_k4_d4)) {
        throw std::invalid_argument(
            "HC repacked projection diagnostic unsupported shape/schedule");
    }
    const auto geometry = hc_repacked_down::geometry(shape.width, shape.rank);
    const bool with_inject = inject.address != 0U;
    if (with_inject != (injection.address != 0U) ||
        (!with_inject && (inject.bytes != 0U || injection.bytes != 0U))) {
        throw std::invalid_argument(
            "HC repacked projection diagnostic injection owner pair differs");
    }
    const auto check_span = [](Span owner, std::uint64_t required,
                               std::uint64_t alignment) {
        if (owner.address == 0U || owner.address % alignment != 0U ||
            owner.bytes < required ||
            owner.bytes > std::numeric_limits<std::uintptr_t>::max() - owner.address) {
            throw std::invalid_argument(
                "HC repacked projection diagnostic owner extent/alignment differs");
        }
    };
    check_span(normalized, static_cast<std::uint64_t>(shape.rows) * shape.width * 4U, 16U);
    check_span(codes, geometry.codes_bytes, 256U);
    check_span(scales, geometry.scales_bytes, 256U);
    check_span(low, static_cast<std::uint64_t>(shape.rows) * shape.rank * 4U, 4U);
    if (with_inject) {
        check_span(inject, static_cast<std::uint64_t>(shape.width) * 4U * 4U, 16U);
        check_span(injection, static_cast<std::uint64_t>(shape.rows) * 4U * 4U, 4U);
    }
    // Every addition below is safe after check_span. Use whole supplied spans,
    // so a caller cannot hide an alias in unused owner bytes.
    const auto overlap = [](Span a, Span b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    if (overlap(codes, scales) || overlap(low, normalized) ||
        overlap(low, codes) || overlap(low, scales) ||
        (with_inject &&
         (overlap(low, inject) || overlap(low, injection) ||
          overlap(injection, normalized) || overlap(injection, codes) ||
          overlap(injection, scales) || overlap(injection, inject)))) {
        throw std::invalid_argument("HC repacked projection diagnostic owners overlap");
    }
    check_hip(hipPeekAtLastError(), "HC repacked projection diagnostic stale error");
    const auto launch = [&]<unsigned Pairs, unsigned Depth, bool LdsInput>() {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(word_repacked_hyper_projection_kernel<Pairs, Depth, LdsInput>),
            dim3(rank_pad / 32U + (with_inject ? 1U : 0U), shape.rows),
            dim3(32U), 0, stream,
            reinterpret_cast<const float*>(normalized.address),
            reinterpret_cast<const U32x4*>(codes.address),
            reinterpret_cast<const std::uint32_t*>(scales.address),
            reinterpret_cast<const float*>(inject.address),
            shape.width, shape.streams, shape.rank, rank_pad,
            reinterpret_cast<float*>(low.address),
            reinterpret_cast<float*>(injection.address), rank_pad / 32U);
    };
    if (schedule == Schedule::w1_scalar_k8_d8) launch.template operator()<8U, 8U, false>();
    else if (schedule == Schedule::w2_lds_k8_d8) launch.template operator()<8U, 8U, true>();
    else launch.template operator()<4U, 4U, false>();
    check_hip(hipGetLastError(), "HC repacked projection diagnostic enqueue");
}


} // namespace strix
