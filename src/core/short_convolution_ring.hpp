#pragma once
// F7 short convolution, decode form with a coded per-channel history (the companion of short_convolution.hpp, whose
// FP32 form keeps history [taps - 1][channel]). Here each channel owns `taps` history slots in the binder's codec,
// oldest first, with the current row last: [channel][taps]. One row of one channel:
//   history[0 .. taps-2] = history[1 .. taps-1];  history[taps-1] = Codec::encode(input)
//   sum = weighted_taps<false>(decode(history[t]) * weight[channel * taps + t])   (from +0, tap order, TU contraction)
//   output = Epilogue::apply(sum)                                                  (default: SiLU, then Codec::round)
// The output overwrites the input in place. Rows of one channel run in order (the history is sequential); channels are
// independent. Hooks: Codec {Slot, encode(float), decode(Slot), round(float)}, Load {Kind, load(const void*, Kind, index)}.
// The expressions are the former GDN conv update text verbatim (bits of every existing binder unchanged).
#include <cmath>  // expf
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "core/short_convolution.hpp"
#define STRIX_CONV_RING_FHD __host__ __device__ __forceinline__
#else
#define STRIX_CONV_RING_FHD inline
#endif

namespace strix::core::causal_conv {

// SiLU of the tap sum, rounded by the history codec (the stored activation has the codec's precision).
template <class Codec> struct SiluRound {
    STRIX_CONV_RING_FHD static float apply(float sum) {
        const float silu =
            sum /
            (1.0F +
             expf(-sum));
        return Codec::round(
            silu);
    }
};

// Shift one channel's history by one row and append the coded input. The input is read after the shift (the order of
// the former text: a value read before the shift schedules the loads differently).
template <class Codec>
STRIX_CONV_RING_FHD void ring_push(typename Codec::Slot* state, std::uint32_t taps, const float* input) {
    for (std::uint32_t index = 0;
         index + 1U < taps;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[taps - 1U] =
        Codec::encode(
            *input);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template <class Codec, class Load, class Epilogue = SiluRound<Codec>>
__device__ __forceinline__ void ring_update(
    float* qkv, typename Codec::Slot* conv_state, const void* conv_weights,
    typename Load::Kind weight_kind_value, std::uint32_t kernel_size, std::uint32_t channel) {
    auto* state =
        conv_state +
        static_cast<std::uint64_t>(
            channel) *
            kernel_size;

    ring_push<Codec>(state, kernel_size, qkv + channel);

    const float sum = weighted_taps<false>(kernel_size,
        [&](std::uint32_t tap) { return Codec::decode(state[tap]); },
        [&](std::uint32_t tap) {
            return Load::load(conv_weights, weight_kind_value,
                              static_cast<std::uint64_t>(channel) * kernel_size + tap);
        });
    qkv[channel] =
        Epilogue::apply(
            sum);
}

// One row, one thread per channel; a non-null nonzero `reject` (graph status) skips the launch's work.
template <class Codec, class Load, class Epilogue = SiluRound<Codec>>
__global__ void ring_update_kernel(
    float* qkv,
    typename Codec::Slot* conv_state,
    const void* conv_weights,
    typename Load::Kind weight_kind_value,
    std::uint32_t channels,
    std::uint32_t kernel_size, std::uint32_t* reject) {
    if (reject != nullptr && *reject != 0) return;
    const auto channel =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (channel >= channels) {
        return;
    }

    ring_update<Codec, Load, Epilogue>(qkv, conv_state, conv_weights, weight_kind_value, kernel_size, channel);
}

// Rows form: one thread per channel walks rows [row][channels] in order (bitwise the per-row launches).
template <class Codec, class Load, class Epilogue = SiluRound<Codec>>
__global__ void ring_update_rows_kernel(float* qkv, typename Codec::Slot* conv_state, const void* conv_weights,
                                        typename Load::Kind kind, std::uint32_t channels, std::uint32_t kernel_size,
                                        std::uint32_t rows) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    for (std::uint32_t r = 0; r < rows; ++r)
        ring_update<Codec, Load, Epilogue>(qkv + static_cast<std::uint64_t>(r) * channels, conv_state, conv_weights, kind,
                                           kernel_size, channel);
}

// Register form of the rows kernel (Taps = the kernel size at compile time): the same operations on the same values in
// the same order; history and taps live in registers, the inputs of Ahead rows are loaded together, the history is
// written back after the last row. Same bits as ring_update_rows_kernel.
template <class Codec, class Load, unsigned Taps, unsigned Ahead = 8U, class Epilogue = SiluRound<Codec>>
__global__ void ring_update_rows_registers_kernel(float* qkv, typename Codec::Slot* conv_state, const void* conv_weights,
                                                  typename Load::Kind kind, std::uint32_t channels, std::uint32_t rows) {
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    typename Codec::Slot* state = conv_state + static_cast<std::uint64_t>(channel) * Taps;
    typename Codec::Slot ring[Taps];
    float weight[Taps];
#pragma unroll
    for (unsigned i = 0; i < Taps; ++i) {
        ring[i] = state[i];
        weight[i] = Load::load(conv_weights, kind, static_cast<std::uint64_t>(channel) * Taps + i);
    }
    for (std::uint32_t r0 = 0; r0 < rows; r0 += Ahead) {
        float in[Ahead];
#pragma unroll
        for (unsigned j = 0; j < Ahead; ++j)
            if (r0 + j < rows) in[j] = qkv[static_cast<std::uint64_t>(r0 + j) * channels + channel];
#pragma unroll
        for (unsigned j = 0; j < Ahead; ++j) {
            if (r0 + j >= rows) break;
#pragma unroll
            for (unsigned i = 0; i + 1U < Taps; ++i) ring[i] = ring[i + 1U];
            ring[Taps - 1U] = Codec::encode(in[j]);
            float sum = 0.0F;
#pragma unroll
            for (unsigned i = 0; i < Taps; ++i) sum += Codec::decode(ring[i]) * weight[i];
            qkv[static_cast<std::uint64_t>(r0 + j) * channels + channel] = Epilogue::apply(sum);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < Taps; ++i) state[i] = ring[i];
}
#endif

}  // namespace strix::core::causal_conv
