#include "rows_linear.hpp"
#include "internal.hpp"
#include "chunk.hpp"
#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
constexpr unsigned tile_waves = 8U;
// One workgroup = eight waves over T * 8 consecutive outputs (pairs for Gate)
// and R consecutive rows. Lane L of every wave is the shared-pass lane L: it
// owns chunks L, L+32, ... of K. Per 512-element K block, each wave decodes its
// lane chunk of T matrices once (the shared WeightChunk decoder) and applies
// every decoded value to R rows. The rows' FP32 inputs for that block are
// staged in LDS, G groups of four elements per barrier, as [row][group][lane]
// float4 so that the 32 lanes of a read touch 32 consecutive float4 values.
// Each accumulator (output, row) still starts at +0 and adds RN(w * x) for its
// lane's chunks and elements in ascending order, and its 32 lane values are
// combined by wave_sum, exactly as in pair_linear_kernel; no row affects
// another row's value.
template <WeightFormat Format, bool Gate, unsigned T, unsigned R, unsigned G>
__global__ void __launch_bounds__(256) rows_linear_kernel(LinearView w, LinearView up, const float* x, unsigned rows,
                                                         float* out, unsigned* failure, bool hardware_half) {
    static_assert(4U % G == 0U && (R * 32U * G) % 256U == 0U);
    __shared__ float4 tile[R * G * 32U];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned first = (blockIdx.y * tile_waves + wave) * T, row0 = blockIdx.x * R;
    const unsigned chunks = w.inputs / 16U, blocks = (chunks + 31U) / 32U;
    const std::uint8_t* wp[T];
    const std::uint8_t* uptr[T];
#pragma unroll
    for (unsigned t = 0; t < T; ++t) {
        // A tail wave reuses a valid row for its unused loads; it never stores it.
        const auto o = min(first + t, w.outputs - 1U);
        wp[t] = w.data + std::size_t(o) * w.row_bytes;
        uptr[t] = Gate ? up.data + std::size_t(o) * up.row_bytes : wp[t];
    }
    float acc[T][R] = {}, uacc[T][R] = {};
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned c = b * 32U + lane;
        const bool active = c < chunks && first < w.outputs;
        WeightChunk<Format> a[T], u[T];
        if (active) {
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = WeightChunk<Format>(wp[t], c * 16U, true, hardware_half);
                if constexpr (Gate)
                    u[t] = WeightChunk<Format>(uptr[t], c * 16U, true, hardware_half);
            }
        }
#pragma unroll
        for (unsigned h = 0; h < 4U / G; ++h) {
            __syncthreads();
#pragma unroll
            for (unsigned j = 0; j < R * 32U * G / 256U; ++j) {
                const unsigned i = threadIdx.x + 256U * j;
                const unsigned rr = i / (32U * G), l = (i / G) % 32U, g = i % G, cc = b * 32U + l;
                if (cc < chunks) {
                    const unsigned row = min(row0 + rr, rows - 1U);
                    tile[(rr * G + g) * 32U + l] = *reinterpret_cast<const float4*>(
                        x + std::size_t(row) * w.inputs + cc * 16U + (h * G + g) * 4U);
                }
            }
            __syncthreads();
            if (active) {
#pragma unroll
                for (unsigned g = 0; g < G; ++g) {
                    float v[T][4], uv[T][4];
#pragma unroll
                    for (unsigned t = 0; t < T; ++t)
#pragma unroll
                        for (unsigned k = 0; k < 4; ++k) {
                            v[t][k] = a[t].at((h * G + g) * 4U + k);
                            if constexpr (Gate)
                                uv[t][k] = u[t].at((h * G + g) * 4U + k);
                        }
#pragma unroll
                    for (unsigned rr = 0; rr < R; ++rr) {
                        const float4 in4 = tile[(rr * G + g) * 32U + lane];
#pragma unroll
                        for (unsigned k = 0; k < 4; ++k) {
                            const float in = k == 0 ? in4.x : k == 1 ? in4.y : k == 2 ? in4.z : in4.w;
#pragma unroll
                            for (unsigned t = 0; t < T; ++t) {
                                acc[t][rr] = acc[t][rr] + v[t][k] * in;
                                if constexpr (Gate)
                                    uacc[t][rr] = uacc[t][rr] + uv[t][k] * in;
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < T; ++t) {
#pragma unroll
        for (unsigned rr = 0; rr < R; ++rr) {
            const float sa = wave_sum(acc[t][rr]);
            float sb = 0.0F;
            if constexpr (Gate)
                sb = wave_sum(uacc[t][rr]);
            if (lane == 0 && first + t < w.outputs && row0 + rr < rows) {
                const float value = Gate ? gelu_tanh(sa) * sb : sa;
                out[std::size_t(row0 + rr) * w.outputs + first + t] = value;
                if (failure)
                    finite_flag(value, failure);
            }
        }
    }
}
bool hardware_half() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_HARDWARE_HALF");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Measured on gfx1151 (rows 512, production shapes): 4x16x2 fastest except
// Q6_K, where 2x32x2 is. STRIX_G4_ROWS_TILE forces one variant everywhere.
int default_variant(WeightFormat format) {
    static const int forced = [] {
        const auto* p = std::getenv("STRIX_G4_ROWS_TILE");
        const int v = p ? std::atoi(p) : -1;
        return v >= 0 && v < rows_linear_variants ? v : -1;
    }();
    return forced >= 0 ? forced : format == WeightFormat::q6_k ? 3 : 4;
}
template <WeightFormat Format, bool Gate, unsigned T, unsigned R, unsigned G>
void launch(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure) {
    const dim3 grid((rows + R - 1U) / R, (w.outputs + tile_waves * T - 1U) / (tile_waves * T));
    hipLaunchKernelGGL((rows_linear_kernel<Format, Gate, T, R, G>), grid, dim3(256), 0, static_cast<hipStream_t>(q), w,
                       up, x, rows, out, failure, hardware_half());
}
// Variants (outputs per wave, rows, groups per barrier): 4x16x4, 4x32x2, 8x16x4,
// 2x32x2, 4x16x2, 4x16x1, 4x8x4, 8x8x2. Gate kernels hold two accumulator sets with half the outputs per wave.
template <WeightFormat Format, bool Gate>
void launch_variant(int variant, LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q,
                    unsigned* failure) {
    switch (variant) {
    case 1: launch<Format, Gate, Gate ? 2U : 4U, 32U, 2U>(w, up, x, rows, out, q, failure); break;
    case 2: launch<Format, Gate, Gate ? 4U : 8U, 16U, 4U>(w, up, x, rows, out, q, failure); break;
    case 3: launch<Format, Gate, Gate ? 1U : 2U, 32U, 2U>(w, up, x, rows, out, q, failure); break;
    case 4: launch<Format, Gate, Gate ? 2U : 4U, 16U, 2U>(w, up, x, rows, out, q, failure); break;
    case 5: launch<Format, Gate, Gate ? 2U : 4U, 16U, 1U>(w, up, x, rows, out, q, failure); break;
    case 6: launch<Format, Gate, Gate ? 2U : 4U, 8U, 4U>(w, up, x, rows, out, q, failure); break;
    case 7: launch<Format, Gate, Gate ? 4U : 8U, 8U, 2U>(w, up, x, rows, out, q, failure); break;
    default: launch<Format, Gate, Gate ? 2U : 4U, 16U, 4U>(w, up, x, rows, out, q, failure); break;
    }
}
template <bool Gate>
bool many_rows(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
               int variant) {
    if (rows <= 8U || !w.data || w.inputs % 16U || (Gate && (up.format != w.format || up.inputs != w.inputs)))
        return false;
    if (variant < 0)
        variant = default_variant(w.format);
    switch (w.format) {
#define G4_ROWS_FORMAT(F)                                                                                                  \
    case WeightFormat::F: launch_variant<WeightFormat::F, Gate>(variant, w, up, x, rows, out, q, failure); break
        G4_ROWS_FORMAT(bf16);
        G4_ROWS_FORMAT(f32);
        G4_ROWS_FORMAT(q8_0);
        G4_ROWS_FORMAT(q4_k);
        G4_ROWS_FORMAT(q6_k);
#undef G4_ROWS_FORMAT
    default: return false;
    }
    launch_checked();
    return true;
}
} // namespace
bool rows_linear_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_EXACT_MANYROW");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
void linear_rows(LinearView w, const float* x, unsigned rows, float* out, Queue q, unsigned* failure, bool many,
                 int variant) {
    if (!rows)
        return;
    if (!w.data || !w.inputs || !w.outputs)
        throw std::runtime_error("empty device linear");
    if (many && many_rows<false>(w, {}, x, rows, out, q, failure, variant))
        return;
    for (unsigned r = 0; r < rows; r += 8U)
        linear(w, x + std::size_t(r) * w.inputs, std::min(8U, rows - r), out + std::size_t(r) * w.outputs, true, q,
               failure);
}
void gate_up_rows(LinearView gate, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
                  bool many, int variant, float* scratch, int split, bool fuse_finite) {
    if (!rows)
        return;
    if (!gate.data || !up.data || gate.outputs != up.outputs || gate.inputs != up.inputs)
        throw std::runtime_error("gate/up geometry");
    static const bool split_default = [] {
        const auto* p = std::getenv("STRIX_G4_ROWS_GATE_SPLIT");
        return !p || std::string_view(p) != "0";
    }();
    if (many && scratch && rows > 8U && (split < 0 ? split_default : split != 0)) {
        linear_rows(gate, x, rows, out, q, failure, true, variant);
        linear_rows(up, x, rows, scratch, q, failure, true, variant);
        if (failure && fuse_finite) {
            gelu_product_finite(out, scratch, rows * gate.outputs, failure, q);
            return;
        }
        gelu_product(out, scratch, rows * gate.outputs, q);
        if (failure)
            check_finite(out, std::size_t(rows) * gate.outputs, failure, q);
        return;
    }
    if (many && many_rows<true>(gate, up, x, rows, out, q, failure, variant))
        return;
    for (unsigned r = 0; r < rows; r += 8U)
        gpu::gate_up(gate, up, x + std::size_t(r) * gate.inputs, std::min(8U, rows - r),
                     out + std::size_t(r) * gate.outputs, true, q, failure);
}
} // namespace strix::gemma4::gpu
