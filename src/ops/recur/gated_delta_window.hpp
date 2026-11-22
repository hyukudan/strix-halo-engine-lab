#pragma once
// F9 `recur`, kind gated_delta, window form (HIP backend body). A window is a short run of known rows (1..max_rows) of
// one recurrent layer: one launch prepares every row (short conv + SiLU, q/k L2 scaling, gate prologue), one launch runs
// the recurrence of every row with the state tile in registers, one launch applies the gated RMS output norm.
// Deferred commit: the rows of a window are saved into slot `next` and committed (conv history + recurrence, no output)
// by the following window before its own rows; a window never commits its own rows. The cohort forms run several
// windows (each with its own state, history, saved rows and control) in one launch, blockIdx.y = segment. The prefill
// form is the same per-row arithmetic over T strided rows with an empty pending set, committed once after the last row.
//
// Arithmetic (every expression and reduction tree is part of the form; lifted verbatim from the former private kernels):
//   conv     x = encode(in); sum = sum_t decode(hist[t]) * w[ch * taps + t] (from +0, tap order, TU contraction);
//            v = Round(sum / (1 + expf(-sum)))
//   q/k      v = v * rsqrtf(sum128(v * v) + eps) * (key ? 1 : rsqrtf(head_dim))
//   gates    gated_delta_gate_body<Gate, Round, Load> (gated_delta_prologue.hpp)
//   step     decay = expf(gate); S *= decay; delta = (v - key_sum(S * k)) * beta; S += k * delta; o = key_sum(S * q)
//   norm     o = v * rsqrtf(sum128(v * v) / head_dim + eps) * w[lane] * (1 / (1 + expf(-z)))
// sum128 is the 128-leaf tree (strides 64 and 32 in LDS, 16..1 by shuffles); key_sum is the same tree with lane l
// holding keys l, l+32, l+64, l+96 ((p0 + p2) + (p1 + p3) in lane, then 16..1 by permlanex16 / DPP row_shl); adds are
// plain IEEE adds hidden from contraction (`add`).
//
// Binder B (engine hooks): B::Codec (Slot, encode, decode, round), B::Load (Kind, load), B::Gate (gate prologue policy),
// B::Control (pending_rows, pending_slot), B::max_rows (saved slot rows, segment table size), B::taps (conv taps).
// Shapes: head_dim 128 per wave (4 keys per lane), Columns value columns per wave (8 default; 4/2 or scatter variants).
#include <hip/hip_runtime.h>

#include "attn_core/recurrent_state.hpp"
#include "ops/recur/gated_delta_policy.hpp"
#include "ops/recur/gated_delta_prologue.hpp"
#include "strix/gdn_recurrent_storage.hpp"
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip_wave32_dpp.hpp"
#include "strix/recurrent_state_type.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ops::recur::window {

namespace rstate = ::strix::attn_core::rstate;
using ::strix::GdnRecurrentStorage;
using ::strix::RecurrentStateType;

inline constexpr std::uint32_t kWindowThreads = kStepThreads;  // prepare / norm block: one 128-channel head
inline constexpr std::uint32_t kWindowColumns = 8;             // value columns per recurrence wave (default)
inline constexpr std::uint32_t kTiledColumns = 32;             // value columns per tiled-recurrence block

// Plain IEEE add hidden from contraction; not volatile, so independent trees interleave.
__device__ __forceinline__ float add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
template <unsigned Rows> struct RowSet {
    float* qkv[Rows]; float* z[Rows]; float* beta[Rows]; float* gate[Rows]; float* core[Rows];
};
// Saved rows: slot s, row r at base + (s * max_rows + r) * row_bytes; offset 0 holds the conv input slots.
struct SavedRows { std::uintptr_t base; std::uint64_t row_bytes, k, v, beta, gate; };
template <std::uint32_t MaxRows, class T>
__device__ __forceinline__ T* saved_row(const SavedRows& saved, std::uint32_t slot, std::uint32_t row,
    std::uint64_t offset) {
    return reinterpret_cast<T*>(saved.base + (static_cast<std::uint64_t>(slot) * MaxRows + row) *
        saved.row_bytes + offset);
}
template <std::uint32_t MaxRows, class Control>
__device__ __forceinline__ std::uint32_t pending_rows(const Control* control) {
    return min(control->pending_rows, MaxRows);
}
// The 128-leaf tree: strides 64 and 32 through LDS, 16..1 by shuffles.
__device__ __forceinline__ float sum128(float* scratch, std::uint32_t lane, float value) {
    scratch[lane] = value;
    __syncthreads();
    if (lane < 64U) scratch[lane] = add(scratch[lane], scratch[lane + 64U]);
    __syncthreads();
    if (lane < 32U) {
        float sum = add(scratch[lane], scratch[lane + 32U]);
#pragma unroll
        for (unsigned distance = 16U; distance; distance /= 2U)
            sum = add(sum, __shfl_down(sum, distance, 32));
        if (lane == 0U) scratch[0] = sum;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}
template <class Slot, std::uint32_t Taps>
__device__ __forceinline__ void shift(Slot (&window)[Taps], Slot value) {
#pragma unroll
    for (std::uint32_t index = 0; index + 1U < Taps; ++index) window[index] = window[index + 1U];
    window[Taps - 1U] = value;
}
// Conv history commit of the pending rows, then conv/SiLU, q/k L2 norm and beta/gate transforms of every window row.
// Blocks own one 128-channel head; the trailing block owns the per-head beta/gate scalars.
template <class B, unsigned Rows, bool ParallelGates = false>
__device__ __forceinline__ void prepare_rows(const RowSet<Rows>& rows, typename B::Codec::Slot* conv_state,
    const void* conv_weights, typename B::Load::Kind conv_kind, const void* dt, typename B::Load::Kind dt_kind, const void* a,
    typename B::Load::Kind a_kind, const SavedRows& saved, const typename B::Control* control, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, float epsilon, float* scratch) {
    using Codec = typename B::Codec;
    using Load = typename B::Load;
    using Slot = typename Codec::Slot;
    constexpr std::uint32_t taps = B::taps, max_rows = B::max_rows;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), block = static_cast<std::uint32_t>(blockIdx.x);
    const auto pending = pending_rows<max_rows>(control), slot = control->pending_slot & 1U, next = slot ^ 1U;
    const auto key_width = key_heads * head_dim;
    if (block >= key_heads * 2U + value_heads) {
        if constexpr (ParallelGates) {
            // One (head, row) pair per thread instead of a serial row loop per head; same per-element arithmetic.
            for (auto item = lane; item < value_heads * Rows; item += kWindowThreads) {
                const auto head = item % value_heads, row = item / value_heads;
                float* beta = rows.beta[row]; float* gate = rows.gate[row];
                gated_delta_gate_body<typename B::Gate, Codec, Load>(beta, gate, dt, dt_kind, a, a_kind, head);
                saved_row<max_rows, float>(saved, next, row, saved.beta)[head] = beta[head];
                saved_row<max_rows, float>(saved, next, row, saved.gate)[head] = gate[head];
            }
            return;
        }
        for (auto head = lane; head < value_heads; head += kWindowThreads) {
            for (unsigned row = 0; row < Rows; ++row) {
                float* beta = rows.beta[row]; float* gate = rows.gate[row];
                gated_delta_gate_body<typename B::Gate, Codec, Load>(beta, gate, dt, dt_kind, a, a_kind, head);
                saved_row<max_rows, float>(saved, next, row, saved.beta)[head] = beta[head];
                saved_row<max_rows, float>(saved, next, row, saved.gate)[head] = gate[head];
            }
        }
        return;
    }
    const auto channel = block * head_dim + lane;
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * taps;
    Slot window[taps];
#pragma unroll
    for (std::uint32_t index = 0; index < taps; ++index) window[index] = history[index];
    for (std::uint32_t row = 0; row < pending; ++row)
        shift(window, saved_row<max_rows, const Slot>(saved, slot, row, 0)[channel]);
    if (pending) {
#pragma unroll
        for (std::uint32_t index = 0; index < taps; ++index) history[index] = window[index];
    }
    const bool is_qk = block < key_heads * 2U, is_key = is_qk && block >= key_heads;
    for (unsigned row = 0; row < Rows; ++row) {
        float* qkv = rows.qkv[row];
        shift(window, Codec::encode(qkv[channel]));
        saved_row<max_rows, Slot>(saved, next, row, 0)[channel] = window[taps - 1U];
        float sum = 0.0F;
        for (std::uint32_t index = 0; index < taps; ++index)
            sum += Codec::decode(window[index]) *
                Load::load(conv_weights, conv_kind, static_cast<std::uint64_t>(channel) * taps + index);
        const float silu = sum / (1.0F + expf(-sum));
        float value = Codec::round(silu);
        if (is_qk) {
            const float inverse = rsqrtf(sum128(scratch, lane, value * value) + epsilon);
            const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
            value = value * inverse * q_scale;
            if (is_key) saved_row<max_rows, float>(saved, next, row, saved.k)[channel - key_width] = value;
        } else saved_row<max_rows, float>(saved, next, row, saved.v)[channel - key_width * 2U] = value;
        qkv[channel] = value;
    }
}
template <class B, unsigned Rows, bool ParallelGates = false>
__global__ __launch_bounds__(kWindowThreads)
void prepare_kernel(RowSet<Rows> rows, typename B::Codec::Slot* conv_state, const void* conv_weights,
    typename B::Load::Kind conv_kind, const void* dt, typename B::Load::Kind dt_kind, const void* a,
    typename B::Load::Kind a_kind, SavedRows saved, const typename B::Control* control, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kWindowThreads];
    prepare_rows<B, Rows, ParallelGates>(rows, conv_state, conv_weights, conv_kind, dt, dt_kind, a, a_kind, saved,
        control, key_heads, value_heads, head_dim, epsilon, scratch);
}
// Cohorts: the prepare / recurrence of every session's segment (its own state, history, saved rows and control) in one
// launch, blockIdx.y = segment; each segment runs the per-session kernel's body.
template <class B> struct Segment {
    RowSet<B::max_rows> rows;
    std::uint32_t count;
    typename B::Codec::Slot* conv_state;
    float* matrix;
    SavedRows saved;
    const typename B::Control* control;
};
template <class B> struct Segments { Segment<B> segment[B::max_rows]; };
template <unsigned Rows, unsigned All> __device__ __forceinline__ RowSet<Rows> rows_prefix(const RowSet<All>& all) {
    RowSet<Rows> out;
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        out.qkv[r] = all.qkv[r]; out.z[r] = all.z[r]; out.beta[r] = all.beta[r]; out.gate[r] = all.gate[r]; out.core[r] = all.core[r];
    }
    return out;
}
// Established segment kernel: 2, 4 or max_rows rows.
template <class B, bool ParallelGates>
__global__ __launch_bounds__(kWindowThreads)
void prepare_segments_kernel(Segments<B> segments, const void* conv_weights, typename B::Load::Kind conv_kind,
    const void* dt, typename B::Load::Kind dt_kind, const void* a, typename B::Load::Kind a_kind, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    static_assert(B::max_rows == 8U);
    __shared__ float scratch[kWindowThreads];
    const auto& s = segments.segment[blockIdx.y];
    switch (s.count) {
    case 2: prepare_rows<B, 2, ParallelGates>(rows_prefix<2>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    case 4: prepare_rows<B, 4, ParallelGates>(rows_prefix<4>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    default: prepare_rows<B, 8, ParallelGates>(rows_prefix<8>(s.rows), s.conv_state, conv_weights, conv_kind, dt, dt_kind,
        a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break;
    }
}
template <GdnRecurrentStorage Storage>
__device__ __forceinline__ std::uint64_t state_index(std::uint32_t value_head, std::uint32_t lane,
    std::uint32_t value_index, std::uint32_t head_dim) {
    return (static_cast<std::uint64_t>(value_head) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? value_index : lane)) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? lane : value_index);
}
// One wave per (value head, Columns-column tile). Lane l holds keys l, l+32, l+64, l+96, so the 128-leaf tree's
// stride-64 and stride-32 levels are in-lane, (p[l] + p[l+64]) + (p[l+32] + p[l+96]), and strides 16..1 are cross-lane.
// Lane 0 holds the root; it is broadcast so every lane derives the same delta.
// Lane l < d receives lane l + d: permlanex16 across the two rows, then DPP row_shl.
template <unsigned Distance>
__device__ __forceinline__ float down(float value) {
    if constexpr (Distance == 16U) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(STRIX_HIP_EXCHANGE16(bits, bits, false, false));
    } else {
        const auto bits = __float_as_int(value);
        return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x100 + static_cast<int>(Distance), 0xf, 0xf, false));
    }
}
__device__ __forceinline__ float key_sum(const float (&p)[4]) {
    float sum = add(add(p[0], p[2]), add(p[1], p[3]));
    sum = add(sum, down<16U>(sum));
    sum = add(sum, down<8U>(sum));
    sum = add(sum, down<4U>(sum));
    sum = add(sum, down<2U>(sum));
    sum = add(sum, down<1U>(sum));
    return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
}
template <std::uint32_t Columns> struct Inputs { float key[4], query[4], value[Columns], beta, gate; };
// Scatter variant: the Columns roots are reduced together, column c's tree finishing in lane root(c).
template <unsigned Distance, unsigned Count>
__device__ __forceinline__ void scatter(float* sums, unsigned lane) {
#pragma unroll
    for (unsigned c = 0; c < Count / 2U; ++c) {
        const bool upper = (lane & Distance) != 0U;
        const float local = sums[2U * c + (upper ? 1U : 0U)];
        const float peer = ::strix::hip_wave32_dpp::xor32<Distance>(sums[2U * c + (upper ? 0U : 1U)]);
        // Always add the lower-key subtree first, including in upper lanes.
        sums[c] = upper ? add(peer, local) : add(local, peer);
    }
}
template <unsigned Columns>
__device__ __forceinline__ void column_sums(float (&sums)[Columns]) {
    const auto lane = static_cast<unsigned>(threadIdx.x) & 31U;
    static_assert(Columns == 4U || Columns == 8U);
    scatter<16U, Columns>(sums, lane);
    scatter<8U, Columns / 2U>(sums, lane);
    if constexpr (Columns == 8U) scatter<4U, 2U>(sums, lane);
    float sum = sums[0];
    if constexpr (Columns == 4U) sum = add(sum, down<4U>(sum));
    sum = add(sum, down<2U>(sum));
    sum = add(sum, down<1U>(sum));
#pragma unroll
    for (unsigned c = 0; c < Columns; ++c) {
        const auto root = (c & 1U) * 16U + (c & 2U) * 4U + (c & 4U);
        sums[c] = __shfl(sum, static_cast<int>(root));
    }
}

template <bool Output, std::uint32_t Columns = kWindowColumns, bool Scatter = false>
__device__ __forceinline__ void step(float (&retained)[4][Columns], const Inputs<Columns>& in,
    float* core) {
    float deltas[Columns];
    const float decay = expf(in.gate);
#pragma unroll
    for (std::uint32_t column = 0; column < Columns; ++column) {
        float products[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            retained[part][column] = retained[part][column] * decay;
            products[part] = retained[part][column] * in.key[part];
        }
        if constexpr (Scatter)
            deltas[column] = add(add(products[0], products[2]), add(products[1], products[3]));
        else deltas[column] = (in.value[column] - key_sum(products)) * in.beta;
    }
    if constexpr (Scatter) {
        column_sums(deltas);
#pragma unroll
        for (unsigned c = 0; c < Columns; ++c) deltas[c] = (in.value[c] - deltas[c]) * in.beta;
    }
    float output[Columns];
#pragma unroll
    for (std::uint32_t column = 0; column < Columns; ++column) {
        float products[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            retained[part][column] += in.key[part] * deltas[column];
            products[part] = retained[part][column] * in.query[part];
        }
        if constexpr (Output) {
            if constexpr (Scatter)
                output[column] = add(add(products[0], products[2]), add(products[1], products[3]));
            else {
                const float sum = key_sum(products);
                if ((threadIdx.x & 31U) == 0U) core[column] = sum;
            }
        }
    }
    if constexpr (Output && Scatter) {
        column_sums(output);
        if ((threadIdx.x & 31U) == 0U)
#pragma unroll
            for (unsigned c = 0; c < Columns; ++c) core[c] = output[c];
    }
}
// 16-bit state contract: a stored state is the narrowed value, so the register state of a window is narrowed and widened
// after every row exactly as the serial row stores it and the next row loads it (identity, compiled out, for f32).
template <RecurrentStateType S, std::uint32_t Columns>
__device__ __forceinline__ void round_state(float (&retained)[4][Columns]) {
    if constexpr (S != RecurrentStateType::f32) {
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
            for (std::uint32_t column = 0; column < Columns; ++column)
                retained[part][column] = rstate::round_in_register<S>(retained[part][column]);
    }
}
// Read the tile once, apply the pending rows and write it once (if any), then run the window rows in registers. Row
// inputs are loaded ahead of their step. WholeHead: the block's waves own consecutive tiles of one head.
template <class B, unsigned Rows, GdnRecurrentStorage Storage, std::uint32_t Columns = kWindowColumns, bool Scatter = false,
    bool WholeHead = false, RecurrentStateType S = RecurrentStateType::f32>
__device__ __forceinline__ void recurrence(const RowSet<Rows ? Rows : 1U>& rows, float* state_raw,
    const SavedRows& saved, const typename B::Control* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    constexpr std::uint32_t max_rows = B::max_rows;
    auto* state = reinterpret_cast<typename rstate::Codec<S>::Elem*>(state_raw);
    const auto lane = WholeHead ? static_cast<std::uint32_t>(threadIdx.x) & 31U : static_cast<std::uint32_t>(threadIdx.x);
    const auto tiles = head_dim / Columns;
    const auto raw_tile = WholeHead ? static_cast<std::uint32_t>(blockIdx.x) * tiles + static_cast<std::uint32_t>(threadIdx.x) / 32U :
        static_cast<std::uint32_t>(blockIdx.x);
    // Each complete wave owns one tile. Make that wave-uniform address explicit
    // so its row values, beta and gate retain the scalar placement.
    const auto tile = WholeHead ? static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(raw_tile))) : raw_tile;
    const auto value_head = tile / tiles;
    const auto value_base = tile % tiles * Columns;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto load = [&](const float* key, const float* query, const float* value, float beta, float gate) {
        Inputs<Columns> in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = key[part * 32U + lane];
            if (query) in.query[part] = query[part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) in.value[column] = value[column];
        in.beta = beta; in.gate = gate;
        return in;
    };
    const auto pending_inputs = [&](std::uint32_t slot, std::uint32_t row) {
        return load(saved_row<max_rows, const float>(saved, slot, row, saved.k) + key_offset, nullptr,
            saved_row<max_rows, const float>(saved, slot, row, saved.v) + value_offset,
            saved_row<max_rows, const float>(saved, slot, row, saved.beta)[value_head],
            saved_row<max_rows, const float>(saved, slot, row, saved.gate)[value_head]);
    };
    const auto window_inputs = [&](unsigned row) {
        const float* qkv = rows.qkv[row];
        return load(qkv + key_width + key_offset, qkv + key_offset, qkv + key_width * 2U + value_offset,
            rows.beta[row][value_head], rows.gate[row][value_head]);
    };
    Inputs<Columns> window[Rows ? Rows : 1U];
    if constexpr (Rows != 0U && !Scatter) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) window[row] = window_inputs(row);
    }
    float retained[4][Columns];
    // 16-bit key-value runs (Columns contiguous elements per key row) move packed: one access per row run.
    constexpr bool packed = S != RecurrentStateType::f32 && Storage == GdnRecurrentStorage::key_value;
    if constexpr (packed) {
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
            rstate::load_run<S, Columns>(state + state_index<Storage>(value_head, part * 32U + lane, value_base, head_dim),
                retained[part]);
    } else {
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column)
            retained[part][column] = rstate::Codec<S>::load(
                state[state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)]);
    }
    const auto pending = pending_rows<max_rows>(control), slot = control->pending_slot & 1U;
    if (pending) {
        auto next = pending_inputs(slot, 0U);
        for (std::uint32_t row = 0; row < pending; ++row) {
            const auto in = next;
            if (row + 1U < pending) next = pending_inputs(slot, row + 1U);
            step<false, Columns, Scatter>(retained, in, nullptr);
            round_state<S, Columns>(retained);
        }
        if constexpr (packed) {
#pragma unroll
            for (std::uint32_t part = 0; part < 4U; ++part)
                rstate::store_run<S, Columns>(
                    state + state_index<Storage>(value_head, part * 32U + lane, value_base, head_dim), retained[part]);
        } else {
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
            for (std::uint32_t column = 0; column < Columns; ++column)
                state[state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)] =
                    rstate::Codec<S>::store(retained[part][column]);
        }
    }
    if constexpr (Rows != 0U) {
        if constexpr (Scatter) {
            // Keep only the current row's inputs live next to the scatter
            // accumulators. Preloading all eight rows raises VGPR pressure.
#pragma unroll 1
            for (unsigned row = 0; row < Rows; ++row) {
                step<true, Columns, true>(retained, window_inputs(row), rows.core[row] + value_offset);
                round_state<S, Columns>(retained);
            }
        } else {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            step<true, Columns, Scatter>(retained, window[row], rows.core[row] + value_offset);
            round_state<S, Columns>(retained);
        }
        }
    }
}
// Tiled (key-value storage, f32 state): one block of 32/Columns waves per (value head, 32 columns). The block reads and
// writes its 128 x 32 state slice through LDS with 128-byte coalesced rows; each wave then runs Columns columns with the
// same lane/key mapping and the same per-column arithmetic as recurrence().
template <class B, unsigned Rows, std::uint32_t Columns>
__global__ __launch_bounds__(32U * (kTiledColumns / Columns))
void recurrent_tiled_kernel(RowSet<Rows> rows, float* state, SavedRows saved,
    const typename B::Control* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    constexpr std::uint32_t max_rows = B::max_rows;
    constexpr std::uint32_t threads = 32U * (kTiledColumns / Columns);
    __shared__ float tile[128][kTiledColumns + 1U];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U, wave = static_cast<std::uint32_t>(threadIdx.x) / 32U;
    const auto blocks = head_dim / kTiledColumns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / blocks;
    const auto block_base = static_cast<std::uint32_t>(blockIdx.x) % blocks * kTiledColumns;
    const auto local = wave * Columns, value_base = block_base + local;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto load = [&](const float* key, const float* query, const float* value, float beta, float gate) {
        Inputs<Columns> in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = key[part * 32U + lane];
            if (query) in.query[part] = query[part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) in.value[column] = value[column];
        in.beta = beta; in.gate = gate;
        return in;
    };
    const auto pending_inputs = [&](std::uint32_t slot, std::uint32_t row) {
        return load(saved_row<max_rows, const float>(saved, slot, row, saved.k) + key_offset, nullptr,
            saved_row<max_rows, const float>(saved, slot, row, saved.v) + value_offset,
            saved_row<max_rows, const float>(saved, slot, row, saved.beta)[value_head],
            saved_row<max_rows, const float>(saved, slot, row, saved.gate)[value_head]);
    };
    Inputs<Columns> window[Rows];
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        const float* qkv = rows.qkv[row];
        window[row] = load(qkv + key_width + key_offset, qkv + key_offset, qkv + key_width * 2U + value_offset,
            rows.beta[row][value_head], rows.gate[row][value_head]);
    }
    // Coalesced slice read: key k, 16-byte chunk c of the 32 columns (key-value storage: row k is contiguous).
    float* const slice = state + static_cast<std::uint64_t>(value_head) * head_dim * head_dim + block_base;
    for (std::uint32_t i = threadIdx.x; i < 128U * (kTiledColumns / 4U); i += threads) {
        const auto key = i / (kTiledColumns / 4U), chunk = i % (kTiledColumns / 4U);
        const float4 v = *reinterpret_cast<const float4*>(slice + static_cast<std::uint64_t>(key) * head_dim + chunk * 4U);
        tile[key][chunk * 4U] = v.x; tile[key][chunk * 4U + 1U] = v.y; tile[key][chunk * 4U + 2U] = v.z; tile[key][chunk * 4U + 3U] = v.w;
    }
    __syncthreads();
    float retained[4][Columns];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < Columns; ++column) retained[part][column] = tile[part * 32U + lane][local + column];
    const auto pending = pending_rows<max_rows>(control), slot = control->pending_slot & 1U;
    if (pending) {
        auto next = pending_inputs(slot, 0U);
        for (std::uint32_t row = 0; row < pending; ++row) {
            const auto in = next;
            if (row + 1U < pending) next = pending_inputs(slot, row + 1U);
            step<false, Columns>(retained, in, nullptr);
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
            for (std::uint32_t column = 0; column < Columns; ++column) tile[part * 32U + lane][local + column] = retained[part][column];
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < 128U * (kTiledColumns / 4U); i += threads) {
            const auto key = i / (kTiledColumns / 4U), chunk = i % (kTiledColumns / 4U);
            *reinterpret_cast<float4*>(slice + static_cast<std::uint64_t>(key) * head_dim + chunk * 4U) =
                float4{tile[key][chunk * 4U], tile[key][chunk * 4U + 1U], tile[key][chunk * 4U + 2U], tile[key][chunk * 4U + 3U]};
        }
    }
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) step<true, Columns>(retained, window[row], rows.core[row] + value_offset);
}
template <class B, unsigned Rows, GdnRecurrentStorage Storage, std::uint32_t Columns = kWindowColumns, bool Scatter = false,
    RecurrentStateType S = RecurrentStateType::f32>
__global__ __launch_bounds__(32)
void recurrent_kernel(RowSet<Rows> rows, float* state, SavedRows saved,
    const typename B::Control* control, std::uint32_t key_heads, std::uint32_t head_dim) {
    recurrence<B, Rows, Storage, Columns, Scatter, false, S>(rows, state, saved, control, key_heads, head_dim);
}
template <class B, GdnRecurrentStorage Storage, std::uint32_t Columns, RecurrentStateType S = RecurrentStateType::f32>
__global__ __launch_bounds__(32)
void recurrent_segments_kernel(Segments<B> segments, std::uint32_t key_heads, std::uint32_t head_dim) {
    static_assert(B::max_rows == 8U);
    const auto& s = segments.segment[blockIdx.y];
    switch (s.count) {
    case 2: recurrence<B, 2, Storage, Columns, false, false, S>(rows_prefix<2>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    case 4: recurrence<B, 4, Storage, Columns, false, false, S>(rows_prefix<4>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    default: recurrence<B, 8, Storage, Columns, false, false, S>(rows_prefix<8>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break;
    }
}
// Separate entry points for any count 1..8: adding cases must not change the register allocation of the established
// 2/4/8 segment kernels above. Each case uses the exact single-session Rows specialization.
template <class B, bool ParallelGates>
__global__ __launch_bounds__(kWindowThreads)
void prepare_anyseg_kernel(Segments<B> segments, const void* conv_weights, typename B::Load::Kind conv_kind,
    const void* dt, typename B::Load::Kind dt_kind, const void* a, typename B::Load::Kind a_kind, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    static_assert(B::max_rows == 8U);
    __shared__ float scratch[kWindowThreads];
    const auto& s = segments.segment[blockIdx.y];
#define STRIX_RECUR_WINDOW_PREPARE_CASE(N) case N: \
    prepare_rows<B, N, ParallelGates>(rows_prefix<N>(s.rows), s.conv_state, conv_weights, conv_kind, \
        dt, dt_kind, a, a_kind, s.saved, s.control, key_heads, value_heads, head_dim, epsilon, scratch); break
    switch (s.count) {
    STRIX_RECUR_WINDOW_PREPARE_CASE(1); STRIX_RECUR_WINDOW_PREPARE_CASE(2); STRIX_RECUR_WINDOW_PREPARE_CASE(3);
    STRIX_RECUR_WINDOW_PREPARE_CASE(4); STRIX_RECUR_WINDOW_PREPARE_CASE(5); STRIX_RECUR_WINDOW_PREPARE_CASE(6);
    STRIX_RECUR_WINDOW_PREPARE_CASE(7); STRIX_RECUR_WINDOW_PREPARE_CASE(8);
    }
#undef STRIX_RECUR_WINDOW_PREPARE_CASE
}
template <class B, GdnRecurrentStorage Storage, std::uint32_t Columns, RecurrentStateType S = RecurrentStateType::f32>
__global__ __launch_bounds__(32)
void recurrent_anyseg_kernel(Segments<B> segments, std::uint32_t key_heads, std::uint32_t head_dim) {
    static_assert(B::max_rows == 8U);
    const auto& s = segments.segment[blockIdx.y];
#define STRIX_RECUR_WINDOW_RECURRENT_CASE(N) case N: \
    recurrence<B, N, Storage, Columns, false, false, S>(rows_prefix<N>(s.rows), s.matrix, s.saved, s.control, key_heads, head_dim); break
    switch (s.count) {
    STRIX_RECUR_WINDOW_RECURRENT_CASE(1); STRIX_RECUR_WINDOW_RECURRENT_CASE(2); STRIX_RECUR_WINDOW_RECURRENT_CASE(3);
    STRIX_RECUR_WINDOW_RECURRENT_CASE(4); STRIX_RECUR_WINDOW_RECURRENT_CASE(5); STRIX_RECUR_WINDOW_RECURRENT_CASE(6);
    STRIX_RECUR_WINDOW_RECURRENT_CASE(7); STRIX_RECUR_WINDOW_RECURRENT_CASE(8);
    }
#undef STRIX_RECUR_WINDOW_RECURRENT_CASE
}
// Standalone commit of the pending rows: recurrence tiles, then 32-channel conv groups.
template <class B, GdnRecurrentStorage Storage, RecurrentStateType S = RecurrentStateType::f32>
__global__ __launch_bounds__(32)
void commit_kernel(float* state, typename B::Codec::Slot* conv_state, SavedRows saved,
    const typename B::Control* control, std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim) {
    using Slot = typename B::Codec::Slot;
    constexpr std::uint32_t taps = B::taps, max_rows = B::max_rows;
    const auto tiles = value_heads * (head_dim / kWindowColumns);
    if (blockIdx.x < tiles) {
        recurrence<B, 0U, Storage, kWindowColumns, false, false, S>({}, state, saved, control, key_heads, head_dim);
        return;
    }
    const auto pending = pending_rows<max_rows>(control), slot = control->pending_slot & 1U;
    if (!pending) return;
    const auto channel = (static_cast<std::uint32_t>(blockIdx.x) - tiles) * 32U + static_cast<std::uint32_t>(threadIdx.x);
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * taps;
    Slot window[taps];
#pragma unroll
    for (std::uint32_t index = 0; index < taps; ++index) window[index] = history[index];
    for (std::uint32_t row = 0; row < pending; ++row)
        shift(window, saved_row<max_rows, const Slot>(saved, slot, row, 0)[channel]);
#pragma unroll
    for (std::uint32_t index = 0; index < taps; ++index) history[index] = window[index];
}
// Gated RMS output norm of every window row: grid (value heads, Rows), one 128-lane block per (head, row).
template <class B, unsigned Rows>
__global__ __launch_bounds__(kWindowThreads)
void norm_kernel(RowSet<Rows> rows, const void* norm_weights, typename B::Load::Kind norm_kind,
    std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kWindowThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), row = static_cast<std::uint32_t>(blockIdx.y);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    float* core = rows.core[row];
    const float value = core[index];
    const float inv = rsqrtf(sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
    const float gate_value = 1.0F / (1.0F + expf(-rows.z[row][index]));
    core[index] = value * inv * B::Load::load(norm_weights, norm_kind, lane) * gate_value;
}

// Prefill form: T consecutive known rows of one layer, committed in order with no deferral. Row r's workspace is
// base + r * stride. Same per-row expressions and trees as the window kernels with an empty pending set; the conv
// history and the recurrent state are written once, after the last row.
struct StridedRows {
    float* qkv; float* z; float* beta; float* gate; float* core; std::uint64_t stride;
};
template <class B>
__global__ __launch_bounds__(kWindowThreads)
void prefill_prepare_kernel(StridedRows rows, std::uint32_t count, typename B::Codec::Slot* conv_state,
    const void* conv_weights, typename B::Load::Kind conv_kind, const void* dt, typename B::Load::Kind dt_kind, const void* a,
    typename B::Load::Kind a_kind, std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim, float epsilon) {
    using Codec = typename B::Codec;
    using Load = typename B::Load;
    using Slot = typename Codec::Slot;
    constexpr std::uint32_t taps = B::taps;
    __shared__ float scratch[kWindowThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), block = static_cast<std::uint32_t>(blockIdx.x);
    if (block >= key_heads * 2U + value_heads) {
        for (auto head = lane; head < value_heads; head += kWindowThreads) {
            for (std::uint32_t row = 0; row < count; ++row) {
                float* beta = rows.beta + row * rows.stride; float* gate = rows.gate + row * rows.stride;
                gated_delta_gate_body<typename B::Gate, Codec, Load>(beta, gate, dt, dt_kind, a, a_kind, head);
            }
        }
        return;
    }
    const auto channel = block * head_dim + lane;
    auto* history = conv_state + static_cast<std::uint64_t>(channel) * taps;
    Slot window[taps];
#pragma unroll
    for (std::uint32_t index = 0; index < taps; ++index) window[index] = history[index];
    const bool is_qk = block < key_heads * 2U, is_key = is_qk && block >= key_heads;
    for (std::uint32_t row = 0; row < count; ++row) {
        float* qkv = rows.qkv + row * rows.stride;
        shift(window, Codec::encode(qkv[channel]));
        float sum = 0.0F;
        for (std::uint32_t index = 0; index < taps; ++index)
            sum += Codec::decode(window[index]) *
                Load::load(conv_weights, conv_kind, static_cast<std::uint64_t>(channel) * taps + index);
        const float silu = sum / (1.0F + expf(-sum));
        float value = Codec::round(silu);
        if (is_qk) {
            const float inverse = rsqrtf(sum128(scratch, lane, value * value) + epsilon);
            const float q_scale = is_key ? 1.0F : rsqrtf(static_cast<float>(head_dim));
            value = value * inverse * q_scale;
        }
        qkv[channel] = value;
    }
#pragma unroll
    for (std::uint32_t index = 0; index < taps; ++index) history[index] = window[index];
}
// f32 state only (the binder refuses 16-bit states for this form).
template <GdnRecurrentStorage Storage>
__global__ __launch_bounds__(32)
void prefill_recurrent_kernel(StridedRows rows, std::uint32_t count, float* state, std::uint32_t key_heads,
    std::uint32_t head_dim) {
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), tiles = head_dim / kWindowColumns;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / tiles;
    const auto value_base = static_cast<std::uint32_t>(blockIdx.x) % tiles * kWindowColumns;
    const auto key_offset = value_head % key_heads * head_dim, key_width = key_heads * head_dim;
    const auto value_offset = static_cast<std::uint64_t>(value_head) * head_dim + value_base;
    const auto row_inputs = [&](std::uint32_t row) {
        const float* qkv = rows.qkv + row * rows.stride;
        Inputs<kWindowColumns> in{};
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            in.key[part] = qkv[key_width + key_offset + part * 32U + lane];
            in.query[part] = qkv[key_offset + part * 32U + lane];
        }
#pragma unroll
        for (std::uint32_t column = 0; column < kWindowColumns; ++column)
            in.value[column] = qkv[key_width * 2U + value_offset + column];
        in.beta = rows.beta[row * rows.stride + value_head]; in.gate = rows.gate[row * rows.stride + value_head];
        return in;
    };
    float retained[4][kWindowColumns];
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < kWindowColumns; ++column)
            retained[part][column] = state[state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)];
    auto next = row_inputs(0U);
    for (std::uint32_t row = 0; row < count; ++row) {
        const auto in = next;
        if (row + 1U < count) next = row_inputs(row + 1U);
        step<true>(retained, in, rows.core + row * rows.stride + value_offset);
    }
#pragma unroll
    for (std::uint32_t part = 0; part < 4U; ++part)
#pragma unroll
        for (std::uint32_t column = 0; column < kWindowColumns; ++column)
            state[state_index<Storage>(value_head, part * 32U + lane, value_base + column, head_dim)] =
                retained[part][column];
}
template <class B>
__global__ __launch_bounds__(kWindowThreads)
void prefill_norm_kernel(StridedRows rows, const void* norm_weights, typename B::Load::Kind norm_kind,
    std::uint32_t head_dim, float epsilon) {
    __shared__ float scratch[kWindowThreads];
    const auto lane = static_cast<std::uint32_t>(threadIdx.x), row = static_cast<std::uint32_t>(blockIdx.y);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * head_dim + lane;
    float* core = rows.core + row * rows.stride;
    const float value = core[index];
    const float inv = rsqrtf(sum128(scratch, lane, value * value) / static_cast<float>(head_dim) + epsilon);
    const float gate_value = 1.0F / (1.0F + expf(-rows.z[row * rows.stride + index]));
    core[index] = value * inv * B::Load::load(norm_weights, norm_kind, lane) * gate_value;
}

}  // namespace strix::ops::recur::window
