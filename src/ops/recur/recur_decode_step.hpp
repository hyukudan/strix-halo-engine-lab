#pragma once

#include "core/decode_state_rows.hpp"
#include "recur_decode_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_RECUR_DECODE_HD __host__ __device__
#else
#define STRIX_RECUR_DECODE_HD
#endif

namespace strix::ops::recur {

struct DecodeRowStrides {
    // Element strides, with beta and log_decay using the same stride.
    std::size_t input, qkv, z, prepared, gate, out, raw_capture, a8_codes;
};

// Default to dense rows for the supplied, validated geometry.
[[nodiscard]] constexpr DecodeRowStrides packed_decode_strides(GdnDecodeShape shape) noexcept {
    return {shape.hidden, shape.channels(), shape.value_width(), shape.channels(),
            shape.value_heads, shape.value_width(), shape.channels(), shape.value_width()};
}

struct DecodeStep {
    const exec::StepBlock* descriptor;
    DecodeRow rows;
    DecodeRowStrides strides;
    ::strix::core::decode_state::SlotView history, matrix;
    // Disjoint, contiguous images indexed by descriptor row, including padding.
    // The caller publishes the selected pair by changing the two slot entries.
    // Pending images must be disjoint from all published images for this step.
    std::uint16_t* pending_history;
    void* pending_matrix;
    RecurrentStateType type;
    DecodeA8 a8; // Deltas are [value_width / 32][a8_pitch], column r.
};

template <class T>
struct DecodeOutputRef {
    T* direct = nullptr;
    T* const* slot = nullptr; // Device pointer word, stable for the life of the graph.
    std::uint64_t byte_offset = 0;
};

// One linear prefix per step, including R=1. The caller keeps the published
// input images and retains the inputs needed to rerun an accepted prefix.
// Replay words and the directory must stay stable throughout each invocation.
struct DecodeReplayStep {
    const exec::StepBlock* descriptor;
    DecodeRow rows;
    DecodeRowStrides strides;
    ::strix::core::decode_state::SlotView history, matrix;
    DecodeOutputRef<std::uint16_t> pending_history;
    DecodeOutputRef<float> pending_matrix; // Float-sized offsets for all codecs.
    RecurrentStateType type;
    DecodeA8 a8;
    std::uint64_t history_byte_offset = 0, matrix_byte_offset = 0; // From published entry bases.
};

template <class Binding>
[[nodiscard]] inline bool valid_decode_step_rows(GdnDecodeShape shape, unsigned width, const Binding& step) noexcept {
    if (!shape.valid() || width > 16U || !session::is_step_replay_width(width) ||
        !step.descriptor || !step.history.valid() || !step.matrix.valid() ||
        (step.type != RecurrentStateType::f32 && step.type != RecurrentStateType::f16 &&
         step.type != RecurrentStateType::bf16))
        return false;

    const auto fits = [width](std::size_t stride, std::size_t extent, std::size_t element) {
        const auto limit = std::numeric_limits<std::size_t>::max() / element;
        return extent <= limit && stride >= extent && (width == 1U || stride <= (limit - extent) / (width - 1U));
    };
    const auto& r = step.rows;
    const auto& s = step.strides;
    return r.input && r.qkv && r.z && r.prepared && r.beta && r.log_decay && r.out &&
        reinterpret_cast<std::uintptr_t>(r.prepared) % 16U == 0U && s.prepared % 4U == 0U &&
        (r.qkv != r.prepared || s.qkv == s.prepared) &&
        fits(s.input, shape.hidden, sizeof(std::uint16_t)) && fits(s.qkv, shape.channels(), sizeof(float)) &&
        fits(s.z, shape.value_width(), sizeof(float)) && fits(s.prepared, shape.channels(), sizeof(float)) &&
        fits(s.gate, shape.value_heads, sizeof(float)) && fits(s.out, shape.value_width(), sizeof(float)) &&
        (!r.raw_capture || fits(s.raw_capture, shape.channels(), sizeof(float))) &&
        bool(step.a8.codes) == bool(step.a8.deltas) &&
        (!step.a8.codes || (width <= shape.a8_pitch && fits(s.a8_codes, shape.value_width(), sizeof(std::int8_t))));
}

[[nodiscard]] inline bool valid_decode_step(GdnDecodeShape shape, unsigned width, const DecodeStep& step) noexcept {
    return step.pending_history && step.pending_matrix && valid_decode_step_rows(shape, width, step);
}

struct DecodeSource {
    const state::SlotEntry* history;
    const state::SlotEntry* matrix;
    unsigned parent;
};

template <class Binding>
[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeSource decode_step_source(const Binding& args, unsigned r) noexcept {
    using namespace ::strix::core::decode_state;
    const auto& step = *args.descriptor;
    if (!exec::step_row_valid(step, r)) return {};
    const auto& row = step.row[r];
    const auto* history = args.history.entry(row.slot);
    const auto* matrix = args.matrix.entry(row.slot);
    if (!accepts(history, row.position) || !accepts(matrix, row.position)) return {};
    const auto parent = row.parent;
    // Each link is in the same slot and advances one position. Checking the
    // root against both frontiers also rejects children of an inadmissible row.
    unsigned root = r;
    while (step.row[root].parent != exec::kStepNoParent) {
        root = parent_row(step, root);
        if (root == exec::kStepNoParent) return {};
    }
    const auto position = step.row[root].position;
    if (history->frontier != position || matrix->frontier != position) return {};
    return {history, matrix, parent};
}

[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeSource decode_source(const DecodeStep& args, unsigned r) noexcept {
    return decode_step_source(args, r);
}

template <class Binding>
[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeRow decode_row(const Binding& step, std::size_t r) noexcept {
    const auto& b = step.rows;
    const auto& s = step.strides;
    return {b.input + r * s.input, b.qkv + r * s.qkv, b.z + r * s.z,
            b.prepared + r * s.prepared, b.beta + r * s.gate, b.log_decay + r * s.gate,
            b.out + r * s.out, b.raw_capture ? b.raw_capture + r * s.raw_capture : nullptr};
}

struct DecodeImageRange {
    std::uint64_t begin, end;

    [[nodiscard]] STRIX_RECUR_DECODE_HD constexpr bool disjoint(DecodeImageRange other) const noexcept {
        return end <= other.begin || other.end <= begin;
    }
};

[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeImageRange decode_image_range(std::uint64_t base, std::uint64_t offset,
    std::uint64_t bytes, std::uint64_t alignment) noexcept {
    constexpr auto limit = UINTPTR_MAX;
    if (!base || base > limit || offset > limit - base) return {};
    const auto begin = base + offset;
    if (begin % alignment || bytes > limit - begin) return {};
    return {begin, begin + bytes};
}

template <class T>
[[nodiscard]] inline bool valid_decode_output(DecodeOutputRef<T> ref, std::uint64_t bytes) noexcept {
    if (bool(ref.direct) == bool(ref.slot) || ref.byte_offset % alignof(T)) return false;
    if (ref.slot) return reinterpret_cast<std::uintptr_t>(ref.slot) % alignof(T*) == 0U;
    return decode_image_range(reinterpret_cast<std::uintptr_t>(ref.direct), ref.byte_offset, bytes, alignof(T)).begin != 0U;
}

[[nodiscard]] inline bool valid_decode_replay_step(GdnDecodeShape shape, unsigned width,
                                                   const DecodeReplayStep& step) noexcept {
    return valid_decode_step_rows(shape, width, step) &&
        step.history_byte_offset % alignof(std::uint16_t) == 0U && step.matrix_byte_offset % alignof(float) == 0U &&
        valid_decode_output(step.pending_history, std::uint64_t{shape.history_elements()} * sizeof(std::uint16_t)) &&
        valid_decode_output(step.pending_matrix, shape.matrix_bytes(step.type));
}

template <class T>
[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeImageRange decode_output_range(DecodeOutputRef<T> ref,
                                                                               std::uint64_t bytes) noexcept {
    return decode_image_range(reinterpret_cast<std::uintptr_t>(ref.slot ? *ref.slot : ref.direct),
                              ref.byte_offset, bytes, alignof(T));
}

struct DecodeReplayImages {
    const std::uint16_t* history;
    const float* matrix;
    std::uint16_t* pending_history;
    float* pending_matrix;
    unsigned rows;
};

// Resolve and check before any writes, in every stage. A malformed replay
// masks off the entire prefix, including the row outputs.
template <GdnDecodeShape Shape>
[[nodiscard]] STRIX_RECUR_DECODE_HD inline DecodeReplayImages decode_replay_images(unsigned width,
                                                                                  const DecodeReplayStep& args) noexcept {
    using L = GdnDecodeLayout<Shape>;
    using ::strix::core::decode_state::parent_row;
    const auto& step = *args.descriptor;
    if (step.h.rows > width || step.row[0].parent != exec::kStepNoParent) return {};
    const auto src = decode_step_source(args, 0U);
    if (!src.history) return {};

    unsigned rows = 1U;
    for (unsigned r = 1U; r < step.h.rows; ++r) {
        if (!exec::step_row_valid(step, r)) continue;
        const auto position = step.row[r].position;
        if (r != rows || parent_row(step, r) != r - 1U ||
            position >= src.history->capacity || position >= src.matrix->capacity)
            return {};
        ++rows;
    }

    constexpr auto history_bytes = std::uint64_t{L::history_elements} * sizeof(std::uint16_t);
    constexpr auto f32_bytes = L::template matrix_bytes<RecurrentStateType::f32>;
    constexpr auto narrow_bytes = L::template matrix_bytes<RecurrentStateType::f16>;
    static_assert(narrow_bytes == L::template matrix_bytes<RecurrentStateType::bf16>);
    const auto matrix_bytes = args.type == RecurrentStateType::f32 ? f32_bytes : narrow_bytes;
    const auto history = decode_image_range(src.history->base, args.history_byte_offset, history_bytes, alignof(std::uint16_t));
    const auto matrix = decode_image_range(src.matrix->base, args.matrix_byte_offset, matrix_bytes, alignof(float));
    const auto pending_history = decode_output_range(args.pending_history, history_bytes);
    const auto pending_matrix = decode_output_range(args.pending_matrix, matrix_bytes);
    if (!history.begin || !matrix.begin || !pending_history.begin || !pending_matrix.begin ||
        !history.disjoint(matrix) || !history.disjoint(pending_history) || !history.disjoint(pending_matrix) ||
        !matrix.disjoint(pending_history) || !matrix.disjoint(pending_matrix) || !pending_history.disjoint(pending_matrix))
        return {};
    return {reinterpret_cast<const std::uint16_t*>(history.begin), reinterpret_cast<const float*>(matrix.begin),
            reinterpret_cast<std::uint16_t*>(pending_history.begin), reinterpret_cast<float*>(pending_matrix.begin), rows};
}

}  // namespace strix::ops::recur

#undef STRIX_RECUR_DECODE_HD
