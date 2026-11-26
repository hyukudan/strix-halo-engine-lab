#pragma once

// HIP side of the deepseek41 K2 routed experts (ds41_k2_experts.hpp): device copy of a LayerImage and the decode of
// (row, slot) entries on the shared trellis MoE family (src/moe_core/exl3_trellis.hpp expert_split_rows, Bits = 2).
//
// One entry = one (row, slot): expert ids[row * top_k + slot] (global id) with weight weights[...]. Steps, all on `stream`:
//   gate/up  per entry, matrix and gate split: the split's 128-blocks of x transformed with the matrix's suh into LDS,
//            one wave per 16-column tile column -> gu_part (one partial per split);
//   finish   per entry and 128-block of the rank's I range: ((0 + p_0) + p_1) + ... in ascending split order, H32, x svh,
//            ds41::swiglu_clamped(gate, up, limit), x down suh, RN32 then RN16, H32 -> h (F16, the down input);
//   down     per entry and down split of the rank (SplitPlan): the split's h channels into LDS -> down_part;
//   combine  (enqueue_k2_down_finish) per row and 128-block of H: per slot ((0 + p_0) + ...) over ALL down splits (a TP
//            caller first places the peer rank's partials in the same array), H32, x svh, then
//            y = ((w_0 y_0 + w_1 y_1) + ...) in slot order (FP32, no contraction).
// Bit-changing (FAST lever decode_k2_experts, identity ds41-expert-k2-v1); never on the default path.

#include "strix/ds41_k2_experts.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::k2 {

struct DeviceLayer {
    std::uint32_t layer = 0, expert_first = 0, expert_count = 0;
    std::uint32_t hidden = 0, inter = 0, local = 0;  // H, the expert's I, the rank's I range
    SplitPlan plan{};
    std::uint64_t tile_bytes = 0;  // per expert and matrix
    void* memory = nullptr;        // one allocation: gate | up | down | gate_suh | up_suh | down_suh | gate_svh | up_svh | down_svh
    const std::uint32_t *gate = nullptr, *up = nullptr, *down = nullptr;
    const std::uint16_t *gate_suh = nullptr, *up_suh = nullptr, *down_suh = nullptr;
    const std::uint16_t *gate_svh = nullptr, *up_svh = nullptr, *down_svh = nullptr;
};

// Uploads the image (one hipMalloc + one copy); free_layer releases it.
[[nodiscard]] DeviceLayer upload_layer(const LayerImage& image);
void free_layer(DeviceLayer& layer) noexcept;

struct DecodeView {
    const DeviceLayer* layer = nullptr;
    std::uint32_t rows = 0, top_k = 0;
    float swiglu_limit = 10.0F;
    const float* x = nullptr;             // [rows][H], original basis
    const std::int32_t* ids = nullptr;    // [rows][top_k], global expert ids inside [expert_first, expert_first + count)
    const float* weights = nullptr;       // [rows][top_k]
    float* gu_part = nullptr;             // scratch [entries][2][gate_splits][local]
    _Float16* h = nullptr;                // scratch [entries][local]
    float* down_part = nullptr;           // [entries][down_splits][H]; this rank writes splits [down_first, down_end)
    float* y = nullptr;                   // [rows][H] (enqueue_k2_down_finish)
};

// Scratch elements of a view (floats / halves).
[[nodiscard]] std::size_t gu_part_floats(const DeviceLayer& layer, std::uint32_t entries) noexcept;
[[nodiscard]] std::size_t down_part_floats(const DeviceLayer& layer, std::uint32_t entries) noexcept;

// gate/up + finish + this rank's down splits.
void enqueue_k2_experts(const DecodeView& view, void* stream);
// Sum of all down splits, output transform and the weighted combine into y.
void enqueue_k2_down_finish(const DecodeView& view, void* stream);

}  // namespace strix::ds41::k2
