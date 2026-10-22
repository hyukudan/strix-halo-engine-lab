#pragma once

// deepseek41 expert tensor parallelism over two ranks, declared identity ds41-expert-tp2-i1152-v1
// (docs/ds41-expert-tp2-i1152-v1.md). Default off; STRIX_DS41_EXPERT_TP=1 turns it on in the host forward, the device
// forward and the batch forward (read once at startup).
//
// Every routed expert is split by its intermediate dimension (I = 2304) over the two ranks: rank r holds 1152 gate and up
// rows (IQ2_XXS, H = 5120 each) and, for every down row (Q2_K, K = I = 9 blocks of 256), the 36 groups of 32 that
// multiply its 1152 activations. 1152 = 4.5 Q2_K blocks: the boundary block 4 is stored on both ranks (rank 0 uses its
// first half, groups 0..3; rank 1 its second half, groups 4..7, moved into the first half of a block of its own), so a
// rank's down row is 5 blocks (420 bytes) and the existing Q2_K reader takes its 36 groups as a row of width 1152.
//
// Local orders (local index -> global index):
//   rank 0: intermediate j -> j (j < 1152); down group g -> g (g < 36).
//   rank 1: intermediate j -> 1280 + j (j < 1024), 1152 + (j - 1024) (1024 <= j < 1152);
//           down group g -> 40 + g (g < 32), 36 + (g - 32) (32 <= g < 36).
// The local orders move whole 32-element groups (A8 groups and Q2_K groups) and the 128-element tail of rank 1 is one
// half block, so every A8 group, every gate/up output and every Q2_K group is the same as in the unsplit expert.

#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix::ds41::expert_tp {

inline constexpr const char* identity = "ds41-expert-tp2-i1152-v1";
inline constexpr std::uint32_t ranks = 2;
inline constexpr std::uint32_t inter = 2304;         // the full intermediate size this layout is declared for
inline constexpr std::uint32_t half = inter / 2U;     // 1152 intermediate rows per rank
inline constexpr std::uint32_t half_groups = half / 32U;  // 36 Q2_K / A8 groups per rank
inline constexpr std::uint32_t down_blocks = 5;       // Q2_K blocks stored per rank and down row (4.5 used)
inline constexpr std::uint64_t down_row_bytes = down_blocks * q2_k_block_bytes;  // 420
inline constexpr std::uint64_t half_block_bytes = 44;  // packed half block: scales 8, codes 32, d, dmin
inline constexpr std::uint64_t down_row_bytes_packed = 4U * q2_k_block_bytes + half_block_bytes;  // 380

// STRIX_DS41_EXPERT_TP=1 (read once): the flag set explicitly (the CPU host forward and the tests follow only this).
[[nodiscard]] bool enabled_from_env();
// The decode forwards' choice (DeviceForward, BatchForward), default on since P16: STRIX_DS41_EXPERT_TP=1 on (the
// caller rejects an unsuitable model or layout), =0 off, unset = the default when the layout allows it (two ranks,
// resident experts, I = 2304). The prefill keeps the expert-parallel identity (ds41-prefill-ep+decode-tp).
[[nodiscard]] bool decode_enabled(std::uint32_t world, std::uint32_t moe_inter, bool stream_experts);
// The default for an unset STRIX_DS41_EXPERT_TP (true; serve sets false: its device prefill reads the decode weights in place).
void set_decode_default(bool on);
// Down-row storage on the device (read once): the packed half-block rows (380 B, default) or, with
// STRIX_DS41_EXPERT_TP_LAYOUT=dup, the duplicated boundary block (420 B). Same group values, same bits.
[[nodiscard]] bool packed_from_env();
// Overrides the layout before any load (1 packed, 0 dup, -1 the environment): the expert-TP device prefill reads the
// dup rows (whole Q2_K blocks), so a server with expert TP and the device prefill loads them in that layout.
void set_packed(int mode);
[[nodiscard]] inline std::uint64_t device_down_row_bytes() { return packed_from_env() ? down_row_bytes_packed : down_row_bytes; }

// Global intermediate index of local index j of a rank (the table above).
[[nodiscard]] std::uint32_t inter_index(unsigned rank, std::uint32_t j);
// Global down group of local group g of a rank.
[[nodiscard]] std::uint32_t group_index(unsigned rank, std::uint32_t g);

// One rank's gate or up rows of one expert: [half][row_bytes] from the expert's [inter][row_bytes].
void pack_gate_rows(std::span<const std::uint8_t> expert, std::uint64_t row_bytes, unsigned rank, std::span<std::uint8_t> out);
// One rank's part of one Q2_K down row (9 blocks, 756 bytes) as 5 blocks (420 bytes): rank 0 blocks 0..4; rank 1 blocks
// 5..8 then block 4 with its second half (scales 8..15, codes 32..63) copied over its first half (d and dmin kept).
void pack_down_row(std::span<const std::uint8_t> row, unsigned rank, std::span<std::uint8_t> out);
// The packed form (380 bytes): the 4 whole blocks as above, then the rank's half of block 4 as one half block: its 8 scale
// bytes, its 32 code bytes, d and dmin.
void pack_down_row_packed(std::span<const std::uint8_t> row, unsigned rank, std::span<std::uint8_t> out);
// Group g of a packed row (the 4 whole blocks, then the half block for g >= 32), as q2_k_group.
[[nodiscard]] Q2kGroup packed_group(std::span<const std::uint8_t> row, std::uint32_t g);

// Declared K-half product: as product_q2_k over `groups` groups of a packed row (lane l accumulates groups l, l + 32, ...
// in ascending order from 0, then the butterfly), the width not restricted to whole blocks.
[[nodiscard]] float product_q2_k_groups(std::span<const std::uint8_t> row, const A8Activation& x, std::uint32_t groups);

struct WindowTp {
    std::vector<float> partial[ranks];  // [slot][hidden]: each rank's unweighted down part of every slot
    std::vector<float> sum[ranks];      // [row][hidden]: each rank's weighted fold of its parts (ascending expert id)
    std::vector<float> y;               // [row][hidden]: (shared + sum[0]) + sum[1]
};
// Reference of one window in the identity: tensors are the whole (unsplit) expert tensors, inter must be 2304 and the
// formats IQ2_XXS / Q2_K; routes.order gives every row's slots in ascending expert id.
[[nodiscard]] WindowTp reference_window(const ExpertTensors& tensors, const std::vector<A8Activation>& x, const WindowRoutes& routes,
                                        std::span<const float> weights, std::span<const float> shared, std::uint32_t top_k);

}  // namespace strix::ds41::expert_tp
