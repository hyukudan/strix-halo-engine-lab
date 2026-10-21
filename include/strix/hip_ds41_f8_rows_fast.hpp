#pragma once
// F8 projection rows (F16RowsView::weight_f8: the MX-FP8-valued Engram kv rows) with 16-byte weight loads, several output
// rows per wave, the declared order of f16_rows (lane class c = element index mod 32 sums acc = acc + w * x over its
// elements in ascending order, then the xor butterfly 16, 8, 4, 2, 1): the same bits as enqueue_f16_rows on F8 rows.
// Each lane reads 16 contiguous codes (512 contiguous bytes per wave per load) and widens them to F32 with two exact
// multiplies (no F16 round trip); each wave owns several output rows and the block stages the activations once for
// all of them (the row kernel of hip_ds41_f16_rows.hip re-stages the whole activation row for every 8 weight rows).
// enqueue_f16_rows' F8 rows come here when f8_rows_fast_enabled() && f8_rows_fast_supported():
//   1 row, STRIX_DS41_ENGRAM_FAST unset or 1: the round-1 kernel (the P13 default); =2: the compact kernel (round 2:
//   codes kept compact in LDS, one linear stream per wave); =0: off (lanes row kernel).
//   2..8 rows, STRIX_DS41_ENGRAM_RN=1 (default off): the RN kernels (exact R); otherwise the lanes kernel.
// Every form gives the same bits.
#include "strix/hip_ds41_layer.hpp"

namespace strix::ds41 {

// width % 512 == 0, 16-byte aligned F8 rows, 1 row (the engine path; the bench variants also take 2..8 rows).
[[nodiscard]] bool f8_rows_fast_supported(const F16RowsView& view) noexcept;
// STRIX_DS41_ENGRAM_FAST != 0 (default on) or STRIX_DS41_ENGRAM_RN=1 (read once)
[[nodiscard]] bool f8_rows_fast_enabled() noexcept;
void enqueue_f8_rows_fast(const F16RowsView& view, void* stream);
// Bench variants (tests only): rpw = output rows per wave (2, 4 or 8); dequant 0 = the exact formula of
// the row kernel without its F16 round trip, 1 = the code bits as an F32 times 2^120 times the scale (exact, default),
// 2 = timing roof only (codes as integers, NOT the same values).
void enqueue_f8_rows_fast_variant(const F16RowsView& view, unsigned rpw, unsigned dequant, void* stream);
// Round 2 (bench variants; see hip_ds41_f8_rows_fast.hip). Compact: 1 row, codes kept as bytes in LDS, one linear
// stream per wave over `rpw` rows (4, 8, 16), `depth` pairs in flight (2, 4, 8), `waves` 4 or 8 per block.
[[nodiscard]] bool f8_rows_compact_supported(const F16RowsView& view) noexcept;
void enqueue_f8_rows_compact_variant(const F16RowsView& view, unsigned waves, unsigned rpw, unsigned depth, void* stream);
// RN: 2..8 rows (exact R), `cl` classes per lane (8: 8-byte loads, 16: 16-byte loads), `k` outputs per lane (1, 2),
// `waves` 4 or 8, `ahead` steps in flight (4, 8). Returns false (no launch) for a shape not instantiated.
[[nodiscard]] bool f8_rows_rn_supported(const F16RowsView& view) noexcept;
bool enqueue_f8_rows_rn_variant(const F16RowsView& view, unsigned cl, unsigned k, unsigned waves, unsigned ahead, void* stream);

}  // namespace strix::ds41
