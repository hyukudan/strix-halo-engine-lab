#pragma once

#include "attn_core/dense_flash_keysplit.hpp"
#include "core/looped_layers.hpp"
#include "strix/env_flags.hpp"

#include <cstddef>

namespace strix::inkling2 {

using core::LayerVisit;
inline constexpr unsigned kPhysicalLayers = 24;
inline constexpr unsigned kCoreFirst = 6;
inline constexpr unsigned kCoreLast = 18;
inline constexpr unsigned kCorePasses = 3;
inline constexpr unsigned kResidualBlockLayers = 6;
using MiniLayers = core::LoopedLayers<kPhysicalLayers, kCoreFirst, kCoreLast, kCorePasses>;
inline constexpr unsigned kLogicalVisits = MiniLayers::visits;

constexpr auto mini_schedule() { return MiniLayers::schedule(); }
constexpr bool valid_mini_visit(LayerVisit visit) { return MiniLayers::valid_visit(visit); }
constexpr bool is_core(unsigned physical_layer) { return MiniLayers::is_core(physical_layer); }
constexpr unsigned state_slot(unsigned physical_layer, unsigned pass) {
    return MiniLayers::state_slot(physical_layer, pass);
}

constexpr bool starts_residual_block(LayerVisit visit) {
    return visit.physical_layer % kResidualBlockLayers == 0;
}

constexpr bool is_full_attention(unsigned physical_layer) {
    return physical_layer % kResidualBlockLayers == kResidualBlockLayers - 1U;
}

// A repeated pass reads the incoming residual at its second core block
// boundary, plus the hidden output at the end of that pass.
constexpr bool captures_message_boundary(LayerVisit visit) {
    return visit.physical_layer == kCoreFirst + kResidualBlockLayers &&
           visit.pass + 1U < kCorePasses;
}

// The official reference numbers pass-zero slots by physical layer, then
// appends caches for repeats. Takes a visit from mini_schedule().
constexpr unsigned reference_state_slot(LayerVisit visit) {
    return visit.pass == 0 ? visit.physical_layer
        : kPhysicalLayers + (visit.pass - 1U) * MiniLayers::core_width
              + (visit.physical_layer - kCoreFirst);
}

// Prefill tail absorption (session::prompt_windows tail_rows) keeps the bits of the greedy schedule only when every
// tail row takes the same route forms in the enlarged final chunk as in its own short chunk:
// - attention: a chunk of at most kKeySplitMaxRows rows takes the small-row policy with key splits merged in split
//   order, a longer one a single online softmax per row (inkling2_mini_attention_hip.hpp); the 1-row chunk with packed
//   decode also takes the packed R1 MoE body instead of the grouped one. A tail longer than the key-split bound takes
//   the long-chunk forms in both schedules.
// - the long-chunk attention gives each row the bits of its 16-position wave tile, counted from the chunk start, so
//   the tail starts a wave tile in both schedules only when the nominal chunk is a multiple of kMiniPrefillWaveRows.
// - the enlarged chunk must pass the backend row/grid admission (enlarged_admitted), else the nominal chunk is kept.
// Returns the admitted tail rows; 0 = greedy chunks, the unabsorbed schedule itself.
inline constexpr unsigned kMiniPrefillWaveRows = 16U;

constexpr unsigned mini_prefill_tail_rows(unsigned chunk_rows, unsigned tail_rows, bool enlarged_admitted) {
    const bool same_forms = tail_rows > attn_core::dense::kKeySplitMaxRows && chunk_rows != 0U &&
                            chunk_rows % kMiniPrefillWaveRows == 0U;
    return same_forms && enlarged_admitted ? tail_rows : 0U;
}

// Tail rows ONE prefill call of `rows` rows absorbs into its final chunk: the remainder after its whole chunks when it is at
// most the decoder's admitted capacity and keeps the greedy bits (mini_prefill_tail_rows on the ACTUAL remainder;
// admits(chunk + remainder) = the device row/grid admission). The one rule of the decoder's prefill() (the CLI) and of its
// served prefill_slot_first() calls: a remainder of 1..16 rows runs as its own chunk in both (8192 + 5 -> 8192, 5).
template <class Admits>
constexpr unsigned mini_call_tail_rows(std::size_t rows, unsigned chunk, unsigned capacity, Admits&& admits) {
    if (chunk == 0U || capacity == 0U || rows <= chunk) return 0U;
    const auto remainder = static_cast<unsigned>(rows % chunk);
    if (remainder == 0U || remainder > capacity) return 0U;
    return mini_prefill_tail_rows(chunk, remainder, admits(chunk + remainder));
}

// Tail capacity the decoder keeps: the largest one <= `capacity` whose enlarged chunk the device admits (admission is
// monotone in rows), instead of dropping a capacity whose full enlargement fails (HIP chunk 60000 + capacity 7500 is
// above the 65535-row bound, 60000 + 17 is not). Each call still absorbs only its actual remainder (mini_call_tail_rows).
// 0 when no tail longer than the key-split bound keeps the forms or fits.
template <class Admits>
constexpr unsigned mini_admitted_tail_capacity(unsigned chunk, unsigned capacity, Admits&& admits) {
    constexpr unsigned shortest = attn_core::dense::kKeySplitMaxRows + 1U;
    if (chunk == 0U || chunk % kMiniPrefillWaveRows != 0U || capacity < shortest || !admits(chunk + shortest)) return 0U;
    unsigned lo = shortest, hi = capacity;  // admits(chunk + lo) holds
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1U) / 2U;
        if (admits(chunk + mid)) lo = mid;
        else hi = mid - 1U;
    }
    return lo;
}
// Rows of the FIRST chunk one prefill call of `rows` rows runs (chunks of `chunk` rows, the final one enlarged by the
// call's absorbed tail, mini_call_tail_rows): what a served piece may cut without crossing a chunk boundary.
constexpr unsigned long long mini_call_first_chunk(unsigned long long rows, unsigned chunk, unsigned call_tail) {
    if (rows <= chunk) return rows;
    return call_tail != 0U && rows == static_cast<unsigned long long>(chunk) + call_tail ? rows : chunk;
}
// Bounded served slices (serve::CohortConfig::prompt_slice_decoding) cut a chunk only where every row keeps its route
// forms (above): at multiples of the 16-position wave tile from the chunk start, every piece longer than the key-split
// bound (so no piece takes the small-row attention policy or the one-row packed MoE body).
inline constexpr unsigned kMiniPieceAlignRows = kMiniPrefillWaveRows;
inline constexpr unsigned kMiniPieceShortRows = attn_core::dense::kKeySplitMaxRows;
// Prefill schedule: ONE resolver for the CLI (apps/inkling2_mini_gpu_main.hpp) and the server (inkling2_mini_serve.cpp).
// Every chunk re-streams all expert weights, so the chunk is as large as the backend qualified (HIP 8192 rows, measured
// in apps/inkling2_mini_gpu_main.hpp); CUDA keeps 2048-row greedy chunks until measured there. The tail divisor lets a
// remainder of up to chunk / divisor rows ride in the final chunk (0 = greedy chunks) when mini_call_tail_rows admits it.
// Bits: a chunking keeps the bits of another only where every row takes the same route forms (see above); the served
// lane runs THIS schedule over its whole prompt, so served bits equal the CLI's on the same decode route. The server
// decodes grouped (resident decoder packed_decode off), so a 1-row prefill chunk (a 1-token prompt, or a 1-row
// remainder such as 8192 + 1) takes the grouped MoE body: served == CLI --grouped-decode at every length, while the
// CLI's HIP default (packed) differs at those lengths until the served packed route is bound. Receipts of the earlier
// server default (128-row slices) differ for prompts whose last slice had 1..16 rows.
// Regimes (serve): with no other lane decoding a served prompt runs these chunks whole (8192 rows ~4.7 s on gfx1151);
// while another lane decodes it runs pieces of the same chunks (kMiniPiece*) sized by the lane gap target below, so a
// decoding lane waits about one piece and the prompt keeps the bits of the unbounded schedule. CUDA keeps whole chunks
// (decoding_rows 0 = off) until its pieces are qualified.
// Gap-bounded pieces (serve::CohortConfig::prompt_gap_ms): a piece takes the most rows whose pass (pass_fixed_ms + the
// previous pass's ms per row) plus the last decode round fits gap_ms, at least decoding_rows (512). pass_fixed_ms = one
// prefill pass's fixed cost on gfx1151 (all expert weights re-streamed; t(rows) ~ 240 + 0.545 rows ms from the 8192 /
// 1024 / 512-row passes). decode_share = ms of decode owed per ms of prompt while lanes decode (CohortConfig::
// prompt_share): 0 = one decode round per piece. Measured on gfx1151 (srvmd_fix3 t1, EXACT, one lane streaming while a
// prose prompt arrives; first token / worst lane gap, 8K and 32K; alone 4.7 s and 21.2 s):
//   gap 1000 ms, share 0 (default)  5.96 / 1.03 s   26.7 / 1.03 s
//   fixed 1024, share 0             6.63 / 0.82     27.6 / 0.95
//   fixed 512, share 0              8.83 / 0.56     36.7 / 0.62
//   fixed 2048 / 4096, share 0      5.55 / 1.35, 4.91 / 2.36
//   gap 1500 ms, share 0            5.29 / 1.57
//   share 1: gap 1000 10.47 / 1.03; fixed 512 18.5 / 0.65 and 60.5 / 0.69; 1024 13.8 / 0.93; 2048 10.4 / 1.35
//   whole chunks (no pieces, share 1) 4.80 / 4.75 s   36.1 / 5.90 s
// Share 1 gives the streaming lane ~5x more tokens while the prompt arrives (121 vs 26 at 8K) but about the same total
// delay (it loses ~100 tokens of decode time to the same prefill work either way), and doubles the arriving first token.
struct MiniPrefillSchedule {
    unsigned chunk;
    unsigned tail_divisor;
    unsigned decoding_rows;
    double gap_ms = 0.0;
    double pass_fixed_ms = 0.0;
    double decode_share = 1.0;
};

constexpr MiniPrefillSchedule mini_prefill_schedule(bool hip) {
    return hip ? MiniPrefillSchedule{8192U, 8U, 512U, 1000.0, 240.0, 0.0} : MiniPrefillSchedule{2048U, 0U, 0U};
}

// The schedule with the run's switch: STRIX_INKLING_PREFILL_TAIL=0 restores greedy chunks. Read here only.
inline MiniPrefillSchedule mini_prefill_schedule_from_env(bool hip) {
    auto schedule = mini_prefill_schedule(hip);
    if (!env_flag_default_on("STRIX_INKLING_PREFILL_TAIL")) schedule.tail_divisor = 0U;
    return schedule;
}

// Rows of the nominal chunk: the request (0 = the backend default), never above the context.
constexpr unsigned mini_resolve_prefill_chunk(unsigned requested, unsigned context, bool hip) {
    const auto chunk = requested != 0U ? requested : mini_prefill_schedule(hip).chunk;
    return chunk < context ? chunk : context;
}

// Tail rows a chunk may absorb (before the decoder's own admission, mini_admitted_tail_capacity). No capacity when the
// context leaves no room past the chunk (an enlarged pass could never run there).
constexpr unsigned mini_prefill_tail_capacity(unsigned chunk, unsigned divisor, unsigned context = ~0U) {
    if (divisor == 0U || context <= chunk) return 0U;
    const unsigned capacity = chunk / divisor;
    return capacity < context - chunk ? capacity : context - chunk;
}

// Rows of a served prompt piece while another lane decodes: the request (0 = the schedule's), off (0) when it does
// not cut the chunk into at least two pieces.
constexpr unsigned mini_resolve_decoding_rows(unsigned requested, unsigned chunk, bool hip) {
    const auto rows = requested != 0U ? requested : mini_prefill_schedule(hip).decoding_rows;
    return rows != 0U && rows < chunk ? rows : 0U;
}

// Decode slot cap of the server when --slots is absent. It does not depend on the prefill chunk.
inline constexpr unsigned kMiniServeDefaultSlotRows = 128U;

}  // namespace strix::inkling2
