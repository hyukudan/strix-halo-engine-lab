#pragma once

#include "attn_core/dense_flash_keysplit.hpp"
#include "core/looped_layers.hpp"
#include "strix/env_flags.hpp"
#include "strix/serve/prompt_piece.hpp"

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

// Tail rows ONE prefill call of `rows` rows absorbs into its final chunk: the remainder after its whole chunks when it is
// at most the decoder's admitted capacity and keeps the greedy bits (mini_prefill_tail_rows on the ACTUAL remainder;
// admits(chunk + remainder) = the device row/grid admission). The one rule of the decoder's prefill() (the command line)
// and of its served prefill_slot_first() calls: a remainder of 1..kKeySplitMaxRows rows runs as its own chunk in both.
template <class Admits>
constexpr unsigned mini_call_tail_rows(std::size_t rows, unsigned chunk, unsigned capacity, Admits&& admits) {
    if (chunk == 0U || capacity == 0U || rows <= chunk) return 0U;
    const auto remainder = static_cast<unsigned>(rows % chunk);
    if (remainder == 0U || remainder > capacity) return 0U;
    return mini_prefill_tail_rows(chunk, remainder, admits(chunk + remainder));
}

// Tail capacity the decoder keeps: the largest one <= `capacity` whose enlarged chunk the device admits (admission is
// monotone in rows). Each call still absorbs only its actual remainder (mini_call_tail_rows). 0 when no tail longer
// than the key-split bound keeps the forms or fits.
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
// call's absorbed tail, mini_call_tail_rows): where a served piece may cut without crossing a chunk boundary.
constexpr unsigned long long mini_call_first_chunk(unsigned long long rows, unsigned chunk, unsigned call_tail) {
    if (rows <= chunk) return rows;
    return call_tail != 0U && rows == static_cast<unsigned long long>(chunk) + call_tail ? rows : chunk;
}

// Prefill schedule: ONE resolver for the command line (apps/inkling2_mini_gpu_main.hpp) and the server
// (inkling2_mini_serve.cpp), so the same request is prefilled by the same chunks on both entry points.
// Every chunk re-streams all expert weights, so the chunk is as large as the backend qualified. gfx1151 (prose
// prompts, one chunk size per run, 2048 -> 8192 rows): 4K 3.47 -> 3.13 s, 8K 7.27 -> 6.23 s, 16K 15.69 -> 14.06 s,
// 32K 35.6 -> 35.8 s. CUDA keeps 2048-row greedy chunks until it is measured there.
// Tail absorption: a remainder of up to chunk / tail_divisor rows rides in the final chunk when mini_call_tail_rows
// admits it (gfx1151 8K prose, 8236 = 8192 + 44 rows: 5,021 -> 4,810 ms with one chunk, logits bitwise equal).
// STRIX_INKLING_PREFILL_TAIL=0 restores greedy chunks on both entry points.
struct MiniPrefillSchedule {
    unsigned chunk;
    unsigned tail_divisor;
};

constexpr MiniPrefillSchedule mini_prefill_schedule(bool hip) {
    return hip ? MiniPrefillSchedule{8192U, 8U} : MiniPrefillSchedule{2048U, 0U};
}

inline MiniPrefillSchedule mini_prefill_schedule_from_env(bool hip) {
    auto schedule = mini_prefill_schedule(hip);
    if (!env_flag_default_on("STRIX_INKLING_PREFILL_TAIL")) schedule.tail_divisor = 0U;
    return schedule;
}

// Tail rows a chunk may absorb (before the decoder's own admission, mini_admitted_tail_capacity). None when the
// context leaves no room past the chunk.
constexpr unsigned mini_prefill_tail_capacity(unsigned chunk, unsigned divisor, unsigned context = ~0U) {
    if (divisor == 0U || chunk == 0U || context <= chunk) return 0U;
    const unsigned capacity = chunk / divisor;
    return capacity < context - chunk ? capacity : context - chunk;
}

// Served prompt pieces while another lane decodes (serve::CohortConfig::prompt_slice_decoding) cut a chunk only where
// every row keeps its route forms: at multiples of the 16-position wave tile from the chunk start, every piece longer
// than the key-split bound (no piece takes the small-row attention policy or the one-row packed MoE body). HIP only;
// the CUDA forms are not qualified for cuts, so a CUDA prompt runs whole chunks.
constexpr serve::PromptPieceRule mini_prefill_piece_rule(bool hip) {
    return hip ? serve::PromptPieceRule{kMiniPrefillWaveRows, attn_core::dense::kKeySplitMaxRows}
               : serve::PromptPieceRule{};
}

// Served lane cap without --slots, and the rows of one served prompt piece while another lane decodes (HIP).
inline constexpr unsigned kMiniServeLaneRows = 128U;
inline constexpr unsigned kMiniServeDecodingRows = 128U;
static_assert(kMiniServeDecodingRows % kMiniPrefillWaveRows == 0U &&
              kMiniServeDecodingRows > attn_core::dense::kKeySplitMaxRows, "a served piece keeps the long-chunk forms");

}  // namespace strix::inkling2
