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

// Prefill schedule: ONE resolver for the CLI (apps/inkling2_mini_gpu_main.hpp) and the server (inkling2_mini_serve.cpp).
// Every chunk re-streams all expert weights, so the chunk is as large as the backend qualified (HIP 8192 rows, measured
// in apps/inkling2_mini_gpu_main.hpp); CUDA keeps 2048-row greedy chunks until measured there. The tail divisor lets a
// remainder of up to chunk / divisor rows ride in the final chunk (0 = greedy chunks) when mini_call_tail_rows admits it.
// Bits: a chunking keeps the bits of another only where every row takes the same route forms (see above); the served
// lane runs THIS schedule over its whole prompt, so served bits equal the CLI's. Receipts of the earlier server default
// (128-row slices) differ for prompts whose last slice had 1..16 rows.
// Interleave: a served prompt pass runs whole, so a decoding lane waits up to one chunk pass (8192 rows ~4.7 s on gfx1151).
// The trade-off is the one shared datum --prefill-rows (CLI and server alike), not a server-only slice.
struct MiniPrefillSchedule {
    unsigned chunk;
    unsigned tail_divisor;
};

constexpr MiniPrefillSchedule mini_prefill_schedule(bool hip) {
    return hip ? MiniPrefillSchedule{8192U, 8U} : MiniPrefillSchedule{2048U, 0U};
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

// Tail rows a chunk may absorb (before the decoder's own admission, mini_prefill_tail_rows).
constexpr unsigned mini_prefill_tail_capacity(unsigned chunk, unsigned divisor) {
    return divisor == 0U ? 0U : chunk / divisor;
}

// Decode slot cap of the server when --slots is absent. It does not depend on the prefill chunk.
inline constexpr unsigned kMiniServeDefaultSlotRows = 128U;

}  // namespace strix::inkling2
