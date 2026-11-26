#pragma once

#include "attn_core/dense_flash_keysplit.hpp"
#include "core/looped_layers.hpp"
#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "strix/env_flags.hpp"
#include "strix/serve/prompt_piece.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

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
// Every chunk re-streams all expert weights, so the chunk is as large as the backend qualified: the declared rows per
// backend and profile (mini_cli_prefill_rows, inkling2_mini_prefill_rows.hpp). CUDA keeps greedy chunks (no tail).
// Tail absorption: a remainder of up to chunk / tail_divisor rows rides in the final chunk when mini_call_tail_rows
// admits it (gfx1151 8K prose, 8236 = 8192 + 44 rows: 5,021 -> 4,810 ms with one chunk, logits bitwise equal).
// STRIX_INKLING_PREFILL_TAIL=0 restores greedy chunks on both entry points.
struct MiniPrefillSchedule {
    unsigned chunk;
    unsigned tail_divisor;
};

constexpr MiniPrefillSchedule mini_prefill_schedule(bool hip, profile::Name tier) {
    return {mini_cli_prefill_rows(hip ? MiniPrefillBackend::hip : MiniPrefillBackend::cuda, tier), hip ? 8U : 0U};
}

inline MiniPrefillSchedule mini_prefill_schedule_from_env(bool hip, profile::Name tier) {
    auto schedule = mini_prefill_schedule(hip, tier);
    if (!env_flag_default_on("STRIX_INKLING_PREFILL_TAIL")) schedule.tail_divisor = 0U;
    return schedule;
}

// Served prompt route. A HIP server with the served route lever (kMiniServedCliRoute, opt-in in every profile column)
// serves the command line's schedule above with the first id taken from the final prefill row (emits-first; device
// evidence: served prefill == CLI prefill bitwise, chunk and tail edges included). Without it, and on CUDA (not
// qualified there), a server keeps the held-back route: greedy chunks, no tail, the last prompt token held back and
// decoded as the first decode row. Its chunk is the command line's chunk (the one schedule resolver above, no tail)
// where the backend's prefill bits do not depend on the chunk (mini_prefill_chunk_invariant), else the incumbent 128
// rows. Only the chunk changes: the held-back token, the decode rows and the lane cap (kMiniServeLaneRows) stay.
inline constexpr unsigned kMiniServedHeldBackChunk = 128U;
constexpr bool mini_served_prefill_emits_first(bool hip, bool cli_route) noexcept { return hip && cli_route; }

// Backends whose resident prefill rows carry the same bits at any chunk (every chunk form, small remainders included):
// CUDA (complete EXACT generations equal at 2048 and 8192 rows, 2050 / 5200 / 8192 / 8195-token prompts, RTX receipt
// q15; FAST and TURBO run the same kernels while no lossy lever resolves there). HIP is not declared: its held-back
// served route keeps 128 rows.
constexpr bool mini_prefill_chunk_invariant(bool hip) noexcept { return !hip; }

constexpr MiniPrefillSchedule mini_served_held_back_schedule(bool hip, profile::Name tier) noexcept {
    return {mini_prefill_chunk_invariant(hip) ? mini_prefill_schedule(hip, tier).chunk : kMiniServedHeldBackChunk, 0U};
}

inline MiniPrefillSchedule mini_served_prefill_schedule_from_env(bool hip, bool cli_route, profile::Name tier) {
    return mini_served_prefill_emits_first(hip, cli_route) ? mini_prefill_schedule_from_env(hip, tier)
                                                           : mini_served_held_back_schedule(hip, tier);
}

// Fallback rows of the served chunk: the command line's fallback rows when the held-back route took the command line's
// chunk; 0 = no smaller candidate (an explicit --prefill-rows is never changed).
constexpr unsigned mini_served_prefill_fallback_rows(bool hip, bool cli_route, profile::Name tier) noexcept {
    if (mini_served_prefill_emits_first(hip, cli_route) || !mini_prefill_chunk_invariant(hip)) return 0U;
    return mini_cli_prefill_fallback_rows(hip ? MiniPrefillBackend::hip : MiniPrefillBackend::cuda, tier);
}

// Candidate chunks of the served load (serve::load_with_chunk_ladder), largest first, the incumbent last: the resolved
// chunk, then (an implicit chunk with fallback rows) the fallback rows and the incumbent held-back 128 rows, each only
// when smaller than the one before. The server keeps the largest candidate whose whole load keeps the incumbent's slot
// capacity, so the tall chunk never costs a session slot. One candidate: an explicit --prefill-rows, HIP, the lever.
struct MiniChunkLadder {
    std::array<std::uint32_t, 3> rows{};
    std::size_t count = 0;
    [[nodiscard]] constexpr std::span<const std::uint32_t> view() const noexcept { return {rows.data(), count}; }
};

constexpr MiniChunkLadder mini_served_chunk_ladder(unsigned chunk, unsigned fallback_rows) noexcept {
    MiniChunkLadder ladder;
    ladder.rows[ladder.count++] = chunk;
    if (fallback_rows == 0U) return ladder;
    if (fallback_rows < chunk) ladder.rows[ladder.count++] = fallback_rows;
    if (kMiniServedHeldBackChunk < ladder.rows[ladder.count - 1U])
        ladder.rows[ladder.count++] = kMiniServedHeldBackChunk;
    return ladder;
}

// Tail rows a chunk may absorb (before the decoder's own admission, mini_admitted_tail_capacity). None when the
// context leaves no room past the chunk.
constexpr unsigned mini_prefill_tail_capacity(unsigned chunk, unsigned divisor, unsigned context = ~0U) {
    if (divisor == 0U || chunk == 0U || context <= chunk) return 0U;
    const unsigned capacity = chunk / divisor;
    return capacity < context - chunk ? capacity : context - chunk;
}

// Served prompt pieces while another lane decodes (serve::CohortConfig::prompt_slice_decoding) cut a chunk only where
// every row keeps its route forms (`cli_route`: the ADMITTED served route lever):
// - HIP command line's route: at multiples of the 16-position wave tile from the chunk start, every piece longer than
//   the key-split bound (no piece takes the small-row attention policy or the one-row packed MoE body).
// - held-back route of a chunk-invariant backend (CUDA): at multiples of kMiniCudaPieceRows from the chunk start, every
//   piece longer than kMiniCudaPieceRows, so no piece takes the short-row dense body (rows <= cuda_core::kMaxRows = 32,
//   plan_bf16_row_projection) or the one-row projected-bias image route (mini_cuda_bias_in_attention): every piece
//   runs the forms of its whole chunk. Device evidence: the served_prefill fixture's held-back arms (incumbent 128-row
//   calls == whole chunk == 2048-row calls == pieces beside a decoding lane: first id, scores, KV, histories).
// - HIP held-back route: none (its 128-row chunk is already the piece).
inline constexpr unsigned kMiniCudaPieceRows = 32U;
constexpr serve::PromptPieceRule mini_prefill_piece_rule(bool hip, bool cli_route) {
    if (mini_served_prefill_emits_first(hip, cli_route))
        return {kMiniPrefillWaveRows, attn_core::dense::kKeySplitMaxRows};
    if (mini_prefill_chunk_invariant(hip)) return {kMiniCudaPieceRows, kMiniCudaPieceRows};
    return {};
}

// Served lane cap without --slots (an explicit --prefill-rows without --slots caps the lanes at its rows, as it did
// before the served chunk and the lane cap were separate), and the rows of one served prompt piece while another lane
// decodes (the shared short mixed-service pieces, serve::kMixedServiceShortPieces: the incumbent 128-row bound).
inline constexpr unsigned kMiniServeLaneRows = 128U;
inline constexpr unsigned kMiniServeDecodingRows = serve::kMixedServiceShortPieces.piece_rows;
static_assert(kMiniServeDecodingRows % kMiniPrefillWaveRows == 0U &&
              kMiniServeDecodingRows > attn_core::dense::kKeySplitMaxRows, "a served piece keeps the long-chunk forms");
static_assert(kMiniServeDecodingRows % kMiniCudaPieceRows == 0U && kMiniServeDecodingRows > 3U * kMiniCudaPieceRows,
              "a CUDA held-back piece is cuttable (serve::validate_prompt_pieces)");

// CUDA held-back route beside decoding lanes: the shared long pieces (serve::kMixedServiceLongPieces, 1024 rows). RTX
// PRO q35 (one decoding lane): a 128-row piece ~135 ms, an 8192-row prompt alone 848 ms (~124 ms per pass + ~0.088 ms
// per row), so 128-row pieces made an 8K prompt beside a lane take 9.7 s. Fixed, not priced: with the shared gap
// pricing (CohortConfig::prompt_gap_ms 250 ms, 124 ms per pass) the learned row cost came out near zero and the planner
// granted whole 8K calls beside a decoding lane (q39: 849-853 ms gaps); the long pieces held 190 ms (q41).
inline constexpr serve::MixedServicePieces kMiniCudaServePieces = serve::kMixedServiceLongPieces;
static_assert(kMiniCudaServePieces.piece_rows % kMiniCudaPieceRows == 0U &&
                  kMiniCudaServePieces.piece_rows > 3U * kMiniCudaPieceRows,
              "a CUDA held-back piece is cuttable (serve::validate_prompt_pieces)");

}  // namespace strix::inkling2
