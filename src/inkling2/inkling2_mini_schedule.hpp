#pragma once

#include "attn_core/dense_flash_keysplit.hpp"
#include "core/looped_layers.hpp"
#include "strix/env_flags.hpp"
#include "strix/model_data/inkling2_mini.hpp"
#include "strix/serve/prompt_piece.hpp"
#include "strix/serve/row_budget.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
// Bits: a chunking keeps the bits of another only where every row takes the same route forms (see above), and a
// prefill row's MoE body does not depend on the decode route (mini_resident_route), so a served lane running THIS
// schedule over its whole prompt prefills with the CLI's bits; decode rows follow the decode route (server: grouped,
// = CLI --grouped-decode). Receipts of the earlier server default (128-row slices, the final prompt row decoded)
// differ for prompts whose last slice had 1..16 rows.
struct MiniPrefillSchedule {
    unsigned chunk;
    unsigned tail_divisor;
};

constexpr MiniPrefillSchedule mini_prefill_schedule(bool hip) {
    return hip ? MiniPrefillSchedule{8192U, 8U} : MiniPrefillSchedule{2048U, 0U};
}

// Served prefill regimes (strix-serve), model data [serve_prefill.<backend>] in models/inkling2-mini.toml, overridden
// by the device class's table where a receipt measured it: with no other lane decoding a served prompt runs whole
// chunks (chunk 0 = the CLI's); while another lane decodes it runs pieces of the same chunks (decoding_rows, cut by
// kMiniPiece* only, so the bits stay those of the unbounded schedule), sized by the lane gap target gap_ms where the
// device class declares it (serve::CohortConfig::prompt_gap_ms: pass_fixed_ms + the previous pass's ms per row + one
// decode round), with decode_share ms of decode owed per prompt ms (0 = one decode round per piece). The receipts and
// the measured table live with the data (TOML) and in docs/models/inkling2-mini/README.md.
struct MiniServePrefill {
    unsigned chunk;
    unsigned decoding_rows;
    double gap_ms;
    double pass_fixed_ms;
    double decode_share;
    std::string_view device_class;  // the class whose table applied, empty = the backend's row
    std::string_view receipt;
};

// Device class of a cost identity key (strix/device/cost_identity.hpp `arch|name|...`; a HIP arch may carry
// `:feature` suffixes): gfx1151, sm_120, ...
constexpr std::string_view mini_device_class(std::string_view identity) {
    identity = identity.substr(0, identity.find('|'));
    return identity.substr(0, identity.find(':'));
}

constexpr MiniServePrefill mini_serve_prefill(bool hip, std::string_view device_class = {}) {
    namespace data = model_data::inkling2_mini::serve_prefill;
    if (!hip)
        return {static_cast<unsigned>(data::cuda::chunk), static_cast<unsigned>(data::cuda::decoding_rows),
                data::cuda::gap_ms, data::cuda::pass_fixed_ms, data::cuda::decode_share, {}, data::cuda::receipt};
    MiniServePrefill out{static_cast<unsigned>(data::hip::chunk), static_cast<unsigned>(data::hip::decoding_rows),
                         data::hip::gap_ms, data::hip::pass_fixed_ms, data::hip::decode_share, {}, data::hip::receipt};
    if (device_class == "gfx1151") {
        out.decoding_rows = static_cast<unsigned>(data::hip::gfx1151::decoding_rows);
        out.gap_ms = data::hip::gfx1151::gap_ms;
        out.pass_fixed_ms = data::hip::gfx1151::pass_fixed_ms;
        out.decode_share = data::hip::gfx1151::decode_share;
        out.device_class = "gfx1151";
        out.receipt = data::hip::gfx1151::receipt;
    }
    return out;
}

// The served prompt route of strix-serve --model inkling2-mini, bound from the shared profile resolution (the identity
// variant kMiniServedCliPrefill of kMiniProfile, include/strix/profiles/inkling2_mini.hpp; requested only by name and
// admitted by profile::admit_variants, never chosen here):
// - incumbent (the default): the prompt minus its final token by prefill, in the forms of greedy form_rows-row chunks
//   (main's served 128-row slices: a final remainder of 1..kKeySplitMaxRows rows is its own short chunk), then the
//   final token by the grouped decode row, which gives the first id. Any chunk that is a multiple of form_rows keeps
//   these bits (cuts at multiples of the 16-row wave tile, every piece longer than the key-split bound), so the CLI's
//   large chunk (HIP 8192) serves the incumbent bits with one pass where main ran 64.
// - cli_prefill (variant served-cli-prefill): the whole prompt by prefill with the CLI's chunks and tail rule
//   (mini_served_prefill_rows), the first id from its final row (= strix-inkling2-mini --grouped-decode bit for bit).
enum class MiniServedIdentity { incumbent, cli_prefill };
// serve::FixedKvCohort route: the final prompt row by prefill (true) or by the decode row (false).
constexpr bool mini_served_emits_first(MiniServedIdentity identity) {
    return identity == MiniServedIdentity::cli_prefill;
}
// Rows of main's served chunk whose forms served-incumbent keeps (data: [serve_prefill] form_rows).
inline constexpr unsigned kMiniIncumbentFormRows =
    static_cast<unsigned>(model_data::inkling2_mini::serve_prefill::form_rows);
static_assert(kMiniIncumbentFormRows % kMiniPrefillWaveRows == 0U && 8192U % kMiniIncumbentFormRows == 0U,
              "the incumbent form chunk is a multiple of the wave tile and divides the CLI's chunk");

// served-incumbent: rows of the short chunk one prefill call of `rows` rows (starting at a multiple of form_rows)
// ends with: the remainder after whole form_rows-row chunks when it has 1..kKeySplitMaxRows rows (main ran it as its
// own short chunk), else 0. A call that IS that remainder runs whole.
constexpr unsigned mini_incumbent_short_tail(unsigned long long rows, unsigned form_rows) {
    if (form_rows == 0U) return 0U;
    const auto r = static_cast<unsigned>(rows % form_rows);
    return r != 0U && r <= attn_core::dense::kKeySplitMaxRows && rows > r ? r : 0U;
}
// served-incumbent: rows of the FIRST chunk one prefill call of `rows` rows runs (chunks of `chunk` rows over the call
// minus its short tail, then the tail).
constexpr unsigned long long mini_incumbent_first_chunk(unsigned long long rows, unsigned chunk, unsigned form_rows) {
    const unsigned long long head = rows - mini_incumbent_short_tail(rows, form_rows);
    return head < chunk ? head : chunk;
}

// The bounded-piece cut rule the resident decoder declares (MiniResidentDecoder::prefill_piece_rule): HIP only, CUDA's
// forms are not qualified (uncuttable). Cuts at the 16-row wave tile, every piece longer than the key-split bound; for
// served-incumbent with a chunk above its form chunk (form_rows, 0 = the chunk itself) at multiples of form_rows, so
// every call starts at a multiple of form_rows and its short tail is the prompt's (mini_incumbent_short_tail). The
// server validates its options against it before any device work.
constexpr serve::PromptPieceRule mini_serve_piece_rule(bool hip, MiniServedIdentity identity, unsigned form_rows,
                                                       unsigned chunk) {
    if (!hip) return {};
    if (identity == MiniServedIdentity::cli_prefill || form_rows == 0U || chunk <= form_rows)
        return {kMiniPieceAlignRows, kMiniPieceShortRows};
    return form_rows % kMiniPrefillWaveRows == 0U ? serve::PromptPieceRule{form_rows, kMiniPieceShortRows}
                                                  : serve::PromptPieceRule{};
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

// Rows of the served chunk: the request, else the served data's (0 = the CLI's chunk), never above the context.
constexpr unsigned mini_resolve_serve_chunk(unsigned requested, unsigned context, bool hip) {
    return mini_resolve_prefill_chunk(requested != 0U ? requested : mini_serve_prefill(hip).chunk, context, hip);
}

// Served prefill values of an identity: the chunk and the tail divisor (0 = greedy chunks). The incumbent takes the
// served data's chunk (the caller rounds it to its form chunk) with no tail absorption; served-cli-prefill takes the
// CLI's own values on every backend (mini_resolve_prefill_chunk and the CLI's tail rule with its
// STRIX_INKLING_PREFILL_TAIL switch), so a request without --prefill-rows dispatches the CLI's calls (CUDA: one
// 129-row call for a 129-token prompt, not 128 + 1).
struct MiniServedPrefillRows {
    unsigned chunk;
    unsigned tail_divisor;
};
constexpr MiniServedPrefillRows mini_served_prefill_rows(MiniServedIdentity identity, unsigned requested,
                                                         unsigned context, bool hip, unsigned cli_tail_divisor) {
    if (identity == MiniServedIdentity::cli_prefill)
        return {mini_resolve_prefill_chunk(requested, context, hip), cli_tail_divisor};
    return {mini_resolve_serve_chunk(requested, context, hip), 0U};
}

// Tail rows a chunk may absorb (before the decoder's own admission, mini_admitted_tail_capacity). No capacity when the
// context leaves no room past the chunk (an enlarged pass could never run there).
constexpr unsigned mini_prefill_tail_capacity(unsigned chunk, unsigned divisor, unsigned context = ~0U) {
    if (divisor == 0U || context <= chunk) return 0U;
    const unsigned capacity = chunk / divisor;
    return capacity < context - chunk ? capacity : context - chunk;
}

// Rows of a served prompt piece while another lane decodes: an explicit request is kept as given (the server refuses
// one the scheduler cannot run, serve::validate_prompt_pieces); the data's default (the device class's row where one
// applies, resolved at launch) is off (0) when it does not cut the chunk into at least two pieces.
constexpr unsigned mini_resolve_decoding_rows(unsigned requested, unsigned chunk, bool hip,
                                              std::string_view device_class = {}) {
    if (requested != 0U) return requested;
    const auto rows = mini_serve_prefill(hip, device_class).decoding_rows;
    return rows != 0U && rows < chunk ? rows : 0U;
}

// Decode slot cap of the server when --slots is absent: main's, the requested --prefill-rows (else its 128-row
// default) clipped to the context (--context 64 -> 64 slots, --prefill-rows 512 -> 512). The served chunk resolved from
// the data does not move it.
inline constexpr unsigned kMiniServeDefaultSlotRows = 128U;
constexpr unsigned mini_serve_slot_cap(unsigned requested_prefill_rows, unsigned context) {
    const unsigned rows = requested_prefill_rows != 0U ? requested_prefill_rows : kMiniServeDefaultSlotRows;
    return rows < context ? rows : context;
}

// Session slots the budget would hold with the decoder built at another row capacity (the incumbent chunk). `free` is
// measured after the current decoder is built, its rows-proportional reservation and first slot committed
// (`current_reservation`, MiniResidentDecoder::allocated_row_reservation_bytes); the other plan gets that back and pays
// its own (`other_reservation`, its first slot included, MiniResidentDecoder::row_reservation_bytes), then its slots
// are counted by the same rule as the current ones (serve::resident_kv_slots) with its own slot bytes.
inline unsigned mini_counterfactual_slots(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                          std::uint64_t kv_limit, unsigned rows, std::uint64_t current_reservation,
                                          std::uint64_t other_reservation, std::uint64_t other_slot_bytes) {
    if (free > total || current_reservation > total - free) return 0U;  // the reservation is part of the used bytes
    const std::uint64_t unreserved = free + current_reservation;
    if (other_reservation > unreserved) return 0U;
    return serve::resident_kv_slots(unreserved - other_reservation, total, device_limit, kv_limit, other_slot_bytes,
                                    rows);
}

// True when the default chunk (--slots and --prefill-rows both absent) is above the incumbent chunk and leaves fewer
// slots than the incumbent plan: the incumbent identity then serves its own chunk (same bits, the incumbent's slots).
constexpr bool mini_slot_keep_incumbent(unsigned lanes, std::optional<unsigned> lanes_incumbent,
                                        unsigned requested_slots, bool chunk_requested, unsigned chunk,
                                        unsigned incumbent_chunk) {
    return requested_slots == 0U && !chunk_requested && chunk > incumbent_chunk && lanes_incumbent.has_value() &&
           lanes < *lanes_incumbent;
}

// Session slots the server admits after the decoder is built, sized before any slot is committed. `lanes` = the slots
// the budget holds with the decoder at its chunk, `lanes_incumbent` = the slots it would hold with the decoder at the
// incumbent chunk (mini_counterfactual_slots; nullopt = not priced on this backend). Refused with a message (empty =
// admitted): a default chunk above the incumbent one that leaves fewer slots than the incumbent chunk, or that cannot
// be compared, when --slots and --prefill-rows are both absent (also when it leaves none); a budget with no slot; and
// explicit --slots the budget cannot hold (never silently fewer).
inline std::string mini_slot_refusal(unsigned lanes, std::optional<unsigned> lanes_incumbent, unsigned requested_slots,
                                     bool chunk_requested, unsigned chunk, unsigned incumbent_chunk,
                                     std::uint64_t slot_bytes) {
    if (requested_slots == 0U && !chunk_requested && chunk > incumbent_chunk) {
        if (!lanes_incumbent)
            return "Inkling: the slots of the default prefill chunk of " + std::to_string(chunk) +
                   " rows cannot be compared with the incumbent " + std::to_string(incumbent_chunk) +
                   "-row chunk on this backend; pass --prefill-rows " + std::to_string(incumbent_chunk) +
                   " (or the chunk you want) or --slots N";
        if (lanes < *lanes_incumbent)
            return "Inkling: the default prefill chunk of " + std::to_string(chunk) + " rows leaves " +
                   std::to_string(lanes) + " session slots where the incumbent " + std::to_string(incumbent_chunk) +
                   "-row chunk leaves " + std::to_string(*lanes_incumbent) + "; pass --prefill-rows " +
                   std::to_string(incumbent_chunk) + " to keep the slots" +
                   (lanes != 0U ? ", or --slots " + std::to_string(lanes) + " to accept fewer" : std::string{});
    }
    if (lanes == 0U)
        return "Inkling KV budget holds no session slots (prefill chunk " + std::to_string(chunk) + " rows, " +
               std::to_string(slot_bytes) + " bytes per slot)";
    if (requested_slots != 0U && lanes < requested_slots)
        return "Inkling --slots " + std::to_string(requested_slots) + " do not fit: with the prefill chunk of " +
               std::to_string(chunk) + " rows the memory budget holds " + std::to_string(lanes) + " slots of " +
               std::to_string(slot_bytes) + " bytes; lower --slots, --context or --prefill-rows";
    return {};
}

}  // namespace strix::inkling2
