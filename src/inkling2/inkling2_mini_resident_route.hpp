#pragma once

#include "strix/serve/decode_form.hpp"

#include <string>
#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Decode expert form default of the command line (and of a server running its route through the served route lever,
// kMiniServedCliRoute): the packed one-row form (moe_core::microscaled_r1), whose row bits equal a one-row launch at any
// cohort width. --grouped-decode selects the grouped tile kernel on the command line and on a CUDA server
// (mini_served_packed_decode); a HIP server without the lever keeps its incumbent grouped decode and takes no form flag.
inline constexpr bool kMiniPackedDecodeDefault = true;

// The same value from the two flags, through the resolver the command line also calls (serve::resolve_decode_form).
constexpr bool mini_packed_decode(bool packed_flag, bool grouped_flag) {
    return ::strix::serve::resolve_decode_form(packed_flag, grouped_flag, kMiniPackedDecodeDefault);
}

constexpr const char* mini_decode_form_name(bool packed_decode) noexcept { return packed_decode ? "packed" : "grouped"; }

// Served decode expert form, fixed at decoder construction. HIP: the packed form only with the admitted served route
// lever (`cli_route`, which also packs cohort rows); without it the incumbent grouped form. CUDA: the packed form, as the
// command line: a lone lane runs the packed one-row body (RTX q36, origin/main: CLI grouped 66.3 vs packed 89.1 tok/s,
// served C1 66.5 = the grouped form); a cohort pass of two or more lanes keeps the grouped tile kernel (no packed cohort
// rows without the lever), and every lane's bits equal its lone packed run (cohort-lanes fixture, widths 2..17). The
// CUDA server takes the command line's flags through the same resolver (--grouped-decode = the grouped form for a lone
// lane too; naming both is refused); a HIP server refuses them (its form follows the lever row, resolve).
constexpr bool mini_served_packed_decode(bool hip, bool cli_route, bool packed_flag = false, bool grouped_flag = false) {
    return hip ? cli_route && kMiniPackedDecodeDefault : mini_packed_decode(packed_flag, grouped_flag);
}

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// With packed decode a lone decode row and every verify row run the packed expert form. A multi-row decode (a cohort
// of more than kMiniMaxVerifyRows lanes) runs it only with packed cohort rows (the served route lever); prefill chunks
// of more than one row keep the grouped tile kernel.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph, bool cohort_rows = false) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || verify || (cohort_rows && kind == MiniAppendKind::decode)),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

// Row cap of one batched cohort decode pass of the served route lever (decode_slots_served; 0 = the whole cohort in one
// pass, serve::plan_cohort_passes). Packed decode promises that a lane's bits do not depend on the cohort width, so it
// runs passes of at most the backend's one-row cap (the dense rows family keeps its one-row body up to that many rows).
constexpr unsigned mini_cohort_pass_cap(bool packed_decode, unsigned backend_one_row_cap) noexcept {
    return packed_decode ? backend_one_row_cap : 0U;
}

// HIP targets the exact down-image lever (inkling2-mini-combine-down-image@1) is qualified on: the NaN identity fixture and
// the model receipts cover gfx1151 only; the resident device policy admits more targets for the rest of the model.
// `arch` is the gcnArchName up to the first ':'.
constexpr bool mini_combine_image_arch(std::string_view arch) { return arch == "gfx1151"; }

// Where the prepared-attention round fold (inkling2-mini-prepared-round-fold@1) runs: the HIP prepared attention on the
// qualified target (the fold fixture and the model receipts cover gfx1151 only). The CUDA preparation has no fold, so a
// selection there is declined; `arch` is the gcnArchName up to the first ':' (ignored off HIP).
constexpr bool mini_prepared_round_fold_admitted(bool hip_backend, std::string_view arch) {
    return hip_backend && arch == "gfx1151";
}

// The greedy head a resident decoder runs (screened-head lever, inkling2-mini-screened-head@1). Screened: a greedy pick
// that returns no score row runs the shared EXACT screened head (same token and status as the full head); a step that
// returns or reads scores keeps the full head. `reason` says why: lever_off (unset or 0), lever (selected and built),
// build_refused (selected; the plane could not be built: unsupported shape, non-finite weight, no device memory),
// kv_slot (served: the plane would cost a KV slot and was released), hip_binding (HIP keeps its own binding).
enum class MiniHeadReason : unsigned char { lever_off, lever, build_refused, kv_slot, hip_binding };
struct MiniHead {
    bool screened = false;
    MiniHeadReason reason = MiniHeadReason::lever_off;
    friend constexpr bool operator==(MiniHead, MiniHead) = default;
};

// The lever decides; `build()` (the plane) runs only when the lever selects the screened head, and a plane it cannot
// build keeps the full head.
template <class Build>
MiniHead mini_select_head(bool selected, Build&& build) {
    if (!selected) return {};
    return build() ? MiniHead{true, MiniHeadReason::lever} : MiniHead{false, MiniHeadReason::build_refused};
}

// The startup "head" field, the same text in the command line, the gate and the server: "screened lever",
// "full lever-off", "full build-refused", "full kv-slot", "screened hip-binding" or "full hip-binding".
constexpr std::string_view mini_head_reason_name(MiniHeadReason reason) noexcept {
    switch (reason) {
    case MiniHeadReason::lever_off: return "lever-off";
    case MiniHeadReason::lever: return "lever";
    case MiniHeadReason::build_refused: return "build-refused";
    case MiniHeadReason::kv_slot: return "kv-slot";
    case MiniHeadReason::hip_binding: return "hip-binding";
    }
    return "unknown";
}
inline std::string mini_head_field(MiniHead head) {
    return std::string(head.screened ? "screened " : "full ") + std::string(mini_head_reason_name(head.reason));
}

// Whether one append's greedy pick runs the built screened head (skips the score row): a single pick without host
// scores. The screened-head lever admits only prefill / decode picks: a verify append (a one-row verify window, the
// partial-commit replay of a kept row) keeps the full head under the lever. HIP's own binding keeps its rule.
constexpr bool mini_screened_pick(MiniHeadReason reason, bool built, MiniAppendKind kind, bool pick_next,
                                  bool multi_pick, bool host_scores) {
    if (!built || !pick_next || multi_pick || host_scores) return false;
    return kind != MiniAppendKind::verify || reason == MiniHeadReason::hip_binding;
}

// The one prompt-row rule of an opted-in W4A4 expert class, shared by the CLI session and the served lanes: every
// prefill-kind row runs the class whatever the chunk geometry, and decode / verify rows keep the BF16 tile. A one-row
// prefill chunk therefore takes the grouped route (the packed one-row route has no W4A4 class).
constexpr bool mini_w4a4_rows(bool w4a4_bound, MiniAppendKind kind) {
    return w4a4_bound && kind == MiniAppendKind::prefill;
}

constexpr MiniResidentRoute mini_resident_chunk_route(MiniAppendKind kind, unsigned rows, bool packed_decode,
                                                      bool w4a4_bound, bool cohort_rows = false) {
    return mini_resident_route(kind, rows, packed_decode && !mini_w4a4_rows(w4a4_bound, kind), false, cohort_rows);
}

}  // namespace strix::inkling2
