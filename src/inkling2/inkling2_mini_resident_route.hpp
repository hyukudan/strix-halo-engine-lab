#pragma once

#include "strix/serve/decode_form.hpp"

#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Decode expert form default of the command line (and of a server running its route through the served route lever,
// kMiniServedCliRoute): the packed one-row form (moe_core::microscaled_r1), whose row bits equal a one-row launch at any
// cohort width. --grouped-decode selects the grouped tile kernel on the command line. Without the lever a server keeps
// its incumbent grouped decode and takes no form flag.
inline constexpr bool kMiniPackedDecodeDefault = true;

// The same value from the two flags, through the resolver the command line also calls (serve::resolve_decode_form).
constexpr bool mini_packed_decode(bool packed_flag, bool grouped_flag) {
    return ::strix::serve::resolve_decode_form(packed_flag, grouped_flag, kMiniPackedDecodeDefault);
}

constexpr const char* mini_decode_form_name(bool packed_decode) noexcept { return packed_decode ? "packed" : "grouped"; }

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
