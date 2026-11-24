#pragma once

#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// The reference a served decode form is checked against, named with the served identity (route line, capabilities):
// packed = every lane's own lone one-row decode, bit for bit (the CLI identity; device proof: HIP cohort rows C17 / C34
// lanes == lone); grouped = the served grouped incumbent (one pass of every lane, no batch-invariance contract).
constexpr const char* mini_decode_reference(bool packed) { return packed ? "lone-decode" : "served-grouped"; }

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// With packed decode, every decode and verify row (a lone token, a verify window or a cohort of lanes of any size)
// runs the packed form, whose row bits equal a one-row launch: a lane's tokens do not depend on the cohort size.
// Prefill chunks of more than one row keep the grouped tile kernel.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || kind != MiniAppendKind::prefill),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

// Row cap of one batched cohort decode pass (0 = the whole cohort in one pass, serve::plan_cohort_passes).
// Packed decode promises that a lane's bits do not depend on the cohort width, so it runs passes of at most the
// backend's one-row cap (HIP: the dense rows family keeps the one-row body up to core::gemv_rows::max_rows rows and
// takes the WMMA tile GEMM above). Grouped decode makes no such promise and keeps its single pass of every lane.
constexpr unsigned mini_cohort_pass_cap(bool packed_decode, unsigned backend_one_row_cap) {
    return packed_decode ? backend_one_row_cap : 0U;
}

// HIP targets the exact down-image lever (inkling2-mini-combine-down-image@1) is qualified on: the NaN identity fixture and
// the model receipts cover gfx1151 only; the resident device policy admits more targets for the rest of the model.
// `arch` is the gcnArchName up to the first ':'.
constexpr bool mini_combine_image_arch(std::string_view arch) { return arch == "gfx1151"; }

}  // namespace strix::inkling2
