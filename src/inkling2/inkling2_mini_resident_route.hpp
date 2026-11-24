#pragma once

#include <optional>
#include <stdexcept>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// The decode expert form of every Mini entry point: ONE resolver for the CLI and strix-serve.
// The incumbent identities stay as they are on both:
// - the CLI decodes on the packed one-row form (moe_core::microscaled_r1; single stream on gfx1151: 19.7 against
//   16.0 tok/s on the grouped tile kernel);
// - strix-serve decodes on the grouped tile kernel (both backends).
// On strix-serve, packed is the identity variant inkling2-mini-moe-packed (strix/profiles/inkling2_mini.hpp, HIP):
// requested by name (--packed-decode / --identity-variant) and admitted by the profile resolver, never by a Boolean. A
// served lane then keeps the CLI's EXACT bits at any cohort width (mini_cohort_pass_cap); its bits differ from the
// grouped served answer, the incumbent serving identity of both backends.
enum class MiniDecodeEntry { cli, serve };

constexpr bool mini_packed_decode(MiniDecodeEntry entry, std::optional<bool> requested) {
    return requested.value_or(entry == MiniDecodeEntry::cli);
}

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

}  // namespace strix::inkling2
