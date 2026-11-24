#pragma once

#include <optional>
#include <stdexcept>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Both CLI backends default to the packed one-row form (moe_core::microscaled_r1). Serving defaults to grouped.
// On strix-serve, packed is the identity variant inkling2-mini-moe-packed (strix/profiles/inkling2_mini.hpp, HIP):
// requested by name (--packed-decode / --identity-variant) and admitted by the profile resolver.
enum class MiniDecodeEntry { cli, serve };

constexpr bool mini_packed_decode(MiniDecodeEntry entry, std::optional<bool> requested) {
    return requested.value_or(entry == MiniDecodeEntry::cli);
}

// The reference a served decode form is checked against, named with the served identity (route line, capabilities):
// Packed references each lane's lone packed decode. Grouped references the served grouped incumbent.
constexpr const char* mini_decode_reference(bool packed) { return packed ? "lone-decode" : "served-grouped"; }

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// With packed decode, decode and verify rows use the packed form.
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
// Packed decode runs passes of at most the
// backend's one-row cap (HIP: the dense rows family keeps the one-row body up to core::gemv_rows::max_rows rows and
// takes the WMMA tile GEMM above). Grouped decode keeps its single pass of every lane.
constexpr unsigned mini_cohort_pass_cap(bool packed_decode, unsigned backend_one_row_cap) {
    return packed_decode ? backend_one_row_cap : 0U;
}

}  // namespace strix::inkling2
