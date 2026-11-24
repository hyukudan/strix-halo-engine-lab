#pragma once

#include "strix/model_data/inkling2_mini.hpp"

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Decode experts on the packed one-row form (moe_core::microscaled_r1). The CLI and HIP serving below
// kMiniGroupedDecodeMinSlots resolve to this value, so a served lane and the CLI share one EXACT identity;
// --grouped-decode keeps the grouped tile kernel.
inline constexpr bool kMiniPackedDecodeDefault = true;
// Cohort width (server slots) from which HIP serving decodes experts on the grouped tile kernel (model data).
inline constexpr unsigned kMiniGroupedDecodeMinSlots =
    static_cast<unsigned>(model_data::inkling2_mini::moe::grouped_decode_min_slots);

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

// A cohort of `lanes` decode lanes runs as `count` batched passes of at most `cap` rows, the row count up to which
// the backend's dense rows keep every row on its one-row body (with the packed experts above, a lane's bits then do
// not depend on the cohort size). Balanced: the first `extra` passes take base + 1 rows and the others base, so a
// cohort above the cap never leaves a pass of one row.
struct MiniCohortPasses {
    unsigned count;
    unsigned base;
    unsigned extra;
    [[nodiscard]] constexpr unsigned rows(unsigned pass) const noexcept { return base + (pass < extra ? 1U : 0U); }
    [[nodiscard]] constexpr unsigned first(unsigned pass) const noexcept {
        return pass * base + (pass < extra ? pass : extra);
    }
};

constexpr MiniCohortPasses mini_cohort_passes(unsigned lanes, unsigned cap) {
    if (lanes == 0U || cap == 0U) return {0U, 0U, 0U};
    const unsigned count = lanes / cap + (lanes % cap != 0U ? 1U : 0U);
    return {count, lanes / count, lanes % count};
}

}  // namespace strix::inkling2
