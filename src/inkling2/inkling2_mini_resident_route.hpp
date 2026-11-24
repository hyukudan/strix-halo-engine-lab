#pragma once

#include <optional>

#include "strix/serve/decode_form.hpp"

#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Decode expert form, ONE default for every entry point (the command line and strix-serve): the packed one-row form
// (moe_core::microscaled_r1), whose row bits equal a one-row launch at any cohort width. --grouped-decode selects the
// grouped tile kernel on both entry points (a declared lever value, printed as moe_decode in the route line); the
// grouped form makes no batch-invariance contract. The same request gets the same form wherever it arrives.
inline constexpr bool kMiniPackedDecodeDefault = true;

// --packed-decode (true) / --grouped-decode (false) / neither (the default above).
constexpr bool mini_packed_decode(std::optional<bool> requested) noexcept {
    return requested.value_or(kMiniPackedDecodeDefault);
}

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

// With packed decode every decode and verify row (a lone token, a verify window or a cohort of lanes of any width)
// runs the packed expert form; prefill chunks of more than one row keep the grouped tile kernel.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || kind != MiniAppendKind::prefill),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

// Row cap of one batched cohort decode pass (0 = the whole cohort in one pass, serve::plan_cohort_passes). Packed
// decode promises that a lane's bits do not depend on the cohort width, so it runs passes of at most the backend's
// one-row cap (the dense rows family keeps its one-row body up to that many rows). Grouped runs passes of ONE lane:
// a one-lane grouped pass is the lone grouped decode, so an explicit --grouped-decode lane also keeps its bits at
// any cohort width (the route line prints moe_decode=grouped).
constexpr unsigned mini_cohort_pass_cap(bool packed_decode, unsigned backend_one_row_cap) noexcept {
    return packed_decode ? backend_one_row_cap : 1U;
}

// HIP targets the exact down-image lever (inkling2-mini-combine-down-image@1) is qualified on: the NaN identity fixture and
// the model receipts cover gfx1151 only; the resident device policy admits more targets for the rest of the model.
// `arch` is the gcnArchName up to the first ':'.
constexpr bool mini_combine_image_arch(std::string_view arch) { return arch == "gfx1151"; }

}  // namespace strix::inkling2
