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

// Served default: the one default above only where the server runs the command line's route (a HIP server with the
// served route lever, kMiniServedCliRoute: served lanes == lone decode on the device at every cohort width). Without
// the lever, and on CUDA, a server without a flag keeps the grouped form it served before. Either flag selects the same
// form on both entry points.
constexpr bool mini_served_packed_default(bool hip_backend, bool cli_route) noexcept {
    return hip_backend && cli_route && kMiniPackedDecodeDefault;
}

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
// one-row cap (the dense rows family keeps its one-row body up to that many rows). Grouped keeps the incumbent served
// schedule: the whole cohort in one batched forward (the grouped form makes no batch-invariance contract; its lane bits
// depend on the cohort width, as they did before).
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

}  // namespace strix::inkling2
