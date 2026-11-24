#pragma once

#include "strix/model_data/inkling2_mini.hpp"

#include <optional>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Cohort width (server slots) from which the slot rule decodes experts on the grouped tile kernel (model data,
// [moe] grouped_decode_min_slots of models/inkling2-mini.toml).
inline constexpr unsigned kMiniGroupedDecodeMinSlots =
    static_cast<unsigned>(model_data::inkling2_mini::moe::grouped_decode_min_slots);

// The decode expert form of every Mini entry point: ONE resolver for the CLI and strix-serve. Precedence:
// 1. an explicit form (--packed-decode / --grouped-decode) wins at any width on either entry point;
// 2. the slot rule (--decode-by-slots, strix-serve) chooses once at start by cohort width: on HIP the packed one-row
//    form (the CLI's form and EXACT identity) below kMiniGroupedDecodeMinSlots, the grouped tile kernel from there on
//    (served gfx1151 EXACT, packed / grouped aggregate tok/s: C1 27.1 / 23.0, C8 72.5 / 67.4, C16 79.2 / 88.6);
//    CUDA keeps the grouped tile kernel (its packed form runs one launch per lane);
// 3. without a request each entry keeps its incumbent identity: the CLI decodes packed (single stream on gfx1151:
//    19.7 against 16.0 tok/s), strix-serve grouped (both backends).
// An explicit form wins because the operator names the identity it serves. The slot rule is opt-in because a
// server's form fixes the bits of every answer it serves and packed bits differ from the grouped served answer:
// it becomes the serving default only as its own admitted identity, never silently. Each form is batch invariant
// on its own (packed: mini_cohort_pass_cap) and a server keeps its form for its whole life.
enum class MiniDecodeEntry { cli, serve };
enum class MiniDecodeRequest { packed, grouped, by_slots };
enum class MiniDecodeReason { entry, requested, slots_below, slots_from, backend };

struct MiniDecodeForm {
    bool packed;
    MiniDecodeReason reason;
};

constexpr MiniDecodeForm mini_decode_form(MiniDecodeEntry entry, std::optional<MiniDecodeRequest> request,
                                          unsigned cohort_rows = 1U, bool hip = true) {
    if (!request) return {entry == MiniDecodeEntry::cli, MiniDecodeReason::entry};
    if (*request != MiniDecodeRequest::by_slots)
        return {*request == MiniDecodeRequest::packed, MiniDecodeReason::requested};
    if (!hip) return {false, MiniDecodeReason::backend};
    return cohort_rows < kMiniGroupedDecodeMinSlots ? MiniDecodeForm{true, MiniDecodeReason::slots_below}
                                                    : MiniDecodeForm{false, MiniDecodeReason::slots_from};
}

constexpr bool mini_packed_decode(MiniDecodeEntry entry, std::optional<bool> requested) {
    return mini_decode_form(entry, requested ? std::optional{*requested ? MiniDecodeRequest::packed
                                                                        : MiniDecodeRequest::grouped}
                                             : std::nullopt).packed;
}

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
