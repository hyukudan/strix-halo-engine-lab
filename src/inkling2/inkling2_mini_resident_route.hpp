#pragma once

#include "strix/model_data/inkling2_mini.hpp"

#include <optional>
#include <stdexcept>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Cohort width (server slots) from which the slot rule decodes experts on the grouped tile kernel (model data,
// [moe] grouped_decode_min_slots of models/inkling2-mini.toml).
inline constexpr unsigned kMiniGroupedDecodeMinSlots =
    static_cast<unsigned>(model_data::inkling2_mini::moe::grouped_decode_min_slots);

// The decode expert form of every Mini entry point. The incumbent identities stay as they are:
// - the CLI decodes on the packed one-row form (moe_core::microscaled_r1), its lone decode;
// - strix-serve decodes on the grouped tile kernel on both backends: the incumbent serving identity.
// On strix-serve, packed is the identity variant inkling2-mini-moe-packed (strix/profiles/inkling2_mini.hpp, HIP only):
// requested by name and admitted by the profile resolver, never chosen by a Boolean. mini_decode_form is the pure rule
// that turns the served requests into a variant request (mini_serve_decode_request), by precedence:
// 1. an explicit form (--packed-decode / --identity-variant inkling2-mini-moe-packed / --grouped-decode) wins at any
//    width, also beside --decode-by-slots; packed and grouped exclude each other;
// 2. the slot rule (--decode-by-slots, opt-in) chooses once at start by the configured cohort width: on HIP it requests
//    the packed variant below kMiniGroupedDecodeMinSlots and keeps the grouped incumbent from there on; CUDA keeps the
//    grouped incumbent (the variant is not declared there);
// 3. no request: each entry keeps its incumbent.
// Only the packed variant is batch invariant (a lane's bits equal its lone one-row decode at any cohort width,
// mini_cohort_pass_cap); the grouped incumbent makes no batch-invariance contract. A server keeps its form for life.
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
