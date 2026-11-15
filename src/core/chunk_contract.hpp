#pragma once
// Chunk-invariance contract for attention key segments and block recurrences (chunkinv-attn-state-v1).
// Home: shared (host shared: session / prefix-cache planner and identity; backend cores bind the same functions).
//
// Goal: a row's attention output and a recurrent layer's state depend only on the token sequence and the absolute
// position, never on the call's chunk size, its grouping with neighbouring queries, or a prefix-cache hit.
//
// 1. Attention key segments (split-K over keys). Segment s of a query covers absolute key tiles
//    [s * seg_tiles, (s + 1) * seg_tiles) clipped to the query's own visible range; partials combine in ascending s.
//    The boundaries are a function of the absolute tile index only. The current dense flash split
//    (src/attn_core/dense_flash_prefill.hpp:90-94) derives `share` from the tiles of the whole row block, so the
//    boundaries move with the first/last query of the block: that form is kept as block_share_segment() for the
//    negative control and is the default today (bit-exact paths do not change). splits == 1 without a window is
//    already invariant: fully masked 16-key tiles are exact no-ops (src/attn_core/online_softmax.hpp:9,36-41).
// 2. Block recurrences (Qwen GDN FAST WY C32, src/hip_gdn_fp16_chunk.inc:13,32-33; GLM CUDA KDA B32,
//    kda.cu:11,168-175) close their partial block at the end of every call, so a cut that is not a multiple of the
//    block from the sequence start is another arithmetic. Contract: a non-final call commits state and KV only up
//    to the last absolute block boundary; the open tail tokens are carried and recomputed by the next call (the
//    "recompute its prefix" option). A prefix-cache entry is stored only at a committed boundary. The final call
//    closes its tail exactly as a whole-prompt run does, at the same absolute position.
// 3. Identity: every state / prefix-cache version that relies on 1-2 folds chunk_contract_tag() in, so entries made
//    under the old per-call closing are never reused by the new contract (and vice versa).
#include <cstdint>
#include <string>

namespace strix::chunk_contract {

inline constexpr const char* kContractName = "chunkinv-attn-state-v1";

struct Segment {
    unsigned tile_begin = 0;  // inclusive, absolute 16-key tile index
    unsigned tile_end = 0;    // exclusive
    bool empty() const noexcept { return tile_end <= tile_begin; }
};

// Absolute segment s for a query whose visible key tiles are [lo_tile, hi_tile] (inclusive).
constexpr Segment absolute_segment(unsigned lo_tile, unsigned hi_tile, unsigned seg_tiles, unsigned s) noexcept {
    if (seg_tiles == 0U || hi_tile < lo_tile) return {};
    const unsigned b = s * seg_tiles, e = b + seg_tiles;
    const unsigned cb = b > lo_tile ? b : lo_tile, ce = e < hi_tile + 1U ? e : hi_tile + 1U;
    return cb < ce ? Segment{cb, ce} : Segment{cb, cb};
}
// Number of absolute segments a query visible up to hi_tile touches (segments are indexed from tile 0).
constexpr unsigned absolute_segment_count(unsigned hi_tile, unsigned seg_tiles) noexcept {
    return seg_tiles ? hi_tile / seg_tiles + 1U : 0U;
}

// Negative control / current default: dense_flash_prefill.hpp:90-94 form. key_begin_tile and last_tile come from the
// row BLOCK (first and last query), so the boundaries depend on neighbouring queries.
constexpr Segment block_share_segment(unsigned key_begin_tile, unsigned last_tile, unsigned splits, unsigned z) noexcept {
    const unsigned total = last_tile - key_begin_tile + 1U, share = (total + splits - 1U) / splits;
    const unsigned b = key_begin_tile + (z * share < total ? z * share : total);
    const unsigned e = key_begin_tile + ((z + 1U) * share < total ? (z + 1U) * share : total);
    return {b, e};
}

// Open-block plan for one call of a block recurrence.
struct CallPlan {
    std::uint64_t run_begin = 0;     // absolute first token this call processes (= committed boundary of the session)
    std::uint64_t commit_end = 0;    // absolute end of state / KV / cache commit (exclusive)
    std::uint64_t carry_tokens = 0;  // open-tail tokens [commit_end, call_end) recomputed by the next call
};

// committed: tokens already committed by the session (always a block multiple, or the prompt end).
// call_end: absolute end of the tokens available to this call. final: no more tokens follow in this prefill.
constexpr CallPlan plan_call(std::uint64_t committed, std::uint64_t call_end, unsigned block, bool final) noexcept {
    CallPlan p{committed, call_end, 0};
    if (final || block <= 1U || call_end <= committed) return p;
    const std::uint64_t aligned = call_end / block * block;
    p.commit_end = aligned > committed ? aligned : committed;
    p.carry_tokens = call_end - p.commit_end;
    return p;
}
// A prefix-cache entry may be stored at `end` only when it is a committed boundary.
constexpr bool cacheable_boundary(std::uint64_t end, unsigned block) noexcept { return block <= 1U || end % block == 0U; }

// A block recurrence that closes its partial block per call is chunk-invariant over a call plan when every call that
// starts before the last absolute block boundary floor(L / block) * block also ends on a block boundary: then the
// only partial blocks are in the final suffix, whose split is a function of L (and the tail ladder) alone.
// ends: absolute end of each call in order, last == L.
template <class Ends>
constexpr bool block_plan_canonical(const Ends& ends, std::uint64_t begin, unsigned block) noexcept {
    if (block <= 1U) return true;
    std::uint64_t last = begin, total = begin;
    for (const auto e : ends) total = e;
    const std::uint64_t suffix = total / block * block;
    for (const auto e : ends) {
        if (last < suffix && e % block != 0U) return false;
        last = e;
    }
    return true;
}

// Fold the contract into an existing state / cache version string.
inline std::string chunk_contract_tag(const std::string& base_version, unsigned block, unsigned seg_tiles) {
    return base_version + "+" + kContractName + "/b" + std::to_string(block) + "/s" + std::to_string(seg_tiles);
}

}  // namespace strix::chunk_contract
