#pragma once

#include "strix/deepseek41_attention_state.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

// Prefix cache of deepseek41 attention states (strix/ds41-attention-state-v1) for prompts that share a beginning:
// multi-turn chat (each turn extends the previous prompt), a common system prompt, RAG documents.
//
// A prompt is cut into blocks of `block` tokens (a multiple of 8, the prefill's window, and of every compressor ratio,
// so the compressed entries of a block are its own and the ratio-2 carry is empty at a boundary). Block k of a prompt
// is a node keyed by the hash chain key_k = FNV(key_{k-1}, tokens of block k) and holds what the block adds to the
// state: its tokens (compared on lookup, so a hash collision is a miss, never a wrong hit) and, per KV-source layer,
// its compressed KV and index-key entries. Those never change once written, so blocks are shared by every prompt with
// the same beginning. A node may also hold a snapshot: the per-sequence pieces that are not append-only (every layer's
// window ring, the ratio-2 carries) plus the greedy token after the boundary and the FNV of its logits. A snapshot at
// boundary p, the KV pieces of blocks 0..p/block - 1 and their tokens are exactly the AttentionState a prefill of those
// p tokens exports (Engram history = the tokens; the last row's top-k is left out: neither a prefill extend nor a decode
// step reads it, both select again), so the device prefill resumes from it (DevicePrefill::import_state + extend) and
// gives every row, the logits and the final state of a fresh prefill bitwise.
//
// Snapshots are taken where a prefill run ends (a run split costs the decoder cone of the exact suffix replay once
// more, see ds41_prefill.hpp): plan_prefill_pieces cuts a prompt at the resume point, at the deepest block shared with
// cached prompts (a branch point, e.g. the end of a common system prompt; once), at max_rows and at the prompt's last
// full block; every piece end on a block boundary is inserted.
//
// Memory: about 890 B per cached token (KV + index keys) + 4 B (token) and 2.9 MB per snapshot (40 rings of 136 x
// 528 B). The budget counts these bytes; on insert the least recently used snapshots go first (and the blocks only
// they needed), never anything a live request holds (the cache holds copies on the host, apart from the KV pool).
//
// Determinism for two ranks: lookup / state / insert / eviction depend only on the calls and the state bytes (the
// attention state is replicated on both ranks), with a logical clock, so two ranks making the same calls hold the
// same cache. Serving lets rank 0 decide (hit position and key go to rank 1 in the control block) and rank 1 checks.
namespace strix::deepseek41 {

struct PrefixCacheOptions {
    std::uint32_t block = 256;            // tokens per block (multiple of 8 and of the compressor ratios)
    std::uint64_t budget_bytes = 0;       // 0: no limit
};

struct PrefixCacheStats {
    std::uint64_t bytes = 0, blocks = 0, snapshots = 0;
    std::uint64_t lookups = 0, hits = 0, hit_tokens = 0, inserts = 0, evicted_snapshots = 0, evicted_blocks = 0, refused = 0;
};

struct PrefixLookup {
    std::uint32_t resume = 0;   // longest snapshot boundary (multiple of block, <= tokens.size()); 0: none
    std::uint64_t key = 0;      // its node's chain key (0 when resume is 0)
    std::uint32_t shared = 0;   // deepest block boundary whose blocks are cached (>= resume): a branch point
    std::uint32_t next_token = 0;  // the snapshot's greedy token after `resume`
    std::uint64_t logits_fnv = 0;  // FNV of the logits at resume - 1
};

class PrefixCache {
public:
    explicit PrefixCache(const PrefixCacheOptions& options);
    ~PrefixCache();
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] std::uint32_t block() const noexcept { return options_.block; }
    [[nodiscard]] const PrefixCacheOptions& options() const noexcept { return options_; }
    // The longest resumable prefix of `tokens` (counts a lookup, and a hit when resume > 0; no LRU change). With
    // min_rows > 1 a snapshot that would leave a prefill of fewer than min_rows rows is passed over (resume is the whole
    // prompt or leaves at least min_rows): the two-rank row-split prefill needs every run to give each rank a row.
    [[nodiscard]] PrefixLookup lookup(std::span<const std::uint32_t> tokens, std::uint32_t min_rows = 1);
    // The state at boundary `position` of `tokens` (a lookup's resume), its snapshot made most recently used. Throws when
    // the cache does not hold it (two ranks out of step). topk empty and unknown; next_token from the snapshot.
    [[nodiscard]] AttentionState state(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key);
    // Inserts what `s` (an exported state at a block boundary, s.tokens its prompt, s.next_token its greedy token) adds:
    // the blocks the cache lacks and the snapshot at s.position (an existing one is only made most recently used).
    // Evicts least recently used snapshots to stay within the budget; false (nothing inserted) when it cannot fit or
    // the position is not a block boundary.
    bool insert(const AttentionState& s, std::uint64_t logits_fnv);
    void clear();
    [[nodiscard]] PrefixCacheStats stats() const;
    // Bytes the cache would take for one snapshot at `position` of a prompt sharing nothing (the budget unit).
    [[nodiscard]] static std::uint64_t snapshot_bytes(const AttentionState& s);

    struct Node;

private:
    PrefixCacheOptions options_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> nodes_;
    std::vector<AttentionLayerState> layout_;  // roles of each layer (owns_kv, owns_index, ratio), from the first insert
    std::uint64_t clock_ = 0;
    PrefixCacheStats stats_;
    void evict_one();
    void prune(Node* n);
};

// Hash chain of blocks: key of the first `blocks` blocks of `tokens` (0 for none).
[[nodiscard]] std::uint64_t prefix_key(std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint32_t blocks);

// The piece ends of a prefill of `total` tokens that resumes at `resume`: the deepest shared block boundary (when it
// lies between resume and the last full block), every max_rows rows (rounded down to a block boundary when possible,
// else to 8), the last full block boundary that leaves at least min_rows rows after it, and total. Strictly increasing,
// all > resume, the last = total, every piece at least min_rows rows when total - resume >= min_rows (a boundary that
// would make a shorter piece is dropped); empty when resume == total.
[[nodiscard]] std::vector<std::uint32_t> plan_prefill_pieces(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                                             std::uint32_t max_rows, std::uint32_t min_rows = 1);

}  // namespace strix::deepseek41
