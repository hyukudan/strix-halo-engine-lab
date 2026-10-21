#include "strix/deepseek41_prefix_cache.hpp"

#include "strix/ds41_attention.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::deepseek41 {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL, kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kNodeOverhead = 256;  // bookkeeping bytes counted per node and per snapshot

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

std::uint64_t chain(std::uint64_t parent, std::span<const std::uint32_t> block) {
    std::uint64_t h = fnv_bytes(&parent, sizeof(parent), kFnvOffset);
    h = fnv_bytes(block.data(), block.size() * 4U, h);
    return h == 0U ? 1U : h;  // 0 means "no prefix"
}

}  // namespace

struct PrefixCache::Node {
    std::uint64_t key = 0;
    Node* parent = nullptr;
    std::uint32_t depth = 0;  // blocks up to and including this one
    std::uint32_t children = 0;
    std::vector<std::uint32_t> tokens;                       // the block's tokens
    std::vector<std::vector<std::uint8_t>> kv, idx;          // per layer (empty outside the KV sources): the block's entries
    struct Snapshot {
        std::vector<std::vector<std::uint8_t>> rings;        // per layer
        std::vector<std::vector<float>> carries;             // per layer (ratio-2 sources)
        std::vector<char> carry_valid;
        std::uint32_t next_token = 0;
        std::uint64_t logits_fnv = 0;
        std::uint64_t last_use = 0;
        std::uint64_t bytes = 0;
    };
    std::unique_ptr<Snapshot> snapshot;
    std::uint64_t bytes = 0;  // the block's own bytes (without the snapshot)
};

std::uint64_t prefix_key(std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint32_t blocks) {
    std::uint64_t key = 0;
    for (std::uint32_t k = 0; k < blocks; ++k) key = chain(key, tokens.subspan(std::size_t{k} * block, block));
    return key;
}

std::vector<std::uint32_t> plan_prefill_pieces(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                               std::uint32_t max_rows, std::uint32_t min_rows) {
    if (block == 0U || block % 8U != 0U || max_rows < 8U) throw std::invalid_argument("plan_prefill_pieces: block a multiple of 8, max_rows >= 8");
    if (resume > total) throw std::invalid_argument("plan_prefill_pieces: resume beyond the prompt");
    min_rows = std::max<std::uint32_t>(min_rows, 1U);
    if (min_rows > max_rows / 2U) throw std::invalid_argument("plan_prefill_pieces: min_rows > max_rows / 2");
    std::vector<std::uint32_t> ends;
    std::uint32_t last_full = total / block * block;  // the last full block, unless it leaves a too short tail
    if (total - last_full != 0U && total - last_full < min_rows) last_full = last_full >= block ? last_full - block : 0U;
    if (shared > resume && shared < last_full) ends.push_back(shared);
    if (last_full > resume && last_full < total) ends.push_back(last_full);
    ends.push_back(total);
    if (resume == total) return {};
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    // split pieces longer than max_rows: at block boundaries when max_rows allows, else at multiples of 8
    std::vector<std::uint32_t> out;
    std::uint32_t at = resume;
    const std::uint32_t step = max_rows >= block ? max_rows / block * block : max_rows / 8U * 8U;
    for (const auto e : ends) {
        if (e != total && (e - at < min_rows || total - e < min_rows)) continue;  // would make a piece too short
        while (e - at > max_rows) {
            // a max_rows step, unless it leaves less than min_rows before e
            const std::uint32_t next = e - (at + step) >= min_rows ? at + step : (e - min_rows) / 8U * 8U;
            at = next;
            out.push_back(at);
        }
        out.push_back(e);
        at = e;
    }
    return out;
}

PrefixCache::PrefixCache(const PrefixCacheOptions& options) : options_(options) {
    if (options.block == 0U || options.block % 8U != 0U) throw std::invalid_argument("deepseek41 prefix cache: block must be a positive multiple of 8");
}

PrefixCache::~PrefixCache() = default;

void PrefixCache::clear() {
    nodes_.clear();
    const auto lookups = stats_.lookups, hits = stats_.hits, hit_tokens = stats_.hit_tokens, inserts = stats_.inserts;
    stats_ = {};
    stats_.lookups = lookups;
    stats_.hits = hits;
    stats_.hit_tokens = hit_tokens;
    stats_.inserts = inserts;
}

PrefixCacheStats PrefixCache::stats() const { return stats_; }

PrefixLookup PrefixCache::lookup(std::span<const std::uint32_t> tokens, std::uint32_t min_rows) {
    PrefixLookup out;
    ++stats_.lookups;
    const std::uint32_t B = options_.block;
    std::uint64_t key = 0;
    for (std::uint32_t k = 0; std::size_t{k + 1U} * B <= tokens.size(); ++k) {
        const auto piece = tokens.subspan(std::size_t{k} * B, B);
        key = chain(key, piece);
        const auto it = nodes_.find(key);
        if (it == nodes_.end()) break;
        const Node& n = *it->second;
        if (n.depth != k + 1U || !std::equal(piece.begin(), piece.end(), n.tokens.begin(), n.tokens.end())) break;  // collision
        if (k > 0U && (n.parent == nullptr || n.parent->depth != k)) break;
        out.shared = (k + 1U) * B;
        const std::size_t left = tokens.size() - out.shared;
        if (n.snapshot && (left == 0U || left >= min_rows)) {
            out.resume = out.shared;
            out.key = key;
            out.next_token = n.snapshot->next_token;
            out.logits_fnv = n.snapshot->logits_fnv;
        }
    }
    if (out.resume > 0U) {
        ++stats_.hits;
        stats_.hit_tokens += out.resume;
    }
    return out;
}

AttentionState PrefixCache::state(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key) {
    const std::uint32_t B = options_.block;
    const auto fail = [](const std::string& what) { throw std::runtime_error("deepseek41 prefix cache: " + what); };
    if (position == 0U || position % B != 0U || position > tokens.size()) fail("state: not a block boundary of the prompt");
    if (prefix_key(tokens, B, position / B) != key) fail("state: the key does not match the prompt");
    const auto it = nodes_.find(key);
    if (it == nodes_.end() || !it->second->snapshot || it->second->depth != position / B) fail("state: no snapshot at the boundary");
    Node* leaf = it->second.get();
    std::vector<const Node*> path(position / B);
    for (const Node* n = leaf; n != nullptr; n = n->parent) {
        if (n->depth == 0U || n->depth > path.size()) fail("state: broken chain");
        path[n->depth - 1U] = n;
    }
    AttentionState s;
    s.position = position;
    s.next_token = leaf->snapshot->next_token;
    s.tokens.reserve(position);
    for (std::uint32_t k = 0; k < path.size(); ++k) {
        if (path[k] == nullptr) fail("state: missing block");
        if (!std::equal(path[k]->tokens.begin(), path[k]->tokens.end(), tokens.begin() + std::size_t{k} * B)) fail("state: tokens differ");
        s.tokens.insert(s.tokens.end(), path[k]->tokens.begin(), path[k]->tokens.end());
    }
    s.layers.resize(layout_.size());
    const auto& snap = *leaf->snapshot;
    for (std::size_t l = 0; l < layout_.size(); ++l) {
        auto& L = s.layers[l];
        L.owns_kv = layout_[l].owns_kv;
        L.owns_index = layout_[l].owns_index;
        L.ratio = layout_[l].ratio;
        L.topk_known = false;
        L.ring = snap.rings[l];
        if (L.owns_kv) {
            std::size_t kv = 0, ix = 0;
            for (const auto* n : path) {
                kv += n->kv[l].size();
                ix += n->idx[l].size();
            }
            L.kv.reserve(kv);
            L.idx_keys.reserve(ix);
            for (const auto* n : path) {
                L.kv.insert(L.kv.end(), n->kv[l].begin(), n->kv[l].end());
                L.idx_keys.insert(L.idx_keys.end(), n->idx[l].begin(), n->idx[l].end());
            }
            L.carry = snap.carries[l];
            L.carry_valid = snap.carry_valid[l] != 0;
        }
    }
    leaf->snapshot->last_use = ++clock_;
    return s;
}

std::uint64_t PrefixCache::snapshot_bytes(const AttentionState& s) {
    std::uint64_t n = kNodeOverhead + s.tokens.size() * 4U;
    for (const auto& L : s.layers) n += L.ring.size() + L.kv.size() + L.idx_keys.size() + L.carry.size() * 4U;
    return n;
}

bool PrefixCache::insert(const AttentionState& s, std::uint64_t logits_fnv) {
    namespace attn = ds41::attn;
    const std::uint32_t B = options_.block;
    if (s.position == 0U || s.position % B != 0U || s.tokens.size() != s.position) return false;
    if (layout_.empty()) {
        for (const auto& L : s.layers) {
            AttentionLayerState r;
            r.owns_kv = L.owns_kv;
            r.owns_index = L.owns_index;
            r.ratio = L.ratio;
            if (L.owns_kv && (L.ratio == 0U || B % L.ratio != 0U)) throw std::invalid_argument("deepseek41 prefix cache: block not a multiple of a compressor ratio");
            layout_.push_back(r);
        }
    }
    if (s.layers.size() != layout_.size()) throw std::invalid_argument("deepseek41 prefix cache: layer count differs from the cache's");
    for (std::size_t l = 0; l < layout_.size(); ++l) {
        const auto& L = s.layers[l];
        if (L.owns_kv != layout_[l].owns_kv || L.ratio != layout_[l].ratio) throw std::invalid_argument("deepseek41 prefix cache: layer roles differ");
        if (L.owns_kv && (L.kv.size() != std::size_t{s.position / L.ratio} * attn::kKvEntryBytes ||
                          L.idx_keys.size() != std::size_t{s.position / L.ratio} * attn::kIdxEntryBytes))
            throw std::invalid_argument("deepseek41 prefix cache: cache sizes of layer " + std::to_string(l));
    }
    const std::uint32_t blocks = s.position / B;
    const std::span<const std::uint32_t> tokens(s.tokens);
    // the chain as far as the cache has it
    std::vector<std::uint64_t> keys(blocks);
    std::uint32_t have = 0;
    {
        std::uint64_t key = 0;
        bool known = true;
        for (std::uint32_t k = 0; k < blocks; ++k) {
            const auto piece = tokens.subspan(std::size_t{k} * B, B);
            key = chain(key, piece);
            keys[k] = key;
            if (!known) continue;
            const auto it = nodes_.find(key);
            if (it != nodes_.end() && it->second->depth == k + 1U && std::equal(piece.begin(), piece.end(), it->second->tokens.begin(), it->second->tokens.end()))
                have = k + 1U;
            else
                known = false;
        }
    }
    // a key in use by another chain (hash collision): do not insert
    for (std::uint32_t k = have; k < blocks; ++k)
        if (nodes_.contains(keys[k])) {
            ++stats_.refused;
            return false;
        }
    if (have == blocks && nodes_.at(keys[blocks - 1U])->snapshot) {
        nodes_.at(keys[blocks - 1U])->snapshot->last_use = ++clock_;
        return true;
    }
    // bytes this insert adds
    const auto block_bytes = [&](std::uint32_t) {
        std::uint64_t n = kNodeOverhead + std::uint64_t{B} * 4U;
        for (const auto& L : s.layers)
            if (L.owns_kv) n += std::uint64_t{B / L.ratio} * (attn::kKvEntryBytes + attn::kIdxEntryBytes);
        return n;
    };
    std::uint64_t snap_bytes = kNodeOverhead;
    for (const auto& L : s.layers) snap_bytes += L.ring.size() + L.carry.size() * 4U;
    std::uint64_t need = snap_bytes;
    for (std::uint32_t k = have; k < blocks; ++k) need += block_bytes(k);
    if (options_.budget_bytes != 0U) {
        if (need > options_.budget_bytes) {
            ++stats_.refused;
            return false;
        }
        // evict, but never the chain this insert extends (pin its existing nodes by giving them a reference)
        Node* tip = have > 0U ? nodes_.at(keys[have - 1U]).get() : nullptr;
        if (tip != nullptr) ++tip->children;  // a pin: prune() never removes a node with children
        while (stats_.bytes + need > options_.budget_bytes && stats_.snapshots > 0U) evict_one();
        if (tip != nullptr) --tip->children;
        if (stats_.bytes + need > options_.budget_bytes) {
            if (tip != nullptr) prune(tip);
            ++stats_.refused;
            return false;
        }
    }
    Node* parent = have > 0U ? nodes_.at(keys[have - 1U]).get() : nullptr;
    for (std::uint32_t k = have; k < blocks; ++k) {
        auto n = std::make_unique<Node>();
        n->key = keys[k];
        n->parent = parent;
        n->depth = k + 1U;
        n->tokens.assign(tokens.begin() + std::size_t{k} * B, tokens.begin() + std::size_t{k + 1U} * B);
        n->kv.resize(layout_.size());
        n->idx.resize(layout_.size());
        for (std::size_t l = 0; l < layout_.size(); ++l) {
            const auto& L = s.layers[l];
            if (!L.owns_kv) continue;
            const std::size_t per = B / L.ratio, first = std::size_t{k} * per;
            n->kv[l].assign(L.kv.begin() + first * attn::kKvEntryBytes, L.kv.begin() + (first + per) * attn::kKvEntryBytes);
            n->idx[l].assign(L.idx_keys.begin() + first * attn::kIdxEntryBytes, L.idx_keys.begin() + (first + per) * attn::kIdxEntryBytes);
        }
        n->bytes = block_bytes(k);
        stats_.bytes += n->bytes;
        ++stats_.blocks;
        if (parent != nullptr) ++parent->children;
        parent = n.get();
        nodes_.emplace(keys[k], std::move(n));
    }
    Node* leaf = parent;
    auto snap = std::make_unique<Node::Snapshot>();
    snap->rings.resize(layout_.size());
    snap->carries.resize(layout_.size());
    snap->carry_valid.assign(layout_.size(), 0);
    for (std::size_t l = 0; l < layout_.size(); ++l) {
        snap->rings[l] = s.layers[l].ring;
        snap->carries[l] = s.layers[l].carry;
        snap->carry_valid[l] = s.layers[l].carry_valid ? 1 : 0;
    }
    snap->next_token = s.next_token;
    snap->logits_fnv = logits_fnv;
    snap->last_use = ++clock_;
    snap->bytes = snap_bytes;
    stats_.bytes += snap_bytes;
    ++stats_.snapshots;
    leaf->snapshot = std::move(snap);
    ++stats_.inserts;
    return true;
}

void PrefixCache::evict_one() {
    Node* victim = nullptr;
    for (auto& [key, n] : nodes_)
        if (n->snapshot && (victim == nullptr || n->snapshot->last_use < victim->snapshot->last_use)) victim = n.get();
    if (victim == nullptr) return;
    stats_.bytes -= victim->snapshot->bytes;
    --stats_.snapshots;
    ++stats_.evicted_snapshots;
    victim->snapshot.reset();
    prune(victim);
}

// Removes `n` and its ancestors while they have neither a snapshot nor children (no resume point needs them).
void PrefixCache::prune(Node* n) {
    while (n != nullptr && !n->snapshot && n->children == 0U) {
        Node* parent = n->parent;
        if (parent != nullptr) --parent->children;
        stats_.bytes -= n->bytes;
        --stats_.blocks;
        ++stats_.evicted_blocks;
        nodes_.erase(n->key);
        n = parent;
    }
}

}  // namespace strix::deepseek41
