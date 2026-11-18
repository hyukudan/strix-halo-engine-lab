#include "strix/spec/block_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix::spec {

void validate(const BlockPolicy& p) {
    if (p.block < 2U || p.block > 32U) throw std::invalid_argument("block_parallel: block must be 2..32 rows");
    if (p.draft_layers == 0U || p.draft_layers > max_draft_layers)
        throw std::invalid_argument("block_parallel: draft_layers must be 1..8");
    if (p.target_layer_count > max_draft_layers) throw std::invalid_argument("block_parallel: too many target layers");
    if (p.kv == KvSource::own_ring && p.ring == 0U) throw std::invalid_argument("block_parallel: own_ring needs ring > 0");
    if (p.kv == KvSource::target_share) {
        if (p.share_budget == 0U) throw std::invalid_argument("block_parallel: target_share needs share_budget > 0");
        if (p.target_layer_count != p.draft_layers)
            throw std::invalid_argument("block_parallel: target_share needs one source layer per draft layer");
    }
    if (max_drafts(p, 0U) > 16U) throw std::invalid_argument("block_parallel: more than 16 proposals per pass");
}

std::uint32_t max_drafts(const BlockPolicy& p, std::uint32_t prefix) {
    if (prefix > 0U && !p.prefix_in_block) throw std::invalid_argument("block_parallel: policy takes no prefix");
    const std::uint32_t total = p.anchor_proposes ? p.block : p.block - 1U;
    return prefix < total ? total - prefix : 0U;
}

std::uint32_t rows_needed(const BlockPolicy& p, std::uint32_t prefix, std::uint32_t k) {
    // Bidirectional rows see the trailing fill rows, so every row's output depends on the whole block: run it all.
    if (p.mask == BlockMask::bidirectional) return p.block;
    const std::uint32_t last = proposal_row(p, prefix + (k > 0U ? k - 1U : 0U));
    return std::max(last, prefix) + 1U;
}

std::uint32_t proposal_row(const BlockPolicy& p, std::uint32_t j) { return p.anchor_proposes ? j : j + 1U; }

void block_ids(const BlockPolicy& p, std::uint32_t anchor, std::span<const std::uint32_t> prefix, std::uint32_t k,
               std::vector<std::uint32_t>& out) {
    const auto n = static_cast<std::uint32_t>(prefix.size());
    if (k > max_drafts(p, n)) throw std::invalid_argument("block_parallel: k exceeds the block");
    const std::uint32_t rows = rows_needed(p, n, k);
    out.assign(rows, p.fill_token);
    out[0] = anchor;
    std::copy(prefix.begin(), prefix.end(), out.begin() + 1);
}

bool row_visible(const BlockPolicy& p, std::uint32_t q, std::uint32_t r) {
    if (q >= p.block || r >= p.block) return false;
    return p.mask == BlockMask::bidirectional || r <= q;
}

BlockPolicy hy4_dfly2() {
    BlockPolicy p;
    p.block = 6U;  // num_speculative_tokens 5 + anchor (SGLang dfly block size 6)
    p.fill_token = 120009U;
    p.kv = KvSource::target_share;
    p.share_budget = 2048U;  // dfly_target_index_topk
    // causal + correction off: measured on the official drafter (strix-hy4-dfly accept, oracle taps, 89 anchors): E 2.43
    // causal vs 1.84 bidirectional; every hidden_correction placement tried lowers E (the hook stays for its owner).
    p.mask = BlockMask::causal;
    p.fusion = Fusion::slot0;
    p.correction = false;
    p.draft_layers = 5U;
    p.target_layer_count = 5U;
    p.target_layers = {25, 29, 33, 37, 41, 0, 0, 0};
    return p;
}

BlockPolicy ds41_dspark() {
    BlockPolicy p;
    p.block = 5U;            // last committed token + 4 noise tokens
    p.fill_token = 128799U;  // ds41_dspark.hpp noise_token
    p.kv = KvSource::own_ring;
    p.ring = 128U;
    p.mask = BlockMask::bidirectional;
    p.anchor_proposes = true;  // one token per block row
    p.draft_layers = 3U;
    p.target_layer_count = 3U;  // hidden taps (HC mean of the input stream)
    p.target_layers = {37, 38, 39, 0, 0, 0, 0, 0};
    return p;
}

BlockPolicy g4_dflash() {
    BlockPolicy p;
    p.block = 8U;  // [anchor, 4 x 7]
    p.fill_token = 4U;
    p.kv = KvSource::own_ring;
    p.ring = 1024U;
    p.mask = BlockMask::bidirectional;
    p.draft_layers = 5U;
    p.target_layer_count = 5U;
    p.target_layers = {5, 11, 17, 23, 35, 0, 0, 0};
    return p;
}

ParallelBlockDrafter::ParallelBlockDrafter(const BlockPolicy& policy, BlockForward& forward)
    : policy_(policy), forward_(forward) {
    validate(policy_);
}

void ParallelBlockDrafter::start(std::uint32_t anchor, std::uint64_t anchor_pos) {
    anchor_ = anchor;
    anchor_pos_ = anchor_pos;
}

void ParallelBlockDrafter::draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) {
    const auto n = static_cast<std::uint32_t>(prefix.size());
    k = std::min(k, max_drafts(policy_, n));  // a block holds at most this many; the caller sees out.size()
    out.clear();
    confidence_.clear();
    if (k == 0U) return;
    block_ids(policy_, anchor_, prefix, k, ids_);
    BlockRequest req;
    req.ids = ids_;
    req.anchor_pos = anchor_pos_;
    req.prefix = n;
    req.k = k;
    proposals_.clear();
    forward_.forward(req, proposals_, confidence_);
    ++passes_;
    if (proposals_.size() != k) throw std::logic_error("block_parallel: forward returned a wrong proposal count");
    if (!confidence_.empty() && confidence_.size() != k) throw std::logic_error("block_parallel: wrong confidence count");
    out.assign(proposals_.begin(), proposals_.end());
}

void ParallelBlockDrafter::accept(std::span<const std::uint32_t> emitted) {
    if (emitted.empty()) return;
    anchor_ = emitted.back();
    anchor_pos_ += emitted.size();
    forward_.accept(emitted);
}

BlockConfig parallel_block_config(const BlockPolicy& p, double pass_ms, BlockConfig base) {
    validate(p);
    base.chain = true;
    base.chain_max_k = max_drafts(p, 0U);
    base.chain_seed_flush_ms = pass_ms;
    base.chain_seed_step_ms = 0.0;
    base.hybrid = base.hybrid && p.prefix_in_block;
    return base;
}

double uniform_distinct_experts(std::uint32_t experts, std::uint32_t top_k, std::uint32_t rows) {
    if (experts == 0U || top_k > experts) throw std::invalid_argument("block_parallel: bad expert geometry");
    const double miss = 1.0 - double(top_k) / double(experts);
    return double(experts) * (1.0 - std::pow(miss, double(rows)));
}

double verify_gb(const MoeVerifyBytes& m, std::uint32_t rows) {
    const double distinct = m.experts > 0U ? uniform_distinct_experts(m.experts, m.top_k, rows) : 0.0;
    return m.fixed_gb + m.per_row_gb * double(rows) + distinct * double(m.moe_layers) * m.expert_gb;
}

double break_even_emitted(double plain_ms, double verify_ms, double draft_ms) {
    if (!(plain_ms > 0.0)) throw std::invalid_argument("block_parallel: plain_ms must be > 0");
    return (verify_ms + draft_ms) / plain_ms;
}

std::vector<double> moe_verify_seed_ms(const MoeVerifyBytes& m, double plain_ms, double gbps, std::uint32_t max_rows) {
    if (!(plain_ms > 0.0) || !(gbps > 0.0) || max_rows == 0U)
        throw std::invalid_argument("block_parallel: moe_verify_seed_ms needs plain_ms > 0, gbps > 0, max_rows > 0");
    const double fixed_ms = plain_ms - verify_gb(m, 1U) / gbps * 1000.0;
    if (fixed_ms < 0.0) throw std::invalid_argument("block_parallel: verify bytes exceed the plain wall at gbps");
    std::vector<double> ms(max_rows);
    for (std::uint32_t r = 1; r <= max_rows; ++r) ms[r - 1U] = fixed_ms + verify_gb(m, r) / gbps * 1000.0;
    return ms;
}

MoeVerifyBytes hy4_verify_bytes() {
    // Decode trace (one node, one token): q8_rows Linears 6.00 GB, absorbed K_b / V_b 0.56, BF16 head 0.74, F32 router
    // 0.25 = 7.55 GB read once per pass; routed experts 1.74 + 1.27 = 3.01 GB = 16 x 41 x 4.59 MB.
    constexpr double elems = 1536.0 * 1536.0;
    constexpr double expert_bytes = 2.0 * elems * 144.0 / 256.0 + elems * 210.0 / 256.0;  // Q4_K gate + up, Q6_K down
    MoeVerifyBytes m;
    m.per_row_gb = 0.046;  // selected latent keys + indexer group keys of one row at short context
    m.fixed_gb = 7.55 - m.per_row_gb;
    m.expert_gb = expert_bytes * 1e-9;
    m.experts = 512U;
    m.top_k = 16U;
    m.moe_layers = 41U;
    return m;
}

double expected_emitted(double alpha, std::uint32_t rows) {
    double e = 0.0, a = 1.0;
    for (std::uint32_t i = 0; i < rows; ++i) {
        e += a;
        a *= alpha;
    }
    return e;
}

}  // namespace strix::spec
