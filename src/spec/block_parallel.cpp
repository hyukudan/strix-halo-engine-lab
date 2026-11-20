#include "strix/spec/block_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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
    pending_.clear();
    pending_conf_.clear();
    if (row_verify_ms_ && n == 0U && !confidence_.empty()) {
        const std::uint32_t keep = trim(k);
        if (keep < k) {
            out.resize(keep);
            ++trimmed_;
        }
        pending_.assign(out.begin(), out.end());
        pending_conf_.assign(confidence_.begin(), confidence_.begin() + std::ptrdiff_t(keep));
    }
}

void ParallelBlockDrafter::bind_row_prior(std::function<double(std::uint32_t rows)> verify_ms, RowPriorConfig config) {
    if (!verify_ms || config.bins == 0U || !(config.half_life > 0.0) || !(config.draft_ms >= 0.0))
        throw std::invalid_argument("block_parallel: row prior needs a verify cost, bins > 0, half_life > 0, draft_ms >= 0");
    row_verify_ms_ = std::move(verify_ms);
    row_cfg_ = config;
    bins_.assign(config.bins, {0.0, 0.0});
}

std::size_t ParallelBlockDrafter::bin_of(float confidence) const {
    if (!(confidence > 0.0f)) return 0;
    const auto b = static_cast<std::size_t>(double(confidence) * double(bins_.size()));
    return std::min(b, bins_.size() - 1U);
}

double ParallelBlockDrafter::row_prior(float confidence) const {
    if (bins_.empty()) return -1.0;
    const auto& b = bins_[bin_of(confidence)];
    const double evidence = b[0] + b[1];
    if (evidence < row_cfg_.min_evidence) return -1.0;
    return (b[0] + 1.0) / (evidence + 2.0);
}

// Pass rate of keeping kk proposals: expected emitted ids over (verify(kk + 1) + draft). Ties keep the longer block (the
// controller's pick); every row must be priced by evidence or the block is kept whole.
std::uint32_t ParallelBlockDrafter::trim(std::uint32_t k) {
    for (std::uint32_t j = 0; j < k; ++j)
        if (row_prior(confidence_[j]) < 0.0) return k;
    double best = -1.0, emitted = 1.0, survive = 1.0;
    std::uint32_t pick = k;
    for (std::uint32_t kk = 0; kk <= k; ++kk) {
        if (kk > 0U) {
            survive *= row_prior(confidence_[kk - 1U]);
            emitted += survive;
        }
        const double score = emitted / std::max(row_verify_ms_(kk + 1U) + row_cfg_.draft_ms, 1e-9);
        if (score >= best) {
            best = score;
            pick = kk;
        }
    }
    if (pick == 0U) {
        ++zero_picks_;
        if (row_cfg_.explore_every > 0U && zero_picks_ % row_cfg_.explore_every == 0U) pick = 1U;
    }
    return pick;
}

// Labels of the verified rows: row j was accepted iff it equals emitted[j]; the first rejected row ends the labels (the
// rows after it were conditioned on a wrong prefix).
void ParallelBlockDrafter::label(std::span<const std::uint32_t> emitted) {
    const double decay = std::exp2(-1.0 / row_cfg_.half_life);
    const std::size_t n = std::min(pending_.size(), emitted.size());
    for (std::size_t j = 0; j < n; ++j) {
        for (auto& b : bins_) {
            b[0] *= decay;
            b[1] *= decay;
        }
        const bool ok = pending_[j] == emitted[j];
        bins_[bin_of(pending_conf_[j])][ok ? 0U : 1U] += 1.0;
        if (!ok) break;
    }
}

void ParallelBlockDrafter::accept(std::span<const std::uint32_t> emitted) {
    if (!pending_.empty()) label(emitted);
    pending_.clear();
    pending_conf_.clear();
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

namespace {
void check_geometry(std::uint32_t experts, std::uint32_t top_k, std::uint32_t layers) {
    if (experts == 0U || top_k == 0U || top_k > experts || layers == 0U)
        throw std::invalid_argument("block_parallel: MoE verify cost needs 0 < top_k <= experts and layers > 0");
}
}  // namespace

MoeVerifyCost moe_verify_cost(const MoeVerifyBytes& m, double plain_ms, double gbps, std::uint32_t ranks, double trip_ms,
                              std::uint32_t trips, bool grouped) {
    check_geometry(m.experts, m.top_k, m.moe_layers);
    if (!(plain_ms > 0.0) || !(gbps > 0.0) || ranks == 0U || !(trip_ms >= 0.0))
        throw std::invalid_argument("block_parallel: moe_verify_cost needs plain_ms > 0, gbps > 0, ranks > 0, trip_ms >= 0");
    MoeVerifyCost c;
    const double ms_per_gb = 1000.0 / (gbps * double(ranks));
    c.per_expert_ms = m.expert_gb * ms_per_gb;
    c.per_row_ms = m.per_row_gb * ms_per_gb;
    c.experts = m.experts;
    c.top_k = m.top_k;
    c.layers = m.moe_layers;
    c.trip_ms = trip_ms;
    c.trips = trips;
    c.grouped = grouped;
    // R = 1 reads top_k experts per layer in both kernel forms.
    c.fixed_ms = plain_ms - double(trips) * trip_ms - c.per_row_ms - double(m.top_k) * double(m.moe_layers) * c.per_expert_ms;
    if (plain_ms - double(trips) * trip_ms - verify_gb(m, 1U) * ms_per_gb < 0.0)
        throw std::invalid_argument("block_parallel: verify bytes exceed the plain wall at gbps");
    return c;
}

double prior_reads(const MoeVerifyCost& c, std::uint32_t rows) {
    check_geometry(c.experts, c.top_k, c.layers);
    if (!c.grouped) return double(c.layers) * double(c.top_k) * double(rows);
    return double(c.layers) * uniform_distinct_experts(c.experts, c.top_k, rows);
}

double verify_ms(const MoeVerifyCost& c, std::uint32_t rows, double reads) {
    return c.fixed_ms + double(c.trips) * c.trip_ms + c.per_row_ms * double(rows) + c.per_expert_ms * reads;
}

MoeVerifyFit::MoeVerifyFit(const MoeVerifyCost& seed, MoeFitConfig config) : seed_(seed), config_(config) {
    check_geometry(seed.experts, seed.top_k, seed.layers);
    if (!(config.half_life > 0.0) || !(config.ridge > 0.0) || !(config.ridge_fixed > 0.0) || !(config.clip > 0.0) ||
        !(config.reads_half_life > 0.0))
        throw std::invalid_argument("block_parallel: MoeFitConfig half lives, ridges and clip must be > 0");
    unit_reads_ = double(seed.top_k) * double(seed.layers);
    theta0_ = {seed.fixed_ms + double(seed.trips) * seed.trip_ms, seed.per_row_ms, seed.per_expert_ms * unit_reads_};
    theta_ = theta0_;
}

double MoeVerifyFit::raw_ms(std::uint32_t rows, double r) const {
    return theta_[0] + theta_[1] * double(rows) + theta_[2] * (r / unit_reads_);
}

double MoeVerifyFit::reads(std::uint32_t rows) const {
    if (rows == 0U) return 0.0;
    const double prior = prior_reads(seed_, rows);
    const std::uint32_t at = std::min(rows, max_rows);
    for (std::uint32_t d = 0; d < max_rows; ++d) {  // nearest seen R, the smaller first
        if (at > d && ratio_w_[at - d] > 0.0) return prior * ratio_[at - d];
        if (at + d <= max_rows && ratio_w_[at + d] > 0.0) return prior * ratio_[at + d];
    }
    return prior;
}

void MoeVerifyFit::observe(std::uint32_t rows, double r, double ms) {
    if (rows == 0U || rows > max_rows || !(r > 0.0) || !(ms > 0.0)) return;
    const double rd = std::exp2(-1.0 / config_.reads_half_life);
    ratio_w_[rows] = ratio_w_[rows] * rd + 1.0;
    ratio_[rows] += (r / prior_reads(seed_, rows) - ratio_[rows]) / ratio_w_[rows];
    ++passes_;
    const double pred = raw_ms(rows, r), limit = config_.clip * pred;
    double y = ms;
    if (std::fabs(y - pred) > limit) {
        if (!last_skipped_) {  // an isolated stall is not folded; a repeat is a level change, followed clipped
            last_skipped_ = true;
            ++skipped_;
            return;
        }
        y = y > pred ? pred + limit : pred - limit;
    }
    last_skipped_ = false;
    const double d = std::exp2(-1.0 / config_.half_life);
    const std::array<double, 3> x{1.0, double(rows), r / unit_reads_};
    for (std::size_t i = 0; i < 3; ++i) {
        xy_[i] = xy_[i] * d + x[i] * y;
        for (std::size_t j = 0; j < 3; ++j) xx_[i * 3 + j] = xx_[i * 3 + j] * d + x[i] * x[j];
    }
    solve();
}

// (xx + ridge I) theta = xy + ridge theta0, Gaussian elimination with partial pivoting (3 x 3, deterministic).
void MoeVerifyFit::solve() {
    std::array<std::array<double, 4>, 3> a{};
    for (std::size_t i = 0; i < 3; ++i) {
        const double ridge = i == 0 ? config_.ridge_fixed : config_.ridge;
        for (std::size_t j = 0; j < 3; ++j) a[i][j] = xx_[i * 3 + j] + (i == j ? ridge : 0.0);
        a[i][3] = xy_[i] + ridge * theta0_[i];
    }
    for (std::size_t c = 0; c < 3; ++c) {
        std::size_t p = c;
        for (std::size_t i = c + 1; i < 3; ++i)
            if (std::fabs(a[i][c]) > std::fabs(a[p][c])) p = i;
        std::swap(a[c], a[p]);
        for (std::size_t i = c + 1; i < 3; ++i) {
            const double f = a[i][c] / a[c][c];
            for (std::size_t j = c; j < 4; ++j) a[i][j] -= f * a[c][j];
        }
    }
    for (std::size_t c = 3; c-- > 0;) {
        double v = a[c][3];
        for (std::size_t j = c + 1; j < 3; ++j) v -= a[c][j] * theta_[j];
        theta_[c] = v / a[c][c];
    }
}

double MoeVerifyFit::verify_ms(std::uint32_t rows) const {
    double ms = 0.0;
    for (std::uint32_t r = 1; r <= rows; ++r) ms = std::max(ms, raw_ms(r, reads(r)));
    return ms;
}

std::vector<double> MoeVerifyFit::table(std::uint32_t rows) const {
    std::vector<double> t(rows);
    double ms = 0.0;
    for (std::uint32_t r = 1; r <= rows; ++r) t[r - 1U] = ms = std::max(ms, raw_ms(r, reads(r)));
    return t;
}

MoeVerifyCost MoeVerifyFit::current() const {
    MoeVerifyCost c = seed_;
    c.fixed_ms = theta_[0] - double(seed_.trips) * seed_.trip_ms;
    c.per_row_ms = theta_[1];
    c.per_expert_ms = theta_[2] / unit_reads_;
    return c;
}

}  // namespace strix::spec
