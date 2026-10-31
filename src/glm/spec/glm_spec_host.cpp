// GLM speculative decoding host reference (see glm_spec_host.hpp). Build with -ffp-contract=off -fno-fast-math.
#include "glm/spec/glm_spec_host.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::glm::spec {

Accepted accept_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> target, unsigned first_flagged,
                       Terminal term) {
    const std::size_t valid = std::min<std::size_t>(drafts.size() + 1, first_flagged);
    if (target.size() < valid) throw std::invalid_argument("accept_greedy: target rows missing");
    if (term.budget == 0) throw std::invalid_argument("accept_greedy: no budget left (the request already ended)");
    Accepted r;
    unsigned emitted = 0;
    for (unsigned j = 0;; ++j) {  // row j is required: plain greedy processes it
        if (j >= first_flagged) {  // emits d1 .. dj (produced by valid rows), then fails at row j
            r.m = j;
            r.abort = true;
            r.rows = j;
            return r;
        }
        const std::uint32_t g = target[j];
        ++emitted;
        const bool terminal = g == term.eos || emitted >= term.budget;
        if (j < drafts.size() && drafts[j] == g) {  // d_{j+1} accepted and emitted
            if (terminal) {
                r.m = j + 1;
                r.stop = true;
                r.rows = j + 1;  // rows 0 .. j processed; row j+1 (input d_{j+1}) never is
                return r;
            }
            continue;
        }
        r.m = j;
        r.bonus = g;
        r.has_bonus = true;
        r.stop = terminal;
        r.rows = j + 1;
        return r;
    }
}

bool window_extend(const CostTable& t, float survival_k, unsigned k, unsigned kmax) {
    if (k >= kmax || k + 2 > 8) return false;
    return survival_k * t.plain_ms > (t.verify_ms[k + 2] - t.verify_ms[k + 1]) + t.step_ms;
}
bool window_gate_open(const CostTable& t, float conf1) { return conf1 * t.plain_ms > t.verify_ms[2] - t.verify_ms[1]; }

bool valid_cost_table(const CostTable& t) {
    if (!std::isfinite(t.plain_ms) || t.plain_ms <= 0 || !std::isfinite(t.step_ms) || t.step_ms <= 0 ||
        !std::isfinite(t.catchup_ms) || t.catchup_ms <= 0) return false;
    for (unsigned r = 1; r <= 8; ++r)
        if (!std::isfinite(t.verify_ms[r]) || t.verify_ms[r] <= 0 ||
            (r > 1 && t.verify_ms[r] < t.verify_ms[r - 1])) return false;
    return true;
}
CostTable seeded_cost_table() {
    // TP2 decode on the Xfer exchange (verify + MTP drafter), mean over 384 cycles: R1 43.2, R2 53.9, R3 64.0,
    // R4 73.7 ms (~10 ms per extra row), extension 2.4 ms, catch-up 3.1 ms, plain step 40.0 ms.
    CostTable t;
    t.version = 1;
    t.plain_ms = 40.0F;
    t.verify_ms[1] = 43.2F;
    t.verify_ms[2] = 53.9F;
    t.verify_ms[3] = 64.0F;
    t.verify_ms[4] = 73.7F;
    for (unsigned r = 5; r <= 8; ++r) t.verify_ms[r] = t.verify_ms[r - 1] + 10.0F;
    t.step_ms = 2.4F;
    t.catchup_ms = 3.1F;
    return t;
}
void store_cost_table(const CostTable& t, std::span<float, 11> wire) {
    wire[0] = t.plain_ms;
    for (unsigned r = 1; r <= 8; ++r) wire[r] = t.verify_ms[r];
    wire[9] = t.step_ms;
    wire[10] = t.catchup_ms;
}
bool load_cost_table(std::span<const float, 11> wire, std::uint32_t version, CostTable& out) {
    CostTable t;
    t.version = version;
    t.plain_ms = wire[0];
    for (unsigned r = 1; r <= 8; ++r) t.verify_ms[r] = wire[r];
    t.step_ms = wire[9];
    t.catchup_ms = wire[10];
    if (!valid_cost_table(t)) return false;
    out = t;
    return true;
}
void observe_cost(CostTable& t, unsigned index, double ms) {
    if (index > 10 || !std::isfinite(ms) || ms <= 0 || ms > std::numeric_limits<float>::max()) return;
    float* cell = index == 0 ? &t.plain_ms : index <= 8 ? &t.verify_ms[index] : index == 9 ? &t.step_ms : &t.catchup_ms;
    *cell += (static_cast<float>(ms) - *cell) * 0.125F;
    if (index >= 1 && index <= 8) {
        ++t.verify_samples[index];
        // Least-squares slope over the sampled rows (seed slope until two are sampled), clamped >= 0; each unsampled
        // row = nearest sampled row + slope * distance.
        unsigned n = 0;
        double sr = 0, sv = 0, srr = 0, srv = 0;
        for (unsigned r = 1; r <= 8; ++r)
            if (t.verify_samples[r] != 0) {
                ++n;
                sr += r; sv += t.verify_ms[r]; srr += double(r) * r; srv += double(r) * t.verify_ms[r];
            }
        const auto seed = seeded_cost_table();
        float slope = (seed.verify_ms[4] - seed.verify_ms[1]) / 3.0F;
        if (n >= 2) slope = static_cast<float>(std::max(0.0, (n * srv - sr * sv) / (n * srr - sr * sr)));
        for (unsigned r = 1; r <= 8; ++r) {
            if (t.verify_samples[r] != 0) continue;
            unsigned best = 0;
            for (unsigned q = 1; q <= 8; ++q)
                if (t.verify_samples[q] != 0 && (best == 0 || (q > r ? q - r : r - q) < (best > r ? best - r : r - best)))
                    best = q;
            t.verify_ms[r] = std::max(0.01F, t.verify_ms[best] + slope * (static_cast<float>(r) - static_cast<float>(best)));
        }
    }
    for (unsigned r = 2; r <= 8; ++r) t.verify_ms[r] = std::max(t.verify_ms[r], t.verify_ms[r - 1]);
    ++t.version;
}
bool window_extend(const CostTable& t, const WindowEstimate& w, float next, unsigned kmax) {
    if (!valid_cost_table(t) || w.k == 0 || w.k >= std::min(kmax, 7U) ||
        !std::isfinite(next) || !std::isfinite(w.survival) || !std::isfinite(w.expected) ||
        !std::isfinite(w.cycle_ms) || w.expected <= 0 || w.cycle_ms <= 0) return false;
    return std::clamp(w.survival, 0.0F, 1.0F) * std::clamp(next, 0.0F, 1.0F) * (w.cycle_ms / w.expected) >
           1.03F * (t.verify_ms[w.k + 2] - t.verify_ms[w.k + 1] + t.step_ms);
}
unsigned draft_limit(unsigned kmax, unsigned remaining, std::uint64_t p, std::uint64_t capacity) {
    if (remaining == 0 || p >= capacity) return 0;
    return static_cast<unsigned>(std::min<std::uint64_t>(std::min({kmax, 7U, remaining - 1}), capacity - p - 1));
}
unsigned Calibration::bin(float c) {
    if (!std::isfinite(c)) return 0;
    return std::min(7U, static_cast<unsigned>(std::clamp(c, 0.0F, 1.0F) * 8));
}
float Calibration::probability(unsigned depth, float c) const {
    const unsigned d = std::clamp(depth, 1U, 7U), b = bin(c);
    const float prior = d == 1 ? 0.90F : d == 2 ? 0.80F : 0.70F;
    // Weight-2 virtual prior: a 64-token request reaches ~20 cycles, so its own acceptance must decide within a few
    // observations (weight 8 held the depth-3 prior 0.70 at the K2/K3 break-even and kept chat at K3).
    return (2 * prior + static_cast<float>(successes_[d][b])) / (2 + static_cast<float>(trials_[d][b]));
}
void Calibration::observe(unsigned depth, float c, bool accepted) {
    const unsigned d = std::clamp(depth, 1U, 7U), b = bin(c);
    ++trials_[d][b];
    successes_[d][b] += accepted ? 1U : 0U;
}

// ---------------------------------------------------------------- KDA deferred replay
unsigned KdaVerify::run(std::span<const KdaRow> rows, const KdaWeights& w, std::vector<std::vector<float>>& y) {
    const unsigned C = st_.heads * kKdaDim;
    std::vector<float> discard(C);
    for (const KdaRow& r : pending_) host::kda_step(st_, r.in(), w, discard);  // committed state catches up
    replays_ += static_cast<unsigned>(pending_.size());
    pending_.clear();
    KdaState work = st_;  // the device keeps this in registers; only st_ is stored
    y.assign(rows.size(), std::vector<float>(C));
    unsigned first = static_cast<unsigned>(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        try {
            host::kda_step(work, rows[r].in(), w, y[r]);
        } catch (const host::NonFinite&) {
            first = static_cast<unsigned>(r);
            break;
        }
    }
    y.resize(first);
    cand_.assign(rows.begin(), rows.begin() + first);  // a flagged row is never committed without an abort
    return first;
}

void KdaVerify::commit(unsigned n_rows) {
    if (n_rows > cand_.size()) throw std::logic_error("KdaVerify::commit: more rows than the last run");
    pending_.assign(cand_.begin(), cand_.begin() + n_rows);
    cand_.clear();
}

void KdaVerify::flush(const KdaWeights& w) {
    std::vector<float> discard(st_.heads * kKdaDim);
    for (const KdaRow& r : pending_) host::kda_step(st_, r.in(), w, discard);
    replays_ += static_cast<unsigned>(pending_.size());
    pending_.clear();
}

// ---------------------------------------------------------------- DSA rows of one slot
unsigned dsa_append_rows(DsaCache& c, std::span<const float> latents, std::span<const float> x, unsigned in,
                         const IndexerWeights& w, unsigned R) {
    if (latents.size() < static_cast<std::size_t>(R) * kKvLora || x.size() < static_cast<std::size_t>(R) * in)
        throw std::invalid_argument("dsa_append_rows geometry");
    const unsigned t0 = c.tokens();
    for (unsigned r = 0; r < R; ++r) {
        try {
            host::dsa_append(c, latents.subspan(static_cast<std::size_t>(r) * kKvLora, kKvLora),
                             x.subspan(static_cast<std::size_t>(r) * in, in), w);
        } catch (const host::NonFinite&) {
            c.truncate(t0 + r);  // drop the partly appended row
            return r;
        }
    }
    return R;
}

// ---------------------------------------------------------------- MTP draft layer
std::vector<MtpOut> mtp_forward(DsaCache& c, std::span<const float> h, std::span<const std::uint32_t> x,
                                const MtpWeights& w, MtpShare* share, bool draft_step) {
    const unsigned R = static_cast<unsigned>(x.size()), D = kHidden, Hq = w.heads * kMlaDim;
    if (h.size() != static_cast<std::size_t>(R) * D) throw std::invalid_argument("mtp_forward: h rows");
    auto reduce = [&](std::span<float> v) {
        if (w.reduce) w.reduce(v);
    };
    const unsigned t0 = c.tokens();
    std::vector<float> z(static_cast<std::size_t>(R) * D), xin(static_cast<std::size_t>(R) * D);
    std::vector<float> qres(static_cast<std::size_t>(R) * kQLora), lat(static_cast<std::size_t>(R) * kKvLora);
    std::vector<float> cat(2 * D), e(D), tmp(kQLora);
    for (unsigned r = 0; r < R; ++r) {  // z = eh_proj([enorm(e) | hnorm(h)]), then the attention inputs
        w.embed(x[r], e);
        host::rmsnorm(e, &w.enorm, kEps, std::span<float>(cat).first(D));
        host::rmsnorm(h.subspan(static_cast<std::size_t>(r) * D, D), &w.hnorm, kEps, std::span<float>(cat).last(D));
        std::span<float> zr(&z[static_cast<std::size_t>(r) * D], D), xr(&xin[static_cast<std::size_t>(r) * D], D);
        w.eh_proj(cat, zr);
        host::rmsnorm(zr, &w.input_ln, kEps, xr);
        w.q_a(xr, tmp);
        host::rmsnorm(tmp, &w.q_a_norm, kEps, std::span<float>(&qres[static_cast<std::size_t>(r) * kQLora], kQLora));
        float kv[kKvLora];
        w.kv_a(xr, kv);
        host::rmsnorm(kv, &w.kv_a_norm, kEps, std::span<float>(&lat[static_cast<std::size_t>(r) * kKvLora], kKvLora));
    }
    if (dsa_append_rows(c, lat, xin, D, w.idx, R) != R) throw host::NonFinite("C3/C4 mtp cache");
    // all rows stored before any row attends (row j sees <= t0 + j)
    const bool sharing = share && share->on;
    if (sharing && draft_step && (R != 1 || !share->valid)) throw std::logic_error("mtp_forward: draft step without a catch-up");
    std::vector<MtpOut> out(R);
    std::vector<float> q(Hq), qI(kIdxHeads * kIdxDim), o(Hq), a(D), y(D), yn(D), sh(D);
    std::vector<float> eo(static_cast<std::size_t>(kTopE) * D);
    for (unsigned r = 0; r < R; ++r) {
        std::span<const float> qr(&qres[static_cast<std::size_t>(r) * kQLora], kQLora), xr(&xin[static_cast<std::size_t>(r) * D], D);
        w.q_b(qr, q);
        w.idx_wq_b(qr, qI);
        std::vector<std::uint32_t> tokens;
        const unsigned t = t0 + r;
#if defined(GLMSPEC_HOST_MUTANT) && GLMSPEC_HOST_MUTANT == 1
        if (false) {  // negative control: extensions select fresh (must fail the sharing reference)
#else
        if (sharing && draft_step) {  // reuse the catch-up pools, then every position from 4 * P0 to t
#endif
            tokens = share->pools;
            for (unsigned pos = share->P0 * kKpool; pos <= t; ++pos) tokens.push_back(pos);
        } else {
            tokens = host::indexer_select(c, t, qI, xr, w.idx);
            if (sharing && r + 1 == R) {
                share->P0 = (t + 1) / kKpool;
                share->pools.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(tokens.size() - (t + 1 - share->P0 * kKpool)));
                share->valid = true;
            }
        }
        host::mla_attention(c, tokens, q, w.kv_b, w.heads, o);
        w.o_proj(o, a);
        reduce(a);
        for (unsigned d = 0; d < D; ++d) y[d] = z[static_cast<std::size_t>(r) * D + d] + a[d];  // r = z + attn
        host::rmsnorm(y, &w.post_ln, kEps, yn);
        const host::Route rt = host::route(yn, w.router, w.router_bias);
        for (unsigned k = 0; k < kTopE; ++k) w.expert(rt.id[k], yn, std::span<float>(&eo[static_cast<std::size_t>(k) * D], D));
        w.shared(yn, sh);
        host::moe_combine(rt, eo, sh, a);
        reduce(a);
        for (unsigned d = 0; d < D; ++d) y[d] = y[d] + a[d];  // y = r + moe
        host::check_finite(y, "C1 mtp residual");
        out[r].h.resize(D);
        host::rmsnorm(y, &w.head_norm, kEps, out[r].h);
        std::vector<float> logits(w.vocab_n);
        w.head(out[r].h, logits);
        out[r].top = host::argmax_local(logits, w.vocab0);
    }
    return out;
}

}  // namespace strix::glm::spec
