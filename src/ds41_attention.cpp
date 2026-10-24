// Host references of the deepseek41 decode attention kernels. Every reduction is written lane by lane in the
// device's order (32 lanes, lane l owns dims 16l..16l+15 of a 512 vector, xor butterflies 16..1), so the results
// are the device's bits; see docs/models/deepseek-v4.1/attention.md for the declared order.
#include "strix/ds41_attention.hpp"

#include "strix/deepseek41_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace strix::ds41::attn {
namespace {

using Lanes = std::array<float, 32>;
float butterfly_sum(Lanes v) {
    for (unsigned d = 16; d != 0; d /= 2) {
        Lanes n{};
        for (unsigned l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ d];
        v = n;
    }
    return v[0];
}
// RMS norm of a 512 vector: lane chains of x*x over 16 dims, butterfly, mean, 1/sqrt, (x * r) * w.
void rms_norm_512(const float* x, const float* w, float* y) {
    Lanes ss{};
    for (unsigned l = 0; l < 32; ++l) {
        float acc = 0.0F;
        for (unsigned i = 0; i < 16; ++i) acc = std::fma(x[16 * l + i], x[16 * l + i], acc);
        ss[l] = acc;
    }
    const float r = 1.0F / std::sqrt(butterfly_sum(ss) / 512.0F + kNormEps);
    for (unsigned d = 0; d < kHeadDim; ++d) y[d] = (x[d] * r) * w[d];
}
void rope_tail(float* v, unsigned dim, const RopeRow& rope, bool inverse) {
    for (unsigned i = 0; i < kRopePairs; ++i) rope_pair(v[dim - kRopeDim + 2 * i], v[dim - kRopeDim + 2 * i + 1], rope.c[i], rope.s[i], inverse);
}
void quant_fp4_e4m3_16(const float* x, std::uint8_t* entry) {  // 512 values -> 288 B
    for (unsigned g = 0; g < 32; ++g) {
        float amax = 0.0F;
        for (unsigned i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(x[16 * g + i]));
        amax = std::fmax(amax, 0.01171875F);
        const std::uint8_t sc = e4m3_encode(amax / 6.0F);
        const float s = e4m3_value(sc);
        for (unsigned i = 0; i < 16; i += 2) {
            const std::uint8_t lo = e2m1_encode(clampf(x[16 * g + i] / s, -6.0F, 6.0F));
            const std::uint8_t hi = e2m1_encode(clampf(x[16 * g + i + 1] / s, -6.0F, 6.0F));
            entry[8 * g + i / 2] = static_cast<std::uint8_t>(lo | (hi << 4U));
        }
        entry[256 + g] = sc;
    }
}
// FP4 with power-of-two (E8M0) scales per 32: codes and exponents (s = 2^e).
void quant_fp4_e8m0_32(const float* x, unsigned n, std::uint8_t* codes, int* exps) {
    for (unsigned g = 0; g < n / 32; ++g) {
        float amax = 0.0F;
        for (unsigned i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[32 * g + i]));
        amax = std::fmax(amax, 7.05296657e-38F);  // 6 * 2^-126
        const int e = log2_ceil(amax * (1.0F / 6.0F));
        const float s = pow2f(e);
        for (unsigned i = 0; i < 32; ++i) codes[32 * g + i] = e2m1_encode(clampf(x[32 * g + i] / s, -6.0F, 6.0F));
        exps[g] = e;
    }
}
void check_rows(std::size_t rows) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("ds41 attention: 1..8 rows");
}

}  // namespace

RopeRow rope_row(RopeKind kind, std::uint64_t position) {
    // rope_head_dim 64: frequencies 1 / base^(2i/64); compressed layers: base 160000 with YaRN (factor 16, original
    // 65536, beta_fast 32, beta_slow 1); window-only layers: base 10000 without YaRN. Double, rounded once.
    const double dim = kRopeDim, base = kind == RopeKind::window ? 10000.0 : 160000.0;
    RopeRow r{};
    double low = 0, high = 0;
    if (kind == RopeKind::compressed) {
        const auto corrected = [&](double rotations) {
            return dim * std::log(65536.0 / (rotations * 2.0 * std::numbers::pi)) / (2.0 * std::log(base));
        };
        low = std::max(std::floor(corrected(32.0)), 0.0);
        high = std::min(std::ceil(corrected(1.0)), dim - 1.0);
    }
    for (unsigned i = 0; i < kRopePairs; ++i) {
        double f = 1.0 / std::pow(base, (2.0 * i) / dim);
        if (kind == RopeKind::compressed) {
            const double ramp = std::clamp((i - low) / std::max(high - low, 1e-3), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            f = f / 16.0 * (1.0 - smooth) + f * smooth;
        }
        const double angle = static_cast<double>(position) * f;
        r.c[i] = static_cast<float>(std::cos(angle));
        r.s[i] = static_cast<float>(std::sin(angle));
    }
    return r;
}

std::vector<LayerAttention> plan_from_contract(const std::vector<deepseek41::LayerAttributes>& layers) {
    std::vector<LayerAttention> p(layers.size());
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& a = layers[l];
        auto& o = p[l];
        if (a.compress_ratio > 2U) throw std::invalid_argument("ds41 attention: compress ratio " + std::to_string(a.compress_ratio) + " at layer " + std::to_string(l));
        o.ratio = a.compress_ratio;
        o.rope = o.ratio == 0U ? RopeKind::window : RopeKind::compressed;
        if (o.ratio == 0U) continue;
        o.kv_source = a.kv_source_layer;
        o.index_source = a.index_source_layer;
        o.owns_kv = a.kv_source;
        o.owns_index = a.index_source;
        o.candidate_source = a.candidate_source;
        o.candidate_user = a.uses_candidates;
    }
    return p;
}
void check_contract(const deepseek41::Config& c) {
    const auto need = [](bool ok, const char* what) { if (!ok) throw std::invalid_argument(std::string("ds41 attention: unsupported ") + what); };
    need(c.head_dim == kHeadDim && c.qk_rope_head_dim == kRopeDim && c.num_attention_heads == kHeads && c.num_key_value_heads == 1U, "attention shape");
    need(c.sliding_window == kWindow, "sliding window");
    need(c.index_n_heads == kIdxHeads && c.index_head_dim == kIdxDim && c.index_topk == kTopK, "indexer shape");
    need(c.candidate_topk_blocks == kCandBlocks && c.candidate_block_size == kCandBlockSize, "candidate pool");
    need(c.rope_theta == 10000.0 && c.compress_rope_theta == 160000.0, "RoPE bases");
    need(c.rope_scaling.type == "yarn" && c.rope_scaling.factor == 16.0 && c.rope_scaling.original_max_position_embeddings == 65536.0 &&
             c.rope_scaling.beta_fast == 32.0 && c.rope_scaling.beta_slow == 1.0, "RoPE scaling");
    need(static_cast<float>(c.rms_norm_eps) == kNormEps, "norm epsilon");
}
std::array<LayerAttention, 40> released_plan() {
    std::array<LayerAttention, 40> p{};
    const int kv_sources[] = {2, 8, 14, 20}, index_sources[] = {2, 8, 14, 20, 24, 28, 32, 36};
    int kv = -1, ix = -1;
    for (int layer = 0; layer < 40; ++layer) {
        auto& a = p[static_cast<std::size_t>(layer)];
        a.ratio = layer < 2 ? 0U : layer < 20 ? 2U : 1U;
        a.rope = a.ratio == 0U ? RopeKind::window : RopeKind::compressed;
        if (std::find(std::begin(kv_sources), std::end(kv_sources), layer) != std::end(kv_sources)) kv = layer;
        if (std::find(std::begin(index_sources), std::end(index_sources), layer) != std::end(index_sources)) ix = layer;
        if (a.ratio == 0U) continue;
        a.kv_source = kv;
        a.index_source = ix;
        a.owns_kv = kv == layer;
        a.owns_index = ix == layer;
        a.candidate_source = layer == 20;
        a.candidate_user = a.owns_index && layer > 20;
    }
    return p;
}

std::array<float, kHeadDim> decode_window_entry(const std::uint8_t* e) {
    std::array<float, kHeadDim> v{};
    for (unsigned d = 0; d < kHeadDim; ++d) v[d] = e4m3_value(e[d]) * e8m0_value(e[512 + d / 32]);
    return v;
}
std::array<float, kHeadDim> decode_kv_entry(const std::uint8_t* e) {
    std::array<float, kHeadDim> v{};
    for (unsigned d = 0; d < kHeadDim; ++d)
        v[d] = e2m1_value(static_cast<std::uint8_t>((e[d / 2] >> (4U * (d & 1U))) & 15U)) * e4m3_value(e[256 + d / 16]);
    return v;
}
std::array<float, kIdxDim> decode_idx_entry(const std::uint8_t* e) {
    std::array<float, kIdxDim> v{};
    for (unsigned d = 0; d < kIdxDim; ++d)
        v[d] = e2m1_value(static_cast<std::uint8_t>((e[d / 2] >> (4U * (d & 1U))) & 15U)) * e8m0_value(e[64 + d / 32]);
    return v;
}

void window_write(const WindowArgs& a, std::span<std::uint8_t> ring) {
    const std::size_t rows = a.pos.size();
    check_rows(rows);
    if (a.capacity < kWindow + rows || ring.size() < std::size_t{a.capacity} * kWindowEntryBytes)
        throw std::invalid_argument("ds41 window ring capacity");
    for (std::size_t r = 0; r < rows; ++r) {
        float y[kHeadDim];
        rms_norm_512(&a.kv[r * kHeadDim], a.norm.data(), y);
        rope_tail(y, kHeadDim, rope_row(a.rope, a.pos[r]), false);
        std::uint8_t* e = &ring[(a.pos[r] % a.capacity) * kWindowEntryBytes];
        for (unsigned b = 0; b < 16; ++b) {
            float amax = 0.0F;
            for (unsigned i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(y[32 * b + i]));
            amax = std::fmax(amax, 1e-4F);
            const int ex = log2_ceil(amax * (1.0F / 448.0F));
            const float s = pow2f(ex);
            for (unsigned i = 0; i < 32; ++i) e[32 * b + i] = e4m3_encode(clampf(y[32 * b + i] / s, -448.0F, 448.0F));
            e[512 + b] = static_cast<std::uint8_t>(ex + 127);
        }
    }
}

void compress(const CompressArgs& a, CompressState& state, std::span<std::uint8_t> kv_cache, std::span<std::uint8_t> idx_cache) {
    const std::size_t rows = a.pos.size();
    check_rows(rows);
    if (a.ratio != 1 && a.ratio != 2) throw std::invalid_argument("ds41 compress ratio");
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint64_t p = a.pos[r];
        const float* kv = &a.kv[r * kHeadDim];
        float pooled[kHeadDim];
        if (a.ratio == 2) {
            const float* sc = &a.score[r * kHeadDim];
            if (p % 2 == 0) {
                std::copy(kv, kv + kHeadDim, state.kv.begin());
                std::copy(sc, sc + kHeadDim, state.score.begin());
                state.valid = true;
                continue;
            }
            if (!state.valid) throw std::invalid_argument("ds41 compress: odd position without its group's first token");
            for (unsigned d = 0; d < kHeadDim; ++d) {
                const float m = std::fmax(state.score[d], sc[d]);
                const float e0 = exp_f(state.score[d] - m), e1 = exp_f(sc[d] - m);
                const float sum = e0 + e1;
                pooled[d] = state.kv[d] * (e0 / sum) + kv[d] * (e1 / sum);
            }
            state.valid = false;
        } else {
            std::copy(kv, kv + kHeadDim, pooled);
        }
        float latent[kHeadDim];
        rms_norm_512(pooled, a.norm.data(), latent);
        const std::uint64_t slot = p / a.ratio, rope_pos = slot * a.ratio;
        const RopeRow rope = rope_row(RopeKind::compressed, rope_pos);
        // index key: wk (F16) x latent, lane l owns outputs 4l..4l+3, each a 512-term chain
        float k[kIdxDim];
        for (unsigned o = 0; o < kIdxDim; ++o) {
            float acc = 0.0F;
            for (unsigned d = 0; d < kHeadDim; ++d) acc = std::fma(f16_to_float(a.wk[o * kHeadDim + d]), latent[d], acc);
            k[o] = acc;
        }
        Lanes ss{};
        for (unsigned l = 0; l < 32; ++l) {
            float acc = 0.0F;
            for (unsigned i = 0; i < 4; ++i) acc = std::fma(k[4 * l + i], k[4 * l + i], acc);
            ss[l] = acc;
        }
        const float rk = 1.0F / std::sqrt(butterfly_sum(ss) / 128.0F + kNormEps);
        for (unsigned o = 0; o < kIdxDim; ++o) k[o] = (k[o] * rk) * a.knorm[o];
        rope_tail(k, kIdxDim, rope, false);
        std::uint8_t codes[kIdxDim];
        int exps[4];
        quant_fp4_e8m0_32(k, kIdxDim, codes, exps);
        std::uint8_t* ie = &idx_cache[slot * kIdxEntryBytes];
        for (unsigned i = 0; i < 64; ++i) ie[i] = static_cast<std::uint8_t>(codes[2 * i] | (codes[2 * i + 1] << 4U));
        for (unsigned g = 0; g < 4; ++g) ie[64 + g] = static_cast<std::uint8_t>(exps[g] + 127);
        rope_tail(latent, kHeadDim, rope, false);
        quant_fp4_e4m3_16(latent, &kv_cache[slot * kKvEntryBytes]);
    }
}

void index_scores(const IndexArgs& a, std::span<const std::uint8_t> idx_cache, std::vector<std::vector<float>>& scores) {
    const std::size_t rows = a.pos.size();
    check_rows(rows);
    scores.assign(rows, {});
    for (std::size_t r = 0; r < rows; ++r) {
        const RopeRow rope = rope_row(a.rope, a.pos[r]);
        std::uint8_t qc[kIdxHeads][kIdxDim];
        int qe[kIdxHeads][4];
        float w[kIdxHeads];
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            float q[kIdxDim];
            std::copy_n(&a.q[(r * kIdxHeads + h) * kIdxDim], kIdxDim, q);
            rope_tail(q, kIdxDim, rope, false);
            quant_fp4_e8m0_32(q, kIdxDim, qc[h], qe[h]);
            w[h] = a.w[r * kIdxHeads + h] * kIdxWeightScale;
        }
        const std::uint64_t count = (a.pos[r] + 1) / a.ratio;
        auto& out = scores[r];
        out.resize(count);
        for (std::uint64_t t = 0; t < count; ++t) {
            const std::uint8_t* e = &idx_cache[t * kIdxEntryBytes];
            float score = 0.0F;
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                float term[4];
                for (unsigned b = 0; b < 4; ++b) {
                    int dot = 0;
                    for (unsigned i = 0; i < 32; ++i) {
                        const unsigned dim = 32 * b + i;
                        dot += e2m1_int2(qc[h][dim]) * e2m1_int2(static_cast<std::uint8_t>((e[dim / 2] >> (4U * (dim & 1U))) & 15U));
                    }
                    term[b] = ((static_cast<float>(dot) * pow2f(qe[h][b])) * e8m0_value(e[64 + b])) * 0.25F;
                }
                const float d = ((term[0] + term[1]) + term[2]) + term[3];
                score = std::fma(std::fmax(d, 0.0F), w[h], score);
            }
            out[t] = score;
        }
    }
}

namespace {
// The k best (key descending, index ascending), returned in ascending index order.
std::vector<std::uint32_t> best_k(std::span<const std::uint32_t> keys, unsigned k) {
    std::vector<std::uint32_t> order(keys.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    const std::size_t take = std::min<std::size_t>(k, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(),
        [&](std::uint32_t x, std::uint32_t y) { return keys[x] != keys[y] ? keys[x] > keys[y] : x < y; });
    order.resize(take);
    std::sort(order.begin(), order.end());
    return order;
}
}  // namespace

std::vector<std::uint32_t> select_topk(std::span<const float> scores, Pool pool, std::span<std::uint8_t> candidates,
    unsigned topk, unsigned cand_blocks, unsigned block_size) {
    const std::size_t n = scores.size();
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<std::uint32_t> keys(n);
    if (pool == Pool::candidate_source && n != 0) {
        const std::size_t nb = (n + block_size - 1) / block_size, last = (n - 1) / block_size;
        std::vector<std::uint32_t> bk(nb);
        std::vector<float> bs(nb, ninf);
        for (std::size_t t = 0; t < n; ++t) bs[t / block_size] = std::fmax(bs[t / block_size], scores[t]);
        bs[last] = std::numeric_limits<float>::infinity();
        for (std::size_t b = 0; b < nb; ++b) bk[b] = order_key(bs[b]);
        std::fill(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(nb), std::uint8_t{0});
        for (const auto b : best_k(bk, cand_blocks)) if (bs[b] > ninf) candidates[b] = 1;
    }
    for (std::size_t t = 0; t < n; ++t)
        keys[t] = order_key(pool == Pool::candidate_user && candidates[t / block_size] == 0 ? ninf : scores[t]);
    return best_k(keys, topk);
}

void attention(const AttentionArgs& a, std::span<const std::uint8_t> ring, std::span<const std::uint8_t> kv_cache, std::span<float> o) {
    const std::size_t rows = a.pos.size();
    check_rows(rows);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint64_t p = a.pos[r];
        const RopeRow rope = rope_row(a.rope, p);
        const unsigned nw = static_cast<unsigned>(std::min<std::uint64_t>(p + 1, kWindow));
        const unsigned nc = a.idx_count.empty() ? 0U : a.idx_count[r];
        const unsigned n = nw + nc, chunks = (n + kChunk - 1) / kChunk;
        const auto key = [&](unsigned j) {
            if (j < nw) return decode_window_entry(&ring[((p + 1 - nw + j) % a.capacity) * kWindowEntryBytes]);
            return decode_kv_entry(&kv_cache[std::size_t{a.idx[r * a.topk + (j - nw)]} * kKvEntryBytes]);
        };
        std::vector<std::array<float, kHeadDim>> keys(n);
        for (unsigned j = 0; j < n; ++j) keys[j] = key(j);
        for (unsigned h = 0; h < kHeads; ++h) {
            float q[kHeadDim];
            std::copy_n(&a.q[(r * kHeads + h) * kHeadDim], kHeadDim, q);
            rope_tail(q, kHeadDim, rope, false);
            float m[kMaxChunks], l[kMaxChunks];
            std::vector<std::array<float, kHeadDim>> acc(chunks);
            for (unsigned c = 0; c < chunks; ++c) {
                // tiles of 16 keys, online softmax: s_j, tile max, rescale by exp(m - m_new), then p_j in key order
                float mc = ninf, lc = 0.0F;
                acc[c].fill(0.0F);
                for (unsigned t0 = c * kChunk; t0 < std::min(n, (c + 1) * kChunk); t0 += kTile) {
                    const unsigned cnt = std::min(kTile, n - t0);
                    float s[kTile], tm = ninf;
                    for (unsigned j = 0; j < cnt; ++j) {
                        Lanes part{};
                        for (unsigned lane = 0; lane < 32; ++lane) {
                            float t = 0.0F;
                            for (unsigned i = 0; i < 16; ++i) t = std::fma(q[16 * lane + i], keys[t0 + j][16 * lane + i], t);
                            part[lane] = t;
                        }
                        s[j] = butterfly_sum(part) * kAttnScale;
                        tm = std::fmax(tm, s[j]);
                    }
                    const float mn = std::fmax(mc, tm), sc = exp_f(mc - mn);
                    lc = lc * sc;
                    float pr[kTile];
                    for (unsigned j = 0; j < cnt; ++j) { pr[j] = exp_f(s[j] - mn); lc = lc + pr[j]; }
                    for (unsigned d = 0; d < kHeadDim; ++d) {
                        float t = acc[c][d] * sc;
                        for (unsigned j = 0; j < cnt; ++j) t = std::fma(pr[j], keys[t0 + j][d], t);
                        acc[c][d] = t;
                    }
                    mc = mn;
                }
                m[c] = mc; l[c] = lc;
            }
            float mm = ninf;
            for (unsigned c = 0; c < chunks; ++c) mm = std::fmax(mm, m[c]);
            float e[kMaxChunks], lsum = 0.0F;
            for (unsigned c = 0; c < chunks; ++c) { e[c] = exp_f(m[c] - mm); lsum = std::fma(l[c], e[c], lsum); }
            lsum = lsum + exp_f(a.sink[h] - mm);
            float* out = &o[(r * kHeads + h) * kHeadDim];
            for (unsigned d = 0; d < kHeadDim; ++d) {
                float t = 0.0F;
                for (unsigned c = 0; c < chunks; ++c) t = std::fma(acc[c][d], e[c], t);
                out[d] = t / lsum;
            }
            rope_tail(out, kHeadDim, rope, true);
        }
    }
}

}  // namespace strix::ds41::attn
