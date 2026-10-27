// MiniMax DSpark drafter: host reference in the declared arithmetic "mm-dspark-v1" (see minimax_dspark.hpp).
#include "minimax_dspark.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace strix::minimax::dspark {
namespace {

// Lane combine of one wave: v[l] = v[l] + v[l ^ d], d = 16, 8, 4, 2, 1; every lane ends with the same value.
float butterfly(std::array<float, 32> v) {
    for (std::uint32_t d = 16; d != 0; d >>= 1U) {
        std::array<float, 32> t{};
        for (std::uint32_t l = 0; l < 32; ++l) t[l] = v[l] + v[l ^ d];
        v = t;
    }
    return v[0];
}

// Chunk-16 wave order over `width` inputs; weight(i) gives the FP32 weight of input i.
template <class WeightAt>
float chunk16_dot(WeightAt weight, const float* x, std::uint32_t width) {
    std::array<float, 32> lane{};
    const std::uint32_t chunks = width / 16U;
    for (std::uint32_t l = 0; l < 32; ++l) {
        float acc = 0.0F;
        for (std::uint32_t c = l; c < chunks; c += 32U)
            for (std::uint32_t e = 0; e < 16; ++e) {
                const std::uint32_t i = c * 16U + e;
                acc = acc + weight(i) * x[i];
            }
        lane[l] = acc;
    }
    return butterfly(lane);
}

// Lane l takes l, l + 32, l + 64, l + 96.
float dot128(const float* a, const float* b) {
    std::array<float, 32> lane{};
    for (std::uint32_t l = 0; l < 32; ++l) {
        float acc = 0.0F;
        for (std::uint32_t j = 0; j < 4; ++j) acc = acc + a[l + 32U * j] * b[l + 32U * j];
        lane[l] = acc;
    }
    return butterfly(lane);
}

// Split [0, n) over the hardware threads.
template <class Body>
void parallel_for(std::size_t n, Body body) {
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(std::thread::hardware_concurrency(), n / 64 + 1));
    if (workers == 1) {
        body(std::size_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    const std::size_t step = (n + workers - 1) / workers;
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t b = w * step, e = std::min(n, b + step);
        if (b < e) pool.emplace_back([=] { body(b, e); });
    }
    for (auto& t : pool) t.join();
}

float rstd_of(float sum_squares, std::uint32_t width) {
    return 1.0F / std::sqrt(sum_squares / static_cast<float>(width) + norm_eps);
}

} // namespace

void mxfp8_linear(const Mxfp8Matrix& w, const float* x, std::size_t rows, float* out) {
    parallel_for(w.outputs, [&](std::size_t o0, std::size_t o1) {
        std::vector<float> wrow(w.width);
        for (std::size_t o = o0; o < o1; ++o) {
            const std::uint8_t* codes = w.codes + o * w.width;
            const std::uint8_t* scales = w.scales + o * (w.width / 32U);
            for (std::uint32_t i = 0; i < w.width; ++i) wrow[i] = e4m3_value(codes[i]) * e8m0_value(scales[i / 32U]);
            for (std::size_t r = 0; r < rows; ++r)
                out[r * w.outputs + o] = chunk16_dot([&](std::uint32_t i) { return wrow[i]; }, x + r * w.width, w.width);
        }
    });
}

void bf16_linear(const Bf16Matrix& w, const float* x, std::size_t rows, float* out) {
    parallel_for(w.outputs, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o) {
            const std::uint16_t* row = w.bits + o * w.width;
            for (std::size_t r = 0; r < rows; ++r)
                out[r * w.outputs + o] = chunk16_dot([&](std::uint32_t i) { return bf16_value(row[i]); }, x + r * w.width, w.width);
        }
    });
}

void rms_norm(const float* x, const std::uint16_t* weight, std::size_t rows, float* out) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * hidden;
        float total = 0.0F;
        for (std::uint32_t wave = 0; wave < 8; ++wave) {
            std::array<float, 32> lane{};
            for (std::uint32_t l = 0; l < 32; ++l) {
                float acc = 0.0F;
                for (std::uint32_t i = wave * 32U + l; i < hidden; i += 256U) acc = acc + xr[i] * xr[i];
                lane[l] = acc;
            }
            total = total + butterfly(lane);
        }
        const float rstd = rstd_of(total, hidden);
        for (std::uint32_t i = 0; i < hidden; ++i) out[r * hidden + i] = (xr[i] * rstd) * (1.0F + bf16_value(weight[i]));
    }
}

void rope_table(std::uint64_t position, float* cos_out, float* sin_out) {
    for (std::uint32_t i = 0; i < rope_dim / 2U; ++i) {
        const double inv = 1.0 / std::pow(rope_theta, static_cast<double>(2U * i) / static_cast<double>(rope_dim));
        const double a = static_cast<double>(position) * inv;
        cos_out[i] = static_cast<float>(std::cos(a));
        sin_out[i] = static_cast<float>(std::sin(a));
    }
}

void head_norm_rope(float* x, std::uint32_t count, const std::uint16_t* weight, const float* cos_v, const float* sin_v) {
    for (std::uint32_t h = 0; h < count; ++h) {
        float* v = x + h * head_dim;
        const float rstd = rstd_of(dot128(v, v), head_dim);
        for (std::uint32_t d = 0; d < head_dim; ++d) v[d] = (v[d] * rstd) * (1.0F + bf16_value(weight[d]));
        for (std::uint32_t i = 0; i < rope_dim / 2U; ++i) {
            const float a = v[i], b = v[i + 32U];
            v[i] = a * cos_v[i] - b * sin_v[i];
            v[i + 32U] = b * cos_v[i] + a * sin_v[i];
        }
    }
}

Ring::Ring()
    : k(static_cast<std::size_t>(layers) * window * kv_width), v(static_cast<std::size_t>(layers) * window * kv_width),
      position(window, -1) {}

void append_context(const Weights& w, Ring& ring, const float* taps_rows, std::size_t n, std::uint64_t pos0) {
    std::vector<float> fc(n * hidden), c(n * hidden), k(n * kv_width), v(n * kv_width);
    bf16_linear(w.fc, taps_rows, n, fc.data());
    rms_norm(fc.data(), w.hidden_norm, n, c.data());
    for (std::uint32_t L = 0; L < layers; ++L) {
        const LayerWeights& y = w.layer[L];
        mxfp8_linear(y.k, c.data(), n, k.data());
        mxfp8_linear(y.v, c.data(), n, v.data());
        for (std::size_t r = 0; r < n; ++r) {
            const std::uint64_t p = pos0 + r;
            std::array<float, 32> cs{}, sn{};
            rope_table(p, cs.data(), sn.data());
            head_norm_rope(k.data() + r * kv_width, kv_heads, y.k_norm, cs.data(), sn.data());
            const std::size_t slot = p % window, base = (static_cast<std::size_t>(L) * window + slot) * kv_width;
            for (std::uint32_t i = 0; i < kv_width; ++i) {
                ring.k[base + i] = bf16_bits(k[r * kv_width + i]);
                ring.v[base + i] = bf16_bits(v[r * kv_width + i]);
            }
        }
    }
    for (std::size_t r = 0; r < n; ++r) ring.position[(pos0 + r) % window] = static_cast<std::int64_t>(pos0 + r);
}

void block_forward(const Weights& w, const Ring& ring, const float* embedded, std::uint64_t position, float* y,
                   std::uint32_t chunk) {
    const std::uint64_t first = position > context_keys ? position - context_keys : 0;
    const std::size_t ctx = static_cast<std::size_t>(position - first), keys = ctx + block;
    for (std::uint64_t p = first; p < position; ++p)
        if (ring.position[p % window] != static_cast<std::int64_t>(p))
            throw std::runtime_error("dspark block: context position " + std::to_string(p) + " missing from the ring");
    std::vector<float> r(embedded, embedded + block * hidden), x(block * hidden), q(block * q_width), kb(block * kv_width),
        vb(block * kv_width), att(block * q_width), o(block * hidden), gate(block * inter), up(block * inter);
    std::vector<std::array<float, 32>> cs(block), sn(block);
    for (std::uint32_t row = 0; row < block; ++row) rope_table(position + row, cs[row].data(), sn[row].data());
    for (std::uint32_t L = 0; L < layers; ++L) {
        const LayerWeights& lw = w.layer[L];
        rms_norm(r.data(), lw.input_norm, block, x.data());
        mxfp8_linear(lw.q, x.data(), block, q.data());
        mxfp8_linear(lw.k, x.data(), block, kb.data());
        mxfp8_linear(lw.v, x.data(), block, vb.data());
        for (std::uint32_t row = 0; row < block; ++row) {
            head_norm_rope(q.data() + row * q_width, heads, lw.q_norm, cs[row].data(), sn[row].data());
            head_norm_rope(kb.data() + row * kv_width, kv_heads, lw.k_norm, cs[row].data(), sn[row].data());
        }
        // Keys and values of the window as FP32 views of their BF16 storage: context ascending, then the block rows.
        std::vector<float> kk(keys * kv_width), vv(keys * kv_width);
        for (std::size_t j = 0; j < ctx; ++j) {
            const std::size_t base = (static_cast<std::size_t>(L) * window + (first + j) % window) * kv_width;
            for (std::uint32_t i = 0; i < kv_width; ++i) {
                kk[j * kv_width + i] = bf16_value(ring.k[base + i]);
                vv[j * kv_width + i] = bf16_value(ring.v[base + i]);
            }
        }
        for (std::uint32_t row = 0; row < block; ++row)
            for (std::uint32_t i = 0; i < kv_width; ++i) {
                kk[(ctx + row) * kv_width + i] = bf16_value(bf16_bits(kb[row * kv_width + i]));
                vv[(ctx + row) * kv_width + i] = bf16_value(bf16_bits(vb[row * kv_width + i]));
            }
        parallel_for(static_cast<std::size_t>(block) * heads, [&](std::size_t b, std::size_t e) {
            std::vector<float> s(keys);
            for (std::size_t rh = b; rh < e; ++rh) {
                const std::size_t row = rh / heads, h = rh % heads, g = h / group;
                const float* qv = q.data() + row * q_width + h * head_dim;
                float m = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < keys; ++j) {
                    s[j] = dot128(qv, kk.data() + j * kv_width + g * head_dim) * attention_scale;
                    m = s[j] > m ? s[j] : m;
                }
                for (std::size_t j = 0; j < keys; ++j) s[j] = declared_exp(s[j] - m);
                float sum = 0.0F;
                for (std::size_t c0 = 0; c0 < keys; c0 += chunk) {
                    float part = 0.0F;
                    for (std::size_t j = c0; j < std::min<std::size_t>(keys, c0 + chunk); ++j) part = part + s[j];
                    sum = sum + part;
                }
                float* out = att.data() + row * q_width + h * head_dim;
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0F;
                    for (std::size_t c0 = 0; c0 < keys; c0 += chunk) {
                        float part = 0.0F;
                        for (std::size_t j = c0; j < std::min<std::size_t>(keys, c0 + chunk); ++j)
                            part = part + s[j] * vv[j * kv_width + g * head_dim + d];
                        acc = acc + part;
                    }
                    out[d] = acc / sum;
                }
            }
        });
        mxfp8_linear(lw.o, att.data(), block, o.data());
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] + o[i];
        rms_norm(r.data(), lw.post_norm, block, x.data());
        mxfp8_linear(lw.gate, x.data(), block, gate.data());
        mxfp8_linear(lw.up, x.data(), block, up.data());
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = swiglu_oai(gate[i], up[i]);
        mxfp8_linear(lw.down, gate.data(), block, o.data());
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] + o[i];
    }
    rms_norm(r.data(), w.final_norm, block, y);
}

Candidate markov_step(const Weights& w, std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1) {
    std::array<float, markov_rank> m{};
    for (std::uint32_t i = 0; i < markov_rank; ++i) m[i] = bf16_value(w.markov_w1[static_cast<std::size_t>(prev) * markov_rank + i]);
    const std::size_t n = v1 - v0;
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(std::thread::hardware_concurrency(), 32));
    std::vector<Candidate> part(workers, Candidate{-std::numeric_limits<float>::infinity(), 0xffffffffU});
    const std::size_t step = (n + workers - 1) / workers;
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < workers; ++t)
        pool.emplace_back([&, t] {
            Candidate best = part[t];
            for (std::size_t i = t * step; i < std::min(n, (t + 1) * step); ++i) {
                const auto v = static_cast<std::uint32_t>(v0 + i);
                const std::uint16_t* row = w.markov_w2 + static_cast<std::size_t>(v) * markov_rank;
                std::array<float, 32> lane{};
                for (std::uint32_t l = 0; l < 32; ++l) {
                    float acc = 0.0F;
                    for (std::uint32_t e = 0; e < 8; ++e) acc = acc + m[8U * l + e] * bf16_value(row[8U * l + e]);
                    lane[l] = acc;
                }
                const float value = base[i] + butterfly(lane);
                if (better(value, v, best.value, best.id)) best = {value, v};
            }
            part[t] = best;
        });
    for (auto& t : pool) t.join();
    Candidate best = part[0];
    for (std::size_t t = 1; t < workers; ++t) best = combine(best, part[t]);
    return best;
}

float confidence(const Weights& w, const float* y_row, std::uint32_t prev) {
    std::array<float, confidence_width> x{};
    std::copy(y_row, y_row + hidden, x.begin());
    for (std::uint32_t i = 0; i < markov_rank; ++i)
        x[hidden + i] = bf16_value(w.markov_w1[static_cast<std::size_t>(prev) * markov_rank + i]);
    const float z = chunk16_dot([&](std::uint32_t i) { return bf16_value(w.confidence_w[i]); }, x.data(), confidence_width);
    return sigmoid(z + w.confidence_b);
}

Draft sample(const Weights& w, std::uint32_t anchor, const float* base, const float* y) {
    Draft d;
    std::uint32_t prev = anchor;
    for (std::uint32_t k = 0; k < block; ++k) {
        d.confidence[k] = confidence(w, y + static_cast<std::size_t>(k) * hidden, prev);
        prev = markov_step(w, prev, base + static_cast<std::size_t>(k) * vocab, 0, vocab).id;
        d.tokens[k] = prev;
    }
    return d;
}

Verdict verify_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> greedy) {
    if (greedy.size() != drafts.size() + 1) throw std::invalid_argument("verify_greedy: greedy needs k + 1 values");
    Verdict v;
    while (v.accepted < drafts.size() && drafts[v.accepted] == greedy[v.accepted]) ++v.accepted;
    v.emitted.assign(greedy.begin(), greedy.begin() + v.accepted + 1);
    return v;
}

std::uint32_t choose_window(std::span<const float> conf, std::span<const double> cost, std::uint32_t min_k) {
    if (cost.empty()) throw std::invalid_argument("choose_window: no costs");
    const auto max_k = static_cast<std::uint32_t>(std::min<std::size_t>(cost.size() - 1, conf.size()));
    std::uint32_t best = std::min(min_k, max_k);
    double expected = 1.0, survival = 1.0, best_rate = -1.0;
    for (std::uint32_t k = 0; k <= max_k; ++k) {
        if (k > 0) {
            survival *= static_cast<double>(conf[k - 1]);
            expected += survival;
        }
        const double rate = expected / cost[k];
        if (k >= min_k && rate > best_rate) {
            best_rate = rate;
            best = k;
        }
    }
    return best;
}

double SurvivalController::probability(std::uint32_t depth, float confidence) const {
    const std::uint32_t b = bin(confidence);
    double ok_b = 0, tries_b = 0;
    for (std::uint32_t i = 0; i < block; ++i) {
        ok_b += ok_[i][b];
        tries_b += tries_[i][b];
    }
    const double pooled = (ok_b + 1.2) / (tries_b + 2.0);
    const double n = static_cast<double>(tries_[depth][b]) + 4.0;
    const double m = (static_cast<double>(ok_[depth][b]) + 4.0 * pooled) / n;
    return std::min(1.0, m + 0.5 * std::sqrt(m * (1.0 - m) / n));
}

std::uint32_t SurvivalController::choose(std::span<const float> conf, std::span<const double> cost) {
    const auto max_k = static_cast<std::uint32_t>(std::min<std::size_t>({cost.size() - 1, conf.size(), std::size_t{block}}));
    if (calls_++ == 0) return max_k;
    const bool explore = explore_ != 0 && calls_ % explore_ == 0;
    std::uint32_t best = 0;
    double expected = 1.0, survival = 1.0, best_rate = expected / cost[0];
    for (std::uint32_t k = 1; k <= max_k; ++k) {
        survival *= probability(k - 1, conf[k - 1]);
        expected += survival;
        const double rate = expected / cost[k];
        if (rate > best_rate) {
            best_rate = rate;
            best = k;
        }
    }
    return explore ? std::min(max_k, best + 2) : best;
}

void SurvivalController::observe(std::span<const float> conf, std::uint32_t k, std::uint32_t accepted) {
    for (std::uint32_t i = 0; i < k && i < conf.size(); ++i) {
        const std::uint32_t b = bin(conf[i]);
        ++tries_[i][b];
        if (i >= accepted) break;  // the depths after the first rejection are not observed
        ++ok_[i][b];
    }
}

bool DraftGate::draft() {
    if (skip_ == 0) return true;
    --skip_;
    ++paused_;
    return false;
}

void DraftGate::observe(bool drafted, std::uint32_t emitted, double seconds, double plain_estimate) {
    if (!drafted) {
        t_plain_ = t_plain_ <= 0 ? seconds : 0.8 * t_plain_ + 0.2 * seconds;
        return;
    }
    const double t_plain = t_plain_ > 0 ? t_plain_ : plain_estimate;
    if (t_plain <= 0) return;
    const double x = static_cast<double>(emitted) - seconds / t_plain;
    if (agreed_) {
        n_ = 0.97 * n_ + 1;
        sx_ = 0.97 * sx_ + x;
        sxx_ = 0.97 * sxx_ + x * x;
        return;
    }
    n_ += 1;
    sx_ += x;
    sxx_ += x * x;
    if (n_ < min_) return;
    const double mean = sx_ / n_;
    const double half = z_ * std::sqrt(std::max(0.0, sxx_ / n_ - mean * mean) / n_);
    if (mean + half < 0) {
        skip_ = backoff_;
        backoff_ = std::min<std::uint32_t>(128, 2 * backoff_);
    } else if (mean - half > 0) {
        backoff_ = 8;
    }
}

std::uint32_t DraftGate::proposal() {
    if (n_ < min_) return 0;
    const double mean = sx_ / n_;
    const double half = z_ * std::sqrt(std::max(0.0, sxx_ / n_ - mean * mean) / n_);
    if (mean + half < 0) return backoff_;
    if (mean - half > 0) backoff_ = 8;
    return 0;
}

void DraftGate::pause(std::uint32_t cycles) {
    if (cycles == 0) return;
    skip_ = cycles;
    backoff_ = std::min<std::uint32_t>(128, 2 * cycles);
}

} // namespace strix::minimax::dspark
