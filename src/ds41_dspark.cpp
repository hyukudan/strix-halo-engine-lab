#include "strix/ds41_dspark.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::ds41::dspark {
namespace {

// Runs body(i) for i < n on the available hardware threads; every index is independent, so the
// result does not depend on the split.
template <class Body>
void parallel_for(std::size_t n, Body body) {
    const std::size_t threads = std::max<std::size_t>(1U, std::min<std::size_t>(std::thread::hardware_concurrency(), 32U));
    if (n < 64U || threads == 1U) {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }
    std::vector<std::thread> pool;
    const std::size_t chunk = (n + threads - 1U) / threads;
    for (std::size_t t = 0; t < threads; ++t) {
        const std::size_t begin = t * chunk, end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([begin, end, &body] {
            for (std::size_t i = begin; i < end; ++i) body(i);
        });
    }
    for (auto& t : pool) t.join();
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ds41 dspark: ") + what);
}

} // namespace

float f16_value(std::uint16_t bits) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU, mantissa = bits & 0x3ffU;
    std::uint32_t out = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) return std::bit_cast<float>(sign);
        const float v = static_cast<float>(mantissa) * 0x1p-24F; // exact
        return sign != 0U ? -v : v;
    }
    if (exponent == 31U) out = sign | 0x7f800000U | (mantissa << 13U);
    else out = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    return std::bit_cast<float>(out);
}

std::vector<float> main_hidden_row(std::span<const float> layer37, std::span<const float> layer38, std::span<const float> layer39) {
    require(layer37.size() == stream_width && layer38.size() == stream_width && layer39.size() == stream_width, "main hidden streams");
    std::vector<float> out(main_width);
    const std::span<const float> layers[3] = {layer37, layer38, layer39};
    for (std::uint32_t l = 0; l < target_layers; ++l)
        for (std::uint32_t i = 0; i < hidden; ++i) {
            const auto& s = layers[l];
            out[l * hidden + i] = (((s[i] + s[hidden + i]) + s[2U * hidden + i]) + s[3U * hidden + i]) * 0.25F;
        }
    return out;
}

std::vector<float> rms_norm(std::span<const float> x, std::uint32_t rows, std::span<const float> weight) {
    const auto width = static_cast<std::uint32_t>(weight.size());
    require(rows != 0U && width % 32U == 0U && x.size() == static_cast<std::size_t>(rows) * width, "rms norm geometry");
    std::vector<float> out(x.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = x.data() + static_cast<std::size_t>(r) * width;
        const float rstd = rstd_of(chunk_sum(width, [&](std::uint32_t i) { return row[i] * row[i]; }), width, norm_eps);
        for (std::uint32_t i = 0; i < width; ++i) out[static_cast<std::size_t>(r) * width + i] = (row[i] * rstd) * weight[i];
    }
    return out;
}

std::vector<float> q8_linear(const Q8Matrix& w, std::span<const float> x, std::uint32_t rows) {
    require(rows != 0U && w.width % 32U == 0U && x.size() == static_cast<std::size_t>(rows) * w.width &&
                w.bytes.size() == w.row_bytes() * w.outputs,
            "q8 linear geometry");
    std::vector<float> out(static_cast<std::size_t>(rows) * w.outputs);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto a8 = quantize_a8(x.subspan(static_cast<std::size_t>(r) * w.width, w.width));
        parallel_for(w.outputs, [&](std::size_t o) {
            out[static_cast<std::size_t>(r) * w.outputs + o] = product_q8_0(w.bytes.subspan(o * w.row_bytes(), w.row_bytes()), a8, w.width);
        });
    }
    return out;
}

std::array<float, mix_count> sinkhorn_split(std::span<const float, mix_count> mixes, std::span<const float> scale,
                                            std::span<const float> base) {
    require(scale.size() == 3U && base.size() == mix_count, "hc scale/base");
    std::array<float, mix_count> out{};
    sinkhorn_core(mixes.data(), scale.data(), base.data(), out.data());
    return out;
}

std::vector<float> hc_mixes(std::span<const float> streams, std::uint32_t rows, const F16Matrix& fn, std::span<const float> scale,
                            std::span<const float> base) {
    require(rows != 0U && streams.size() == static_cast<std::size_t>(rows) * stream_width && fn.width == stream_width &&
                fn.outputs == mix_count && fn.bits.size() == static_cast<std::size_t>(mix_count) * stream_width,
            "hc mixes geometry");
    std::vector<float> out(static_cast<std::size_t>(rows) * mix_count);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* x = streams.data() + static_cast<std::size_t>(r) * stream_width;
        const float rstd = rstd_of(chunk_sum(stream_width, [&](std::uint32_t i) { return x[i] * x[i]; }), stream_width, norm_eps);
        std::array<float, mix_count> mixes{};
        for (std::uint32_t j = 0; j < mix_count; ++j) {
            const std::uint16_t* f = fn.bits.data() + static_cast<std::size_t>(j) * stream_width;
            mixes[j] = chunk_sum(stream_width, [&](std::uint32_t i) { return f16_value(f[i]) * x[i]; }) * rstd;
        }
        const auto split = sinkhorn_split(mixes, scale, base);
        std::copy(split.begin(), split.end(), out.begin() + static_cast<std::ptrdiff_t>(r) * mix_count);
    }
    return out;
}

std::vector<float> hc_pre(std::span<const float> streams, std::uint32_t rows, std::span<const float> coefficients, std::uint32_t stride) {
    require(streams.size() == static_cast<std::size_t>(rows) * stream_width && coefficients.size() >= static_cast<std::size_t>(rows) * stride,
            "hc pre geometry");
    std::vector<float> out(static_cast<std::size_t>(rows) * hidden);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* x = streams.data() + static_cast<std::size_t>(r) * stream_width;
        const float* p = coefficients.data() + static_cast<std::size_t>(r) * stride;
        for (std::uint32_t i = 0; i < hidden; ++i)
            out[static_cast<std::size_t>(r) * hidden + i] =
                ((p[0] * x[i] + p[1] * x[hidden + i]) + p[2] * x[2U * hidden + i]) + p[3] * x[3U * hidden + i];
    }
    return out;
}

std::vector<float> hc_post(std::span<const float> x, std::span<const float> residual, std::uint32_t rows, std::span<const float> mixes) {
    require(x.size() == static_cast<std::size_t>(rows) * hidden && residual.size() == static_cast<std::size_t>(rows) * stream_width &&
                mixes.size() == static_cast<std::size_t>(rows) * mix_count,
            "hc post geometry");
    std::vector<float> out(residual.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* m = mixes.data() + static_cast<std::size_t>(r) * mix_count;
        const float* s = residual.data() + static_cast<std::size_t>(r) * stream_width;
        for (std::uint32_t k = 0; k < hc; ++k) {
            const float post = m[hc + k];
            const float* c = m + 2U * hc;
            for (std::uint32_t i = 0; i < hidden; ++i)
                out[static_cast<std::size_t>(r) * stream_width + k * hidden + i] =
                    post * x[static_cast<std::size_t>(r) * hidden + i] +
                    (((c[0U * hc + k] * s[i] + c[1U * hc + k] * s[hidden + i]) + c[2U * hc + k] * s[2U * hidden + i]) +
                     c[3U * hc + k] * s[3U * hidden + i]);
        }
    }
    return out;
}

RopeRow rope_row(std::uint64_t position) {
    RopeRow row;
    for (std::uint32_t i = 0; i < rope_dim / 2U; ++i) {
        const double frequency = 1.0 / std::pow(static_cast<double>(rope_theta), static_cast<double>(2U * i) / static_cast<double>(rope_dim));
        const double angle = static_cast<double>(position) * frequency;
        row.c[i] = static_cast<float>(std::cos(angle));
        row.s[i] = static_cast<float>(std::sin(angle));
    }
    return row;
}

void apply_rope(std::span<float> x, std::uint32_t rows, std::uint32_t head_width, std::uint64_t first, bool inverse) {
    const std::size_t row_width = rows == 0U ? 0U : x.size() / rows;
    require(rows != 0U && head_width >= rope_dim && row_width % head_width == 0U && row_width * rows == x.size(), "rope geometry");
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto table = rope_row(first + r);
        for (std::size_t h = 0; h < row_width / head_width; ++h) {
            float* tail = x.data() + static_cast<std::size_t>(r) * row_width + h * head_width + (head_width - rope_dim);
            for (std::uint32_t i = 0; i < rope_dim / 2U; ++i) rope_pair(tail[2U * i], tail[2U * i + 1U], table.c[i], table.s[i], inverse);
        }
    }
}

void fp8_entry(std::span<const float> x, std::span<std::uint8_t> entry) {
    require(x.size() == head_dim && entry.size() == entry_bytes, "fp8 entry geometry");
    for (std::uint32_t g = 0; g < head_dim / 32U; ++g) {
        float amax = 0.0F;
        for (std::uint32_t i = 0; i < 32U; ++i) amax = std::max(amax, std::fabs(x[32U * g + i]));
        const auto code = fp8_scale_code(amax);
        const float s = scale_value(code);
        entry[head_dim + g] = code;
        for (std::uint32_t i = 0; i < 32U; ++i) entry[32U * g + i] = fp8_element(x[32U * g + i], s);
    }
}

float entry_value(std::span<const std::uint8_t> entry, std::uint32_t i) noexcept {
    return decode_e4m3(entry[i]) * scale_value(entry[head_dim + i / 32U]);
}

std::vector<std::uint8_t> kv_entries(const StageWeights& w, std::span<const float> x, std::uint32_t rows, std::uint64_t first) {
    auto kv = rms_norm(q8_linear(w.kv, x, rows), rows, w.kv_norm);
    apply_rope(kv, rows, head_dim, first, false);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(rows) * entry_bytes);
    for (std::uint32_t r = 0; r < rows; ++r)
        fp8_entry(std::span<const float>(kv).subspan(static_cast<std::size_t>(r) * head_dim, head_dim),
                  std::span<std::uint8_t>(out).subspan(static_cast<std::size_t>(r) * entry_bytes, entry_bytes));
    return out;
}

std::vector<float> block_attention(std::span<const float> q, std::uint32_t rows, const Ring& ring, std::span<const std::uint8_t> block_entries,
                                   std::span<const float> sinks, std::uint64_t first) {
    require(q.size() == static_cast<std::size_t>(rows) * q_width && block_entries.size() == static_cast<std::size_t>(rows) * entry_bytes &&
                sinks.size() == heads && ring.next == first,
            "block attention geometry");
    const std::uint64_t oldest = first > window ? first - window : 0U;
    const auto ring_keys = static_cast<std::uint32_t>(first - oldest), keys = ring_keys + rows;
    std::vector<std::span<const std::uint8_t>> key(keys);
    for (std::uint32_t j = 0; j < ring_keys; ++j)
        key[j] = std::span<const std::uint8_t>(ring.entries).subspan(((oldest + j) % window) * entry_bytes, entry_bytes);
    for (std::uint32_t j = 0; j < rows; ++j) key[ring_keys + j] = block_entries.subspan(static_cast<std::size_t>(j) * entry_bytes, entry_bytes);
    // Decoded keys (exact values).
    std::vector<float> values(static_cast<std::size_t>(keys) * head_dim);
    for (std::uint32_t j = 0; j < keys; ++j)
        for (std::uint32_t d = 0; d < head_dim; ++d) values[static_cast<std::size_t>(j) * head_dim + d] = entry_value(key[j], d);
    std::vector<float> o(q.size());
    parallel_for(static_cast<std::size_t>(rows) * heads, [&](std::size_t rh) {
        const float* qh = q.data() + rh * head_dim;
        const auto h = static_cast<std::uint32_t>(rh % heads);
        std::vector<float> s(keys), e(keys);
        float m = 0.0F;
        for (std::uint32_t j = 0; j < keys; ++j) {
            const float* v = values.data() + static_cast<std::size_t>(j) * head_dim;
            s[j] = wave_sum(head_dim, [&](std::uint32_t d) { return qh[d] * v[d]; }) * attention_scale;
            m = j == 0U || s[j] > m ? s[j] : m;
        }
        float sum = 0.0F;
        for (std::uint32_t j = 0; j < keys; ++j) {
            e[j] = declared_exp(s[j] - m);
            sum = sum + e[j];
        }
        sum = sum + declared_exp(sinks[h] - m);
        for (std::uint32_t d = 0; d < head_dim; ++d) {
            float acc = 0.0F;
            for (std::uint32_t j = 0; j < keys; ++j) acc = acc + e[j] * values[static_cast<std::size_t>(j) * head_dim + d];
            o[rh * head_dim + d] = acc / sum;
        }
    });
    apply_rope(o, rows, head_dim, first, true);
    return o;
}

std::vector<float> output_projection(const StageWeights& w, std::span<const float> o, std::uint32_t rows) {
    require(o.size() == static_cast<std::size_t>(rows) * q_width && w.out_a.width == o_group_width && w.out_a.outputs == o_width,
            "output projection geometry");
    std::vector<float> low(static_cast<std::size_t>(rows) * o_width);
    const auto row_bytes = w.out_a.row_bytes();
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t g = 0; g < o_groups; ++g) {
            const auto a8 = quantize_a8(o.subspan(static_cast<std::size_t>(r) * q_width + g * o_group_width, o_group_width));
            parallel_for(o_rank, [&](std::size_t i) {
                const std::size_t out_row = g * o_rank + i;
                low[static_cast<std::size_t>(r) * o_width + out_row] =
                    product_q8_0(w.out_a.bytes.subspan(out_row * row_bytes, row_bytes), a8, o_group_width);
            });
        }
    return q8_linear(w.out_b, low, rows);
}

std::vector<float> moe(const StageWeights& w, std::span<const float> x, std::uint32_t rows) {
    const auto router = reference_router(x, rows, hidden, w.router, experts, w.bias, w.bias_image, {}, top_k, route_scale);
    std::vector<A8Activation> xs;
    for (std::uint32_t r = 0; r < rows; ++r) xs.push_back(quantize_a8(x.subspan(static_cast<std::size_t>(r) * hidden, hidden)));
    const std::vector<std::uint32_t> zeros(rows, 0U);
    const std::vector<float> ones(rows, 1.0F), none(static_cast<std::size_t>(rows) * hidden, 0.0F);
    const auto shared = reference_window(w.shared, xs, build_window_routes(zeros, rows, 1U, 1U), ones, none, 1U).raw;
    const auto routes = build_window_routes(router.ids, rows, top_k, experts / 2U);
    return reference_window(w.routed, xs, routes, router.weights, shared, top_k).y;
}

std::vector<float> main_rows(const Weights& w, std::span<const float> main_hidden, std::uint32_t rows) {
    return rms_norm(q8_linear(w.main_proj, main_hidden, rows), rows, w.main_norm);
}

void append(const Weights& w, State& state, std::span<const float> main_x, std::uint32_t rows) {
    require(main_x.size() == static_cast<std::size_t>(rows) * hidden, "append geometry");
    const auto first = state.rings[0].next;
    for (std::uint32_t s = 0; s < stages; ++s) {
        auto& ring = state.rings[s];
        require(ring.next == first, "rings out of step");
        const auto entries = kv_entries(w.stage[s], main_x, rows, first);
        for (std::uint32_t r = 0; r < rows; ++r)
            std::memcpy(ring.entries.data() + ((first + r) % window) * entry_bytes, entries.data() + static_cast<std::size_t>(r) * entry_bytes,
                        entry_bytes);
        ring.next = first + rows;
    }
}

void draft_head(const Weights& w, std::span<const float> streams, std::span<const float> pre_mix, Draft& out, bool keep_logits,
                const Vocab& vocabulary) {
    constexpr std::uint32_t rows = block_size;
    const std::uint32_t size = vocabulary.size();
    require(vocabulary.core <= vocab && size >= 1U, "draft vocabulary");
    out.head_in = hc_pre(streams, rows, pre_mix, mix_count);
    const auto h = rms_norm(out.head_in, rows, w.final_norm);
    // Head logits of the vocabulary rows (the declared Q8_0 product, A8 of each row).
    std::vector<float> logits(static_cast<std::size_t>(rows) * size);
    const auto row_bytes = w.output.row_bytes();
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto a8 = quantize_a8(std::span<const float>(h).subspan(static_cast<std::size_t>(r) * hidden, hidden));
        parallel_for(size, [&](std::size_t j) {
            const std::size_t v = vocabulary.id(static_cast<std::uint32_t>(j));
            logits[static_cast<std::size_t>(r) * size + j] = product_q8_0(w.output.bytes.subspan(v * row_bytes, row_bytes), a8, hidden);
        });
    }
    out.markov.assign(static_cast<std::size_t>(rows) * markov_rank, 0.0F);
    std::vector<float> value(size);
    for (std::uint32_t i = 0; i < rows; ++i) {
        float* e = out.markov.data() + static_cast<std::size_t>(i) * markov_rank;
        const std::uint16_t* row = w.markov_embed.bits.data() + static_cast<std::size_t>(out.tokens[i]) * markov_rank;
        for (std::uint32_t j = 0; j < markov_rank; ++j) e[j] = f16_value(row[j]);
        const float* l = logits.data() + static_cast<std::size_t>(i) * size;
        parallel_for(size, [&](std::size_t j) {
            const std::uint16_t* hv = w.markov_head.bits.data() + static_cast<std::size_t>(vocabulary.id(static_cast<std::uint32_t>(j))) * markov_rank;
            value[j] = l[j] + markov_dot(hv, e);
        });
        float best = -__builtin_huge_valf();
        std::uint32_t best_index = vocab;
        for (std::uint32_t j = 0; j < size; ++j) {
            const std::uint32_t v = vocabulary.id(j);
            if (better(value[j], v, best, best_index)) { best = value[j]; best_index = v; }
        }
        out.tokens[i + 1U] = best_index;
        out.candidates[i] = size;
        const float* x = out.head_in.data() + static_cast<std::size_t>(i) * hidden;
        out.confidence[i] = wave_sum(confidence_width, [&](std::uint32_t j) { return w.confidence[j] * (j < hidden ? x[j] : e[j - hidden]); });
    }
    if (keep_logits) out.logits = std::move(logits);
}

float markov_dot(const std::uint16_t* head_row, const float* e) noexcept {
    float lane[32];
    for (std::uint32_t l = 0; l < 32U; ++l) {
        float acc = 0.0F;
        for (std::uint32_t k = 0; k < 8U; ++k) acc = acc + f16_value(head_row[8U * l + k]) * e[8U * l + k];
        lane[l] = acc;
    }
    for (std::uint32_t d = 16; d != 0U; d /= 2U) {
        float next[32];
        for (std::uint32_t l = 0; l < 32U; ++l) next[l] = lane[l] + lane[l ^ d];
        for (std::uint32_t l = 0; l < 32U; ++l) lane[l] = next[l];
    }
    return lane[0];
}

Draft draft(const Weights& w, const State& state, std::uint32_t token, bool keep_logits, const Vocab& vocabulary) {
    constexpr std::uint32_t rows = block_size;
    require(token < vocab, "draft token");
    const auto first = state.rings[0].next;
    Draft out;
    out.tokens[0] = token;
    std::vector<float> streams(static_cast<std::size_t>(rows) * stream_width);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint32_t t = r == 0U ? token : noise_token;
        const std::uint16_t* e = w.token_embd.bits.data() + static_cast<std::size_t>(t) * hidden;
        for (std::uint32_t k = 0; k < hc; ++k)
            for (std::uint32_t i = 0; i < hidden; ++i) streams[static_cast<std::size_t>(r) * stream_width + k * hidden + i] = f16_value(e[i]);
    }
    std::vector<float> pre(static_cast<std::size_t>(rows) * mix_count, 0.0F);
    for (std::uint32_t r = 0; r < rows; ++r) pre[static_cast<std::size_t>(r) * mix_count] = 1.0F;
    for (std::uint32_t s = 0; s < stages; ++s) {
        const auto& sw = w.stage[s];
        const auto attn_mix = hc_mixes(streams, rows, sw.hc_attn_fn, sw.hc_attn_scale, sw.hc_attn_base);
        auto x = rms_norm(hc_pre(streams, rows, pre, mix_count), rows, sw.attn_norm);
        auto q = q8_linear(sw.q_b, rms_norm(q8_linear(sw.q_a, x, rows), rows, sw.q_norm), rows);
        apply_rope(q, rows, head_dim, first, false);
        const auto block = kv_entries(sw, x, rows, first);
        const auto o = block_attention(q, rows, state.rings[s], block, sw.sinks, first);
        streams = hc_post(output_projection(sw, o, rows), streams, rows, attn_mix);
        const auto ffn_mix = hc_mixes(streams, rows, sw.hc_ffn_fn, sw.hc_ffn_scale, sw.hc_ffn_base);
        x = rms_norm(hc_pre(streams, rows, attn_mix, mix_count), rows, sw.ffn_norm);
        streams = hc_post(moe(sw, x, rows), streams, rows, ffn_mix);
        pre = ffn_mix;
        out.stage_out[s] = streams;
    }
    draft_head(w, streams, pre, out, keep_logits, vocabulary);
    return out;
}

VerifyOutcome verify_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> target) {
    require(target.size() == drafts.size() + 1U && drafts.size() <= block_size, "verify window");
    VerifyOutcome out;
    while (out.accepted < drafts.size() && drafts[out.accepted] == target[out.accepted]) ++out.accepted;
    out.committed_rows = out.accepted + 1U;
    out.next_token = target[out.accepted];
    return out;
}

std::uint32_t choose_drafts(std::span<const float, block_size> confidence, std::span<const float, block_size + 1U> cost,
                            std::uint32_t min_drafts) {
    require(min_drafts <= block_size, "minimum drafts");
    float survival = 1.0F, expected = 1.0F;
    std::array<float, block_size + 1U> tokens{};
    tokens[0] = 1.0F;
    for (std::uint32_t j = 0; j < block_size; ++j) {
        survival = survival * sigmoid(confidence[j]);
        expected = expected + survival;
        tokens[j + 1U] = expected;
    }
    std::uint32_t best = min_drafts;
    for (std::uint32_t k = min_drafts + 1U; k <= block_size; ++k)
        if (tokens[k] / cost[k] > tokens[best] / cost[best]) best = k;
    return best;
}

} // namespace strix::ds41::dspark
