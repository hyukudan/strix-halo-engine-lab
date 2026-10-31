#pragma once
#include "gemma4/ref/target.hpp"
#include "gemma4/exl3/format.hpp"
#include "gemma4/kv/layout.hpp"
#include <stdexcept>

namespace strix::gemma4::prefill::detail {
inline void expect(LinearView v, unsigned outputs, unsigned inputs) {
    if (!v.data || v.outputs != outputs || v.inputs != inputs ||
        (v.format == WeightFormat::exl3_mul1 ? !exl3::geometry(v) : v.row_bytes < packed_row_bytes(v.format, inputs)))
        throw std::invalid_argument("incomplete prefill tensor geometry");
}
inline void validate_weights(const HostWeights& w) {
    const auto& c = w.config;
    gpu::validate_config(c);
    if (w.layer.size() != c.layers) throw std::invalid_argument("incomplete prefill layers");
    expect(w.embedding, c.vocab, c.hidden); expect(w.ple_table, c.vocab, c.layers * c.ple_dim);
    if (w.head.data) expect(w.head, c.vocab, c.hidden);
    expect(w.ple_model, c.layers * c.ple_dim, c.hidden); expect(w.ple_model_norm, 1, c.ple_dim); expect(w.final_norm, 1, c.hidden);
    for (unsigned l = 0; l < c.layers; ++l) {
        const auto& y = w.layer[l]; const auto d = c.layer[l].head_dim;
        for (const auto v : {y.input_norm, y.post_attention_norm, y.pre_ffn_norm, y.post_ffn_norm, y.post_ple_norm}) expect(v, 1, c.hidden);
        expect(y.q, c.heads * d, c.hidden); expect(y.o, c.hidden, c.heads * d); expect(y.q_norm, 1, d);
        if (c.layer[l].producer == l) { expect(y.k, c.kv_heads * d, c.hidden); expect(y.v, c.kv_heads * d, c.hidden); expect(y.k_norm, 1, d); }
        expect(y.gate, c.inter, c.hidden); expect(y.up, c.inter, c.hidden); expect(y.down, c.hidden, c.inter);
        expect(y.ple_gate, c.ple_dim, c.hidden); expect(y.ple_projection, c.hidden, c.ple_dim); expect(y.scalar, 1, 1);
    }
}
inline void validate_tables(RopeTables t) {
    if (!t.positions || t.positions > max_positions || t.sliding.size() != std::size_t(t.positions) * 256 || t.global.size() != std::size_t(t.positions) * 512)
        throw std::invalid_argument("invalid prefill rotary tables");
    for (const auto table : {t.sliding, t.global})
        for (float a : table) if (!std::isfinite(a)) throw std::invalid_argument("nonfinite prefill rotary table");
    for (unsigned p = 0; p < t.positions; ++p)
        for (unsigned i = 64; i < 256; ++i)
            if (bit_cast<unsigned>(t.global[std::size_t(p) * 512 + 2 * i]) != 0x3f800000U || bit_cast<unsigned>(t.global[std::size_t(p) * 512 + 2 * i + 1]) != 0)
                throw std::invalid_argument("noncanonical inactive prefill rotary coordinates");
}
} // namespace strix::gemma4::prefill::detail
