// Hy4-flash DFly2 drafter pack: the shard as Hy4 blocks (see hy4_dfly_pack.hpp).
#include "hy4/ref/hy4_dfly_pack.hpp"

#include "core/iu4_onfly.hpp"
#include "strix/q8_0_rule.hpp"

#include <cstring>
#include <span>
#include <stdexcept>

namespace strix::hy4::dfly {
namespace {

std::uint16_t bf16_bits(float v) {  // the shard's values are BF16 already (exact); folded products round here (RNE)
    std::uint32_t x = 0;
    std::memcpy(&x, &v, 4U);
    x += 0x7fffU + ((x >> 16U) & 1U);
    return static_cast<std::uint16_t>(x >> 16U);
}
float bf16_round(float v) {
    const std::uint32_t x = std::uint32_t{bf16_bits(v)} << 16U;
    float r = 0.0F;
    std::memcpy(&r, &x, 4U);
    return r;
}

void put_f32(Pack& p, std::string name, std::span<const float> v) {
    PackTensor t{std::move(name), kGgmlF32, {v.size()}, std::vector<std::uint8_t>(v.size() * 4U)};
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    p.tensors.push_back(std::move(t));
}

// rows x width values (row o = output o) -> BF16 or Q8_0 rows.
void put_linear(Pack& p, std::string name, std::span<const float> v, std::uint64_t rows, std::uint64_t width, PackFormat f) {
    if (v.size() != rows * width) throw std::runtime_error("dfly pack: shape of " + name);
    PackTensor t{std::move(name), f == PackFormat::bf16 ? kGgmlBf16 : kGgmlQ8_0, {width, rows}, {}};
    if (f == PackFormat::bf16) {
        t.bytes.resize(v.size() * 2U);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const std::uint16_t b = bf16_bits(v[i]);
            std::memcpy(t.bytes.data() + 2U * i, &b, 2U);
        }
    } else {
        if (width % q8_0::group_values != 0U) throw std::runtime_error("dfly pack: Q8_0 width of " + t.name);
        t.bytes.resize(static_cast<std::size_t>(rows * (width / q8_0::group_values) * q8_0::block_bytes));
        std::uint8_t* out = t.bytes.data();
        for (std::size_t gi = 0; gi < v.size(); gi += q8_0::group_values) {
            const auto grp = q8_0::quantize_group(v.data() + gi);
            const std::uint16_t d = core::onfly::f32_to_f16_bits(grp.d);
            std::memcpy(out, &d, 2U);
            std::memcpy(out + 2, grp.q, q8_0::group_values);
            out += q8_0::block_bytes;
        }
    }
    p.tensors.push_back(std::move(t));
}

std::string layer(std::uint32_t i, const char* name) { return std::to_string(i) + "." + name; }

}  // namespace

std::string block_name(std::uint32_t first_block, std::uint32_t layer_index, const char* suffix) {
    return "blk." + std::to_string(first_block + layer_index) + "." + suffix;
}

PackFormat parse_pack_format(const std::string& text) {
    if (text == "bf16") return PackFormat::bf16;
    if (text == "q8") return PackFormat::q8_0;
    throw std::invalid_argument("dfly pack format: bf16 | q8");
}

const PackTensor& Pack::at(const std::string& name) const {
    for (const auto& t : tensors)
        if (t.name == name) return t;
    throw std::runtime_error("dfly pack: no tensor " + name);
}

std::uint64_t Pack::total_bytes() const {
    std::uint64_t n = 0;
    for (const auto& t : tensors) n += t.bytes.size();
    return n;
}

Pack build_pack(const Geometry& g, Weights& w, std::uint32_t first_block, PackFormat f) {
    Pack p;
    p.first_block = first_block;
    const std::uint64_t h = g.hidden, H = g.heads, kl = g.kv_lora, nope = g.nope, vd = g.v_head, qk = g.qk();
    for (std::uint32_t i = 0; i < g.layers; ++i) {
        const auto name = [&](const char* s) { return block_name(first_block, i, s); };
        put_f32(p, name("attn_norm.weight"), w.tensor(layer(i, "input_layernorm.weight")));
        put_linear(p, name("attn_q_a.weight"), w.tensor(layer(i, "self_attn.q_a_proj.weight")), g.q_lora, h, f);
        put_f32(p, name("attn_q_a_norm.weight"), w.tensor(layer(i, "self_attn.q_a_layernorm.weight")));
        put_linear(p, name("attn_q_b.weight"), w.tensor(layer(i, "self_attn.q_b_proj.weight")), H * qk, g.q_lora, f);
        put_linear(p, name("attn_kv_a_mqa.weight"), w.tensor(layer(i, "self_attn.kv_a_proj_with_mqa.weight")), g.latent(), h, f);
        put_f32(p, name("attn_kv_a_norm.weight"), w.tensor(layer(i, "self_attn.kv_a_layernorm.weight")));
        {  // kv_b [H][nope + vd][kl]: K block per head transposed to [kl][nope] (q_nope -> latent), V block per head [vd][kl]
            const auto& kvb = w.tensor(layer(i, "self_attn.kv_b_proj.weight"));
            if (kvb.size() != H * (nope + vd) * kl) throw std::runtime_error("dfly pack: shape of kv_b_proj");
            std::vector<float> kb(H * kl * nope), vb(H * vd * kl);
            for (std::uint64_t hd = 0; hd < H; ++hd) {
                const float* src = kvb.data() + hd * (nope + vd) * kl;
                for (std::uint64_t j = 0; j < kl; ++j)
                    for (std::uint64_t n = 0; n < nope; ++n) kb[(hd * kl + j) * nope + n] = src[n * kl + j];
                std::memcpy(vb.data() + hd * vd * kl, src + nope * kl, vd * kl * sizeof(float));
            }
            put_linear(p, name("attn_k_b.weight"), kb, H * kl, nope, f);
            put_linear(p, name("attn_v_b.weight"), vb, H * vd, kl, f);
        }
        put_linear(p, name("attn_gate.weight"), w.tensor(layer(i, "self_attn.linear_gate.weight")), H * vd, h, f);
        put_linear(p, name("attn_output.weight"), w.tensor(layer(i, "self_attn.o_proj.weight")), h, H * vd, f);
        put_f32(p, name("post_attention_norm.weight"), w.tensor(layer(i, "self_attn.post_proj_norm.weight")));
        put_f32(p, name("attn_sinks.weight"), w.tensor(layer(i, "self_attn.learnable_sink_param")));
        put_f32(p, name("ffn_norm.weight"), w.tensor(layer(i, "post_attention_layernorm.weight")));
        put_linear(p, name("ffn_gate.weight"), w.tensor(layer(i, "mlp.gate_proj.weight")), g.ff, h, f);
        put_linear(p, name("ffn_up.weight"), w.tensor(layer(i, "mlp.up_proj.weight")), g.ff, h, f);
        put_linear(p, name("ffn_down.weight"), w.tensor(layer(i, "mlp.down_proj.weight")), h, g.ff, f);
    }
    put_f32(p, "dfly.enorm", w.tensor("mtp_fusion.enorm.weight"));
    put_f32(p, "dfly.hnorm", w.tensor("mtp_fusion.hnorm.weight"));
    {
        const auto& eh = w.tensor("mtp_fusion.eh_proj.weight");  // [h][2h]
        const auto& gate = w.tensor("mtp_fusion.gate");          // [h]
        if (eh.size() != h * 2U * h || gate.size() != h) throw std::runtime_error("dfly pack: shape of mtp_fusion");
        std::vector<float> folded(eh.size());
        for (std::uint64_t o = 0; o < h; ++o)
            for (std::uint64_t k = 0; k < 2U * h; ++k) folded[o * 2U * h + k] = bf16_round(gate[o] * eh[o * 2U * h + k]);
        put_linear(p, "dfly.eh_proj", folded, h, 2U * h, f);
    }
    put_f32(p, "dfly.final_norm", w.tensor("final_norm.weight"));
    put_f32(p, "dfly.hc_hnorm", w.tensor("hidden_correction.hidden_norm.weight"));
    put_f32(p, "dfly.hc_enorm", w.tensor("hidden_correction.embed_norm.weight"));
    put_linear(p, "dfly.hc_gate", w.tensor("hidden_correction.gate_proj.weight"), h, 2U * h, f);
    put_linear(p, "dfly.hc_up", w.tensor("hidden_correction.up_proj.weight"), h, 2U * h, f);
    put_linear(p, "dfly.hc_down", w.tensor("hidden_correction.down_proj.weight"), h, h, f);
    return p;
}

}  // namespace strix::hy4::dfly
