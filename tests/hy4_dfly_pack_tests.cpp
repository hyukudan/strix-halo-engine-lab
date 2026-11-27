// DFly2 drafter pack (src/hy4/ref/hy4_dfly_pack.hpp): the shard as Hy4 blocks. Invariants: every device name present with
// GGUF dims {inputs, outputs}; kv_b split = the absorb forms (K block transposed per head, V block per head); the slot-0
// gate folded into eh_proj rows (one BF16 rounding of the FP32 product); norms / sinks exact F32; Q8_0 rows = the shared
// ggml rule on the BF16 values (bytes, scale, codes within half a step).
#include "hy4/ref/hy4_dfly_pack.hpp"

#include "core/iu4_onfly.hpp"
#include "strix/q8_0_rule.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dfly = strix::hy4::dfly;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}
float bf16_at(const dfly::PackTensor& t, std::size_t i) {
    std::uint16_t b = 0;
    std::memcpy(&b, t.bytes.data() + 2U * i, 2U);
    const std::uint32_t x = std::uint32_t{b} << 16U;
    float v = 0.0F;
    std::memcpy(&v, &x, 4U);
    return v;
}
float bf16_round(float v) {
    std::uint32_t x = 0;
    std::memcpy(&x, &v, 4U);
    x += 0x7fffU + ((x >> 16U) & 1U);
    x &= 0xffff0000U;
    float r = 0.0F;
    std::memcpy(&r, &x, 4U);
    return r;
}
const char* const kBlockNames[] = {"attn_norm.weight",   "attn_q_a.weight",   "attn_q_a_norm.weight", "attn_q_b.weight",
                                   "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
                                   "attn_gate.weight",   "attn_output.weight", "post_attention_norm.weight", "attn_sinks.weight",
                                   "ffn_norm.weight",    "ffn_gate.weight",   "ffn_up.weight",        "ffn_down.weight"};
const char* const kGlobals[] = {"dfly.enorm", "dfly.hnorm", "dfly.eh_proj", "dfly.final_norm", "dfly.hc_hnorm",
                                "dfly.hc_enorm", "dfly.hc_gate", "dfly.hc_up", "dfly.hc_down"};

void bf16_layout() {
    const auto g = dfly::Geometry::tiny();
    auto w = dfly::synthetic_weights(g, 7U);
    const std::uint32_t first = 43U;
    const auto p = dfly::build_pack(g, *w, first, dfly::PackFormat::bf16);
    check(p.tensors.size() == std::size_t{g.layers} * 16U + 9U, "tensor count");
    for (std::uint32_t i = 0; i < g.layers; ++i)
        for (const char* s : kBlockNames) {
            try {
                (void)p.at(dfly::block_name(first, i, s));
            } catch (const std::exception&) {
                check(false, "missing " + dfly::block_name(first, i, s));
            }
        }
    for (const char* s : kGlobals) {
        try {
            (void)p.at(s);
        } catch (const std::exception&) {
            check(false, std::string("missing ") + s);
        }
    }
    const std::uint64_t H = g.heads, kl = g.kv_lora, nope = g.nope, vd = g.v_head;
    const auto& kvb = w->tensor("1.self_attn.kv_b_proj.weight");
    const auto& kb = p.at(dfly::block_name(first, 1, "attn_k_b.weight"));
    const auto& vb = p.at(dfly::block_name(first, 1, "attn_v_b.weight"));
    check(kb.dims == std::vector<std::uint64_t>{nope, H * kl} && kb.ggml_type == dfly::kGgmlBf16, "k_b dims {nope, H kl}");
    check(vb.dims == std::vector<std::uint64_t>{kl, H * vd}, "v_b dims {kl, H vd}");
    bool k_ok = true, v_ok = true;
    for (std::uint64_t hd = 0; hd < H; ++hd)
        for (std::uint64_t j = 0; j < kl; ++j) {
            for (std::uint64_t n = 0; n < nope; ++n)  // k_b[hd][j][n] = kv_b[hd][n][j]
                k_ok = k_ok && bf16_at(kb, (hd * kl + j) * nope + n) == kvb[(hd * (nope + vd) + n) * kl + j];
            for (std::uint64_t o = 0; o < vd; ++o)  // v_b[hd][o][j] = kv_b[hd][nope + o][j]
                v_ok = v_ok && bf16_at(vb, (hd * vd + o) * kl + j) == kvb[(hd * (nope + vd) + nope + o) * kl + j];
        }
    check(k_ok, "k_b = K block transposed per head");
    check(v_ok, "v_b = V block per head");
    const auto& q = p.at(dfly::block_name(first, 0, "attn_q_b.weight"));
    check(q.dims == std::vector<std::uint64_t>{g.q_lora, H * g.qk()}, "q_b dims {q_lora, H qk}");
    const auto& eh = w->tensor("mtp_fusion.eh_proj.weight");
    const auto& gate = w->tensor("mtp_fusion.gate");
    const auto& ehp = p.at("dfly.eh_proj");
    bool fold_ok = ehp.dims == std::vector<std::uint64_t>{2U * g.hidden, g.hidden};
    for (std::size_t o = 0; o < g.hidden; ++o)
        for (std::size_t k = 0; k < 2U * g.hidden; ++k)
            fold_ok = fold_ok && bf16_at(ehp, o * 2U * g.hidden + k) == bf16_round(gate[o] * eh[o * 2U * g.hidden + k]);
    check(fold_ok, "eh_proj rows = bf16(gate[o] * eh_proj[o][k])");
    const auto& sinks = p.at(dfly::block_name(first, 1, "attn_sinks.weight"));
    const auto& sref = w->tensor("1.self_attn.learnable_sink_param");
    check(sinks.ggml_type == dfly::kGgmlF32 && sinks.bytes.size() == sref.size() * 4U &&
              std::memcmp(sinks.bytes.data(), sref.data(), sinks.bytes.size()) == 0,
          "sinks exact F32");
    const auto& fn = p.at("dfly.final_norm");
    check(std::memcmp(fn.bytes.data(), w->tensor("final_norm.weight").data(), fn.bytes.size()) == 0, "final norm exact F32");
}

void q8_rows() {
    dfly::Geometry g = dfly::Geometry::tiny();
    g.hidden = 64; g.heads = 2; g.q_lora = 32; g.kv_lora = 32; g.nope = 32; g.rope = 32; g.v_head = 32; g.ff = 96; g.layers = 1;
    auto w = dfly::synthetic_weights(g, 11U);
    const auto p = dfly::build_pack(g, *w, 5U, dfly::PackFormat::q8_0);
    const auto& t = p.at(dfly::block_name(5U, 0, "ffn_gate.weight"));
    const auto& src = w->tensor("0.mlp.gate_proj.weight");
    check(t.ggml_type == dfly::kGgmlQ8_0 && t.bytes.size() == src.size() / 32U * 34U, "q8 bytes = rows width/32 34");
    bool ok = true;
    for (std::size_t b = 0; b < src.size() / 32U; ++b) {
        const auto ref = strix::q8_0::quantize_group(src.data() + 32U * b);
        std::uint16_t d = 0;
        std::memcpy(&d, t.bytes.data() + 34U * b, 2U);
        ok = ok && d == strix::core::onfly::f32_to_f16_bits(ref.d) && std::memcmp(t.bytes.data() + 34U * b + 2U, ref.q, 32U) == 0;
        for (std::size_t j = 0; j < 32U; ++j)
            ok = ok && std::fabs(static_cast<float>(ref.q[j]) * ref.d - src[32U * b + j]) <= 0.5F * ref.d + 1e-7F;
    }
    check(ok, "q8 blocks = shared rule, within half a step");
    bool bad_width = false;
    try {
        dfly::Geometry t2 = dfly::Geometry::tiny();  // q_lora 16: not a Q8_0 width
        auto w2 = dfly::synthetic_weights(t2, 3U);
        (void)dfly::build_pack(t2, *w2, 1U, dfly::PackFormat::q8_0);
    } catch (const std::runtime_error&) {
        bad_width = true;
    }
    check(bad_width, "q8 refuses a width that is not a multiple of 32");
    check(dfly::parse_pack_format("q8") == dfly::PackFormat::q8_0 && dfly::parse_pack_format("bf16") == dfly::PackFormat::bf16, "format names");
}

}  // namespace

int main() {
    try {
        bf16_layout();
        q8_rows();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::printf("hy4 dfly pack tests: PASS\n");
    return 0;
}
