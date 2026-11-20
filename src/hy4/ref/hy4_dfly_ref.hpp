#pragma once
// Hy4-flash DFly2 drafter: host reference of ONE draft block (the CPU reference every backend's drafter is compared with).
// Model numerics only (the draft layer geometry is the target DSA layer's, mla::hy4_dsa; the controller contract is the
// shared strix/spec/block_parallel.hpp, preset hy4_dfly2).
//
// Block (anchor position p): rows j = 0..R-1 at positions p + j; row 0 = the anchor token (the last emitted id, not yet
// in the target cache), rows 1.. = the mask id. Inputs from the target: its embedding rows, its cached latent rows
// [kv_lora normed | qk_rope roped] of the KV-share layer of each draft layer at positions 0..p-1 (Index Share: every
// position while p <= share budget 2048), and the last-layer hidden of position p - 1 (hc_head output).
//   slot-0 fusion   x0 = e0 + gate * eh_proj(cat(enorm(e0), hnorm(h))) (measured, see Variant), rows j >= 1: x_j = e_mask
//   draft layer i   x += post_proj_norm(o_proj(sigmoid(gate) * MLA(input_layernorm(x)))), keys = shared latents + the
//                   block's own latents (own kv_a / kv_a_layernorm / RoPE), own kv_b expands both, learnable sinks;
//                   x += SwiGLU(post_attention_layernorm(x))
//   output          y = final_norm(x); proposal j = argmax(target head . y_j) for j = 1..R-1 (token p + j);
//                   confidence_j = sigmoid(conf . y_j + b)
// Arithmetic: FP32 with BF16 rounding of every Linear / norm output and of the residual stream (the model is BF16);
// softmax, RoPE angles and the head in FP32; ascending sums; argmax ties take the lower id. Draft numerics never change
// the emitted ids (the target verifies greedily); they decide acceptance only.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "hy4/ref/hy4_host_ref.hpp"

namespace strix::hy4::dfly {

struct Geometry {
    std::uint32_t hidden = 3072, heads = 32, q_lora = 1024, kv_lora = 512, nope = 192, rope = 64, v_head = 256;
    std::uint32_t ff = 9216, layers = 5, block = 6;
    float theta = 1e7F, eps = 1e-5F;
    std::uint32_t mask_id = 120009;
    [[nodiscard]] std::uint32_t qk() const { return nope + rope; }
    [[nodiscard]] std::uint32_t latent() const { return kv_lora + rope; }
    [[nodiscard]] static Geometry official();  // config.json dfly_* + the shard's shapes
    [[nodiscard]] static Geometry tiny();      // test geometry with the same structure
};

// Numerics points the files on disk do not settle (settled by measured acceptance on this reference). The default is the measured choice; the
// other values exist for the acceptance measurement only (strix-hy4-dfly accept --variant). Measured on the official
// drafter + oracle taps (golden copy loop 29 anchors, prose 60): default E 2.43 (4.03 / 1.65); bidirectional 1.84,
// h pre-norm 2.20, fusion eh 1.10, sigmoid mix 1.21 (bidir), every Correction placement lower (1.07 .. 2.37).
enum class HiddenForm : std::uint8_t { post_norm, pre_norm };                 // h = output_norm(hc_head) or hc_head
enum class FusionForm : std::uint8_t { eh, e_plus_gate, sigmoid_mix };      // x0 = eh | e0 + g*eh | s(g)*eh + (1-s(g))*e0
// h += C(e0, h) | x_j>=1 = e_mask + C(e_mask, h) | x0 += C(e0, h) | before final_norm x_j += C(e_j, x_j)
enum class Correction : std::uint8_t { off, hidden_residual, mask_residual, slot0_residual, output_residual };
enum class Mask : std::uint8_t { causal, bidirectional };                   // block rows among themselves
struct Variant {
    HiddenForm hidden = HiddenForm::post_norm;
    FusionForm fusion = FusionForm::e_plus_gate;
    Correction correction = Correction::off;
    Mask mask = Mask::causal;
};
// "h=post,fus=eh,hc=off,mask=causal" form; throws std::invalid_argument on an unknown key / value.
[[nodiscard]] Variant parse_variant(const std::string& text);

// Drafter tensors by their shard name without the "model.dfly_layers." prefix (e.g. "0.self_attn.q_a_proj.weight").
[[nodiscard]] std::unique_ptr<Weights> shard_weights(const std::filesystem::path& shard);
[[nodiscard]] std::unique_ptr<Weights> synthetic_weights(const Geometry& g, std::uint64_t seed);

// What the drafter reads from the target.
class Target {
public:
    virtual ~Target() = default;
    [[nodiscard]] virtual std::span<const float> embed(std::uint32_t token) = 0;
    // latent row of the KV-share layer of draft layer `layer` at target position `pos` (pos < anchor position)
    [[nodiscard]] virtual std::span<const float> latent(std::uint32_t layer, std::uint32_t pos) = 0;
    // logits of the target head for one final row y [hidden]
    virtual void logits(std::span<const float> y, std::vector<float>& out) = 0;
};

struct Block {
    std::vector<std::uint32_t> proposals;  // tokens p + 1 .. p + R - 1
    std::vector<float> confidence;         // per proposal
    std::vector<float> final_rows;         // [R][hidden] final_norm output (device compare)
};

// One draft block at anchor position `pos` (>= 1) for `anchor` with the target hidden `h` of position pos - 1 in the
// form Variant::hidden selects; `rows` <= Geometry::block.
[[nodiscard]] Block draft(const Geometry& g, const Variant& v, Weights& w, Target& target, std::uint32_t anchor,
                          std::uint32_t pos, std::span<const float> h, std::uint32_t rows);

}  // namespace strix::hy4::dfly
