#pragma once
// Hy4-flash (hy_v4_flash) model numerics as data for the shared policy types: the shared headers keep the types and their
// validators, the model package keeps its values (source: the model's config.json and reference forward).
// Home: specific (model numerics: Hy4 indexer, MLA layers, iHC, router, LatentMoE geometry).
#include "strix/attn/group_indexer.hpp"
#include "strix/attn/mla_contract.hpp"
#include "strix/hc/hc_contract.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/moe_q4k_a8_contract.hpp"

namespace strix::hy4::preset {

// Compressed indexer (config compressed_indexer_*: ratio 4, 16 heads x 128, group_topk 512, gated pooling).
inline constexpr attn::group_indexer::Shape indexer{4U, 16U, 128U, 512U, attn::group_indexer::Pooling::GatedSoftmaxApe,
                                                   attn::group_indexer::OwnGroup::AlwaysOwn,
                                                   attn::group_indexer::ScoreScale::InHeadWeight};

// Sparse MLA layers: DSA (indexed list of 2048 keys) and SWA (window of 512), learned sink, sigmoid gate.
constexpr attn::mla::LayerSpec dsa() {
    return {{32U, 512U, 64U, 192U, 256U}, {attn::mla::Select::List, 2048U}, attn::mla::Sink::Learned,
            attn::mla::Gate::SigmoidElementwise, 1.0e7};
}
constexpr attn::mla::LayerSpec swa() {
    return {{64U, 512U, 64U, 192U, 256U}, {attn::mla::Select::Window, 512U}, attn::mla::Sink::Learned,
            attn::mla::Gate::SigmoidElementwise, 1.0e4};
}

// iHC (config: hc_mult 4, hidden_size 3072, hc_eps 1e-6, hc_magnitude 2.0, rms_norm_eps 1e-5; fn F32 in the pack;
// streams stored in bfloat16 by the reference, which also rounds the collapse output).
[[nodiscard]] constexpr hc::Spec hc_spec(hc::Site site) {
    hc::Spec s;
    s.streams = 4;
    s.hidden = 3072;
    s.site = site;
    s.comb = hc::Comb::identity;
    s.fn = hc::FnType::f32;
    s.store = hc::Store::bf16;
    s.post_magnitude = 2.0F;
    s.pre_eps = 1e-6F;
    s.post_eps = 1e-6F;
    s.norm_eps = 1e-5F;
    return s;
}

// Router (config: sigmoid, e_score_correction_bias for selection, top-16 of 512, n_group 1, renorm (+1e-20), x 3.998).
// Output order ascending id: the reference accumulates the experts with index_add_ in ascending id and the CPU oracle
// sorts the ids (src/hy4/ref/hy4_host_ref.cpp route), so the denominator and the BF16 merge run in id order. Key
// ordered_bits: the oracle's stable sort on score + bias (lower id on ties, -0 == +0) for every non-NaN key; the HIP
// binding is moe_core::route_wave_kernel.
inline constexpr moe::RoutePolicy route{moe::RouteScore::sigmoid, true, moe::RouteKey::ordered_bits,
                                        moe::RouteOrder::ascending_id, 512U, 16U, 1U, 1U, true, true, 1e-20F, 3.998F};
static_assert(moe::validate(route) == nullptr);

// LatentMoE geometry of the A8 expert forms: latent 1536 in and out, 512 experts, top-16.
inline constexpr moe_q4k_a8::Geometry moe_geometry{1536, 1536, 512, 16};
static_assert(moe_q4k_a8::fits_c1(moe_geometry));

}  // namespace strix::hy4::preset
