#include "strix/ernie/engine.hpp"

#include "attn_core/causal_gqa_reference.hpp"
#include "attn_core/value_head_mix_reference.hpp"
#include "core/bf16_reference.hpp"
#include "moe_core/latent_experts_reference.hpp"
#include "moe_core/reference.hpp"
#include "strix/attn/head_rms_reference.hpp"
#include "strix/attn/hybrid_rope.hpp"
#include "strix/hc/hc_ref.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::ernie {
namespace {

namespace d = data::dims;
namespace a = data::attention;
namespace m = data::moe;
namespace x = data::extras;
using model::nearest_float;
using model::narrow;
constexpr auto kStreams = narrow<unsigned>(x::hc_mult);
constexpr auto kRmsEps = nearest_float(a::rms_norm_eps);
constexpr auto kHcEps = nearest_float(x::hc_eps);
constexpr auto kSinkhornIters = narrow<unsigned>(x::hc_sinkhorn_iters);
static_assert(d::n_kv_heads == 1 && a::kv_lora_rank == d::head_dim && a::v_head_dim == d::head_dim);
static_assert(a::rope_interleave && a::gate_use_q_lora && a::use_vha && a::qk_norm == "per_head");
static_assert(m::latent_moe_use_norm && m::moe_split_feature_routing && m::n_shared_experts == 1 &&
              !m::moe_shared_expert_gate && m::routed_scaling_factor_learnable && !x::tie_word_embeddings);

std::size_t elements(unsigned rows, unsigned columns) {
    return std::size_t{rows} * columns;
}

hc::Spec hc_spec(unsigned hidden, hc::Site site) {
    hc::Spec spec{};
    spec.hidden = hidden;
    spec.streams = kStreams;
    spec.site = site;
    spec.sinkhorn_iters = kSinkhornIters;
    spec.pre_eps = spec.comb_eps = spec.norm_eps = kHcEps;
    return spec;
}

struct RouteMath {
    static float exp(float x) { return std::exp(x); }
};

} // namespace

ReferenceShape checkpoint_reference_shape() {
    return {
        narrow<unsigned>(d::hidden), narrow<unsigned>(d::vocab), narrow<unsigned>(d::intermediate),
        narrow<unsigned>(m::moe_latent_size), narrow<unsigned>(m::moe_intermediate_size),
        narrow<unsigned>(d::n_heads), narrow<unsigned>(d::head_dim), narrow<unsigned>(a::qk_rope_head_dim),
        narrow<unsigned>(a::q_lora_rank), narrow<unsigned>(a::o_groups), narrow<unsigned>(a::o_lora_rank),
        narrow<unsigned>(a::vha_rank),
        {.score = moe::route_score_named(m::scoring_func), .selection_bias = true,
         .experts = kExpertCount, .top_k = narrow<unsigned>(m::num_experts_per_tok),
         .n_group = narrow<unsigned>(m::n_group), .topk_group = narrow<unsigned>(m::topk_group),
         .renormalize = m::norm_topk_prob, .scale = nearest_float(m::routed_scaling_factor)}
    };
}

FirstRowReference::FirstRowReference(ReferenceShape shape, ReferenceRead read_weights, ReferenceActivation activation)
    : shape_(shape), read_(std::move(read_weights)), activation_(activation) {
    const auto& s = shape_;
    if (!read_ || !activation_ || !s.hidden || !s.vocab || !s.dense_width || !s.latent_width || !s.expert_width ||
        !s.heads || !s.head_dim || s.heads > std::numeric_limits<unsigned>::max() / s.head_dim ||
        !s.q_rank || !s.o_groups || !s.o_rank || s.heads % s.o_groups != 0 ||
        s.o_groups > std::numeric_limits<unsigned>::max() / s.o_rank ||
        !s.vha_rank || s.rope_dim > s.head_dim || s.rope_dim % 2 != 0 ||
        moe::validate(s.route) || s.route.math != moe::RouteMath::native)
        throw std::invalid_argument("ERNIE: invalid first-row binding");
}

std::vector<float> FirstRowReference::read(const std::string& name, std::size_t count, std::size_t first) const {
    std::vector<float> values(count);
    read_(name, values, first);
    return values;
}

void FirstRowReference::project(const std::string& name, std::span<const float> input, std::span<float> output) const {
    core::linear_reference(read(name, output.size() * input.size()), input, output);
}

std::vector<float> FirstRowReference::attention(const std::string& prefix, std::span<const float> input) const {
    const auto& s = shape_;
    std::vector<float> qr(s.q_rank), query(elements(s.heads, s.head_dim)), kv(s.head_dim);
    project(prefix + "linear_q_down_proj.weight", input, qr);
    core::rms_norm_bf16_reference(qr, read(prefix + "q_layernorm.weight", qr.size()), kRmsEps);
    project(prefix + "linear_q_up_proj.weight", qr, query);
    project(prefix + "linear_kv_proj.weight", input, kv);
    core::rms_norm_bf16_reference(kv, read(prefix + "kv_layernorm.weight", kv.size()), kRmsEps);
    attn::head_rms_reference(query, s.head_dim, kRmsEps);
    attn::head_rms_reference(kv, s.head_dim, kRmsEps);

    std::vector<float> cosines(s.rope_dim / 2U), sines(cosines.size());
    attn::hybrid_rope_row(0, static_cast<double>(a::rope_theta), nullptr, cosines, sines);
    attn::hybrid_rope_tail(query, s.head_dim, cosines, sines, false);
    attn::hybrid_rope_tail(kv, s.head_dim, cosines, sines, false);
    std::vector<float> keys(kv.size()), values(kv.size()), head_output(query.size());
    const auto sinks = read(prefix + "core_attention.attn_sink", s.heads);
    attn_core::causal_gqa_reference({s.heads, 1, s.head_dim}, {0, query, kv, kv, 0, {}, sinks},
                                   {keys, values}, head_output);
    attn::hybrid_rope_tail(head_output, s.head_dim, cosines, sines, true);
    const auto mix_size = elements(s.heads, s.vha_rank);
    attn_core::residual_value_head_mix(head_output, s.heads, s.vha_rank,
        read(prefix + "vha_postmix_U", mix_size), read(prefix + "vha_postmix_V", mix_size));

    std::vector<float> gate(elements(s.o_groups, s.o_rank)), grouped(gate.size()), output(s.hidden);
    const auto group_weights = read(prefix + "linear_o_group_proj", head_output.size() / s.o_groups * grouped.size());
    core::grouped_linear_reference(group_weights, head_output, grouped, s.o_groups);
    project(prefix + "gate_proj.weight", qr, gate);
    core::sigmoid_product_reference(grouped, gate, grouped, core::SigmoidGatePrecision::f32);
    project(prefix + "o_proj.weight", grouped, output);
    return output;
}

// GCC 15 -O2 reports -Wfree-nonheap-object on the routed reference's scratch vectors once they are inlined through
// the latent-expert lambdas (a vector freed at its own data pointer); a false positive, scoped to this function.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
void FirstRowReference::feed_forward(const std::string& layer, bool routed, std::span<const float> input,
                                     LayerResult& result) const {
    const auto& s = shape_;
    const auto activate = [this](float gate, float up) {
        return activation_(gate, up, nearest_float(m::activation_situ_beta), nearest_float(m::activation_situ_linear_beta));
    };
    const auto glu = [&](const std::string& gate, const std::string& up, const std::string& down,
                         unsigned width, std::span<const float> in, std::span<float> out) {
        std::vector<float> gate_values(width), up_values(width);
        project(gate, in, gate_values);
        project(up, in, up_values);
        std::ranges::transform(gate_values, up_values, gate_values.begin(), activate);
        project(down, gate_values, out);
    };
    if (!routed) {
        glu(layer + "mlp.gate_proj.weight", layer + "mlp.up_proj.weight", layer + "mlp.down_proj.weight",
            s.dense_width, input, result.ffn);
        return;
    }
    const auto prefix = layer + "block_sparse_moe.";
    std::vector<float> select(s.route.experts), mix(s.route.experts);
    project(prefix + "gate.weight", input, select);
    project(prefix + "gate.weight_1", input, mix);
    moe::route_scores_ref<RouteMath>(s.route.score, select, select);
    moe::route_scores_ref<RouteMath>(s.route.score, mix, mix);
    result.experts.resize(s.route.top_k);
    result.expert_weights.resize(s.route.top_k);
    moe::route_select_split_ref(s.route, select, mix, read(prefix + "e_score_correction_bias", s.route.experts),
        read(prefix + "gate.routed_scaling_factor_param", s.route.experts), result.experts, result.expert_weights);

    const auto project_expert = [&](unsigned id, moe::ExpertProjection kind,
                                    std::span<const float> value, std::span<float> out) {
        const auto expert = prefix + "experts." + std::to_string(id) + ".";
        if (kind == moe::ExpertProjection::gate_up) {
            project(expert + "w1.weight", value, out.first(s.expert_width));
            project(expert + "w3.weight", value, out.subspan(s.expert_width));
        } else {
            project(expert + "w2.weight", value, out);
        }
    };
    moe_core::latent_experts_reference(input, read(prefix + "latent_norm.weight", s.latent_width), kRmsEps,
        [&](auto in, auto out) { project(prefix + "fc1_latent_proj.weight", in, out); },
        [&](auto in, auto out) {
            moe::combine_routed_reference(s.expert_width, in, result.experts, result.expert_weights,
                project_expert, activate, [](float v) { return v; }, out);
        },
        [&](auto in, auto out) { project(prefix + "fc2_latent_proj.weight", in, out); },
        [&](auto in, auto out) {
            const auto shared = prefix + "shared_experts.";
            glu(shared + "w1.weight", shared + "w3.weight", shared + "w2.weight", s.expert_width, in, out);
        }, result.ffn);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

LayerResult FirstRowReference::block(unsigned layer, std::span<const float> streams) const {
    if (layer >= kLayerCount || layer_is_mla(layer))
        throw std::invalid_argument("ERNIE: first-row reference requires a target HCA layer");
    const auto site = hc_spec(shape_.hidden, hc::Site::sublayer);
    const auto prefix = target_layer_prefix(layer);
    const auto run_site = [&](const char* name, const char* norm, std::span<const float> residual,
                              std::span<float> next, const auto& compute) {
        const auto hp = prefix + name;
        const float scale[] = {read(hp + "alpha_pre", 1).front(), read(hp + "alpha_post", 1).front(),
                               read(hp + "alpha_res", 1).front()};
        const auto fn = read(hp + "mapping_proj.weight", site.flat() * site.mix_rows());
        const auto base = read(hp + "bias", site.mix_rows());
        const auto mix = hc::mixes(site, residual, {{}, fn, base, scale, hc::FnLayout::feature_major});
        std::vector<float> normalized(shape_.hidden);
        hc::collapse(site, residual, mix.pre, normalized);
        core::rms_norm_bf16_reference(normalized, read(prefix + norm, normalized.size()), kRmsEps);
        hc::expand(site, compute(normalized), residual, mix, next);
    };
    LayerResult result{std::vector<float>(site.flat()), std::vector<float>(shape_.hidden),
                       std::vector<float>(site.flat()), {}, {}};
    run_site("self_attention_hyper_connection.", "input_layernorm.weight", streams, result.attention_streams,
             [&](auto row) { return attention(prefix + "self_attn.", row); });
    run_site("mlp_hyper_connection.", "post_attention_layernorm.weight", result.attention_streams, result.streams,
             [&](auto row) -> std::span<const float> {
                 feed_forward(prefix, layer_has_moe(layer), row, result);
                 return result.ffn;
             });
    return result;
}

std::vector<float> FirstRowReference::embedding_first_block_logits(std::uint32_t token) const {
    if (token >= shape_.vocab) throw std::out_of_range("ERNIE: token outside reference vocabulary");
    const auto site = hc_spec(shape_.hidden, hc::Site::head);
    const auto embedding = read("model.embed_tokens.weight", shape_.hidden, elements(token, shape_.hidden));
    std::vector<float> streams(site.flat());
    for (unsigned i = 0; i < site.streams; ++i)
        std::ranges::copy(embedding, streams.begin() + static_cast<std::ptrdiff_t>(elements(i, shape_.hidden)));
    const auto result = block(0, streams);

    const auto fn = read("model.mhc_contract.hc_head_fn", site.flat() * site.mix_rows());
    const auto base = read("model.mhc_contract.hc_head_base", site.mix_rows());
    const auto scale = read("model.mhc_contract.hc_head_scale", 1);
    const auto mix = hc::mixes(site, result.streams, {{}, fn, base, scale, hc::FnLayout::feature_major});
    std::vector<float> row(shape_.hidden), logits(shape_.vocab);
    hc::collapse(site, result.streams, mix.pre, row);
    core::rms_norm_bf16_reference(row, read("model.norm.weight", row.size()), kRmsEps);
    project("lm_head.weight", row, logits);
    return logits;
}

} // namespace strix::ernie
