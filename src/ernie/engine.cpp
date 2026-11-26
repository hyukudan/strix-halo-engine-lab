#include "strix/ernie/engine.hpp"
#include "strix/ernie/attention.hpp"

#include "core/bf16_reference.hpp"
#include "moe_core/latent_experts_reference.hpp"
#include "moe_core/reference.hpp"
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

std::vector<float> feature_order(std::span<const float> streams, unsigned hidden) {
    std::vector<float> ordered(streams.size());
    for (unsigned d = 0; d < hidden; ++d)
        for (unsigned c = 0; c < kStreams; ++c)
            ordered[std::size_t{d} * kStreams + c] = streams[std::size_t{c} * hidden + d];
    return ordered;
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
    : FirstRowReference(shape, std::move(read_weights), activation, {}, {}) {}

FirstRowReference::FirstRowReference(ReferenceShape shape, ReferenceRead read_weights, ReferenceActivation activation,
                                     ReferenceProject projection, ExpertObserver observe_expert)
    : shape_(shape), read_(std::move(read_weights)), activation_(activation), project_(std::move(projection)),
      observe_expert_(std::move(observe_expert)) {
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
    if (project_) project_(name, input, output, 1);
    else core::linear_reference(read(name, output.size() * input.size()), input, output);
}

std::vector<float> FirstRowReference::attention(const std::string& prefix, std::span<const float> input) const {
    return attention_reference_first_hca(prefix, shape_, read_, input, project_);
}

// GCC 15 -O2 reports -Wfree-nonheap-object on the routed reference's scratch vectors once they are inlined through
// the latent-expert lambdas (a vector freed at its own data pointer); a false positive, scoped to this function.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
void FirstRowReference::feed_forward(unsigned layer_id, const std::string& layer, std::span<const float> input,
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
    if (!layer_has_moe(layer_id)) {
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
    if (s.contracts.router == RouterContract::swapped_planes) std::swap(select, mix);
    if (s.contracts.router == RouterContract::single_plane) mix = select;
    auto gain = read(prefix + "gate.routed_scaling_factor_param", s.route.experts);
    if (s.contracts.router == RouterContract::gain_before_normalize)
        for (std::size_t i = 0; i < mix.size(); ++i) mix[i] *= gain[i];
    if (s.contracts.router == RouterContract::unit_gain || s.contracts.router == RouterContract::gain_before_normalize)
        std::ranges::fill(gain, 1.0F);
    if (s.contracts.router == RouterContract::summed_scores) {
        std::ranges::transform(select, mix, select.begin(), std::plus<>{});
        mix = select;
    }
    result.experts.resize(s.route.top_k);
    result.expert_weights.resize(s.route.top_k);
    moe::route_select_split_ref(s.route, select, mix, read(prefix + "e_score_correction_bias", s.route.experts),
        gain, result.experts, result.expert_weights);

    const auto project_expert = [&](unsigned id, moe::ExpertProjection kind,
                                    std::span<const float> value, std::span<float> out) {
        if (observe_expert_) observe_expert_(layer_id, id, value, kind == moe::ExpertProjection::down);
        const auto expert = prefix + "experts." + std::to_string(id) + ".";
        if (kind == moe::ExpertProjection::gate_up) {
            project(expert + "w1.weight", value, out.first(s.expert_width));
            project(expert + "w3.weight", value, out.subspan(s.expert_width));
        } else {
            project(expert + "w2.weight", value, out);
        }
    };
    const auto epsilon = s.contracts.rms_epsilon > 0 ? s.contracts.rms_epsilon : kRmsEps;
    moe_core::latent_experts_reference(input, read(prefix + "latent_norm.weight", s.latent_width), epsilon,
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
    const auto prefix = target_layer_prefix(layer) + "self_attn.";
    return block(layer, streams, [&](auto row) { return attention(prefix, row); });
}

LayerResult FirstRowReference::block(unsigned layer, std::span<const float> streams,
                                     const AttentionRow& attend) const {
    if (layer >= kLayerCount || !attend)
        throw std::invalid_argument("ERNIE: invalid target block binding");
    auto site = hc_spec(shape_.hidden, hc::Site::sublayer);
    const auto residual_contract = shape_.contracts.residual;
    if (residual_contract == ResidualContract::half_post) site.post_magnitude = 1.0F;
    if (residual_contract == ResidualContract::bf16) site.store = hc::Store::bf16;
    const auto epsilon = shape_.contracts.rms_epsilon > 0 ? shape_.contracts.rms_epsilon : kRmsEps;
    const auto prefix = target_layer_prefix(layer);
    const auto run_site = [&](const char* name, const char* norm, std::span<const float> residual,
                              std::span<float> next, const auto& compute) {
        const auto hp = prefix + name;
        const float scale[] = {read(hp + "alpha_pre", 1).front(), read(hp + "alpha_post", 1).front(),
                               read(hp + "alpha_res", 1).front()};
        const auto fn = read(hp + "mapping_proj.weight", site.flat() * site.mix_rows());
        const auto base = read(hp + "bias", site.mix_rows());
        const auto ordered = residual_contract == ResidualContract::feature_interleave
            ? feature_order(residual, shape_.hidden) : std::vector<float>{};
        auto mix = hc::mixes(site, ordered.empty() ? residual : std::span<const float>(ordered),
                            {{}, fn, base, scale, hc::FnLayout::feature_major});
        if (residual_contract == ResidualContract::transpose)
            for (unsigned i = 0; i < kStreams; ++i)
                for (unsigned j = 0; j < i; ++j)
                    std::swap(mix.comb[i * kStreams + j], mix.comb[j * kStreams + i]);
        std::vector<float> normalized(shape_.hidden);
        hc::collapse(site, residual, mix.pre, normalized);
        core::rms_norm_bf16_reference(normalized, read(prefix + norm, normalized.size()), epsilon);
        hc::expand(site, compute(normalized), residual, mix, next);
    };
    LayerResult result{std::vector<float>(site.flat()), std::vector<float>(shape_.hidden),
                       std::vector<float>(site.flat()), {}, {}};
    run_site("self_attention_hyper_connection.", "input_layernorm.weight", streams, result.attention_streams,
             attend);
    run_site("mlp_hyper_connection.", "post_attention_layernorm.weight", result.attention_streams, result.streams,
             [&](auto row) -> std::span<const float> {
                 feed_forward(layer, prefix, row, result);
                 return result.ffn;
             });
    return result;
}

std::vector<float> FirstRowReference::embed(std::uint32_t token) const {
    if (token >= shape_.vocab) throw std::out_of_range("ERNIE: token outside reference vocabulary");
    const auto site = hc_spec(shape_.hidden, hc::Site::head);
    const auto embedding = read("model.embed_tokens.weight", shape_.hidden, elements(token, shape_.hidden));
    std::vector<float> streams(site.flat());
    for (unsigned i = 0; i < site.streams; ++i)
        std::ranges::copy(embedding, streams.begin() + static_cast<std::ptrdiff_t>(elements(i, shape_.hidden)));
    return streams;
}

std::vector<float> FirstRowReference::head_input(std::span<const float> streams) const {
    auto site = hc_spec(shape_.hidden, hc::Site::head);
    if (shape_.contracts.residual == ResidualContract::bf16) site.store = hc::Store::bf16;
    const auto fn = read("model.mhc_contract.hc_head_fn", site.flat() * site.mix_rows());
    const auto base = read("model.mhc_contract.hc_head_base", site.mix_rows());
    const auto scale = read("model.mhc_contract.hc_head_scale", 1);
    const auto ordered = shape_.contracts.residual == ResidualContract::feature_interleave
        ? feature_order(streams, shape_.hidden) : std::vector<float>{};
    const auto mix = hc::mixes(site, ordered.empty() ? streams : std::span<const float>(ordered),
                              {{}, fn, base, scale, hc::FnLayout::feature_major});
    std::vector<float> row(shape_.hidden);
    hc::collapse(site, streams, mix.pre, row);
    const auto epsilon = shape_.contracts.rms_epsilon > 0 ? shape_.contracts.rms_epsilon : kRmsEps;
    core::rms_norm_bf16_reference(row, read("model.norm.weight", row.size()), epsilon);
    return row;
}

std::vector<float> FirstRowReference::raw_logits(std::span<const float> streams) const {
    const auto row = head_input(streams);
    std::vector<float> logits(shape_.vocab);
    project("lm_head.weight", row, logits);
    return logits;
}

std::vector<float> FirstRowReference::embedding_first_block_logits(std::uint32_t token) const {
    return raw_logits(block(0, embed(token)).streams);
}

} // namespace strix::ernie
