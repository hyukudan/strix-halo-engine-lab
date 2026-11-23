#include "inkling2_mini_dspark_reference.hpp"

#include "attn_core/block_gqa_reference.hpp"
#include "inkling2_mini_forward.hpp"
#include "core/bf16_reference.hpp"
#include "ops/glu/glu.hpp"
#include "core/markov_head_reference.hpp"
#include "strix/serial_request.hpp"

#include <utility>

namespace strix::inkling2 {
namespace {

using D = MiniDSparkCheckpoint;
constexpr unsigned tap_count = static_cast<unsigned>(D::target_layers.size());
constexpr unsigned tap_stride = tap_count * kHiddenWidth;
constexpr auto epsilon = D::epsilon;
constexpr auto rope_theta = D::rope_theta;

void check(const weight::Status& status) {
    if (!status) throw std::runtime_error(status.detail);
}

void norm_rotary(std::span<float> row, std::span<const float> gain, std::uint64_t position) {
    for (std::size_t offset = 0; offset < row.size(); offset += kHeadWidth)
        core::rms_norm_bf16_reference(row.subspan(offset, kHeadWidth), gain, epsilon);
    attn_core::split_half_rotary_reference(row, kHeadWidth, rope_theta, position);
    core::round_bf16_reference(row);
}

struct ExactExp {
    static float exp(float x) { return std::exp(x); }
};

}  // namespace

MiniDSparkReference::MiniDSparkReference(const std::string& sidecar, ReadTarget read_target)
    : read_target_(std::move(read_target)) {
    if (!read_target_) throw std::invalid_argument("DSpark needs target weight slices");
    check(weights_.open_safetensors(sidecar, D::schema()));
    check(weights_.read_floats("fuse_weight", fusion_));
    for (auto offset = 0U; offset < fusion_.size(); offset += tap_count)
        core::softmax_bf16_reference(std::span(fusion_).subspan(offset, tap_count));
}

spec::BlockPolicy MiniDSparkReference::policy() {
    return D::policy();
}

std::vector<float> MiniDSparkReference::read(std::string_view name, std::size_t count, std::size_t first) const {
    std::vector<float> result(count);
    check(weights_.read_floats(name, result, first));
    return result;
}

void MiniDSparkReference::append_context(std::span<const float> taps) {
    if (taps.empty() || taps.size() % tap_stride != 0 || taps.size() / tap_stride > kMaxContext - positions_)
        throw std::invalid_argument("DSpark: invalid committed tap shape");
    const auto count = taps.size() / tap_stride;
    for (unsigned layer = 0; layer < D::layers; ++layer) {
        auto& cache = context_[layer];
        const auto local = D::window(layer);
        const auto rows = local ? local : positions_ + count;
        cache.keys.resize(static_cast<std::size_t>(rows) * kKvWidth);
        cache.values.resize(cache.keys.size());
        const auto gain = read(mini_layer_parameter(layer, "dflare_norm.weight"), kHiddenWidth);
        const auto k_gain = read(mini_layer_parameter(layer, "self_attn.k_norm_ctx.weight"), kHeadWidth);
        const auto k_weights = read(mini_layer_parameter(layer, "self_attn.k_proj_ctx.weight"), kKvWidth * kHiddenWidth);
        const auto v_weights = read(mini_layer_parameter(layer, "self_attn.v_proj_ctx.weight"), kKvWidth * kHiddenWidth);
        for (std::size_t row = 0; row < count; ++row) {
            const auto position = positions_ + row;
            const auto row_offset = attn::cache_rows::row_offset({kKvWidth, local}, position);
            const auto fused = core::mix_rms_norm_bf16_reference(taps.subspan(row * tap_stride, tap_stride),
                                   std::span(fusion_).subspan(layer * tap_count, tap_count), gain, epsilon);
            auto key = std::span(cache.keys).subspan(row_offset, kKvWidth);
            auto value = std::span(cache.values).subspan(row_offset, kKvWidth);
            core::linear_reference(k_weights, fused, key);
            core::round_bf16_reference(key);
            norm_rotary(key, k_gain, position);
            core::linear_reference(v_weights, fused, value);
            core::round_bf16_reference(value);
        }
    }
    last_markov_context_ = core::mix_rms_norm_bf16_reference(taps.last(tap_stride), std::span(fusion_).last(tap_count),
                               read("markov_head.dflare_norm.weight", kHiddenWidth), epsilon);
    positions_ += count;
}

void MiniDSparkReference::run_layer(unsigned layer, std::uint64_t anchor, std::span<float> hidden,
                                     std::span<float> normalized) const {
    const auto get = [&](std::string_view suffix, std::size_t size) {
        return read(mini_layer_parameter(layer, suffix), size);
    };
    auto q = core::linear_rows_bf16_reference(get("self_attn.q_proj.weight", kQueryWidth * kHiddenWidth), normalized, kHiddenWidth, kQueryWidth);
    auto k = core::linear_rows_bf16_reference(get("self_attn.k_proj.weight", kKvWidth * kHiddenWidth), normalized, kHiddenWidth, kKvWidth);
    auto v = core::linear_rows_bf16_reference(get("self_attn.v_proj.weight", kKvWidth * kHiddenWidth), normalized, kHiddenWidth, kKvWidth);
    const auto q_gain = get("self_attn.q_norm.weight", kHeadWidth);
    const auto k_gain = get("self_attn.k_norm.weight", kHeadWidth);
    for (unsigned row = 0; row < D::block; ++row) {
        norm_rotary(std::span(q).subspan(row * kQueryWidth, kQueryWidth), q_gain, anchor + row);
        norm_rotary(std::span(k).subspan(row * kKvWidth, kKvWidth), k_gain, anchor + row);
    }

    const auto& ctx = context_[layer];
    const auto local = D::window(layer);
    const auto seen = local ? std::min<std::uint64_t>(local, anchor) : anchor;
    std::vector<float> keys, values;
    keys.reserve(static_cast<std::size_t>(seen + D::block) * kKvWidth);
    values.reserve(keys.capacity());
    for (auto position = anchor - seen; position < anchor; ++position) {
        const auto offset = attn::cache_rows::row_offset({kKvWidth, local}, position);
        const auto key = std::span(ctx.keys).subspan(offset, kKvWidth);
        const auto value = std::span(ctx.values).subspan(offset, kKvWidth);
        keys.insert(keys.end(), key.begin(), key.end());
        values.insert(values.end(), value.begin(), value.end());
    }
    keys.insert(keys.end(), k.begin(), k.end());
    values.insert(values.end(), v.begin(), v.end());
    std::vector<float> attended(q.size());
    attn_core::bidirectional_gqa_reference(kQueryHeads, kKvHeads, kHeadWidth, q, keys, values, attended);
    core::round_bf16_reference(attended);

    const auto projected = core::linear_rows_bf16_reference(get("self_attn.o_proj.weight", kHiddenWidth * kQueryWidth),
                                   attended, kQueryWidth, kHiddenWidth);
    core::residual_rms_norm_bf16_reference(hidden, projected, normalized,
                                         get("post_attention_layernorm.weight", kHiddenWidth), epsilon);
    auto gate = core::linear_rows_bf16_reference(get("mlp.gate_proj.weight", D::intermediate * kHiddenWidth),
                        normalized, kHiddenWidth, D::intermediate);
    const auto up = core::linear_rows_bf16_reference(get("mlp.up_proj.weight", D::intermediate * kHiddenWidth),
                            normalized, kHiddenWidth, D::intermediate);
    for (std::size_t i = 0; i < gate.size(); ++i)
        gate[i] = ops::glu::swiglu<ExactExp, ops::glu::Round::none>(gate[i], up[i], ops::glu::no_clamp);
    core::round_bf16_reference(gate);
    const auto down = core::linear_rows_bf16_reference(get("mlp.down_proj.weight", kHiddenWidth * D::intermediate),
                              gate, D::intermediate, kHiddenWidth);
    const auto next_gain = layer + 1U < D::layers
        ? read(mini_layer_parameter(layer + 1U, "input_layernorm.weight"), kHiddenWidth)
        : read("norm.weight", kHiddenWidth);
    core::residual_rms_norm_bf16_reference(hidden, down, normalized, next_gain, epsilon);
}

void MiniDSparkReference::fold_head_tile(std::size_t first, std::span<const float> lm_head,
                                         std::span<const float> weight, std::span<const float> bias) {
    constexpr auto rank = D::markov_rank;
    std::array<double, rank> workspace{};
    for (std::size_t row = 0; row < lm_head.size() / kHiddenWidth; ++row) {
        auto folded = std::span(folded_head_).subspan((first + row) * rank, rank);
        folded_bias_[first + row] = core::fold_markov_head_row_reference(
            lm_head.subspan(row * kHiddenWidth, kHiddenWidth), weight, bias, workspace, folded);
    }
}

std::vector<float> MiniDSparkReference::readout(std::span<const float> hidden, unsigned proposals) {
    constexpr unsigned tile = 128U;
    const bool fold = !folded_;
    std::vector<float> weight, bias;
    if (fold) {
        weight = read("markov_head.hidden_out.weight", kHiddenWidth * D::markov_rank);
        bias = read("markov_head.hidden_out.bias", kHiddenWidth);
        folded_head_.resize(kOutputVocabulary * D::markov_rank);
        folded_bias_.resize(kOutputVocabulary);
    }
    std::vector<float> output(proposals * kOutputVocabulary);
    std::vector<float> matrix(tile * kHiddenWidth);
    for (std::size_t first = 0; first < kOutputVocabulary; first += tile) {
        const auto rows = std::min<std::size_t>(tile, kOutputVocabulary - first);
        const auto weights = std::span(matrix).first(rows * kHiddenWidth);
        read_target_("unembed.weight", weights, first * kHiddenWidth);
        if (fold) fold_head_tile(first, weights, weight, bias);
        for (unsigned j = 0; j < proposals; ++j)
            core::linear_reference(weights, hidden.subspan((j + 1U) * kHiddenWidth, kHiddenWidth),
                                   std::span(output).subspan(j * kOutputVocabulary + first, rows));
    }
    if (fold) folded_ = true;
    core::round_bf16_reference(output);
    return output;
}

void MiniDSparkReference::forward(const spec::BlockRequest& request, std::vector<std::uint32_t>& proposals,
                                  std::vector<float>& confidence) {
    if (request.anchor_pos != positions_ || positions_ == 0U || positions_ > kMaxContext - D::block ||
        request.ids.size() != D::block || request.prefix != 0 || request.k == 0 || request.k >= D::block ||
        request.ids.front() >= kVocabulary)
        throw std::invalid_argument("DSpark: invalid proposal frontier or block");

    std::vector<float> hidden(D::block * kHiddenWidth);
    auto anchor = std::span(hidden).first(kHiddenWidth);
    read_target_("embed.weight", anchor, std::size_t{request.ids.front()} * kHiddenWidth);
    std::vector<float> embed_gain(kHiddenWidth);
    read_target_("embed_norm.weight", embed_gain, 0U);
    core::rms_norm_bf16_reference(anchor, embed_gain, epsilon);
    const auto mask = read("mask_embed", kHiddenWidth);
    for (unsigned row = 1; row < D::block; ++row)
        std::ranges::copy(mask, hidden.begin() + row * kHiddenWidth);
    auto normalized = hidden;
    const auto first_gain = read("layers.0.input_layernorm.weight", kHiddenWidth);
    for (unsigned row = 0; row < D::block; ++row)
        core::rms_norm_bf16_reference(std::span(normalized).subspan(row * kHiddenWidth, kHiddenWidth), first_gain, epsilon);
    for (unsigned layer = 0; layer < D::layers; ++layer)
        run_layer(layer, request.anchor_pos, hidden, normalized);

    auto logits = readout(normalized, request.k);
    constexpr unsigned rank = D::markov_rank;
    constexpr unsigned recurrent_inputs = 2U * rank;
    constexpr unsigned affine_width = recurrent_inputs + 2U * kHiddenWidth;
    constexpr unsigned outputs = 3U * rank;
    const auto matrix = read("markov_head.rnn_proj.weight", outputs * affine_width);
    const auto bias = read("markov_head.rnn_proj.bias", outputs);
    std::array<float, recurrent_inputs> recurrent{};
    std::array<float, 2U * kHiddenWidth> features{};
    std::ranges::copy(last_markov_context_, features.begin() + kHiddenWidth);
    std::array<float, outputs> affine{};
    std::array<float, rank> latent{};
    std::vector<float> correction(kOutputVocabulary);
    auto prev = request.ids.front();
    proposals.clear();
    confidence.clear();
    for (unsigned proposal = 0; proposal < request.k; ++proposal) {
        auto embed = std::span(recurrent).last(rank);
        check(weights_.read_floats("markov_head.markov_w1.weight", embed, std::size_t{prev} * rank));
        std::ranges::copy_n(normalized.begin() + (proposal + 1U) * kHiddenWidth, kHiddenWidth, features.begin());
        for (unsigned i = 0; i < outputs; ++i) {
            const auto row = std::span(matrix).subspan(i * affine_width, affine_width);
            affine[i] = (bias[i] + core::dot_f32(row.last(features.size()), features)) +
                        core::dot_f32(row.first(recurrent_inputs), recurrent);
        }
        core::bf16_markov_gate_reference(affine, std::span(recurrent).first(rank), latent);
        core::linear_reference(folded_head_, latent, correction);
        auto scores = std::span(logits).subspan(proposal * kOutputVocabulary, kOutputVocabulary);
        for (unsigned i = 0; i < kOutputVocabulary; ++i) correction[i] += folded_bias_[i];
        core::round_bf16_reference(correction);
        for (unsigned i = 0; i < kOutputVocabulary; ++i) scores[i] += correction[i];
        core::round_bf16_reference(scores);
        for (auto& score : scores) score = kLogitSoftcap * std::tanh(score / kLogitSoftcap);
        core::round_bf16_reference(scores);
        prev = select_greedy_token(scores, kOutputVocabulary);
        proposals.push_back(prev);
    }
}

}  // namespace strix::inkling2
