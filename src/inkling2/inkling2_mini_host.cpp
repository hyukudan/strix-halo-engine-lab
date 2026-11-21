#include "inkling2_mini_host.hpp"
#include "inkling2_mini_expert_layout.hpp"

#include "attn_core/causal_gqa_reference.hpp"
#include "core/linear_microscaled_reference.hpp"
#include "core/linear_reference.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_forward.hpp"
#include "inkling2_mini_moe_reference.hpp"
#include "inkling2_mini_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string_view>

namespace strix::inkling2 {
namespace {

void require_checkpoint(const weight::Status& status) {
    if (!status) throw std::runtime_error(status.detail);
}

std::string text_parameter(std::string_view suffix) {
    return "model.llm." + std::string(suffix);
}

using VisitStorage = session::GrowingVisitCache<float, MiniVisitStatePlan::stream_count>;

}  // namespace

struct MiniHostDecoder::Impl {
    Impl(const std::string& path, unsigned capacity) : max_positions(capacity) {
        if (capacity == 0 || capacity > kMaxContext) throw std::invalid_argument("invalid Mini context capacity");
        require_checkpoint(weights.open(path));
    }

    std::vector<float> parameter(std::string_view name, std::size_t elements, std::size_t first = 0) const {
        std::vector<float> result(elements);
        require_checkpoint(weights.read(text_parameter(name), result, first));
        return result;
    }

    std::vector<float> attention(LayerVisit visit, unsigned position, std::span<const float> q,
                                 std::span<const float> k, std::span<const float> v) {
        auto& state = storage[visit.logical_layer];
        const auto plan = visit_state_plan(visit, max_positions, 1);
        const bool full = is_full_attention(visit.physical_layer);
        const unsigned window = plan.kv.ring;

        state.prepare(plan, position);

        const auto layer = visit.physical_layer;
        const auto query_gain = parameter(mini_layer_parameter(layer, "attn.q_l2_clip.scale"), kQueryWidth);
        const auto relative_gain =
            parameter(mini_layer_parameter(layer, "attn.rel_l2_clip.scale"), kQueryHeads * kRelativeWidth);
        auto key_conv = parameter(mini_layer_parameter(layer, full ? "attn.k_sconv.static_w" : "attn.k_sconv.weight"),
                                  kConvTaps * kKvWidth);
        auto value_conv = parameter(mini_layer_parameter(layer, full ? "attn.v_sconv.static_w" : "attn.v_sconv.weight"),
                                    kConvTaps * kKvWidth);
        if (!full) {
            key_conv = mini_short_conv_weights(key_conv, kKvWidth);
            value_conv = mini_short_conv_weights(value_conv, kKvWidth);
        }
        constexpr auto relative = mini_attention_projection_spec(MiniAttentionProjection::relative_bias);
        const auto relative_source = parameter(mini_layer_parameter(layer, relative.tensor_suffix),
                                               kRelativeWidth * kRelativeExtent);
        std::array<float, kRelativeWidth * kRelativeExtent> relative_table;
        for (std::size_t i = 0; i < kRelativeWidth; ++i) {
            for (std::size_t lag = 0; lag < kRelativeExtent; ++lag) {
                relative_table[lag * kRelativeWidth + i] = relative_source[i * kRelativeExtent + lag];
            }
        }
        const auto project = [&](MiniAttentionProjection kind, std::span<const float> input, std::span<float> output) {
            const auto spec = mini_attention_projection_spec(kind);
            if (input.size() != spec.input_width || output.size() != spec.output_width) {
                throw std::invalid_argument("invalid Inkling 2 Mini projection operands");
            }
            if (spec.transpose_for_rows) {
                core::linear_reference(relative_table, input, output);
            } else {
                const auto matrix = parameter(mini_layer_parameter(layer, spec.tensor_suffix),
                                              input.size() * output.size());
                core::linear_reference(matrix, input, output);
            }
        };

        const auto inputs = prepare_mini_attention(visit, position, q, k, v,
            {query_gain, relative_gain, key_conv, value_conv}, state.history[0], state.history[1], project);
        std::array<float, kQueryWidth> attended{};
        attn_core::causal_gqa_reference(
            {kQueryHeads, kKvHeads, kHeadWidth, window, attn_core::ReferenceAttentionArithmetic::bf16_eager},
            {position, inputs.query, inputs.key, inputs.value, kRelativeExtent, inputs.relative_logits, inputs.sinks},
            {state.keys, state.values}, attended);
        std::vector<float> hidden(kHiddenWidth);
        project_mini_attention_output(attended, inputs.output_gate, project, hidden);
        return hidden;
    }

    std::vector<float> convolve(LayerVisit visit, bool after_moe, std::span<const float> input) {
        auto& state = storage[visit.logical_layer];
        constexpr core::ShortConvolution plan{kHiddenWidth, kConvTaps, 0U, 1.0F, 2.2360679774997898F};
        const auto suffix = after_moe ? "mlp_sconv.weight" : "attn_sconv.weight";
        const auto weights_by_lag = mini_short_conv_weights(
            parameter(mini_layer_parameter(visit.physical_layer, suffix), kConvTaps * kHiddenWidth), kHiddenWidth);
        std::vector<float> output(kHiddenWidth);
        core::causal_short_convolution(plan, input, weights_by_lag, {}, state.history[after_moe ? 3U : 2U], output);
        mini_round_bf16(output);
        return output;
    }

    std::vector<float> moe(LayerVisit visit, std::span<const float> input) const {
        const auto layer = visit.physical_layer;
        const auto router = parameter(mini_layer_parameter(layer, "mlp.gate.weight"), kExperts * kHiddenWidth);
        const auto bias = parameter(mini_layer_parameter(layer, "mlp.gate.bias"), kExperts);
        const auto project = [&](unsigned expert, strix::moe::ExpertProjection kind,
                                 std::span<const float> row, std::span<float> output) {
            const auto projection = kind == strix::moe::ExpertProjection::gate_up
                ? MiniExpertProjection::gate_up : MiniExpertProjection::down;
            weight::microscaled::MatrixView view;
            require_checkpoint(weights.expert_matrix(layer, expert, projection, view));
            weight::microscaled::StagedMatrix matrix;
            require_checkpoint(matrix.load(view, mini_expert_row_order(projection)));
            using weight::microscaled::MatrixFormat;
            if (matrix.format() == MatrixFormat::nvfp4) {
                core::microscaled_matvec_reference<MatrixFormat::nvfp4>(matrix.packed(), row, output);
            } else {
                core::microscaled_matvec_reference<MatrixFormat::mxfp8>(matrix.packed(), row, output);
            }
        };
        std::vector<float> output(kHiddenWidth);
        mini_moe_reference(input, router, bias, project, output);
        return output;
    }

    std::vector<float> logits(std::span<const float> hidden) const {
        constexpr std::size_t rows_per_read = 128;
        std::vector<float> output(kOutputVocabulary);
        for (std::size_t first = 0; first < output.size(); first += rows_per_read) {
            const auto rows = std::min(rows_per_read, output.size() - first);
            const auto matrix = parameter("unembed.weight", rows * kHiddenWidth, first * kHiddenWidth);
            core::linear_reference(matrix, hidden, std::span<float>(output).subspan(first, rows));
        }
        mini_round_bf16(output);
        for (float& value : output) {
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite Inkling 2 Mini logits");
            value = kLogitSoftcap * std::tanh(value / kLogitSoftcap);
        }
        return output;
    }

    std::vector<float> step(unsigned token, bool want_logits) {
        if (failed) throw std::runtime_error("Inkling 2 Mini session failed during a prior step");
        if (frontier >= max_positions || token >= kVocabulary) {
            throw std::out_of_range("Inkling 2 Mini session capacity or token id exceeded");
        }

        failed = true;
        const auto embedding = parameter("embed.weight", kHiddenWidth, std::size_t{token} * kHiddenWidth);
        const auto normalized = mini_rms_norm(embedding, parameter("embed_norm.weight", kHiddenWidth));
        const auto hidden = mini_forward_row(*this, frontier, normalized);
        auto result = want_logits ? logits(hidden) : std::vector<float>{};
        ++frontier;
        failed = false;
        return result;
    }

    MiniTextWeights weights;
    unsigned max_positions;
    unsigned frontier = 0;
    bool failed = false;
    std::array<VisitStorage, kLogicalVisits> storage;
};

MiniHostDecoder::MiniHostDecoder(const std::string& path, unsigned max_positions)
    : impl_(std::make_unique<Impl>(path, max_positions)) {}
MiniHostDecoder::~MiniHostDecoder() = default;

std::vector<float> MiniHostDecoder::step(unsigned token, bool want_logits) {
    return impl_->step(token, want_logits);
}

}  // namespace strix::inkling2
