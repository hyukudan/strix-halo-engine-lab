#include "strix/ernie/forward.hpp"

#include "strix/moe/activation_policy.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strix::ernie {
namespace {
struct ActivationMath {
    static float exp(float value) { return std::exp(value); }
    static float tanh(float value) { return std::tanh(value); }
};

float swiglu(float gate, float up) {
    return (gate * (1.0F / (1.0F + std::exp(-gate)))) * up;
}
} // namespace

ReferenceActivation candidate_activation(CandidateActivation activation) {
    switch (activation) {
    case CandidateActivation::swiglu:
        return [](float gate, float up, float, float) {
            return swiglu(gate, up);
        };
    case CandidateActivation::soft_clip:
        return [](float gate, float up, float beta, float linear_beta) {
            return moe::SoftcappedSwiglu<ActivationMath>{beta, linear_beta}(gate, up);
        };
    case CandidateActivation::hard_clip:
        return [](float gate, float up, float beta, float linear_beta) {
            return swiglu(std::min(gate, beta), std::clamp(up, -linear_beta, linear_beta));
        };
    }
    throw std::invalid_argument("ERNIE: invalid candidate activation");
}

CpuReference::CpuReference(ReferenceShape shape, ReferenceRead read, ReferenceActivation activation,
                           ReferenceProject project, ExpertObserver expert_observer,
                           AttentionHeadExecutor execute_heads)
    : shape_(shape), read_(std::move(read)), project_(std::move(project)),
      execute_heads_(std::move(execute_heads)),
      block_(shape_, read_, activation, project_, std::move(expert_observer)) {}

NextTokenScore score_next_token(std::span<const float> logits, std::uint32_t target) {
    if (target >= logits.size()) throw std::out_of_range("ERNIE: target outside vocabulary");
    if (!std::ranges::all_of(logits, [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("ERNIE: nonfinite logits");
    const auto best = std::ranges::max_element(logits);
    double partition = 0;
    for (float value : logits) partition += std::exp(double{value} - *best);
    return {std::log(partition) + (double{*best} - logits[target]),
            static_cast<std::uint32_t>(best - logits.begin())};
}

ReferenceRows CpuReference::embed_rows(std::span<const std::uint32_t> tokens) const {
    if (tokens.empty() || tokens.size() > data::dims::max_context)
        throw std::invalid_argument("ERNIE: invalid prefill token count");
    ReferenceRows rows;
    rows.reserve(tokens.size());
    for (const auto token : tokens) rows.push_back(block_.embed(token));
    return rows;
}

void CpuReference::run_layer(ReferenceRows& rows, unsigned layer, AttentionHistory& history,
                             std::size_t position, const LayerObserver& observe) const {
    for (std::size_t row = 0; row < rows.size(); ++row) {
        auto result = block_.block(layer, rows[row], [&](auto input) {
            return attention_reference(layer, shape_, read_, input, history, project_, execute_heads_);
        });
        if (observe) observe(position + row, layer, result);
        rows[row] = std::move(result.streams);
    }
}

ReferenceRows CpuReference::prefill(ReferenceRows rows, unsigned first_layer, unsigned end_layer,
                                    const LayerBoundaryObserver& boundary) const {
    if (first_layer > end_layer || end_layer > kLayerCount || rows.empty() ||
        rows.size() > data::dims::max_context ||
        std::ranges::any_of(rows, [&](const auto& row) {
            return row.size() != std::size_t{shape_.hidden} * data::extras::hc_mult;
        }))
        throw std::invalid_argument("ERNIE: invalid prefill boundary");
    if (boundary) boundary(first_layer, rows);
    for (auto layer = first_layer; layer < end_layer; ++layer) {
        AttentionHistory history;
        run_layer(rows, layer, history, 0, {});
        if (boundary) boundary(layer + 1, rows);
    }
    return rows;
}

std::vector<float> CpuReference::head_input(std::span<const float> row) const {
    return block_.head_input(row);
}

std::vector<float> CpuReference::append(std::span<const std::uint32_t> tokens, const LayerObserver& observe) {
    if (failed_) throw std::logic_error("ERNIE: discard the sequence after a failed forward");
    if (tokens.empty() || tokens.size() > static_cast<std::size_t>(data::dims::max_context) - position_)
        throw std::invalid_argument("ERNIE: invalid token count for the sequence");
    if (std::ranges::any_of(tokens, [&](auto id) { return id >= shape_.vocab; }))
        throw std::out_of_range("ERNIE: token outside reference vocabulary");
    try {
        auto rows = embed_rows(tokens);
        for (unsigned layer = 0; layer < kLayerCount; ++layer)
            run_layer(rows, layer, history_[layer], position_, observe);
        auto logits = block_.raw_logits(rows.back());
        position_ += tokens.size();
        return logits;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

} // namespace strix::ernie
