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
                           ReferenceProject project, ExpertObserver expert_observer)
    : shape_(shape), read_(std::move(read)), project_(std::move(project)),
      block_(shape_, read_, activation, project_, std::move(expert_observer)) {}

std::vector<float> CpuReference::append(std::span<const std::uint32_t> tokens, const LayerObserver& observe) {
    if (failed_) throw std::logic_error("ERNIE: discard the sequence after a failed forward");
    if (tokens.empty() || tokens.size() > static_cast<std::size_t>(data::dims::max_context) - position_)
        throw std::invalid_argument("ERNIE: invalid token count for the sequence");
    if (std::ranges::any_of(tokens, [&](auto id) { return id >= shape_.vocab; }))
        throw std::out_of_range("ERNIE: token outside reference vocabulary");
    try {
        std::vector<std::vector<float>> rows;
        rows.reserve(tokens.size());
        for (const auto token : tokens) rows.push_back(block_.embed(token));
        for (unsigned layer = 0; layer < kLayerCount; ++layer) {
            for (std::size_t row = 0; row < rows.size(); ++row) {
                auto result = block_.block(layer, rows[row], [&](auto input) {
                    return attention_reference(layer, shape_, read_, input, history_[layer], project_);
                });
                if (observe) observe(position_ + row, layer, result);
                rows[row] = std::move(result.streams);
            }
        }
        auto logits = block_.raw_logits(rows.back());
        position_ += tokens.size();
        return logits;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

} // namespace strix::ernie
