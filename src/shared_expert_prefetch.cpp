#include "strix/shared_expert_prefetch.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
namespace strix::shared_expert_prefetch {
namespace {
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a)
        throw std::invalid_argument("shared prefetch geometry overflow");
    return a * b;
}
int slot(const ManifestTensor &t, const ModelContract &m) {
    if (t.role != TensorRole::dense_projection || t.dimensions.size() != 2 ||
        !t.name.ends_with(".weight"))
        return -1;
    if (t.dimensions[0] == m.shared_expert_intermediate_size && t.dimensions[1] == m.hidden_size &&
        t.name.find("down") != t.name.npos)
        return 2;
    if (t.dimensions[0] == m.hidden_size && t.dimensions[1] == m.shared_expert_intermediate_size) {
        if (t.name.find("up") != t.name.npos)
            return 1;
        if (t.name.find("gate") != t.name.npos)
            return 0;
    }
    return -1;
}
} // namespace
bool eligible(const ManifestTensor &t) noexcept {
    return t.ggml_type == 8U && t.dimensions.size() == 2U && t.dimensions[0] > 0 &&
           t.dimensions[0] % 32U == 0 && t.dimensions[1] > 0;
}
std::uint64_t required_weight_bytes(std::uint64_t width, std::uint64_t outputs) {
    if (width == 0 || width % 32U != 0 || outputs == 0)
        throw std::invalid_argument("shared prefetch Q8 geometry invalid");
    return product(product(width / 32U, outputs), 34U);
}
void validate_layers(std::uint64_t hidden_layers, bool enabled,
                     std::span<const std::uint32_t> layers) {
    if (!enabled && !layers.empty())
        throw std::invalid_argument("shared prefetch layers require opt-in");
    if (enabled && layers.empty())
        throw std::invalid_argument("shared prefetch requires eligible step");
    std::optional<std::uint32_t> previous;
    for (auto layer : layers) {
        if (layer >= hidden_layers || (previous && layer <= *previous))
            throw std::invalid_argument("shared prefetch eligible layers invalid");
        previous = layer;
    }
}
bool selected(bool enabled, std::span<const std::uint32_t> layers,
              std::optional<std::uint32_t> layer) noexcept {
    return enabled && layer && std::binary_search(layers.begin(), layers.end(), *layer);
}
Plan build_plan(const ModelContract &m, const ModelManifest &manifest) {
    if (!m.validate().empty() || m.hidden_layers > UINT32_MAX)
        throw std::invalid_argument("shared prefetch model invalid");
    Plan p;
    for (std::uint32_t layer = 0; layer < m.hidden_layers; ++layer) {
        std::array<const ManifestTensor *, 3> tensors{};
        for (const auto &t : manifest.tensors) {
            if (t.layer != layer)
                continue;
            const auto s = slot(t, m);
            if (s < 0)
                continue;
            auto &found = tensors[static_cast<unsigned>(s)];
            if (found)
                throw std::invalid_argument("shared prefetch ambiguous projection");
            found = &t;
        }
        bool any = false;
        for (const auto *t : tensors) {
            if (!t)
                throw std::invalid_argument("shared prefetch missing projection family");
            const bool use = eligible(*t);
            if (use && (t->absolute_offset % 16U != 0 ||
                        t->byte_size < required_weight_bytes(t->dimensions[0], t->dimensions[1])))
                throw std::invalid_argument("shared prefetch weight alignment/extent invalid");
            p.projections.push_back({t->name, layer, use});
            if (use) {
                any = true;
                ++p.eligible_projections;
            } else
                ++p.fallback_projections;
        }
        if (any) {
            p.layers.push_back(layer);
            ++p.eligible_steps;
        } else
            ++p.fallback_steps;
    }
    validate_layers(m.hidden_layers, true, p.layers);
    return p;
}
void validate_tensor_access(const Plan &plan, const ModelManifest &manifest,
                            const SerialExecutionPlan &execution,
                            const ExecutionTensorAccessPlan &accesses) {
    for (const auto &projection : plan.projections) {
        const auto step =
            std::find_if(execution.steps.begin(), execution.steps.end(), [&](const auto &s) {
                return s.op == SerialOp::shared_expert && s.layer == projection.layer;
            });
        const auto tensor =
            std::find_if(manifest.tensors.begin(), manifest.tensors.end(), [&](const auto &t) {
                return t.name == projection.tensor_name && t.layer == projection.layer;
            });
        if (step == execution.steps.end() || tensor == manifest.tensors.end())
            throw std::invalid_argument("shared prefetch projection access missing");
        const auto *access = find_execution_tensor_access(accesses, step->id, 0);
        const auto index = static_cast<std::uint32_t>(tensor - manifest.tensors.begin());
        if (!access || access->op != SerialOp::shared_expert ||
            std::count(access->tensor_indices.begin(), access->tensor_indices.end(), index) != 1)
            throw std::invalid_argument("shared prefetch projection differs from tensor access");
    }
}
} // namespace strix::shared_expert_prefetch
