#include "strix/qsa_input_prefetch.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
namespace strix::qsa_input_prefetch {
namespace {
int slot(std::string_view name) {
    if (!name.ends_with(".weight") || name.find("norm") != name.npos)
        return -1;
    if (name.find("indexer.q_proj") != name.npos || name.find("indexer_q_proj") != name.npos)
        return 3;
    if (name.find("indexer.k_proj") != name.npos || name.find("indexer_k_proj") != name.npos)
        return 4;
    if (name.find("attn_output") != name.npos || name.find("o_proj") != name.npos)
        return -1;
    if (name.find("attn_q") != name.npos || name.find("q_proj") != name.npos)
        return 0;
    if (name.find("attn_k") != name.npos || name.find("k_proj") != name.npos)
        return 1;
    if (name.find("attn_v") != name.npos || name.find("v_proj") != name.npos)
        return 2;
    return -1;
}
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a)
        throw std::invalid_argument("QSA prefetch geometry overflow");
    return a * b;
}
} // namespace
bool eligible(const ManifestTensor &t) noexcept {
    return t.ggml_type == 8U && t.dimensions.size() == 2U && t.dimensions[0] > 0 &&
           t.dimensions[0] % 32U == 0 && t.dimensions[1] > 0;
}
std::uint64_t required_weight_bytes(std::uint64_t width, std::uint64_t outputs) {
    if (width == 0 || width % 32U != 0 || outputs == 0)
        throw std::invalid_argument("QSA prefetch Q8 geometry invalid");
    return product(product(width / 32U, outputs), 34U);
}
void validate_layers(const ModelContract &m, bool enabled, std::span<const std::uint32_t> layers) {
    if (!enabled && !layers.empty())
        throw std::invalid_argument("QSA prefetch layers require opt-in");
    if (enabled && layers.empty())
        throw std::invalid_argument("QSA prefetch requires at least one eligible step");
    std::optional<std::uint32_t> previous;
    for (auto layer : layers) {
        if (m.full_attention_interval == 0 || layer >= m.hidden_layers ||
            (static_cast<std::uint64_t>(layer) + 1U) % m.full_attention_interval != 0 ||
            (previous && layer <= *previous))
            throw std::invalid_argument("QSA prefetch eligible layers invalid");
        previous = layer;
    }
}
bool selected(bool enabled, std::span<const std::uint32_t> layers,
              std::optional<std::uint32_t> layer) noexcept {
    return enabled && layer && std::binary_search(layers.begin(), layers.end(), *layer);
}
Plan build_plan(const ModelContract &m, const ModelManifest &manifest) {
    if (!m.validate().empty() || m.hidden_layers > UINT32_MAX)
        throw std::invalid_argument("QSA prefetch model invalid");
    Plan p;
    const auto q = product(product(m.attention_heads, m.head_dim), 2U),
               kv = product(m.kv_heads, m.head_dim),
               iq = product(m.indexer_n_heads, m.indexer_head_dim);
    const std::array outputs{q, kv, kv, iq, m.indexer_head_dim};
    for (std::uint32_t layer = 0; layer < m.hidden_layers; ++layer) {
        if ((static_cast<std::uint64_t>(layer) + 1U) % m.full_attention_interval != 0)
            continue;
        std::array<const ManifestTensor *, 5> tensors{};
        for (const auto &t : manifest.tensors) {
            if (t.layer != layer || t.role == TensorRole::mtp)
                continue;
            const int s = slot(t.name);
            if (s < 0)
                continue;
            auto &found = tensors[static_cast<unsigned>(s)];
            if (found)
                throw std::invalid_argument("QSA prefetch ambiguous projection tensor");
            found = &t;
        }
        bool any = false;
        for (unsigned s = 0; s < 5; ++s) {
            const auto *t = tensors[s];
            if (!t || t->dimensions.size() != 2 || t->dimensions[0] != m.hidden_size ||
                t->dimensions[1] != outputs[s])
                throw std::invalid_argument("QSA prefetch projection geometry missing or invalid");
            const bool use = eligible(*t);
            if (use && (t->absolute_offset % 16U != 0 ||
                        t->byte_size < required_weight_bytes(m.hidden_size, outputs[s])))
                throw std::invalid_argument("QSA prefetch weight alignment/extent invalid");
            p.projections.push_back({t->name, layer, use});
            if (use) {
                ++p.eligible_projections;
                any = true;
            } else
                ++p.fallback_projections;
        }
        if (any) {
            p.layers.push_back(layer);
            ++p.eligible_steps;
        } else
            ++p.fallback_steps;
    }
    validate_layers(m, true, p.layers);
    return p;
}
} // namespace strix::qsa_input_prefetch
