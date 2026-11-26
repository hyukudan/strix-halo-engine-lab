#pragma once

#include "strix/ernie/inventory.hpp"
#include "strix/moe/route_policy.hpp"

#include <functional>

namespace strix::ernie {

struct ReferenceShape {
    unsigned hidden, vocab, dense_width, latent_width, expert_width;
    unsigned heads, head_dim, rope_dim, q_rank, o_groups, o_rank, vha_rank;
    moe::RoutePolicy route;
};

[[nodiscard]] ReferenceShape checkpoint_reference_shape();

using ReferenceRead = std::function<void(std::string_view name, std::span<float> values, std::size_t first_element)>;
using ReferenceActivation = float (*)(float gate, float up, float beta, float linear_beta);

struct LayerResult {
    std::vector<float> attention_streams;
    std::vector<float> ffn;
    std::vector<float> streams;
    std::vector<std::uint32_t> experts;
    std::vector<float> expert_weights;
};

// Position-zero diagnostic. Uses the window's current entry, a zero-value sink,
// residual VHA (V contracts, U expands), gate after grouped O, and two sigmoid
// score planes (gate.weight selects, gate.weight_1 mixes). The caller supplies
// the GLU arithmetic. These boundaries need a checkpoint oracle before serving.
class FirstRowReference {
public:
    FirstRowReference(ReferenceShape shape, ReferenceRead read, ReferenceActivation activation);

    [[nodiscard]] LayerResult block(unsigned layer, std::span<const float> streams) const;
    [[nodiscard]] std::vector<float> embedding_first_block_logits(std::uint32_t token) const;

private:
    [[nodiscard]] std::vector<float> read(const std::string& name, std::size_t count, std::size_t first = 0) const;
    void project(const std::string& name, std::span<const float> x, std::span<float> y) const;
    [[nodiscard]] std::vector<float> attention(const std::string& prefix, std::span<const float> input) const;
    void feed_forward(const std::string& prefix, bool routed, std::span<const float> input, LayerResult& out) const;

    ReferenceShape shape_;
    ReferenceRead read_;
    ReferenceActivation activation_;
};

} // namespace strix::ernie
