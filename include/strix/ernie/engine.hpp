#pragma once

#include "strix/ernie/inventory.hpp"
#include "strix/moe/route_policy.hpp"

#include <functional>

namespace strix::ernie {

// Numerics of the released checkpoint. The default value of each field is the plain reference form used
// by the unit fixtures; the checkpoint contract sets all of them.
struct HostContracts {
    bool norm_routed_output = false;
    bool checkpoint_attention = false;
    bool checkpoint_hc = false;
    bool checkpoint_stores = false;
};

inline constexpr HostContracts kCheckpointContracts{true, true, true, true};

struct ReferenceShape {
    unsigned hidden, vocab, dense_width, latent_width, expert_width;
    unsigned heads, head_dim, rope_dim, q_rank, o_groups, o_rank, vha_rank;
    moe::RoutePolicy route;
    HostContracts contracts{};
};

[[nodiscard]] ReferenceShape checkpoint_reference_shape();

using ReferenceRead = std::function<void(std::string_view name, std::span<float> values, std::size_t first_element)>;
using ReferenceActivation = float (*)(float gate, float up, float beta, float linear_beta);
using ReferenceProject = std::function<void(const std::string& name, std::span<const float> input,
                                             std::span<float> output, unsigned groups)>;
using ExpertObserver = std::function<void(unsigned layer, unsigned expert, std::span<const float> value,
                                          bool down)>;
using AttentionRow = std::function<std::vector<float>(std::span<const float>)>;

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
    FirstRowReference(ReferenceShape shape, ReferenceRead read, ReferenceActivation activation,
                      ReferenceProject project, ExpertObserver observe_expert);

    [[nodiscard]] LayerResult block(unsigned layer, std::span<const float> streams) const;
    [[nodiscard]] LayerResult block(unsigned layer, std::span<const float> streams,
                                    const AttentionRow& attend) const;
    [[nodiscard]] std::vector<float> embed(std::uint32_t token) const;
    [[nodiscard]] std::vector<float> head_input(std::span<const float> streams) const;
    [[nodiscard]] std::vector<float> raw_logits(std::span<const float> streams) const;
    [[nodiscard]] std::vector<float> embedding_first_block_logits(std::uint32_t token) const;

private:
    [[nodiscard]] std::vector<float> read(const std::string& name, std::size_t count, std::size_t first = 0) const;
    void project(const std::string& name, std::span<const float> x, std::span<float> y) const;
    [[nodiscard]] std::vector<float> attention(const std::string& prefix, std::span<const float> input) const;
    void feed_forward(unsigned layer, const std::string& prefix, std::span<const float> input, LayerResult& out) const;

    ReferenceShape shape_;
    ReferenceRead read_;
    ReferenceActivation activation_;
    ReferenceProject project_;
    ExpertObserver observe_expert_;
};

} // namespace strix::ernie
