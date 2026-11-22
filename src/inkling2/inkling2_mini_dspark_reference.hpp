#pragma once

#include "inkling2_mini_dspark_checkpoint.hpp"
#include "strix/spec/block_parallel.hpp"

#include <functional>

namespace strix::inkling2 {

// CPU oracle. Tap rows are supplied only after the target publishes them.
// The target owns its embedding and unembedding weights.
class MiniDSparkReference final : public spec::BlockForward {
public:
    using ReadTarget = std::function<void(std::string_view, std::span<float>, std::size_t)>;

    MiniDSparkReference(const std::string& sidecar, ReadTarget read_target);
    static spec::BlockPolicy policy();
    // Flat [position][tap][hidden]; appends contiguous committed positions.
    void append_context(std::span<const float> taps);
    void forward(const spec::BlockRequest&, std::vector<std::uint32_t>& proposals,
                 std::vector<float>& confidence) override;
    void accept(std::span<const std::uint32_t>) override {}  // tap publication is owned by the target

private:
    struct Context {
        std::vector<float> keys, values;
    };

    std::vector<float> read(std::string_view, std::size_t count, std::size_t first = 0) const;
    void run_layer(unsigned layer, std::uint64_t anchor, std::span<float> hidden, std::span<float> normalized) const;
    std::vector<float> readout(std::span<const float> hidden, unsigned proposals);
    void fold_head_tile(std::size_t first, std::span<const float> lm_head,
                        std::span<const float> weight, std::span<const float> bias);

    weight::TensorLookup weights_;
    ReadTarget read_target_;
    std::array<Context, MiniDSparkCheckpoint::layers> context_;
    std::array<float, 16> fusion_{};
    std::vector<float> last_markov_context_;
    std::vector<float> folded_head_, folded_bias_;
    std::uint64_t positions_ = 0;
    bool folded_ = false;
};

}  // namespace strix::inkling2
