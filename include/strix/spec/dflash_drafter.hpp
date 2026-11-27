#pragma once

// Block-diffusion drafter (the DFlash family: DFlash and DFlash 2): host reference forward.
// Home: shared (strix::spec, host, backend-neutral). The model data is the drafter's own config.json; the target
// supplies its embedding boundary, its head and its committed hidden taps through dflash::Target. No engine, model,
// container or device types. Sources: the drafter's published config.json / README (block diffusion drafting,
// two-tap grouped convolutions, candidate path selector); implemented from that description, no reference code copied.
//
// One pass at anchor position P (block rows r = 0 .. R-1 at positions P + r, ids [anchor, mask, mask, ...]):
//   context (committed target positions j < P, computed once per position):
//     c_j = hidden_norm(fc * concat_t tap_t(j));  per draft layer  K_j = rope_j(k_norm(W_k c_j)),  V_j = W_v c_j
//   x = target_embed(ids) * input_embedding_scale       (the target's embedding boundary, e.g. its embedding norm)
//   per layer:  h = norm_in(x);  [conv] (h, d) = conv_pre(h);  a = W_o attend(h);  [conv] a = conv_post(a, d);  x += a
//               h = norm_post(x); [conv] (h, d) = conv_pre(h); m = W_d(silu(W_g h) * W_u h); [conv] m = conv_post(m, d); x += m
//   y = norm(x); proposal j reads block row j + 1 (the anchor row never proposes).
//   attend: Qwen3 GQA with per-head Q/K RMSNorm and NeoX RoPE; block row r sees the context keys j with
//     P + r - window < j < P (window 0: all) and every block row (non-causal).
//   conv (DFlash 2, conv_kernel_size > 0): d = kernel_projection(h) as [side][tap][group]; side 0 filters h, side 1
//     filters the sub-block output: out_r[c] = sum_{t <= r} (base[side][t][c] + d_r[side][t][c / group]) * in_{r-t}[c].
//   head: without a selector, argmax of the target logits per row. With the selector (DFlash 2): the top-k target
//     logits of each row (unary), edge scores s[l][p][c] = unary[l][c] + sum_q pred[prev(l, p)][q] h_l[q] succ[cand[l][c]][q]
//     (h_l = hidden_projection(y_l); prev(0, p) = anchor, prev(l, p) = cand[l-1][p]), softcap after the correction;
//     the path follows the chosen predecessor: greedy = argmax per step, sampled = a draw from softmax(s / T) per step
//     (its probability is returned: the lossless acceptance rule needs q of the drafted token and q's support).

#include "strix/spec/block_parallel.hpp"
#include "strix/weight/resident_slices.hpp"
#include "strix/weight/tensor_access.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::spec::dflash {

struct Config {
    std::uint32_t hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, layers = 0;
    std::uint32_t block = 0;        // rows of one pass including the anchor
    std::uint32_t window = 0;       // context positions a row sees (0: all)
    std::uint32_t mask_token = 0;   // fill id of the rows after the anchor
    std::uint32_t vocab = 0;        // rows of the selector codebooks (the target's padded table)
    float eps = 1e-6F, rope_theta = 10000.0F, embed_scale = 1.0F;
    std::vector<std::uint32_t> target_layers;  // the target's tap visits, in concatenation order
    std::uint32_t conv_taps = 0, conv_group = 0;          // grouped convolution (0: none)
    std::uint32_t selector_rank = 0, selector_top_k = 0;  // candidate path selector (0: argmax rows)
    float softcap = 0.0F;                                 // selector softcap (0: none)
    bool softcap_after_correction = true;

    // The drafter's config.json (block size, window, taps, conv and selector from its dflash_config).
    static Config from_json_text(std::string_view text);
    [[nodiscard]] std::uint32_t tap_width() const { return hidden * static_cast<std::uint32_t>(target_layers.size()); }
    [[nodiscard]] std::uint32_t conv_groups() const { return conv_group ? hidden / conv_group : 0U; }
};

// Every tensor the forward reads: name -> extents (row-major, the file's order).
[[nodiscard]] std::vector<std::pair<std::string, std::vector<std::uint64_t>>> inventory(const Config& config);
// The drafter directory's config.json.
[[nodiscard]] Config load_config(const std::filesystem::path& dir);
// The checked schema of model.safetensors (every inventory() tensor, BF16), shared by the host and device loaders.
// weight::TensorSchema is a view (a span of specs): this owns the specs; view() is valid while the Schema lives.
struct Schema {
    std::vector<weight::TensorSpec> specs;
    [[nodiscard]] weight::TensorSchema view() const { return {weight::Container::safetensors, "safetensors@1", specs, {}}; }
};
[[nodiscard]] Schema schema(const Config& config);
// model.safetensors of a drafter directory opened against schema(config) (throws with the reader's reason).
[[nodiscard]] weight::TensorLookup open_lookup(const Config& config, const std::filesystem::path& dir);
// The resident slice plan of every inventory() tensor of `lookup` (the device loader's input). The plan copies the names
// it keeps; the inventory it reads them from lives for the whole call.
[[nodiscard]] weight::ResidentSlicePlan resident_plan(const Config& config, const weight::TensorLookup& lookup,
                                                      std::size_t alignment);

// FP32 tensors by file name (BF16 / F32 expanded once at load).
class Weights {
public:
    virtual ~Weights() = default;
    [[nodiscard]] virtual std::span<const float> tensor(const std::string& name) const = 0;
};
// config.json + model.safetensors of a drafter directory; every inventory() tensor is checked (name, extents, dtype).
[[nodiscard]] std::unique_ptr<Weights> load_weights(const Config& config, const std::filesystem::path& dir);
[[nodiscard]] std::unique_ptr<Weights> synthetic_weights(const Config& config, std::uint64_t seed);

// What the drafter reads from the target.
class Target {
public:
    virtual ~Target() = default;
    // The target's input row of `token` (table row and its embedding boundary), config.hidden floats.
    virtual void embed(std::uint32_t token, std::span<float> row) = 0;
    // Valid output vocabulary of the head (ids >= it are never proposed).
    [[nodiscard]] virtual std::uint32_t vocab_out() const = 0;
    // Head logits of `count` rows [count][hidden] -> [count][vocab_out()].
    virtual void head(std::span<const float> rows, std::uint32_t count, std::span<float> logits) = 0;
};

enum class Path : std::uint8_t { greedy, viterbi };

struct Draft {
    std::vector<std::uint32_t> tokens;  // the proposals
    std::vector<float> q;               // drafter probability of each proposal given its predecessor (selector softmax at
                                        // the sampling temperature; greedy: at T = 1); argmax rows: target softmax
    std::vector<std::uint32_t> candidates;  // selector: [proposal][top_k] candidate ids (empty without a selector)
    std::vector<float> scores;          // selector: [proposal][top_k] scores from the chosen predecessor
};

// The proposal distribution q of step `step` of a selector draft (sampled verification: the shared sampler's lossless
// rule, strix::decide_token, accepts d ~ q with min(1, p(d) / q(d)) and otherwise samples normalize(max(0, p - q))):
// softmax(scores / temperature) over the step's candidates from the chosen predecessor. Its support is the step's
// top-k candidates; ids outside it have q = 0. The drafted token's entry equals Draft::q[step]. Empty without a selector.
struct Proposal {
    std::vector<std::uint32_t> tokens;
    std::vector<double> probabilities;
};
[[nodiscard]] Proposal proposal(const Draft& draft, std::uint32_t step, std::uint32_t top_k, double temperature);

class Drafter final : public BlockForward {
public:
    Drafter(Config config, std::unique_ptr<Weights> weights, Target& target, unsigned threads = 0);
    ~Drafter() override;
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] static BlockPolicy policy(const Config& config);
    // Committed tap rows [rows][tap_width] (the target's tap order) of positions positions() .. positions() + rows - 1.
    void append_context(std::span<const float> taps, std::uint32_t rows);
    // Forget positions >= `positions` (a rejected suffix; context rows are per position, so this is exact).
    void truncate(std::uint32_t positions);
    [[nodiscard]] std::uint32_t positions() const noexcept { return positions_; }
    // One pass with the anchor at position positions(): k (<= block - 1) proposals. temperature 0 = greedy.
    void draft(std::uint32_t anchor, std::uint32_t k, Draft& out, Path path = Path::greedy, double temperature = 0.0,
               std::uint64_t seed = 0);

    void forward(const BlockRequest& req, std::vector<std::uint32_t>& proposals, std::vector<float>& confidence) override;
    void accept(std::span<const std::uint32_t>) override {}  // context rows come from the target's committed taps

private:
    struct Impl;
    Config config_;
    std::unique_ptr<Weights> weights_;
    Target& target_;
    std::unique_ptr<Impl> impl_;
    std::uint32_t positions_ = 0;
};

}  // namespace strix::spec::dflash
