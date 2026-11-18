#pragma once
// Hy4-flash host reference (the CPU oracle every backend is compared with): one decode step at a time over the text
// model, every tensor read by its GGUF name (the names of the shared naming map, src/weight hy_v4_flash table).
// Package notes and the contract: docs/models/hy4-flash/README.md.
//
// Arithmetic contract hy4-host-ref-v1 (what EXACT means for this oracle):
//   * weights: every tensor decoded to its exact stored value by the shared decoders (ggml_dequant for Q8_0 / Q4_K /
//     Q6_K / BF16 / F32, the shared E4M3 codec for the Engram table), widened to FP32; no requantisation;
//   * activations: BF16 at the reference model's rounding points (every Linear output, every norm output with the
//     model's double rounding w * bf16(x * r), the four residual streams after every update); FP32 elsewhere (router,
//     hyper-connection mixes, indexer pooling and scores, softmax, lm_head);
//   * reductions: the shared declared orders (ds41_layer.hpp chunked_dot_of: lane-strided chunks of 640, ascending),
//     small sums ascending by index, exp / sqrt / sigmoid by the declared functions (ds41_moe.hpp); no fused
//     multiply-add (build with contraction off);
//   * ties: top-k and argmax take the lower index.
// A run reproduces its logits bit for bit on every host. A backend identity is EXACT when it matches these bits;
// an identity that changes activation precision (A8 / FP8 activations, fused or reordered sums) is a declared FAST
// class compared by logits distance and token agreement, never called exact.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strix::hy4 {

struct Config {
    std::uint32_t hidden = 3072, vocab = 120832, layers = 42;
    float rms_eps = 1e-5F, hc_eps = 1e-6F;
    float engram_norm_eps = 0.0078125F;  // nn.RMSNorm(eps=None) on BF16 = finfo(bf16).eps (forward-pass Q2)
    std::uint32_t q_lora = 1024, kv_lora = 512, qk_nope = 192, qk_rope = 64, v_head = 256;
    float rope_theta_indexed = 1e7F, rope_theta_sliding = 1e4F;
    std::uint32_t window = 512;  // keys q-window+1 .. q (the query included)
    std::uint32_t index_heads = 16, index_dim = 128, index_ratio = 4, index_groups = 512;  // groups incl. the own group
    std::uint32_t experts = 512, top_k = 16, latent = 1536, expert_ff = 1536, shared_ff = 1536, dense_ff = 13824;
    float routed_scale = 3.998F, swiglu_limit = 10.0F;
    std::vector<std::uint8_t> indexed;  // per layer: compressed-indexer (DSA) attention, else sliding window
    std::vector<std::uint32_t> heads;   // per layer attention heads
    std::vector<std::uint8_t> dense;    // per layer: dense SwiGLU FFN, else LatentMoE
    // Engram
    std::vector<std::uint32_t> engram_layers;  // layer ids (0-based)
    std::uint32_t engram_dim = 96, ngram = 3, heads_per_ngram = 8;
    std::vector<std::int64_t> engram_mult;     // [engram layer][window slot]
    std::vector<std::uint64_t> head_offsets;   // [engram layer][head] rows into the one table
    std::vector<std::uint64_t> head_sizes;     // [engram layer][head] prime moduli
    std::vector<std::int32_t> token_map;
    std::uint32_t pad_token = 120002;

    [[nodiscard]] std::uint32_t engram_heads() const { return (ngram - 1U) * heads_per_ngram; }
    [[nodiscard]] std::uint32_t qk_head() const { return qk_nope + qk_rope; }
    // Structure checks (array lengths, divisibility); throws std::invalid_argument.
    void validate() const;
    // The Hy4-flash pack: scalars and arrays from its GGUF metadata (hy_v4_flash.*); reads the header only.
    [[nodiscard]] static Config from_gguf(const std::filesystem::path& first_part);
    // A small configuration with the real structure: layer 0 indexed + dense, layer 1 sliding + MoE + Engram,
    // layer 2 indexed + MoE + Engram.
    [[nodiscard]] static Config tiny();
    // The same model cut to its first `n` trunk layers (1 <= n <= layers): per-layer arrays cut, Engram layers >= n
    // dropped with their multipliers / head offsets / moduli; the output head follows layer n - 1. A truncated
    // model is a complete model: the depth-N compare of a backend against this oracle runs on it.
    [[nodiscard]] Config truncated(std::uint32_t n) const;
};

// Decoded tensors by GGUF name, row-major in GGUF order: a tensor of extents [ne0, ne1, ...] is ne1*... rows of ne0.
class Weights {
public:
    virtual ~Weights() = default;
    [[nodiscard]] virtual const std::vector<float>& tensor(const std::string& name) = 0;
    // One row of a large table (the Engram table) without decoding the whole tensor.
    [[nodiscard]] virtual std::vector<float> row(const std::string& name, std::uint64_t row) = 0;
};

// Random weights with the pack's names and shapes for `config` (deterministic in seed). Tensors whose name contains
// `reseed_filter` (non-empty) are drawn from seed + 1: a way to perturb one component in tests.
[[nodiscard]] std::unique_ptr<Weights> synthetic_weights(const Config& config, std::uint64_t seed,
                                                         const std::string& reseed_filter = {});
// The pack's tensors (all parts of a split GGUF), decoded on first use by the shared decoders.
[[nodiscard]] std::unique_ptr<Weights> pack_weights(const std::vector<std::filesystem::path>& parts);

// Router of a LatentMoE layer: selection key = sigmoid(logit) + bias, top_k by key (lower id on ties); weights =
// unbiased scores of the selection / (their sum in ascending expert id + 1e-20) * scale. Ids ascending.
struct Route {
    std::vector<std::uint32_t> ids;
    std::vector<float> weights;
};
[[nodiscard]] Route route(const std::vector<float>& logits, const std::vector<float>& bias, std::uint32_t top_k, float scale);

// Host threads for the oracle's large row loops (default 1). Each row is one independent declared dot, so the thread
// count never changes a bit; it only bounds the oracle's wall time on the real pack. A loop is split only when
// rows x width >= min_work (tests pass 1 to split the tiny model's loops).
void set_ref_threads(unsigned threads, std::size_t min_work = std::size_t{1} << 20) noexcept;

// Engram table rows of one token for Engram layer ordinal `ordinal`: window = (tok[q], tok[q-1], tok[q-2]) with the
// pad id before the sequence start; heads order-2 first, then order-3.
[[nodiscard]] std::vector<std::uint64_t> engram_rows(const Config& config, std::uint32_t ordinal,
                                                     const std::vector<std::uint32_t>& window);

class HostRef {
public:
    HostRef(Config config, Weights& weights);
    ~HostRef();
    HostRef(const HostRef&) = delete;
    HostRef& operator=(const HostRef&) = delete;
    // Feeds one token at the next position; returns the FP32 logits of the next token.
    const std::vector<float>& step(std::uint32_t token);
    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] std::uint32_t next_token() const noexcept;  // argmax of the last logits, lower index on ties
    [[nodiscard]] const Config& config() const noexcept;
    // Read-only taps for a drafter that shares this model's state (DFly2 KV Share + last-layer hidden); no bit change.
    // The latent row of `layer` at `position`: [kv_lora (kv_a_norm output) | qk_rope (after RoPE)], as cached.
    [[nodiscard]] const std::vector<float>& latent_row(std::uint32_t layer, std::uint32_t position) const;
    // The last step's hc_head output (the one stream before output_norm) or, when `normed`, the output_norm output.
    [[nodiscard]] const std::vector<float>& last_hidden(bool normed) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace strix::hy4
