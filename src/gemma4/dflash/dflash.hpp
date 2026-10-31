#pragma once

// Nemotron Bolt E4B DFlash: host reference in the declared arithmetic "g4-dflash-v1.1".
// Semantics: NVIDIA Model-Optimizer DFlash plugin and the official drafter configuration.
// This is an independent arithmetic reference, pending device and full-workload qualification.
//
// - Five pre-norm residual layers: width 2560, 32 Q / 8 KV heads of 128 (GQA 4), FFN 5120.
//   Weighted Q/K RMSNorm, no V norm, full-head NeoX RoPE (i,i+64), theta 1000000;
//   attention scale 128^-0.5; ordinary SiLU(gate)*up SwiGLU, then final weighted RMSNorm.
// - Committed target taps after blocks [5,11,17,23,35], including PLE and layer scalar,
//   concatenate in that order: c = enc.output_norm(fc*taps), fc:12800->2560. Each draft
//   layer projects this same c independently into K/V; c never traverses the residual stack.
// - Context K/V is FP32 in position-tagged rings of 1024 entries per layer. A query q at
//   anchor P sees q-1024 < j < P, then all eight temporary block rows, non-causally.
//   Block inputs [anchor,4,4,4,4,4,4,4] use target embeddings with the target BF16 boundary.
//   Target head on normalized mask rows 1..7 gives d1..d7; anchor row 0 is never a proposal.
// - BF16 matrices expand exactly; FP32 norm vectors retain their stored values and multiply
//   directly (no +1). Activations/residuals/KV stay FP32. Use g4-v1 scalar, norm and dot
//   routines, separate rounded multiply/add, no contraction/reassociation/fast math, eps 1e-6.
//   Immutable caller-supplied FP32 RoPE tables follow the shared frequency/angle recipe;
//   attention partials follow absolute 64-position boundaries, visible keys ascending.
//   SiLU is g/(1+declared_exp(-g)), then multiply by up. For finite g < -88 the
//   declared exponential saturates to +inf: SiLU is -0 (finite/+inf), not an error.
//   At g=-88 retain the finite denominator; g>103 has exp(-g)=+0 and returns g.
//   Nonfinite inputs/results still fail. This extends v1's previously rejected domain;
//   all successful v1 operations retain their order/bits. Draft logits use scale 1/no cap;
//   a draft metadata scale/cap never inherits target selection policy. Ties prefer low IDs.
// - Verify k=0..7 proposals through the target transaction. Accept the longest greedy prefix a,
//   emit [d1..da,g[a]], commit a+1 processed inputs [anchor,d1..da], discard every other row.
//   After EOS/budget truncation to m emitted tokens, commit exactly m processed inputs;
//   the final emitted token remains the next unprocessed anchor. Only committed taps append.
//   k=0 skips the drafter; a full block crossing position 65535 forces k=0.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>
#include "../contracts/g4_types.hpp"

namespace strix::gemma4::dflash {

inline constexpr const char* arithmetic_id = "g4-dflash-v1.1";
inline constexpr std::uint32_t block = 8, max_drafts = 7, window = 1024, tap_count = 5;
inline constexpr std::uint32_t mask_id = 4, position_limit = 65536;
inline constexpr std::array<std::uint32_t, tap_count> tap_layers{5, 11, 17, 23, 35};
inline constexpr float attention_scale = 0.08838834764831845F;

// Explicit negative saturation avoids an infinite intermediate without clipping finite
// gates or changing the division order elsewhere. Nonfinite gates return a failing NaN.
STRIX_G4_HD inline float declared_silu(float gate) {
    if ((gemma4::bit_cast<std::uint32_t>(gate) & 0x7f800000U) == 0x7f800000U)
        return gemma4::bit_cast<float>(0x7fc00000U);
    if (gate < -88.0F) return -0.0F;
    return gate / (1.0F + gemma4::declared_exp(-gate));
}

// Non-default shapes are for small host fixtures, never a production model override.
struct Shape {
    std::uint32_t hidden = 2560, layers = 5, heads = 32, kv_heads = 8, head_dim = 128, inter = 5120;
    [[nodiscard]] std::size_t q_width() const { return std::size_t{heads} * head_dim; }
    [[nodiscard]] std::size_t kv_width() const { return std::size_t{kv_heads} * head_dim; }
    [[nodiscard]] std::size_t tap_width() const { return std::size_t{tap_count} * hidden; }
    void validate() const;
    bool operator==(const Shape&) const = default;
};

struct Bf16Matrix {
    std::uint32_t outputs = 0, width = 0;
    std::vector<std::uint16_t> bits; // [outputs][width], original BF16 coefficients
};
struct LayerWeights {
    Bf16Matrix q, k, v, o, gate, up, down;
    std::vector<float> input_norm, ffn_norm, q_norm, k_norm;
};
struct Weights {
    Shape shape;
    Bf16Matrix fc;
    std::vector<float> context_norm, final_norm;
    std::vector<LayerWeights> layer;
    float logit_scale = 1.0F, logit_cap = 0.0F;
    float checkpoint_embedding_scale = 0.0F; // informational; target owns embeddings
    std::string license, license_name, license_link;
    std::uint64_t tensor_bytes = 0;
    void validate() const;
};

class DraftFile {
public:
    static DraftFile open(const std::filesystem::path& path);
    // Explicit fixture path: shape is validated against the file just as in open().
    static DraftFile open_fixture(const std::filesystem::path& path, const Shape& shape);
    [[nodiscard]] const Weights& weights() const { return weights_; }
private:
    Weights weights_;
};

// Tables are immutable, position-major [positions][head_dim/2], and supplied by the caller.
// Generation/hashing belongs to the shared arithmetic contract, never runtime device libm.
struct RopeView {
    std::span<const float> cosine, sine;
    std::uint32_t positions = 0, head_dim = 128;
    void validate() const;
};
struct RopeFile {
    std::vector<float> cosine, sine;
    std::uint32_t positions = 0, head_dim = 0;
    // G4DR, uint32 version=1, positions, head_dim; cosine then sine, little-endian FP32.
    static RopeFile open(const std::filesystem::path& path);
    [[nodiscard]] RopeView view() const { return {cosine, sine, positions, head_dim}; }
};

class DraftContext {
public:
    explicit DraftContext(Shape shape = {});
    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] std::uint64_t frontier() const { return frontier_; }
    [[nodiscard]] bool contains(std::uint64_t position) const;
    [[nodiscard]] std::span<const float> key(std::uint32_t layer, std::uint64_t position) const;
    [[nodiscard]] std::span<const float> value(std::uint32_t layer, std::uint64_t position) const;
    // Initial long-prefill tail may start after zero. Subsequent appends must be contiguous;
    // duplicates/rollback/gaps are refused. All calculations finish before publishing tags.
    void append(const Weights& weights, RopeView rope, std::span<const float> taps,
                std::span<const std::uint64_t> positions);
private:
    Shape shape_;
    std::vector<float> k_, v_;
    std::array<std::int64_t, window> tags_{};
    std::uint64_t frontier_ = 0;
    bool initialized_ = false;
};

// Adapter to the shared committed-tap interface. Weights, ring and immutable tables
// outlive this adapter; the scheduling owner calls it only after target prefix publication.
class ContextAppender final : public gemma4::DraftContext {
public:
    ContextAppender(const Weights& weights, dflash::DraftContext& context, RopeView rope)
        : weights_(weights), context_(context), rope_(rope) {}
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        context_.append(weights_, rope_, taps, positions);
    }
private:
    const Weights& weights_;
    dflash::DraftContext& context_;
    RopeView rope_;
};

[[nodiscard]] bool context_visible(std::uint64_t query, std::uint64_t anchor, std::uint64_t key);
[[nodiscard]] std::vector<std::uint64_t> visible_positions(std::uint64_t anchor, std::uint32_t row);
// embedded and return value are [8][hidden]. This operation cannot publish context or target KV.
// Optional diagnostics observe finite pre-SiLU gates without changing the arithmetic.
struct GateRange {
    float minimum = 0.0F, maximum = 0.0F;
    std::size_t negative_saturated = 0;
};
struct ForwardDiagnostics {
    std::vector<GateRange> gates; // one entry per draft layer, replaced on each call
};
std::vector<float> block_forward(const Weights& weights, const DraftContext& context, RopeView rope,
                                 std::span<const float> embedded, std::uint64_t anchor_position,
                                 ForwardDiagnostics* diagnostics = nullptr);
// Target-owned table/head views use the common format decoder and dot recipe. Embedding
// scaling is the g4-v1 BF16 boundary; head policy comes only from drafter metadata.
std::vector<float> embed_block(std::uint32_t anchor, LinearView target_embedding);
std::array<std::uint32_t, max_drafts> proposals_from_hidden(const Weights& weights,
    std::span<const float> normalized_block, LinearView target_head);

struct Acceptance {
    std::uint64_t cycles = 0, attempted = 0, eligible = 0, accepted = 0, emitted = 0, all_accepted = 0;
    std::array<std::uint64_t, max_drafts> depth_eligible{}, depth_accepted{}, rejection_depth{};
};
struct Verdict {
    std::uint32_t accepted = 0, processed = 0, eligible = 0;
    bool stopped = false; // EOS encountered; caller also enforces the remaining output budget
    std::vector<std::uint32_t> emitted;
};
// The target's greedy values include its own cap/policies. EOS is included in emitted.
Verdict verify_greedy(std::span<const std::uint32_t> proposals, std::span<const std::uint32_t> greedy,
                      std::size_t budget, std::span<const std::uint32_t> eos = {}, Acceptance* stats = nullptr);
// Pure admission: no measured estimate means k=0. Explicit k still respects OFF, transforms,
// position reservation and budget. Gains must include draft/head/context/commit costs.
std::uint32_t admit_window(std::uint32_t requested, std::uint64_t position, std::size_t budget,
                           bool enabled, bool serial_policy, double conservative_speedup = 0.0,
                           bool automatic = false);
[[nodiscard]] bool enabled_from_environment();

struct CycleResult {
    Verdict verdict;
    std::uint32_t next_anchor = 0;
    std::vector<float> committed_taps;
    std::vector<std::uint64_t> committed_positions;
};
using DraftCallback = std::function<std::array<std::uint32_t, max_drafts>()>;
// One row-invariant target pass, then exactly one prefix commit. The callback runs once
// for a full eight-row draft when admitted k>0, and never for k=0/disabled/empty budget.
// Returned taps contain only committed rows; the owner appends them at its publication
// boundary. No helper can write target live KV or publish rejected target/draft state.
CycleResult run_cycle(Target& target, KvTxn& transaction, std::uint32_t anchor, std::uint32_t k,
    const DraftCallback& draft, std::size_t budget, std::span<const std::uint32_t> eos = {}, Acceptance* stats = nullptr,
    std::uint64_t stream = 0, std::uint64_t generation = 0);

} // namespace strix::gemma4::dflash
