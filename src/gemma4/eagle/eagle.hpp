#pragma once

// Nemotron 3.5 Bolt E4B EAGLE-3-style drafter, engine side.
// Semantics: the drafter training contract v1 (rows_step/ttt). File format: FORMAT.md.
//
// - Row i (position i) consumes taps(i) of target row i and e(t_{i+1}); its depth-1 output predicts t_{i+2}.
//   h = fc(concat_j RMSNorm_j(tap_j)), e = bf16(bf16(E[t]) * 50.5) (target table), x = in_proj([RMSNorm_h(h);
//   RMSNorm_e(e)]) (or the sum). One gemma-style layer: pre-norm, Q/K RMSNorm, unscaled V RMSNorm, NeoX RoPE theta 1e4
//   over all 256 dims (= target sliding tables), 8q/2kv x 256, scale folded into the Q norm, post-attn norm,
//   GeGLU(gelu_tanh), post-FFN norm. logits = 30 tanh(head(RMSNorm_f(s))/30) over the draft vocab; d2t maps to target ids.
// - Committed rows (target-verified) are primed: their K/V (from target taps) enter a 512-slot ring with absolute tags.
//   Accepted drafted positions are therefore recomputed from target taps; chain K/V live only in a per-draft
//   overlay and are discarded. Depth d >= 2 feeds the previous depth's pre-final-norm state and the previous draft.
// - The drafter only proposes. Every emitted token is the target's greedy choice (dflash::verify_greedy), so ids are
//   identical to OFF whenever the target forward is row-invariant (C protocol). Nothing here touches target KV.
// - Arithmetic: g4-v1 scalar recipe (FP32 RNE, separate multiply/add, lane-ordered dots, correctly rounded rsqrt,
//   declared exp/tanh, absolute 64-position attention chunks). Device and host reference are bit-identical by design.

#include "gemma4/contracts/g4_types.hpp"
#include "gemma4/dflash/dflash.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strix::gemma4::eagle {

inline constexpr const char* arithmetic_id = "g4-eagle-v1";
inline constexpr std::uint32_t max_depth = 7, max_rows = 8, engine_taps = 5;
inline constexpr std::int32_t final_tap = -1;
inline constexpr std::uint32_t select_threads = 1024; // fixed reduction tree shared by host and device

struct Config {
    std::uint32_t hidden = 2560, heads = 8, kv_heads = 2, head_dim = 256, ffn = 5120, window = 512, vocab = 65536;
    std::uint32_t target_hidden = 2560, target_vocab = 262144;
    std::vector<std::int32_t> taps{11, 23, final_tap};
    bool concat = true;
    float attention_scale = 0.0625F, softcap = 30.0F, eps = 1e-6F, rope_theta = 10000.0F, embedding_scale = 50.5F;
    [[nodiscard]] std::uint32_t q_width() const { return heads * head_dim; }
    [[nodiscard]] std::uint32_t kv_width() const { return kv_heads * head_dim; }
    [[nodiscard]] std::uint32_t tap_count() const { return std::uint32_t(taps.size()); }
    [[nodiscard]] bool needs_final() const;
    // Engine tap slot of fc input j: 0..4 = taps after blocks 5/11/17/23/35, 5 = final-norm hidden row.
    [[nodiscard]] std::uint32_t tap_slot(std::uint32_t j) const;
    void validate() const;
};

// Owning packed matrix, row-major [outputs][inputs]; view() is valid while this object lives (moves keep data).
struct Matrix {
    std::vector<std::uint8_t> bytes;
    std::uint32_t outputs = 0, inputs = 0;
    WeightFormat format = WeightFormat::q8_0;
    [[nodiscard]] std::size_t row_bytes() const;
    [[nodiscard]] LinearView view() const;
};
[[nodiscard]] std::size_t row_bytes(WeightFormat format, std::uint32_t inputs);
// Packs FP32 rows into f32/bf16/q8_0 (ggml quantize_row_q8_0 rounding: d=amax/127 as fp16, q=round(x/d)).
[[nodiscard]] Matrix pack(std::span<const float> values, std::uint32_t outputs, std::uint32_t inputs, WeightFormat format);

struct Weights {
    Config config;
    std::vector<std::vector<float>> tap_norm;
    Matrix fc, in_proj, q, k, v, o, gate, up, down, head;
    std::vector<float> h_norm, e_norm, input_norm, q_norm, k_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm, final_norm;
    std::vector<std::uint32_t> d2t;
    std::string license, license_name, license_link;
    std::uint64_t tensor_bytes = 0;
    // q_norm * attention_scale; exact because the scale is a power of two (see FORMAT.md).
    [[nodiscard]] std::vector<float> q_norm_effective() const;
    void validate() const;
    static Weights load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const; // fixture/export writer, same layout as load()
    // Deterministic synthetic weights for tests/benchmarks only (never a production file).
    static Weights random(const Config& config, std::uint64_t seed, WeightFormat matrices = WeightFormat::q8_0,
                          float head_std = 0.5F);
};

// Target sliding RoPE table, layout of RopeTables::sliding: [position][128 pairs][cos, sin].
struct RopeView {
    std::span<const float> table;
    std::uint32_t positions = 0;
    void validate() const;
};
// Test/bench helper: libm cos/sin table in that layout (production uses the target's hashed tables).
[[nodiscard]] std::vector<float> make_rope_table(std::uint32_t positions, float theta = 10000.0F);

// Committed target rows, contiguous positions. taps use ForwardResult order [row][5][target_hidden];
// hidden is [row][target_hidden] (RowPlan::export_hidden) and required iff config.needs_final().
// next_tokens[j] = t_{j+1}, the token that follows row j (verdict.emitted for decode rows, prompt[j+1] in prefill).
struct Rows {
    std::span<const float> taps, hidden;
    std::span<const std::uint32_t> next_tokens;
    std::span<const std::uint64_t> positions;
};

struct Draft {
    std::uint32_t depth = 0;
    std::array<std::uint32_t, max_depth> tokens{}, draft_ids{};
    std::array<float, max_depth> prob{}; // drafter top-1 softmax probability per depth (policy/calibration input)
};
struct Timings {
    double stage_ms = 0, publish_ms = 0, draft_ms = 0; // last call of each
    std::uint64_t launches = 0;                         // device kernel launches of the last draft()
    std::uint64_t stage_launches = 0;                   // device kernel launches of the last stage()
};

// Common drafter interface (host reference and device). Single owner, serialized calls.
class Drafter {
public:
    virtual ~Drafter() = default;
    [[nodiscard]] virtual const Config& config() const = 0;
    // First position without committed drafter K/V (= target frontier between cycles).
    [[nodiscard]] virtual std::uint64_t frontier() const = 0;
    [[nodiscard]] virtual const Timings& timings() const = 0;
    virtual void reset() = 0;
    // 1..8 committed rows. The first stage after reset may start at any position (prompt tail); later stages must
    // start at frontier(). Stage computes everything; publish makes it visible; discard drops it (C protocol).
    virtual void stage(const Rows& rows) = 0;
    virtual void publish() = 0;
    virtual void discard() noexcept = 0;
    void prime(const Rows& rows) { stage(rows); publish(); }
    // position == frontier() > 0 and anchor == next token of the last committed row (row position-1). Proposes
    // k in 1..7 target ids for positions position+1..position+k. Chain K/V are temporary; no state changes.
    virtual Draft draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) = 0;
};

// FP32 g4-v1 host reference. Weights/table/embedding views outlive it.
class HostDrafter final : public Drafter {
public:
    HostDrafter(const Weights& weights, RopeView rope, LinearView target_embedding);
    ~HostDrafter() override;
    [[nodiscard]] const Config& config() const override;
    [[nodiscard]] std::uint64_t frontier() const override;
    [[nodiscard]] const Timings& timings() const override;
    void reset() override;
    void stage(const Rows& rows) override;
    void publish() override;
    void discard() noexcept override;
    Draft draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) override;
    // Diagnostics: committed K/V of a position (empty when absent).
    [[nodiscard]] std::vector<float> key(std::uint64_t position) const;
    [[nodiscard]] std::vector<float> value(std::uint64_t position) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Host recipe of the device selection kernel: capped argmax (low draft id on ties) and top-1 probability with
// the fixed 1024-way strided partial sums and binary tree. Returns {draft id, probability}.
std::pair<std::uint32_t, float> select(std::span<const float> raw_logits, float cap);

[[nodiscard]] bool enabled_from_environment(); // STRIX_G4_EAGLE=0 disables drafting (OFF route)
enum class Kind { dflash, eagle, mtp };
[[nodiscard]] Kind kind_from_environment();      // STRIX_G4_DRAFT_KIND=dflash|eagle|mtp (default dflash)

struct CycleResult {
    dflash::Verdict verdict;
    std::uint32_t next_anchor = 0, k = 0;
    Draft draft;
    double draft_ms = 0, verify_ms = 0, stage_ms = 0, commit_ms = 0;
};
// One greedy verification cycle with the C protocol: draft (k>0) -> one target pass over [anchor, d1..dk] ->
// verify_greedy -> stage committed rows -> target commit -> drafter publish. Rejected rows and chain K/V are never
// published. Emitted ids are target greedy choices only.
CycleResult run_cycle(Target& target, KvTxn& transaction, Drafter& drafter, std::uint32_t anchor, std::uint32_t k,
                      std::size_t budget, std::span<const std::uint32_t> eos = {}, dflash::Acceptance* stats = nullptr,
                      std::uint64_t stream = 0, std::uint64_t generation = 0);
// Processes prompt[0..n-2] through the target in <=8-row transactions and primes the drafter with each chunk
// (next token = following prompt token). Returns the first decode anchor prompt.back().
std::uint32_t prefill(Target& target, const std::function<std::unique_ptr<KvTxn>()>& begin, Drafter& drafter,
                      std::span<const std::uint32_t> prompt, std::uint64_t stream = 0, std::uint64_t generation = 0);

} // namespace strix::gemma4::eagle
