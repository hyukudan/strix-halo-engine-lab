#pragma once

#include "strix/deepseek41_model.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Execution plan of one deepseek41 decode step (one token row) over a range of target layers, derived from the
// admitted model contract: the ordered operations, the weights each reads, the per-row activation buffers they
// pass, the persistent state they touch, which piece implements them and the numerical identity they follow.
// A runner interprets the plan; each operation can be bound to a host reference or a device kernel, so the
// host-only, mixed and device-only runs of the same plan are comparable step by step.
namespace strix::deepseek41 {

enum class Op : std::uint8_t {
    embed,            // token row (F16) -> hc copies of the residual stream
    engram,           // n-gram hash, row gather, decode, engram_kv projection, gated add into each copy
    hc_mix,           // RMS of the flattened stream, hc_*_fn projection, Sinkhorn: pre/post/comb
    hc_pre,           // collapse the copies with pre (previous sublayer's) -> one row
    rms_norm,         // attn_norm / ffn_norm / q_a_norm (kv_a_norm belongs to window_write, output_norm to head)
    q_a,              // attn_q_a  (Q8_0, 5120 -> 1280)
    q_b,              // attn_q_b  (Q8_0, 1280 -> 64 x 512)
    kv,               // attn_kv   (Q8_0, 5120 -> 512): wkv(x), normalised and rotated by the attention piece
    window_write,     // kv_norm + RoPE of wkv(x), FP8 E4M3/32 (UE8M0) into the layer's 128-entry ring
    compress,         // KV source: compressor (gated pooling for ratio 2) -> FP4 KV entry + FP4 index key
    index,            // index source: FP4 queries, ReLU-weighted head scores, (candidate pool), top-k
    attention,        // RoPE of q, D512 MQA over window + selected compressed positions, per-head sink, inverse RoPE
    o_a,              // attn_output_a (Q8_0, 8 groups of 4096 -> 1024, block diagonal)
    o_b,              // attn_output_b (Q8_0, 8192 -> 5120)
    hc_post,          // expand the sublayer output with post and mix the residual with comb
    router,           // sqrt(softplus) scores, biased top-6, weights x 1.5
    shared_expert,    // Q8_0 SwiGLU (clamped)
    routed_experts,   // IQ2_XXS gate/up + Q2_K down per selected expert (raw slot vectors)
    combine,          // (shared + low half) + high half, expert halves = ranks
    head,             // final hc_pre + output_norm + output (Q8_0, 5120 -> 129280)
    argmax,           // lowest index on ties (split vocabulary: per-rank argmax then min index)
};
[[nodiscard]] std::string_view to_string(Op op) noexcept;

// Who implements the operation (wave 2 fronts); `integration` marks a stand-in kept until the owner lands.
enum class Owner : std::uint8_t { formats_moe, attention, layer, integration };
[[nodiscard]] std::string_view to_string(Owner owner) noexcept;

// Per-row activation buffers of one step (FP32 unless noted). Sizes in elements.
enum class Buffer : std::uint8_t {
    stream,      // hc_mult x hidden residual copies
    mix_attn,    // pre, post, comb of the attention site (hc + hc + hc*hc)
    mix_ffn,     // the same for the FFN site
    pre_carry,   // pre of the previous FFN site (identity one-hot before layer 0): collapses the next attention input
    x,           // collapsed sublayer input / normed input (hidden)
    qr,          // q_a output after q_norm (q_lora_rank)
    q,           // heads x head_dim (before RoPE)
    kvrow,       // head_dim: wkv(x)
    o,           // heads x head_dim
    o_low,       // o_groups x o_lora_rank
    y,           // sublayer output (hidden)
    shared,      // shared expert output (hidden)
    route_ids,   // top_k expert ids (u32)
    route_w,     // top_k weights
    raw,         // top_k x hidden unweighted expert outputs
    logits,      // vocab
    token,       // u32
};
[[nodiscard]] std::string_view to_string(Buffer buffer) noexcept;

struct Step {
    Op op = Op::embed;
    std::int32_t layer = -1;              // -1: outside the layer stack
    std::string site;                     // "attn" / "ffn" for hc and norms, else empty
    std::vector<std::string> weights;     // GGUF tensor names read by the step (manifest entries)
    std::vector<Buffer> inputs, outputs;
    Owner owner = Owner::integration;
    std::string identity;                 // declared numerical identity of the operation
};

// Persistent per-sequence state of one layer.
struct LayerState {
    std::uint32_t layer = 0;
    std::uint64_t window_ring_bytes = 0;          // 128 entries x (512 FP8 + 16 UE8M0)
    std::uint64_t kv_bytes_per_position = 0;      // compressed FP4 KV (KV sources), amortised per token
    std::uint64_t index_key_bytes_per_position = 0;
    std::uint64_t compress_carry_bytes = 0;       // ratio-2 partial group (kv + score, FP32)
};

struct Plan {
    std::uint32_t first_layer = 0, end_layer = 0;  // [first, end)
    bool embed = true, head = true;
    std::uint32_t ranks = 2;                       // expert halves of the combine (one node emulates both)
    std::uint32_t expert_split = 0;                // experts [0, split) on rank 0, [split, n) on rank 1
    std::vector<Step> steps;
    std::vector<LayerState> state;
    // fixed per-sequence bytes (rings, carries) and bytes per token of context (KV + index keys + Engram history)
    std::uint64_t fixed_state_bytes = 0, state_bytes_per_token = 0;
    [[nodiscard]] std::size_t buffer_elements(Buffer buffer) const noexcept;
    Config config;  // copy of the geometry the plan was built for
};

// The decode-step plan for target layers [first, end); a trimmed model keeps the embedding and the head.
// Throws std::invalid_argument for an empty or out-of-range layer range.
[[nodiscard]] Plan build_decode_plan(const TargetModel& model, std::uint32_t first, std::uint32_t end, bool embed = true,
                                     bool head = true);
// Every weight a plan names must be in the model manifest with the format its operation expects.
void validate_plan(const Plan& plan, const TargetModel& model);

}  // namespace strix::deepseek41
