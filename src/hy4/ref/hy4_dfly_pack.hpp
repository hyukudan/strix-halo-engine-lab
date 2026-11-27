#pragma once
// Hy4-flash DFly2 drafter: the shard's tensors laid out as Hy4 blocks for the device forward (host, no device types).
// Model data only: the draft layers ARE target DSA layers (same attention form, hy4_dfly_ref.hpp header), so draft
// layer i becomes block `first_block + i` with the target's GGUF tensor names and orientation ([outputs][inputs] rows,
// dims {inputs, outputs}); the device forward runs them through the target's own attention rows form.
//   input_layernorm -> attn_norm, q_a_proj -> attn_q_a, q_a_layernorm -> attn_q_a_norm, q_b_proj -> attn_q_b,
//   kv_a_proj_with_mqa -> attn_kv_a_mqa, kv_a_layernorm -> attn_kv_a_norm, linear_gate -> attn_gate,
//   kv_b_proj [H][nope + v][kv_lora] -> attn_k_b (per head [kv_lora][nope]: the K block transposed, the absorb form) and
//   attn_v_b (per head [v][kv_lora]), o_proj -> attn_output, post_proj_norm -> post_attention_norm,
//   learnable_sink_param -> attn_sinks, post_attention_layernorm -> ffn_norm, mlp.{gate,up,down}_proj -> ffn_{gate,up,down}
// Globals: dfly.enorm / dfly.hnorm / dfly.eh_proj (mtp_fusion; the slot-0 gate g folded into the rows of eh_proj:
// x0 = e0 + eh_proj'(cat), eh_proj'[o][:] = g[o] * eh_proj[o][:], FP32 product rounded once to BF16), dfly.final_norm,
// dfly.hc_hnorm / dfly.hc_enorm / dfly.hc_gate / dfly.hc_up / dfly.hc_down (hidden_correction).
// Norms and sinks are F32 (the device reads them as FP32 vectors); Linears BF16 (the shard's bits) or Q8_0 (strix/q8_0_rule.hpp
// on the BF16 values). The drafter's numerics change acceptance only, never an emitted id (the target verifies greedily).
#include "hy4/ref/hy4_dfly_ref.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix::hy4::dfly {

enum class PackFormat : std::uint8_t { bf16, q8_0 };

inline constexpr std::uint32_t kGgmlF32 = 0U, kGgmlQ8_0 = 8U, kGgmlBf16 = 30U;

struct PackTensor {
    std::string name;                  // "blk.<b>.attn_q_a.weight" / "dfly.eh_proj"
    std::uint32_t ggml_type = 0;
    std::vector<std::uint64_t> dims;   // {inputs, outputs} (GGUF order); norms / sinks {n}
    std::vector<std::uint8_t> bytes;
};

struct Pack {
    std::uint32_t first_block = 0;
    std::vector<PackTensor> tensors;
    [[nodiscard]] const PackTensor& at(const std::string& name) const;
    [[nodiscard]] std::uint64_t total_bytes() const;
};

// Every tensor the device forward reads (88 shard tensors minus the confidence head = 86 sources; kv_b gives two).
[[nodiscard]] Pack build_pack(const Geometry& g, Weights& w, std::uint32_t first_block, PackFormat format);
// Device name of draft layer i's tensor `suffix` ("attn_q_a.weight").
[[nodiscard]] std::string block_name(std::uint32_t first_block, std::uint32_t layer, const char* suffix);
[[nodiscard]] PackFormat parse_pack_format(const std::string& text);  // "bf16" | "q8"

}  // namespace strix::hy4::dfly
