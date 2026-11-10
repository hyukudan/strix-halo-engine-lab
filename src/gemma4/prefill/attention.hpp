#pragma once
#include "emulator.hpp"
#include <functional>

namespace strix::gemma4::prefill {
inline constexpr char attention_identity[] = "g4-wmma-attn-v1";
inline constexpr char flash_attention_identity[] = "g4-flash-attn-v1"; // shared src/attn_core dense flash core
enum class AttentionMode { canonical, wmma, flash };
[[nodiscard]] const char* attention_name(AttentionMode mode);
[[nodiscard]] AttentionMode resolve_attention(std::string_view value);
[[nodiscard]] AttentionMode attention_mode(); // startup-cached STRIX_G4_PREFILL_ATTN
[[nodiscard]] bool attention_pack_enabled(); // startup-cached STRIX_G4_PREFILL_ATTN_PACK

// Independent per-query instruction emulator. read(p,value) returns one KV
// head's FP32 vector at absolute p; masked keys are never read. See README for
// the frozen K16 score, online softmax and PV recurrence and FP16 boundaries.
using AttentionRead = std::function<const float*(std::uint64_t, bool)>;
[[nodiscard]] std::vector<float> emulate_attention(std::span<const float> query, std::uint64_t position,
                                                   unsigned window, const AttentionRead& read);
} // namespace strix::gemma4::prefill
