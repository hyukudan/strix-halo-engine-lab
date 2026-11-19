#pragma once
// K/V cache storage classes: one policy for TARGET caches and DRAFTER caches on every backend.
//
// A form names how one cache stores its K/V rows; the device bodies that write and read each form live in the backend's
// shared attention home and are selected as a trait of the shared kernel entry (CUDA: cuda_core::attention_decode_kv), never
// as a model-local copy.
//   bf16        canonical BF16 rows (every backend).
//   int8_token  kv-int8-tok-v1: per (token, kv head, K/V half) int8 codes rint(x / s), s = absmax / 127, one FP32 scale
//               (HIP sparse QSA walk; strix/model_contract.hpp AttentionKvStorage::int8_token).
//   e4m3_token  per (token, kv head) e4m3 codes of x / s, s = absmax / 448, one FP32 scale for K and one for V (CUDA
//               cuda_core::KvShadowE4m3: attn8 prefill, attention_decode8 through attention_decode_kv).
//
// Role decides what a non-BF16 form may change. A TARGET cache feeds the logits: its form is part of the arithmetic identity
// and is a FAST lever behind the quality gate. A DRAFT cache feeds only a drafter whose proposals pass exact verification
// against the target (greedy prefix match on the target's own logits): its form can move acceptance, never the emitted ids,
// so it is allowed in every tier and never enters an identity string.
//
// Per-channel key layouts (4-bit keys need them: per-token 4-bit keys lose most of the score signal on keys with outlier
// channels, see tests/kv_storage_tests.cpp) are not a form here: no backend has a reader for them.
#include <cstdint>
#include <optional>
#include <string_view>

namespace strix::kv {

enum class Form : std::uint32_t { bf16 = 0, int8_token = 1, e4m3_token = 2 };
enum class Role : std::uint32_t { target = 0, draft = 1 };

struct Shape {
    std::uint32_t kv_heads = 0, head_dim = 0;
};

// Bytes of one token's K and V rows in one layer, scales included.
[[nodiscard]] constexpr std::uint64_t bytes_per_token(Form f, Shape s) noexcept {
    const std::uint64_t elems = static_cast<std::uint64_t>(s.kv_heads) * s.head_dim;
    if (f == Form::bf16) return 2U * elems * 2U;
    return 2U * elems + 2U * static_cast<std::uint64_t>(s.kv_heads) * 4U;  // codes K + V, one FP32 scale per kv head and half
}

// Whether storing a cache of this role in this form can change the emitted token ids.
[[nodiscard]] constexpr bool changes_output(Form f, Role r) noexcept { return r == Role::target && f != Form::bf16; }

// Whether a tier admits the form for the role: exact tiers take only forms that cannot change the output.
[[nodiscard]] constexpr bool allowed(Form f, Role r, bool fast_tier) noexcept { return fast_tier || !changes_output(f, r); }

[[nodiscard]] constexpr std::string_view name(Form f) noexcept {
    switch (f) {
        case Form::bf16: return "bf16";
        case Form::int8_token: return "kv-int8-tok-v1";
        case Form::e4m3_token: return "kv-e4m3-tok-v1";
    }
    return "?";
}

// CLI word of a draft / target cache form: "bf16" or "kv8" (the backend's 8-bit token form: e4m3_token on CUDA, int8_token
// on HIP, where gfx1151 has no FP8). nullopt for any other word.
[[nodiscard]] constexpr std::optional<Form> parse(std::string_view word, Form eight_bit) noexcept {
    if (word == "bf16") return Form::bf16;
    if (word == "kv8") return eight_bit;
    return std::nullopt;
}

}  // namespace strix::kv
