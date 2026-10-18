#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace strix {
// Arithmetic choices are independent of submission/organization defaults.
// Every alternative has a separate identity and requires model qualification.
enum class DecodeNumericalProfile {
    exact_v5,
    hc_splitk_f32_v1,
    q51_a8_down_v1,
    hc_splitk_f32_q51_a8_down_v1,
    q51_q80_a8_down_v1,
    hc_splitk_f32_q51_q80_a8_down_v1,
    hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
    hc_deferred_norm_f32_q51_q80_a8_down_v1,
    hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
    // The q5k-a8-gateup profiles plus dense A8: every dense Q8_0 projection of the GDN,
    // QSA and shared-expert steps runs Q8_0 x per-32 int8 activations (dense-a8-q80-v1).
    hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
    hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
    // Prefill-only (never a decode profile): the kernel plan of
    // hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1 whose prompt launches run FP16 WMMA
    // arithmetic in the components of NumericalProfilePair::fast_components (prefill HC projections,
    // dense projections of 32..256 rows, routed MoE). Launches outside those keep the sibling's arithmetic.
    fast_prefill_f16wmma_v1
};

inline constexpr std::string_view numerical_profile_name(DecodeNumericalProfile value) {
    switch (value) {
    case DecodeNumericalProfile::exact_v5: return "exact-v5";
    case DecodeNumericalProfile::hc_splitk_f32_v1: return "hc-splitk-f32-v1";
    case DecodeNumericalProfile::q51_a8_down_v1: return "q51-a8-down-v1";
    case DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1: return "hc-splitk-f32+q51-a8-down-v1";
    case DecodeNumericalProfile::q51_q80_a8_down_v1: return "q51-q80-a8-down-v1";
    case DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1: return "hc-splitk-f32+q51-q80-a8-down-v1";
    case DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1: return "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1";
    case DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1: return "hc-deferred-norm-f32+q51-q80-a8-down-v1";
    case DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1: return "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1";
    case DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1:
        return "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    case DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1:
        return "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    case DecodeNumericalProfile::fast_prefill_f16wmma_v1: return "fast-prefill-f16wmma-v1";
    }
    return {};
}

inline constexpr std::optional<DecodeNumericalProfile> parse_decode_numerical_profile(std::string_view name) {
    for (const auto value : {DecodeNumericalProfile::exact_v5, DecodeNumericalProfile::hc_splitk_f32_v1,
             DecodeNumericalProfile::q51_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1,
             DecodeNumericalProfile::q51_q80_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
             DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
             DecodeNumericalProfile::fast_prefill_f16wmma_v1})
        if (numerical_profile_name(value) == name) return value;
    return std::nullopt;
}

inline constexpr bool numerical_profile_fast_prefill(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::fast_prefill_f16wmma_v1;
}
// The decode profile a fast prefill profile is paired with (its kernel plan and fallback arithmetic).
inline constexpr std::optional<DecodeNumericalProfile> numerical_profile_fast_prefill_decode(DecodeNumericalProfile value) {
    if (value == DecodeNumericalProfile::fast_prefill_f16wmma_v1)
        return DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
    return std::nullopt;
}
// Recipe predicates of a fast prefill profile are those of its decode sibling (same kernel plan).
inline constexpr bool numerical_profile_dense_a8(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1 ||
        numerical_profile_fast_prefill(value);
}
inline constexpr bool numerical_profile_q5k_a8_gateup(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1 ||
        numerical_profile_dense_a8(value);
}
// (A fast prefill profile is not deferred-norm: numerical_profile_dense_a8 covers it through its sibling.)
inline constexpr bool numerical_profile_hc_deferred_norm(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
}
inline constexpr bool numerical_profile_hc_splitk(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_splitk_f32_v1 ||
        value == DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1 ||
        numerical_profile_q5k_a8_gateup(value) || numerical_profile_hc_deferred_norm(value);
}
inline constexpr bool numerical_profile_q51_a8_down(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::q51_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1 ||
        value == DecodeNumericalProfile::q51_q80_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1 ||
        numerical_profile_q5k_a8_gateup(value) || numerical_profile_hc_deferred_norm(value);
}
inline constexpr bool numerical_profile_q80_a8_down(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::q51_q80_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1 ||
        numerical_profile_q5k_a8_gateup(value) || numerical_profile_hc_deferred_norm(value);
}

// Stable 64-bit identity of a profile: FNV-1a over its canonical name. Recorded in
// goldens and receipts; the enum value is not an identity.
inline constexpr std::uint64_t numerical_profile_identity(DecodeNumericalProfile value) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : numerical_profile_name(value)) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// The profile that differs from `value` only in the dense Q8_0 projections (dense A8 on/off).
// Only the q5k-a8-gateup profiles have a dense-A8 sibling.
inline constexpr std::optional<DecodeNumericalProfile> numerical_profile_dense_a8_sibling(DecodeNumericalProfile value) {
    switch (value) {
    case DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1:
        return DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
    case DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1:
        return DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1;
    case DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1:
        return DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
    case DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1:
        return DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1;
    default: return std::nullopt;
    }
}

// Components of a fast prefill profile (any non-empty subset; all by default).
inline constexpr std::uint32_t kFastPrefillHc = 1U, kFastPrefillDense = 2U, kFastPrefillMoe = 4U, kFastPrefillQsa = 8U,
    kFastPrefillRouter = 16U,
    kFastPrefillAll = kFastPrefillHc | kFastPrefillDense | kFastPrefillMoe | kFastPrefillQsa | kFastPrefillRouter;

// Prefill/decode arithmetic pair. The prompt prefill may run a different (prefill) profile from
// the decode/MTP windows; decode always runs `decode`. A pair is admitted when both profiles are
// equal (the default), when they differ only in the dense Q8_0 projections, which are the only
// kernels whose prefill and decode variants share every arena, state layout and MoE/HC kernel,
// or when the prefill profile is a fast prefill profile paired with its decode sibling (same
// kernel plan and layouts; prefill-only arithmetic in `fast_components`).
struct NumericalProfilePair {
    DecodeNumericalProfile prefill = DecodeNumericalProfile::exact_v5;
    DecodeNumericalProfile decode = DecodeNumericalProfile::exact_v5;
    std::uint32_t fast_components = 0;  // non-zero exactly when prefill is a fast prefill profile
    [[nodiscard]] constexpr bool split() const noexcept { return prefill != decode; }
    friend constexpr bool operator==(const NumericalProfilePair&, const NumericalProfilePair&) = default;
};
inline constexpr bool numerical_profile_pair_admitted(NumericalProfilePair pair) {
    if (numerical_profile_fast_prefill(pair.decode)) return false;
    if (numerical_profile_fast_prefill(pair.prefill))
        return numerical_profile_fast_prefill_decode(pair.prefill) == pair.decode && pair.fast_components != 0U &&
            (pair.fast_components & ~kFastPrefillAll) == 0U;
    return pair.fast_components == 0U && (!pair.split() || numerical_profile_dense_a8_sibling(pair.decode) == pair.prefill);
}
// Identity of a pair. A pure pair keeps the identity of its single profile, so records that
// predate the split (one profile) compare equal to the same profile used for both phases.
inline constexpr std::uint64_t numerical_profile_pair_identity(NumericalProfilePair pair) {
    const auto decode = numerical_profile_identity(pair.decode);
    if (!pair.split()) return decode;
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const std::uint64_t word : {std::uint64_t{0x50524546494c4c31ULL} /* PREFILL1 */,
             numerical_profile_identity(pair.prefill), decode})
        for (unsigned i = 0; i < 8; ++i) {
            hash ^= (word >> (8U * i)) & 0xffU;
            hash *= 0x100000001b3ULL;
        }
    // A fast prefill profile with fewer than all components (an ablation) has its own identity.
    if (numerical_profile_fast_prefill(pair.prefill) && pair.fast_components != kFastPrefillAll)
        for (const std::uint64_t word : {std::uint64_t{0x46415354434f4d50ULL} /* FASTCOMP */, std::uint64_t{pair.fast_components}})
            for (unsigned i = 0; i < 8; ++i) {
                hash ^= (word >> (8U * i)) & 0xffU;
                hash *= 0x100000001b3ULL;
            }
    return hash;
}
} // namespace strix
