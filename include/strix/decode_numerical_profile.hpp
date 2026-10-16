#pragma once

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
    hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1
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
    }
    return {};
}

inline constexpr std::optional<DecodeNumericalProfile> parse_decode_numerical_profile(std::string_view name) {
    for (const auto value : {DecodeNumericalProfile::exact_v5, DecodeNumericalProfile::hc_splitk_f32_v1,
             DecodeNumericalProfile::q51_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1,
             DecodeNumericalProfile::q51_q80_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1})
        if (numerical_profile_name(value) == name) return value;
    return std::nullopt;
}

inline constexpr bool numerical_profile_q5k_a8_gateup(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1;
}
inline constexpr bool numerical_profile_hc_deferred_norm(DecodeNumericalProfile value) {
    return value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1 ||
        value == DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1;
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
} // namespace strix
