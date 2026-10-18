#pragma once
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include <cstdint>
#include <atomic>
#include <string_view>
#include <vector>

namespace strix::moe_q4k_a8 {
inline constexpr std::string_view runtime_kernel_suffix = ".q4k-a8g32-gateup-v2";
inline constexpr std::string_view grouped_q51_kernel_ending =
    ".grouped-exact-q51-down-v1.q4k-a8g32-gateup-v2";
// Compact unambiguous RAW tag keeps existing combined LM/A8 records <=512.
inline constexpr std::string_view grouped_q51_raw_identity = "q51g-v1:v256:slot-fma:+0";
inline constexpr std::string_view grouped_q51_numerical_identity =
    "exact-grouped-q51-down-v1:original-f32-activation:virtual256-tree:ascending-slot-fma:positive-zero-seed";
inline constexpr std::string_view runtime_numerical_identity =
    "q4k-a8g32-native-v2:g32-sym127-f32div-rne-noflush:dot-s8xu8-i32:"
    "separate-f32-pair4-c0:legacy-silu-down-slot-order:finite-pipeline-failclosed";
inline constexpr std::string_view q51_a8_down_kernel_marker = ".q51-a8g32-down-v1";
inline constexpr std::string_view q80_a8_down_kernel_marker = ".q80-a8g32-down-v1";
inline constexpr std::string_view q5k_gate_up_kernel_marker = ".q5k-a8g32-gateup-v1";
inline constexpr std::string_view q5k_gate_up_raw_identity = "q5ka8v1:g32s127:u5qh:pair4c0";
inline constexpr std::string_view a8_down_kernel_ending =
    ".mixed-a8-down-v1.q4k-a8g32-gateup-v2";
// Compact RAW tags identify the new arithmetic without exceeding the RAW ABI.
inline constexpr std::string_view q51_a8_down_raw_identity =
    "q51a8v1:g32f32:t119:iu5:k5x12:fma";
inline constexpr std::string_view q80_a8_down_raw_identity =
    "q80a8v1:g32f32:t119:is8:k5x12:fma";
inline constexpr std::string_view a8_down_gate_up_raw_identity = "q4ka8v2:g32s127:pair4c0";
inline constexpr std::string_view a8_down_gate_up_numerical_identity =
    "q4k-a8g32-native-v2:g32-sym127-f32div-rne-noflush:dot-s8xu8-i32:separate-f32-pair4-c0";
struct RuntimeWorkspace {
    std::uint64_t gate_up_offset = 0, gate_up_bytes = 0;
    std::uint64_t accumulation_offset = 0, accumulation_bytes = 0;
    std::uint64_t status_offset = 0;
    std::uint64_t codes_offset = 0, codes_bytes = 0;
    std::uint64_t deltas_offset = 0, deltas_bytes = 0;
    std::uint64_t sums_offset = 0, sums_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t rawdots_offset = 0, rawdots_bytes = 0;
    std::uint64_t down_codes_offset = 0, down_codes_bytes = 0;
    std::uint64_t down_deltas_offset = 0, down_deltas_bytes = 0;
    std::uint64_t down_sums_offset = 0, down_sums_bytes = 0;
};
// Pure host admission. All regions begin on relative 256-byte boundaries,
// except status, which immediately follows accumulation for one ordered clear.
[[nodiscard]] RuntimeWorkspace build_runtime_workspace(
    std::uint64_t hidden, std::uint64_t rows, std::uint64_t experts, std::uint64_t top_k);
[[nodiscard]] RuntimeWorkspace build_runtime_workspace(
    std::uint64_t hidden, std::uint64_t rows, std::uint64_t experts, std::uint64_t top_k,
    bool q51_slot_rawdots);
// New post-SiLU quantization owns independent regions. It always reserves raw
// dots, including Q8_0 layers, independently of the old Q5_1 organization flag.
[[nodiscard]] RuntimeWorkspace build_runtime_workspace(
    std::uint64_t hidden, std::uint64_t rows, std::uint64_t experts, std::uint64_t top_k,
    bool q51_slot_rawdots, bool a8_down);
// Host submissions include graph captures; replay counts are not inferred.
struct RuntimeDispatchStats {
    std::atomic<std::uint64_t> batched_activation_launches{0};
    std::atomic<std::uint64_t> batched_activation_slots{0};
    std::atomic<std::uint64_t> quantized_submissions{0};
    std::atomic<std::uint64_t> legacy_q5k_submissions{0};
    std::atomic<std::uint64_t> grouped_q51_submissions{0};
    std::atomic<std::uint64_t> slot_parallel_q51_submissions{0};
    std::atomic<std::uint64_t> a8_down_q51_submissions{0};
    std::atomic<std::uint64_t> a8_down_q80_submissions{0};
    std::atomic<std::uint64_t> a8_down_q80_legacy_submissions{0};
    std::atomic<std::uint64_t> q5k_a8_gate_up_submissions{0};
    std::atomic<std::uint64_t> prefill_f16wmma_submissions{0}; // prompt rows of FP16 WMMA prefill blocks
};
struct RuntimeLayerClassification {
    std::vector<std::uint32_t> quantized_layers;
    std::vector<std::uint32_t> legacy_q5k_layers;
    std::vector<std::uint32_t> q4k_with_q51_down_layers;
    std::vector<std::uint32_t> q4k_with_q80_down_layers;
    std::vector<std::uint32_t> q5k_with_q80_down_layers;
};
// Requires unambiguous separate gate/up/down tensors for every routed layer.
// This runs before GPU ownership or full weight loading.
[[nodiscard]] RuntimeLayerClassification classify_runtime_manifest(
    const ModelContract& model, const ModelManifest& manifest);
// Explicit current-model slot-parallel admission before residency. The five
// noneligible layers retain their existing per-slot fallback.
void validate_slot_parallel_manifest(
    const ModelContract& model, const RuntimeLayerClassification& classification);
// Source/content scope is distinct from numerical qualification. The explicit
// arithmetic candidates retain one canonical Q5_K fallback and, if Q8_0 is not
// requested, the four canonical Q4_K/Q8_0 fallback layers.
void validate_a8_down_manifest(const ModelContract&, const RuntimeLayerClassification&,
    bool a8_down_q51, bool a8_down_q80);
void validate_a8_down_content(const ModelContentIdentity&);
// The explicit Q5_K candidate promotes only canonical layer 2. Classification
// remains by stored weight type; it does not relabel Q5_K as Q4_K.
void validate_q5k_a8_manifest(const ModelContract&, const RuntimeLayerClassification&);
}  // namespace strix::moe_q4k_a8
