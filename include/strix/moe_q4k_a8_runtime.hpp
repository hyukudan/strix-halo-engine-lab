#pragma once
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
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
struct RuntimeWorkspace {
    std::uint64_t gate_up_offset = 0, gate_up_bytes = 0;
    std::uint64_t accumulation_offset = 0, accumulation_bytes = 0;
    std::uint64_t status_offset = 0;
    std::uint64_t codes_offset = 0, codes_bytes = 0;
    std::uint64_t deltas_offset = 0, deltas_bytes = 0;
    std::uint64_t sums_offset = 0, sums_bytes = 0;
    std::uint64_t total_bytes = 0;
};
// Pure host admission. All regions begin on relative 256-byte boundaries,
// except status, which immediately follows accumulation for one ordered clear.
[[nodiscard]] RuntimeWorkspace build_runtime_workspace(
    std::uint64_t hidden, std::uint64_t rows, std::uint64_t experts, std::uint64_t top_k);
// Host submissions include graph captures; replay counts are not inferred.
struct RuntimeDispatchStats {
    std::atomic<std::uint64_t> batched_activation_launches{0};
    std::atomic<std::uint64_t> batched_activation_slots{0};
    std::atomic<std::uint64_t> quantized_submissions{0};
    std::atomic<std::uint64_t> legacy_q5k_submissions{0};
    std::atomic<std::uint64_t> grouped_q51_submissions{0};
};
struct RuntimeLayerClassification {
    std::vector<std::uint32_t> quantized_layers;
    std::vector<std::uint32_t> legacy_q5k_layers;
    std::vector<std::uint32_t> q4k_with_q51_down_layers;
};
// Requires unambiguous separate gate/up/down tensors for every routed layer.
// This runs before GPU ownership or full weight loading.
[[nodiscard]] RuntimeLayerClassification classify_runtime_manifest(
    const ModelContract& model, const ModelManifest& manifest);
}  // namespace strix::moe_q4k_a8
