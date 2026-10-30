#pragma once
#include "strix/model_content_hash.hpp"
#include <cstdint>
#include <vector>
#include <filesystem>
namespace strix {
struct HipMtpForwardGoldenRow {
    std::uint64_t source_position=0;
    std::uint32_t source_input_token=0,next_input_token=0,selected_token=0;
    std::vector<std::uint16_t> target_hidden_bf16,next_hidden_bf16;
    std::vector<float> full_logits;
};
struct HipMtpForwardGolden {
    ModelContentIdentity independent_source_content,target_content,sidecar_content;
    std::uint64_t independent_reference_identity=0,structural_plan_identity=0,semantic_identity=0,numerical_profile_identity=0;
    // Target arithmetic that produced the prompt rows (numerical_profile_identity above is the
    // MTP head's own cast profile): the prefill and decode profile identities. Both zero in a
    // version-1 file, which predates the split and records neither.
    std::uint64_t target_prefill_profile_identity=0,target_decode_profile_identity=0;
    // V3: shared target/predictor rotary, actual owner capacity and fresh calibration.
    // Zero factor preserves the historical native v1/v2 serialization and frame.
    std::uint64_t rotary_factor=0,context_capacity=0,calibration_begin=0,calibration_end=0;
    ModelContentIdentity rotary_descriptor_sha256,rotary_table_sha256;
    std::uint64_t prompt_prefill_profile_identity=0;
    std::vector<HipMtpForwardGoldenRow> rows;
};
struct HipMtpForwardReplayRowReceipt {
    std::uint64_t source_position=0,seed_position=0;
    std::uint32_t selected_token=0,completed_steps=0;
    ModelContentIdentity actual_hidden,actual_logits;
};
// Reads at most 288 header bytes; verifies the exact file extent before allocation.
void require_mtp_forward_golden_geometry(const std::filesystem::path&,std::uint64_t rows,
    std::uint64_t width,std::uint64_t vocabulary,std::uint64_t capacity);
} // namespace strix
