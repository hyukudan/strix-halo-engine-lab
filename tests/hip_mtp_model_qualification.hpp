#pragma once
#include "../apps/hip_mtp_decode_bridge.hpp"
namespace strix::tests {
void set_mtp_product_profile(const qwen::profile::Resolution&);
void run_mtp_model_qualification(decode_cli::MtpDecodeQualificationContext&);
[[nodiscard]] bool mtp_component_diagnostic_requested() noexcept;
void run_mtp_component_diagnostic(const std::shared_ptr<HipMtpResources>&,std::uint64_t actual_native_profile_identity);
[[nodiscard]] bool mtp_target_input_export_requested() noexcept;
[[nodiscard]] bool mtp_prefix_gate_requested() noexcept;
void validate_mtp_target_input_export(const decode_cli::Options&,bool resolved);
void export_mtp_target_inputs(const decode_cli::Options&,const ModelContract&,std::shared_ptr<::HipDecodeExperimentResources>,
    const CompiledSerialExecution&,const ExecutionRuntimeResources&,SerialSessionContext&,DirectLaunchDispatcher&,
    AllocationGenerationSource&,bool deferred_status);
}
