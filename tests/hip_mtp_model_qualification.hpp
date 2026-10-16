#pragma once
#include "../apps/hip_mtp_decode_bridge.hpp"
namespace strix::tests {
void run_mtp_model_qualification(decode_cli::MtpDecodeQualificationContext&);
[[nodiscard]] bool mtp_target_input_export_requested() noexcept;
void validate_mtp_target_input_export(const decode_cli::Options&);
void export_mtp_target_inputs(const decode_cli::Options&,const ModelContract&,std::shared_ptr<::HipDecodeExperimentResources>,
    const CompiledSerialExecution&,const ExecutionRuntimeResources&,SerialSessionContext&,DirectLaunchDispatcher&,
    AllocationGenerationSource&,bool deferred_status);
}
