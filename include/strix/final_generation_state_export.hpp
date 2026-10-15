#pragma once

#include "strix/raw_baseline_recording.hpp"
#include "strix/serial_request.hpp"
#include "strix/session_state_memory.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace strix {
struct FinalGenerationStateMetadata {
    std::string model_sha256, model_content_identity, executable_sha256;
    std::string numerical_identity, output_mode, gpu_identity, gpu_architecture;
    std::uint64_t kernel_plan_identity = 0, launch_plan_identity = 0, runtime_binding_identity = 0;
    std::uint32_t hip_runtime_version = 0, hip_driver_version = 0;
    bool prepared_eager = false, shared_overlap = false, batched_moe_activation = false;
};
struct FinalGenerationStatePolicy {
    std::uint64_t maximum_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
};
// Admission only: fresh destination in an existing canonical parent; no mkdir.
void validate_final_generation_state_destination(const std::filesystem::path&);
void validate_final_generation_state_budget(const SerialSessionStateLayout&, FinalGenerationStatePolicy);

// Validation-only synchronous whole-seven-component export. No logits/model
// execution. Exclude mutation and retain real leases through original-device
// quiescence, including errors. require_binding checks actual device/runtime/
// state generations. before_publication rechecks model/executable hashes.
// quiesce_or_fail_stop MUST drain original work or terminate without unwinding
// unproved async ownership. Failures preserve .partial; the sole publication is
// an atomic no-replace directory rename after every artifact is complete.
void export_final_generation_state(const std::filesystem::path&, FinalGenerationStatePolicy,
    const ModelContract&, const SerialSessionStateLayout&, const SessionStateMemoryBinding&,
    const SerialSessionContext&, const SerialGreedyRequest&, const SerialGreedyRequestResult&,
    const FinalGenerationStateMetadata&, std::shared_ptr<const void> actual_state_owner,
    const RawBaselineStateReader&, const std::function<void()>& require_binding,
    const std::function<void()>& quiesce_or_fail_stop, const std::function<void()>& before_publication);
} // namespace strix
