#pragma once

#include "strix/integer_projection_execution.hpp"
#include <hip/hip_runtime_api.h>

namespace strix::integer_projection {
// Sealed event factory checks the current device and stream before recording;
// callers cannot relabel an event from another device as input readiness.
class HipProjectionReadyEvent {
public:
    [[nodiscard]] static std::shared_ptr<const HipProjectionReadyEvent> record(
        DeviceIdentity,hipStream_t,AllocationPins producer_pins,std::shared_ptr<void> stream_owner={});
    ~HipProjectionReadyEvent();
    HipProjectionReadyEvent(const HipProjectionReadyEvent&)=delete;
    HipProjectionReadyEvent& operator=(const HipProjectionReadyEvent&)=delete;
    [[nodiscard]] const DeviceIdentity& device() const noexcept;
private:
    HipProjectionReadyEvent()=default;
    [[nodiscard]] hipEvent_t native_handle() const noexcept;
    [[nodiscard]] bool matches_pins(const AllocationPins&) const noexcept;
    friend class HipIntegerProjectionLauncher;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct HipProjectionInputs {
    ExecutionBindings buffers;
    AllocationPins pins;
    // Recorded after all input producers; retained through final stream proof.
    std::shared_ptr<const HipProjectionReadyEvent> producer_ready;
};
enum class HipProjectionStatus { complete, rejected, failed, failed_retained };
struct HipProjectionOutcome {
    HipProjectionStatus status=HipProjectionStatus::rejected;
    CompletionEvidence evidence;
    std::string reason;
    std::uint64_t actual_device_bytes=0; // Whole HIP-reported allocations, once.
};
// Opt-in blocking laboratory launcher. One serialized owner; no default stream
// inference, production registry, automatic mode or model qualification.
// Every input/scratch/destination allocation is retained through stream proof.
// Failed submission/copy poisons the launcher; unproved quiescence retains pins,
// ready event and stream owner until drain or process-lifetime quarantine.
class HipIntegerProjectionLauncher {
public:
    HipIntegerProjectionLauncher(DeviceIdentity,hipStream_t,std::shared_ptr<void> stream_owner={});
    ~HipIntegerProjectionLauncher();
    HipIntegerProjectionLauncher(const HipIntegerProjectionLauncher&)=delete;
    HipIntegerProjectionLauncher& operator=(const HipIntegerProjectionLauncher&)=delete;
    [[nodiscard]] HipProjectionOutcome execute(const ExecutionPlan&,HipProjectionInputs);
    [[nodiscard]] bool drain() noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool retains_resources() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::integer_projection
