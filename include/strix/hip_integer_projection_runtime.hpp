#pragma once
#include "strix/hip_integer_projection.hpp"
#include "strix/integer_projection_runtime.hpp"

namespace strix::integer_projection {
// Native extent discovery for an existing device allocation. The strong token
// must own its actual allocator/model lease; a copied raw pointer is insufficient.
// A single scheduler advances content epochs before producers overwrite data.
class HipIntegerRuntimeAllocationOwner final:public AllocationPin {
public:
    [[nodiscard]] static std::shared_ptr<HipIntegerRuntimeAllocationOwner> pin(
        DeviceIdentity,std::uintptr_t view_address,std::uint64_t allocation_generation,
        std::shared_ptr<void> allocation_owner);
    [[nodiscard]] AllocationBinding current_binding()const override;
    void advance_access(std::uint64_t next_epoch);
private:
    AllocationBinding binding_;
    std::shared_ptr<void> owner_;
};
struct RuntimeOutcome {
    bool output_committed=false, destination_poisoned=false, stream_quiescent=false;
    std::uint64_t completed_tiles=0, actual_internal_bytes=0, actual_retained_bytes=0;
    std::string numerical_identity, runtime_identity, reason;
};
// Real eager consumer: original GGUF device blocks -> lossless tile repack,
// source BF16/F32 widening -> shared IU8/IU4 launcher -> checked FP32 conversion
// -> one whole-logits commit. Blocking; never captures graphs or auto-qualifies.
class HipIntegerProjectionRuntimeBundle {
public:
    HipIntegerProjectionRuntimeBundle(RuntimePlan,RuntimeOwners,hipStream_t,
        std::shared_ptr<void> stream_owner={});
    ~HipIntegerProjectionRuntimeBundle();
    HipIntegerProjectionRuntimeBundle(const HipIntegerProjectionRuntimeBundle&)=delete;
    HipIntegerProjectionRuntimeBundle& operator=(const HipIntegerProjectionRuntimeBundle&)=delete;
    void register_for_compiled_execution(const CompiledSerialExecution&,ExecutionLauncherRegistry&);
    [[nodiscard]] RuntimeOutcome execute(const ExecutionDispatchContext&);
    [[nodiscard]] bool drain()noexcept;
    [[nodiscard]] bool failed()const noexcept;
private:
    static void dispatch(const ExecutionDispatchContext&,void*);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::integer_projection
