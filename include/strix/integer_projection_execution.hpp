#pragma once

#include "strix/integer_projection.hpp"
#include <array>
#include <memory>

namespace strix::integer_projection {
enum class Phase { unspecified, prefill, decode, verify };
struct Geometry {
    std::uint32_t tile_rows=16, tile_columns=16, tile_inner=16;
    std::uint32_t wave_size=32, threads=32, preparation_threads=128;
    bool operator==(const Geometry&) const = default;
};
struct DeviceIdentity {
    std::int32_t index=-1;
    std::uint64_t instance_generation=0;
    std::string architecture;
    bool operator==(const DeviceIdentity&) const = default;
};
struct ExecutionRequest {
    bool experimental_opt_in=false;
    Phase phase=Phase::unspecified;
    std::uint64_t operation_generation=0; // Nonzero, increasing per launcher/owner.
    Shape shape{0,0,0};
    WeightEncoding weights=WeightEncoding::q4_0;
    low_precision::ActivationEncoding activations=low_precision::ActivationEncoding::symmetric_a8;
    Instruction instruction=Instruction::iu8;
    Geometry geometry;
    DeviceIdentity device;
    std::string approved_numerical_identity;
    std::uint64_t maximum_device_bytes=0; // Explicit admission ceiling, <=256 MiB.
};
enum class BufferRole : std::size_t {
    input_f64, packed_weights, weight_scales_f64, activation_codes,
    activation_scales_f64, activation_sums_i32, staged_output_f64,
    destination_f64, integer_groups_i32, status_u32, count
};
inline constexpr std::size_t buffer_count=static_cast<std::size_t>(BufferRole::count);
[[nodiscard]] constexpr std::size_t index(BufferRole role) { return static_cast<std::size_t>(role); }
struct BufferRequirement { std::uint64_t bytes=0; std::uint64_t alignment=1; };
struct ExecutionAccounting {
    std::array<BufferRequirement,buffer_count> buffers;
    std::uint64_t used_device_bytes=0, scratch_bytes=0;
    std::uint64_t source_checkpoint_bytes=0; // Host staging source, not device-resident twice.
    std::uint64_t preparation_elements=0, preparation_groups=0;
    std::uint64_t useful_products=0, issued_products=0, matrix_instructions=0;
    std::uint32_t grid_columns=0, grid_rows=0, preparation_blocks=0, weight_scale_blocks=0;
};
class ExecutionPlan {
public:
    [[nodiscard]] static ExecutionPlan build(const ExecutionRequest&);
    [[nodiscard]] const ExecutionRequest& request() const noexcept { return request_; }
    [[nodiscard]] const ExecutionAccounting& accounting() const noexcept { return accounting_; }
    [[nodiscard]] const std::string& identity() const noexcept { return identity_; }
private:
    ExecutionRequest request_;
    ExecutionAccounting accounting_;
    std::string identity_;
};
struct AllocationBinding {
    std::uintptr_t base=0;
    std::uint64_t bytes=0, identity=0, generation=0;
    DeviceIdentity device;
    std::uint64_t access_generation=0; // Advances before any new content/write epoch.
    bool operator==(const AllocationBinding&) const = default;
};
struct BufferBinding {
    AllocationBinding allocation;
    std::uint64_t offset=0, bytes=0;
    bool operator==(const BufferBinding&) const = default;
};
using ExecutionBindings=std::array<BufferBinding,buffer_count>;
// A trusted allocation owner pins the underlying device allocation. Its single
// scheduler excludes conflicting producers/consumers and advances access epochs
// before content reuse; stale completions then fail. Current metadata is checked
// at every stage; this is not an untrusted DMA or hardware-generation proof API.
class AllocationPin {
public:
    virtual ~AllocationPin()=default;
    [[nodiscard]] virtual AllocationBinding current_binding() const=0;
};
using AllocationPins=std::array<std::shared_ptr<const AllocationPin>,buffer_count>;
using AllocationSnapshots=std::array<AllocationBinding,buffer_count>;
[[nodiscard]] AllocationSnapshots snapshot_allocation_pins(const AllocationPins&);
void validate_allocation_snapshots(const AllocationPins&,const AllocationSnapshots&);
// Validates the complete allocation set (deduplicated arena allocations count
// once), all view extents/generations/devices/alignment and used-range aliases.
// Returns actual retained allocation bytes, including unused capacity.
[[nodiscard]] std::uint64_t validate_execution_bindings(const ExecutionPlan&,const ExecutionBindings&);
void validate_execution_pins(const ExecutionPlan&,const ExecutionBindings&,const AllocationPins&);
[[nodiscard]] std::uintptr_t buffer_address(const BufferBinding&);
struct CompletionEvidence {
    std::string execution_identity;
    BufferBinding destination;
    std::uint32_t device_status=0;
    bool stream_quiescent=false, destination_copy_complete=false;
    bool destination_poisoned=false;
    std::shared_ptr<const AllocationPin> destination_owner; // Survives successful scratch retirement.
};
// Only trusted launcher evidence with matching phase/shape/profile/destination
// generation and a successful final copy can authorize output consumption.
void validate_completed_output(const ExecutionPlan&,const ExecutionBindings&,const CompletionEvidence&);
} // namespace strix::integer_projection
