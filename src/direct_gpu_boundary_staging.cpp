#include "strix/direct_gpu_boundary_staging.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
using transport::TransportBufferView;
bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation &&
        a.domain == b.domain && a.access == b.access;
}
bool same_descriptor(const transport::TransferDescriptor& a, const transport::TransferDescriptor& b) {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch && a.model_plan_id == b.model_plan_id &&
        a.request_id == b.request_id && a.generation == b.generation && a.boundary_id == b.boundary_id && a.sequence == b.sequence &&
        a.shape_fingerprint == b.shape_fingerprint && a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
bool same_boundary(const DistributedExecutionBoundary& a, const DistributedExecutionBoundary& b) {
    return a.boundary_id == b.boundary_id && a.source_step == b.source_step && a.destination_step == b.destination_step &&
        a.source_node == b.source_node && a.destination_node == b.destination_node && a.payload_bytes == b.payload_bytes && a.payloads == b.payloads;
}
bool same_lease(const transport::DirectTransportLease& a, const transport::DirectTransportLease& b) {
    return a.physical_link_id == b.physical_link_id && a.connection_epoch == b.connection_epoch && a.pool_binding_identity == b.pool_binding_identity &&
        a.link_index == b.link_index && a.lease_generation == b.lease_generation;
}
bool valid_range(std::uintptr_t address, std::uint64_t bytes) {
    return address && bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
bool overlaps(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
} // namespace

struct DirectGpuBoundaryStaging::Impl : std::enable_shared_from_this<Impl> {
    struct Active {
        DistributedExecutionBoundary boundary;
        DistributedBoundaryTransfer transfer;
        transport::TransferDescriptor physical;
        TransportBufferView staged;
        std::uint64_t serial = 0;
        bool consumed = false, finish_issued = false;
    };
    struct Completion final : DistributedInboundCompletion {
        std::shared_ptr<Impl> owner;
        std::uint64_t serial;
        DistributedInboundCompletionStatus status = DistributedInboundCompletionStatus::pending;
        Completion(std::shared_ptr<Impl> p, std::uint64_t s) : owner(std::move(p)), serial(s) {}
        DistributedInboundCompletionStatus poll() override {
            if (status != DistributedInboundCompletionStatus::pending) return status;
            try { owner->complete(serial); status = DistributedInboundCompletionStatus::complete; }
            catch (...) { owner->fail(); status = DistributedInboundCompletionStatus::failed; }
            return status;
        }
    };
    DistributedPlacementPlan placement;
    std::uint32_t endpoint_node = 0;
    DistributedRuntimeRequest request;
    transport::DirectTransportLease lease;
    std::shared_ptr<DirectGpuBoundaryMemoryOwner> memory;
    std::shared_ptr<transport::DirectGpuStreamGpu> gpu;
    std::shared_ptr<transport::DirectGpuBoundaryReceiveIo> receive;
    std::vector<ExecutionBufferAddress> buffers;
    transport::DirectGpuExportBinding exported;
    std::optional<Active> active;
    std::uint64_t last_tx = 0, last_rx = 0, next_serial = 1;
    bool failed = false;
    void fail() noexcept { failed = true; gpu->request_stop(); }
    void current() {
        if (failed) throw std::logic_error("GPU boundary staging is failed");
        gpu->require_current(); receive->require_current();
        const auto observed = memory->current_buffers();
        if (observed.size() != buffers.size()) throw std::runtime_error("GPU boundary memory inventory changed");
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            const auto& a = observed[i]; const auto& b = buffers[i];
            if (a.use != b.use || a.address != b.address || a.bytes != b.bytes || a.allocation_generation != b.allocation_generation) {
                throw std::runtime_error("GPU boundary arena binding or generation changed");
            }
        }
        const auto& now = gpu->exported_binding();
        if (now.device != exported.device || now.pool_binding_identity != exported.pool_binding_identity ||
            !same_view(now.pool.tx, exported.pool.tx) || !same_view(now.pool.rx, exported.pool.rx)) {
            throw std::runtime_error("GPU boundary exported pool changed");
        }
    }
    std::vector<TransportBufferView> segments(const DistributedExecutionBoundary& boundary) const {
        std::vector<TransportBufferView> result;
        if (boundary.payloads.empty() || boundary.payloads.size() > 64) throw std::invalid_argument("GPU boundary requires bounded ordered execution payloads");
        std::uint64_t total = 0;
        for (const auto& use : boundary.payloads) {
            const auto found = std::find_if(buffers.begin(), buffers.end(), [&](const auto& b) { return b.use == use; });
            if (found == buffers.end() || !found->allocation_generation || !valid_range(found->address, found->bytes) ||
                total > boundary.payload_bytes || found->bytes > boundary.payload_bytes - total) {
                throw std::invalid_argument("GPU boundary payload lacks exact resident arena extent");
            }
            TransportBufferView view{found->address, found->bytes, found->allocation_generation,
                transport::TransportMemoryDomain::gpu_device, transport::TransportBufferAccess::read_write};
            if (overlaps(view, exported.pool.tx) || overlaps(view, exported.pool.rx) ||
                std::any_of(result.begin(), result.end(), [&](const auto& other) { return overlaps(view, other); })) {
                throw std::invalid_argument("GPU boundary payload aliases another live payload or transport pool");
            }
            total += view.bytes; result.push_back(view);
        }
        if (total != boundary.payload_bytes) throw std::invalid_argument("GPU boundary ordered payloads do not cover descriptor bytes");
        return result;
    }
    transport::TransferDescriptor validate(const DistributedExecutionBoundary& boundary, const DistributedBoundaryTransfer& transfer, bool inbound) {
        current();
        if (std::none_of(placement.boundaries.begin(), placement.boundaries.end(), [&](const auto& b) { return same_boundary(b, boundary); }) ||
            (inbound ? boundary.destination_node : boundary.source_node) != endpoint_node ||
            transfer.flow.direction != (inbound ? transport::LinkDirection::inbound : transport::LinkDirection::outbound) ||
            transfer.link_plan.kind != transport::PlanKind::single_link || transfer.link_plan.slices.size() != 1 || transfer.native_slices.size() != 1 ||
            transfer.link_plan.slices[0].link_index != lease.link_index || transfer.link_plan.slices[0].byte_offset != 0 ||
            transfer.link_plan.slices[0].byte_length != boundary.payload_bytes || transfer.native_slices[0].connection_epoch != lease.connection_epoch ||
            !transfer.native_slices[0].sequence || transfer.native_slices[0].sequence == std::numeric_limits<std::uint64_t>::max() ||
            !transfer.descriptor.sequence || transfer.descriptor.sequence == std::numeric_limits<std::uint64_t>::max()) {
            throw std::invalid_argument("GPU staging requires exact C1 placement and physical link slice");
        }
        const auto expected = make_distributed_boundary_descriptor(placement, boundary, request.connection_epoch,
            request.request_id, request.generation, transfer.descriptor.sequence, request.dtype_id);
        if (!same_descriptor(expected, transfer.descriptor)) throw std::invalid_argument("GPU boundary descriptor is outside request authority");
        const auto expected_flow = classify_distributed_boundary_flow(placement, boundary, {}, endpoint_node);
        if (transfer.flow.flow_id != expected_flow.flow_id || transfer.flow.priority != expected_flow.priority) throw std::invalid_argument("GPU boundary flow identity is foreign");
        auto physical = expected; physical.connection_epoch = lease.connection_epoch; physical.sequence = transfer.native_slices[0].sequence;
        return physical;
    }
    TransportBufferView pool_view(bool inbound, std::uint64_t bytes) const {
        auto view = inbound ? exported.pool.rx : exported.pool.tx;
        if (view.bytes <= 4096 || bytes > view.bytes - 4096) throw std::length_error("GPU boundary payload would overlap reserved terminal frame");
        view.bytes = bytes; return view;
    }
    void complete(std::uint64_t serial) {
        current();
        if (!active || active->serial != serial || !active->consumed || !active->finish_issued) throw std::logic_error("GPU receive completion is stale");
        const auto proof = receive->complete_receive_consumer(active->physical);
        const auto& op = proof.operation;
        if (proof.visibility != transport::DirectDriverVisibility::consumer_release_complete || !op.receive_path || !op.handle ||
            !op.imported.import_generation || !op.imported.device.device_instance_generation || op.imported.device.device_identity.empty() ||
            !same_lease(op.imported.lease, lease) || !same_view(op.imported.pool.tx, exported.pool.tx) || !same_view(op.imported.pool.rx, exported.pool.rx) ||
            !same_view(op.buffer, active->staged) || !same_descriptor(op.descriptor, active->physical)) {
            throw std::runtime_error("GPU boundary consumer completion proof is foreign or incomplete");
        }
        current(); last_rx = active->transfer.descriptor.sequence; active.reset();
    }
};

DirectGpuBoundaryStaging::DirectGpuBoundaryStaging(DistributedPlacementPlan placement, DistributedRuntimeRequest request,
    transport::DirectTransportLease lease, std::shared_ptr<DirectGpuBoundaryMemoryOwner> memory,
    std::shared_ptr<transport::DirectGpuStreamGpu> gpu, std::shared_ptr<transport::DirectGpuBoundaryReceiveIo> receive)
    : DirectGpuBoundaryStaging(std::move(placement), request, std::move(lease), std::move(memory),
        std::move(gpu), std::move(receive), 0U, false) {}

DirectGpuBoundaryStaging::DirectGpuBoundaryStaging(DistributedPlacementPlan placement, DistributedRuntimeRequest request,
    transport::DirectTransportLease lease, std::shared_ptr<DirectGpuBoundaryMemoryOwner> memory,
    std::shared_ptr<transport::DirectGpuStreamGpu> gpu, std::shared_ptr<transport::DirectGpuBoundaryReceiveIo> receive,
    std::uint32_t endpoint_node)
    : DirectGpuBoundaryStaging(std::move(placement), request, std::move(lease), std::move(memory),
        std::move(gpu), std::move(receive), endpoint_node, true) {}

DirectGpuBoundaryStaging::DirectGpuBoundaryStaging(DistributedPlacementPlan placement, DistributedRuntimeRequest request,
    transport::DirectTransportLease lease, std::shared_ptr<DirectGpuBoundaryMemoryOwner> memory,
    std::shared_ptr<transport::DirectGpuStreamGpu> gpu, std::shared_ptr<transport::DirectGpuBoundaryReceiveIo> receive,
    std::uint32_t endpoint_node, bool explicit_endpoint)
    : impl_(std::make_shared<Impl>()) {
    auto& i = *impl_;
    if (!explicit_endpoint) endpoint_node = placement.local_node;
    if (explicit_endpoint && (endpoint_node > 1U || std::none_of(placement.boundaries.begin(), placement.boundaries.end(),
        [&](const auto& b) {return b.source_node == endpoint_node || b.destination_node == endpoint_node;}) ||
        std::any_of(placement.boundaries.begin(), placement.boundaries.end(),
            [](const auto& b) {return b.source_node>1U || b.destination_node>1U;})))
        throw std::invalid_argument("GPU boundary endpoint is outside the two-peer placement");
    if (!placement.executable || !placement.plan_identity || !placement.compiled_execution_identity ||
        !placement.manifest_layout_identity || !placement.tensor_access_identity || placement.boundaries.empty() || placement.boundaries.size() > 64 ||
        !request.connection_epoch || !request.request_id || !request.generation || !request.dtype_id || !memory || !gpu || !receive ||
        lease.physical_link_id.empty() || !lease.connection_epoch || !lease.pool_binding_identity || !lease.lease_generation) {
        throw std::invalid_argument("GPU boundary staging requires exact request, lease and retained owners");
    }
    i.endpoint_node = endpoint_node;
    i.placement = std::move(placement); i.request = request; i.lease = std::move(lease); i.memory = std::move(memory);
    i.gpu = std::move(gpu); i.receive = std::move(receive); i.exported = i.gpu->exported_binding(); i.buffers = i.memory->current_buffers();
    if (i.buffers.empty() || i.buffers.size() > 4096 || i.exported.pool_binding_identity != i.lease.pool_binding_identity ||
        !valid_range(i.exported.pool.tx.address, i.exported.pool.tx.bytes) || !valid_range(i.exported.pool.rx.address, i.exported.pool.rx.bytes) ||
        !i.exported.pool.tx.allocation_generation || !i.exported.pool.rx.allocation_generation ||
        i.exported.pool.tx.domain != transport::TransportMemoryDomain::gpu_transport_pool || i.exported.pool.rx.domain != transport::TransportMemoryDomain::gpu_transport_pool ||
        i.exported.pool.tx.access != transport::TransportBufferAccess::read_write || i.exported.pool.rx.access != transport::TransportBufferAccess::read_write ||
        overlaps(i.exported.pool.tx, i.exported.pool.rx)) throw std::invalid_argument("GPU boundary pool/inventory is incompatible");
    for (std::size_t n = 0; n < i.buffers.size(); ++n) for (std::size_t m = 0; m < n; ++m) {
        if (i.buffers[n].use == i.buffers[m].use) throw std::invalid_argument("GPU boundary memory inventory has duplicate execution identities");
    }
    for (const auto& b : i.placement.boundaries) if (b.source_node == i.endpoint_node || b.destination_node == i.endpoint_node) {
        if (!b.boundary_id || !b.payload_bytes || b.source_node == b.destination_node) throw std::invalid_argument("GPU boundary placement is invalid");
        static_cast<void>(i.segments(b)); static_cast<void>(i.pool_view(false, b.payload_bytes)); static_cast<void>(i.pool_view(true, b.payload_bytes));
    }
    i.current();
}
DirectGpuBoundaryStaging::~DirectGpuBoundaryStaging() = default;
TransportBufferView DirectGpuBoundaryStaging::stage_outbound(const DistributedExecutionBoundary& boundary, const DistributedBoundaryTransfer& transfer) {
    try {
        auto& i = *impl_; const auto physical = i.validate(boundary, transfer, false);
        if (transfer.descriptor.sequence <= i.last_tx) throw std::invalid_argument("GPU outbound logical sequence replay");
        const auto parts = i.segments(boundary); const auto view = i.pool_view(false, boundary.payload_bytes);
        i.gpu->stage_outbound_segments(physical, parts, view); i.current(); i.last_tx = transfer.descriptor.sequence; return view;
    } catch (...) { impl_->fail(); throw; }
}
TransportBufferView DirectGpuBoundaryStaging::stage_inbound(const DistributedExecutionBoundary& boundary, const DistributedBoundaryTransfer& transfer) {
    try {
        auto& i = *impl_; const auto physical = i.validate(boundary, transfer, true);
        if (i.active || transfer.descriptor.sequence <= i.last_rx || i.next_serial == std::numeric_limits<std::uint64_t>::max()) throw std::invalid_argument("GPU inbound is active or replayed");
        static_cast<void>(i.segments(boundary)); const auto view = i.pool_view(true, boundary.payload_bytes);
        i.active = Impl::Active{boundary, transfer, physical, view, i.next_serial++, false, false}; return view;
    } catch (...) { impl_->fail(); throw; }
}
void DirectGpuBoundaryStaging::consume_inbound(const DistributedExecutionBoundary& boundary, const TransportBufferView& staged) {
    try {
        auto& i = *impl_; i.current();
        if (!i.active || i.active->consumed || !same_boundary(boundary, i.active->boundary) || !same_view(staged, i.active->staged)) throw std::invalid_argument("GPU inbound consumer binding is foreign");
        const auto acquired = i.receive->acquired_receive_view(i.active->physical);
        if (!valid_range(acquired.address, acquired.bytes) || acquired.bytes != boundary.payload_bytes || !acquired.allocation_generation ||
            acquired.domain != transport::TransportMemoryDomain::gpu_device || acquired.access != transport::TransportBufferAccess::read_write ||
            overlaps(acquired, i.exported.pool.tx) || overlaps(acquired, i.exported.pool.rx)) throw std::runtime_error("GPU received payload lacks separate acquired destination");
        const auto parts = i.segments(boundary); i.receive->consume_received_segments(i.active->physical, parts);
        i.current(); i.active->consumed = true;
    } catch (...) { impl_->fail(); throw; }
}
std::unique_ptr<DistributedInboundCompletion> DirectGpuBoundaryStaging::finish_inbound(const DistributedExecutionBoundary& boundary, const DistributedBoundaryTransfer& transfer) {
    try {
        auto& i = *impl_; const auto physical = i.validate(boundary, transfer, true);
        if (!i.active || !i.active->consumed || i.active->finish_issued || !same_boundary(boundary, i.active->boundary) ||
            !same_descriptor(physical, i.active->physical) || !same_descriptor(transfer.descriptor, i.active->transfer.descriptor)) throw std::invalid_argument("GPU inbound finish is foreign or premature");
        auto completion = std::make_unique<Impl::Completion>(impl_, i.active->serial); i.active->finish_issued = true; return completion;
    } catch (...) { impl_->fail(); throw; }
}
bool DirectGpuBoundaryStaging::failed() const noexcept { return impl_->failed; }
} // namespace strix
