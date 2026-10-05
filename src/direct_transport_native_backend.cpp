#include "strix/direct_transport_native_backend.hpp"

#include <stdexcept>

namespace strix::transport {

DirectNativeReadiness assess_direct_native_readiness(
    const BackendCapabilities& backend,
    const DirectNativeCapabilities& native) {
    DirectNativeReadiness result;
    const auto require = [&](bool condition, const char* blocker) {
        if (!condition) {
            result.blockers.emplace_back(blocker);
        }
    };

    require(backend.external_dma,
            "backend lacks external DMA support");
    require(backend.supports_gpu_transport_pool,
            "backend cannot bind GPU transport pools");
    require(backend.requires_exclusive_lease,
            "backend does not enforce exclusive direct-link leases");
    require(backend.explicit_receive_repost,
            "backend lacks explicit receive repost");
    require(backend.gpu_commit_record,
            "backend lacks GPU commit records");
    require(backend.buffer_mode ==
                TransportBufferMode::imported_gpu_dma_buf,
            "backend is not using imported GPU DMA-BUF memory");

    require(native.dma_buf_export,
            "native GPU DMA-BUF export is unavailable");
    require(native.dma_buf_import,
            "native transport DMA-BUF import is unavailable");
    require(native.uncached_gpu_pool,
            "native uncached GPU transport pool is unavailable");
    require(native.completion_event_source,
            "native completion event source is unavailable");
    require(native.system_scope_gpu_commit,
            "system-scope GPU commit primitive is unavailable");
    require(native.remote_consumer_credit,
            "remote consumer-complete credit is unavailable");
    require(native.receive_repost,
            "native receive repost is unavailable");

    result.ready = result.blockers.empty();
    return result;
}

void prepare_direct_native_link(
    DirectTransportNativeBackend& backend,
    const DirectTransportLeaseRegistry& leases,
    const DirectLinkLifecycle& link,
    const DirectTransportPoolMemoryBinding& pools,
    const DirectTransportLease& lease) {
    const auto readiness =
        assess_direct_native_readiness(
            backend.capabilities(),
            backend.direct_capabilities());
    if (!readiness.ready) {
        throw std::logic_error(
            "native direct transport backend is not capability-complete");
    }
    if (!leases.validate(lease, link, pools)) {
        throw std::invalid_argument(
            "native direct transport link requires a valid exclusive lease");
    }
    if (lease.link_index >= pools.links.size()) {
        throw std::out_of_range(
            "native direct transport lease link index exceeds pool binding");
    }

    backend.import_link_pool(
        lease,
        pools.links[lease.link_index]);
}

void publish_direct_native_commit(
    DirectTransportNativeBackend& backend,
    TransferHandle handle,
    const TransferDescriptor& descriptor,
    const GpuCommitRecord& record) {
    if (handle == 0 ||
        !validate_gpu_commit_record(record, descriptor)) {
        throw std::invalid_argument(
            "native direct transport commit is invalid");
    }
    backend.publish_gpu_commit(handle, record);
}

void repost_direct_native_receive(
    DirectTransportNativeBackend& backend,
    DirectReceiveCompletionBridge& receives,
    TransferHandle handle) {
    const auto& state =
        receives.state(handle);
    if (state.failed ||
        state.ownership != BufferOwnership::consumer_released) {
        throw std::logic_error(
            "native receive repost requires consumer-released ownership");
    }

    backend.repost_receive(handle);
    receives.submit_repost(handle);
}

bool observe_direct_native_event(
    DirectTransportNativeBackend& backend,
    DirectReceiveCompletionBridge& receives,
    DirectSendCompletionBridge& sends) {
    const auto event =
        backend.poll_direct_event();
    if (!event.has_value()) {
        return false;
    }
    if (event->completion.handle == 0) {
        throw std::invalid_argument(
            "native direct transport event has zero handle");
    }

    if (event->receive_path) {
        receives.observe(event->completion);
    } else {
        sends.observe(event->completion);
    }
    return true;
}

}  // namespace strix::transport
