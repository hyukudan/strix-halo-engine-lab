#pragma once

#include "strix/direct_transport_native_backend.hpp"

// Synthetic portable declarations exercise the gate only. They are neither
// a real kernel ABI nor evidence of native GPU or transport qualification.
inline strix::transport::DirectNativeInterfaceIdentity synthetic_native_identity(
    const strix::transport::DirectLinkLifecycle& link) {
    return {1, "synthetic-kernel", "synthetic-driver-build", "synthetic-abi-v1",
            "synthetic-local-node", "synthetic-peer-node",
            link.physical_link_id, link.connection_epoch};
}

inline strix::transport::DirectNativeCapabilities synthetic_native_capabilities(
    const strix::transport::DirectNativeInterfaceIdentity& identity) {
    strix::transport::DirectNativeCapabilities value;
    value.dma_buf_export = true;
    value.dma_buf_import = true;
    value.uncached_gpu_pool = true;
    value.completion_event_source = true;
    value.system_scope_gpu_commit = true;
    value.remote_consumer_credit = true;
    value.receive_repost = true;
    value.query_status = strix::transport::DirectNativeQueryStatus::verified;
    value.observed_interface = identity;
    value.imported_tx_rx_ownership_verified = true;
    value.completion_ownership_verified = true;
    value.external_dma_visibility_qualified = true;
    return value;
}
