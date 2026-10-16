#pragma once
#include "hip_decode_receipt_json.hpp"
#include "strix/uma_owner_inventory.hpp"
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace strix::decode_receipt {
// Actual retained view snapshots supplied by the model/runtime owner producers.
// This is a covered subset, never an exhaustive allocation or physical census.
inline std::string runtime_owner_inventory_receipt(const UmaOwnerInventory& combined,
    const UmaOwnerInventory& runtime, bool deferred_status_present) {
    std::ostringstream out;out.imbue(std::locale::classic());
    out << "  \"runtime_backing_inventory\":{"
        "\"scope\":\"retained-model-and-base-scratch-state-virtual-backing-subset\","
        "\"boundary\":\"after-base-runtime-D1-catalog-before-optional-request-setup\","
        "\"complete_resource_census\":false,\"physical_fit\":null,"
        "\"physical_residency_independently_verified\":null,"
        "\"opaque_owners\":{\"D1\":\"" << (deferred_status_present?"unaccounted":"absent")
        << "\",\"PLE\":\"unaccounted\",\"catalog\":\"unaccounted\","
        "\"other\":\"unaccounted\",\"optional_request_workspaces\":\"outside-observation-boundary\"},"
        "\"covered_view_count\":" << combined.pinned_views.size()
        << ",\"covered_known_backing_count\":" << combined.known_backings.size()
        << ",\"covered_unknown_backing_views\":" << combined.unknown_backing_views
        << ",\"covered_logical_view_bytes\":" << combined.logical_view_bytes
        << ",\"covered_known_allocated_storage_bytes\":" << combined.known_allocated_storage_bytes
        << ",\"covered_known_file_mapping_bytes\":" << combined.known_file_mapping_bytes
        << ",\"covered_reservation_upper_bound_bytes\":" << combined.reservation_upper_bound_bytes
        << ",\"covered_unique_backing_reserved_bytes\":";
    if(combined.unique_backing_reserved_bytes) out << *combined.unique_backing_reserved_bytes;
    else out << "null";
    out << ",\"runtime_pools\":[";
    bool first=true;
    for(const auto& view:runtime.pinned_views) {
        if(view.view_offset>view.allocation_bytes || view.view_bytes>view.allocation_bytes-view.view_offset)
            throw std::invalid_argument("runtime inventory view exceeds declared live allocation");
        const char* space=nullptr;
        switch(view.address_space) {
        case UmaAddressSpace::host:space="host";break;
        case UmaAddressSpace::device:space="device";break;
        case UmaAddressSpace::unspecified:space="unspecified";break;
        default:throw std::invalid_argument("runtime inventory address space is invalid");
        }
        if(!first) out << ',';
        first=false;
        out << "{\"category\":\"" << escape_json_string(view.category) << "\",\"view_identity\":" << view.view_identity
            << ",\"address_space\":\"" << space << "\",\"allocation_bytes\":" << view.allocation_bytes
            << ",\"view_offset\":" << view.view_offset << ",\"view_bytes\":" << view.view_bytes
            << ",\"outside_view_reservation_bytes\":" << view.allocation_bytes-view.view_bytes
            << ",\"backing\":";
        if(view.backing.issuer && view.backing.allocation && view.backing.generation)
            out << "{\"issuer\":" << view.backing.issuer << ",\"allocation\":" << view.backing.allocation
                << ",\"generation\":" << view.backing.generation << '}';
        else out << "null";
        out << '}';
    }
    out << "]},\n";return out.str();
}
} // namespace strix::decode_receipt