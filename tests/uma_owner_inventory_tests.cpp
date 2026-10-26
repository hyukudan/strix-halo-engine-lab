#include "strix/uma_owner_inventory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace strix;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class Function> void invalid(Function&& work) {
    bool caught = false;
    try { work(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "invalid owner inventory was accepted");
}
UmaOwnerView view(std::uint64_t id, UmaBackingIdentity key, std::uint64_t extent,
                  std::uint64_t offset, std::uint64_t bytes) {
    return {id,key,UmaBackingKind::allocated_storage,"state",extent,offset,bytes,std::make_shared<int>(7)};
}
void aliases_and_copy() {
    const auto first=next_uma_owner_view_identity(),second=next_uma_owner_view_identity();
    require(first && second>first,"shared view identity issuer is not nonzero and monotonic");
    std::array<UmaOwnerView,3> records{view(1,{1,9,3},100,0,60),view(2,{1,9,3},100,40,60),view(3,{1,10,3},100,0,100)};
    // Host and GPU mappings share key1. Byte-identical copied data has key2.
    records[0].category = "CPU-state"; records[1].category = "GPU-state";
    const auto result = build_uma_owner_inventory(records);
    require(result.logical_view_bytes == 220 && result.unique_backing_reserved_bytes == 200 &&
            result.known_alias_view_bytes == 20 && result.known_covered_view_bytes == 200 &&
            result.reservation_upper_bound_bytes == 200 && result.known_backings.size() == 2,
            "aliases must be unioned once and copies counted separately");
    std::reverse(records.begin(), records.end());
    const auto reverse = build_uma_owner_inventory(records);
    require(reverse.unique_backing_reserved_bytes == result.unique_backing_reserved_bytes &&
            reverse.known_covered_view_bytes == result.known_covered_view_bytes,
            "inventory depends on input order");
}
void sparse_and_nested() {
    std::array<UmaOwnerView,5> records{view(1,{1,1,1},100,5,10),view(2,{1,1,1},100,10,2),
        view(3,{1,1,1},100,30,20),view(4,{1,1,1},100,50,10),view(5,{1,1,1},100,100,0)};
    const auto result = build_uma_owner_inventory(records);
    require(result.logical_view_bytes == 42 && result.known_covered_view_bytes == 40 &&
            result.known_alias_view_bytes == 2 && result.unique_backing_reserved_bytes == 100,
            "nested/disjoint/touching/empty view interval union is wrong");
}
void unknown_and_observed() {
    std::array<UmaOwnerView,3> records{view(1,{1,2,1},100,0,30),view(2,{},50,0,20),view(3,{},50,10,20)};
    records[0].kind = UmaBackingKind::file_mapping;
    UmaMemoryObservation observed;
    observed.observation_identity = 99; observed.mem_total_bytes = 1000; observed.mem_available_bytes = 400;
    observed.process_rss_bytes = 300; observed.gtt_reported_bytes = 300; observed.process_peak_bytes = 350;
    const auto result = build_uma_owner_inventory(records, observed);
    require(!result.unique_backing_reserved_bytes && result.unknown_backing_views == 2 &&
            result.reservation_upper_bound_bytes == 200 && result.known_file_mapping_bytes == 100 &&
            result.known_allocated_storage_bytes == 0 && result.observation.process_rss_bytes == 300 &&
            result.observation.gtt_reported_bytes == 300 && result.observation.mem_available_bytes == 400,
            "unknown backing must stay unknown and RSS/GTT must remain separate");
    const auto empty = build_uma_owner_inventory({});
    require(empty.unique_backing_reserved_bytes == 0 && !empty.observation.mem_total_bytes,
            "empty inventory must not invent observed capacity");
}
void identities_and_lifetime() {
    std::vector<UmaOwnerView> records{view(1,{1,1,1},10,0,10),view(2,{2,1,1},10,0,10),view(3,{1,1,2},10,0,10)};
    require(build_uma_owner_inventory(records).unique_backing_reserved_bytes == 30,
            "issuer or generation changed but distinct backing was merged");
    auto lease = std::make_shared<int>(9);
    std::weak_ptr<int> weak = lease;
    records.resize(1); records[0].allocation_pin = lease;
    {
        const auto snapshot = build_uma_owner_inventory(records);
        records.clear(); lease.reset();
        require(!weak.expired() && snapshot.pinned_views.size() == 1, "snapshot did not retain the actual lease");
    }
    require(weak.expired(), "snapshot leaked the lease");
}
void negative() {
    const auto original = view(1,{1,2,3},100,0,10);
    for (unsigned mutation = 0; mutation < 8; ++mutation) {
        auto record = original;
        if (mutation == 0) record.view_identity = 0;
        if (mutation == 1) record.allocation_pin.reset();
        if (mutation == 2) record.backing.generation = 0;
        if (mutation == 3) record.category.clear();
        if (mutation == 4) record.allocation_bytes = 0;
        if (mutation == 5) record.view_offset = 101;
        if (mutation == 6) { record.view_offset = 99; record.view_bytes = 2; }
        if (mutation == 7) record.kind = static_cast<UmaBackingKind>(99);
        invalid([&] { static_cast<void>(build_uma_owner_inventory(std::span<const UmaOwnerView>(&record,1))); });
    }
    std::array<UmaOwnerView,2> duplicates{original,original};
    invalid([&] { static_cast<void>(build_uma_owner_inventory(duplicates)); });
    duplicates[1].view_identity = 2; duplicates[1].allocation_bytes = 101;
    invalid([&] { static_cast<void>(build_uma_owner_inventory(duplicates)); });
    duplicates[1].allocation_bytes = 100; duplicates[1].kind = UmaBackingKind::file_mapping;
    invalid([&] { static_cast<void>(build_uma_owner_inventory(duplicates)); });
    UmaMemoryObservation bad; bad.mem_available_bytes = 1;
    invalid([&] { static_cast<void>(build_uma_owner_inventory({},bad)); });
    bad.observation_identity = 1; bad.mem_total_bytes = 0;
    invalid([&] { static_cast<void>(build_uma_owner_inventory({},bad)); });
    bad.mem_total_bytes = 1; bad.mem_available_bytes = 2;
    invalid([&] { static_cast<void>(build_uma_owner_inventory({},bad)); });
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    std::array<UmaOwnerView,2> overflow{view(1,{1,1,1},maximum,0,0),view(2,{1,2,1},1,0,0)};
    bool caught = false;
    try { static_cast<void>(build_uma_owner_inventory(overflow)); } catch (const std::overflow_error&) { caught = true; }
    require(caught, "reserved-byte sum overflow was accepted");
}
void address_evidence_and_reservation_budget() {
    auto mapping = view(20,{7,8,9},4096,0,4096); mapping.kind = UmaBackingKind::file_mapping;
    mapping.category = "weights.mapping"; mapping.address_space = UmaAddressSpace::host; mapping.address = 0x10000;
    auto registered = mapping; registered.view_identity = 21; registered.category = "weights.registered-alias";
    registered.address_space = UmaAddressSpace::device; registered.address = 0x80000;
    auto repack = view(22,{7,10,9},1024,0,1024); repack.category = "repack";
    repack.address_space = UmaAddressSpace::device; repack.address = 0x90000;
    auto staging = view(23,{7,11,9},256,0,256); staging.category = "staging";
    const std::array retained{mapping,registered,repack};
    const std::array setup{staging};
    const auto plan = build_uma_reservation_plan(retained,setup,{5120,5376});
    require(plan.retained.unique_backing_reserved_bytes == 5120 && plan.setup_peak.unique_backing_reserved_bytes == 5376 &&
        plan.retained.logical_view_bytes == 9216 && plan.retained.known_alias_view_bytes == 4096,
        "CPU/device aliases inflated retained or peak reservation; repack/staging copy was omitted");
    require(plan.retained.pinned_views[1].address == 0x80000 && !plan.retained.observation.process_peak_bytes,
        "actual alias evidence was lost or declared peak became observed peak");
    const std::array source_views{mapping,registered};
    const auto before_repack = build_uma_owner_inventory(source_views);
    const auto estimated = plan_uma_reservation_growth(before_repack,1024,256,{5120,5376});
    require(estimated.retained_reserved_bytes == plan.retained.reservation_upper_bound_bytes &&
            estimated.setup_peak_reserved_bytes == plan.setup_peak.reservation_upper_bound_bytes,
        "pre-allocation repack/staging estimate disagrees with actual retained/setup backing inventory");
    invalid([&] { static_cast<void>(plan_uma_reservation_growth(before_repack,1024,256,{5120,5375})); });
    bool overflow = false;
    try { static_cast<void>(plan_uma_reservation_growth(before_repack,std::numeric_limits<std::uint64_t>::max(),0)); }
    catch (const std::overflow_error&) { overflow = true; }
    require(overflow, "pre-allocation reservation growth overflow was accepted");
    invalid([&] { static_cast<void>(build_uma_reservation_plan(retained,setup,{5119,5376})); });
    invalid([&] { static_cast<void>(build_uma_reservation_plan(retained,setup,{5120,5375})); });
    // Repeated exports of the same backing have distinct view IDs but one
    // reservation, including at the setup peak.
    auto setup_alias = registered; setup_alias.view_identity = 24;
    const std::array setup_aliases{setup_alias};
    require(build_uma_reservation_plan(retained,setup_aliases).setup_peak.unique_backing_reserved_bytes == 5120,
        "setup alias was counted as a new backing reservation");
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto bad = mapping;
        if (mutation == 0) bad.address = 0;
        if (mutation == 1) bad.address_space = UmaAddressSpace::unspecified;
        if (mutation == 2) bad.address_space = static_cast<UmaAddressSpace>(99);
        if (mutation == 3) bad.address = std::numeric_limits<std::uintptr_t>::max() - 1U;
        invalid([&] { static_cast<void>(build_uma_owner_inventory(std::span<const UmaOwnerView>(&bad,1))); });
    }
    auto unknown = staging; unknown.backing = {};
    const std::array unknown_setup{unknown};
    const auto conservative = build_uma_reservation_plan(retained,unknown_setup);
    require(!conservative.setup_peak.unique_backing_reserved_bytes && conservative.setup_peak.reservation_upper_bound_bytes == 5376,
        "unknown setup owner acquired false physical identity or escaped the conservative budget");
    std::weak_ptr<const void> lease;
    UmaReservationPlan pinned;
    {
        auto only = view(30,{7,12,9},128,0,128);
        lease = only.allocation_pin;
        const std::array temporary{only};
        pinned = build_uma_reservation_plan({},temporary);
    }
    require(!lease.expired() && pinned.setup_peak.pinned_views.size() == 1,
        "setup budget snapshot discarded its consumer lease");
    pinned = {};
    require(lease.expired(), "setup budget snapshot leaked its last consumer lease");
}
} // namespace
int main() try {
    aliases_and_copy(); sparse_and_nested(); unknown_and_observed(); identities_and_lifetime(); negative();
    address_evidence_and_reservation_budget();
    UmaOwnerInventory prompt_base;
    prompt_base.reservation_upper_bound_bytes = 70U;
    const auto prompt = plan_uma_reservation_growth(prompt_base, 8U + 2U + 4U, 1U, {85U,85U});
    require(prompt.retained_reserved_bytes == 84U && prompt.setup_peak_reserved_bytes == 85U,
        "prompt planner counts model once, workspace, both metadata copies, headroom and runtime payload");
    invalid([&] { (void)plan_uma_reservation_growth(prompt_base,14U,1U,{85U,84U}); });
    std::cout << "UMA owner inventory source fixtures completed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
