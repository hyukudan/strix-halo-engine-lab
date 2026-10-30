#pragma once

#include "strix/uma_owner_inventory.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace strix::decode_cli {
inline std::uint64_t prefill_reservation_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("known prefill reservation overflow");
    return a + b;
}
// Shared with the pre-device resolver and the post-registration CLI ledger.
// Missing masks mean retained across all five phases.
inline std::array<std::uint64_t,5> prefill_owner_phase_peaks(
    const std::map<std::string,std::uint64_t>& owners,const std::map<std::string,unsigned>& masks,
    std::uint64_t reserve=0) {
    std::array<std::uint64_t,5> peaks{};peaks.fill(reserve);
    for(const auto& [name,bytes]:owners) {
        const auto mask=masks.contains(name)?masks.at(name):31U;
        for(unsigned i=0;i<peaks.size();++i)
            if(mask&(1U<<i)) peaks[i]=prefill_reservation_add(peaks[i],bytes);
    }
    return peaks;
}
struct MtpPrefillReservation {
    std::uint64_t sidecar_retained = 0, runtime = 0, target_capture = 0;
    std::uint64_t replay = 0, golden = 0;
    std::uint64_t window_arena = 0, window_capture = 0, window_checkpoint = 0, window_total = 0;
    std::uint32_t window_owners = 0;
    bool prefix = false;
};
struct PrefillReservationGrowth { std::uint64_t retained = 0, temporary = 0; };
// Caps bound owners; they are not allocations. The window factory enforces one
// combined arena cap across all R2/R4/R8 owners and four capture pools per owner.
// Prefix commit uses deferred GDN/PLE snapshots and allocates no full checkpoint.
// Sidecar setup has finished before the verified PLE drop and is not live here.
inline PrefillReservationGrowth mtp_prefill_reservation(const MtpPrefillReservation& m) {
    const auto add = prefill_reservation_add;
    auto retained = add(m.sidecar_retained, add(m.runtime, m.target_capture));
    if (m.window_owners) {
        auto window = m.window_arena;
        for (std::uint32_t owner = 0; owner < m.window_owners; ++owner)
            for (unsigned pool = 0; pool < 4U; ++pool) window = add(window, m.window_capture);
        if (!m.prefix) window = add(window, m.window_checkpoint);
        retained = add(retained, std::min(window, m.window_total));
    }
    return {retained, add(m.replay, m.golden)};
}
// Shared plain/MTP admission. Verified credit already equals dropped interior
// minus C+D+margin. Adding that bound or decode quota again would count it twice.
// Metadata is a separate owner. No observed physical/GTT quantity receives credit.
inline UmaReservationEstimate plan_known_prefill_reservation(const UmaOwnerInventory& live,
    std::uint64_t growth, std::uint64_t temporary, UmaReservationLimits limits,
    std::uint64_t verified_credit, std::uint64_t rolling_metadata,
    PrefillReservationGrowth mtp = {}) {
    return plan_uma_reservation_growth(live,
        prefill_reservation_add(growth, prefill_reservation_add(rolling_metadata, mtp.retained)),
        prefill_reservation_add(temporary, mtp.temporary), limits, verified_credit);
}
} // namespace strix::decode_cli
