#pragma once

#include "strix/prefix_state_cache.hpp"
#include "strix/session/kv_pages.hpp"

namespace strix::session {

// Sizes supplied by the allocation owner. State kept in one allocation per
// slot (including a context-sized index) goes in whole_slot_bytes.
struct KvPageLayout {
    std::uint64_t page_tokens = 1;
    std::uint64_t page_bytes = 0;
    std::uint64_t whole_slot_bytes = 0;
};

class KvCapacity {
    struct State;
public:
    using Credit = PrefixStateCache::Reservation;
    struct Lease {
        // A retained handle may still own its backend, result or opaque arena.
        Credit retained;
        struct Execution {
            Execution(const Execution&) = delete;
            Execution& operator=(const Execution&) = delete;
            ~Execution();
            KvPages<Credit> kv;
        private:
            friend class KvCapacity;
            Execution(std::shared_ptr<State>, std::size_t, Credit, KvPages<Credit>);
            std::shared_ptr<State> state_;
            std::size_t slots_;
            Credit whole_slots_;
        };
        std::unique_ptr<Execution> execution;
    };
    KvCapacity(std::uint64_t budget, std::uint64_t fixed_bytes, std::size_t slots);
    enum class Admission { accepted, slot_limit, byte_limit };
    struct Result { Admission admission; Lease lease; };
    [[nodiscard]] Result reserve(KvPageLayout, std::uint64_t context, std::size_t slots,
                                 std::uint64_t retained_bytes);
    [[nodiscard]] std::uint64_t reserved_bytes() const;
    // Optional snapshots use the same strict byte owner and its existing eviction policy.
    // Serialize capture/restore with admission. Only evictable entries may
    // survive the visit. Keep lookup pins and allocation leases within it.
    void with_optional_cache(const std::function<void(PrefixStateCache&)>&) const;
private:
    std::shared_ptr<State> state_;
};

} // namespace strix::session
