#pragma once

#include "strix/sparse_lookup.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class SparseLookupIoSource {
public:
    virtual ~SparseLookupIoSource() = default;

    virtual void read(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) = 0;
};

struct SparseLookupIoExecution {
    std::uint64_t operations_completed = 0;
    std::uint64_t peak_inflight = 0;
    std::uint64_t bytes_read = 0;
    std::vector<std::vector<std::byte>> operation_payloads;
};

[[nodiscard]] SparseLookupIoExecution execute_sparse_lookup_io_plan(
    const SparseLookupPlan& plan,
    SparseLookupIoSource& source);

// Explicit CPU budget, 1..8 executing helpers INCLUDING this calling thread.
// The caller processes one operation while at most budget-1 async readers run.
// Inflight I/O is independently capped by plan.max_inflight_ios. Source-owned
// internal threads are outside this contract; source.read must be concurrently
// safe and retain its buffers until it returns. All submitted readers are joined
// on every exit; errors are selected in logical operation order after joining.
// No persistent pool, affinity policy, busy polling or GPU lifetime publication.
[[nodiscard]] SparseLookupIoExecution execute_sparse_lookup_io_plan(
    const SparseLookupPlan& plan, SparseLookupIoSource& source,
    std::uint32_t cpu_worker_budget);

}  // namespace strix
