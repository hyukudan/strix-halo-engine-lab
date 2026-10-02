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

}  // namespace strix
