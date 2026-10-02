#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct SparseLookupRequest {
    std::uint64_t table_rows = 0;
    std::uint64_t row_bytes = 0;
    std::uint64_t cache_capacity_rows = 0;
    std::uint64_t max_rows_per_io = 0;
    std::uint64_t max_inflight_ios = 1;
    std::span<const std::uint64_t> row_ids;
};

struct SparseLookupRun {
    std::uint64_t first_row = 0;
    std::uint64_t row_count = 0;
};

struct SparseLookupIoOperation {
    std::uint64_t first_row = 0;
    std::uint64_t row_count = 0;
    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
};

struct SparseLookupPlan {
    std::uint64_t requests = 0;
    std::uint64_t unique_rows = 0;
    std::uint64_t duplicate_requests = 0;

    std::uint64_t requested_bytes = 0;
    std::uint64_t unique_bytes = 0;

    std::uint64_t contiguous_runs = 0;
    std::uint64_t largest_run_rows = 0;
    std::vector<SparseLookupRun> runs;

    std::uint64_t io_operations = 0;
    std::uint64_t largest_io_rows = 0;
    std::uint64_t largest_io_bytes = 0;
    std::uint64_t max_inflight_ios = 1;
    std::uint64_t io_waves = 0;
    std::vector<SparseLookupIoOperation> io_plan;

    std::uint64_t cache_capacity_rows = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_read_bytes = 0;
    double cache_hit_rate = 0.0;
};

[[nodiscard]] SparseLookupPlan build_sparse_lookup_plan(
    const SparseLookupRequest& request);

}  // namespace strix
