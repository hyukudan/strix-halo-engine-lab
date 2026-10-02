#include "strix/sparse_lookup.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const std::vector<std::uint64_t> rows{
        10, 11, 12, 50, 10, 51, 52, 53, 50, 90,
    };

    strix::SparseLookupRequest request;
    request.table_rows = 100;
    request.row_bytes = 320;
    request.cache_capacity_rows = 4;
    request.max_rows_per_io = 2;
    request.max_inflight_ios = 2;
    request.row_ids = rows;

    const auto plan =
        strix::build_sparse_lookup_plan(request);

    expect(plan.requests == 10,
           "lookup planner counts row requests");
    expect(plan.unique_rows == 8 &&
               plan.duplicate_requests == 2,
           "lookup planner counts unique and duplicate rows");
    expect(plan.requested_bytes == 3'200 &&
               plan.unique_bytes == 2'560,
           "lookup planner accounts row bytes");
    expect(plan.contiguous_runs == 3,
           "lookup planner coalesces adjacent unique rows");
    expect(plan.runs[0].first_row == 10 &&
               plan.runs[0].row_count == 3 &&
               plan.runs[1].first_row == 50 &&
               plan.runs[1].row_count == 4 &&
               plan.runs[2].first_row == 90 &&
               plan.runs[2].row_count == 1,
           "lookup planner emits deterministic sorted runs");
    expect(plan.largest_run_rows == 4,
           "lookup planner tracks largest contiguous run");
    expect(plan.io_operations == 5 &&
               plan.io_waves == 3 &&
               plan.largest_io_rows == 2 &&
               plan.largest_io_bytes == 640,
           "lookup planner bounds coalesced I/O and computes inflight waves");
    expect(plan.io_plan[0].first_row == 10 &&
               plan.io_plan[0].row_count == 2 &&
               plan.io_plan[0].byte_offset == 3'200 &&
               plan.io_plan[1].first_row == 12 &&
               plan.io_plan[1].row_count == 1 &&
               plan.io_plan[2].first_row == 50 &&
               plan.io_plan[2].row_count == 2 &&
               plan.io_plan[4].first_row == 90,
           "lookup I/O plan preserves deterministic sorted run order");
    expect(plan.cache_hits == 1 &&
               plan.cache_misses == 9,
           "bounded LRU simulation accounts eviction");
    expect(plan.cache_read_bytes == 2'880,
           "cache planner exposes physical row bytes read");
    expect(plan.cache_hit_rate == 0.1,
           "cache planner exposes hit rate");

    auto no_cache = request;
    no_cache.cache_capacity_rows = 0;
    no_cache.max_rows_per_io = 0;
    no_cache.max_inflight_ios = 4;
    const auto uncached =
        strix::build_sparse_lookup_plan(no_cache);
    expect(uncached.cache_hits == 0 &&
               uncached.cache_misses == uncached.requests &&
               uncached.cache_read_bytes == uncached.requested_bytes,
           "zero cache models one read per request");
    expect(uncached.io_operations == uncached.contiguous_runs &&
               uncached.io_waves == 1 &&
               uncached.largest_io_rows == uncached.largest_run_rows,
           "zero max rows leaves each contiguous run as one I/O");

    bool zero_inflight = false;
    try {
        auto broken = request;
        broken.max_inflight_ios = 0;
        static_cast<void>(
            strix::build_sparse_lookup_plan(broken));
    } catch (const std::invalid_argument&) {
        zero_inflight = true;
    }
    expect(zero_inflight,
           "lookup planner rejects zero inflight I/O depth");

    bool invalid = false;
    try {
        const std::vector<std::uint64_t> bad_rows{100};
        request.row_ids = bad_rows;
        static_cast<void>(
            strix::build_sparse_lookup_plan(request));
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid,
           "out-of-range sparse lookup row is rejected");

    if (failures == 0) {
        std::cout << "sparse lookup tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
