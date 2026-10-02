#include "strix/sparse_lookup.hpp"

#include <algorithm>
#include <limits>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strix {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("sparse lookup addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("sparse lookup byte calculation overflow");
    }
    return left * right;
}

}  // namespace

SparseLookupPlan build_sparse_lookup_plan(
    const SparseLookupRequest& request) {
    if (request.table_rows == 0 || request.row_bytes == 0) {
        throw std::invalid_argument(
            "sparse lookup table rows and row bytes must be non-zero");
    }
    if (request.max_inflight_ios == 0) {
        throw std::invalid_argument(
            "sparse lookup max inflight I/O must be non-zero");
    }

    SparseLookupPlan plan;
    plan.requests =
        static_cast<std::uint64_t>(request.row_ids.size());
    plan.requested_bytes =
        checked_mul(plan.requests, request.row_bytes);
    plan.cache_capacity_rows = request.cache_capacity_rows;

    std::unordered_set<std::uint64_t> unique;
    unique.reserve(request.row_ids.size());

    for (const auto row : request.row_ids) {
        if (row >= request.table_rows) {
            throw std::invalid_argument(
                "sparse lookup row is outside the table");
        }
        unique.insert(row);
    }

    plan.unique_rows =
        static_cast<std::uint64_t>(unique.size());
    plan.duplicate_requests =
        plan.requests - plan.unique_rows;
    plan.unique_bytes =
        checked_mul(plan.unique_rows, request.row_bytes);

    std::vector<std::uint64_t> sorted(
        unique.begin(), unique.end());
    std::sort(sorted.begin(), sorted.end());

    if (!sorted.empty()) {
        std::uint64_t run_first = sorted.front();
        std::uint64_t previous = sorted.front();

        for (std::size_t index = 1;
             index < sorted.size();
             ++index) {
            const auto row = sorted[index];
            if (row == previous + 1U) {
                previous = row;
                continue;
            }

            const auto count = previous - run_first + 1U;
            plan.runs.push_back(
                SparseLookupRun{run_first, count});
            plan.largest_run_rows =
                std::max(plan.largest_run_rows, count);
            run_first = row;
            previous = row;
        }

        const auto count = previous - run_first + 1U;
        plan.runs.push_back(
            SparseLookupRun{run_first, count});
        plan.largest_run_rows =
            std::max(plan.largest_run_rows, count);
    }

    plan.contiguous_runs =
        static_cast<std::uint64_t>(plan.runs.size());

    plan.max_inflight_ios =
        request.max_inflight_ios;
    for (const auto& run : plan.runs) {
        const auto chunk_rows =
            request.max_rows_per_io == 0
                ? run.row_count
                : request.max_rows_per_io;
        std::uint64_t consumed = 0;
        while (consumed < run.row_count) {
            const auto remaining =
                run.row_count - consumed;
            const auto rows =
                std::min(chunk_rows, remaining);
            const auto first_row =
                checked_add(run.first_row, consumed);
            const auto byte_offset =
                checked_mul(first_row, request.row_bytes);
            const auto byte_count =
                checked_mul(rows, request.row_bytes);
            plan.io_plan.push_back(
                SparseLookupIoOperation{
                    first_row,
                    rows,
                    byte_offset,
                    byte_count,
                });
            plan.largest_io_rows =
                std::max(plan.largest_io_rows, rows);
            plan.largest_io_bytes =
                std::max(plan.largest_io_bytes, byte_count);
            consumed =
                checked_add(consumed, rows);
        }
    }
    plan.io_operations =
        static_cast<std::uint64_t>(plan.io_plan.size());
    plan.io_waves =
        plan.io_operations == 0
            ? 0
            : (plan.io_operations +
               plan.max_inflight_ios - 1ULL) /
                  plan.max_inflight_ios;

    if (request.cache_capacity_rows != 0) {
        using Lru = std::list<std::uint64_t>;
        Lru lru;
        std::unordered_map<std::uint64_t, Lru::iterator> positions;
        positions.reserve(
            static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    request.cache_capacity_rows,
                    plan.unique_rows)));

        for (const auto row : request.row_ids) {
            const auto found = positions.find(row);
            if (found != positions.end()) {
                ++plan.cache_hits;
                lru.splice(lru.begin(), lru, found->second);
                found->second = lru.begin();
                continue;
            }

            ++plan.cache_misses;
            if (lru.size() >= request.cache_capacity_rows) {
                const auto victim = lru.back();
                positions.erase(victim);
                lru.pop_back();
            }
            lru.push_front(row);
            positions[row] = lru.begin();
        }
    } else {
        plan.cache_misses = plan.requests;
    }

    plan.cache_read_bytes =
        checked_mul(plan.cache_misses, request.row_bytes);
    plan.cache_hit_rate =
        plan.requests == 0
            ? 0.0
            : static_cast<double>(plan.cache_hits) /
                  static_cast<double>(plan.requests);

    return plan;
}

}  // namespace strix
