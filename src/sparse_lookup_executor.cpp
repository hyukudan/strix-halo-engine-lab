#include "strix/sparse_lookup_executor.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "sparse lookup execution byte count overflow");
    }
    return left + right;
}

void validate_io_plan(
    const SparseLookupPlan& plan) {
    if (plan.max_inflight_ios == 0) {
        throw std::invalid_argument(
            "sparse lookup execution requires non-zero inflight depth");
    }
    if (plan.io_operations !=
        static_cast<std::uint64_t>(
            plan.io_plan.size())) {
        throw std::invalid_argument(
            "sparse lookup I/O operation count does not match plan");
    }

    for (const auto& operation : plan.io_plan) {
        if (operation.row_count == 0 ||
            operation.byte_count == 0) {
            throw std::invalid_argument(
                "sparse lookup I/O operation must be non-empty");
        }
    }
}

}  // namespace

SparseLookupIoExecution execute_sparse_lookup_io_plan(
    const SparseLookupPlan& plan,
    SparseLookupIoSource& source) {
    validate_io_plan(plan);

    SparseLookupIoExecution execution;
    execution.operation_payloads.resize(
        plan.io_plan.size());

    std::size_t begin = 0;
    while (begin < plan.io_plan.size()) {
        const auto remaining =
            plan.io_plan.size() - begin;
        const auto width =
            std::min<std::size_t>(
                remaining,
                static_cast<std::size_t>(
                    plan.max_inflight_ios));

        execution.peak_inflight =
            std::max(
                execution.peak_inflight,
                static_cast<std::uint64_t>(
                    width));

        std::vector<std::future<std::vector<std::byte>>> futures;
        futures.reserve(width);

        for (std::size_t local = 0;
             local < width;
             ++local) {
            const auto operation =
                plan.io_plan[begin + local];
            futures.push_back(
                std::async(
                    std::launch::async,
                    [&source, operation] {
                        if (operation.byte_count >
                            static_cast<std::uint64_t>(
                                std::numeric_limits<std::size_t>::max())) {
                            throw std::overflow_error(
                                "sparse lookup I/O payload exceeds size_t");
                        }

                        std::vector<std::byte> payload(
                            static_cast<std::size_t>(
                                operation.byte_count));
                        source.read(
                            operation.byte_offset,
                            payload);
                        return payload;
                    }));
        }

        for (std::size_t local = 0;
             local < width;
             ++local) {
            auto payload =
                futures[local].get();
            execution.bytes_read =
                checked_add(
                    execution.bytes_read,
                    static_cast<std::uint64_t>(
                        payload.size()));
            execution.operation_payloads[
                begin + local] =
                std::move(payload);
            ++execution.operations_completed;
        }

        begin += width;
    }

    return execution;
}

}  // namespace strix
