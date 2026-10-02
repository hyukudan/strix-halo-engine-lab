#include "strix/sparse_lookup.hpp"
#include "strix/sparse_lookup_executor.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>\n#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeSource final
    : public strix::SparseLookupIoSource {
public:
    void read(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) override {
        const auto active_now =
            active_.fetch_add(1) + 1;
        auto observed =
            peak_.load();
        while (active_now > observed &&
               !peak_.compare_exchange_weak(
                   observed,
                   active_now)) {
        }

        if (byte_offset == fail_offset_) {
            active_.fetch_sub(1);
            throw std::runtime_error(
                "injected sparse lookup read failure");
        }

        // Later operations deliberately complete sooner so result ordering
        // cannot accidentally depend on completion order.
        const auto delay =
            static_cast<unsigned>(
                4U -
                std::min<std::uint64_t>(
                    3ULL,
                    byte_offset / 8ULL));
        std::this_thread::sleep_for(
            std::chrono::milliseconds(delay));

        for (std::size_t index = 0;
             index < destination.size();
             ++index) {
            destination[index] =
                static_cast<std::byte>(
                    (byte_offset + index) &
                    0xffULL);
        }

        active_.fetch_sub(1);
    }

    void fail_at(std::uint64_t offset) {
        fail_offset_ = offset;
    }

    [[nodiscard]] std::uint64_t peak() const {
        return peak_.load();
    }

private:
    std::atomic<std::uint64_t> active_{0};
    std::atomic<std::uint64_t> peak_{0};
    std::uint64_t fail_offset_ =
        std::numeric_limits<std::uint64_t>::max();
};

}  // namespace

int main() {
    const std::vector<std::uint64_t> rows{
        0, 1, 2, 5, 6, 9,
    };

    strix::SparseLookupRequest request;
    request.table_rows = 16;
    request.row_bytes = 8;
    request.max_rows_per_io = 2;
    request.max_inflight_ios = 2;
    request.row_ids = rows;

    const auto plan =
        strix::build_sparse_lookup_plan(
            request);
    expect(plan.io_operations == 4 &&
               plan.io_waves == 2,
           "fixture builds two bounded I/O waves");

    FakeSource source;
    const auto execution =
        strix::execute_sparse_lookup_io_plan(
            plan,
            source);

    expect(execution.operations_completed == 4 &&
               execution.bytes_read == 48,
           "executor completes every planned byte range");
    expect(execution.peak_inflight == 2 &&
               source.peak() <= 2,
           "executor never exceeds planned inflight depth");
    expect(execution.operation_payloads.size() == 4 &&
               execution.operation_payloads[0].size() == 16 &&
               execution.operation_payloads[1].size() == 8 &&
               execution.operation_payloads[2].size() == 16 &&
               execution.operation_payloads[3].size() == 8,
           "executor preserves I/O plan ordering despite async completion");
    expect(
        std::to_integer<unsigned>(
            execution.operation_payloads[0][0]) == 0 &&
        std::to_integer<unsigned>(
            execution.operation_payloads[1][0]) == 16 &&
        std::to_integer<unsigned>(
            execution.operation_payloads[2][0]) == 40 &&
        std::to_integer<unsigned>(
            execution.operation_payloads[3][0]) == 72,
        "executor returns payload bytes from the corresponding offsets");

    bool malformed_rejected = false;
    try {
        auto malformed = plan;
        ++malformed.io_operations;
        static_cast<void>(
            strix::execute_sparse_lookup_io_plan(
                malformed,
                source));
    } catch (const std::invalid_argument&) {
        malformed_rejected = true;
    }
    expect(malformed_rejected,
           "executor rejects inconsistent operation counts");

    bool failure_propagated = false;
    try {
        FakeSource failing;
        failing.fail_at(16);
        static_cast<void>(
            strix::execute_sparse_lookup_io_plan(
                plan,
                failing));
    } catch (const std::runtime_error&) {
        failure_propagated = true;
    }
    expect(failure_propagated,
           "executor propagates source failures and does not report success");

    if (failures == 0) {
        std::cout
            << "sparse lookup executor tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
