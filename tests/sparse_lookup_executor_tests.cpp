#include "strix/sparse_lookup.hpp"
#include "strix/sparse_lookup_executor.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
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

class BudgetSource final : public strix::SparseLookupIoSource {
public:
    std::atomic<unsigned> active{0}, peak{0}, calls{0}, caller_reads{0};
    const std::thread::id caller = std::this_thread::get_id();
    bool fail = false;
    void read(std::uint64_t offset, std::span<std::byte> output) override {
        const auto current = active.fetch_add(1) + 1;
        struct Finish { std::atomic<unsigned>& count; ~Finish() { count.fetch_sub(1); } } finish{active};
        calls.fetch_add(1);
        if (std::this_thread::get_id() == caller) caller_reads.fetch_add(1);
        auto previous = peak.load();
        while (current > previous && !peak.compare_exchange_weak(previous, current)) {}
        // Force a later operation to fail first in wall-clock order. Admission
        // must nevertheless return the earliest logical operation's failure.
        std::this_thread::sleep_for(std::chrono::milliseconds(offset == 0 ? 4 : 1));
        if (fail) throw std::runtime_error("read failed offset " + std::to_string(offset));
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = static_cast<std::byte>((offset + i) & 255U);
    }
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

    const std::array<std::uint64_t, 8> budget_rows{0,2,4,6,8,10,12,14};
    auto budget_request = request; budget_request.max_rows_per_io = 1;
    budget_request.max_inflight_ios = 8; budget_request.row_ids = budget_rows;
    const auto budget_plan = strix::build_sparse_lookup_plan(budget_request);
    FakeSource baseline_source;
    const auto baseline = strix::execute_sparse_lookup_io_plan(budget_plan, baseline_source);
    for (const auto budget : {1U,2U,4U,8U}) {
        BudgetSource bounded;
        const auto result = strix::execute_sparse_lookup_io_plan(budget_plan, bounded, budget);
        expect(result.operation_payloads == baseline.operation_payloads && result.bytes_read == baseline.bytes_read &&
            result.operations_completed == baseline.operations_completed,
            "explicit CPU budgets preserve every byte and logical operation order");
        expect(result.peak_inflight == budget && bounded.peak.load() <= budget && bounded.active.load() == 0 &&
            bounded.caller_reads.load() != 0,
            "CPU budget includes the caller as a reader and joins all bounded helpers");
        auto depth_one = budget_plan; depth_one.max_inflight_ios = 1;
        BudgetSource serial_source;
        const auto serial = strix::execute_sparse_lookup_io_plan(depth_one, serial_source, budget);
        expect(serial.peak_inflight == 1 && serial_source.peak.load() == 1 && serial_source.caller_reads.load() == budget_rows.size(),
            "I/O depth remains an independent tighter cap without an idle coordinator worker");
        BudgetSource failing; failing.fail = true;
        std::string reason;
        try { static_cast<void>(strix::execute_sparse_lookup_io_plan(budget_plan, failing, budget)); }
        catch (const std::runtime_error& error) { reason = error.what(); }
        expect(reason == "read failed offset 0" && failing.active.load() == 0 && failing.calls.load() == budget,
            "all submitted failures are joined before the earliest logical failure is propagated");
    }
    for (const auto invalid_budget : {0U,9U,std::numeric_limits<unsigned>::max()}) {
        BudgetSource untouched; bool rejected = false;
        try { static_cast<void>(strix::execute_sparse_lookup_io_plan(budget_plan, untouched, invalid_budget)); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && untouched.calls.load() == 0, "invalid CPU budget rejects before any source I/O");
    }

    if (failures == 0) {
        std::cout
            << "sparse lookup executor tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
