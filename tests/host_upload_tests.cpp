#include "strix/host_upload.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template <typename Function>
bool rejected(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; }
    return false;
}

struct Counts {
    int freed = 0;
    int prepared = 0;
    int submitted = 0;
    int polled = 0;
    int canceled = 0;
    int retired = 0;
};

class Source final : public strix::HostUploadSource {
public:
    Source(std::uint64_t bytes, std::uint64_t generation, std::shared_ptr<Counts> counts)
        : storage_(static_cast<std::size_t>(bytes)), generation_(generation), counts_(std::move(counts)) {}
    ~Source() override { ++counts_->freed; }
    std::span<std::byte> writable_bytes() noexcept override { return storage_; }
    std::span<const std::byte> bytes() const noexcept override { return storage_; }
    std::uint64_t allocation_generation() const noexcept override { return generation_; }

private:
    std::vector<std::byte> storage_;
    std::uint64_t generation_;
    std::shared_ptr<Counts> counts_;
};

class DelayedBackend final : public strix::HostUploadBackend {
public:
    struct Operation {
        std::shared_ptr<const strix::HostUploadSource> source;
        strix::ExecutionBufferAddress destination;
        std::uint64_t bytes = 0;
        int remaining_polls = 0;
        bool submitted = false;
        bool cancel_requested = false;
        bool released = false;
    };
    std::shared_ptr<strix::HostUploadSource> allocate_source(
        std::uint64_t bytes, std::uint64_t generation) override {
        auto source = std::make_shared<Source>(bytes, foreign_source_generation ? generation + 1 : generation, counts);
        last_source = source;
        return source;
    }
    strix::HostUploadHandle prepare(const strix::ExecutionBufferAddress& destination,
                                    std::shared_ptr<const strix::HostUploadSource> source,
                                    std::uint64_t bytes) override {
        ++counts->prepared;
        const auto handle = next_handle++;
        operations.emplace(handle, Operation{std::move(source), destination, bytes, pending_polls});
        last_handle = handle;
        return handle;
    }
    void submit(strix::HostUploadHandle handle) override {
        ++counts->submitted;
        operations.at(handle).submitted = true;
        if (throw_submit_after_enqueue) throw std::runtime_error("injected submit callback error after enqueue");
    }
    strix::HostUploadStatus poll(strix::HostUploadHandle handle) override {
        ++counts->polled;
        auto& operation = operations.at(handle);
        if (throw_poll) throw std::runtime_error("injected completion callback error");
        if (fail_retained) return strix::HostUploadStatus::failed_retained;
        if (operation.remaining_polls-- > 0) return strix::HostUploadStatus::pending;
        if (!operation.released) {
            // The simulated DMA reads the owned source only when it completes.
            // This makes a released or overwritten input observable in tests.
            std::memcpy(reinterpret_cast<void*>(operation.destination.address),
                        operation.source->bytes().data(), static_cast<std::size_t>(operation.bytes));
            operation.released = true;
        }
        return operation.cancel_requested ? strix::HostUploadStatus::canceled : strix::HostUploadStatus::complete;
    }
    void cancel(strix::HostUploadHandle handle) override {
        ++counts->canceled;
        operations.at(handle).cancel_requested = true;
        if (throw_cancel) throw std::runtime_error("injected cancellation callback error");
    }
    void retire(strix::HostUploadHandle handle) override {
        if (!operations.at(handle).released) throw std::logic_error("retire lacks actual source-release proof");
        ++counts->retired;
        operations.erase(handle);
    }
    void finish_all() {
        fail_retained = false;
        throw_poll = false;
        for (auto& [handle, operation] : operations) {
            static_cast<void>(handle);
            operation.remaining_polls = 0;
        }
    }
    void drain_all() {
        finish_all();
        while (!operations.empty()) {
            const auto handle = operations.begin()->first;
            static_cast<void>(poll(handle));
            retire(handle);
        }
    }

    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    std::weak_ptr<strix::HostUploadSource> last_source;
    strix::HostUploadHandle next_handle = 1;
    strix::HostUploadHandle last_handle = 0;
    int pending_polls = 2;
    bool throw_submit_after_enqueue = false;
    bool throw_poll = false;
    bool fail_retained = false;
    bool throw_cancel = false;
    bool foreign_source_generation = false;
    std::unordered_map<strix::HostUploadHandle, Operation> operations;
};

strix::ExecutionBufferAddress destination(std::span<std::byte> bytes, std::uint64_t generation = 7) {
    strix::ExecutionBufferAddress result;
    result.use.generation = 1;
    result.address = reinterpret_cast<std::uintptr_t>(bytes.data());
    result.bytes = bytes.size();
    result.allocation_generation = generation;
    return result;
}

}  // namespace

int main() {
    using namespace strix;
    const std::array<std::byte, 4> original{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output{};
        const auto binding = destination(output);
        HostUploadPool pool(backend, generations, {1, 16, 16, 4});
        auto caller_input = original;
        const auto first = pool.start(caller_input, binding);
        caller_input.fill(std::byte{99});
        expect(pool.active_uploads() == 1 && pool.allocated_bytes() == 16 &&
                   pool.poll(first, binding) == HostUploadStatus::pending &&
                   backend.counts->freed == 0 && output[0] == std::byte{0},
               "copy remains pending after CPU return and caller input mutation");
        expect(rejected([&] { static_cast<void>(pool.start(caller_input, binding)); }),
               "pending source/destination cannot be overwritten by another upload");
        auto foreign_ticket = first;
        ++foreign_ticket.source_allocation_generation;
        expect(rejected([&] { static_cast<void>(pool.poll(foreign_ticket, binding)); }),
               "foreign source allocation generation cannot retire pending storage");
        foreign_ticket = first;
        ++foreign_ticket.pool_generation;
        expect(rejected([&] { static_cast<void>(pool.poll(foreign_ticket, binding)); }),
               "foreign pool ticket cannot progress another pool's source");
        pool.wait(first, binding);
        expect(std::equal(original.begin(), original.end(), output.begin()) && pool.active_uploads() == 0,
               "delayed DMA consumes the original owned bytes rather than changed caller input");
        const auto second = pool.start(caller_input, binding);
        expect(second.source_allocation_generation == first.source_allocation_generation &&
                   second.sequence > first.sequence &&
                   rejected([&] { static_cast<void>(pool.poll(first, binding)); }),
               "safe physical reuse keeps allocation generation but rejects old content sequence");
        pool.wait(second, binding);
        expect(output[0] == std::byte{99}, "next upload reuses source only after explicit release proof");
    }

    for (int changed = 0; changed < 4; ++changed) {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output{};
        const auto binding = destination(output);
        HostUploadPool pool(backend, generations, {1, 16, 16, 4});
        const auto ticket = pool.start(original, binding);
        auto foreign_binding = binding;
        if (changed == 0) ++foreign_binding.allocation_generation;
        if (changed == 1) ++foreign_binding.use.generation;
        if (changed == 2) ++foreign_binding.address;
        if (changed == 3) --foreign_binding.bytes;
        expect(rejected([&] { static_cast<void>(pool.poll(ticket, foreign_binding)); }) &&
                   backend.counts->polled == 0 && pool.failed() && pool.active_uploads() == 1 &&
                   backend.counts->freed == 0 && backend.counts->canceled == 1,
               "destination allocation/use/address/extent drift poisons new work before native progress");
        expect(pool.drain(ticket, binding) && pool.active_uploads() == 0 && pool.failed() &&
                   rejected([&] { static_cast<void>(pool.start(original, binding)); }),
               "later quiescence retires a drifted operation without recovering a failed pool");
    }

    {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output0{}, output1{}, output2{};
        HostUploadPool pool(backend, generations, {2, 16, 32, 4});
        const auto first = pool.start(original, destination(output0));
        auto overlapping = destination(output0, 99);
        overlapping.address += 2;
        overlapping.bytes -= 2;
        expect(rejected([&] { static_cast<void>(pool.start(original, overlapping)); }) &&
                   pool.allocated_bytes() == 16 && backend.counts->prepared == 1,
               "overlapping destination is rejected despite spare slot and different allocation generation");
        const auto second = pool.start(original, destination(output1));
        expect(pool.allocated_bytes() == 32 && pool.active_uploads() == 2 &&
                   rejected([&] { static_cast<void>(pool.start(original, destination(output2))); }),
               "bounded slot/byte budget rejects a third distinct pending upload");
        std::array<std::byte, 17> oversized{};
        expect(rejected([&] { static_cast<void>(pool.start(oversized, destination(output2))); }),
               "oversized payload is rejected before allocation or submission");
        pool.wait(first, destination(output0));
        pool.wait(second, destination(output1));
    }

    {
        DelayedBackend backend;
        AllocationGenerationSource generations0(1), generations1(1);
        std::array<std::byte, 16> output0{}, output1{};
        HostUploadPool pool0(backend, generations0, {1, 16, 16, 4});
        HostUploadPool pool1(backend, generations1, {1, 16, 16, 4});
        const auto first = pool0.start(original, destination(output0));
        const auto second = pool1.start(original, destination(output1));
        expect(first.source_allocation_generation == second.source_allocation_generation &&
                   first.sequence == second.sequence && first.slot == second.slot &&
                   first.pool_generation != second.pool_generation &&
                   rejected([&] { static_cast<void>(pool1.poll(first, destination(output1))); }) &&
                   rejected([&] { static_cast<void>(pool0.poll(second, destination(output0))); }) &&
                   backend.counts->polled == 0,
               "independent counters starting at one cannot authorize another pool's upload");
        pool0.wait(first, destination(output0));
        pool1.wait(second, destination(output1));
        AllocationGenerationSource restarted(1);
        HostUploadPool recreated(backend, restarted, {1, 16, 16, 4});
        const auto current = recreated.start(original, destination(output0));
        expect(current.pool_generation != first.pool_generation &&
                   rejected([&] { static_cast<void>(recreated.poll(first, destination(output0))); }),
               "recreated pool cannot accept a prior pool's ticket when allocation counters restart");
        recreated.wait(current, destination(output0));
    }

    {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output{};
        HostUploadPool pool(backend, generations, {1, 16, 16, 4});
        const auto ticket = pool.start(original, destination(output));
        pool.cancel(ticket);
        expect(backend.counts->canceled == 1 && pool.active_uploads() == 1 &&
                   pool.poll(ticket, destination(output)) == HostUploadStatus::pending &&
                   rejected([&] { static_cast<void>(pool.start(original, destination(output))); }),
               "cancel callback return cannot authorize slot reuse");
        expect(pool.drain(ticket, destination(output)) && pool.active_uploads() == 0 &&
                   backend.counts->retired == 1,
               "delayed cancellation completion explicitly proves source release");
    }

    for (int failure = 0; failure < 4; ++failure) {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output{};
        std::weak_ptr<HostUploadSource> retained;
        {
            HostUploadPool pool(backend, generations, {1, 16, 16, 1});
            const auto ticket = pool.start(original, destination(output));
            retained = backend.last_source;
            backend.throw_poll = failure == 1;
            backend.fail_retained = failure == 2;
            backend.throw_cancel = failure == 3;
            expect(rejected([&] { pool.wait(ticket, destination(output)); }) && pool.failed() &&
                       backend.counts->freed == 0 && pool.active_uploads() == 1,
                   "timeout/completion failure/callback error retains pending buffer and poisons new work");
            expect(rejected([&] { static_cast<void>(pool.start(original, destination(output))); }),
                   "failed upload pool cannot silently reuse pending storage");
        }
        expect(!retained.expired() && backend.counts->freed == 0 && backend.operations.size() == 1,
               "backend keeps owned source alive after failed pool destruction");
        backend.throw_cancel = false;
        backend.drain_all();
        expect(retained.expired() && backend.counts->freed == 1 &&
                   std::equal(original.begin(), original.end(), output.begin()),
               "pending source is freed only after explicit later copy/cancel quiescence proof");
    }

    {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        std::array<std::byte, 16> output{};
        backend.throw_submit_after_enqueue = true;
        {
            HostUploadPool pool(backend, generations, {1, 16, 16, 1});
            expect(rejected([&] { static_cast<void>(pool.start(original, destination(output))); }) &&
                       pool.failed() && pool.active_uploads() == 1,
                   "submit callback error after enqueue retains its prepared operation");
        }
        expect(!backend.last_source.expired() && backend.counts->freed == 0,
               "partial submit cleanup cannot free a source without completion");
        backend.drain_all();
        expect(backend.last_source.expired() && std::equal(original.begin(), original.end(), output.begin()),
               "partially submitted copy still sees retained original bytes");
    }

    {
        DelayedBackend backend;
        AllocationGenerationSource generations;
        expect(rejected([&] { HostUploadPool pool(backend, generations, {2, 17, 32, 1}); }) &&
                   rejected([&] { HostUploadPool pool(backend, generations, {0, 16, 16, 1}); }) &&
                   rejected([&] { HostUploadPool pool(backend, generations, {1, 16, 16, 0}); }) &&
                   rejected([&] { HostUploadPool pool(backend, generations,
                       {2, std::numeric_limits<std::uint64_t>::max(),
                        std::numeric_limits<std::uint64_t>::max(), 1}); }),
               "pool policy rejects zero limits, excessive total bytes and multiplication overflow");
        HostUploadPool pool(backend, generations, {1, 16, 16, 1});
        std::array<std::byte, 16> output{};
        auto undersized = destination(output);
        undersized.bytes = 3;
        expect(rejected([&] { static_cast<void>(pool.start(original, undersized)); }) && backend.counts->prepared == 0,
               "destination byte bound is checked before any backend callback");
        auto overflowing = destination(output);
        overflowing.address = std::numeric_limits<std::uintptr_t>::max() - 1;
        expect(rejected([&] { static_cast<void>(pool.start(original, overflowing)); }) && backend.counts->prepared == 0,
               "overflowing destination range is rejected before any backend callback");
        backend.foreign_source_generation = true;
        expect(rejected([&] { static_cast<void>(pool.start(original, destination(output))); }) &&
                   pool.failed() && backend.counts->prepared == 0,
               "foreign allocator generation cannot enter native copy preparation");
    }

    if (failures == 0) std::cout << "host upload tests passed\n";
    return failures == 0 ? 0 : 1;
}
