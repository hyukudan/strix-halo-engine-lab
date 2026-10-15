#include "strix/hip_generation_output.hpp"
#include "strix/hip_generation_output_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t e, const char* message) {
    if (e != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(e));
}
template<class Work> void rejected(Work&& work) {
    bool caught = false;
    try { work(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "invalid input was accepted");
}
std::uint32_t oracle(const std::vector<float>& values) {
    require(!values.empty(), "oracle requires vocabulary");
    std::uint32_t best = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "oracle requires all finite logits");
        if (values[i] > values[best]) best = static_cast<std::uint32_t>(i);
    }
    return best;
}
struct Allocation {
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard;
    void* pointer = nullptr;
    std::size_t bytes = 0;
    Allocation(std::shared_ptr<const strix::HipSetupQuiescenceGuard> q, std::size_t n)
        : guard(std::move(q)), bytes(n) {
        guard->require_original(); check(hipMalloc(&pointer, bytes), "allocate fixture");
    }
    ~Allocation() {
        guard->recover_or_fail_stop();
        if (pointer && hipFree(pointer) != hipSuccess) std::_Exit(3);
    }
};
struct Fixture {
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard =
        std::make_shared<strix::HipSetupQuiescenceGuard>();
    std::shared_ptr<Allocation> allocation;
    strix::ExecutionMemoryBinding scratch;
    strix::ExecutionBufferAddress view;
    std::vector<unsigned char> image;
    explicit Fixture(std::uint32_t vocabulary)
        : allocation(std::make_shared<Allocation>(guard, static_cast<std::size_t>(vocabulary) * 4U + 256U)),
          scratch{0x4752454544593031ULL, reinterpret_cast<std::uintptr_t>(allocation->pointer),
                  static_cast<std::uint64_t>(allocation->bytes), 17},
          image(allocation->bytes, 0xa5) {
        view.address = scratch.arena_base + 128U;
        view.bytes = static_cast<std::uint64_t>(vocabulary) * 4U;
        view.allocation_generation = scratch.allocation_generation;
    }
    void upload(const std::vector<float>& values) {
        require(view.bytes == values.size() * 4U, "fixture extent mismatch");
        std::memcpy(image.data() + 128U, values.data(), static_cast<std::size_t>(view.bytes));
        check(hipMemcpy(allocation->pointer, image.data(), image.size(), hipMemcpyHostToDevice), "fixture producer complete upload");
    }
    void readonly_and_guards() const {
        std::vector<unsigned char> result(image.size());
        check(hipMemcpy(result.data(), allocation->pointer, result.size(), hipMemcpyDeviceToHost), "fixture readback");
        require(result == image, "logits changed or source guards damaged");
    }
};
void finite_family(std::uint32_t n) {
    Fixture fixture(n);
    strix::HipGreedyOutput output(n, fixture.scratch, fixture.allocation, fixture.guard);
    std::vector<float> values(n);
    std::uint64_t epoch = 0;
    for (unsigned pattern = 0; pattern < 8; ++pattern) {
        for (std::uint32_t i = 0; i < n; ++i)
            values[i] = -static_cast<float>((static_cast<std::uint64_t>(i) * 37U + pattern) % 16381U) - 1.0F;
        if (pattern == 0) std::fill(values.begin(), values.end(), -7.0F);
        if (pattern == 1) { std::fill(values.begin(), values.end(), -3.0F); values.back() = 0.0F; }
        if (pattern == 2) {
            std::fill(values.begin(), values.end(), -1.0F);
            values[0] = -0.0F; values.back() = 0.0F;
        }
        if (pattern == 3) {
            std::fill(values.begin(), values.end(), -1.0F);
            values[0] = 0.0F; values.back() = -0.0F;
        }
        if (pattern == 4) { values[n / 2U] = std::numeric_limits<float>::max(); values.back() = values[n / 2U]; }
        if (pattern == 5) std::fill(values.begin(), values.end(), -std::numeric_limits<float>::max());
        if (pattern == 6) { std::fill(values.begin(), values.end(), -std::numeric_limits<float>::denorm_min()); values.back() = std::numeric_limits<float>::denorm_min(); }
        if (pattern == 7 && n > 256U) { values[255] = 99.0F; values[256] = 99.0F; }
        fixture.upload(values);
        const auto actual = output.select(fixture.view, n, 42, ++epoch);
        require(actual == oracle(values), "finite first-max oracle mismatch");
        fixture.readonly_and_guards();
    }
    const auto before = output.stats();
    require(output.scratch_bytes() == 16ULL * ((static_cast<std::uint64_t>(n) + 255ULL) / 256ULL) + 64ULL,
            "retained pool payload must include device and pinned host replies");
    require(before.completed == 8 && before.pool_allocations == 3 && before.reduction_kernel_submissions == 16 &&
            before.result_d2h_bytes == 256 && before.poison_h2d_bytes == 256 && before.stream_completions == 8,
            "actual output counters mismatch");
    rejected([&] { static_cast<void>(output.select(fixture.view, n, 42, epoch)); });
    rejected([&] { static_cast<void>(output.select(fixture.view, n, 42, 0)); });
    auto bad = fixture.view; bad.address += 2U;
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    bad = fixture.view; --bad.bytes;
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    bad = fixture.view; ++bad.bytes;
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    bad = fixture.view; ++bad.allocation_generation;
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    // Valid aligned same-arena span, but a different view after binding.
    bad = fixture.view; bad.address += sizeof(float);
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    bad = fixture.view; bad.address = fixture.scratch.arena_base + fixture.scratch.arena_bytes;
    rejected([&] { static_cast<void>(output.select(bad, n, 42, epoch + 1)); });
    if (n != std::numeric_limits<std::uint32_t>::max())
        rejected([&] { static_cast<void>(output.select(fixture.view, n + 1U, 42, epoch + 1)); });
    const auto after = output.stats();
    require(after.calls == before.calls && after.reduction_kernel_submissions == before.reduction_kernel_submissions &&
            after.pool_allocations == before.pool_allocations && !output.failed(),
            "host admission changed queue or allocation counts");
    require(output.select(fixture.view, n, 43, ++epoch) == oracle(values), "owner unusable after host-only rejection");
}
void repeats() {
    constexpr std::uint32_t n = 513;
    Fixture fixture(n);
    strix::HipGreedyOutput output(n, fixture.scratch, fixture.allocation, fixture.guard);
    std::vector<float> values(n, -5.0F);
    for (std::uint64_t epoch = 1; epoch <= 100; ++epoch) {
        const auto expected = static_cast<std::uint32_t>((epoch * 97U) % n);
        std::fill(values.begin(), values.end(), -5.0F); values[expected] = 3.0F;
        fixture.upload(values);
        require(output.select(fixture.view, n, epoch - 1U, epoch) == expected,
                "persistent owner stale result across repeat");
    }
    const auto stats = output.stats();
    require(stats.completed == 100 && stats.pool_allocations == 3 && stats.result_d2h_bytes == 3200,
            "persistent output allocated per call or copied full logits");
    fixture.readonly_and_guards();
}
void nonfinite() {
    constexpr std::uint32_t n = 513;
    for (const auto position : std::array<std::uint32_t,3>{0, 256, 512}) {
        for (const auto bits : std::array<std::uint32_t,4>{0x7fc12345U, 0xffc23456U, 0x7f800000U, 0xff800000U}) {
            Fixture fixture(n);
            std::vector<float> values(n, 9.0F); values[position] = std::bit_cast<float>(bits);
            fixture.upload(values);
            strix::HipGreedyOutput output(n, fixture.scratch, fixture.allocation, fixture.guard);
            rejected([&] { static_cast<void>(output.select(fixture.view, n, 0, 1)); });
            require(output.failed() && output.stats().completed == 0 && output.stats().failures == 1,
                    "nonfinite logits published a token");
            const auto calls = output.stats().calls;
            rejected([&] { static_cast<void>(output.select(fixture.view, n, 1, 2)); });
            require(output.stats().calls == calls, "failed owner reused");
            fixture.readonly_and_guards();
        }
    }
}
template<class Work> void boundary_failure(Work&& work) {
    bool caught = false;
    try { work(); } catch (const std::exception&) { caught = true; }
    strix::clear_hip_greedy_output_diagnostic_fault();
    require(caught, "diagnostic boundary did not reject");
}
void host_fault_boundaries() {
    using Fault = strix::HipGreedyOutputDiagnosticFault;
    for (const auto fault : {Fault::partial_allocation, Fault::reply_allocation, Fault::pinned_allocation}) {
        Fixture fixture(513);
        strix::arm_hip_greedy_output_diagnostic_fault(fault);
        boundary_failure([&] {
            strix::HipGreedyOutput output(513, fixture.scratch, fixture.allocation, fixture.guard);
        });
        fixture.guard->require_original();
        fixture.upload(std::vector<float>(513, -7.0F));
        strix::HipGreedyOutput replacement(513, fixture.scratch, fixture.allocation, fixture.guard);
        require(replacement.select(fixture.view, 513, 0, 1) == 0, "partial setup cleanup left device unusable");
        fixture.readonly_and_guards();
    }
    for (const auto fault : {Fault::after_poison, Fault::after_partials, Fault::after_finish,
            Fault::after_readback, Fault::after_completion, Fault::corrupt_epoch,
            Fault::corrupt_position, Fault::corrupt_identity, Fault::corrupt_status, Fault::corrupt_token}) {
        Fixture fixture(513);
        fixture.upload(std::vector<float>(513, -3.0F));
        strix::HipGreedyOutput output(513, fixture.scratch, fixture.allocation, fixture.guard);
        require(output.select(fixture.view, 513, 42, 1) == 0, "fault fixture warmup failed");
        strix::arm_hip_greedy_output_diagnostic_fault(fault);
        boundary_failure([&] { static_cast<void>(output.select(fixture.view, 513, 43, 2)); });
        const auto stats = output.stats();
        require(output.failed() && stats.calls == 2 && stats.completed == 1 && stats.failures == 1,
                "fault boundary published a result or left owner reusable");
        rejected([&] { static_cast<void>(output.select(fixture.view, 513, 44, 3)); });
        require(output.stats().calls == stats.calls, "faulted owner submitted more work");
        fixture.guard->require_original();
        fixture.readonly_and_guards();
    }
}
void all_nan_and_retained_owner() {
    Fixture fixture(513);
    fixture.upload(std::vector<float>(513, std::bit_cast<float>(0x7fc12345U)));
    {
        strix::HipGreedyOutput output(513, fixture.scratch, fixture.allocation, fixture.guard);
        rejected([&] { static_cast<void>(output.select(fixture.view, 513, 0, 1)); });
        require(output.failed() && output.stats().completed == 0, "all-NaN vocabulary published a token");
        fixture.readonly_and_guards();
    }
    fixture.upload(std::vector<float>(513, -9.0F));
    std::weak_ptr<Allocation> weak = fixture.allocation;
    {
        strix::HipGreedyOutput output(513, fixture.scratch, fixture.allocation, fixture.guard);
        fixture.allocation.reset();
        require(!weak.expired(), "greedy owner did not retain source allocation");
        require(output.select(fixture.view, 513, 0, 1) == 0, "retained source selection failed");
    }
    require(weak.expired(), "greedy owner leaked source allocation");
}
} // namespace
int main() try {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || !count) return 77;
    for (const auto n : std::array<std::uint32_t,15>{1,2,31,32,33,255,256,257,511,512,513,769,1025,4097,248320}) finite_family(n);
    // Invalid fixed contracts reject before creating a stream/allocation.
    rejected([] { strix::HipGreedyOutput invalid(0, {}, {}, {}); });
    {
        Fixture fixture(1);
        auto bad = fixture.scratch; bad.execution_identity = 0;
        rejected([&] { strix::HipGreedyOutput invalid(1, bad, fixture.allocation, fixture.guard); });
        bad = fixture.scratch; bad.allocation_generation = 0;
        rejected([&] { strix::HipGreedyOutput invalid(1, bad, fixture.allocation, fixture.guard); });
        bad = fixture.scratch; bad.arena_base = std::numeric_limits<std::uintptr_t>::max() - 4U;
        rejected([&] { strix::HipGreedyOutput invalid(1, bad, fixture.allocation, fixture.guard); });
        rejected([&] { strix::HipGreedyOutput invalid(1, fixture.scratch, {}, fixture.guard); });
        rejected([&] { strix::HipGreedyOutput invalid(1, fixture.scratch, fixture.allocation, {}); });
    }
    repeats(); nonfinite(); host_fault_boundaries(); all_nan_and_retained_owner();
    std::cout << "GPU greedy exact first-max/finite/owners/repeats fixtures passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "GPU greedy output fixture: " << e.what() << '\n'; return 1;
}
