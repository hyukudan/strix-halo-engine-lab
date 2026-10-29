#include "hip_prefill_pool.hpp"
#include <array>
#include <iostream>
#include <thread>

namespace {
void expect(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    expect(rejected, "expected capacity/device rejection");
}
void exercise(std::uint32_t maximum) {
    strix::HipPrefillPoolOwner owner;
    strix::HipPrefillPool image;
    const auto allocation_count = prefill_mock::allocations.load();
    rejects([&] { (void)image.get(16U, "unreserved"); });
    const auto setup = [&] {
        owner.setup(maximum, [&] {
            (void)image.get(owner.rows() * sizeof(unsigned), "image", true, true);
        });
    };
    // Concurrent setup is serialized; exactly one maximum-sized allocation is exposed.
    std::array<std::thread, 4> threads;
    for (auto& thread : threads) thread = std::thread(setup);
    for (auto& thread : threads) thread.join();
    expect(prefill_mock::allocations == allocation_count + 1U, "setup allocated more than once");
    const auto capacity = strix::known_prefill_pool_rows(maximum);
    expect(owner.rows() == capacity, "historical floor or configured maximum changed");
    auto* image_ptr = static_cast<unsigned*>(image.get(capacity * sizeof(unsigned), "image"));
    expect(image_ptr[capacity - 1U] == 0U, "setup initialization missed retained tail");
    // Poison the actual full reservation, then model large/small/large submissions.
    (void)hipMemset(image_ptr, 0xa5, capacity * sizeof(unsigned));
    const auto generation = image.generation();
    for (const auto rows : {maximum, 16U, maximum}) {
        expect(owner.rows(rows) == capacity, "a smaller call lowered high-water capacity");
        expect(image.get(rows * sizeof(unsigned), "image") == image_ptr, "submission changed the poisoned allocation");
        expect(image_ptr[rows - 1U] == 0xa5a5a5a5U, "submission consumed an unpoisoned replacement");
        expect(image.generation() == generation, "allocation generation changed through submission");
    }
    owner.setup(16U, [] { throw std::runtime_error("repeat setup must not reserve again"); });
    expect(owner.rows(maximum) == capacity, "tail setup lowered the shared high-water capacity");
    owner.setup(maximum, [] { throw std::runtime_error("largest owner setup must remain idempotent"); });
    expect(image.get(owner.rows() * sizeof(unsigned), "after tail setup") == image_ptr &&
        image_ptr[capacity - 1U] == 0xa5a5a5a5U && image.generation() == generation,
        "tail setup changed the retained extent, allocation or contents");
    rejects([&] { (void)owner.rows(capacity + 1U); });
    rejects([&] { (void)image.get((capacity + 1U) * sizeof(unsigned), "oversized"); });

    prefill_mock::device = 1;
    rejects([&] { (void)owner.rows(16U); });
    rejects([&] { owner.setup(maximum, [] {}); });
    rejects([&] { (void)image.get(16U, "other device"); });
    prefill_mock::device = 0;
    expect(prefill_mock::allocations == allocation_count + 1U && prefill_mock::frees == 0U,
        "execution allocated, replaced or freed scratch");
}
}
int main() try {
    const strix::PrefillStageReservation copy_stage{8192U, 262144U, 4U, 0, false};
    expect(copy_stage.fits({2048U, 32768U, 4U, 0, false}), "copy-wide plus group reservation fits");
    expect(!copy_stage.fits({16384U, 262144U, 4U, 0, false}), "larger group declines optional ring");
    expect(!copy_stage.fits({8192U, 262144U, 4U, 0, true}), "compact mismatch declines optional ring");
    expect(!copy_stage.fits({8192U, 524288U, 4U, 0, false}), "context mismatch declines optional ring");
    expect(!copy_stage.fits({8192U, 262144U, 8U, 0, false}), "ratio mismatch declines optional ring");
    expect(!copy_stage.fits({8192U, 262144U, 4U, 1, false}), "device mismatch declines optional ring");
    exercise(2048U);
    exercise(8192U);
    exercise(16384U);
    exercise(8224U);
    expect(strix::known_prefill_pool_rows(8224U) == 8704U, "512-row band capacity");
    strix::HipPrefillPoolOwner copy_wide;
    strix::HipPrefillPool image;
    copy_wide.setup(8192U, [&] { (void)image.get(copy_wide.rows(), "copy-wide", true); });
    const auto* previous = image.get(8192U, "copy-wide");
    copy_wide.setup(16384U, [&] { (void)image.get(copy_wide.rows(), "group", true); });
    expect(copy_wide.rows() == 16384U && image.generation() == 2U, "later group setup grows at setup");
    expect(image.get(16384U, "group") != previous, "growth replaces scratch only during setup");
    std::cout << "prefill immutable pool host contracts PASS\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
