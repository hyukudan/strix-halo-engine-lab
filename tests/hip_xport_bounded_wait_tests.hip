// Bounded device wait of the shared channel consumer (src/xport/channel_device.hpp wait_view, RxView::abort / budget).
// One GPU, no transport: the RX flag lives in host-mapped memory and is raised (or never raised) by this test.
//   1. unraised flag, bounded view, poller form (gate nullptr, 4 blocks): the kernel returns after the budget, the sticky
//      word holds the sequence, elapsed >= budget;
//   2. unraised flag, bounded view, gate form (8 blocks): block 0 opens the gate on expiry, every block returns;
//   3. flag already raised: no abort recorded;
//   4. unbounded view, flag raised by the host 20 ms after launch: the kernel waits for it and records nothing;
//   5. first expiry wins: a second expired view does not overwrite the word;
//   6. wait_budget_ticks: 2 x timeout x rate, 0 for a zero timeout / rate, saturation;
//   7. bounded gate form in the regime of a TP2 verify window (64 blocks of 256 threads, the flag raised by the host
//      50 ms after launch, a budget far above it): every block returns, nothing is recorded. With the former meter
//      (the wall clock read by s_sendmsg_rtn inside the poll loop) the same bounded form hung the GLM pair MTP verify.
// Budgets are shader cycles at the peak clock (hipDeviceAttributeClockRate, channel_device.hpp WaitMeter). Every launch
// is awaited with a 10 s host bound, so a hung wait fails the test instead of blocking it.
#include <hip/hip_runtime.h>

#include "xport/channel_hip.hpp"
#include "xport/channel_device.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace x = strix::xport;

namespace {

int failures = 0;
void require(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
void hk(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

__global__ void consume(x::RxView v, unsigned* passed) {
    x::wait_view(v);
    if (threadIdx.x == 0) atomicAdd(passed, 1U);
}

struct Mapped {
    std::uint64_t* host = nullptr;  // [0] RX flag, [8] abort word
    std::uint64_t* dev = nullptr;
    Mapped() {
        hk(hipHostMalloc(reinterpret_cast<void**>(&host), 4096, hipHostMallocMapped | hipHostMallocCoherent), "host flags");
        hk(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev), host, 0), "flags device pointer");
        reset();
    }
    ~Mapped() { (void)hipHostFree(host); }
    void reset() {
        __atomic_store_n(&host[0], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&host[8], 0, __ATOMIC_RELEASE);
    }
    std::uint64_t aborted() const { return __atomic_load_n(&host[8], __ATOMIC_ACQUIRE); }
};

double run(const x::RxView& v, unsigned blocks, unsigned* passed_dev, unsigned* passed_host, Mapped* raise_later = nullptr,
           unsigned threads = 64, unsigned raise_ms = 20) {
    hk(hipMemset(passed_dev, 0, sizeof(unsigned)), "passed reset");
    hk(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    consume<<<blocks, threads>>>(v, passed_dev);
    hk(hipGetLastError(), "launch");
    if (raise_later != nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(raise_ms));
        __atomic_store_n(&raise_later->host[0], v.sequence, __ATOMIC_RELEASE);
    }
    for (;;) {  // bounded host wait: a wait that never returns is a failure, not a blocked test
        const hipError_t q = hipStreamQuery(nullptr);
        if (q == hipSuccess) break;
        if (q != hipErrorNotReady) hk(q, "kernel");
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
            std::printf("FAIL: consumer kernel still waiting after 10 s (hung wait)\n");
            std::fflush(stdout);
            std::_Exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    hk(hipMemcpy(passed_host, passed_dev, sizeof(unsigned), hipMemcpyDeviceToHost), "passed");
    return ms;
}

}  // namespace

int main() {
    int dev = 0, khz = 0;
    hk(hipGetDevice(&dev), "device");
    hk(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, dev), "peak shader clock rate");
    require(khz > 0, "device reports a peak shader clock rate");
    if (khz <= 0) return EXIT_FAILURE;
    std::printf("peak shader clock %d kHz\n", khz);
    Mapped m;
    std::uint64_t* gate = nullptr;
    unsigned* passed_dev = nullptr;
    unsigned passed = 0;
    hk(hipMalloc(reinterpret_cast<void**>(&gate), 64), "gate");
    hk(hipMalloc(reinterpret_cast<void**>(&passed_dev), sizeof(unsigned)), "passed");
    const std::uint64_t budget = 5ULL * static_cast<std::uint64_t>(khz);  // 5 ms at the peak clock (longer when slower)

    {  // 1. poller form
        m.reset();
        const x::RxView v{nullptr, m.dev, 7, nullptr, m.dev + 8, budget};
        const double ms = run(v, 4, passed_dev, &passed);
        std::printf("  poller form: %.2f ms, abort word %llu, blocks passed %u\n", ms, static_cast<unsigned long long>(m.aborted()), passed);
        require(m.aborted() == 7 && passed == 4, "unraised flag: every block returns and the sequence is recorded");
        require(ms >= 5.0 && ms < 2000.0, "returns after the budget, not before, not hung");
    }
    {  // 2. gate form
        m.reset();
        hk(hipMemset(gate, 0, 64), "gate reset");
        const x::RxView v{nullptr, m.dev, 9, gate, m.dev + 8, budget};
        const double ms = run(v, 8, passed_dev, &passed);
        std::printf("  gate form: %.2f ms, abort word %llu, blocks passed %u\n", ms, static_cast<unsigned long long>(m.aborted()), passed);
        require(m.aborted() == 9 && passed == 8, "gate form: block 0 opens the gate on expiry, every block returns");
    }
    {  // 3. raised flag
        m.reset();
        hk(hipMemset(gate, 0, 64), "gate reset");
        __atomic_store_n(&m.host[0], 11, __ATOMIC_RELEASE);
        const x::RxView v{nullptr, m.dev, 11, gate, m.dev + 8, budget};
        (void)run(v, 8, passed_dev, &passed);
        require(m.aborted() == 0 && passed == 8, "raised flag: no abort");
    }
    {  // 4. unbounded, raised later
        m.reset();
        hk(hipMemset(gate, 0, 64), "gate reset");
        const x::RxView v{nullptr, m.dev, 13, gate};
        const double ms = run(v, 8, passed_dev, &passed, &m);
        std::printf("  unbounded: %.2f ms\n", ms);
        require(m.aborted() == 0 && passed == 8 && ms >= 20.0, "unbounded view waits for the host and records nothing");
    }
    {  // 5. first expiry wins
        m.reset();
        const x::RxView a{nullptr, m.dev, 21, nullptr, m.dev + 8, budget};
        const x::RxView b{nullptr, m.dev, 22, nullptr, m.dev + 8, budget};
        (void)run(a, 1, passed_dev, &passed);
        (void)run(b, 1, passed_dev, &passed);
        require(m.aborted() == 21, "the sticky word keeps the first expired sequence");
    }
    {  // 6. budget arithmetic
        using ms = std::chrono::milliseconds;
        require(x::wait_budget_ticks(ms{120000}, 100000) == 2ULL * 120000ULL * 100000ULL, "budget = 2 x timeout x rate");
        require(x::wait_budget_ticks(ms{0}, 100000) == 0 && x::wait_budget_ticks(ms{1000}, 0) == 0, "zero timeout / rate: unbounded");
        require(x::wait_budget_ticks(ms{INT64_MAX}, 100000) == ~std::uint64_t{0}, "saturates");
    }
    {  // 7. verify-window regime: 64 x 256 threads, bounded gate form, raised after 50 ms, budget 2 s at the peak clock
        m.reset();
        hk(hipMemset(gate, 0, 64), "gate reset");
        const x::RxView v{nullptr, m.dev, 31, gate, m.dev + 8, 2000ULL * static_cast<std::uint64_t>(khz)};
        const double ms = run(v, 64, passed_dev, &passed, &m, 256, 50);
        std::printf("  verify regime: %.2f ms, abort word %llu, blocks passed %u\n", ms, static_cast<unsigned long long>(m.aborted()), passed);
        require(m.aborted() == 0 && passed == 64 && ms >= 50.0 && ms < 2000.0,
                "bounded gate form, 64 blocks, late flag: every block returns when the host raises it, nothing recorded");
    }
    (void)hipFree(gate);
    (void)hipFree(passed_dev);
    if (failures == 0) std::printf("bounded wait tests PASS\n");
    else std::printf("bounded wait tests: %d failure(s)\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
