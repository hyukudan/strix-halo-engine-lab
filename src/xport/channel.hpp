#pragma once
#include <chrono>
#include <cstdint>

namespace strix::xport {

// Where the TX / RX slots live. host_coherent: hipHostMalloc coherent+mapped (CPU-cached, GPU uncached). device_fine:
// fine-grained device memory (hipExtMallocWithFlags hipDeviceMallocFinegrained), which libraries that only accept
// device pointers (strix_gemm) can write directly; the CPU reads / writes it in place.
// device_uncached: hipDeviceMallocUncached. Validation found device_fine UNSAFE for the CPU side (stale TX bytes read
// after the flag; see design notes): only host_coherent is a default; the device kinds are for the coherence probe.
enum class Memory : std::uint32_t { host_coherent = 0, device_fine = 1, device_uncached = 2 };

// How the TX flag of a message is written once its producers are enqueued.
enum class Publish : std::uint32_t {
    kernel = 0,       // a 1-block kernel after the producers (system-scope release store)
    write_value = 1,  // hipStreamWriteValue64 after the producers
    producer = 2,     // the producer's own last block publishes (glm_xfer_device.hpp), nothing is enqueued
};

// A received message for consumers that wait for it themselves (glmx_wait in glm_xfer_device.hpp) instead of a
// stream wait packet: every block of the first consumer kernel calls glmx_wait(view) before reading the payload.
struct RxView {
    const void* payload = nullptr;
    const std::uint64_t* flag = nullptr;  // device pointer of the RX flag
    std::uint64_t sequence = 0;
    // Device-memory gate (measurement: with every block polling the host flag, a 196 KB consumer's 192 pollers slowed the whole
    // exchange 2.4x): block 0 polls the host flag and opens the gate, the other blocks poll the gate in L2. nullptr:
    // every block polls the host flag.
    std::uint64_t* gate = nullptr;
    // Bounded wait (backstop for a host side that never releases the flag: dead or stuck progress thread): after
    // `budget` shader cycles spent polling (channel_device.hpp WaitMeter: clock64 deltas, no message inside the poll
    // loop) the poller records `sequence` once in `abort` (host-mapped, sticky: the first expiry stays), opens the gate
    // and falls through; the owner's check() throws with that sequence, so the window that read the payload fails
    // instead of the kernel hanging.
    // budget 0 (or abort nullptr) = unbounded: the form of every view built without them.
    std::uint64_t* abort = nullptr;
    std::uint64_t budget = 0;
};

// What a producer kernel needs to publish its own message (Publish::producer): see glm_xfer_device.hpp.
// Device wait budget of a channel whose host timeout is `timeout`: twice it, in cycles of a clock of `khz` kHz (the
// device's peak shader clock, hipDeviceAttributeClockRate: at a lower clock the backstop only comes later, never
// earlier); saturates instead of wrapping; 0 (unbounded) for a zero timeout or rate.
constexpr std::uint64_t wait_budget_ticks(std::chrono::milliseconds timeout, std::uint64_t khz) {
    if (timeout.count() <= 0 || khz == 0) return 0;
    const auto ms = static_cast<std::uint64_t>(timeout.count());
    return ms > ~std::uint64_t{0} / 2U / khz ? ~std::uint64_t{0} : 2U * ms * khz;
}

struct ProducerTarget {
    std::uint64_t* flag = nullptr;    // device pointer of the TX flag
    unsigned* counter = nullptr;      // device scratch, reset by the last block
    std::uint64_t sequence = 0;       // value to publish
};

} // namespace strix::xport
