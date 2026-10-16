#pragma once

#include "strix/linux_direct_gpu_stream_uapi.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <span>

namespace strix::transport {

struct GpuQueueSyscall { int result = -1; int error = 0; };
struct GpuQueueIo {
    std::shared_ptr<void> lifetime;
    std::function<GpuQueueSyscall(unsigned long, void*)> ioctl_call;
    std::function<void()> require_current;
    // Optional until GROUP_RESET is used. Never accept an unowned numeric FD.
    std::function<int()> native_descriptor{};
};
// Opens one nonblocking, close-on-exec character-device FD. The preflight must
// verify the deliberately provisioned gpu_direct route/module before opening.
// No probing of an unknown stock/copied-mode stream, reset or mode switching.
[[nodiscard]] GpuQueueIo open_linux_gpu_queue(const std::string& path,
    std::function<void()> require_provisioned_route);

struct GpuQueueExpected {
    std::array<std::uint8_t, 16> local{}, peer{};
    std::uint32_t ring_frames = 0, depth = 0;
};
struct GpuQueuePool {
    sgu_buffer tx{}, rx{};
    std::shared_ptr<void> lifetime;
    std::function<void()> require_current;
    // Must cover EVERY GPU stream using either exported allocation. A controller
    // STOP is not a GPU completion witness. False/exception retains all owners.
    std::function<bool()> gpu_quiescent;
};
struct GpuQueueOperation {
    std::uint64_t epoch = 0, import_generation = 0, handle = 0, sequence = 0;
    std::uint64_t payload_bytes = 0, pool_offset = 0;
    std::uint32_t direction = 0, slot = 0, terminal_bytes = 0;
    bool operator==(const GpuQueueOperation&) const = default;
};
struct GpuQueueCompletion {
    GpuQueueOperation operation;
    std::uint64_t controller_monotonic_ns = 0;
};

// Reusable experimental engine-side adapter for the project-owned QUEUE ABI.
// Serialized caller; 2..8 concurrent operations PER direction. It does not
// implement the old C1 RX_REPOST semantics or grant production qualification.
// GPU and authenticated peer-credit checks are intentionally separate from
// EVENT. A boolean supplied by an untrusted receipt is not a native proof.
struct GpuQueueResetResult {
    bool success = false;
    bool may_have_reset = false; // ioctl attempted: a lost reply is ambiguous
    int error = 0;
};
class LinuxGpuQueue final {
public:
    LinuxGpuQueue(GpuQueueIo, GpuQueueExpected);
    ~LinuxGpuQueue();
    LinuxGpuQueue(const LinuxGpuQueue&) = delete;
    LinuxGpuQueue& operator=(const LinuxGpuQueue&) = delete;
    [[nodiscard]] const sgu_query& observation() const noexcept;
    [[nodiscard]] const sgu_fence& binding() const noexcept;
    [[nodiscard]] std::uint32_t depth() const noexcept;
    [[nodiscard]] std::uint64_t slot_capacity() const noexcept;
    [[nodiscard]] std::uint64_t terminal_offset(std::uint32_t slot) const;
    void import_pool(GpuQueuePool, std::uint64_t epoch, std::uint64_t pool_identity);
    void arm();
    [[nodiscard]] GpuQueueOperation prepare(std::uint32_t direction, std::uint32_t slot,
        std::uint64_t sequence, std::uint64_t payload_bytes, std::uint32_t terminal_bytes);
    // False means bounded caller may retry this SAME operation after progress.
    // Only EAGAIN is retryable. EINTR on a mutating ioctl is ambiguous and fails.
    void gpu_produced(const GpuQueueOperation&);
    [[nodiscard]] bool submit(const GpuQueueOperation&);
    [[nodiscard]] std::optional<GpuQueueCompletion> poll();
    // Call after actual GPU producer/consumer work or authenticated peer credit.
    // Direction and exact epoch/import/handle/geometry are checked independently.
    void gpu_consumed(const GpuQueueOperation&);
    void peer_consumed(const GpuQueueOperation&);
    void retire(const GpuQueueOperation&);
    [[nodiscard]] sgu_stats stats();
    void invalidate() noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool stop() noexcept;
    [[nodiscard]] bool release() noexcept;
    // Caller coordinates peers externally. RECOVER never resets unrelated NHI
    // users. New IMPORT must use an epoch strictly greater than the old epoch.
    [[nodiscard]] bool recover() noexcept;
    // All callers/peers must stop at a shared completed-window boundary. Every
    // member has to be stopped, retired and independently GPU-quiescent. This
    // executes ONE local controller reset, never a distributed rollback. All
    // attempted members become failed until RECOVER + strictly newer IMPORT.
    // NETWORK explicitly allows the local domain's USB4NET down/up outage;
    // the caller must use a control channel independent of that controller.
    [[nodiscard]] static GpuQueueResetResult reset_controller_group(
        std::span<LinuxGpuQueue* const> members, bool include_network = false) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::transport
