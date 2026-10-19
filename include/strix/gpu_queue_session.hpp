#pragma once
#include "strix/linux_gpu_queue.hpp"
#include <chrono>
#include <span>
#include <vector>

namespace strix::transport {
// Wire encoding is canonical big-endian, not memcpy of a padded C++ struct.
// This is a bounded peer-agreement protocol over an independently authenticated
// byte stream, not encryption/authentication or production qualification.
struct GpuQueuePeerFrame {
    std::array<std::uint8_t, 32> campaign{};
    std::array<std::uint8_t, 16> local{}, peer{}, device_instance{};
    std::uint64_t epoch = 0, import_generation = 0, pool_identity = 0;
    std::uint64_t window = 0, first_sequence = 0, payload_bytes = 0;
    std::uint32_t kind = 0, depth = 0, ring_frames = 0, directions = 0, completed_mask = 0, recovery_flags = 0;
    bool operator==(const GpuQueuePeerFrame&) const = default;
};
inline constexpr std::uint32_t gpu_queue_hello = 1, gpu_queue_ready = 2,
    gpu_queue_consumed = 3, gpu_queue_recovery_ready = 4,
    gpu_queue_reset_done = 5, gpu_queue_reset_failed = 6;
inline constexpr std::uint32_t gpu_queue_reset_controller = 0x10,
    gpu_queue_reset_network = 0x20;
// Feature bit in the same flags word: both peers use the kernel-validated
// terminal header (GpuQueueTerminal::kernel_header). Peers must agree exactly;
// a decoder without this bit rejects the frame, so mixed modes never run.
inline constexpr std::uint32_t gpu_queue_terminal_kernel = 0x40;
[[nodiscard]] std::vector<std::byte> encode_gpu_queue_peer_frame(const GpuQueuePeerFrame&);
[[nodiscard]] GpuQueuePeerFrame decode_gpu_queue_peer_frame(std::span<const std::byte>);
void require_gpu_queue_peer(const GpuQueuePeerFrame& local, const GpuQueuePeerFrame& peer,
    const GpuQueuePeerFrame* established_peer = nullptr);
struct GpuQueueSessionIo {
    bool send_first = false;
    std::function<void(std::span<const std::byte>)> send;
    std::function<std::vector<std::byte>()> receive;
};
struct GpuQueueSessionGpu {
    // Must produce/pack/publish terminal and wait for its complete GPU release.
    std::function<void(const GpuQueueOperation&)> produce;
    // Must acquire/unpack/check terminal, verify every payload byte, run the
    // consumer and complete all work BEFORE returning true. False aborts credit.
    std::function<bool(const GpuQueueOperation&)> consume;
};
struct GpuQueueSessionPolicy {
    std::array<std::uint8_t, 32> campaign{};
    std::uint64_t payload_bytes = 0;
    // Must equal queue.required_terminal_bytes() when that is nonzero.
    std::uint32_t windows = 1, directions = 3, terminal_bytes = 32;
    std::uint32_t recovery_every_windows = 0, maximum_progress_calls = 100000;
    std::chrono::milliseconds window_timeout{10000};
    // Additive wire feature negotiation: old decoders reject nonzero flags.
    // Only reset bits here; gpu_queue_terminal_kernel follows queue.terminal().
    std::uint32_t recovery_flags = 0;
    bool independent_control = false;
};
struct GpuQueueProgress {
    std::function<std::chrono::steady_clock::time_point()> now;
    std::function<void()> pause;
    [[nodiscard]] static GpuQueueProgress native();
};
struct GpuQueueSessionReport {
    std::uint64_t completed_tx = 0, completed_rx = 0, final_epoch = 0;
    std::uint32_t completed_windows = 0, recoveries = 0, peak_inflight_tx = 0, peak_inflight_rx = 0, controller_resets = 0;
    bool cleanup_released = false;
    std::string error;
};
// Batch windows are genuinely multi-operation: all RXs are posted and all TXs
// submitted before awaiting payload completion. No pipeline/throughput claim:
// windows are retired together, not a rolling/adaptive queue. The callback API
// can bind real execution buffers without changing the C1 production backend.
[[nodiscard]] GpuQueueSessionReport run_gpu_queue_session(LinuxGpuQueue&, const GpuQueuePool&,
    GpuQueueSessionPolicy, GpuQueueSessionIo, GpuQueueSessionGpu,
    GpuQueueProgress = GpuQueueProgress::native());
} // namespace strix::transport
