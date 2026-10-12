#pragma once

#include "strix/linux_direct_transport_driver.hpp"

#include <chrono>
#include <functional>
#include <utility>

namespace strix::transport {

// Experimental measurements only. This type cannot be used as a production
// native backend, IO provider, staging provider or qualification declaration.
struct DirectGpuCommissioningConfig {
    std::string campaign_id, run_id;
    DirectDriverBinding expected;
    TransferDescriptor descriptor;
    std::uint64_t pattern_seed = 0;
    bool receiver = false;
    std::chrono::milliseconds timeout{10000};
    std::uint32_t maximum_progress_polls = 10000;
};

// Splitmix-derived per-byte opaque pattern. Position and sequence are part of
// the pattern; checking a checksum alone is never complete payload validation.
[[nodiscard]]
#if defined(__HIPCC__)
__host__ __device__
#endif
constexpr std::uint8_t direct_gpu_commissioning_pattern_byte(
    std::uint64_t position, std::uint64_t seed, std::uint64_t sequence) noexcept {
    auto value = position ^ seed ^ (sequence * 0x9e3779b97f4a7c15ULL);
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return static_cast<std::uint8_t>((value ^ (value >> 31U)) & 255U);
}
void validate_direct_gpu_commissioning_config(const DirectGpuCommissioningConfig&);
[[nodiscard]] bool verify_direct_gpu_commissioning_pattern(
    std::span<const std::byte>, std::uint64_t seed, std::uint64_t sequence) noexcept;

struct DirectGpuCommissioningReport {
    bool executed = false, controller_frames_complete = false, gpu_acquire_complete = false,
        consumer_complete = false, receive_reposted = false, local_tx_complete = false,
        transport_ack = false, consumer_credit = false, full_payload_verified = false;
    std::uint64_t completed_count = 0;
    DirectPoolReleaseStatus cleanup = DirectPoolReleaseStatus::failed_retained;
    std::uint64_t pool_binding_identity = 0, import_generation = 0;
    std::string error;
};
struct DirectGpuCommissioningPayload {
    std::vector<TransportBufferView> send_segments, receive_segments;
    // Trusted native GPU producer on the helper-owned stream. Its complete
    // source owner is retained by DirectGpuStreamGpu, including callback error.
    std::function<void()> produce;
    // Runs after acquire, unpack and GPU consumer event completion. Must wait
    // for its own complete D2H/event before comparing every byte. An error or
    // false result cannot authorize peer consumer credit or receive repost.
    std::function<bool()> verify_full_payload;
};
struct DirectGpuCommissioningProgress {
    std::function<std::chrono::steady_clock::time_point()> now;
    std::function<void()> pause;
    [[nodiscard]] static DirectGpuCommissioningProgress native();
};

class LinuxDirectGpuStreamCommissioning final {
public:
    LinuxDirectGpuStreamCommissioning(DirectStreamDeviceIdentity route, DirectDriverBinding discovery,
        std::shared_ptr<DirectGpuStreamGpu>, std::shared_ptr<DistributedControlByteStream>,
        LinuxDirectGpuStreamPolicy, LinuxDirectGpuStreamCalls = LinuxDirectGpuStreamCalls::native());
    ~LinuxDirectGpuStreamCommissioning();
    LinuxDirectGpuStreamCommissioning(const LinuxDirectGpuStreamCommissioning&) = delete;
    LinuxDirectGpuStreamCommissioning& operator=(const LinuxDirectGpuStreamCommissioning&) = delete;
    // Observation contains no positive native qualification. Bind checks a
    // complete operator-expected identity against this same fresh owned FD.
    [[nodiscard]] DirectDriverQuery observe();
    void bind_exact_observed(const DirectDriverBinding&);
    [[nodiscard]] DirectGpuCommissioningReport run_once(const DirectGpuCommissioningConfig&,
        const DirectGpuCommissioningPayload&, DirectGpuCommissioningProgress = DirectGpuCommissioningProgress::native());
    [[nodiscard]] DirectPoolReleaseStatus close() noexcept;
private:
    template<class Function> decltype(auto) dispatch(Function&& function) {
        provider_->begin_commissioning_operation();
        struct Guard { LinuxDirectTransportDriverIo* io; ~Guard() { io->end_commissioning_operation(); } } guard{provider_.get()};
        return std::forward<Function>(function)();
    }
    std::unique_ptr<LinuxDirectTransportDriverIo> provider_;
    std::shared_ptr<DirectGpuStreamGpu> gpu_;
    DirectDriverBinding binding_;
    bool bound_ = false, ran_ = false;
};
} // namespace strix::transport
