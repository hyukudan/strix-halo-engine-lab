#pragma once
#include "strix/direct_gpu_stream_gpu.hpp"
// Exact pattern expression from the engine, isolated from unrelated headers.
namespace strix::transport {
constexpr std::uint8_t direct_gpu_commissioning_pattern_byte(std::uint64_t position,std::uint64_t seed,std::uint64_t sequence) noexcept {
    auto value=position ^ seed ^ (sequence * 0x9e3779b97f4a7c15ULL);
    value+=0x9e3779b97f4a7c15ULL;
    value=(value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value=(value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return static_cast<std::uint8_t>((value ^ (value >> 31U)) & 255U);
}
}

#include <chrono>
#include <functional>
#include <optional>
namespace strix::transport {
struct DirectNativeInterfaceIdentity { std::uint32_t version=0; std::string kernel_release,driver_identity,abi_identity,local_endpoint_identity,peer_endpoint_identity,physical_link_id;std::uint64_t connection_epoch=0; };
struct DirectDriverBinding { DirectNativeInterfaceIdentity interface; std::string device_identity;std::uint64_t device_instance_generation=0; };
enum class DirectNativeQueryStatus { verified, unavailable };
struct DirectDriverQuery { DirectDriverBinding binding;struct { DirectNativeQueryStatus query_status=DirectNativeQueryStatus::unavailable; } native;struct { std::uint64_t maximum_payload_bytes=0; } backend; };
struct DirectGpuCommissioningConfig { std::string campaign_id,run_id;DirectDriverBinding expected;TransferDescriptor descriptor;std::uint64_t pattern_seed=0;bool receiver=false;std::chrono::milliseconds timeout{10000};std::uint32_t maximum_progress_polls=10000; };
struct DirectGpuCommissioningReport { bool executed=false,controller_frames_complete=false,gpu_acquire_complete=false,consumer_complete=false,receive_reposted=false,local_tx_complete=false,transport_ack=false,consumer_credit=false,full_payload_verified=false;std::uint64_t completed_count=0;DirectPoolReleaseStatus cleanup=DirectPoolReleaseStatus::failed_retained;std::uint64_t pool_binding_identity=0,import_generation=0;std::string error; };
struct DirectGpuCommissioningSessionReport { std::vector<DirectGpuCommissioningReport> transfers;std::uint32_t requested_transfers=0;std::uint64_t completed_count=0,pool_binding_identity=0,import_generation=0;DirectPoolReleaseStatus cleanup=DirectPoolReleaseStatus::failed_retained;std::string error; };
struct DirectGpuCommissioningPayload { std::vector<TransportBufferView> send_segments,receive_segments;std::function<void()> produce;std::function<bool()> verify_full_payload; };
void validate_direct_gpu_commissioning_config(const DirectGpuCommissioningConfig&);
bool verify_direct_gpu_commissioning_pattern(std::span<const std::byte>,std::uint64_t,std::uint64_t) noexcept;
struct DirectStreamDeviceIdentity {};
DirectStreamDeviceIdentity inspect_direct_stream_device(const std::string&);
struct DistributedControlByteStream { virtual ~DistributedControlByteStream()=default; };
struct LinuxDirectGpuStreamPolicy { bool experimental_opt_in=false;std::uint64_t maximum_payload_bytes=0,maximum_imported_bytes=0;std::uint32_t maximum_event_calls=0; };
struct LinuxDirectGpuStreamCalls { std::function<std::optional<std::string>()> driver_build_identity;std::function<std::string()> kernel_release;static LinuxDirectGpuStreamCalls native(); };
class LinuxDirectGpuStreamCommissioning {
public:
 LinuxDirectGpuStreamCommissioning(DirectStreamDeviceIdentity,DirectDriverBinding,std::shared_ptr<DirectGpuStreamGpu>,std::shared_ptr<DistributedControlByteStream>,LinuxDirectGpuStreamPolicy);
 ~LinuxDirectGpuStreamCommissioning();
 DirectDriverQuery observe();void bind_exact_observed(const DirectDriverBinding&);
 DirectGpuCommissioningReport run_once(const DirectGpuCommissioningConfig&,const DirectGpuCommissioningPayload&);
 DirectGpuCommissioningSessionReport run_session(std::span<const DirectGpuCommissioningConfig>,std::span<const DirectGpuCommissioningPayload>);
 DirectGpuCommissioningSessionReport run_duplex_session(std::span<const DirectGpuCommissioningConfig>,std::span<const DirectGpuCommissioningPayload>);
};
}
