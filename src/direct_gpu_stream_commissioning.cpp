#include "strix/direct_gpu_stream_commissioning.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strix::transport {
namespace {
bool identifier(const std::string& value) {
    return !value.empty() && value.size() <= 128 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    });
}
bool bounded_identity(const std::string& value) {
    return !value.empty() && value.size() <= 256 && std::all_of(value.begin(), value.end(), [](unsigned char character) { return character >= 32 && character <= 126; });
}
bool same_descriptor(const TransferDescriptor& a, const TransferDescriptor& b) {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch && a.model_plan_id == b.model_plan_id &&
        a.request_id == b.request_id && a.generation == b.generation && a.boundary_id == b.boundary_id && a.sequence == b.sequence &&
        a.shape_fingerprint == b.shape_fingerprint && a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
}
void validate_direct_gpu_commissioning_config(const DirectGpuCommissioningConfig& config) {
    const auto& descriptor = config.descriptor;
    if (!identifier(config.campaign_id) || !identifier(config.run_id) || config.timeout.count() <= 0 || config.timeout > std::chrono::minutes{1} ||
        !config.maximum_progress_polls || config.maximum_progress_polls > 100000 || config.expected.device_identity.empty() || !config.expected.device_instance_generation ||
        !bounded_identity(config.expected.interface.local_endpoint_identity) || !bounded_identity(config.expected.interface.peer_endpoint_identity) ||
        config.expected.interface.local_endpoint_identity == config.expected.interface.peer_endpoint_identity || !config.expected.interface.connection_epoch ||
        !bounded_identity(config.expected.interface.physical_link_id) || !bounded_identity(config.expected.interface.kernel_release) || !bounded_identity(config.expected.interface.driver_identity) || !bounded_identity(config.expected.interface.abi_identity) ||
        !validate_descriptor(descriptor, {1, config.expected.interface.connection_epoch, 16ULL * 1024ULL * 1024ULL}) ||
        !descriptor.sequence || !descriptor.model_plan_id || !descriptor.request_id || !descriptor.generation || !descriptor.boundary_id || !descriptor.shape_fingerprint || !descriptor.dtype_id) {
        throw std::invalid_argument("commissioning requires complete exact identity, descriptor and bounded explicit policy");
    }
}
bool verify_direct_gpu_commissioning_pattern(std::span<const std::byte> bytes, std::uint64_t seed, std::uint64_t sequence) noexcept {
    if (bytes.empty() || !sequence) return false;
    bool equal = true;
    for (std::size_t index = 0; index < bytes.size(); ++index) equal = (std::to_integer<std::uint8_t>(bytes[index]) == direct_gpu_commissioning_pattern_byte(index, seed, sequence)) && equal;
    return equal;
}
DirectGpuCommissioningProgress DirectGpuCommissioningProgress::native() {
    return {[] { return std::chrono::steady_clock::now(); }, [] { std::this_thread::sleep_for(std::chrono::milliseconds{1}); }};
}
LinuxDirectGpuStreamCommissioning::LinuxDirectGpuStreamCommissioning(DirectStreamDeviceIdentity route, DirectDriverBinding discovery,
    std::shared_ptr<DirectGpuStreamGpu> gpu, std::shared_ptr<DistributedControlByteStream> control, LinuxDirectGpuStreamPolicy policy, LinuxDirectGpuStreamCalls calls)
    : gpu_(std::move(gpu)) {
    if (!discovery.device_identity.empty() || discovery.device_instance_generation) throw std::invalid_argument("commissioning must observe its exact newly opened FD before binding");
    provider_ = std::make_unique<LinuxDirectTransportDriverIo>(std::move(route), std::move(discovery), gpu_, std::move(control), policy, nullptr, std::move(calls));
}
LinuxDirectGpuStreamCommissioning::~LinuxDirectGpuStreamCommissioning() = default;
DirectDriverQuery LinuxDirectGpuStreamCommissioning::observe() {
    auto result = provider_->query();
    if (result.native.query_status == DirectNativeQueryStatus::verified || provider_->current_binding()) throw std::logic_error("commissioning must not create production authority");
    return result;
}
void LinuxDirectGpuStreamCommissioning::bind_exact_observed(const DirectDriverBinding& value) {
    if (bound_ || ran_) throw std::logic_error("commissioning binding cannot be replaced");
    provider_->bind_commissioning_observed(value); binding_ = value; bound_ = true;
}
DirectGpuCommissioningReport LinuxDirectGpuStreamCommissioning::run_once(const DirectGpuCommissioningConfig& config,
    const DirectGpuCommissioningPayload& payload, DirectGpuCommissioningProgress progress) {
    validate_direct_gpu_commissioning_config(config);
    if (!bound_ || ran_ || config.expected != binding_ || !progress.now || !progress.pause ||
        (config.receiver ? payload.receive_segments.empty() || !payload.verify_full_payload : payload.send_segments.empty() || !payload.produce)) throw std::invalid_argument("commissioning run requires one exact bound role and retained payload callbacks");
    ran_ = true; DirectGpuCommissioningReport report; report.pool_binding_identity = gpu_->exported_binding().pool_binding_identity;
    try {
        auto last = progress.now();
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(config.timeout);
        if (last > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("commissioning deadline overflow");
        const auto deadline = last + duration;
        const auto check = [&] {
            const auto now = progress.now(); if (now < last || now >= deadline) throw std::runtime_error("commissioning total deadline expired or clock changed"); last = now;
        };
        const auto& pool = gpu_->exported_binding().pool;
        const auto imported = dispatch([&] { return provider_->prepare_import(binding_, {binding_.interface.physical_link_id, binding_.interface.connection_epoch, report.pool_binding_identity, 0, 1}, pool); });
        report.import_generation = imported->binding().import_generation; check();
        dispatch([&] { provider_->activate_import(imported); }); check();
        auto view = config.receiver ? pool.rx : pool.tx; view.bytes = config.descriptor.byte_length;
        DirectDriverOperation operation{imported->binding(), 1, config.descriptor, view, config.receiver};
        if (!config.receiver) {
            report.executed = true;
            payload.produce(); check();
            dispatch([&] { gpu_->stage_outbound_segments(config.descriptor, payload.send_segments, view); }); check();
        }
        dispatch([&] { provider_->prepare(operation); }); check();
        if (!config.receiver) dispatch([&] { static_cast<void>(provider_->publish_gpu_commit(operation, {1, gpu_commit_word, config.descriptor.connection_epoch, config.descriptor.sequence, static_cast<std::uint32_t>(config.descriptor.byte_length), 0, 1})); });
        check(); report.executed = true; dispatch([&] { provider_->submit(operation); }); check();
        const auto wait = [&](CompletionKind expected) {
            for (std::uint32_t polls = 0; polls < config.maximum_progress_polls; ++polls) {
                check(); const auto event = dispatch([&] { return provider_->poll_event(); }); check();
                if (!event) { progress.pause(); continue; }
                if (event->operation.handle != operation.handle || event->operation.receive_path != config.receiver || !same_descriptor(event->operation.descriptor, config.descriptor) ||
                    event->event.completion.handle != operation.handle || event->event.completion.bytes != config.descriptor.byte_length || event->event.completion.kind != expected) throw std::runtime_error("commissioning completion is foreign, failed or out of order");
                return;
            }
            throw std::runtime_error("commissioning bounded progress exhausted");
        };
        if (config.receiver) {
            wait(CompletionKind::receive_complete); report.controller_frames_complete = true;
            dispatch([&] { static_cast<void>(provider_->read_gpu_commit(operation)); }); check(); report.gpu_acquire_complete = true;
            dispatch([&] { provider_->consume_received_segments(config.descriptor, payload.receive_segments); }); check();
            dispatch([&] { static_cast<void>(provider_->consumer_completion(operation)); }); check(); report.consumer_complete = true;
            if (!payload.verify_full_payload()) throw std::runtime_error("commissioning complete acquired/unpacked payload mismatch");
            check(); report.full_payload_verified = true;
            dispatch([&] { provider_->publish_consumer_credit(operation, config.descriptor.byte_length); }); check(); report.consumer_credit = true;
            dispatch([&] { provider_->repost_receive(operation); }); wait(CompletionKind::receive_reposted); report.receive_reposted = true;
        } else {
            wait(CompletionKind::local_buffer_reusable); report.controller_frames_complete = report.local_tx_complete = true;
            wait(CompletionKind::remote_transport_ack); report.transport_ack = true;
            wait(CompletionKind::remote_consumer_complete); report.consumer_credit = true;
        }
        dispatch([&] { provider_->retire(operation); }); check(); report.completed_count = 1;
    } catch (const std::exception& error) { report.error = error.what(); provider_->invalidate(); }
    catch (...) { report.error = "commissioning unknown callback failure"; provider_->invalidate(); }
    report.cleanup = provider_->close();
    if (report.cleanup != DirectPoolReleaseStatus::released && report.error.empty()) report.error = "commissioning cleanup retained pending controller/GPU resources";
    return report;
}
DirectPoolReleaseStatus LinuxDirectGpuStreamCommissioning::close() noexcept { return provider_->close(); }
} // namespace strix::transport
