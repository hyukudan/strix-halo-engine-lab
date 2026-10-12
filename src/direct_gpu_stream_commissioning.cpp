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
    auto session = run_batch(std::span{&config, 1}, std::span{&payload, 1}, std::move(progress));
    DirectGpuCommissioningReport report;
    if (!session.transfers.empty()) report = std::move(session.transfers.front());
    report.pool_binding_identity = session.pool_binding_identity; report.import_generation = session.import_generation;
    report.cleanup = session.cleanup; if (report.error.empty()) report.error = std::move(session.error);
    return report;
}
DirectGpuCommissioningSessionReport LinuxDirectGpuStreamCommissioning::run_session(std::span<const DirectGpuCommissioningConfig> configs,
    std::span<const DirectGpuCommissioningPayload> payloads, DirectGpuCommissioningProgress progress) {
    if (configs.size() < 2) throw std::invalid_argument("commissioning session reuse requires at least two planned transfers");
    return run_batch(configs, payloads, std::move(progress));
}
DirectGpuCommissioningSessionReport LinuxDirectGpuStreamCommissioning::run_batch(std::span<const DirectGpuCommissioningConfig> configs,
    std::span<const DirectGpuCommissioningPayload> payloads, DirectGpuCommissioningProgress progress, bool duplex) {
    if (!bound_ || ran_ || configs.empty() || configs.size() > 128 || payloads.size() != configs.size() || configs.front().descriptor.sequence != 1 || !progress.now || !progress.pause) throw std::invalid_argument("commissioning requires one bounded serialized exact transfer schedule beginning at sequence one");
    if (duplex && (configs.size() % 2 || configs.size() < 2)) throw std::invalid_argument("paired duplex requires an even bounded transfer schedule");
    std::uint64_t total_bytes = 0;
    for (std::size_t index = 0; index < configs.size(); ++index) {
        const auto& config = configs[index]; const auto& payload = payloads[index]; validate_direct_gpu_commissioning_config(config);
        if (config.expected != binding_ || config.campaign_id != configs.front().campaign_id ||
            (duplex && config.receiver != (index % 2 ? !configs.front().receiver : configs.front().receiver)) ||
            (index && (configs[index - 1].descriptor.sequence == UINT64_MAX || config.descriptor.sequence != configs[index - 1].descriptor.sequence + 1)) ||
            (config.receiver ? payload.receive_segments.empty() || !payload.verify_full_payload : payload.send_segments.empty() || !payload.produce)) throw std::invalid_argument("commissioning schedule contains foreign binding, nonmonotonic sequence or missing payload callback");
        for (std::size_t prior = 0; prior < index; ++prior) if (configs[prior].run_id == config.run_id) throw std::invalid_argument("commissioning transfer identity is duplicated");
        const auto& segments = config.receiver ? payload.receive_segments : payload.send_segments;
        if (segments.size() > 64) throw std::invalid_argument("commissioning payload segment count exceeds bound");
        std::uint64_t bytes = 0;
        for (const auto& view : segments) {
            if (view.domain != TransportMemoryDomain::gpu_device || view.bytes > config.descriptor.byte_length - bytes ||
                !(config.receiver ? validate_transport_receive_buffer(view, view.bytes) : validate_transport_send_buffer(view, view.bytes))) throw std::invalid_argument("commissioning segment extent/domain/ownership is invalid");
            bytes += view.bytes;
        }
        if (bytes != config.descriptor.byte_length) throw std::invalid_argument("commissioning segments do not cover the exact payload");
        if (config.descriptor.byte_length > 128ULL * 1024ULL * 1024ULL - total_bytes) throw std::invalid_argument("commissioning session aggregate payload exceeds 128 MiB");
        total_bytes += config.descriptor.byte_length;
    }
    ran_ = true; DirectGpuCommissioningSessionReport result; result.requested_transfers = static_cast<std::uint32_t>(configs.size());
    result.pool_binding_identity = gpu_->exported_binding().pool_binding_identity; result.transfers.reserve(configs.size());
    try {
        if (duplex) provider_->enable_commissioning_duplex();
        auto last = progress.now();
        const auto first_duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(duplex ? std::min(configs.front().timeout, configs[1].timeout) : configs.front().timeout);
        if (last > std::chrono::steady_clock::time_point::max() - first_duration) throw std::overflow_error("commissioning deadline overflow");
        auto deadline = last + first_duration;
        const auto check = [&] {
            const auto now = progress.now(); if (now < last || now >= deadline) throw std::runtime_error("commissioning total deadline expired or clock changed"); last = now;
        };
        const auto& pool = gpu_->exported_binding().pool;
        const auto imported = dispatch([&] { return provider_->prepare_import(binding_, {binding_.interface.physical_link_id, binding_.interface.connection_epoch, result.pool_binding_identity, 0, 1}, pool); });
        result.import_generation = imported->binding().import_generation;
        check(); dispatch([&] { provider_->activate_import(imported); }); check();
        for (std::size_t index = 0; index < configs.size(); index += duplex ? 2U : 1U) {
        const auto& config = configs[index]; const auto& payload = payloads[index];
        if (duplex) {
            result.transfers.emplace_back(); result.transfers.emplace_back();
            std::array<DirectDriverOperation, 2> operations;
            for (std::size_t member = 0; member < 2; ++member) {
                const auto& step = configs[index + member]; auto& report = result.transfers[index + member];
                report.pool_binding_identity = result.pool_binding_identity; report.import_generation = result.import_generation; report.cleanup = DirectPoolReleaseStatus::pending;
                auto view = step.receiver ? pool.rx : pool.tx; view.bytes = step.descriptor.byte_length;
                operations[member] = {imported->binding(), index + member + 1, step.descriptor, view, step.receiver};
            }
            if (index) {
                check(); const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::min(config.timeout, configs[index + 1].timeout));
                if (last > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("commissioning duplex deadline overflow");
                deadline = last + duration;
            }
            const std::size_t tx = config.receiver ? 1U : 0U, rx = 1U - tx;
            auto& sent = result.transfers[index + tx]; auto& received = result.transfers[index + rx];
            sent.executed = true; payloads[index + tx].produce(); check();
            dispatch([&] { gpu_->stage_outbound_segments(operations[tx].descriptor, payloads[index + tx].send_segments, operations[tx].buffer); }); check();
            for (const auto& operation : operations) { dispatch([&] { provider_->prepare(operation); }); check(); }
            const auto& descriptor = operations[tx].descriptor;
            dispatch([&] { static_cast<void>(provider_->publish_gpu_commit(operations[tx], {1, gpu_commit_word, descriptor.connection_epoch, descriptor.sequence, static_cast<std::uint32_t>(descriptor.byte_length), 0, 1})); }); check();
            received.executed = true; dispatch([&] { provider_->submit(operations[rx]); }); check();
            dispatch([&] { provider_->submit(operations[tx]); }); check();
            const auto budget = static_cast<std::uint64_t>(config.maximum_progress_polls) + configs[index + 1].maximum_progress_polls;
            for (std::uint64_t polls = 0; polls < budget; ++polls) {
                check(); const auto event = dispatch([&] { return provider_->poll_event(); }); check();
                if (!event) { progress.pause(); continue; }
                const auto member = event->operation.receive_path ? rx : tx;
                const auto& operation = operations[member]; auto& report = result.transfers[index + member];
                if (event->operation.handle != operation.handle || !same_descriptor(event->operation.descriptor, operation.descriptor) || event->event.completion.handle != operation.handle || event->event.completion.bytes != operation.descriptor.byte_length) throw std::runtime_error("duplex completion is foreign to the bounded owned pair");
                switch (event->event.completion.kind) {
                case CompletionKind::receive_complete:
                    if (member != rx || report.controller_frames_complete) throw std::runtime_error("duplex RX completion is duplicate or out of order");
                    report.controller_frames_complete = true;
                    dispatch([&] { static_cast<void>(provider_->read_gpu_commit(operation)); }); check(); report.gpu_acquire_complete = true;
                    dispatch([&] { provider_->consume_received_segments(operation.descriptor, payloads[index + rx].receive_segments); }); check();
                    dispatch([&] { static_cast<void>(provider_->consumer_completion(operation)); }); check(); report.consumer_complete = true;
                    if (!payloads[index + rx].verify_full_payload()) throw std::runtime_error("duplex complete acquired/unpacked payload mismatch");
                    check(); report.full_payload_verified = true;
                    dispatch([&] { provider_->publish_consumer_credit(operation, operation.descriptor.byte_length); }); check(); report.consumer_credit = true;
                    dispatch([&] { provider_->repost_receive(operation); }); check(); break;
                case CompletionKind::receive_reposted:
                    if (member != rx || !report.consumer_credit || report.receive_reposted) throw std::runtime_error("duplex repost lacks one completed verified consumer");
                    report.receive_reposted = true; break;
                case CompletionKind::local_buffer_reusable:
                    if (member != tx || report.local_tx_complete) throw std::runtime_error("duplex TX completion is duplicate or foreign");
                    report.controller_frames_complete = report.local_tx_complete = true; break;
                case CompletionKind::remote_transport_ack:
                    if (member != tx || !report.local_tx_complete || report.transport_ack) throw std::runtime_error("duplex ACK is duplicate or out of order");
                    report.transport_ack = true; break;
                case CompletionKind::remote_consumer_complete:
                    if (member != tx || !report.transport_ack || report.consumer_credit) throw std::runtime_error("duplex credit is duplicate or out of order");
                    report.consumer_credit = true; break;
                default: throw std::runtime_error("duplex completion is failed or unsupported");
                }
                if (received.receive_reposted && sent.consumer_credit) break;
            }
            if (!received.receive_reposted || !sent.consumer_credit) throw std::runtime_error("duplex bounded pair progress exhausted");
            for (std::size_t member = 0; member < 2; ++member) { dispatch([&] { provider_->retire(operations[member]); }); check(); result.transfers[index + member].completed_count = 1; ++result.completed_count; }
            continue;
        }
        result.transfers.emplace_back(); auto& report = result.transfers.back(); report.pool_binding_identity = result.pool_binding_identity; report.import_generation = result.import_generation;
        report.cleanup = DirectPoolReleaseStatus::pending; // The session deliberately retains the same import between transfers.
        if (index) {
            check();
            const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(config.timeout);
            if (last > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("commissioning deadline overflow");
            deadline = last + duration;
        }
        auto view = config.receiver ? pool.rx : pool.tx; view.bytes = config.descriptor.byte_length;
        DirectDriverOperation operation{imported->binding(), index + 1, config.descriptor, view, config.receiver};
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
        dispatch([&] { provider_->retire(operation); }); check(); report.completed_count = 1; ++result.completed_count;
        }
    } catch (const std::exception& error) { result.error = error.what(); provider_->invalidate(); }
    catch (...) { result.error = "commissioning unknown callback failure"; provider_->invalidate(); }
    if (!result.error.empty()) for (auto& transfer : result.transfers) if (!transfer.completed_count) transfer.error = result.error;
    result.cleanup = provider_->close();
    if (result.cleanup != DirectPoolReleaseStatus::released && result.error.empty()) result.error = "commissioning cleanup retained pending controller/GPU resources";
    return result;
}
DirectGpuCommissioningSessionReport LinuxDirectGpuStreamCommissioning::run_duplex_session(std::span<const DirectGpuCommissioningConfig> configs,
    std::span<const DirectGpuCommissioningPayload> payloads, DirectGpuCommissioningProgress progress) {
    return run_batch(configs, payloads, std::move(progress), true);
}
DirectPoolReleaseStatus LinuxDirectGpuStreamCommissioning::close() noexcept { return provider_->close(); }
} // namespace strix::transport
