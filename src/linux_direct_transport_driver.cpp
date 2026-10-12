#include "strix/linux_direct_transport_driver.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <deque>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#if defined(__linux__)
#include "strix/linux_direct_gpu_stream_uapi.h"
#include <sys/ioctl.h>
#include <sys/utsname.h>
#endif

namespace strix::transport {
namespace {
[[maybe_unused]] bool same_view(const TransportBufferView& a, const TransportBufferView& b) {
    return a.address == b.address && a.bytes == b.bytes && a.allocation_generation == b.allocation_generation && a.domain == b.domain && a.access == b.access;
}
[[maybe_unused]] bool same_descriptor(const TransferDescriptor& a, const TransferDescriptor& b) {
    return a.protocol_version == b.protocol_version && a.connection_epoch == b.connection_epoch && a.model_plan_id == b.model_plan_id &&
        a.request_id == b.request_id && a.generation == b.generation && a.boundary_id == b.boundary_id && a.sequence == b.sequence &&
        a.shape_fingerprint == b.shape_fingerprint && a.dtype_id == b.dtype_id && a.byte_length == b.byte_length;
}
[[maybe_unused]] bool same_import(const DirectDriverImportBinding& a, const DirectDriverImportBinding& b) {
    return a.device == b.device && a.import_generation == b.import_generation && a.lease.physical_link_id == b.lease.physical_link_id &&
        a.lease.connection_epoch == b.lease.connection_epoch && a.lease.pool_binding_identity == b.lease.pool_binding_identity &&
        a.lease.link_index == b.lease.link_index && a.lease.lease_generation == b.lease.lease_generation && same_view(a.pool.tx, b.pool.tx) && same_view(a.pool.rx, b.pool.rx);
}
[[maybe_unused]] bool same_operation(const DirectDriverOperation& a, const DirectDriverOperation& b) {
    return same_import(a.imported, b.imported) && a.handle == b.handle && a.receive_path == b.receive_path && same_descriptor(a.descriptor, b.descriptor) && same_view(a.buffer, b.buffer);
}
class Import final : public DirectDriverImport {
public:
    DirectDriverImportBinding value;
    std::shared_ptr<DirectGpuExportImportLease> lease;
    const DirectDriverImportBinding& binding() const noexcept override { return value; }
};
#if defined(__linux__)
bool same_fence(const sgu_fence& a, const sgu_fence& b) {
    return a.abi_major == b.abi_major && a.abi_minor == b.abi_minor && a.abi_cookie == b.abi_cookie &&
        std::equal(std::begin(a.device_instance), std::end(a.device_instance), std::begin(b.device_instance)) &&
        std::equal(std::begin(a.local_uuid), std::end(a.local_uuid), std::begin(b.local_uuid)) &&
        std::equal(std::begin(a.peer_uuid), std::end(a.peer_uuid), std::begin(b.peer_uuid)) && a.route == b.route &&
        a.in_hopid == b.in_hopid && a.out_hopid == b.out_hopid && a.connection_epoch == b.connection_epoch &&
        a.pool_binding_identity == b.pool_binding_identity && a.import_generation == b.import_generation;
}
bool same_buffer(const sgu_buffer& a, const sgu_buffer& b) {
    return a.fd == b.fd && a.reserved == b.reserved && a.full_bytes == b.full_bytes && a.window_offset == b.window_offset &&
        a.window_bytes == b.window_bytes && a.allocation_generation == b.allocation_generation && a.fd_device == b.fd_device && a.fd_inode == b.fd_inode;
}
std::string hex_uuid(const __u8 (&value)[16]) {
    constexpr char digits[] = "0123456789abcdef"; std::string result; result.reserve(32);
    for (const auto byte : value) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); } return result;
}
std::uint64_t instance_generation(const __u8 (&value)[16]) {
    std::uint64_t low = 0, high = 0;
    for (unsigned i = 0; i < 8; ++i) { low |= static_cast<std::uint64_t>(value[i]) << (i * 8U); high |= static_cast<std::uint64_t>(value[i + 8]) << (i * 8U); }
    return low ? low : high; // Whole UUID remains in device_identity and kernel fence.
}
struct KernelError : std::runtime_error {
    int number;
    explicit KernelError(int error) : std::runtime_error("experimental direct GPU stream syscall failed"), number(error) {}
};
struct BuildUnavailable : std::runtime_error { BuildUnavailable() : std::runtime_error("loaded direct GPU driver GNU build-id is unavailable") {} };
#endif
} // namespace
std::string parse_linux_direct_gpu_stream_module_build_id(std::span<const std::byte> note) {
    if (note.size() < 20 || note.size() > 128) throw std::invalid_argument("module GNU build-id note is truncated or excessive");
    const auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned index = 0; index < 4; ++index) {
            const auto shift = std::endian::native == std::endian::little ? index * 8U : (3U - index) * 8U;
            value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(note[offset + index])) << shift;
        }
        return value;
    };
    const auto names = word(0), bytes = word(4), type = word(8);
    if (names != 4 || !bytes || bytes > 64 || type != 3 || note[12] != std::byte{'G'} || note[13] != std::byte{'N'} || note[14] != std::byte{'U'} || note[15] != std::byte{0} ||
        note.size() != 16 + ((static_cast<std::size_t>(bytes) + 3) & ~std::size_t{3})) throw std::invalid_argument("module note is not one exact GNU build-id");
    constexpr char digits[] = "0123456789abcdef"; std::string identity; identity.reserve(bytes * 2U);
    for (std::uint32_t index = 0; index < bytes; ++index) { const auto byte = std::to_integer<unsigned char>(note[16 + index]); identity.push_back(digits[byte >> 4]); identity.push_back(digits[byte & 15]); }
    for (std::size_t index = 16 + bytes; index < note.size(); ++index) if (note[index] != std::byte{0}) throw std::invalid_argument("module GNU build-id padding is noncanonical");
    return identity;
}
std::optional<std::string> observe_linux_direct_gpu_stream_module_build_id() {
#if defined(__linux__)
    try {
        std::ifstream input("/sys/module/thunderbolt_stream/notes/.note.gnu.build-id", std::ios::binary);
        if (!input) return std::nullopt;
        std::array<char, 129> bytes{}; input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        const auto count = input.gcount(); if (count <= 0 || count > 128 || input.bad()) return std::nullopt;
        return parse_linux_direct_gpu_stream_module_build_id({reinterpret_cast<const std::byte*>(bytes.data()), static_cast<std::size_t>(count)});
    } catch (...) { return std::nullopt; }
#else
    return std::nullopt;
#endif
}
LinuxDirectGpuStreamCalls LinuxDirectGpuStreamCalls::native() {
    LinuxDirectGpuStreamCalls calls; calls.fd_owner = std::make_shared<PosixDirectStreamControlIo>();
#if defined(__linux__)
    calls.ioctl_call = [](DirectStreamFd fd, unsigned long command, void* argument) {
        const auto result = ::ioctl(fd.descriptor, command, argument); return LinuxDirectGpuStreamSyscallResult{result, result < 0 ? errno : 0};
    };
    calls.kernel_release = [] { struct utsname value{}; if (::uname(&value) != 0) throw std::runtime_error("kernel identity unavailable"); return std::string(value.release); };
#else
    calls.ioctl_call = [](DirectStreamFd, unsigned long, void*) { return LinuxDirectGpuStreamSyscallResult{-1, ENOTSUP}; };
    calls.kernel_release = [] { return std::string{}; };
#endif
    calls.driver_build_identity = observe_linux_direct_gpu_stream_module_build_id;
    return calls;
}
struct LinuxDirectTransportDriverIo::Impl {
    struct Operation {
        DirectDriverOperation binding;
        GpuCommitRecord commit;
        std::uint64_t offset = 0;
        std::uint32_t frame_count = 0, completed_frames = 0;
        bool submitted = false, released = false, hardware_done = false, acquired = false, consumer_done = false, credited = false, reposted = false,
            transport_ack = false, remote_consumer = false;
    };
    DirectStreamDeviceIdentity route;
    DirectDriverBinding expected;
    std::shared_ptr<DirectGpuStreamGpu> gpu;
    std::shared_ptr<DistributedControlByteStream> control;
    LinuxDirectGpuStreamPolicy policy;
    std::shared_ptr<DirectGpuStreamQualification> qualification;
    LinuxDirectGpuStreamCalls calls;
    DirectStreamFd fd;
    DirectDriverQuery cached;
    std::shared_ptr<Import> imported;
    DirectDriverImportBinding import_binding;
    std::array<std::optional<Operation>, 2> operations;
    std::deque<DirectDriverEvent> events;
    std::uint64_t last_send = 0, last_receive = 0;
    bool failed = false, closing = false, queried = false, ready = false, armed = false, controller_stopped = false, kernel_released = false;
    bool discovery_placeholder = false, observed_bound = false;
    bool commissioning = false, commissioning_operation = false, commissioning_duplex = false;
    std::string observed_driver_build;
#if defined(__linux__)
    sgu_query observed{};
    sgu_fence fence{};
    std::string build_identity() {
        const auto build = calls.driver_build_identity();
        if (!build || build->empty() || build->size() > 128 || build->size() % 2 ||
            !std::all_of(build->begin(), build->end(), [](char character) { return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'); })) throw BuildUnavailable{};
        return "thunderbolt_stream:gnu-build-id:" + *build;
    }
    void identity() {
        const auto current = calls.fd_owner->inspect(fd);
        if (current.device != route || current.open_generation != fd.open_generation) throw std::runtime_error("direct GPU stream FD/device generation changed");
        if (!observed_driver_build.empty() && build_identity() != observed_driver_build) throw std::runtime_error("direct GPU loaded driver build identity changed");
    }
    LinuxDirectGpuStreamSyscallResult raw_call(unsigned long command, void* argument) {
        identity(); const auto result = calls.ioctl_call(fd, command, argument); identity(); return result;
    }
    void call(unsigned long command, void* argument) {
        const auto result = raw_call(command, argument);
        if (result.result != 0 || result.error) throw KernelError(result.error ? result.error : EIO);
    }
    sgu_query inspect_kernel() {
        sgu_query result{}; result.fence.abi_major = SGU_ABI_MAJOR; result.fence.abi_minor = SGU_ABI_MINOR; call(SGU_IOC_QUERY, &result);
        if (result.fence.abi_major != SGU_ABI_MAJOR || result.fence.abi_minor != SGU_ABI_MINOR || result.fence.abi_cookie != SGU_ABI_COOKIE ||
            !instance_generation(result.fence.device_instance) || hex_uuid(result.fence.local_uuid) == std::string(32, '0') ||
            hex_uuid(result.fence.peer_uuid) == std::string(32, '0') || hex_uuid(result.fence.local_uuid) == hex_uuid(result.fence.peer_uuid) ||
            result.frame_bytes != SGU_FRAME_BYTES || result.ring_frames < 32 || result.ring_frames > SGU_MAX_FRAMES ||
            result.maximum_sg_entries == 0 || result.maximum_sg_entries > SGU_MAX_SG_ENTRIES ||
            result.maximum_window_bytes != static_cast<std::uint64_t>(result.ring_frames) * SGU_FRAME_BYTES ||
            result.maximum_imported_bytes == 0 || result.maximum_imported_bytes > SGU_MAX_IMPORTED_BYTES || result.maximum_event_batch != SGU_EVENT_BATCH || result.reserved) {
            throw std::runtime_error("experimental direct GPU stream QUERY ABI/limits/peer is incompatible");
        }
        return result;
    }
    DirectDriverBinding binding(const sgu_query& result) {
        DirectDriverBinding value;
        value.device_identity = route.canonical_path + "#" + std::to_string(route.character_device_number) + ":" + hex_uuid(result.fence.device_instance) +
            ":" + std::to_string(result.fence.route) + ":" + std::to_string(result.fence.in_hopid) + ":" + std::to_string(result.fence.out_hopid);
        value.device_instance_generation = instance_generation(result.fence.device_instance);
        value.interface = {1, calls.kernel_release(), build_identity(), "sgu-1.0-5354475055303031",
            hex_uuid(result.fence.local_uuid), hex_uuid(result.fence.peer_uuid), expected.interface.physical_link_id, expected.interface.connection_epoch};
        return value;
    }
    void current() {
        const bool authorized = commissioning ? commissioning_operation : ready;
        if (failed || closing || !authorized || !imported) throw std::logic_error("direct GPU stream is unavailable, failed, closing or unimported");
        identity();
        if (!same_import(imported->value, import_binding)) throw std::runtime_error("direct GPU stream import token binding changed");
        imported->lease->require_current(); gpu->require_current();
        if (!commissioning && (!qualification || !qualification->current(expected, gpu->exported_binding()))) throw std::runtime_error("direct GPU stream qualification invalidated");
        imported->lease->require_current(); gpu->require_current(); identity();
    }
    Operation& require(const DirectDriverOperation& operation) {
        current(); auto& entry = operations[operation.receive_path ? 1U : 0U];
        if (!entry || !same_operation(entry->binding, operation)) throw std::invalid_argument("direct GPU stream operation is foreign or stale");
        return *entry;
    }
    Operation& receive(const TransferDescriptor& descriptor) {
        current(); if (!operations[1] || !same_descriptor(operations[1]->binding.descriptor, descriptor)) throw std::invalid_argument("direct GPU receive descriptor is stale or foreign"); return *operations[1];
    }
    void action(unsigned long command, const Operation* operation, std::uint32_t expected_result, std::uint32_t expected_frames) {
        sgu_action request{}; request.fence = fence;
        if (operation) { request.handle = operation->binding.handle; request.sequence = operation->binding.descriptor.sequence; request.direction = operation->binding.receive_path ? SGU_DIRECTION_RX : SGU_DIRECTION_TX; }
        const auto before = request; call(command, &request);
        if (!same_fence(request.fence, before.fence) || request.handle != before.handle || request.sequence != before.sequence || request.direction != before.direction ||
            request.result != expected_result || request.queued_frames != expected_frames || request.reserved) throw std::runtime_error("direct GPU stream action acknowledgement is foreign");
    }
    TransportBufferView terminal(bool receive_path) const {
        auto view = receive_path ? imported->value.pool.rx : imported->value.pool.tx; view.address += view.bytes - SGU_FRAME_BYTES; view.bytes = sizeof(GpuCommitRecord); return view;
    }
    void send_control(const Operation& operation, DistributedDirectGpuControlKind kind, const GpuCommitRecord& commit = {}) {
        DistributedDirectGpuControlFrame frame; frame.kind = kind; frame.physical_link_id = expected.interface.physical_link_id;
        frame.local_endpoint_identity = expected.interface.local_endpoint_identity; frame.peer_endpoint_identity = expected.interface.peer_endpoint_identity;
        frame.descriptor = operation.binding.descriptor; frame.commit = commit;
        current(); control->send(encode_distributed_direct_gpu_control(frame, {2, 2048, 1})); current();
    }
    GpuCommitRecord receive_control(const Operation& operation, DistributedDirectGpuControlKind kind) {
        current(); const auto frame = decode_distributed_direct_gpu_control(control->receive(), {2, 2048, 1}); current();
        if (frame.kind != kind || frame.physical_link_id != expected.interface.physical_link_id || frame.local_endpoint_identity != expected.interface.peer_endpoint_identity ||
            frame.peer_endpoint_identity != expected.interface.local_endpoint_identity || !same_descriptor(frame.descriptor, operation.binding.descriptor)) throw std::runtime_error("direct GPU peer control descriptor/endpoint/kind is foreign");
        return frame.commit;
    }
    void enqueue(Operation& operation, CompletionKind kind) {
        events.push_back({operation.binding, {{operation.binding.handle, kind, operation.binding.descriptor.byte_length}, operation.binding.receive_path, 0.0, operation.binding.descriptor.byte_length}});
    }
#endif
};

LinuxDirectTransportDriverIo::LinuxDirectTransportDriverIo(DirectStreamDeviceIdentity route, DirectDriverBinding expected,
    std::shared_ptr<DirectGpuStreamGpu> gpu, std::shared_ptr<DistributedControlByteStream> control, LinuxDirectGpuStreamPolicy policy,
    std::shared_ptr<DirectGpuStreamQualification> qualification, LinuxDirectGpuStreamCalls calls) : impl_(std::make_unique<Impl>()) {
    auto& i = *impl_; i.route = std::move(route); i.expected = std::move(expected); i.gpu = std::move(gpu); i.control = std::move(control);
    i.policy = policy; i.qualification = std::move(qualification); i.calls = std::move(calls);
    i.discovery_placeholder = i.expected.device_identity.empty() && i.expected.device_instance_generation == 0;
    if (!policy.experimental_opt_in || !policy.maximum_payload_bytes || policy.maximum_payload_bytes > 16ULL * 1024ULL * 1024ULL ||
        !policy.maximum_imported_bytes || policy.maximum_imported_bytes > 128ULL * 1024ULL * 1024ULL || !policy.maximum_event_calls ||
        !i.gpu || !i.gpu->import_lease() || !i.control ||
        !i.calls.fd_owner || !i.calls.ioctl_call || !i.calls.kernel_release || !i.calls.driver_build_identity || !i.expected.interface.connection_epoch || i.expected.interface.physical_link_id.empty() ||
        (!i.discovery_placeholder && (i.expected.device_identity.empty() || !i.expected.device_instance_generation))) throw std::invalid_argument("direct GPU stream requires explicit bounded experimental route, guarded GPU owner and dedicated control stream");
    if (const auto* stream = dynamic_cast<const DirectStreamControlByteStream*>(i.control.get()); stream && (!stream->valid() || stream->maximum_frame_bytes() < 2048)) throw std::invalid_argument("direct GPU copied control stream is invalid or undersized");
#if defined(__linux__)
    i.fd = i.calls.fd_owner->open(i.route);
    try { i.identity(); } catch (...) { i.calls.fd_owner->close(i.fd); i.fd = {}; throw; }
#endif
}
LinuxDirectTransportDriverIo::~LinuxDirectTransportDriverIo() {
    if (close() != DirectPoolReleaseStatus::released) static_cast<void>(impl_.release());
}
bool LinuxDirectTransportDriverIo::failed() const noexcept { return impl_->failed; }
void LinuxDirectTransportDriverIo::invalidate() noexcept { impl_->failed = true; impl_->ready = false; }
DirectDriverQuery LinuxDirectTransportDriverIo::query() {
    auto& i = *impl_; DirectDriverQuery result;
#if defined(__linux__)
    try {
        const auto observed = i.inspect_kernel(); result.binding = i.binding(observed);
        if (!i.observed_driver_build.empty() && result.binding.interface.driver_identity != i.observed_driver_build) throw std::runtime_error("direct GPU loaded driver build changed during QUERY");
        if (i.observed_driver_build.empty()) i.observed_driver_build = result.binding.interface.driver_identity;
        if (i.queried && (result.binding != i.cached.binding || !same_fence(observed.fence, i.fence))) throw std::runtime_error("direct GPU stream QUERY changed live device/import identity");
        if (!i.queried && (observed.state != SGU_STATE_OPEN || observed.fence.connection_epoch || observed.fence.pool_binding_identity || observed.fence.import_generation)) throw std::runtime_error("direct GPU stream endpoint already imported or unavailable");
        const auto expected_state = i.imported ? (i.armed ? SGU_STATE_ARMED : SGU_STATE_IMPORTED) : SGU_STATE_OPEN;
        if (i.failed || i.closing || i.controller_stopped || i.kernel_released || observed.state != expected_state) {
            i.failed = true;
            throw std::runtime_error("direct GPU stream QUERY state is failed, disconnected or incompatible with current phase");
        }
        i.observed = observed; i.fence = observed.fence; i.queried = true;
        result.backend.name = "experimental-linux-direct-gpu-stream"; result.backend.maximum_payload_bytes = std::min<std::uint64_t>(i.policy.maximum_payload_bytes, observed.maximum_window_bytes - SGU_FRAME_BYTES);
        result.backend.full_duplex = false; result.backend.external_dma = result.backend.supports_gpu_transport_pool = true;
        result.backend.requires_exclusive_lease = result.backend.explicit_receive_repost = result.backend.gpu_commit_record = true;
        result.backend.buffer_mode = TransportBufferMode::imported_gpu_dma_buf; result.backend.progress_mode = TransportProgressMode::interrupt_driven;
        result.native.observed_interface = result.binding.interface;
        if (i.commissioning) { result.native.query_status = DirectNativeQueryStatus::unavailable; i.ready = false; }
        else if (result.binding != i.expected) { result.native.query_status = i.discovery_placeholder ? DirectNativeQueryStatus::unavailable : DirectNativeQueryStatus::incompatible; i.ready = false; }
        else if (i.qualification) {
            result.native = i.qualification->proof(result.binding, i.gpu->exported_binding());
            i.ready = !i.failed && result.native.observed_interface == std::optional<DirectNativeInterfaceIdentity>(result.binding.interface) &&
                i.qualification->current(result.binding, i.gpu->exported_binding()) && assess_direct_native_readiness(result.backend, result.native).ready;
            if (!i.ready) result.native.query_status = DirectNativeQueryStatus::incompatible;
        }
        i.gpu->require_current(); i.identity();
        i.cached = result; return result;
    } catch (const BuildUnavailable&) { result.native.query_status = DirectNativeQueryStatus::unavailable; }
    catch (const KernelError& error) {
        result.native.query_status = error.number == ENOTTY || error.number == EOPNOTSUPP ? DirectNativeQueryStatus::unsupported : DirectNativeQueryStatus::inaccessible;
    } catch (...) { result.native.query_status = DirectNativeQueryStatus::incompatible; }
#else
    result.native.query_status = DirectNativeQueryStatus::unsupported;
#endif
    i.ready = false; if (i.imported) i.failed = true; i.cached = result; return result;
}
void LinuxDirectTransportDriverIo::bind_observed(const DirectDriverBinding& binding, std::shared_ptr<DirectGpuStreamQualification> qualification) {
#if defined(__linux__)
    auto& i = *impl_;
    if (!i.discovery_placeholder || i.observed_bound || i.failed || i.closing || !i.queried || i.imported || i.operations[0] || i.operations[1] || binding != i.cached.binding) throw std::invalid_argument("direct GPU observed binding requires one exact pre-import discovery adoption");
    i.identity(); const auto fresh = i.inspect_kernel(); const auto fresh_binding = i.binding(fresh);
    if (fresh.state != SGU_STATE_OPEN || fresh.fence.connection_epoch || fresh.fence.pool_binding_identity || fresh.fence.import_generation ||
        !same_fence(fresh.fence, i.fence) || fresh_binding != binding) throw std::runtime_error("direct GPU observed endpoint changed before binding");
    i.gpu->require_current(); i.identity();
    i.expected = binding; i.qualification = std::move(qualification); i.discovery_placeholder = false; i.observed_bound = true; i.ready = false;
#else
    static_cast<void>(binding); static_cast<void>(qualification); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::bind_commissioning_observed(const DirectDriverBinding& binding) {
    bind_observed(binding, {});
    const auto observed = query();
    if (observed.binding != binding || observed.native.query_status != DirectNativeQueryStatus::unavailable || !observed.backend.maximum_payload_bytes) throw std::runtime_error("commissioning exact observed binding is unavailable");
    impl_->commissioning = true;
}
void LinuxDirectTransportDriverIo::begin_commissioning_operation() {
#if defined(__linux__)
    auto& i = *impl_;
    if (!i.commissioning || !i.observed_bound || i.failed || i.closing || i.commissioning_operation) throw std::logic_error("commissioning requires one isolated current serialized dispatch");
    i.identity(); i.gpu->require_current(); i.identity(); i.commissioning_operation = true;
#else
    throw std::logic_error("Linux direct GPU commissioning unsupported");
#endif
}
void LinuxDirectTransportDriverIo::end_commissioning_operation() noexcept { impl_->commissioning_operation = false; }
void LinuxDirectTransportDriverIo::enable_commissioning_duplex() {
    auto& i = *impl_;
    if (!i.commissioning || i.commissioning_operation || i.imported || i.armed || i.failed || i.closing || i.commissioning_duplex) throw std::logic_error("paired duplex requires one isolated pre-import commissioning owner");
    i.commissioning_duplex = true;
}
std::optional<DirectDriverBinding> LinuxDirectTransportDriverIo::current_binding() const {
#if defined(__linux__)
    try {
        auto& i = *impl_; if (i.commissioning || i.failed || i.closing || !i.ready) return std::nullopt;
        i.identity(); i.gpu->require_current();
        if (!i.qualification || !i.qualification->current(i.expected, i.gpu->exported_binding())) { i.failed = true; i.ready = false; return std::nullopt; }
        i.gpu->require_current(); i.identity();
        return i.expected;
    } catch (...) { impl_->failed = true; impl_->ready = false; }
#endif
    return std::nullopt;
}
void LinuxDirectTransportDriverIo::require_current() {
#if defined(__linux__)
    if (impl_->commissioning) throw std::logic_error("commissioning cannot supply production current/readiness authority");
    try { impl_->current(); } catch (...) { invalidate(); throw; }
#else
    throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
std::shared_ptr<DirectDriverImport> LinuxDirectTransportDriverIo::prepare_import(const DirectDriverBinding& expected, const DirectTransportLease& lease, const DirectTransportLinkPoolBinding& pool) {
#if defined(__linux__)
    auto& i = *impl_;
    const bool authorized = i.commissioning ? i.commissioning_operation && !i.failed && !i.closing : current_binding() == std::optional<DirectDriverBinding>(expected);
    if (!authorized || i.imported || expected != i.expected || lease.connection_epoch != expected.interface.connection_epoch ||
        lease.physical_link_id != expected.interface.physical_link_id || !lease.lease_generation || lease.link_index != 0) throw std::invalid_argument("direct GPU stream import is unavailable or foreign");
    const auto export_lease = i.gpu->import_lease(); const auto& exported = export_lease->binding(); export_lease->require_current();
    if (lease.pool_binding_identity != exported.pool_binding_identity || !same_view(pool.tx, exported.pool.tx) || !same_view(pool.rx, exported.pool.rx) ||
        pool.tx.bytes != i.observed.maximum_window_bytes || pool.rx.bytes != i.observed.maximum_window_bytes ||
        exported.actual_gpu_bytes > i.policy.maximum_imported_bytes || exported.actual_gpu_bytes > i.observed.maximum_imported_bytes) throw std::invalid_argument("direct GPU stream import window/full extents exceed exact QUERY/owner admission");
    auto token = std::make_shared<Import>(); token->lease = export_lease; token->value = {expected, lease, pool, 0}; i.imported = token; i.import_binding = token->value;
    sgu_import request{}; request.fence = i.fence; request.fence.connection_epoch = lease.connection_epoch; request.fence.pool_binding_identity = lease.pool_binding_identity;
    for (std::size_t index = 0; index < 2; ++index) {
        auto& buffer = index ? request.rx : request.tx; buffer.fd = exported.descriptors[index]; buffer.full_bytes = exported.allocations[index].bytes;
        buffer.window_offset = exported.view_offsets[index]; buffer.window_bytes = index ? pool.rx.bytes : pool.tx.bytes;
        buffer.allocation_generation = index ? pool.rx.allocation_generation : pool.tx.allocation_generation;
        buffer.fd_device = exported.descriptor_identities[index].device; buffer.fd_inode = exported.descriptor_identities[index].inode;
    }
    const auto before = request;
    try {
        i.call(SGU_IOC_IMPORT, &request);
        const auto generation = request.fence.import_generation; request.fence.import_generation = 0;
        if (!generation || !same_fence(request.fence, before.fence) || !same_buffer(request.tx, before.tx) || !same_buffer(request.rx, before.rx) || request.reserved[0] || request.reserved[1]) throw std::runtime_error("direct GPU stream IMPORT acknowledgement is incompatible");
        i.fence = before.fence; i.fence.import_generation = generation; token->value.import_generation = generation; i.import_binding = token->value;
        i.gpu->bind_import(token->value); i.current(); return token;
    } catch (...) {
        i.failed = true;
        // QUERY can recover a generation hidden by failed copyout only for this
        // exact exclusive FD, endpoint, epoch and unique pool identity.
        try {
            const auto observed = i.inspect_kernel(); auto candidate = observed.fence; candidate.import_generation = 0;
            if (same_fence(candidate, before.fence) && observed.fence.import_generation) { i.fence = observed.fence; token->value.import_generation = observed.fence.import_generation; i.import_binding = token->value; }
            else if (observed.state == SGU_STATE_OPEN && !observed.fence.import_generation && same_fence(observed.fence, i.observed.fence)) { i.kernel_released = true; i.controller_stopped = true; }
        } catch (...) {}
        throw;
    }
#else
    static_cast<void>(expected); static_cast<void>(lease); static_cast<void>(pool); throw std::logic_error("Linux experimental direct GPU stream is unsupported");
#endif
}
void LinuxDirectTransportDriverIo::activate_import(const std::shared_ptr<DirectDriverImport>& token) {
#if defined(__linux__)
    try {
        auto& i = *impl_; i.current();
        if (token != i.imported || !same_import(token->binding(), i.import_binding) || i.armed) throw std::invalid_argument("direct GPU stream activation token is stale");
        i.action(SGU_IOC_ARM, nullptr, SGU_RESULT_NONE, 0); i.armed = true; i.current();
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(token); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::prepare(const DirectDriverOperation& operation) {
#if defined(__linux__)
    auto& i = *impl_; i.current();
    const bool occupied = i.commissioning && i.commissioning_duplex ? i.operations[operation.receive_path ? 1U : 0U].has_value() : i.operations[0].has_value() || i.operations[1].has_value();
    if (!i.armed || occupied || !same_import(operation.imported, i.import_binding) || !operation.handle ||
        !validate_descriptor(operation.descriptor, {1, i.fence.connection_epoch, i.cached.backend.maximum_payload_bytes}) || !operation.descriptor.sequence || !operation.descriptor.generation) throw std::invalid_argument("direct GPU stream requires one exact bounded C1 operation");
    const auto& pool = operation.receive_path ? operation.imported.pool.rx : operation.imported.pool.tx;
    if (operation.buffer.address < pool.address || operation.buffer.address - pool.address > pool.bytes - SGU_FRAME_BYTES ||
        (operation.buffer.address - pool.address) % SGU_FRAME_BYTES || operation.buffer.allocation_generation != pool.allocation_generation ||
        operation.buffer.domain != pool.domain || operation.buffer.access != TransportBufferAccess::read_write ||
        operation.buffer.bytes < operation.descriptor.byte_length || operation.buffer.bytes > pool.bytes - (operation.buffer.address - pool.address) ||
        operation.descriptor.byte_length > pool.bytes - SGU_FRAME_BYTES - (operation.buffer.address - pool.address)) throw std::invalid_argument("direct GPU stream payload window is foreign or overlaps terminal frame");
    auto& last = operation.receive_path ? i.last_receive : i.last_send;
    if (operation.descriptor.sequence <= last) throw std::invalid_argument("direct GPU stream physical sequence is replayed");
    last = operation.descriptor.sequence;
    auto& entry = i.operations[operation.receive_path ? 1U : 0U]; entry.emplace(); entry->binding = operation;
    entry->offset = operation.buffer.address - pool.address;
    entry->frame_count = static_cast<std::uint32_t>((operation.descriptor.byte_length + SGU_FRAME_BYTES - 1) / SGU_FRAME_BYTES + 1);
    sgu_prepare request{}; request.fence = i.fence; request.handle = operation.handle; request.sequence = operation.descriptor.sequence;
    request.pool_offset = entry->offset; request.payload_bytes = operation.descriptor.byte_length; request.terminal_bytes = sizeof(GpuCommitRecord); request.direction = operation.receive_path ? SGU_DIRECTION_RX : SGU_DIRECTION_TX;
    const auto before = request;
    try {
        i.gpu->prepare(operation); i.call(SGU_IOC_PREPARE, &request); i.current();
        if (!same_fence(request.fence, before.fence) || request.handle != before.handle || request.sequence != before.sequence || request.pool_offset != before.pool_offset ||
            request.payload_bytes != before.payload_bytes || request.terminal_bytes != before.terminal_bytes || request.direction != before.direction || request.reserved) throw std::runtime_error("direct GPU stream PREPARE request changed at driver boundary");
    }
    catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectDriverVisibilityProof LinuxDirectTransportDriverIo::publish_gpu_commit(const DirectDriverOperation& operation, const GpuCommitRecord& record) {
#if defined(__linux__)
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (operation.receive_path || entry.released || entry.submitted || !validate_gpu_commit_record(record, operation.descriptor)) throw std::invalid_argument("direct GPU stream producer record is stale or foreign");
        const auto proof = i.gpu->publish_terminal(operation, i.terminal(false), record); i.current();
        if (proof.visibility != DirectDriverVisibility::producer_release_to_external_dma || !same_operation(proof.operation, operation)) throw std::runtime_error("direct GPU stream producer proof is foreign");
        entry.commit = record; entry.released = true; i.send_control(entry, DistributedDirectGpuControlKind::producer_released, record); return proof;
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); static_cast<void>(record); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::submit(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (entry.submitted || (!operation.receive_path && !entry.released)) throw std::logic_error("direct GPU stream submission lacks one producer release");
        if (operation.receive_path) entry.commit = i.receive_control(entry, DistributedDirectGpuControlKind::producer_released);
        else static_cast<void>(i.receive_control(entry, DistributedDirectGpuControlKind::receive_ready));
        entry.submitted = true; i.action(SGU_IOC_SUBMIT, &entry, SGU_RESULT_NONE, entry.frame_count); i.current();
        if (operation.receive_path) i.send_control(entry, DistributedDirectGpuControlKind::receive_ready);
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::cancel(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try { auto& entry = impl_->require(operation); impl_->action(SGU_IOC_CANCEL, &entry, SGU_RESULT_CANCEL_REQUESTED, 0); invalidate(); impl_->gpu->request_stop(); }
    catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectDriverCommitRead LinuxDirectTransportDriverIo::read_gpu_commit(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (!operation.receive_path || !entry.hardware_done || entry.acquired) throw std::logic_error("direct GPU acquire requires every exact controller frame completed");
        const auto result = i.gpu->read_terminal(operation, i.terminal(true)); i.current();
        if (result.acquire.visibility != DirectDriverVisibility::consumer_acquire_from_external_dma || !same_operation(result.acquire.operation, operation) ||
            !validate_gpu_commit_record(result.record, operation.descriptor) || result.record.protocol_version != entry.commit.protocol_version ||
            result.record.commit_word != entry.commit.commit_word || result.record.connection_epoch != entry.commit.connection_epoch || result.record.sequence != entry.commit.sequence ||
            result.record.payload_bytes != entry.commit.payload_bytes || result.record.stripe_index != entry.commit.stripe_index || result.record.stripe_count != entry.commit.stripe_count) throw std::runtime_error("direct GPU terminal/acquire proof mismatches exact peer producer release");
        entry.acquired = true; i.current(); i.send_control(entry, DistributedDirectGpuControlKind::received); return result;
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectDriverVisibilityProof LinuxDirectTransportDriverIo::consumer_completion(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try {
        auto& entry = impl_->require(operation);
        if (!operation.receive_path || !entry.acquired) throw std::logic_error("direct GPU consumer proof lacks acquired operation");
        const auto proof = impl_->gpu->consumer_completion(operation); impl_->current();
        if (proof.visibility != DirectDriverVisibility::consumer_release_complete || !same_operation(proof.operation, operation)) throw std::runtime_error("direct GPU consumer proof is incomplete or foreign");
        entry.consumer_done = true; return proof;
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::publish_consumer_credit(const DirectDriverOperation& operation, std::uint64_t bytes) {
#if defined(__linux__)
    try {
        auto& entry = impl_->require(operation);
        if (!operation.receive_path || !entry.consumer_done || entry.credited || bytes != operation.descriptor.byte_length) throw std::logic_error("direct GPU credit requires one exact completed consumer");
        entry.credited = true; impl_->send_control(entry, DistributedDirectGpuControlKind::consumed);
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); static_cast<void>(bytes); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::repost_receive(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try {
        auto& entry = impl_->require(operation);
        if (!operation.receive_path || !entry.credited || entry.reposted) throw std::logic_error("direct GPU repost lacks consumer credit");
        entry.reposted = true; impl_->action(SGU_IOC_RX_REPOST, &entry, SGU_RESULT_RX_REPOSTED, entry.frame_count); impl_->current(); impl_->enqueue(entry, CompletionKind::receive_reposted);
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
std::optional<DirectDriverEvent> LinuxDirectTransportDriverIo::poll_event() {
#if defined(__linux__)
    try {
        auto& i = *impl_; i.current();
        if (!i.events.empty()) { auto event = i.events.front(); i.events.pop_front(); i.current(); return event; }
        // The isolated paired commissioning runner processes RX acquire,
        // consumer and repost before allowing a blocking TX ACK read. Otherwise
        // TX-first completion at both peers would prevent RX/controller progress.
        const bool receive_needs_progress = i.commissioning && i.commissioning_duplex && i.operations[1] && !i.operations[1]->reposted;
        if (i.operations[0] && i.operations[0]->hardware_done && !receive_needs_progress) {
            auto& entry = *i.operations[0];
            if (!entry.transport_ack) { static_cast<void>(i.receive_control(entry, DistributedDirectGpuControlKind::received)); entry.transport_ack = true; i.enqueue(entry, CompletionKind::remote_transport_ack); }
            else if (!entry.remote_consumer) { static_cast<void>(i.receive_control(entry, DistributedDirectGpuControlKind::consumed)); entry.remote_consumer = true; i.enqueue(entry, CompletionKind::remote_consumer_complete); }
            if (!i.events.empty()) { auto event = i.events.front(); i.events.pop_front(); i.current(); return event; }
        }
        for (std::uint32_t call = 0; call < i.policy.maximum_event_calls; ++call) {
            sgu_event event{}; event.fence = i.fence; const auto result = i.raw_call(SGU_IOC_EVENT, &event); i.current();
            if (result.result < 0 && (result.error == EAGAIN || result.error == EINTR)) return std::nullopt;
            if (result.result || result.error) throw KernelError(result.error ? result.error : EIO);
            if (event.direction > SGU_DIRECTION_RX || !same_fence(event.fence, i.fence)) throw std::runtime_error("direct GPU controller event has foreign direction/epoch/import");
            auto& entry = i.operations[event.direction];
            if (!entry || !entry->submitted || entry->hardware_done || event.handle != entry->binding.handle || event.sequence != entry->binding.descriptor.sequence ||
                event.payload_bytes != entry->binding.descriptor.byte_length || event.status != SGU_EVENT_COMPLETE || event.first_frame != entry->completed_frames ||
                !event.count || event.count > SGU_EVENT_BATCH || event.total_frames != entry->frame_count || event.count > event.total_frames - event.first_frame || event.reserved) throw std::runtime_error("direct GPU controller event count/descriptor/status is invalid");
            for (std::uint32_t index = 0; index < SGU_EVENT_BATCH; ++index) {
                const auto& frame = event.frames[index];
                if (index >= event.count) {
                    if (frame.pool_offset || frame.capacity || frame.expected_bytes || frame.actual_bytes || frame.error_flags || frame.physical_slot || frame.reserved) throw std::runtime_error("direct GPU controller event unused fields are nonzero");
                    continue;
                }
                const auto logical_frame = event.first_frame + index; const auto terminal = logical_frame + 1 == event.total_frames;
                const auto expected_offset = terminal ? i.observed.maximum_window_bytes - SGU_FRAME_BYTES : entry->offset + static_cast<std::uint64_t>(logical_frame) * SGU_FRAME_BYTES;
                const auto expected_bytes = terminal ? static_cast<std::uint32_t>(sizeof(GpuCommitRecord)) : static_cast<std::uint32_t>(std::min<std::uint64_t>(SGU_FRAME_BYTES, entry->binding.descriptor.byte_length - static_cast<std::uint64_t>(logical_frame) * SGU_FRAME_BYTES));
                if (frame.pool_offset != expected_offset || frame.physical_slot != expected_offset / SGU_FRAME_BYTES || frame.capacity != SGU_FRAME_BYTES ||
                    frame.expected_bytes != expected_bytes || frame.actual_bytes != expected_bytes || frame.error_flags || frame.reserved) throw std::runtime_error("direct GPU controller frame offset/size/error mismatch");
            }
            entry->completed_frames += event.count;
            if (entry->completed_frames == entry->frame_count) {
                entry->hardware_done = true; i.enqueue(*entry, entry->binding.receive_path ? CompletionKind::receive_complete : CompletionKind::local_buffer_reusable);
                auto completed = i.events.front(); i.events.pop_front(); completed.event.monotonic_seconds = static_cast<double>(event.monotonic_ns) / 1.0e9;
                i.current(); return completed;
            }
        }
        return std::nullopt;
    } catch (...) { invalidate(); throw; }
#else
    throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::retire(const DirectDriverOperation& operation) {
#if defined(__linux__)
    try {
        auto& i = *impl_; auto& entry = i.require(operation);
        if (!entry.hardware_done || (operation.receive_path ? !entry.reposted : !entry.transport_ack || !entry.remote_consumer)) throw std::logic_error("direct GPU stream cannot retire incomplete ownership");
        i.action(SGU_IOC_RETIRE, &entry, SGU_RESULT_NONE, 0); i.current(); i.gpu->retire(operation); i.current();
        i.operations[operation.receive_path ? 1U : 0U].reset();
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(operation); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
TransportBufferView LinuxDirectTransportDriverIo::acquired_receive_view(const TransferDescriptor& descriptor) {
#if defined(__linux__)
    auto& entry = impl_->receive(descriptor);
    if (!entry.acquired) throw std::logic_error("direct GPU receive has no completed acquire");
    const auto view = impl_->gpu->acquired_receive_view(entry.binding); impl_->current(); return view;
#else
    static_cast<void>(descriptor); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::consume_received_segments(const TransferDescriptor& descriptor, std::span<const TransportBufferView> destinations) {
#if defined(__linux__)
    try {
        auto& entry = impl_->receive(descriptor);
        if (!entry.acquired || entry.consumer_done) throw std::logic_error("direct GPU receive unpack is unacquired or already completed");
        impl_->gpu->consume_received_segments(entry.binding, destinations); impl_->current();
    } catch (...) { invalidate(); throw; }
#else
    static_cast<void>(descriptor); static_cast<void>(destinations); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectDriverVisibilityProof LinuxDirectTransportDriverIo::complete_receive_consumer(const TransferDescriptor& descriptor) {
#if defined(__linux__)
    const auto binding = impl_->receive(descriptor).binding; return consumer_completion(binding);
#else
    static_cast<void>(descriptor); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
void LinuxDirectTransportDriverIo::request_import_release(const std::shared_ptr<DirectDriverImport>& token) {
#if defined(__linux__)
    auto& i = *impl_;
    if (!i.imported || token != i.imported || !same_import(token->binding(), i.import_binding)) throw std::invalid_argument("direct GPU cleanup import token changed");
    i.closing = true; i.ready = false; i.gpu->request_stop();
    if (!i.import_binding.import_generation && !i.kernel_released) {
        const auto observed = i.inspect_kernel(); auto initial = i.observed.fence;
        initial.connection_epoch = i.import_binding.lease.connection_epoch; initial.pool_binding_identity = i.import_binding.lease.pool_binding_identity;
        auto candidate = observed.fence; candidate.import_generation = 0;
        if (observed.fence.import_generation && same_fence(candidate, initial)) { i.fence = observed.fence; i.import_binding.import_generation = observed.fence.import_generation; i.imported->value = i.import_binding; }
        else if (observed.state == SGU_STATE_OPEN && same_fence(observed.fence, i.observed.fence)) { i.kernel_released = true; i.controller_stopped = true; }
        else throw std::runtime_error("direct GPU partial import ownership remains unknown");
    }
    if (!i.controller_stopped) { i.action(SGU_IOC_STOP, nullptr, SGU_RESULT_CONTROLLER_STOPPED, 0); i.controller_stopped = true; }
#else
    static_cast<void>(token); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectDriverQuiescence LinuxDirectTransportDriverIo::poll_import_quiescence(const std::shared_ptr<DirectDriverImport>& token) {
    DirectDriverQuiescence proof; proof.imported = impl_->import_binding;
#if defined(__linux__)
    try {
        auto& i = *impl_; i.identity();
        if (!i.imported || token != i.imported || !same_import(token->binding(), i.import_binding)) throw std::runtime_error("direct GPU quiescence token changed");
        proof.status = i.controller_stopped ? i.gpu->poll_quiescence() : DirectDriverQuiescenceStatus::pending;
        i.identity(); if (!same_import(token->binding(), i.import_binding)) throw std::runtime_error("direct GPU quiescence token changed during GPU query");
        return proof;
    } catch (...) { impl_->failed = true; proof.status = DirectDriverQuiescenceStatus::failed_retained; }
#else
    static_cast<void>(token); proof.status = DirectDriverQuiescenceStatus::failed_retained;
#endif
    return proof;
}
void LinuxDirectTransportDriverIo::release_import(const std::shared_ptr<DirectDriverImport>& token) {
#if defined(__linux__)
    auto& i = *impl_;
    if (poll_import_quiescence(token).status != DirectDriverQuiescenceStatus::quiescent) throw std::logic_error("direct GPU import lacks controller AND GPU quiescence");
    if (!i.kernel_released) { i.action(SGU_IOC_RELEASE, nullptr, SGU_RESULT_NONE, 0); i.kernel_released = true; }
    if (i.gpu->close() != DirectPoolReleaseStatus::released) throw std::runtime_error("direct GPU event/stream/metadata cleanup remains unproved");
    i.identity(); if (!same_import(token->binding(), i.import_binding)) throw std::runtime_error("direct GPU release token changed during cleanup");
    i.imported.reset(); i.operations = {}; i.events.clear(); i.failed = true; i.ready = false;
#else
    static_cast<void>(token); throw std::logic_error("Linux direct GPU stream unsupported");
#endif
}
DirectPoolReleaseStatus LinuxDirectTransportDriverIo::close() noexcept {
    auto& i = *impl_;
    try {
#if defined(__linux__)
        if (i.imported) {
            request_import_release(i.imported);
            if (poll_import_quiescence(i.imported).status != DirectDriverQuiescenceStatus::quiescent) return DirectPoolReleaseStatus::failed_retained;
            const auto token = i.imported; release_import(token);
        } else if (i.gpu && i.gpu->close() != DirectPoolReleaseStatus::released) return DirectPoolReleaseStatus::failed_retained;
        if (i.fd.descriptor >= 0) { i.identity(); i.calls.fd_owner->close(i.fd); i.fd = {}; }
#else
        if (i.gpu && i.gpu->close() != DirectPoolReleaseStatus::released) return DirectPoolReleaseStatus::failed_retained;
#endif
        return DirectPoolReleaseStatus::released;
    } catch (...) { i.failed = true; return DirectPoolReleaseStatus::failed_retained; }
}
} // namespace strix::transport
