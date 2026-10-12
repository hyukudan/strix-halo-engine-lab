#pragma once

#include "strix/direct_gpu_stream_gpu.hpp"
#include "strix/direct_gpu_boundary_staging.hpp"
#include "strix/direct_stream_control.hpp"
#include <functional>

namespace strix::transport {
class LinuxDirectGpuStreamCommissioning;
struct LinuxDirectGpuStreamSyscallResult { int result = -1; int error = 0; };
struct LinuxDirectGpuStreamCalls {
    std::shared_ptr<DirectStreamControlIo> fd_owner;
    std::function<LinuxDirectGpuStreamSyscallResult(DirectStreamFd, unsigned long, void*)> ioctl_call;
    std::function<std::string()> kernel_release;
    // Actual loaded module GNU build-id observation. Missing/unavailable is not
    // the driver name, ABI cookie or a positive qualification declaration.
    std::function<std::optional<std::string>()> driver_build_identity;
    [[nodiscard]] static LinuxDirectGpuStreamCalls native();
};
[[nodiscard]] std::string parse_linux_direct_gpu_stream_module_build_id(std::span<const std::byte>);
[[nodiscard]] std::optional<std::string> observe_linux_direct_gpu_stream_module_build_id();
// Trusted independent qualification declaration for exact observed binding,
// GPU UUID/runtime/driver/pool generations and the actual ownership/visibility
// path. No native factory supplies a positive declaration. A fixture must mark
// its synthetic evidence explicitly; implementation/query/config is not proof.
class DirectGpuStreamQualification {
public:
    virtual ~DirectGpuStreamQualification() = default;
    [[nodiscard]] virtual DirectNativeCapabilities proof(const DirectDriverBinding&, const DirectGpuExportBinding&) = 0;
    [[nodiscard]] virtual bool current(const DirectDriverBinding&, const DirectGpuExportBinding&) const = 0;
};
struct LinuxDirectGpuStreamPolicy {
    bool experimental_opt_in = false;
    std::uint64_t maximum_payload_bytes = 16ULL * 1024ULL * 1024ULL;
    std::uint64_t maximum_imported_bytes = 128ULL * 1024ULL * 1024ULL;
    std::uint32_t maximum_event_calls = 4096;
};

// Concrete experimental extension provider, never a stock-driver ioctl guess.
// Exact QUERY/FD/ABI/peer/link/epoch/import fencing, production global C1, contiguous
// payload frames (partial final frame permitted) + one dedicated terminal frame.
// Dedicated bounded v2 kind4 control stream supplies peer-ready/ACK/credit;
// caller supplies a trusted bounded byte stream (FD copy or portable network),
// retained for the entire provider lifetime; it must not block indefinitely.
// hardware frame completion and GPU completion remain separate. Stock ENOTTY,
// unsupported platforms and absent independent qualification stay unavailable.
// A private commissioning facade can exercise one TX+RX pair; it cannot promote
// query/current_binding authority or the production full_duplex capability.
// Single serialized owner. Import/export/GPU/FD/control lifetimes survive errors;
// no cancel/reset recovery, ring wrap or automatic retry of exposed identities.
class LinuxDirectTransportDriverIo final : public DirectTransportDriverIo, public DirectGpuBoundaryReceiveIo {
public:
    LinuxDirectTransportDriverIo(DirectStreamDeviceIdentity route, DirectDriverBinding expected,
        std::shared_ptr<DirectGpuStreamGpu>, std::shared_ptr<DistributedControlByteStream> dedicated_control,
        LinuxDirectGpuStreamPolicy, std::shared_ptr<DirectGpuStreamQualification> qualification = {},
        LinuxDirectGpuStreamCalls = LinuxDirectGpuStreamCalls::native());
    ~LinuxDirectTransportDriverIo() override;
    [[nodiscard]] DirectDriverQuery query() override;
    // Same-FD two-phase setup for an unpredictable kernel open-instance UUID.
    // Constructor accepts empty device_identity/generation discovery fields;
    // bind accepts only the exact observed OPEN binding once. It adopts no
    // positive readiness: a later query still needs independent qualification.
    void bind_observed(const DirectDriverBinding&, std::shared_ptr<DirectGpuStreamQualification>);
    [[nodiscard]] std::optional<DirectDriverBinding> current_binding() const override;
    void require_current() override;
    [[nodiscard]] std::shared_ptr<DirectDriverImport> prepare_import(const DirectDriverBinding&, const DirectTransportLease&, const DirectTransportLinkPoolBinding&) override;
    void activate_import(const std::shared_ptr<DirectDriverImport>&) override;
    void prepare(const DirectDriverOperation&) override;
    void submit(const DirectDriverOperation&) override;
    void cancel(const DirectDriverOperation&) override;
    [[nodiscard]] DirectDriverVisibilityProof publish_gpu_commit(const DirectDriverOperation&, const GpuCommitRecord&) override;
    [[nodiscard]] DirectDriverCommitRead read_gpu_commit(const DirectDriverOperation&) override;
    [[nodiscard]] DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) override;
    void publish_consumer_credit(const DirectDriverOperation&, std::uint64_t) override;
    void repost_receive(const DirectDriverOperation&) override;
    [[nodiscard]] std::optional<DirectDriverEvent> poll_event() override;
    void retire(const DirectDriverOperation&) override;
    void request_import_release(const std::shared_ptr<DirectDriverImport>&) override;
    [[nodiscard]] DirectDriverQuiescence poll_import_quiescence(const std::shared_ptr<DirectDriverImport>&) override;
    void release_import(const std::shared_ptr<DirectDriverImport>&) override;
    [[nodiscard]] TransportBufferView acquired_receive_view(const TransferDescriptor&) override;
    void consume_received_segments(const TransferDescriptor&, std::span<const TransportBufferView>) override;
    [[nodiscard]] DirectDriverVisibilityProof complete_receive_consumer(const TransferDescriptor&) override;
    [[nodiscard]] DirectPoolReleaseStatus close() noexcept;
    void invalidate() noexcept;
    [[nodiscard]] bool failed() const noexcept;
private:
    friend class LinuxDirectGpuStreamCommissioning;
    void bind_commissioning_observed(const DirectDriverBinding&);
    void enable_commissioning_duplex();
    void begin_commissioning_operation();
    void end_commissioning_operation() noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::transport
