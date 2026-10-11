#include "strix/hip_host_upload_backend.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace strix {
namespace {

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

class PinnedSource final : public HostUploadSource {
public:
    PinnedSource(std::uint64_t bytes, std::uint64_t generation)
        : bytes_(static_cast<std::size_t>(bytes)), generation_(generation) {
        void* pointer = nullptr;
        check_hip(hipHostMalloc(&pointer, bytes_, hipHostMallocDefault), "hipHostMalloc(host upload)");
        pointer_ = static_cast<std::byte*>(pointer);
    }
    ~PinnedSource() override { static_cast<void>(hipHostFree(pointer_)); }
    std::span<std::byte> writable_bytes() noexcept override { return {pointer_, bytes_}; }
    std::span<const std::byte> bytes() const noexcept override { return {pointer_, bytes_}; }
    std::uint64_t allocation_generation() const noexcept override { return generation_; }

private:
    std::byte* pointer_ = nullptr;
    std::size_t bytes_;
    std::uint64_t generation_;
};

}  // namespace

struct HipHostUploadBackend::Impl {
    struct Transfer {
        std::shared_ptr<const HostUploadSource> source;
        ExecutionBufferAddress destination;
        std::uint64_t bytes = 0;
        int device = -1;
        hipEvent_t event = nullptr;
        bool submitted = false;
        bool event_recorded = false;
        bool cancel_requested = false;
        bool source_released = false;
        bool submission_failed = false;
    };
    std::uint32_t maximum_operations = 0;
    std::uint64_t teardown_polls = 0;
    HostUploadHandle next_handle = 1;
    std::unordered_map<HostUploadHandle, std::unique_ptr<Transfer>> operations;

    Transfer& require(HostUploadHandle handle) {
        const auto found = operations.find(handle);
        if (found == operations.end()) throw std::invalid_argument("HIP host upload handle is unknown");
        return *found->second;
    }
};

HipHostUploadBackend::HipHostUploadBackend(std::uint32_t maximum_operations,
                                         std::uint64_t teardown_polls)
    : impl_(std::make_unique<Impl>()) {
    if (maximum_operations == 0 || teardown_polls == 0) {
        throw std::invalid_argument("HIP host upload backend requires bounded operations/progress");
    }
    impl_->maximum_operations = maximum_operations;
    impl_->teardown_polls = teardown_polls;
}

HipHostUploadBackend::~HipHostUploadBackend() {
    while (!impl_->operations.empty()) {
        const auto handle = impl_->operations.begin()->first;
        auto& operation = *impl_->operations.begin()->second;
        operation.cancel_requested = true;
        for (std::uint64_t count = 0; count < impl_->teardown_polls && !operation.source_released; ++count) {
            static_cast<void>(poll(handle));
        }
        if (operation.source_released) {
            retire(handle);
        } else {
            // No HIP cancel API proves a queued copy stopped reading its host
            // source. A failed/hung context must retain the owning transfer,
            // pinned buffer and event until process teardown; do not free them.
            // At most maximum_operations bounded sources are quarantined by
            // this backend. A failed runtime must not recreate upload pools.
            static_cast<void>(impl_->operations.begin()->second.release());
            impl_->operations.erase(handle);
        }
    }
}

std::shared_ptr<HostUploadSource> HipHostUploadBackend::allocate_source(
    std::uint64_t bytes, std::uint64_t allocation_generation) {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() || allocation_generation == 0) {
        throw std::invalid_argument("HIP host upload source size/generation is invalid");
    }
    return std::make_shared<PinnedSource>(bytes, allocation_generation);
}

HostUploadHandle HipHostUploadBackend::prepare(
    const ExecutionBufferAddress& destination,
    std::shared_ptr<const HostUploadSource> source,
    std::uint64_t bytes) {
    if (!source || source->allocation_generation() == 0 || bytes == 0 ||
        bytes > source->bytes().size() || source->bytes().data() == nullptr || destination.address == 0 ||
        destination.allocation_generation == 0 || destination.use.generation == 0 || bytes > destination.bytes ||
        destination.bytes > std::numeric_limits<std::uintptr_t>::max() - destination.address) {
        throw std::invalid_argument("HIP host upload requires exact bounded source/destination bindings");
    }
    if (impl_->operations.size() >= impl_->maximum_operations ||
        impl_->next_handle == std::numeric_limits<HostUploadHandle>::max()) {
        throw std::runtime_error("HIP host upload operation budget exhausted");
    }
    auto operation = std::make_unique<Impl::Transfer>();
    check_hip(hipGetDevice(&operation->device), "hipGetDevice(host upload prepare)");
    operation->source = std::move(source);
    operation->destination = destination;
    operation->bytes = bytes;
    const auto handle = impl_->next_handle++;
    auto [entry, inserted] = impl_->operations.emplace(handle, std::move(operation));
    if (!inserted) throw std::logic_error("HIP host upload duplicate handle");
    try {
        check_hip(hipEventCreateWithFlags(&entry->second->event, hipEventDisableTiming),
                  "hipEventCreate(host upload)");
    } catch (...) {
        impl_->operations.erase(handle);
        throw;
    }
    return handle;
}

void HipHostUploadBackend::submit(HostUploadHandle handle) {
    auto& operation = impl_->require(handle);
    if (operation.submitted || operation.cancel_requested) {
        throw std::logic_error("HIP host upload cannot be resubmitted or submitted after cancellation");
    }
    int device = -1;
    check_hip(hipGetDevice(&device), "hipGetDevice(host upload submit)");
    if (device != operation.device) {
        throw std::invalid_argument("HIP host upload device changed after preparation");
    }
    // Set before the runtime call: an exception cannot erase the fact that a
    // copy may now reference the source, even if event recording later fails.
    operation.submitted = true;
    const auto copy = hipMemcpyAsync(
        reinterpret_cast<void*>(operation.destination.address), operation.source->bytes().data(),
        static_cast<std::size_t>(operation.bytes), hipMemcpyHostToDevice, hipStreamLegacy);
    if (copy != hipSuccess) {
        operation.submission_failed = true;
        check_hip(copy, "hipMemcpyAsync(host upload)");
    }
    const auto recorded = hipEventRecord(operation.event, hipStreamLegacy);
    if (recorded != hipSuccess) {
        operation.submission_failed = true;
        check_hip(recorded, "hipEventRecord(host upload)");
    }
    operation.event_recorded = true;
}

HostUploadStatus HipHostUploadBackend::poll(HostUploadHandle handle) {
    auto& operation = impl_->require(handle);
    if (!operation.submitted) {
        if (!operation.cancel_requested) return HostUploadStatus::pending;
        operation.source_released = true;
        return HostUploadStatus::canceled;
    }
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != operation.device) {
        return HostUploadStatus::failed_retained;
    }
    // An unrecorded event would look complete without covering the copy.
    // Stream quiescence is the fallback proof after a submit/record error.
    const auto status = operation.event_recorded
        ? hipEventQuery(operation.event) : hipStreamQuery(hipStreamLegacy);
    if (status == hipErrorNotReady) return HostUploadStatus::pending;
    if (status != hipSuccess) return HostUploadStatus::failed_retained;
    operation.source_released = true;
    return operation.cancel_requested || operation.submission_failed
        ? HostUploadStatus::canceled : HostUploadStatus::complete;
}

void HipHostUploadBackend::cancel(HostUploadHandle handle) {
    impl_->require(handle).cancel_requested = true;
}

void HipHostUploadBackend::retire(HostUploadHandle handle) {
    auto& operation = impl_->require(handle);
    if (!operation.source_released) throw std::logic_error("HIP host upload retirement lacks a source-release proof");
    static_cast<void>(hipEventDestroy(operation.event));
    impl_->operations.erase(handle);
}

}  // namespace strix
