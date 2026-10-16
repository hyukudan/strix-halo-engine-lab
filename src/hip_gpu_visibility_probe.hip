#include "strix/hip_gpu_visibility_probe.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace strix::transport {
namespace {
void checked(hipError_t error, const char* what) {
    if (error != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}
__global__ void fill_visibility_bytes(std::uint8_t* data, std::uint64_t bytes,
    std::uint64_t offset, GpuVisibilityPattern pattern) {
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
        data[i] = gpu_visibility_byte(offset + i, pattern);
}
__global__ void initialize_visibility_result(GpuVisibilityResult* out,
    GpuVisibilityPattern pattern, std::uint64_t nonce) {
    if (threadIdx.x || blockIdx.x) return;
    *out = {};
    out->epoch = pattern.epoch; out->sequence = pattern.sequence; out->nonce = nonce;
    out->first_mismatch = ~std::uint64_t{0};
}
__global__ void check_visibility_bytes(const std::uint8_t* data, std::uint64_t bytes,
    std::uint64_t offset, GpuVisibilityPattern pattern, GpuVisibilityResult* out) {
    unsigned long long count = 0, mismatch = 0, first = ~0ULL, digest = 0, consumed = 0;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        // Ordinary GPU loads, not a CPU copy and not volatile cache bypasses.
        const auto value = data[i]; const auto position = offset + i;
        ++count;
        if (value != gpu_visibility_byte(position, pattern)) {
            ++mismatch; if (position < first) first = position;
        }
        digest ^= gpu_visibility_mix(position ^ value);
        consumed += gpu_visibility_consume(position, value);
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->checked_bytes), count);
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->mismatches), mismatch);
    atomicMin(reinterpret_cast<unsigned long long*>(&out->first_mismatch), first);
    atomicXor(reinterpret_cast<unsigned long long*>(&out->observed_digest), digest);
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->consumed_digest), consumed);
}
__global__ void flip_visibility_byte(std::uint8_t* address) {
    if (!blockIdx.x && !threadIdx.x) *address ^= 1;
}
std::atomic<std::uint64_t> next_probe_nonce{1};
std::uint64_t allocate_nonce() {
    auto value = next_probe_nonce.load();
    for (;;) {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("GPU visibility nonce exhausted");
        if (next_probe_nonce.compare_exchange_weak(value, value + 1)) return value;
    }
}
}
struct HipGpuVisibilityProbe::Impl {
    std::shared_ptr<void> owner;
    std::function<hipStream_t()> stream;
    std::function<void()> current;
    HipGpuVisibilityProbePolicy policy;
    GpuVisibilityResult* device = nullptr;
    GpuVisibilityResult* host = nullptr;
    hipEvent_t event = nullptr;
    hipStream_t submitted_stream = nullptr;
    int allocation_device = -1;
    bool queued = false, failed = false, closed = false, cleanup_unknown = false;
    std::uint64_t ranges(std::span<const GpuVisibilityRange> ranges, GpuVisibilityPattern p) {
        if (failed || closed || queued || !p.epoch || !p.sequence || ranges.empty() || ranges.size() > 64)
            throw std::invalid_argument("invalid or non-quiescent GPU visibility request");
        current();
        std::uint64_t total = 0;
        for (const auto& r : ranges) {
            if (!r.address || !r.bytes || r.bytes > policy.maximum_bytes - total ||
                r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
                throw std::invalid_argument("GPU visibility range is empty, overflowing or too large");
            hipPointerAttribute_t attributes{};
            checked(hipPointerGetAttributes(&attributes, reinterpret_cast<void*>(r.address)), "GPU visibility pointer");
            if (attributes.type != hipMemoryTypeDevice || attributes.isManaged)
                throw std::invalid_argument("visibility probe requires resident device memory");
            int device_index = -1; checked(hipGetDevice(&device_index), "GPU visibility device");
            if (attributes.device != device_index)
                throw std::invalid_argument("GPU visibility pointer belongs to another device");
            void* base = nullptr; std::size_t extent = 0;
            checked(hipMemGetAddressRange(&base, &extent, reinterpret_cast<void*>(r.address)), "GPU visibility extent");
            const auto begin = reinterpret_cast<std::uintptr_t>(base);
            if (r.address < begin || r.address - begin > extent || r.bytes > extent - (r.address - begin))
                throw std::invalid_argument("GPU visibility range exceeds its allocation");
            total += r.bytes;
        }
        for (std::size_t i = 0; i < ranges.size(); ++i)
            for (std::size_t j = i + 1; j < ranges.size(); ++j)
                if (ranges[i].address < ranges[j].address + ranges[j].bytes &&
                    ranges[j].address < ranges[i].address + ranges[i].bytes)
                    throw std::invalid_argument("GPU visibility ranges overlap");
        return total;
    }
    hipStream_t begin() {
        current(); auto result = stream();
        if (!result) throw std::logic_error("visibility probe requires the bound helper-owned stream");
        int active = -1; hipDevice_t device_index = -1;
        checked(hipGetDevice(&active), "GPU visibility active device");
        checked(hipStreamGetDevice(result, &device_index), "GPU visibility stream device");
        hipStreamCaptureStatus capture{};
        checked(hipStreamIsCapturing(result, &capture), "GPU visibility stream capture");
        if (active != allocation_device || active != device_index || capture != hipStreamCaptureStatusNone)
            throw std::logic_error("GPU visibility stream is foreign or captured");
        submitted_stream = result; queued = true; return result;
    }
    void finish(hipStream_t submitted) {
        checked(hipEventRecord(event, submitted), "GPU visibility completion record");
        const auto start = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(policy.timeout);
        if (start > std::chrono::steady_clock::time_point::max() - duration)
            throw std::overflow_error("GPU visibility deadline overflow");
        const auto deadline = start + duration;
        for (;;) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("GPU visibility deadline expired; storage retained");
            current();
            if (stream() != submitted) throw std::runtime_error("GPU visibility stream changed");
            const auto status = hipEventQuery(event);
            current();
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("GPU visibility deadline expired; storage retained");
            if (status == hipSuccess) { queued = false; return; }
            if (status != hipErrorNotReady) checked(status, "GPU visibility completion");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("GPU visibility deadline expired; storage retained");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
};
HipGpuVisibilityProbe::HipGpuVisibilityProbe(std::shared_ptr<void> owner,
    std::function<hipStream_t()> stream, std::function<void()> current, HipGpuVisibilityProbePolicy policy)
    : impl_(std::make_unique<Impl>()) {
    auto& i = *impl_; i.owner = std::move(owner); i.stream = std::move(stream);
    i.current = std::move(current); i.policy = policy;
    if (!i.owner || !i.stream || !i.current || !policy.maximum_bytes || policy.maximum_bytes > 128ULL*1024*1024 ||
        policy.timeout.count() <= 0 || policy.timeout > std::chrono::minutes{1})
        throw std::invalid_argument("GPU visibility probe requires an owner and bounded policy");
    try {
        i.current();
        checked(hipGetDevice(&i.allocation_device), "GPU visibility allocation device");
        checked(hipMalloc(reinterpret_cast<void**>(&i.device), sizeof(*i.device)), "GPU result allocation");
        checked(hipHostMalloc(reinterpret_cast<void**>(&i.host), sizeof(*i.host), hipHostMallocDefault), "GPU result host allocation");
        checked(hipEventCreateWithFlags(&i.event, hipEventDisableTiming), "GPU result system-fence event");
    } catch (...) {
        if (!close()) static_cast<void>(impl_.release());
        throw;
    }
}
HipGpuVisibilityProbe::~HipGpuVisibilityProbe() {
    if (impl_ && !close()) static_cast<void>(impl_.release());
}
void HipGpuVisibilityProbe::produce(std::span<const GpuVisibilityRange> ranges, GpuVisibilityPattern pattern) {
    auto& i = *impl_; i.ranges(ranges, pattern);
    try {
        auto stream = i.begin(); std::uint64_t offset = 0;
        for (const auto& r : ranges) {
            hipLaunchKernelGGL(fill_visibility_bytes, dim3(128), dim3(256), 0, stream,
                reinterpret_cast<std::uint8_t*>(r.address), r.bytes, offset, pattern);
            checked(hipGetLastError(), "GPU visibility producer launch"); offset += r.bytes;
        }
        i.finish(stream);
    } catch (...) { i.failed = true; throw; }
}
GpuVisibilityResult HipGpuVisibilityProbe::verify(std::span<const GpuVisibilityRange> ranges, GpuVisibilityPattern pattern) {
    auto& i = *impl_; const auto bytes = i.ranges(ranges, pattern); const auto nonce = allocate_nonce();
    try {
        auto stream = i.begin();
        hipLaunchKernelGGL(initialize_visibility_result, dim3(1), dim3(1), 0, stream, i.device, pattern, nonce);
        checked(hipGetLastError(), "GPU visibility result launch");
        std::uint64_t offset = 0;
        for (const auto& r : ranges) {
            hipLaunchKernelGGL(check_visibility_bytes, dim3(128), dim3(256), 0, stream,
                reinterpret_cast<const std::uint8_t*>(r.address), r.bytes, offset, pattern, i.device);
            checked(hipGetLastError(), "GPU visibility consumer launch"); offset += r.bytes;
        }
        checked(hipMemcpyAsync(i.host, i.device, sizeof(*i.host), hipMemcpyDeviceToHost, stream), "GPU result copy");
        i.finish(stream);
        if (i.host->epoch != pattern.epoch || i.host->sequence != pattern.sequence ||
            i.host->nonce != nonce || i.host->checked_bytes != bytes || i.host->mismatches > bytes ||
            (i.host->mismatches && i.host->first_mismatch >= bytes))
            throw std::runtime_error("GPU visibility result identity/coverage is invalid");
        return *i.host;
    } catch (...) { i.failed = true; throw; }
}
bool HipGpuVisibilityProbe::negative_control(std::span<const GpuVisibilityRange> ranges,
    GpuVisibilityPattern pattern, std::uint64_t offset) {
    auto& i = *impl_; const auto bytes = i.ranges(ranges, pattern);
    if (offset >= bytes) throw std::invalid_argument("negative-control offset outside payload");
    std::uint64_t inside = offset; std::uintptr_t address = 0;
    for (const auto& r : ranges) { if (inside < r.bytes) { address = r.address + inside; break; } inside -= r.bytes; }
    const auto flip = [&] {
        auto stream = i.begin();
        hipLaunchKernelGGL(flip_visibility_byte, dim3(1), dim3(1), 0, stream, reinterpret_cast<std::uint8_t*>(address));
        checked(hipGetLastError(), "GPU sensitivity launch"); i.finish(stream);
    };
    try {
        const auto before = verify(ranges, pattern);
        if (!valid_gpu_visibility_result(before, pattern, before.nonce, bytes)) return false;
        flip(); const auto corrupted = verify(ranges, pattern);
        flip(); const auto restored = verify(ranges, pattern);
        return corrupted.mismatches == 1 && corrupted.first_mismatch == offset &&
            valid_gpu_visibility_result(restored, pattern, restored.nonce, bytes);
    } catch (...) { i.failed = true; throw; }
}
bool HipGpuVisibilityProbe::close() noexcept {
    if (!impl_ || impl_->closed) return true;
    auto& i = *impl_;
    if (i.cleanup_unknown) return false;
    try {
        int active = -1;
        checked(hipGetDevice(&active), "GPU visibility cleanup device");
        if (i.allocation_device >= 0 && active != i.allocation_device) return false;
        if (i.queued) {
            // Even a failed event record may follow successfully queued work.
            // Query its original retained stream; a timeout is not a stop.
            i.current();
            if (!i.submitted_stream || hipStreamQuery(i.submitted_stream) != hipSuccess) return false;
            i.queued = false;
        }
        // An ambiguous native free is sticky. A later close must not discard
        // the retained owner or retry a possibly reused handle.
        if (i.event) {
            i.cleanup_unknown = true;
            checked(hipEventDestroy(i.event), "GPU visibility event destroy");
            i.event = nullptr; i.cleanup_unknown = false;
        }
        if (i.host) {
            i.cleanup_unknown = true;
            checked(hipHostFree(i.host), "GPU visibility host free");
            i.host = nullptr; i.cleanup_unknown = false;
        }
        if (i.device) {
            i.cleanup_unknown = true;
            checked(hipFree(i.device), "GPU visibility device free");
            i.device = nullptr; i.cleanup_unknown = false;
        }
        i.stream = {}; i.current = {}; i.owner.reset(); i.closed = true; return true;
    } catch (...) { i.failed = true; return false; }
}
} // namespace strix::transport
