#pragma once

#include "visibility_kernels.hpp"
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace strix::transport::visibility {
// Owns only a 64-byte GPU summary, a pinned 64-byte readback and one event.
// Payload is never downloaded. The complete owner is retained by the native
// helper, and unknown GPU completion deliberately retains these allocations.
class NativeProbe final {
public:
    NativeProbe(std::function<void()> require_device, std::uint64_t generation)
        : require_device_(std::move(require_device)), generation_(generation) {
        if (!require_device_ || !generation_) throw std::invalid_argument("invalid visibility owner");
        try {
            require_device_();
            check(hipMalloc(reinterpret_cast<void**>(&device_), sizeof(Result)), "allocate GPU summary");
            require_device_();
            check(hipHostMalloc(reinterpret_cast<void**>(&host_), sizeof(Result), hipHostMallocDefault), "allocate summary readback");
            require_device_();
            check(hipEventCreateWithFlags(&event_, hipEventDisableTiming | hipEventReleaseToSystem), "create system-scope summary event");
            require_device_();
            if (!device_ || !host_ || !event_) throw std::runtime_error("missing visibility resources");
        } catch (...) { static_cast<void>(release()); throw; }
    }
    ~NativeProbe() { static_cast<void>(release()); }
    NativeProbe(const NativeProbe&) = delete;
    NativeProbe& operator=(const NativeProbe&) = delete;
    [[nodiscard]] TransportBufferView view() const noexcept {
        return {reinterpret_cast<std::uintptr_t>(device_), sizeof(Result), generation_,
                TransportMemoryDomain::gpu_device, TransportBufferAccess::read_write};
    }
    Observation verify(hipStream_t stream, const TransportBufferView& source,
        std::uint64_t seed, std::uint64_t epoch, std::uint64_t sequence, Stage stage,
        std::chrono::milliseconds timeout, std::chrono::milliseconds pause, std::uint64_t polls) {
        if (!idle_ || !stream || !device_ || !host_ || !event_ || release_unknown_ ||
            !source.address || !source.bytes || !source.allocation_generation ||
            source.domain != TransportMemoryDomain::gpu_device ||
            (source.access != TransportBufferAccess::read_only && source.access != TransportBufferAccess::read_write) ||
            source.bytes > std::numeric_limits<std::uintptr_t>::max() - source.address ||
            source.bytes > 16ULL * 1024ULL * 1024ULL || !epoch || nonce_ == UINT64_MAX ||
            !polls || polls > 1000000 || timeout.count() <= 0 || timeout > std::chrono::minutes{1} ||
            pause.count() <= 0 || pause > std::chrono::seconds{1} || pause > timeout)
            throw std::invalid_argument("invalid or still-owned visibility request");
        require_device_();
        auto last = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeout);
        if (last > std::chrono::steady_clock::time_point::max() - duration)
            throw std::overflow_error("visibility deadline overflow");
        const auto deadline = last + duration;
        const auto check_deadline = [&] {
            const auto now = std::chrono::steady_clock::now();
            if (now < last || now >= deadline) throw std::runtime_error("GPU visibility deadline expired");
            last = now;
        };
        const Expected expected{++nonce_, epoch, sequence, source.bytes, stage};
        idle_ = false; // Set before the first launch, including ambiguous failures.
        hipLaunchKernelGGL(initialize_result, dim3(1), dim3(1), 0, stream,
            device_, expected.nonce, epoch, sequence, static_cast<std::uint64_t>(stage));
        check(hipGetLastError(), "initialize GPU visibility summary"); check_deadline();
        const auto blocks = static_cast<unsigned>((source.bytes + 255) / 256);
        hipLaunchKernelGGL(verify_bytes, dim3(blocks < 256 ? blocks : 256), dim3(256), 0, stream,
            reinterpret_cast<const std::uint8_t*>(source.address), source.bytes, seed, sequence, device_);
        check(hipGetLastError(), "verify every payload byte on GPU"); check_deadline();
        hipLaunchKernelGGL(finish_result, dim3(1), dim3(1), 0, stream, device_, expected.nonce);
        check(hipGetLastError(), "finish GPU visibility summary"); check_deadline();
        check(hipMemcpyAsync(host_, device_, sizeof(Result), hipMemcpyDeviceToHost, stream), "read 64-byte GPU summary");
        check(hipEventRecord(event_, stream), "record system-scope GPU summary event");
        for (std::uint64_t attempt = 0; attempt < polls; ++attempt) {
            require_device_(); check_deadline();
            const auto status = hipEventQuery(event_);
            require_device_(); check_deadline();
            if (status == hipSuccess) {
                idle_ = true;
                return {true, expected, *host_};
            }
            if (status != hipErrorNotReady) check(status, "query GPU visibility event");
            if (attempt + 1 < polls) std::this_thread::sleep_for(pause);
        }
        throw std::runtime_error("GPU visibility poll budget exhausted; resources retained");
    }
    // Only after the native helper has independently proved whole-stream idle.
    void mark_stream_quiescent() noexcept { idle_ = true; }
    [[nodiscard]] bool release() noexcept {
        if (!idle_ || release_unknown_) return false;
        try {
            require_device_();
            if (event_) { release_unknown_ = true; check(hipEventDestroy(event_), "release summary event"); event_ = nullptr; release_unknown_ = false; }
            require_device_();
            if (host_) { release_unknown_ = true; check(hipHostFree(host_), "release summary readback"); host_ = nullptr; release_unknown_ = false; }
            require_device_();
            if (device_) { release_unknown_ = true; check(hipFree(device_), "release GPU summary"); device_ = nullptr; release_unknown_ = false; }
            return true;
        } catch (...) { return false; }
    }
private:
    static void check(hipError_t status, const char* operation) {
        if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
    std::function<void()> require_device_;
    std::uint64_t generation_ = 0, nonce_ = 0;
    Result* device_ = nullptr;
    Result* host_ = nullptr;
    hipEvent_t event_ = nullptr;
    bool idle_ = true, release_unknown_ = false;
};
} // namespace strix::transport::visibility
