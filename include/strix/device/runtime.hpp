#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#if defined(__HIPCC__)
#include "strix/hip_check.hpp"
#define STRIX_DEVICE_RT(name) hip##name
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define STRIX_DEVICE_RT(name) cuda##name
#endif

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include "strix/device_allocation_error.hpp"
#endif

#include "strix/device/declaration.hpp"
#include "strix/session/step_replay.hpp"

namespace strix::device {

struct Capabilities {
    int major = 0, minor = 0;  // compute capability; zero when unavailable
    bool pdl = false, clusters = false;
};

// Non-owning runtime binding of an already resolved descriptor. No enumeration or device selection.
// The descriptor must outlive this binding; a backend supplies the ordinal capability query.
struct Runtime {
    const Descriptor& descriptor;
    Vendor vendor;
    std::string_view arch;
    int ordinal;
    using Query = Capabilities (*)(int) noexcept;
    Query query;

    Runtime(const Descriptor& device, int ordinal_, Query query_) noexcept
        : descriptor(device), vendor(device.vendor), arch(device.arch), ordinal(ordinal_), query(query_) {}
    [[nodiscard]] Capabilities capabilities() const noexcept { return query ? query(ordinal) : Capabilities{}; }
};

}  // namespace strix::device


// Native buffer and stream ownership for the compiled accelerator backend.
#if defined(__HIPCC__) || defined(__CUDACC__)
namespace strix::device::native {

using Status = STRIX_DEVICE_RT(Error_t);
using Stream = STRIX_DEVICE_RT(Stream_t);
using CopyKind = STRIX_DEVICE_RT(MemcpyKind);
inline constexpr auto host_to_device = STRIX_DEVICE_RT(MemcpyHostToDevice);
inline constexpr auto device_to_host = STRIX_DEVICE_RT(MemcpyDeviceToHost);
inline constexpr auto device_to_device = STRIX_DEVICE_RT(MemcpyDeviceToDevice);

inline void check(Status status, const char* operation) {
#if defined(__HIPCC__)
    hip_check(status, operation);
#else
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
#endif
}

// The status of a DEVICE-memory allocation (Buffer and the row-dependent buffers). Only here is an out-of-memory
// status typed as DeviceAllocationError; check() keeps every other operation generic (pinned host staging included).
// The refused call also recorded its status in the runtime's last-error slot; the exception now carries it, so the slot
// is read back to success here: a caller that recovers (a retry with smaller buffers) must not find the old refusal in
// its next launch check.
inline void check_device_allocation(Status status, const char* operation) {
    if (status == STRIX_DEVICE_RT(Success)) return;
    static_cast<void>(STRIX_DEVICE_RT(GetLastError)());
#if defined(__HIPCC__)
    const bool out_of_memory = status == hipErrorOutOfMemory;
#else
    const bool out_of_memory = status == cudaErrorMemoryAllocation;
#endif
    throw_runtime_failure(std::string(operation) + ": " + STRIX_DEVICE_RT(GetErrorString)(status), static_cast<int>(status),
                          out_of_memory, RuntimeOperation::device_allocation);
}

inline int current_device() {
    int device{};
    check(STRIX_DEVICE_RT(GetDevice)(&device), "get device");
    return device;
}

inline void select(int device) { check(STRIX_DEVICE_RT(SetDevice)(device), "set device"); }

struct MemoryInfo {
    std::size_t free;
    std::size_t total;
};

inline MemoryInfo memory_info() {
    MemoryInfo result{};
    check(STRIX_DEVICE_RT(MemGetInfo)(&result.free, &result.total), "read free device memory");
    return result;
}

inline void copy(void* destination, const void* source, std::size_t bytes, CopyKind kind, Stream stream) {
    if (bytes != 0U)
        check(STRIX_DEVICE_RT(MemcpyAsync)(destination, source, bytes, kind, stream), "device copy");
}

inline void clear(void* destination, std::size_t bytes, Stream stream) {
    if (bytes != 0U) check(STRIX_DEVICE_RT(MemsetAsync)(destination, 0, bytes, stream), "device clear");
}

class StreamOwner {
public:
    StreamOwner() { check(STRIX_DEVICE_RT(StreamCreateWithFlags)(&stream_, STRIX_DEVICE_RT(StreamNonBlocking)), "create stream"); }
    ~StreamOwner() { (void)STRIX_DEVICE_RT(StreamDestroy)(stream_); }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    [[nodiscard]] Stream get() const { return stream_; }
private:
    Stream stream_{};
};

inline void synchronize(Stream stream) { check(STRIX_DEVICE_RT(StreamSynchronize)(stream), "synchronize stream"); }

class StepGraph {
public:
    StepGraph() = default;
    ~StepGraph() {
        discard_exec();
        if (device_) (void)STRIX_DEVICE_RT(Free)(device_);
#if defined(__HIPCC__)
        if (staging_) (void)hipHostFree(staging_);
#else
        if (staging_) (void)cudaFreeHost(staging_);
#endif
    }
    StepGraph(const StepGraph&) = delete;
    StepGraph& operator=(const StepGraph&) = delete;

    [[nodiscard]] const session::StepDescriptor* capture_descriptor() const noexcept {
        return capturing_ ? device_ : nullptr;
    }

    void invalidate() noexcept {
        discard_exec();
        policy_.invalidate();
    }

    void reset() noexcept {
        invalidate();
        policy_.reset();
    }

    // False means none of the captured work ran, so the caller can retry
    // eagerly. A launch error has uncertain submission status and propagates.
    // The serial owner drains the stream before returning or replacing storage.
    template <class Work>
    bool try_launch(Stream stream, bool enabled, unsigned rows, bool decode, bool stable_layout,
                    session::StepDescriptor input, Work&& work) {
        const auto action = policy_.select(enabled, rows, decode, stable_layout);
        if (action == session::StepReplayAction::eager) {
            invalidate();
            return false;
        }
        if (action == session::StepReplayAction::capture && !try_capture(stream, work)) return fail();

        *staging_ = input;
        if (STRIX_DEVICE_RT(MemcpyAsync)(device_, staging_, sizeof(input), host_to_device, stream) !=
            STRIX_DEVICE_RT(Success)) return fail();
        const auto status = STRIX_DEVICE_RT(GraphLaunch)(exec_, stream);
        if (status != STRIX_DEVICE_RT(Success)) {
            fail();
            check(status, "launch step graph");
        }
        return true;
    }

    [[nodiscard]] unsigned instantiations() const noexcept { return instantiations_; }

#if defined(__CUDACC__) && !defined(__HIPCC__)
    // Program replay supplies a stable device block and stages its transfers
    // outside the graph. inspect runs on the candidate before it is installed.
    template <class Work, class Inspect>
    bool prepare_program(Stream capture_stream, unsigned rows, bool rebind, Work&& work, Inspect&& inspect) {
        const auto expected = rebind ? session::StepReplayAction::replay : session::StepReplayAction::capture;
        if (policy_.select_bound(rows) != expected || (exec_ != nullptr) != rebind)
            throw std::logic_error("CUDA graph update requires the captured step width");
        const auto graph = capture(capture_stream, work);
        if (!graph) return fail();
        static_assert(noexcept(inspect(graph)));
        const bool valid = inspect(graph);
        bool installed = false;
        if (valid && rebind && exec_) {
            cudaGraphExecUpdateResultInfo info{};
            installed = cudaGraphExecUpdate(exec_, graph, &info) == cudaSuccess &&
                        info.result == cudaGraphExecUpdateSuccess;
        } else if (valid && !rebind && !exec_) {
            installed = instantiate(graph, rows);
        }
        (void)cudaGraphDestroy(graph);
        return installed || fail();
    }

    // A launch error can report prior asynchronous work. The caller treats it
    // as terminal instead of replaying work whose submission is uncertain.
    void launch_program(Stream stream) { check(cudaGraphLaunch(exec_, stream), "launch bound program graph"); }
#endif

private:
    bool allocate_descriptor() noexcept {
        if (!device_) {
            void* address{};
            if (STRIX_DEVICE_RT(Malloc)(&address, sizeof(*device_)) != STRIX_DEVICE_RT(Success)) return false;
            device_ = static_cast<session::StepDescriptor*>(address);
        }
        if (!staging_) {
            void* address{};
#if defined(__HIPCC__)
            const auto status = hipHostMalloc(&address, sizeof(*staging_));
#else
            const auto status = cudaMallocHost(&address, sizeof(*staging_));
#endif
            if (status != STRIX_DEVICE_RT(Success)) return false;
            staging_ = static_cast<session::StepDescriptor*>(address);
        }
        return true;
    }

    template <class Work>
    STRIX_DEVICE_RT(Graph_t) capture(Stream stream, Work& work) {
        if (STRIX_DEVICE_RT(StreamBeginCapture)(stream, STRIX_DEVICE_RT(StreamCaptureModeThreadLocal)) !=
                STRIX_DEVICE_RT(Success)) return nullptr;

        capturing_ = true;
        bool enqueued = false;
        try {
            work();
            enqueued = true;
        } catch (...) {
            // Retry through the ordinary path once capture has ended. Its
            // checkpoint/rollback handles a failure that occurs there too.
        }
        STRIX_DEVICE_RT(Graph_t) graph{};
        const auto end = STRIX_DEVICE_RT(StreamEndCapture)(stream, &graph);
        capturing_ = false;
        if (enqueued && end == STRIX_DEVICE_RT(Success)) return graph;
        if (graph) (void)STRIX_DEVICE_RT(GraphDestroy)(graph);
        return nullptr;
    }

    bool instantiate(STRIX_DEVICE_RT(Graph_t) graph, unsigned rows = 1U) {
#if defined(__HIPCC__)
        const auto status = hipGraphInstantiate(&exec_, graph, nullptr, nullptr, 0U);
#else
        const auto status = cudaGraphInstantiate(&exec_, graph, 0U);
#endif
        if (status != STRIX_DEVICE_RT(Success)) return false;
        policy_.captured(rows);
        ++instantiations_;
        return true;
    }

    template <class Work>
    bool try_capture(Stream stream, Work& work) {
        if (!allocate_descriptor()) return false;
        const auto graph = capture(stream, work);
        if (!graph) return false;
        const bool instantiated = instantiate(graph);
        (void)STRIX_DEVICE_RT(GraphDestroy)(graph);
        return instantiated;
    }

    bool fail() noexcept {
        discard_exec();
        policy_.fail();
        (void)STRIX_DEVICE_RT(GetLastError)();
        return false;
    }

    void discard_exec() noexcept {
        if (exec_) (void)STRIX_DEVICE_RT(GraphExecDestroy)(exec_);
        exec_ = nullptr;
    }

    session::StepReplayPolicy policy_;
    session::StepDescriptor* device_{};
    session::StepDescriptor* staging_{};
    STRIX_DEVICE_RT(GraphExec_t) exec_{};
    bool capturing_{};
    unsigned instantiations_{};
};

template <class T>
class Buffer {
public:
    explicit Buffer(std::size_t elements) : size_(elements) {
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("device buffer size overflow");
        if (elements == 0U) return;
        void* data{};
        check_device_allocation(STRIX_DEVICE_RT(Malloc)(&data, elements * sizeof(T)), "allocate device buffer");
        data_ = static_cast<T*>(data);
    }
    ~Buffer() {
        if (data_) (void)STRIX_DEVICE_RT(Free)(data_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    [[nodiscard]] T* data() const { return data_; }
    [[nodiscard]] std::span<T> span() const { return {data_, size_}; }
private:
    T* data_{};
    std::size_t size_{};
};

}  // namespace strix::device::native

#undef STRIX_DEVICE_RT
#endif
