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

#endif

#include "strix/device/declaration.hpp"

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

inline int current_device() {
    int device{};
    check(STRIX_DEVICE_RT(GetDevice)(&device), "get device");
    return device;
}

inline void select(int device) { check(STRIX_DEVICE_RT(SetDevice)(device), "set device"); }

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
    ~StepGraph() { if (exec_) (void)STRIX_DEVICE_RT(GraphExecDestroy)(exec_); }
    StepGraph(const StepGraph&) = delete;
    StepGraph& operator=(const StepGraph&) = delete;

    void begin(Stream stream) {
        check(STRIX_DEVICE_RT(StreamBeginCapture)(stream, STRIX_DEVICE_RT(StreamCaptureModeThreadLocal)),
              "begin step graph capture");
        capturing_ = true;
    }

    void launch(Stream stream) {
        STRIX_DEVICE_RT(Graph_t) graph{};
        const auto end = STRIX_DEVICE_RT(StreamEndCapture)(stream, &graph);
        capturing_ = false;
        if (end != STRIX_DEVICE_RT(Success) && graph) (void)STRIX_DEVICE_RT(GraphDestroy)(graph);
        check(end, "end step graph capture");
        Status status = STRIX_DEVICE_RT(Success);
        if (exec_) {
#if defined(__HIPCC__)
            hipGraphNode_t error_node{};
            hipGraphExecUpdateResult result{};
            const auto update = STRIX_DEVICE_RT(GraphExecUpdate)(exec_, graph, &error_node, &result);
            const bool updated = update == hipSuccess && result == hipGraphExecUpdateSuccess;
#else
            cudaGraphExecUpdateResultInfo info{};
            const auto update = STRIX_DEVICE_RT(GraphExecUpdate)(exec_, graph, &info);
            const bool updated = update == cudaSuccess && info.result == cudaGraphExecUpdateSuccess;
#endif
            if (updated) {
                ++updates_;
            } else {
                (void)STRIX_DEVICE_RT(GetLastError)();
                (void)STRIX_DEVICE_RT(GraphExecDestroy)(exec_);
                exec_ = nullptr;
            }
        }
        if (!exec_) {
#if defined(__HIPCC__)
            status = STRIX_DEVICE_RT(GraphInstantiate)(&exec_, graph, nullptr, nullptr, 0U);
#else
            status = STRIX_DEVICE_RT(GraphInstantiate)(&exec_, graph, 0U);
#endif
            if (status == STRIX_DEVICE_RT(Success)) ++instantiations_;
        }
        (void)STRIX_DEVICE_RT(GraphDestroy)(graph);
        check(status, "instantiate step graph");
        check(STRIX_DEVICE_RT(GraphLaunch)(exec_, stream), "launch step graph");
    }

    void abandon(Stream stream) noexcept {
        if (!capturing_) return;
        STRIX_DEVICE_RT(Graph_t) graph{};
        (void)STRIX_DEVICE_RT(StreamEndCapture)(stream, &graph);
        capturing_ = false;
        if (graph) (void)STRIX_DEVICE_RT(GraphDestroy)(graph);
    }

    [[nodiscard]] bool capturing() const { return capturing_; }
    [[nodiscard]] unsigned instantiations() const { return instantiations_; }
    [[nodiscard]] unsigned updates() const { return updates_; }
private:
    STRIX_DEVICE_RT(GraphExec_t) exec_{};
    bool capturing_{};
    unsigned instantiations_{};
    unsigned updates_{};
};

template <class T>
class Buffer {
public:
    explicit Buffer(std::size_t elements) : size_(elements) {
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("device buffer size overflow");
        if (elements == 0U) return;
        void* data{};
        check(STRIX_DEVICE_RT(Malloc)(&data, elements * sizeof(T)), "allocate device buffer");
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
