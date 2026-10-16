#pragma once

// Deliberately include the complete SDK before the function-like redirects.
// Function addresses and external-library calls are not interposed. No kernel
// launch is wrapped, counted or timed as a GPU operation by this header.
#include <hip/hip_runtime.h>
#include "strix/execution_telemetry.hpp"
#include <chrono>
#include <limits>

namespace strix::telemetry_hip {
inline CopyDirection direction(hipMemcpyKind kind) noexcept {
    switch (kind) {
    case hipMemcpyHostToHost: return CopyDirection::host_to_host;
    case hipMemcpyHostToDevice: return CopyDirection::host_to_device;
    case hipMemcpyDeviceToHost: return CopyDirection::device_to_host;
    case hipMemcpyDeviceToDevice: return CopyDirection::device_to_device;
    default: return CopyDirection::automatic;
    }
}
template<class Call> inline hipError_t invoke(ExecutionActivityKind kind, std::uint64_t bytes,
    CopyDirection copy, hipStream_t stream, Call&& call, bool byte_saturated = false) {
    if (!current_execution_telemetry()) return call();
    const auto start = std::chrono::steady_clock::now();
    const auto result = call();
    const auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    observe_execution_activity(kind, bytes, 1, result == hipSuccess, ns, copy,
        reinterpret_cast<std::uintptr_t>(stream), static_cast<std::int32_t>(result), byte_saturated);
    return result;
}
inline hipError_t memcpy(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind) {
    return invoke(ExecutionActivityKind::copy_sync, bytes, direction(kind), nullptr,
        [&] { return ::hipMemcpy(dst, src, bytes, kind); });
}
inline hipError_t memcpy_async(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind, hipStream_t stream = nullptr) {
    return invoke(ExecutionActivityKind::copy_async, bytes, direction(kind), stream,
        [&] { return ::hipMemcpyAsync(dst, src, bytes, kind, stream); });
}
inline hipError_t memcpy2d(void* dst, std::size_t dp, const void* src, std::size_t sp,
    std::size_t width, std::size_t height, hipMemcpyKind kind) {
    const bool overflow = height && width > UINT64_MAX / height;
    return invoke(ExecutionActivityKind::copy_sync, overflow ? UINT64_MAX : width * height, direction(kind), nullptr,
        [&] { return ::hipMemcpy2D(dst, dp, src, sp, width, height, kind); }, overflow);
}
inline hipError_t memcpy2d_async(void* dst, std::size_t dp, const void* src, std::size_t sp,
    std::size_t width, std::size_t height, hipMemcpyKind kind, hipStream_t stream = nullptr) {
    const bool overflow = height && width > UINT64_MAX / height;
    return invoke(ExecutionActivityKind::copy_async, overflow ? UINT64_MAX : width * height, direction(kind), stream,
        [&] { return ::hipMemcpy2DAsync(dst, dp, src, sp, width, height, kind, stream); }, overflow);
}
inline hipError_t memset(void* dst, int value, std::size_t bytes) {
    return invoke(ExecutionActivityKind::memset_sync, bytes, CopyDirection::unspecified, nullptr,
        [&] { return ::hipMemset(dst, value, bytes); });
}
inline hipError_t memset_async(void* dst, int value, std::size_t bytes, hipStream_t stream = nullptr) {
    return invoke(ExecutionActivityKind::memset_async, bytes, CopyDirection::unspecified, stream,
        [&] { return ::hipMemsetAsync(dst, value, bytes, stream); });
}
template<class T> inline hipError_t malloc(T** pointer, std::size_t bytes) {
    return invoke(ExecutionActivityKind::device_alloc, bytes, CopyDirection::unspecified, nullptr,
        [&] { return ::hipMalloc(reinterpret_cast<void**>(pointer), bytes); });
}
inline hipError_t free(void* pointer) {
    return invoke(ExecutionActivityKind::device_free, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipFree(pointer); });
}
template<class T> inline hipError_t host_malloc(T** pointer, std::size_t bytes, unsigned flags = hipHostMallocDefault) {
    return invoke(ExecutionActivityKind::host_alloc, bytes, CopyDirection::unspecified, nullptr,
        [&] { return ::hipHostMalloc(reinterpret_cast<void**>(pointer), bytes, flags); });
}
inline hipError_t host_free(void* pointer) {
    return invoke(ExecutionActivityKind::host_free, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipHostFree(pointer); });
}
inline hipError_t host_register(void* pointer, std::size_t bytes, unsigned flags) {
    return invoke(ExecutionActivityKind::host_register, bytes, CopyDirection::unspecified, nullptr,
        [&] { return ::hipHostRegister(pointer, bytes, flags); });
}
inline hipError_t host_unregister(void* pointer) {
    return invoke(ExecutionActivityKind::host_unregister, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipHostUnregister(pointer); });
}
inline hipError_t device_synchronize() {
    return invoke(ExecutionActivityKind::device_sync, 0, CopyDirection::unspecified, nullptr,
        [] { return ::hipDeviceSynchronize(); });
}
inline hipError_t stream_synchronize(hipStream_t stream) {
    return invoke(ExecutionActivityKind::stream_sync, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipStreamSynchronize(stream); });
}
inline hipError_t event_synchronize(hipEvent_t event) {
    return invoke(ExecutionActivityKind::event_sync, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipEventSynchronize(event); });
}
inline hipError_t stream_wait_event(hipStream_t stream, hipEvent_t event, unsigned flags = 0) {
    return invoke(ExecutionActivityKind::stream_wait_event, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipStreamWaitEvent(stream, event, flags); });
}
inline hipError_t event_record(hipEvent_t event, hipStream_t stream = nullptr) {
    return invoke(ExecutionActivityKind::event_record, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipEventRecord(event, stream); });
}
inline hipError_t stream_create(hipStream_t* stream, unsigned flags) {
    return invoke(ExecutionActivityKind::stream_create, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipStreamCreateWithFlags(stream, flags); });
}
inline hipError_t stream_destroy(hipStream_t stream) {
    return invoke(ExecutionActivityKind::stream_destroy, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipStreamDestroy(stream); });
}
inline hipError_t event_create(hipEvent_t* event, unsigned flags) {
    return invoke(ExecutionActivityKind::event_create, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipEventCreateWithFlags(event, flags); });
}
inline hipError_t event_destroy(hipEvent_t event) {
    return invoke(ExecutionActivityKind::event_destroy, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipEventDestroy(event); });
}
inline hipError_t capture_begin(hipStream_t stream, hipStreamCaptureMode mode) {
    const auto result = invoke(ExecutionActivityKind::graph_capture_begin, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipStreamBeginCapture(stream, mode); });
    if (result == hipSuccess) execution_telemetry_capture_change(true);
    return result;
}
inline hipError_t capture_end(hipStream_t stream, hipGraph_t* graph) {
    const auto result = invoke(ExecutionActivityKind::graph_capture_end, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipStreamEndCapture(stream, graph); });
    // Preserve the host capture marker after other failures, just as the
    // owning capture lifecycle retains its drain/cleanup responsibility.
    if (result == hipSuccess || result == hipErrorStreamCaptureInvalidated)
        execution_telemetry_capture_change(false);
    return result;
}
inline hipError_t graph_instantiate(hipGraphExec_t* executable, hipGraph_t graph,
    hipGraphNode_t* error_node, char* log, std::size_t log_bytes) {
    return invoke(ExecutionActivityKind::graph_instantiate, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipGraphInstantiate(executable, graph, error_node, log, log_bytes); });
}
inline hipError_t graph_launch(hipGraphExec_t executable, hipStream_t stream) {
    if (current_execution_telemetry()) {
        // The ordinal identifies an attempted API replay; succeeded/failed remain separate.
        execution_telemetry_replay_attempted();
    }
    return invoke(ExecutionActivityKind::graph_replay, 0, CopyDirection::unspecified, stream,
        [&] { return ::hipGraphLaunch(executable, stream); });
}
inline hipError_t graph_destroy(hipGraph_t graph) {
    return invoke(ExecutionActivityKind::graph_destroy, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipGraphDestroy(graph); });
}
inline hipError_t graph_exec_destroy(hipGraphExec_t executable) {
    return invoke(ExecutionActivityKind::graph_exec_destroy, 0, CopyDirection::unspecified, nullptr,
        [&] { return ::hipGraphExecDestroy(executable); });
}
} // namespace strix::telemetry_hip

#define hipMemcpy(...) strix::telemetry_hip::memcpy(__VA_ARGS__)
#define hipMemcpyAsync(...) strix::telemetry_hip::memcpy_async(__VA_ARGS__)
#define hipMemcpy2D(...) strix::telemetry_hip::memcpy2d(__VA_ARGS__)
#define hipMemcpy2DAsync(...) strix::telemetry_hip::memcpy2d_async(__VA_ARGS__)
#define hipMemset(...) strix::telemetry_hip::memset(__VA_ARGS__)
#define hipMemsetAsync(...) strix::telemetry_hip::memset_async(__VA_ARGS__)
#define hipMalloc(...) strix::telemetry_hip::malloc(__VA_ARGS__)
#define hipFree(...) strix::telemetry_hip::free(__VA_ARGS__)
#define hipHostMalloc(...) strix::telemetry_hip::host_malloc(__VA_ARGS__)
#define hipHostFree(...) strix::telemetry_hip::host_free(__VA_ARGS__)
#define hipHostRegister(...) strix::telemetry_hip::host_register(__VA_ARGS__)
#define hipHostUnregister(...) strix::telemetry_hip::host_unregister(__VA_ARGS__)
#define hipDeviceSynchronize(...) strix::telemetry_hip::device_synchronize(__VA_ARGS__)
#define hipStreamSynchronize(...) strix::telemetry_hip::stream_synchronize(__VA_ARGS__)
#define hipEventSynchronize(...) strix::telemetry_hip::event_synchronize(__VA_ARGS__)
#define hipStreamWaitEvent(...) strix::telemetry_hip::stream_wait_event(__VA_ARGS__)
#define hipEventRecord(...) strix::telemetry_hip::event_record(__VA_ARGS__)
#define hipStreamCreateWithFlags(...) strix::telemetry_hip::stream_create(__VA_ARGS__)
#define hipStreamDestroy(...) strix::telemetry_hip::stream_destroy(__VA_ARGS__)
#define hipEventCreateWithFlags(...) strix::telemetry_hip::event_create(__VA_ARGS__)
#define hipEventDestroy(...) strix::telemetry_hip::event_destroy(__VA_ARGS__)
#define hipStreamBeginCapture(...) strix::telemetry_hip::capture_begin(__VA_ARGS__)
#define hipStreamEndCapture(...) strix::telemetry_hip::capture_end(__VA_ARGS__)
#define hipGraphInstantiate(...) strix::telemetry_hip::graph_instantiate(__VA_ARGS__)
#define hipGraphLaunch(...) strix::telemetry_hip::graph_launch(__VA_ARGS__)
#define hipGraphDestroy(...) strix::telemetry_hip::graph_destroy(__VA_ARGS__)
#define hipGraphExecDestroy(...) strix::telemetry_hip::graph_exec_destroy(__VA_ARGS__)
