#pragma once

// Host-only CUDA runtime stand-in for the tests that exercise the native device runtime header
// (strix/device/runtime.hpp) and the CUDA weight uploader without a GPU. Memory comes from the C heap; each
// allocation entry point returns the status the test installs, so an out-of-memory refusal can be injected at the
// exact call a production allocation makes.

#include <cstddef>
#include <cstdlib>

enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal = 1 };
enum { cudaGraphExecUpdateSuccess = 0 };
constexpr unsigned cudaStreamNonBlocking = 1U;

using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel, cudaGraphNodeTypeMemcpy, cudaGraphNodeTypeMemset, cudaGraphNodeTypeEmpty,
    cudaGraphNodeTypeEventRecord, cudaGraphNodeTypeWaitEvent, cudaGraphNodeTypeHost
};
struct cudaKernelNodeParams {
    void* func{};
    struct Extent { unsigned x = 1, y = 1, z = 1; } gridDim, blockDim;
};
struct StubGraphNode {
    cudaGraphNodeType kind;
    cudaKernelNodeParams kernel;
};
using cudaGraphNode_t = StubGraphNode*;

// The capture fixtures use one native node. A null handle models an empty graph.
inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count) {
    if (graph && nodes && *count) nodes[0] = static_cast<StubGraphNode*>(graph);
    *count = graph ? 1U : 0U;
    return cudaSuccess;
}
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type) {
    *type = node->kind;
    return cudaSuccess;
}
inline cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* result) {
    *result = node->kernel;
    return cudaSuccess;
}
struct cudaGraphExecUpdateResultInfo {
    int result;
};

namespace cuda_status_stub {
inline cudaError_t malloc_status = cudaSuccess;       // cudaMalloc (device memory)
inline cudaError_t malloc_host_status = cudaSuccess;  // cudaMallocHost (pinned host memory)
// Free device memory: a cudaMalloc larger than this is refused with cudaErrorMemoryAllocation (default: no limit).
inline std::size_t malloc_free_bytes = static_cast<std::size_t>(-1);
inline unsigned malloc_calls = 0, malloc_host_calls = 0;
// The runtime's last-error slot: a failed call records its status, cudaGetLastError reads and resets it.
inline cudaError_t last_error = cudaSuccess;
inline cudaError_t fail(cudaError_t status) { return last_error = status; }
}  // namespace cuda_status_stub

inline cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
    ++cuda_status_stub::malloc_calls;
    if (cuda_status_stub::malloc_status != cudaSuccess) return cuda_status_stub::fail(cuda_status_stub::malloc_status);
    if (bytes > cuda_status_stub::malloc_free_bytes) return cuda_status_stub::fail(cudaErrorMemoryAllocation);
    *pointer = std::malloc(bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** pointer, std::size_t bytes) {
    ++cuda_status_stub::malloc_host_calls;
    if (cuda_status_stub::malloc_host_status != cudaSuccess)
        return cuda_status_stub::fail(cuda_status_stub::malloc_host_status);
    *pointer = std::malloc(bytes);
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() {
    const auto status = cuda_status_stub::last_error;
    cuda_status_stub::last_error = cudaSuccess;
    return status;
}
inline const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaErrorMemoryAllocation ? "out of memory" : error == cudaSuccess ? "no error" : "invalid value";
}
inline cudaError_t cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    *free = *total = 0;
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    *stream = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
    *graph = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t, unsigned long long) {
    *exec = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo* info) {
    info->result = cudaGraphExecUpdateSuccess;
    return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
