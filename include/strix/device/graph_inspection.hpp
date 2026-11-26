#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define STRIX_GRAPH_RT(name) hip##name
#else
#include <cuda_runtime.h>
#define STRIX_GRAPH_RT(name) cuda##name
#endif

#include <cstddef>
#include <memory>
#include <new>
#include <span>

namespace strix::device::native {

// Opaque single-stream decode captures have no independently bound list of leaves. Validate
// their physical node vocabulary before installation. Declared programs additionally compare
// each kernel against its bound geometry and check the edges in their program executor.
inline bool inspect_raw_step_graph(STRIX_GRAPH_RT(Graph_t) graph, std::size_t& count, bool allow_events = false) noexcept {
    if (STRIX_GRAPH_RT(GraphGetNodes)(graph, nullptr, &count) != STRIX_GRAPH_RT(Success) || count == 0U)
        return false;
#if !defined(NDEBUG) || defined(STRIX_STRICT_GRAPH_CAPTURE)
    const auto allocated = count;
    using Node = STRIX_GRAPH_RT(GraphNode_t);
    std::unique_ptr<Node[]> nodes{new (std::nothrow) Node[count]};
    if (!nodes || STRIX_GRAPH_RT(GraphGetNodes)(graph, nodes.get(), &count) != STRIX_GRAPH_RT(Success) ||
        count != allocated) return false;
    for (const auto node : std::span(nodes.get(), count)) {
        STRIX_GRAPH_RT(GraphNodeType) kind{};
        if (STRIX_GRAPH_RT(GraphNodeGetType)(node, &kind) != STRIX_GRAPH_RT(Success)) return false;
        if (kind == STRIX_GRAPH_RT(GraphNodeTypeMemcpy) || kind == STRIX_GRAPH_RT(GraphNodeTypeMemset)) continue;
        if (allow_events && (kind == STRIX_GRAPH_RT(GraphNodeTypeEmpty) ||
                             kind == STRIX_GRAPH_RT(GraphNodeTypeEventRecord) ||
                             kind == STRIX_GRAPH_RT(GraphNodeTypeWaitEvent))) continue;
        if (kind != STRIX_GRAPH_RT(GraphNodeTypeKernel)) return false;
        STRIX_GRAPH_RT(KernelNodeParams) args{};
        if (STRIX_GRAPH_RT(GraphKernelNodeGetParams)(node, &args) != STRIX_GRAPH_RT(Success) || !args.func ||
            !args.gridDim.x || !args.gridDim.y || !args.gridDim.z ||
            !args.blockDim.x || !args.blockDim.y || !args.blockDim.z) return false;
    }
#else
    (void)allow_events;
#endif
    return true;
}

// Older segmented graphs can contain stream events and explicit ordering nodes.
// Inspect only at capture/update time. Preserve the native status contract of
// each owner, including its cleanup, retry and memory-accounting paths.
inline STRIX_GRAPH_RT(Error_t) inspect_segment_graph(STRIX_GRAPH_RT(Graph_t) graph) noexcept {
#if !defined(NDEBUG) || defined(STRIX_STRICT_GRAPH_CAPTURE)
    std::size_t count = 0;
    if (!inspect_raw_step_graph(graph, count, true)) return STRIX_GRAPH_RT(ErrorInvalidValue);
#else
    (void)graph;
#endif
    return STRIX_GRAPH_RT(Success);
}

template <class... Options>
inline STRIX_GRAPH_RT(Error_t) instantiate_segment_graph(STRIX_GRAPH_RT(GraphExec_t)* executable,
                                                         STRIX_GRAPH_RT(Graph_t) graph, Options... options) {
    const auto status = inspect_segment_graph(graph);
    if (status != STRIX_GRAPH_RT(Success)) return status;
    return STRIX_GRAPH_RT(GraphInstantiate)(executable, graph, options...);
}

}  // namespace strix::device::native

#undef STRIX_GRAPH_RT
#endif
