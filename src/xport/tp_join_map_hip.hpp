#pragma once
// Join-map joins of the exact tensor-parallel pass on HIP: the gather of the slots' first rows x maxw words then
// out[rows][cols] = the joined rows (core tp_join kernels), FP32 and the FP16 form that feeds an IU4 GEMV. Launch configs and
// op order are the engines' own; slots, maps, rows are arguments.
// Home: shared (backend core, HIP). Bound by: qwen35 dense HIP exact TP decode (src/qwen35/hip/engine.hip tp_join /
// tp_join_x). Bodies moved unchanged.
#include "strix/hip/legacy_launch.hpp"
#include "strix/pair/gather_fn.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../core/dense_linear_iu4.hpp"
#include "../core/tp_join.hpp"
#include "tp_join_exec_hip.hpp"

namespace strix::xport::tp_join_exec {

// All-gather of the slots' first rows x maxw words, then out[rows][cols] = the joined rows (join map base / stride).
inline void join_map(const strix::pair::GatherFn& g, hipStream_t stream, float* slots, unsigned slot_floats, const int* base, const int* stride,
                     unsigned maxw, unsigned cols, unsigned rows, float* out, std::uint64_t& calls) {
    swap(g, stream, slots, std::size_t(slot_floats) * 4U, std::size_t(rows) * maxw * 4U, calls);
    const unsigned n = rows * cols;
    STRIX_HIP_LAUNCH(Rdna3Layout, strix::core::tp_join::tp_join_kernel<<<std::min(1024U, (n + 255U) / 256U), 256, 0, stream>>>(slots, base, stride, cols, rows, out));
}

// The FP16 form: this rank's slot converted with prepare's elementwise kernel, the halves gathered, joined into the IU4
// activation rows xh in prepare's placement.
inline void join_map_f16(const strix::pair::GatherFn& g, hipStream_t stream, const float* own_slot, _Float16* hslots, unsigned rank, unsigned slot_floats,
                         const int* base, const int* stride, unsigned maxw, unsigned cols, unsigned rows, _Float16* xh, std::uint64_t& calls) {
    const std::size_t words = std::size_t(rows) * maxw;
    STRIX_HIP_LAUNCH(Rdna3Layout, strix::core::dense_iu4::to_f16_kernel<0><<<64, 256, 0, stream>>>(own_slot, hslots + std::size_t(rank) * slot_floats, words));
    swap(g, stream, hslots, std::size_t(slot_floats) * 2U, words * 2U, calls);
    const unsigned n = rows * cols;
    STRIX_HIP_LAUNCH(Rdna3Layout, strix::core::tp_join::tp_join16_kernel<<<std::min(1024U, (n + 255U) / 256U), 256, 0, stream>>>(hslots, base, stride, cols, rows, xh,
                                                                                                strix::core::dense_iu4::scalar_x()));
}

}  // namespace strix::xport::tp_join_exec
