#pragma once

#include "moe_core/grouped_microscaled_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace strix::moe_core::device_grouped {

using grouped_microscaled::TileRows;
using weight::microscaled::PackedMatrix;

struct ExpertWeights { PackedMatrix gate_up, down; };

struct CsrTables {
    unsigned* offsets;  // [experts + 1]
    unsigned* touched;  // [experts]
    unsigned* experts;  // [tile_capacity]
    unsigned* begins;   // [tile_capacity + 1]
    unsigned* rows;     // [rows * top_k], also the gather permutation
    unsigned* slots;    // [rows * top_k], original token-major route slot
};

struct Output {
    float* assignment_weights;  // [rows * top_k], expert-major
    unsigned* combine_rows;     // [rows * top_k], token-major in ascending expert order
    TileRows* gate_up;            // [tile_capacity]
    TileRows* down;               // [tile_capacity]
    unsigned* tile_count;
    unsigned* failure;
    unsigned tile_capacity;
};

struct Workspace {
    std::uint16_t* bf16;  // [assignments * max(hidden, intermediate)]
    float* fp32;          // [assignments * max(hidden, 2 * intermediate)]
    unsigned hidden;
    unsigned intermediate;
};

struct Args {
    const unsigned* ids;      // [rows * top_k], token-major selection order
    const float* weights;      // same order as ids
    const unsigned* route_status;  // optional [rows], zero on success
    unsigned rows, top_k, experts, rows_per_tile;
    const ExpertWeights* matrices;  // [experts], validates before upload
    CsrTables csr;
    Output out;
    Workspace workspace;
};

// Number of launch descriptors sufficient for any routing of the chunk.
constexpr unsigned tile_bound(unsigned rows, unsigned top_k, unsigned experts, unsigned rows_per_tile) {
    const auto assignments = std::uint64_t(rows) * top_k;
    if (assignments == 0U || assignments > std::numeric_limits<unsigned>::max() || rows_per_tile == 0U) return 0U;
    const auto bound = std::uint64_t(experts) + (assignments - 1U) / rows_per_tile;
    return static_cast<unsigned>(std::min(assignments, bound));
}

inline bool fits(const Args& a) {
    const auto tiles = tile_bound(a.rows, a.top_k, a.experts, a.rows_per_tile);
    constexpr auto max_fp32 = std::numeric_limits<std::size_t>::max() / sizeof(float);
    const auto& w = a.workspace;
    return a.rows != 0U && a.rows <= 65535U && a.top_k != 0U && a.top_k <= a.experts &&
           a.ids && a.weights && a.matrices && a.csr.offsets && a.csr.touched &&
           a.csr.experts && a.csr.begins && a.csr.rows && a.csr.slots &&
           a.out.assignment_weights && a.out.combine_rows && a.out.gate_up && a.out.down &&
           a.out.tile_count && a.out.failure && a.out.failure != a.out.tile_count &&
           tiles != 0U && a.out.tile_capacity >= tiles && a.out.tile_capacity <= grouped_microscaled::kMaxTiles &&
           w.bf16 && w.fp32 && w.hidden != 0U && w.intermediate != 0U &&
           w.intermediate <= std::numeric_limits<unsigned>::max() / 2U &&
           std::uint64_t(a.rows) * a.top_k <= max_fp32 / std::max(w.hidden, 2U * w.intermediate);
}

}  // namespace strix::moe_core::device_grouped

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "moe_core/device_plan.hpp"

namespace strix::moe_core::device_grouped {

// Bind the existing packed three-phase CSR planner. No new route counting,
// prefix scan or scatter is needed to build the grouped operand views.
template<unsigned Experts, unsigned TopK>
struct CsrPolicy {
    static constexpr unsigned kExperts = Experts, kTopK = TopK;
    static constexpr unsigned kThreads = Experts < 256U ? 256U : Experts;
    static constexpr unsigned kAssignmentsPerWave = 128U, kCountLimit = 65535U;
    static constexpr unsigned kHistogramBins = 1U, kInvalid = 2U;
    static constexpr bool kCheckLaunch = true;
    using View = Args;

    __device__ static const CsrTables& tables(const View& v) { return v.csr; }
    __device__ static const unsigned* routes(const View& v, unsigned row) {
        return v.ids + std::size_t(row) * TopK;
    }
    __device__ static unsigned expert(const View& v, unsigned row, unsigned slot) { return routes(v, row)[slot]; }
    __device__ static unsigned status(const View& v, unsigned row) {
        return v.route_status ? v.route_status[row] : 0U;
    }
    template<bool Certificate>
    __device__ static bool invalid(const View& v, unsigned row, unsigned slot, unsigned id, unsigned experts, bool&) {
        return id >= experts || status(v, row) != 0U ||
               !isfinite(v.weights[std::size_t(row) * TopK + slot]);
    }
    __device__ static void certificate(const View&) {}
    __device__ static void failure(const View& v, unsigned tid, unsigned bits, unsigned, unsigned) {
        if (tid == 0U) {
            *v.out.failure = bits;
            *v.out.tile_count = 0U;
        }
    }
    __device__ static void summary(const View& v, unsigned tid, unsigned status_bits, unsigned, unsigned tiles,
                                   unsigned, unsigned duplicates, unsigned, unsigned, const unsigned*) {
        if (tid == 0U) {
            const unsigned bits = duplicates == 0U ? status_bits : status_bits | kInvalid;
            *v.out.failure = bits;
            *v.out.tile_count = bits == 0U ? tiles : 0U;
        }
    }
};

// One thread owns one assignment and, if present, one descriptor. The stream
// dependency on completed CSR scatter makes both device-resident permutations
// deterministic and avoids a host read of the number of live tiles.
template<unsigned TopK>
__global__ void materialize(Args a) {
    if (*a.out.failure != 0U) return;
    const unsigned assignment = blockIdx.x * blockDim.x + threadIdx.x;
    if (assignment >= a.rows * TopK) return;
    const unsigned row = a.csr.rows[assignment];
    const unsigned slot = a.csr.slots[assignment];
    const auto at = std::size_t(row) * TopK;
    const unsigned expert = a.ids[at + slot];
    unsigned rank = 0U;
#pragma unroll
    for (unsigned i = 0; i < TopK; ++i) rank += a.ids[at + i] < expert ? 1U : 0U;
    a.out.assignment_weights[assignment] = a.weights[at + slot];
    a.out.combine_rows[at + rank] = assignment;

    if (assignment >= *a.out.tile_count) return;
    const unsigned expert_id = a.csr.experts[assignment];
    const unsigned base = a.csr.begins[assignment];
    const unsigned remaining = a.csr.offsets[expert_id + 1U] - base;
    const unsigned count = remaining < a.rows_per_tile ? remaining : a.rows_per_tile;
    const auto& weights = a.matrices[expert_id];
    const auto hidden = a.workspace.hidden;
    const auto intermediate = a.workspace.intermediate;
    const unsigned gate_width = 2U * intermediate;
    const auto input_offset = std::size_t(base) * hidden;
    const auto activation_offset = std::size_t(base) * intermediate;
    a.out.gate_up[assignment] = {weights.gate_up, a.workspace.bf16 + input_offset,
        a.workspace.fp32 + std::size_t(base) * gate_width, hidden, gate_width, count};
    a.out.down[assignment] = {weights.down, a.workspace.bf16 + activation_offset,
        a.workspace.fp32 + input_offset, intermediate, hidden, count};
}

template<unsigned Experts, unsigned TopK>
using Planner = device_plan::PackedPlan<CsrPolicy<Experts, TopK>>;

template<unsigned Experts, unsigned TopK>
#if defined(__HIPCC__)
hipError_t enqueue(const Args& a, std::span<unsigned> scratch, hipStream_t stream) {
    constexpr auto invalid_value = hipErrorInvalidValue;
    constexpr auto success = hipSuccess;
#else
cudaError_t enqueue(const Args& a, std::span<unsigned> scratch, cudaStream_t stream) {
    constexpr auto invalid_value = cudaErrorInvalidValue;
    constexpr auto success = cudaSuccess;
#endif
    using Plan = Planner<Experts, TopK>;
    if (!fits(a) || a.experts != Experts || a.top_k != TopK ||
        scratch.size() < Plan::scratch_bytes(a.rows) / sizeof(unsigned)) return invalid_value;
    const auto status = Plan::enqueue(a, a.rows, Experts, TopK, a.rows_per_tile, scratch.data(), stream);
    if (status != success) return status;
    constexpr unsigned threads = 256U;
    const auto blocks = (a.rows * TopK - 1U) / threads + 1U;
#if defined(__HIPCC__)
    hipLaunchKernelGGL(materialize<TopK>, dim3(blocks), dim3(threads), 0, stream, a);
    return hipGetLastError();
#else
    materialize<TopK><<<blocks, threads, 0, stream>>>(a);
    return cudaGetLastError();
#endif
}

}  // namespace strix::moe_core::device_grouped
#endif
