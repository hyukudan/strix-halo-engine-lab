#pragma once
// Expert-major plan of device-selected (row, expert) slots for the exact row-sweep expert path (core/dense_linear
// enqueue_expert_rows): the shared three-phase CSR planner (device_plan::PackedPlan) bound to a route table ids[rows][TopK]
// that already lives on the device. Output: CSR by expert (offsets), tiles of at most `rows_per_tile` assignments of ONE
// expert (experts / begins), the assignment -> (row, slot) permutation (rows / slots) and the live tile count, all on the
// device (no host read). A failed plan (id >= Experts, a repeated expert in a row, rows > 65535) leaves tile_count 0.
#include <cstddef>
#include <cstdint>

namespace strix::moe_core::expert_rows {

// Largest number of tiles any routing of `rows` rows can need at the smallest tile size `min_rows_per_tile`.
constexpr unsigned tile_capacity(unsigned rows, unsigned top_k, unsigned experts, unsigned min_rows_per_tile) {
    const std::uint64_t assignments = std::uint64_t{rows} * top_k;
    if (assignments == 0U || min_rows_per_tile == 0U) return 0U;
    const std::uint64_t bound = experts + (assignments - 1U) / min_rows_per_tile;
    return static_cast<unsigned>(assignments < bound ? assignments : bound);
}

struct Tables {
    unsigned *offsets, *touched, *experts, *begins, *rows, *slots, *tile_count, *failure;
};

}  // namespace strix::moe_core::expert_rows

#if defined(__HIPCC__)
#include "moe_core/device_plan.hpp"

namespace strix::moe_core::expert_rows {

template <unsigned Experts, unsigned TopK>
struct Policy {
    static constexpr unsigned kExperts = Experts, kTopK = TopK;
    static constexpr unsigned kThreads = Experts < 256U ? 256U : Experts;
    static constexpr unsigned kAssignmentsPerWave = 128U, kCountLimit = 65535U;
    static constexpr unsigned kHistogramBins = 1U, kInvalid = 2U;
    static constexpr bool kCheckLaunch = true;
    struct View {
        const std::uint32_t* ids;  // [rows][TopK]
        Tables tables;
    };
    __device__ static const Tables& tables(const View& v) { return v.tables; }
    __device__ static const unsigned* routes(const View& v, unsigned row) { return v.ids + std::size_t(row) * TopK; }
    __device__ static unsigned expert(const View& v, unsigned row, unsigned slot) { return routes(v, row)[slot]; }
    __device__ static unsigned status(const View&, unsigned) { return 0U; }
    template <bool Certificate>
    __device__ static bool invalid(const View&, unsigned, unsigned, unsigned id, unsigned experts, bool&) {
        return id >= experts;
    }
    __device__ static void certificate(const View&) {}
    __device__ static void failure(const View& v, unsigned tid, unsigned bits, unsigned, unsigned) {
        if (tid == 0U) {
            *v.tables.failure = bits;
            *v.tables.tile_count = 0U;
        }
    }
    __device__ static void summary(const View& v, unsigned tid, unsigned status_bits, unsigned, unsigned tiles, unsigned,
                                   unsigned duplicates, unsigned, unsigned, const unsigned*) {
        if (tid == 0U) {
            const unsigned bits = duplicates == 0U ? status_bits : status_bits | kInvalid;
            *v.tables.failure = bits;
            *v.tables.tile_count = bits == 0U ? tiles : 0U;
        }
    }
};

template <unsigned Experts, unsigned TopK>
using Planner = device_plan::PackedPlan<Policy<Experts, TopK>>;

// Words of the one device buffer `enqueue` slices: planner scratch, then the tables for capacity `rows` and tiles of at least
// `min_rows_per_tile` rows.
template <unsigned Experts, unsigned TopK>
constexpr std::size_t words(unsigned rows, unsigned min_rows_per_tile) {
    const std::size_t assignments = std::size_t{rows} * TopK, tiles = tile_capacity(rows, TopK, Experts, min_rows_per_tile);
    return Planner<Experts, TopK>::scratch_bytes(rows) / sizeof(unsigned) + (Experts + 1U) + Experts + tiles + (tiles + 1U) +
           2U * assignments + 2U;
}

// Slices `buffer` (words<Experts, TopK>(capacity, min_rows_per_tile) words), plans `rows` <= capacity rows of ids and
// returns the tables. cap = rows per tile (>= min_rows_per_tile).
template <unsigned Experts, unsigned TopK>
inline hipError_t enqueue(const std::uint32_t* ids, unsigned rows, unsigned capacity, unsigned cap, unsigned min_rows_per_tile,
                          unsigned* buffer, Tables& out, hipStream_t stream) {
    using Plan = Planner<Experts, TopK>;
    if (rows == 0U || rows > capacity || rows > 65535U || cap < min_rows_per_tile || min_rows_per_tile == 0U)
        return hipErrorInvalidValue;
    const std::size_t assignments = std::size_t{capacity} * TopK, tiles = tile_capacity(capacity, TopK, Experts, min_rows_per_tile);
    unsigned* at = buffer;
    unsigned* const scratch = at;
    at += Plan::scratch_bytes(capacity) / sizeof(unsigned);
    out.offsets = at; at += Experts + 1U;
    out.touched = at; at += Experts;
    out.experts = at; at += tiles;
    out.begins = at; at += tiles + 1U;
    out.rows = at; at += assignments;
    out.slots = at; at += assignments;
    out.tile_count = at++;
    out.failure = at++;
    return Plan::enqueue(typename Policy<Experts, TopK>::View{ids, out}, rows, Experts, TopK, cap, scratch, stream);
}

}  // namespace strix::moe_core::expert_rows
#endif
