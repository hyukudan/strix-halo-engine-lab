#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_dispatch.hpp"
#include "moe_core/route_wave_kernel.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

namespace core = strix::moe_core;
namespace grouped = core::device_grouped;
namespace moe = strix::moe;

constexpr unsigned experts = 128U, top_k = 8U;
constexpr unsigned tile_rows = core::grouped_dispatch::HipBackend::rows_per_tile;
struct Policy {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
        moe::RouteOrder::selection, experts, top_k, 1U, 1U, true, true, 1e-20F, 8.0F};
};
struct Math {
    __host__ __device__ static float exp(float x) { return ::expf(x); }
};
using Traits = moe::RouteWaveTraits<true, true>;
using Planner = grouped::Planner<experts, top_k>;
using Select = hipError_t (*)(const float*, const float*, unsigned*, float*, unsigned*, unsigned, hipStream_t);

void require(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

template<class T>
struct Buffer {
    T* data{};
    std::size_t size;
    explicit Buffer(std::size_t n) : size(n) { require(hipMallocManaged(&data, size * sizeof(T))); }
    ~Buffer() { (void)hipFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::span<T> span() const { return {data, size}; }
    bool same(const Buffer& other) const {
        return size == other.size && std::memcmp(data, other.data, size * sizeof(T)) == 0;
    }
};

struct Plan {
    Buffer<unsigned> ids, status, scratch, tables;
    Buffer<float> weights;
    grouped::Args args{};

    explicit Plan(unsigned rows)
        : ids(std::size_t{rows + 1U} * top_k), status(rows + 1U),
          scratch(Planner::scratch_bytes(rows) / sizeof(unsigned)),
          tables(2U * experts + 2U * grouped::tile_bound(rows, top_k, experts, tile_rows) + 2U * rows * top_k + 4U),
          weights(ids.size) {
        std::ranges::fill(ids.span(), ~0U);
        std::ranges::fill(status.span(), ~0U);
        std::ranges::fill(weights.span(), -1.0F);
        std::ranges::fill(tables.span(), ~0U);
        const unsigned cap = grouped::tile_bound(rows, top_k, experts, tile_rows);
        auto* const offsets = tables.data;
        auto* const touched = offsets + experts + 1U;
        auto* const tile_experts = touched + experts;
        auto* const begins = tile_experts + cap;
        auto* const perm_rows = begins + cap + 1U;
        auto* const perm_slots = perm_rows + rows * top_k;
        auto* const tile_count = perm_slots + rows * top_k;
        args.ids = ids.data;
        args.weights = weights.data;
        args.route_status = status.data;
        args.csr = {offsets, touched, tile_experts, begins, perm_rows, perm_slots};
        args.out.tile_count = tile_count;
        args.out.failure = tile_count + 1U;
    }

    void enqueue(Select select, const float* logits, const float* bias, unsigned rows, hipStream_t stream) {
        require(select(logits, bias, ids.data, weights.data, status.data, rows, stream));
        require(Planner::enqueue(args, rows, experts, top_k, tile_rows, scratch.data, stream));
    }
};

void check_rows(unsigned rows, bool bad, hipStream_t stream) {
    Buffer<float> logits(std::size_t{rows} * experts), bias(experts);
    for (unsigned e = 0; e < experts; ++e) bias.data[e] = static_cast<float>(e % 11U) / 32.0F;
    for (std::size_t i = 0; i < logits.size; ++i) logits.data[i] = static_cast<float>(i % 71U) / 8.0F - 4.0F;
    if (bad) logits.data[experts] = std::numeric_limits<float>::quiet_NaN();
    Plan legacy(rows), grid(rows);
    constexpr Select select_legacy = core::enqueue_route_wave<Policy, Math, Traits>;
    constexpr Select select_grid = core::enqueue_route_wave_grid<Policy, Math, Traits>;
    legacy.enqueue(select_legacy, logits.data, bias.data, rows, stream);
    grid.enqueue(select_grid, logits.data, bias.data, rows, stream);
    require(hipStreamSynchronize(stream));
    if (!legacy.ids.same(grid.ids) || !legacy.weights.same(grid.weights) || !legacy.status.same(grid.status) ||
        !legacy.tables.same(grid.tables) || grid.status.data[rows] != ~0U || grid.ids.span().back() != ~0U ||
        grid.weights.span().back() != -1.0F) {
        throw std::runtime_error("route grid changed router or CSR bytes");
    }
    if ((*grid.args.out.failure != 0U) != bad || (bad && *grid.args.out.tile_count != 0U) ||
        (!bad && grid.args.csr.offsets[experts] != rows * top_k)) {
        throw std::runtime_error("wrong CSR outcome");
    }
    for (unsigned row = 0; row < rows; ++row) {
        const bool invalid = bad && row == 1U;
        if (grid.status.data[row] != (invalid ? 1U : 0U)) throw std::runtime_error("wrong route status");
        for (unsigned slot = 0; slot < top_k; ++slot) {
            const auto i = std::size_t{row} * top_k + slot;
            if (invalid ? grid.ids.data[i] != experts || grid.weights.data[i] != 0.0F : grid.ids.data[i] >= experts)
                throw std::runtime_error("wrong route IDs or sentinel weights");
        }
    }
    if (bad) return;
    for (unsigned e = 0; e < experts; ++e) {
        const unsigned begin = grid.args.csr.offsets[e], end = grid.args.csr.offsets[e + 1U];
        for (unsigned i = begin; i < end; ++i) {
            const unsigned row = grid.args.csr.rows[i], slot = grid.args.csr.slots[i];
            if (row >= rows || slot >= top_k || grid.ids.data[std::size_t{row} * top_k + slot] != e ||
                (i != begin && row <= grid.args.csr.rows[i - 1U])) throw std::runtime_error("wrong CSR ordering");
        }
    }
    if (rows != 4096U && rows != 8192U) return;
    hipEvent_t start{}, end{};
    require(hipEventCreate(&start));
    require(hipEventCreate(&end));
    const auto time = [&](Select select, Plan& plan) {
        constexpr unsigned repeats = 32U;
        require(hipEventRecord(start, stream));
        for (unsigned i = 0; i < repeats; ++i) plan.enqueue(select, logits.data, bias.data, rows, stream);
        require(hipEventRecord(end, stream));
        require(hipEventSynchronize(end));
        float ms = 0.0F;
        require(hipEventElapsedTime(&ms, start, end));
        return ms / static_cast<float>(repeats);
    };
    for (unsigned round = 0; round < 3U; ++round) {
        const bool grid_first = round % 2U != 0U;
        const float first = grid_first ? time(select_grid, grid) : time(select_legacy, legacy);
        const float second = grid_first ? time(select_legacy, legacy) : time(select_grid, grid);
        std::cout << "route+CSR rows=" << rows << " legacy_ms=" << (grid_first ? second : first)
                  << " grid_ms=" << (grid_first ? first : second) << '\n';
    }
    require(hipEventDestroy(start));
    require(hipEventDestroy(end));
}

}  // namespace

int main() try {
    hipStream_t stream{};
    require(hipStreamCreate(&stream));
    for (const unsigned rows : {1U, 3U, 4U, 5U, 4096U, 8192U, 8193U}) check_rows(rows, false, stream);
    check_rows(8193U, true, stream);
    require(hipStreamDestroy(stream));
    std::cout << "route wave grid and CSR byte comparison PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
