#pragma once
// Hi/lo panel form of the shared FP16 tall GEMM: f16_dense_tall::core with a ColsHiLo policy over prepared panels.
// y[r][o] = merge of two WMMA chains (f16_hilo_math.hpp) over hi = fp16(x) and lo = fp16((x - hi) * scale).
//
// Layouts (FP16, prepared by the caller):
//   weights : [ceil(outputs / 16) tiles][stride / 16 units][16 columns][16 k] (a tile's unit is one 512-byte panel)
//   x       : hi panels [stride / 16 units][rows][16 k] at x, lo panels in the same layout at x + rows * stride
// stride % 16 == 0. The core runs whole 8-unit iterations: the units past stride / 16 read as +0 operands (+0
// products leave every chain value unchanged; a chain from +0 never holds -0), and a blocked product closes its last
// block at the last unit as before. Output y[r * outputs + o] (row-major, `outputs` floats per row); a non-finite stored
// value sets *failure |= 1.
#include "f16_dense_tall.hpp"

namespace strix::core::f16_dense_tall::hilo {

// BlockUnits: 0 = unblocked, else chains restart every BlockUnits k units (multiple of 8). Unscale = 1 / scale.
template <unsigned BlockUnits, float Unscale> struct PanelCols {
    static constexpr unsigned kHiLoBlockUnits = BlockUnits;
    static constexpr float kHiLoUnscale = Unscale;
    const _Float16* w;
    const _Float16* x;
    float* y;
    unsigned rows, outputs, units, tiles;  // units = stride / 16 (stored), tiles = ceil(outputs / 16)
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        return {w, y + static_cast<std::size_t>(i) * 16U, outputs, i < tiles};
    }
    // The stager's 4 units of column `col`; dead tiles read tile 0, units past the stored ones are +0.
    __device__ __forceinline__ void stage(unsigned t, unsigned col, unsigned unit, U4* nb) const {
        const unsigned tt = t < tiles ? t : 0U;
        const _Float16* const col_base = w + static_cast<std::size_t>(tt) * units * 256U + col * 16U;  // inside tile tt's unit 0
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            // The unit's address is formed only when the unit is stored (no pointer past the panel array).
            const bool stored = unit + i < units;
            const auto* p = stored ? reinterpret_cast<const U4*>(col_base + static_cast<std::size_t>(unit + i) * 256U) : nullptr;
            nb[2U * i] = stored ? p[0] : U4{};
            nb[2U * i + 1U] = stored ? p[1] : U4{};
        }
    }
    __device__ __forceinline__ Half16 panel(const _Float16* base, unsigned row, unsigned unit) const {
        if (unit >= units) return Half16{};
        return *reinterpret_cast<const Half16*>(base + (static_cast<std::size_t>(unit) * rows + row) * 16U);
    }
    __device__ __forceinline__ Half16 fragment(unsigned row, unsigned unit) const { return panel(x, row, unit); }
    __device__ __forceinline__ Half16 fragment_lo(unsigned row, unsigned unit) const {
        return panel(x + static_cast<std::size_t>(rows) * units * 16U, row, unit);
    }
};

// Stores live columns (< outputs) and raises *failure |= 1 once per thread that stored a non-finite value.
struct CheckedStore {
    unsigned outputs;
    unsigned* failure;
    struct Tile {
        float* y;  // the tile's column-0 element of row 0 (the element address is formed only for a live column)
        unsigned c;
        unsigned column;
    };
    struct Mode {
        const CheckedStore& p;
        unsigned bad = 0;
        __device__ __forceinline__ Tile open(unsigned i, const ColumnTile& ct, unsigned c) const { return {ct.y, c, 16U * i + c}; }
        __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) {
            if (t.column >= p.outputs) return;
            bad |= (__builtin_bit_cast(unsigned, v) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
            t.y[static_cast<std::size_t>(r) * p.outputs + t.c] = v;
        }
        __device__ __forceinline__ void close() const {
            if (bad) atomicOr(p.failure, 1U);
        }
    };
    template <class F> __device__ __forceinline__ void run(F&& f) {
        Mode m{*this};
        f(m);
    }
};

template <unsigned TC, unsigned BlockUnits, float Unscale>
__global__ __launch_bounds__(256) void panel_kernel(PanelCols<BlockUnits, Unscale> cols, CheckedStore store) {
    __shared__ Storage<TC> lds;
    const unsigned in = (cols.units + 7U) / 8U * 128U;  // whole 8-unit iterations (+0 units past the stored ones)
    core<TC>(static_cast<const float*>(nullptr), in, cols.rows, cols, lds, store, true);
}

// One launch: 128 rows per CTA, the widest of 4 / 2 / 1 column tiles per CTA that divides the tiles (no dead tiles).
template <unsigned BlockUnits, float Unscale>
inline void launch_panels(const _Float16* w, const _Float16* x, unsigned rows, unsigned outputs, unsigned stride, float* y,
                          unsigned* failure, hipStream_t s) {
    const unsigned tiles = (outputs + 15U) / 16U, tc = x16_tc(tiles), gr = (rows + kRows - 1U) / kRows;
    const PanelCols<BlockUnits, Unscale> cols{w, x, y, rows, outputs, stride / 16U, tiles};
    const CheckedStore store{outputs, failure};
    switch (tc) {
    case 4: STRIX_HIP_LAUNCH(Rdna3Layout, (panel_kernel<4, BlockUnits, Unscale>)<<<dim3(gr, tiles / 4U), 256, 0, s>>>(cols, store)); break;
    case 2: STRIX_HIP_LAUNCH(Rdna3Layout, (panel_kernel<2, BlockUnits, Unscale>)<<<dim3(gr, tiles / 2U), 256, 0, s>>>(cols, store)); break;
    default: STRIX_HIP_LAUNCH(Rdna3Layout, (panel_kernel<1, BlockUnits, Unscale>)<<<dim3(gr, tiles), 256, 0, s>>>(cols, store)); break;
    }
}

}  // namespace strix::core::f16_dense_tall::hilo
