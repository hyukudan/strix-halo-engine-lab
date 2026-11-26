#pragma once

namespace strix::moe_core {

// Four independent wave32 rows per HIP workgroup. A partial final block owns only live rows.
struct RouteWaveGrid {
    static constexpr unsigned rows_per_block = 4U;
    static constexpr unsigned threads_per_block = rows_per_block * 32U;

    static constexpr bool admits(unsigned rows) { return rows >= rows_per_block; }

    static constexpr unsigned blocks(unsigned rows) {
        return rows / rows_per_block + (rows % rows_per_block != 0U ? 1U : 0U);
    }

#if defined(__HIPCC__)
    __host__ __device__
#endif
    static constexpr unsigned row(unsigned block, unsigned wave) {
        return block * rows_per_block + wave;
    }
};

}  // namespace strix::moe_core
