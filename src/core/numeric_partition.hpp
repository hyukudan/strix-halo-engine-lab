#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_PARTITION_HD __host__ __device__
#else
#define STRIX_PARTITION_HD
#endif

namespace strix::core {

// Frozen numeric partition policy v1. These reference buckets preserve existing
// S values; physical tile tuning must never modify them.
//
// The policy is one set of constexpr functions of plain integers (usable from host planners, static_asserts and
// device planners alike); the Weights templates below only read the matrix descriptor. `codebook` 2 is the mul1
// codebook: only that codebook has the measured shape table.
namespace partition {

STRIX_PARTITION_HD constexpr bool medium(unsigned input, unsigned output, unsigned bits, unsigned codebook,
                                         unsigned rows) {
    if (rows < 32 || codebook != 2) return false;
    if (rows >= 256) return (bits == 6 && input == 5120 && output == 1280) || (bits == 5 && input == 4096 && output == 1024);
    if (bits == 6) return input == 5120 && output == 1280;
    if (bits != 5) return false;
    return (input == 1280 && output == 32768) || (input == 8192 && output == 5120) || (input == 5120 && output == 2304) ||
           (input == 2304 && output == 5120) || (input == 4096 && output == 1024);
}

STRIX_PARTITION_HD constexpr unsigned bucket_rows(unsigned input, unsigned output, unsigned bits, unsigned codebook,
                                                  unsigned rows) {
    return medium(input, output, bits, codebook, rows) || (rows >= 256 && output >= 2304) ||
                   (rows >= 64 && output >= 32768 && bits == 5 && codebook == 2)
               ? 64U
               : 16U;
}

// Vector (<= 8 rows) split count. Do not combine the shared R4 exact-global S16 experiment with unmeasured S8.
STRIX_PARTITION_HD constexpr unsigned vector_splits(unsigned input, unsigned output, unsigned bits, unsigned codebook,
                                                    unsigned rows) {
    if (bits == 5 && codebook == 2) {
        if (input == 4096 && output == 1024 && (rows == 1 || rows == 4 || rows == 8)) return 16U;
        if (input == 5120 && output == 2304 && rows == 8) return 8U;
    }
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}

// The rows-dependent split count of one matrix (v1 contract): a GEMV chain up to 8 rows, an occupancy-driven
// split-K above. Never more than one split per K16 tile.
STRIX_PARTITION_HD constexpr unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned codebook,
                                                  unsigned rows) {
    unsigned long long requested = 1;
    if (rows > 8) {
        const unsigned long long tile_m = bucket_rows(input, output, bits, codebook, rows);
        const unsigned long long buckets = ((rows + tile_m - 1U) / tile_m) * ((output + 63ULL) / 64ULL);
        if (medium(input, output, bits, codebook, rows)) {
            if (output >= 32768) {
                requested = rows < 64 ? 4U : 1U;
            } else {
                const unsigned long long target = rows > 64 && rows % 64 != 0 ? 512ULL : 256ULL;
                while (buckets * requested < target && requested < 16) requested *= 2;
            }
        } else {
            while (buckets * requested < 160 && requested < 32) requested *= 2;
        }
    } else {
        requested = vector_splits(input, output, bits, codebook, rows);
    }
    const unsigned long long k16 = input / 16U;
    return static_cast<unsigned>(requested < k16 ? requested : k16);
}

}  // namespace partition

// Negative rows never reach a plan; they keep the "below every threshold" answer of the signed form.
template<class Weights> inline bool numeric_partition_medium(const Weights& w, int rows) {
    return rows >= 0 && partition::medium(static_cast<unsigned>(w.input), static_cast<unsigned>(w.output),
                                          static_cast<unsigned>(w.bits), static_cast<unsigned>(w.codebook),
                                          static_cast<unsigned>(rows));
}
template<class Weights> inline int numeric_partition_bucket_rows(const Weights& w, int rows) {
    return static_cast<int>(partition::bucket_rows(static_cast<unsigned>(w.input), static_cast<unsigned>(w.output),
                                                   static_cast<unsigned>(w.bits), static_cast<unsigned>(w.codebook),
                                                   rows < 0 ? 0U : static_cast<unsigned>(rows)));
}
} // namespace strix::core

#undef STRIX_PARTITION_HD
