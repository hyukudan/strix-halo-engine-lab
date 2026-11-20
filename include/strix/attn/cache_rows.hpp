#pragma once
// Shared attention cache-row contract (host / device neutral; HIP body src/attn_core/cache_rows.hpp).
//
// An attention cache is an array of fixed-width bf16 rows, one row per token position:
//   - MLA latent caches: [kv_lora latent (normalised) | rope tail (RoPE applied)] (mla_contract.hpp, read by
//     attn_core/sparse_flash_mla.hpp LatentPolicy::key_row);
//   - indexer token-key caches (group_indexer.hpp: the token keys pooled into group keys);
//   - any other per-token bf16 row (GLM DSA latent / index key / index gate rows: GLM's fused store kernels bind
//     encode / finite through glm_math.hpp and index position * width + i = row_offset with ring 0).
// The writer stores an FP32 row produced by the shared stages (projection, norm, rotary) as bf16 RNE at
//   slot(position) = position              (ring == 0: a full cache, List selection), or
//   slot(position) = position % ring       (a ring: the sliding-window latent ring, the indexer token-key ring).
// A ring read by the sparse flash core's Window trait must be a power of two (the reader masks: pos & (ring - 1));
// window_ring() gives that size. Token-key rings (group_indexer::token_ring_slots) need not be.
//
// Encoding: encode() is round-to-nearest-even of a FINITE float (the statement every cache site uses; a non-finite
// input has no defined image: the caller flags it, see store_rows_ref / finite()). Rows are written whole; a slot is
// overwritten when the ring wraps.
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_CR_HD __host__ __device__
#else
#define STRIX_CR_HD
#endif

namespace strix::attn::cache_rows {

struct Rows {
    unsigned width;  // bf16 elements per row
    unsigned ring;   // 0: slot = position; otherwise slot = position % ring
};

STRIX_CR_HD constexpr std::uint64_t slot(const Rows& r, std::uint64_t position) {
    return r.ring == 0U ? position : position % r.ring;
}
// First element of the row of `position` (elements from the cache base).
STRIX_CR_HD constexpr std::size_t row_offset(const Rows& r, std::uint64_t position) {
    return static_cast<std::size_t>(slot(r, position)) * r.width;
}
// Ring rows the sparse flash Window reader needs: a power of two >= window - 1 + rows written per launch (the
// oldest key of the first row and the newest row of the chunk must not share a slot).
constexpr unsigned window_ring(unsigned window, unsigned chunk_rows) {
    const unsigned need = window - 1U + chunk_rows;
    unsigned p = 1U;
    while (p < need) p <<= 1U;
    return p;
}
constexpr bool is_pow2(unsigned x) { return x != 0U && (x & (x - 1U)) == 0U; }

// RNE to bfloat16 of a finite float.
STRIX_CR_HD inline std::uint16_t encode(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
STRIX_CR_HD inline bool finite(std::uint16_t b) { return (b & 0x7f80U) != 0x7f80U; }
STRIX_CR_HD inline bool finite(float x) { return (__builtin_bit_cast(std::uint32_t, x) & 0x7f800000U) != 0x7f800000U; }
STRIX_CR_HD inline float decode(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
// The writer's flag rule for one value (store_rows_ref and every device writer): the input and its encoding are finite.
STRIX_CR_HD inline bool encodes_finite(float x) { return finite(x) && finite(encode(x)); }
// A span of a wide row as a row of a narrow view of the same buffer: columns [first, first + width) of row k of rows
// `wide` elements wide are row subrow(k, ...) of the buffer read as rows `width` wide (subrow_ok: width divides wide
// and first). A writer stores such spans with Rows{width, 0} at those positions (Hy4: the rope tail of a query image).
constexpr bool subrow_ok(unsigned wide, unsigned width, unsigned first) {
    return width != 0U && wide % width == 0U && first % width == 0U && first + width <= wide;
}
constexpr std::uint32_t subrow(std::uint32_t k, unsigned wide, unsigned width, unsigned first) {
    return k * (wide / width) + first / width;
}

// Host reference of the writer: rows r = 0..n-1 of src (stride src_stride floats) go to the row of position
// positions[r] (positions == nullptr: first_position + r). Returns false if any input or encoded value is not finite
// (the device writer raises its flag for the same rows; the stored bits are the same either way).
inline bool store_rows_ref(const float* src, std::size_t src_stride, const std::uint32_t* positions,
                           std::uint32_t first_position, unsigned n, const Rows& r, std::uint16_t* cache) {
    bool ok = true;
    for (unsigned k = 0; k < n; ++k) {
        const std::uint64_t p = positions != nullptr ? positions[k] : std::uint64_t{first_position} + k;
        std::uint16_t* dst = cache + row_offset(r, p);
        for (unsigned i = 0; i < r.width; ++i) {
            const float x = src[static_cast<std::size_t>(k) * src_stride + i];
            const std::uint16_t b = encode(x);
            ok = ok && finite(x) && finite(b);
            dst[i] = b;
        }
    }
    return ok;
}

}  // namespace strix::attn::cache_rows
