#pragma once

// Device decoding of Q6P32 weights (layout in include/strix/q6p_registry.hpp).
//
// Eight consecutive weights 8g..8g+7 of a 32-block come from one qs dword (their
// low nibbles) and half of one qh dword (their high bit pairs). The decode yields
// the same signed int8 codes, in the same byte order, as the 8 code bytes of the
// equivalent Q8_0 block, so callers keep their Q8_0 arithmetic unchanged.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::q6p {

// Signed codes of the four 6-bit values held in the bytes of c (each 0..63): c - 32
// per byte without borrows between bytes.
__device__ __forceinline__ std::uint32_t signed_codes(std::uint32_t c) { return ((c | 0x80808080U) - 0x20202020U) ^ 0x80808080U; }

// Codes of weights 8g..8g+7 (byte l = weight 8g+l) from the qs dword of group g and the
// qh dword of groups 2(g/2), 2(g/2)+1; shift = 4 * (g & 1).
__device__ __forceinline__ std::uint64_t codes8(std::uint32_t qs, std::uint32_t qh, unsigned shift) {
    const std::uint32_t h = qh >> shift;
    const std::uint32_t even = (qs & 0x0f0f0f0fU) | ((h << 4U) & 0x30303030U);
    const std::uint32_t odd = ((qs >> 4U) & 0x0f0f0f0fU) | ((h << 2U) & 0x30303030U);
    return static_cast<std::uint64_t>(signed_codes(odd)) << 32U | signed_codes(even);
}
// Codes of weights 4m..4m+3 of a block (m = 0..7) from qs dword m/2 and qh dword m/4.
__device__ __forceinline__ std::uint32_t codes4(std::uint32_t qs, std::uint32_t qh, unsigned m) {
    return signed_codes(((qs >> (4U * (m & 1U))) & 0x0f0f0f0fU) | (((qh >> (2U * (m & 3U))) & 0x03030303U) << 4U));
}

// Byte offsets inside a row of `cols` weights: chunk c (weights 256c..) starts at 208c and
// holds n = min(256, cols - 256c) weights as qs (n/2 bytes), qh (n/4), d (n/16).
struct Offsets {
    std::uint32_t qs, qh, d;
};
__host__ __device__ __forceinline__ Offsets chunk_offsets(std::uint32_t cols, std::uint32_t chunk) {
    const std::uint32_t n = cols - 256U * chunk < 256U ? cols - 256U * chunk : 256U, at = 208U * chunk;
    return {at, at + n / 2U, at + n / 2U + n / 4U};
}
// Offsets of the qs dword, qh dword and d of weights w..w+7 (w % 8 == 0).
__host__ __device__ __forceinline__ Offsets group_offsets(std::uint32_t cols, std::uint32_t w) {
    const auto c = chunk_offsets(cols, w / 256U);
    const std::uint32_t in = w % 256U;
    return {c.qs + in / 2U, c.qh + (in / 32U) * 8U + ((in / 16U) & 1U) * 4U, c.d + (in / 32U) * 2U};
}
__host__ __device__ __forceinline__ constexpr std::uint64_t row_bytes(std::uint64_t cols) { return cols / 32U * 26U; }

// Row r of a Q6P32 tensor of width `cols` (tensor below 4 GiB): the tensor base plus the
// row's byte offset. Callers whose row is wave-uniform pass it through uniform() so loads
// address an SGPR base plus a 32-bit offset.
struct Row {
    const unsigned char* tensor;
    std::uint32_t offset, cols;
};
__device__ __forceinline__ Row row_of(const unsigned char* base, std::uint32_t cols, std::uint64_t r) {
    return {base, static_cast<std::uint32_t>(r * row_bytes(cols)), cols};
}
__device__ __forceinline__ Row uniform(const Row& row) {
    const auto address = reinterpret_cast<std::uintptr_t>(row.tensor);
    const auto lo = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(address & 0xffffffffU)));
    const auto hi = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(address >> 32U)));
    return {reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(hi) << 32U | lo),
        static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(row.offset))),
        static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(row.cols)))};
}

// Global (not flat) loads: an address passed through readfirstlane has lost its address space.
template <class T> __device__ __forceinline__ T load_global(const unsigned char* tensor, std::uint32_t offset) {
    using Global = const __attribute__((address_space(1))) T*;
    return *reinterpret_cast<Global>(reinterpret_cast<std::uintptr_t>(tensor) + offset);
}

// Raw words of weights w..w+7 (w % 8 == 0): decode with codes8(qs, qh, shift_of(w)).
struct Group {
    std::uint32_t qs, qh;
    unsigned scale;
};
__device__ __forceinline__ Group load_group(const Row& row, std::uint32_t w) {
    const auto o = group_offsets(row.cols, w);
    return {load_global<std::uint32_t>(row.tensor, row.offset + o.qs), load_global<std::uint32_t>(row.tensor, row.offset + o.qh),
        load_global<std::uint16_t>(row.tensor, row.offset + o.d)};
}
__device__ __forceinline__ unsigned shift_of(std::uint32_t w) { return ((w / 8U) & 1U) * 4U; }

}  // namespace strix::q6p
