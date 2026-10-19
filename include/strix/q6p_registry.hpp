#pragma once

// Q6P32 dense weights (6.5 bits per weight) that stand in for Q8_0 tensors.
//
// A Q6P32 tensor of R rows (outputs) and K columns (inputs, K % 32 == 0) stores
// blocks of 32 weights w = d * q with q in [-32, 31] and d an f16. Rows are
// K/32 * 26 bytes, row-major. Inside a row, chunk c (weights 256c.., n =
// min(256, K - 256c) of them) starts at byte 208c and holds, with c_e = q_e + 32:
//   qs  n/2 bytes   block b at 16b:  byte 4k+i = lo4(c[8k+i]) | lo4(c[8k+4+i]) << 4
//   qh  n/4 bytes   block b at 8b:   byte 4j+i = sum_f hi2(c[16j+4f+i]) << 2f
//   d   n/16 bytes  block b at 2b:   the f16 scale
// (e indexes the weights of the block). A wave reading 256 weights of a row
// touches one contiguous 208-byte range.
//
// Every Q6P32 block is also an exact Q8_0 block (same d, codes q), so a model
// whose Q8_0 tensors hold Q6P32 values (the "emulated" checkpoint) is computed
// bit-identically by the Q8_0 kernels and by the Q6P32 kernels.
//
// The registry maps the device address of such a Q8_0 tensor to its Q6P32 copy
// (or of an F32 tensor to its BF16 copy). Launchers that have a Q6P32 kernel look
// their Q8_0 weights up here and fall back to the Q8_0 kernel when the address is
// not registered or STRIX_Q6P=0.

#include <cstdint>

namespace strix::q6p {

enum class Kind : std::uint8_t { q6p32, bf16 };

struct Tensor {
    const unsigned char* base = nullptr;  // device address of the Q6P32 rows (or the BF16 values)
    std::uint32_t rows = 0, cols = 0;
    Kind kind = Kind::q6p32;
};

constexpr std::uint64_t q6p32_bytes(std::uint64_t rows, std::uint64_t cols) noexcept { return rows * (cols / 32U) * 26U; }

// Registers or replaces one copy; its addresses must stay valid until clear().
void register_tensor(const void* original_address, const Tensor& tensor);
void clear() noexcept;
// Registered copy of the tensor that starts at this device address, or nullptr
// (always nullptr when STRIX_Q6P=0).
const Tensor* find(const void* original_address) noexcept;
// STRIX_Q6P=0 keeps every launcher on its original kernel (A/B on one checkpoint).
bool kernels_enabled() noexcept;
std::uint64_t registered_count() noexcept;
// Successful find() calls since start (diagnostics and fixtures).
std::uint64_t hits() noexcept;

// Content identity (32 bytes) of an emulated checkpoint whose dense Q6P32 pack was built
// from the admitted original: checks that admit the original also admit it.
void admit_derived_content(const unsigned char (&identity)[32]) noexcept;
bool derived_content_admitted(const unsigned char* identity) noexcept;

// Host conversions between a Q8_0 tensor (rows x cols, 34-byte blocks) and its Q6P32
// form (q6p32_bytes(rows, cols) bytes). to_q6p32 fails when a code lies outside [-32, 31].
bool to_q6p32(const unsigned char* q8, std::uint64_t rows, std::uint64_t cols, unsigned char* out);
void to_q8_0(const unsigned char* q6, std::uint64_t rows, std::uint64_t cols, unsigned char* out);

}  // namespace strix::q6p
