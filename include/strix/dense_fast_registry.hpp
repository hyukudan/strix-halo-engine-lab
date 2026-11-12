#pragma once

// Dense FAST pack registry (--dense-fast-pack): 4/5-bit K-quant copies of dense Q8_0
// matrices, re-quantised offline (tools/dense_fast_quantize.py). Unlike a Q6P32 copy, a
// copy here holds different values than its Q8_0 tensor: reading it is a FAST numerical
// identity of its own and never the default.
//
// The registry maps the device address of a Q8_0 tensor to its copy and the copy's ggml
// type. The dense A8 decode windows of 1..8 rows (q8_rows::launch_dense_a8 and
// launch_dense_a8_quantized) look their Q8_0 segments up here and run the typed dense A8
// kernel on the copies; every other kernel keeps the Q8_0 tensor.

#include <cstdint>

namespace strix::dense_fast {

struct Tensor {
    const unsigned char* base = nullptr;  // device address of the copy (GGUF block layout of `type`)
    std::uint32_t rows = 0, cols = 0;     // outputs x input width
    std::uint32_t type = 0;               // ggml type of the copy
};

// Registers or replaces one copy; its address must stay valid until clear().
void register_tensor(const void* original_address, const Tensor& tensor);
void clear() noexcept;
// Registered copy of the Q8_0 tensor that starts at this device address for a window of
// `window_rows` rows, or nullptr (always nullptr when STRIX_DENSE_FAST=0 or the row count
// is not admitted by STRIX_DENSE_FAST_ROWS).
const Tensor* find(const void* original_address, unsigned window_rows) noexcept;
// STRIX_DENSE_FAST=0 keeps every launcher on the Q8_0 tensors (A/B with the pack loaded).
bool kernels_enabled() noexcept;
// STRIX_DENSE_FAST_ROWS: bit r set admits windows of r rows. Default 0x6 (1 and 2 rows), the sizes where
// the typed kernel measured faster than the Q8_0 forms; 0x1fe admits every size up to 8 rows.
bool rows_admitted(unsigned window_rows) noexcept;
std::uint64_t registered_count() noexcept;
// Successful find() calls and the Q8_0 bytes they did not read, since start.
std::uint64_t hits() noexcept;
std::uint64_t replaced_q8_bytes() noexcept;
std::uint64_t read_bytes() noexcept;

}  // namespace strix::dense_fast
