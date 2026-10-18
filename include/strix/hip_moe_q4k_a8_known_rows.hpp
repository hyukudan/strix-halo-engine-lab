#pragma once
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/q51_known_rows.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace strix::moe_q4k_a8::known_rows {
inline constexpr std::string_view organization_identity =
    "a8-csr-gate-up-v1:shared-packed-weights:independent-row-pair4";
// Source-declared staging arrays only; compiled LDS/register/private usage
// remains a compiler qualification result. Geometry is bounded before products.
[[nodiscard]] inline std::uint64_t staged_activation_bytes(std::uint32_t rows) {
    if(rows!=2U&&rows!=4U&&rows!=8U)throw std::invalid_argument("A8 CSR staging requires R2/R4/R8");
    return static_cast<std::uint64_t>(rows)*(max_hidden+2ULL*(max_hidden/32U)*sizeof(std::uint32_t));
}
// Per-row Bindings retain the exact R1 views and numerical contract. Quantize
// each row once before this entry. CSR must be the complete stable (row,slot)
// grouping in q51_known_rows::Workspace; repeated row/expert slots are distinct.
// Validates metadata only, never reads host payloads or changes device bytes.
void preflight(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, hipStream_t);
// One bounded device CSR admission, then packed gate/up loads shared across
// each <=R assignment tile. Both output chains retain the incumbent FP order.
// No quantization, clear, allocation, synchronization, SiLU or publication.
// Every owner/descriptor and the explicit eager stream remain retained and
// exclusively leased until quiescence, including after any enqueue exception.
void enqueue(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, hipStream_t);
// Device tables of the window prologue (probe TU, shares the quantizer body).
struct CsrRows { unsigned *summary{}, *offsets{}, *touched{}, *rows{}, *slots{}, *experts{}, *begins{}; };
struct PrologueRows {
    const void* input[8]{}; const unsigned* routes[8]{}; const float* route_weights[8]{};
    signed char* codes[8]{}; float* deltas[8]{}; int* sums[8]{}; unsigned* status[8]{}; float* clear[8]{};
    CsrRows csr;
};
// One CTA per row zeroes clear[r][0..clear_words) and status, validates routes
// and weights and quantizes (the bytes of the per-row clear/validate/quantize
// sequence). A concurrent CTA builds the stable CSR from the routes alone; on
// an invalid route it writes the fail-closed summary instead. Quantizer
// failures are only in row status: consumers must check status as well.
void enqueue_prologue(const Request&, const PrologueRows&, unsigned rows, unsigned clear_words, hipStream_t);
// Window-owned rows: route weights and the per-row FP32 region zeroed by the
// prologue. down_type 7/8 selects the A8 profile (Q5_1/Q8_0 A8G32 down) and
// then needs each row's serial down image (codes/deltas/sums) and raw slot dots.
struct WindowRows {
    std::array<DeviceBytes, 8> route_weights{}, clear{};
    std::uint32_t down_type = 0;
    DeviceBytes down_weights{};
    std::array<DeviceBytes, 8> down_codes{}, down_deltas{}, down_sums{}, rawdots{};
};
// Complete window producer: one prologue launch (rows plus the device CSR
// build into the workspace: complete stable (row,slot) grouping, so no
// separate admission), then shared gate/up with the finite scans and SiLU
// fused per element. Leaves each row's slot workspace exactly as quantize/
// project/scan/SiLU/scan would on success; any failure sets the same local
// status bits. After an early (route/quantizer) failure the caller restores
// the fail-closed summary (touched, batches, histograms zero). A8 profile
// (down_type 7/8, Q4_K or Q5_K gate/up): projection plus the serial down
// activation quantizer, then the shared integer down writes each row's raw
// slot dots exactly as down_activation_quantize/down_rawdots would; the
// caller merges and commits.
void enqueue_window(const Request&, std::span<const Bindings>,
    const q51_known_rows::Workspace&, const WindowRows&, hipStream_t);

// Known-prompt prefill over T rows (multiple of 8, A8 profile only). Every per-row
// view is row 0 plus r times its own byte stride (the prefill arena stripes rows).
template <class T> struct StridedRows { T* base{}; std::uint64_t stride{}; };
struct PrefillRows {
    StridedRows<const void> input; StridedRows<const unsigned> routes; StridedRows<const float> route_weights;
    StridedRows<signed char> codes; StridedRows<float> deltas; StridedRows<int> sums; StridedRows<unsigned> status;
    StridedRows<float> output, clear;
    StridedRows<signed char> down_codes; StridedRows<float> down_deltas; StridedRows<int> down_sums;
    StridedRows<float> rawdots;
    CsrRows csr; // q51_known_rows::scratch_layout(T, experts, top_k, true) in an exclusive allocation
};
// T CTAs: each row's clear, route checks and quantizer exactly as the window prologue.
void enqueue_prefill_prologue(const Request&, const PrefillRows&, unsigned rows, unsigned clear_words, hipStream_t);
// Prologue, GPU CSR over the T*top_k assignments (stable: rows then slots ascending per
// expert, batches of consecutive assignments of one expert; row statuses folded into the
// summary word), gate/up with the serial down activation quantizer and the integer down.
// Each (row, slot) runs the window kernels' per-lane arithmetic; the caller merges and
// commits. Summary histograms count n>=8 in their last bucket. form 1 runs gate/up and down
// as iu8 WMMA kernels over batches of <= 16 assignments, form 0 the dot4 kernels, -1 picks by
// STRIX_PREFILL_MOE_WMMA; every output byte is identical across forms.
void enqueue_prefill(const Request&, const PrefillRows&, unsigned rows, unsigned clear_words,
    const unsigned char* gate, const unsigned char* up, unsigned down_type, const unsigned char* down_weights, hipStream_t,
    int form = -1);
} // namespace strix::moe_q4k_a8::known_rows
