#pragma once
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/q51_known_rows.hpp"
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
} // namespace strix::moe_q4k_a8::known_rows
