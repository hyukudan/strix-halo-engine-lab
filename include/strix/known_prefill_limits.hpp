#pragma once

#include <cstdint>
#include <string>

namespace strix {
// Largest T of one tiled known-prompt prefill block (--known-prefix-prefill-rows). Every T-row
// prefill operator works in row groups or 64-bit row strides; the routed MoE CSR of T * top_k
// assignments is built by one CTA (its wave counters bound T * top_k, see build_prefill_csr_waves).
inline constexpr std::uint32_t known_prefill_max_rows = 8192;
inline std::string known_prefill_rows_range() { return "16.." + std::to_string(known_prefill_max_rows); }
} // namespace strix
