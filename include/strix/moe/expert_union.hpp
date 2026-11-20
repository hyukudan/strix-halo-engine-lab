#pragma once
// Expert union of a verify window: R <= 8 rows, each with its top-k (<= 16) routed experts. The plan lists the distinct
// experts the rows chose, in first-occurrence order of (row, slot) ascending, and for every row the slot that row holds the
// expert in (no_slot = the row did not choose it). A union-rows kernel reads each entry's expert weights ONCE and computes
// the rows that chose it one after another, each with its own activation, so every (row, slot, output) runs the arithmetic
// of the single-row form: outputs are bitwise those of R single-row calls; only weight reads and launches change.
// Backend-neutral: build() is the host reference of the device builder (tests compare them byte for byte).
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::moe::expert_union {

inline constexpr std::uint32_t max_rows = 8U, max_top_k = 16U, max_assignments = max_rows * max_top_k;
inline constexpr std::uint8_t no_slot = 0xFFU;

// Plan layout in device memory: Header, then max_assignments Entry records (fixed extent; entries >= count are unused).
struct Header {
    std::uint32_t count = 0, rows = 0, top_k = 0, reserved = 0;
};
struct Entry {  // trivial (no member initializers): device kernels keep it in __shared__ memory
    std::uint32_t expert;
    std::uint8_t slot[max_rows];  // slot of this expert in row r, no_slot when row r did not choose it
};
static_assert(sizeof(Header) == 16U && sizeof(Entry) == 12U);
inline constexpr std::uint64_t plan_bytes = sizeof(Header) + std::uint64_t{max_assignments} * sizeof(Entry);

[[nodiscard]] constexpr bool admits(std::uint32_t rows, std::uint32_t top_k) noexcept {
    return rows >= 1U && rows <= max_rows && top_k >= 1U && top_k <= max_top_k;
}

// Host reference. ids: rows x top_k expert ids, row-major. Throws on a shape outside admits(), an id >= experts or an id
// repeated within one row (a router never emits that; the device builder raises its invalid-route status instead).
[[nodiscard]] inline std::vector<Entry> build(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k,
                                              std::uint32_t experts) {
    if (!admits(rows, top_k) || ids.size() != std::size_t{rows} * top_k)
        throw std::invalid_argument("expert union: rows <= 8, top_k <= 16 and rows x top_k ids required");
    std::vector<Entry> out;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t s = 0; s < top_k; ++s) {
            const std::uint32_t e = ids[std::size_t{r} * top_k + s];
            if (e >= experts) throw std::invalid_argument("expert union: expert id out of range");
            std::size_t u = 0;
            while (u < out.size() && out[u].expert != e) ++u;
            if (u == out.size()) {
                Entry fresh{};
                fresh.expert = e;
                std::memset(fresh.slot, no_slot, sizeof(fresh.slot));
                out.push_back(fresh);
            }
            if (out[u].slot[r] != no_slot) throw std::invalid_argument("expert union: expert repeated within one row");
            out[u].slot[r] = static_cast<std::uint8_t>(s);
        }
    }
    return out;
}

}  // namespace strix::moe::expert_union
