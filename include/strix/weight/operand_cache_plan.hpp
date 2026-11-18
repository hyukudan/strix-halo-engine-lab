#pragma once
// Resident expanded-operand cache: host planning and per-entry ordering (pure host code, no CUDA).
// Home: shared (host, weight layer). Consumer: src/cuda_core/dense/dense_big.cu (dense_big_gemm), today the only site that
// expands a compressed weight into an MMA operand per call (no such site in the HIP dense or the CUDA MoE paths); any future
// large-M GEMM with a per-call expansion binds this planner instead of adding its own.
//
// Why: the large-M prefill GEMM expands each NVFP4 / q8_0 weight into its MMA operand (FAST8 e4m3, exact f16, q8 bf16 + scales)
// on EVERY call, i.e. once per weight per chunk per slot per request, although the expansion is a pure function of the weight
// bytes. A resident copy expanded once and reused feeds the same bytes to the same kernel, so the outputs are bitwise unchanged;
// only the redundant expansion traffic disappears. Admission is budgeted against free device memory.
#include <algorithm>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::weight::operand_cache {

// off: never. auto: only the operand classes whose per-call expansion was measured as a cost (FAST8 e4m3), under the budget.
// on: every expandable class (also exact f16 / q8 bf16, 2 bytes per element), under the same budget.
enum class Mode : std::uint8_t { off, automatic, on };
// e2m1_tile: FAST4 NVFP4 weight in the W tile staging layout of strix/cuda/core/nvfp4_tile.hpp (a byte permutation of the decode
// records, 0.5625 B per element); same per-call traffic argument as e4m3 (the whole weight re-staged per chunk).
enum class Kind : std::uint8_t { e4m3, f16, q8_bf16, e2m1_tile };

[[nodiscard]] inline const char* mode_name(Mode m) {
    return m == Mode::off ? "off" : m == Mode::on ? "on" : "auto";
}
// Parses off|auto|on; false on anything else (the caller reports the flag).
[[nodiscard]] inline bool parse_mode(const char* s, Mode* m) {
    if (s == nullptr) return false;
    const std::string_view v(s);
    if (v == "off") *m = Mode::off;
    else if (v == "auto") *m = Mode::automatic;
    else if (v == "on") *m = Mode::on;
    else return false;
    return true;
}
[[nodiscard]] inline bool admits(Mode m, Kind k) {
    return m == Mode::on || (m == Mode::automatic && (k == Kind::e4m3 || k == Kind::e2m1_tile));
}

constexpr std::uint64_t kAlign = 256;
constexpr std::uint64_t kNone = ~0ULL;
[[nodiscard]] inline std::uint64_t align_up(std::uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Device bytes of one expanded operand of an out x in weight (layouts of nvfp4_expand_e4m3 / nvfp4_expand_f16 / the q8 expansion:
// bf16 values [out][in] then FP32 scales per 32-block at scales_offset).
[[nodiscard]] inline std::uint64_t scales_offset(Kind k, std::uint32_t out, std::uint32_t in) {
    return k == Kind::q8_bf16 ? align_up(static_cast<std::uint64_t>(out) * in * 2) : kNone;
}
[[nodiscard]] inline std::uint64_t entry_bytes(Kind k, std::uint32_t out, std::uint32_t in) {
    const std::uint64_t e = static_cast<std::uint64_t>(out) * in;
    if (k == Kind::e4m3) return e;
    if (k == Kind::f16) return e * 2;
    if (k == Kind::e2m1_tile) return e / 16 * 9;  // == nvfp4::tile::w_tile_bytes (out % 16 == 0, in % 128 == 0)
    return scales_offset(k, out, in) + e / 32 * 4;
}

// Budget for the arena: free device memory minus a reserve of max(4 GiB, total / 10) kept for activations, KV growth and other
// processes; `cap` > 0 bounds it further. Off -> 0.
[[nodiscard]] inline std::uint64_t budget(Mode m, std::uint64_t free_bytes, std::uint64_t total_bytes, std::uint64_t cap) {
    if (m == Mode::off) return 0;
    const std::uint64_t reserve = std::max<std::uint64_t>(4ULL << 30, total_bytes / 10);
    std::uint64_t b = free_bytes > reserve ? free_bytes - reserve : 0;
    if (cap != 0 && cap < b) b = cap;
    return b;
}

struct Candidate {
    const void* key = nullptr;  // weight data pointer (one entry per weight and kind)
    std::uint64_t bytes = 0;
};
struct Plan {
    std::vector<std::uint64_t> offset;  // per candidate: arena offset, or kNone (not admitted / zero bytes)
    std::uint64_t arena_bytes = 0, wanted_bytes = 0;
    std::uint32_t admitted = 0, rejected = 0, duplicates = 0;
};
// First fit in the given order (the caller passes layer order): an entry is admitted when it still fits in the remaining budget,
// otherwise skipped (later smaller entries may still fit). A key seen before reuses the first offset. Deterministic.
[[nodiscard]] inline Plan plan(const std::vector<Candidate>& c, std::uint64_t budget_bytes) {
    Plan p;
    p.offset.assign(c.size(), kNone);
    std::vector<std::pair<const void*, std::size_t>> seen;
    for (std::size_t i = 0; i < c.size(); ++i) {
        bool dup = false;
        for (const auto& s : seen)
            if (s.first == c[i].key) {
                p.offset[i] = p.offset[s.second];
                ++p.duplicates;
                dup = true;
                break;
            }
        if (dup || c[i].bytes == 0 || c[i].key == nullptr) continue;
        seen.emplace_back(c[i].key, i);
        const std::uint64_t need = align_up(c[i].bytes);
        p.wanted_bytes += need;
        if (p.arena_bytes + need <= budget_bytes) {
            p.offset[i] = p.arena_bytes;
            p.arena_bytes += need;
            ++p.admitted;
        } else {
            ++p.rejected;
        }
    }
    return p;
}

// Ordering of one entry across streams. The first use fills it (the caller enqueues the expansion on its stream, then records the
// entry's event); every later use, on ANY stream including the filling one, waits on that event before reading (a GPU-side
// dependency, no host sync; robust to stream handles being recycled). A failed fill returns the entry to empty.
enum class Use : std::uint8_t { fill, wait };
struct EntryState {
    bool filled = false;
    std::uint32_t fills = 0, waits = 0;
};
[[nodiscard]] inline Use on_use(EntryState& s) {
    if (!s.filled) {
        s.filled = true;
        ++s.fills;
        return Use::fill;
    }
    ++s.waits;
    return Use::wait;
}
inline void on_fill_failed(EntryState& s) { s.filled = false; }

}  // namespace strix::weight::operand_cache
