#pragma once
// Gemma 4 two-node pipeline prefill (opt-in STRIX_G4_PAIR=1; see engine/pair.hpp for the route).
// Stage A (rank 0) runs layers [0, cut) of each prompt chunk, stage B (rank 1) layers [cut, layers) and
// everything after them, on the shared two-slot seam (strix/pair/pipe.hpp) while A computes the next
// chunk. Per-layer arithmetic is unchanged: B receives A's exact residual, its fused next-layer input norm,
// the taps rows and the unpublished K/V overlays of A's layers, and commits them like its own, so B's
// committed stream equals the single-node one byte for byte.
#include "strix/pair/pipe.hpp"
#include "gemma4/contracts/g4_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::gemma4::prefill {
struct PairStage {
    strix::pair::Pipe* pipe = nullptr; // null: single node (no seam work at all)
    unsigned rank = 0, cut = 0;
    std::uint64_t sequence = 0;   // next chunk of this process
    std::uint64_t identity = 0;   // model + profile + build word both ranks must share
    void* queue = nullptr;        // compute queue of the pass in flight (staging copies run on it)
    double wait_ms = 0;           // stage B: time blocked on the next seam frame (A slower or wire)
    double seal_ms = 0;           // stage A: snapshot fence + slot wait (B slower or wire)
    std::vector<double> chunk_ms; // per chunk: pass wall time minus stage B's seam wait (this stage's own work)
    // STRIX_G4_PAIR_BREAKDOWN=1 (diagnostic; event timing serialises the side queue): cumulative
    // linear / attention / other device ms of this stage's passes.
    double linear_ms = 0, attention_ms = 0, other_ms = 0;
};
inline PairStage pair_stage; // one pipeline per process
inline constexpr std::uint64_t pair_seam_magic = 0x4734504149523031ULL; // "G4PAIR01"
// Header words (strix/pair/pipe.hpp owns [1], [40..42]).
struct PairWord {
    static constexpr std::size_t magic = 0, start = 2, rows = 3, cut = 4, identity = 5, tokens = 6, x_ready = 7, bytes = 8;
};
// Seam bytes of one chunk: residual and next input norm rows, the five tap columns, and the K/V overlay
// rows of every KV-producing layer below the cut.
inline std::size_t pair_seam_bytes(const ModelConfig& c, unsigned cut, unsigned rows) {
    std::size_t floats = std::size_t(rows) * c.hidden * (2 + 5);
    for (unsigned l = 0; l < cut && l < c.layers; ++l)
        if (c.layer[l].producer == l) floats += 2ULL * rows * c.kv_heads * c.layer[l].head_dim;
    return floats * sizeof(float);
}
} // namespace strix::gemma4::prefill
