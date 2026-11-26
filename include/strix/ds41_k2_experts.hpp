#pragma once

// deepseek41 routed experts in the EXL3 K2 trellis form (FAST, bit-changing; identity ds41-expert-k2-v1).
//
// EXL3 K2 pack: a root directory with one directory per layer, L<l>/x.L<l>.c<chunk>.safetensors
// (or the chunk files directly in the root), holding per expert e and matrix w1 (gate), w3 (up), w2 (down) the EXL3
// quadruple layers.<l>.ffn.experts.<e>.<w>.{trellis I16 [I/16][N/16][16 K], suh F16 [I], svh F16 [N], mul1 I32}. It is read
// with the deepseek41 trellis pack reader (ds41_trellis.hpp TrellisPack: geometry, mul1 marker, 128-aligned slices).
//
// Device form: the shared trellis MoE family (src/moe_core/exl3_trellis.hpp, expert_split_rows with Bits = 2); the tile
// bytes are used as stored. Two-rank expert TP splits the intermediate dimension like ds41-expert-tp2-i1152-v1: rank r
// holds gate/up output rows and down input columns [r * I / ranks, (r + 1) * I / ranks). The split policy is the single
// node one (moe_core::vector_splits of the whole expert), and a rank runs only the down splits inside its k range, so
// every gate/up output and every down split partial is the same as on one node; the down finish adds the partials of
// both ranks in ascending split order (split_plan below checks that the rank boundary is a split boundary).

#include "strix/lever.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41 {
class TrellisPack;
}

namespace strix::ds41::k2 {

inline constexpr const char* identity = "ds41-expert-k2-v1";

// Candidate FAST lever declaration. The standalone K2 fixtures use the loader directly; production does not yet bind
// this switch. A model binding needs the full gate v2 verdict and pair timing.
inline constexpr lever::Decl decode_k2_experts_lever{
    "decode_k2_experts", "STRIX_DS41_K2_EXPERTS", lever::Exactness::bit_changing, false, {}, {},
    "FAST routed experts EXL3 K2 (2.01 bpw, -10.7 % expert bytes, -8.2 GB/node): pilot dKL -0.0016 on 4 layers; gate v2 on the "
    "40-layer pack and pair R1/R6/N8 timing pending"};
inline constexpr const char* pack_env = "STRIX_DS41_K2_PACK";

struct Geometry {
    std::uint32_t hidden = 5120;  // H: gate/up input, down output
    std::uint32_t inter = 2304;   // I: gate/up output, down input
    std::uint32_t experts = 384;
    std::uint32_t bits = 2;
};

enum class Mat : std::uint8_t { gate, up, down };  // pack w1, w3, w2
[[nodiscard]] const char* mat_suffix(Mat m) noexcept;
// layers.<layer>.ffn.experts.<expert>.<w1|w3|w2>
[[nodiscard]] std::string pack_name(std::uint32_t layer, std::uint32_t expert, Mat m);

// The intermediate range [first, first + count) a rank holds (128-aligned; throws otherwise).
struct RankSlice {
    std::uint32_t rank = 0, ranks = 1;
    std::uint32_t first = 0, count = 0;
};
[[nodiscard]] RankSlice rank_slice(const Geometry& g, std::uint32_t rank, std::uint32_t ranks);

// Split counts of the whole expert (shared policy moe_core::vector_splits) and the down splits a rank runs.
struct SplitPlan {
    std::uint32_t gate_splits = 0;   // over H, for gate and up
    std::uint32_t down_splits = 0;   // over I, for down
    std::uint32_t down_first = 0;    // the rank's down splits [down_first, down_end)
    std::uint32_t down_end = 0;
    std::uint32_t down_k_tile0 = 0;  // first k tile (of I / 16) the rank holds
};
// Throws if a split of the rank's range would cross the rank boundary (then the rank partials would not be single-node bits).
[[nodiscard]] SplitPlan split_plan(const Geometry& g, const RankSlice& s);

// Host image of one layer's routed experts for one rank, in device order: per matrix, expert e's tile bytes at
// e * <matrix>_tile_bytes, suh / svh F16 bits at e * length.
struct LayerImage {
    std::uint32_t layer = 0, expert_first = 0, expert_count = 0;
    Geometry geometry{};
    RankSlice slice{};
    std::uint64_t gate_tile_bytes = 0;  // per expert, gate and up (H x slice.count)
    std::uint64_t down_tile_bytes = 0;  // per expert (slice.count x H)
    std::vector<std::uint8_t> gate, up, down;
    std::vector<std::uint16_t> gate_suh, up_suh, down_suh;  // H, H, slice.count per expert
    std::vector<std::uint16_t> gate_svh, up_svh, down_svh;  // slice.count, slice.count, H per expert
    [[nodiscard]] std::uint64_t bytes() const noexcept;
};

// Bytes one rank holds per layer for `experts` experts (trellis + suh + svh of the three matrices).
[[nodiscard]] std::uint64_t layer_bytes(const Geometry& g, const RankSlice& s, std::uint32_t experts);

// A K2 pack root. Layer l reads <root>/L<l> when that directory exists, else <root> (one directory for all layers).
class Pack {
public:
    explicit Pack(std::filesystem::path root, Geometry g = {});
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;
    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }
    [[nodiscard]] const Geometry& geometry() const noexcept { return geometry_; }
    // Experts [first, first + count) of `layer` for the rank; count 0 = to the last expert. Throws on a missing tensor,
    // a K other than geometry().bits, a geometry other than (H, I) or a codebook other than mul1.
    [[nodiscard]] LayerImage load_layer(std::uint32_t layer, const RankSlice& s, std::uint32_t first = 0, std::uint32_t count = 0);

private:
    TrellisPack& layer_pack(std::uint32_t layer);
    std::filesystem::path root_;
    Geometry geometry_;
    std::map<std::filesystem::path, std::unique_ptr<TrellisPack>> packs_;
};

// Announced identity of a configuration: ds41-expert-k2-v1 plus the TP form, e.g. "ds41-expert-k2-v1:tp2".
[[nodiscard]] std::string step_identity(std::uint32_t ranks);

}  // namespace strix::ds41::k2
