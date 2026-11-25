#pragma once
// Backend-neutral plan data of the grouped rotated A4W4 expert class (contract and HIP body: moe_core/grouped_iu4a4.hpp).
// The grouped dispatcher carries these on every backend; only a backend with the class's hook reads them.
#include "moe_core/grouped_microscaled_rows.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::moe_core::grouped_iu4a4 {

using grouped_microscaled::TileRows;
inline constexpr const char* kArithmeticId = "grouped-iu4a4-rot-v1";
inline constexpr unsigned kInputBlock = 128U;  // rotation block / weight group of dense-iu4a4-rot-v1

// Resident image of one projection of a layer's experts: expert e at codes + e * code_bytes, steps + e * step_floats.
struct ExpertImage {
    std::uint8_t* codes{};
    float* steps{};
    std::size_t code_bytes{};   // per expert, >= image_code_bytes(in, n)
    std::size_t step_floats{};  // per expert, >= image_step_floats(in, n)
    unsigned experts{};         // expert slots in the image (a tile's expert index must be below it)
};
// Rotated images of one layer's experts (both projections; gate/up in the paired column order) and the code address
// rule of the resident matrices they were built from (expert_codes_strided).
struct LayerImages {
    ExpertImage gate_up{}, down{};
    const std::uint8_t* gate_up_codes{};
    std::size_t gate_up_stride{};
    const std::uint8_t* down_codes{};
    std::size_t down_stride{};
};
// Trait value of a grouped expert launch (DevicePlan::iu4): images == nullptr = off (the BF16 tile).
struct Selection {
    const LayerImages* images{};
};
// Bytes of the two activation images of one pass (token rows x hidden, assignment rows x intermediate), 256 B aligned
// each; the HIP pass refuses a smaller workspace.
constexpr std::size_t align256(std::size_t v) { return (v + 255U) / 256U * 256U; }
constexpr std::size_t act_image_bytes(unsigned rows, unsigned in) {
    const std::size_t mp = (std::size_t(rows) + 15U) / 16U * 16U;
    return align256(mp * in / 2U) + align256(mp * (in / kInputBlock) * 4U) + align256(mp * 4U);
}
constexpr std::size_t act_bytes(unsigned rows, unsigned assignments, unsigned hidden, unsigned intermediate) {
    return act_image_bytes(rows, hidden) + act_image_bytes(assignments, intermediate);
}
// One grouped expert pass of the class (the backend-neutral dispatcher fills it; plan, tiles and outputs as the BF16 tile):
// token rows (FP32 rows x hidden) -> A4 token image; gate/up tiles (rows through assignment_rows) with the paired SwiGLU
// epilogue -> h (FP32 [assignment][intermediate]); h -> A4 image; down tiles -> y (FP32 [assignment][hidden], the
// combine's rows). Both destinations are derived from the admitted base, pitch and assignment interval of a tile; the
// tiles' own output fields are not read.
struct Pass {
    const float* tokens{};
    unsigned rows{}, assignments{}, hidden{}, intermediate{};
    const std::uint32_t* assignment_rows{};
    const TileRows* gate_up{};
    const TileRows* down{};
    unsigned tiles{}, max_tile_rows{};
    const unsigned* tile_count{};
    const unsigned* failure{};
    const std::uint16_t* index_base{};  // the base the gate/up tiles' input addresses are relative to (assignment a at a)
    float* h{};
    std::uint8_t* act{};
    std::size_t act_size{};
    const LayerImages* images{};
    float gate_cap{}, up_cap{};
    // Verdict word of the pass (required). A validation launch, complete before any compute launch, copies a planner
    // failure into it and sets kFaultBounds when an active tile names an expert outside its image or an assignment
    // range past `assignments`, when the active tiles do not cover the assignments exactly once (a short or zero device
    // count), or when a map entry names a token past the token image. The compute launches only read it (one value for
    // every wave of a launch) and store nothing when it is set. The dispatcher passes the plan's failure word, so the
    // combine skips and the step fails as on a planner failure.
    unsigned* fault{};
    float* y{};  // down destination: assignment a at y + a * hidden
};
inline constexpr unsigned kFaultBounds = 1U << 30U;

}  // namespace strix::moe_core::grouped_iu4a4
