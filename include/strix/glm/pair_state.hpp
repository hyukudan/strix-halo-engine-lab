#pragma once
// GLM-5.3 prompt-state declaration (models/glm-5.3-flash.toml [pair] with the state table).
// Bound from the model TOML through the generated constants (strix-model-gen); the shared planner is
// strix/pair/state_owners.hpp. The declaration is registered in namespace strix::pair::state::decl next to the
// other families (strix/pair/state_decls.hpp).
#include <array>
#include <cstddef>
#include <cstdint>

#include "strix/model/bind.hpp"
#include "strix/model_data/glm_5_3_flash.hpp"
#include "strix/pair/state_owners.hpp"

namespace strix::pair::state::decl {

// GLM-5.3 pair: 34 KDA layers with 64 heads split 32 + 32 (per rank, FP32 S and conv), 11 DSA + 1 MTP layer whose
// latent / index key / gate / pooled keys are the same bytes on both ranks (runtime.cu:660 one download): shared.
// Component ids = glm_import.hpp:139. Pool: one FP32 128-vector per closed 4-row block. rows = P = N - 1.
// The numbers are models/glm-5.3-flash.toml [pair] with the state table.
namespace glm_data = ::strix::model_data::glm_5_3_flash::pair;
inline constexpr std::array<Piece, glm_data::state::count> glm_pieces = pieces_of(
    glm_data::state::name, glm_data::state::kind, glm_data::state::owner, glm_data::state::extent,
    glm_data::state::bytes, glm_data::state::layers, glm_data::state::layer0, glm_data::state::layer_stride,
    glm_data::state::row_div, glm_data::state::rows_less, glm_data::state::source);
// v2 ("DSA once", gs::dsa_relay) is the shipped relay of the same bytes; default ON since the v1 / v2 rank-1 payload
// equal-bits receipt: rank 1 merged v2 payload == its v1 payload (tests/glm_serve_cpu_tests.cpp "dsa relay protocol").
inline constexpr Family glm{glm_data::family, glm_data::rows_of, glm_pieces,
                            ::strix::model::narrow<std::uint32_t>(glm_data::world),
                            ::strix::model::narrow<std::uint32_t>(glm_data::block_rows), glm_data::prefix_rows_invariant,
                            glm_data::shared_once_default, glm_data::shared_once_source};


// Former literals, pinned field by field: a drifting TOML row is a build error, not a silent state-size change.
namespace previous_glm {
constexpr std::array<Piece, 6> pieces() {
    return {{
        {"kda_s", 0, Owner::per_rank, Extent::fixed, 32U * 128U * 128U * 4U, 34, 0, 1, 1, 0, "src/glm/kernels/glm_tier_a.hip:153-190"},
        {"kda_conv", 1, Owner::per_rank, Extent::fixed, 3U * 32U * 128U * 3U * 4U, 34, 0, 1, 1, 0, "src/glm/kernels/glm_tier_a.hip:153-190"},
        {"dsa_latent", 2, Owner::shared, Extent::rows, 1024, 12, 0, 1, 1, 0, "src/glm/kernels/glm_dsa.hip:54-87"},
        {"dsa_ikey", 3, Owner::shared, Extent::rows, 256, 12, 0, 1, 1, 0, "src/glm/kernels/glm_dsa.hip:120-153"},
        {"dsa_igate", 4, Owner::shared, Extent::rows, 256, 12, 0, 1, 1, 0, "src/glm/kernels/glm_dsa.hip:120-153"},
        {"dsa_pool", 5, Owner::shared, Extent::rows, 512, 12, 0, 1, 4, 0, "src/glm/kernels/glm_dsa.hip:167-180"},
    }};
}
inline constexpr Family family{"glm-5.3", "P = N - 1", {}, 2, 64, false, true,
                               "glm v1/v2 rank payload equal bits (host relay fixture)"};
}  // namespace previous_glm
static_assert(glm_pieces.size() == 6 && equal(glm_pieces, previous_glm::pieces()),
              "GLM pair.state differs from the declaration it replaced");
static_assert(equal(glm, previous_glm::family), "GLM [pair] differs from the declaration it replaced");
static_assert(glm.pieces.data() == glm_pieces.data());

}  // namespace strix::pair::state::decl
