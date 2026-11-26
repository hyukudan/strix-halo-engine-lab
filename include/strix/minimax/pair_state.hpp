#pragma once
// MiniMax M3.1 prompt-state declaration (models/minimax-m3.1-flash.toml [pair] with the state table).
// Bound from the model TOML through the generated constants (strix-model-gen); the shared planner is
// strix/pair/state_owners.hpp. The declaration is registered in namespace strix::pair::state::decl next to the
// other families (strix/pair/state_decls.hpp).
#include <array>
#include <cstddef>
#include <cstdint>

#include "strix/model/bind.hpp"
#include "strix/model_data/minimax_m3_1_flash.hpp"
#include "strix/pair/state_owners.hpp"

namespace strix::pair::state::decl {

// MiniMax M3.1, TP2 (2 KV heads per rank), 60 layers, rows = P = N - 1 (the last prompt token runs on the ranks).
// K/V are per rank, the index keys IK are the same bytes for both ranks (mm_rtx_kernels.cu:632-639 copies the one IK
// into both rank blocks): shared. Today 153,600 B/row (both ranks get IK), one copy 138,240 B/row.
// The numbers are models/minimax-m3.1-flash.toml [pair] with the state table, read through the generated constants.
namespace minimax_data = ::strix::model_data::minimax_m3_1_flash::pair;
inline constexpr std::array<Piece, minimax_data::state::count> minimax_pieces = pieces_of(
    minimax_data::state::name, minimax_data::state::kind, minimax_data::state::owner, minimax_data::state::extent,
    minimax_data::state::bytes, minimax_data::state::layers, minimax_data::state::layer0,
    minimax_data::state::layer_stride, minimax_data::state::row_div, minimax_data::state::rows_less,
    minimax_data::state::source);
inline constexpr Family minimax{minimax_data::family, minimax_data::rows_of, minimax_pieces,
                                ::strix::model::narrow<std::uint32_t>(minimax_data::world),
                                ::strix::model::narrow<std::uint32_t>(minimax_data::block_rows),
                                minimax_data::prefix_rows_invariant, minimax_data::shared_once_default,
                                minimax_data::shared_once_source};


// Former literals, pinned field by field: a drifting TOML row is a build error, not a silent state-size change.
namespace previous_minimax {
constexpr std::array<Piece, 3> pieces() {
    return {{
        {"k", 0, Owner::per_rank, Extent::rows, 512, 60, 0, 1, 1, 0, "src/minimax/engine/engine.hip:3193-3207"},
        {"v", 1, Owner::per_rank, Extent::rows, 512, 60, 0, 1, 1, 0, "src/minimax/engine/engine.hip:3193-3207"},
        {"ik", 2, Owner::shared, Extent::rows, 256, 60, 0, 1, 1, 0, "src/minimax/rtxpf/cuda/mm_rtx_kernels.cu:639"},
    }};
}
inline constexpr Family family{"minimax-m3.1", "P = N - 1", {}, 2, 64, false, false,
                               "opt-in pending the pair receipt"};
}  // namespace previous_minimax
static_assert(minimax_pieces.size() == 3 && equal(minimax_pieces, previous_minimax::pieces()),
              "MiniMax pair.state differs from the declaration it replaced");
static_assert(equal(minimax, previous_minimax::family), "MiniMax [pair] differs from the declaration it replaced");
static_assert(minimax.pieces.data() == minimax_pieces.data());

}  // namespace strix::pair::state::decl
