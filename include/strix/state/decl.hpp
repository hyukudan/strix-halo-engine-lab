#pragma once

#include "strix/pair/state_owners.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string_view>

namespace strix::graph { struct LayerVisit; struct ModelDecl; }

namespace strix::state {

enum class Kind : std::uint8_t {
    kv_dense, kv_window, kv_compressed, index_pool, recurrent, taps,
    history, cache_rows, draft,  // cache_rows includes position-addressed residual banks
};
enum class Dtype : std::uint8_t {
    f32, bf16, f16, e4m3_shadow, i8_token, i8_g64, i4_had, trellis, fp8_block, fp4_block, u32,
};
enum class Rollback : std::uint8_t { rewind, snapshot, recompute, replay };
enum class Per : std::uint8_t { model, layer, visit };
enum class Absent : std::uint8_t { required, recompute };
using Owner = pair::state::Owner;

struct Shadow {
    Dtype dtype{};
    std::uint32_t planes{};
    std::uint32_t plane_bytes{};
};

struct Canonical {
    Dtype dtype{};
    std::string_view layout{};  // empty: use the declared master image
};

struct Decl {
    std::string_view id;
    Kind kind{};
    Per per{};
    Dtype dtype{};
    std::uint32_t dims[3]{};      // contiguous first, trailing zero axes omitted
    std::uint32_t planes = 1;
    std::uint32_t window{}, ratio = 1, group = 1;
    Rollback rollback{};
    std::string_view when{};
    std::int32_t producer = -1; // storage shared with this physical layer of the same declaration
    Owner owner = Owner::per_rank;
    std::uint32_t block = 1;
    std::uint32_t plane_bytes{}; // explicit image bytes per plane, required for packed formats
    Shadow shadow{};
    Canonical canonical{};
    Absent absent{};
    bool position_range{};      // position-sharded per-rank ownership
    std::uint32_t replay_input_bytes{};
    std::uint32_t checkpoint_interval{};  // zero: pure replay, otherwise checkpoints every N rows
    std::uint32_t row_alignment = 4;
};

struct Layout;
struct LayoutOptions {
    std::uint32_t max_pending = 1;
    std::uint64_t dense_slot_budget = std::numeric_limits<std::uint64_t>::max();
    // When supplied, resolves the predicate on the actual visit. Without it,
    // the built-in D1 physical layer traits are available by name.
    std::function<bool(std::string_view, const graph::LayerVisit&)> matches{};
};

[[nodiscard]] Layout layout(std::span<const Decl>, const graph::ModelDecl&, std::uint64_t context,
                            std::uint32_t slots, const LayoutOptions& = {});

}  // namespace strix::state
