#pragma once

#include "strix/spec/block_drafter.hpp"
#include "strix/spec/probe_policy.hpp"
#include "strix/spec/row_rule.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

// One construction of a request's BlockDrafter, shared by every binder of the speculator (the serial generation
// command line, the cohort decoder's lone-lane speculation): the engine's data, the measured probe loss, the target's
// window, the model drafter, the caller's prices, learned walls, the row rule and the engine's seeds, in that order.
// The caller keeps what is its own around it (cost tables, diagnostic prices, verify modes) through the hooks.
namespace strix::spec {

struct BlockSetup {
    std::uint32_t max_rows = 0;        // the target's widest verify window (anchor + drafts): BlockConfig::max_block
    std::string_view engine, backend;  // probe-loss keys (apply_probe_loss)
    // The engine's data (arbiter windows, the copy + chain hybrid, chain seeds), before the request's settings.
    // Null: none.
    std::function<void(BlockConfig&)> tune;
    // After the window is set: the request's mode fields (e.g. force_lookup), then the model drafter (nullptr = lookup
    // only; the hybrid is then off). Null: no model drafter.
    std::function<ChainDrafter*(BlockConfig&)> chain;
    // The caller's price policy (fixed or frozen prices) before the learned-wall rule. Null: learned prices.
    std::function<void(BlockConfig&, ChainDrafter*)> prices;
    RowRule row_rule{};
    // The engine's (or the caller's) cost seeds after construction and the row rule. Null: none.
    std::function<void(BlockDrafter&)> seed;
};

// Builds the drafter into `drafter` and returns it. Throws std::invalid_argument when the window exceeds the arbiter's.
inline BlockDrafter& make_block_drafter(std::optional<BlockDrafter>& drafter, const BlockSetup& setup) {
    BlockConfig config;
    config.hybrid = false;  // The optional chain owns a complete, independent block (an engine may opt in).
    if (setup.tune) setup.tune(config);
    apply_probe_loss(config.arbiter, setup.engine, setup.backend);
    if (config.arbiter.windows.empty() || setup.max_rows > config.arbiter.windows.back())
        throw std::invalid_argument("target exceeds the supported verify row count");
    config.max_block = setup.max_rows;
    ChainDrafter* const chain = setup.chain ? setup.chain(config) : nullptr;
    if (!chain) config.hybrid = false;
    if (setup.prices) setup.prices(config, chain);
    // Learned walls: the first pass of a width can carry its one-time work (graph capture), never a price.
    if (!config.fixed_costs) config.arbiter.cold_walls = 1;
    drafter.emplace(std::move(config), chain);
    // Before seeding: switching the drafter to sampled rows resets its learned observations.
    drafter->set_row_rule(setup.row_rule);
    if (setup.seed) setup.seed(*drafter);
    return *drafter;
}

}  // namespace strix::spec
