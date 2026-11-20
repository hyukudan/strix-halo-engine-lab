#pragma once
// Server start warm-up: one synthetic prompt through the engine's own request path before the server listens.
//
// A server's request path grows device memory on first use: prefill scratch per row class, route tables, draft-head
// buffers. Without a warm-up the FIRST request of each row class pays those allocations on its time to first token
// (on a unified-memory node each allocation reclaims page cache: tens to hundreds of milliseconds each) and every
// repeat of the same size does not. The warm-up runs ONE prompt of the server's prefill tile T (its largest prefill
// unit: the scratch only grows, so every later prompt whose units are at most T rows fits) and one output token, keeps
// nothing (no prefix store, no drafting, no conversation), and is timed as its own start phase.
//
// Exactness: later requests run the same arithmetic on the same buffers (bitwise lever); the cost is one T-row
// prompt at start. Not covered: growth that depends on the context position a prompt reaches (prompts longer than T
// reach positions the warm-up did not); a full-context warm-up would cost a full-context prefill at every start. Nor
// data that depends on the token ids: file-backed per-token rows the memory cannot keep resident are still read on
// the first request that uses those tokens.
// STRIX_SERVE_WARMUP=0 keeps the lazy start.

#include "strix/lever.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

inline constexpr lever::Decl startup_warmup_lever{
    "serve-startup-warmup", "STRIX_SERVE_WARMUP", lever::Exactness::bitwise, true, {}, {},
    "server start, first-use request-path scratch sized before listening (docs/toggles.md)"};

[[nodiscard]] inline bool startup_warmup_enabled() { return lever::select(startup_warmup_lever, {}).on; }

struct StartupWarmupPlan {
    std::uint32_t rows = 0;      // prompt rows of the warm-up (0: no warm-up)
    std::uint32_t outputs = 0;   // generated tokens of the warm-up
    std::string_view reason{};   // why there is no warm-up, or "tile"/"context" for what bounded the rows
};

// rows = min(T, context - 1 - outputs): a prompt must leave room for its output tokens inside the context.
[[nodiscard]] constexpr StartupWarmupPlan plan_startup_warmup(bool enabled, std::uint64_t tile_rows,
                                                              std::uint64_t context_tokens, std::uint32_t outputs = 1U) {
    if (!enabled) return {0U, 0U, "off"};
    if (tile_rows == 0U) return {0U, 0U, "no prefill tile"};
    const std::uint64_t room = context_tokens > std::uint64_t{outputs} + 1U ? context_tokens - outputs - 1U : 0U;
    if (room == 0U) return {0U, 0U, "no context room"};
    const std::uint64_t rows = std::min({tile_rows, room, std::uint64_t{UINT32_MAX}});
    return {static_cast<std::uint32_t>(rows), outputs, rows == tile_rows ? "tile" : "context"};
}

// Deterministic synthetic ids spread over [1, vocabulary / 2): ordinary tokens (the special and control tokens of the
// supported vocabularies sit at the top), varied so a mixture-of-experts router visits many experts.
[[nodiscard]] inline std::vector<std::uint32_t> startup_warmup_prompt(std::uint32_t rows, std::uint64_t vocabulary) {
    std::vector<std::uint32_t> ids(rows);
    const std::uint64_t span = std::max<std::uint64_t>(vocabulary / 2U, 2U) - 1U;
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    for (auto& id : ids) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id = static_cast<std::uint32_t>(1U + (s >> 33U) % span);
    }
    return ids;
}

[[nodiscard]] inline std::string startup_warmup_line(std::string_view who, const StartupWarmupPlan& plan, double ms) {
    std::string out(who);
    out += " warm-up: ";
    if (!plan.rows) return out + "none (" + std::string(plan.reason) + ")";
    out += std::to_string(plan.rows) + " rows + " + std::to_string(plan.outputs) + " token (" + std::string(plan.reason) +
           ") " + std::to_string(static_cast<long long>(ms)) + " ms before listening";
    return out;
}

// Runs `run(prompt)` for a planned warm-up and returns its wall milliseconds (0 without a warm-up). `run` drives the
// engine's request path to completion and throws on failure: a request path that cannot run its tile at start would
// fail the first request too, so the server does not start.
template <class Run>
double run_startup_warmup(const StartupWarmupPlan& plan, std::uint64_t vocabulary, Run&& run) {
    if (!plan.rows) return 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    std::forward<Run>(run)(startup_warmup_prompt(plan.rows, vocabulary));
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace strix::serve
