#pragma once
// Hy4-flash verify rows (host side): the contract of Forward::forward_rows / commit_rows, the cache-ring rewind replay,
// and the binding of the shared speculative layer (strix::spec BlockDrafter + Arbiter + BlockSpeculator through
// EngineVerifyTarget) over any forward with that contract.
//
// forward_rows(tokens[R]), 1 <= R <= kMaxVerifyRows: rows r = 0 .. R-1 at positions p0 + r in ONE pass, each row
// bitwise equal to what step(tokens[r]) gives at that position (class BASE: the same q8_rows FP32 forms with Rows = R,
// the same per-row A8 probe / down, attention per row at its own position). commit_rows(k), 0 <= k <= R: keep the first
// k rows (position p0 + k, token history cut); the device rows of the rejected positions are rewritten by the next pass
// before any read, which holds when every ring is sized for the verify rows (check_rewind below replays it).
// Home: src/hy4 (binding: Hy4 positions / rings data); the controller, proposer and Arbiter are shared (strix::spec).
#include "hy4/hip/hy4_hip_plan.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/engine_target.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::hy4::hipbind::rows {

// Rows of one verify pass: the q8_rows direct / stepped forms keep the single-row arithmetic up to 8 rows and the shared
// device greedy selects 1..8 rows (strix::hip_greedy_rows_max).
inline constexpr std::uint32_t kMaxVerifyRows = 8;
// Routed experts of a verify window in the shared A8 union rows form (each distinct expert read once, rows bitwise == the
// per-row loop). Off switch: STRIX_HY4_ROWS_EXPERT_UNION=0.
inline constexpr bool kExpertUnionDefault = false;

// The decode step's device position words of a row at position p: {position, rope slot of its group, closing group, 0}.
[[nodiscard]] constexpr std::array<std::uint32_t, 4> position_words(std::uint32_t p, std::uint32_t ratio) {
    return {p, p / ratio * ratio, p / ratio, 0U};
}
// Rows one launch / one pass may write ahead of the oldest live key: the caches are sized for the larger of the batched
// prompt window and the verify rows.
[[nodiscard]] constexpr std::uint32_t cache_rows(std::uint32_t prompt_rows, std::uint32_t verify_rows) {
    return std::max({1U, prompt_rows, verify_rows});
}

// Host replay of every ring a layer keeps (sliding latent ring: window keys; indexer token-key ring: the own group's
// keys, which the group pool also reads): plain decode up to p0, one verify pass of R rows (all rows stored before any
// row reads, the order of forward_rows), commit k, then the following plain positions until every rejected slot has
// been rewritten. Every read must find the key of the position it asks for. Returns "" or the first clobber.
[[nodiscard]] inline std::string check_rewind(const Config& c, std::uint32_t layer, std::uint32_t context, std::uint32_t max_rows,
                                              std::uint32_t verify_rows) {
    const auto k = layer_caches(c, layer, context, max_rows);
    struct Ring { std::uint32_t slots; std::uint32_t span; const char* name; };  // span: keys a read covers ending at q
    std::vector<Ring> rings;
    if (c.indexed.at(layer) == 0U) rings.push_back({k.latent.ring, c.window, "latent window ring"});
    else rings.push_back({k.token_keys.ring, 0U, "indexer token ring"});
    for (const auto& ring : rings) {
        if (ring.slots == 0U) continue;  // indexed by position: a rejected row is rewritten before any later read
        const std::uint32_t ratio = c.index_ratio == 0U ? 1U : c.index_ratio;
        const auto first_read = [&](std::uint32_t q) {
            return ring.span ? (q + 1U > ring.span ? q + 1U - ring.span : 0U) : q / ratio * ratio;
        };
        for (std::uint32_t R = 1; R <= verify_rows; ++R)
            for (std::uint32_t kept = 1; kept <= R; ++kept)
                for (std::uint32_t p0 : {2U * ring.slots + 1U, 2U * ring.slots + 2U, 2U * ring.slots + 3U, 2U * ring.slots + 4U}) {
                    if (p0 + 2U * R + 1U >= context) return "context too small for the replay";
                    std::vector<std::uint32_t> owner(ring.slots, ~0U);
                    const auto write = [&](std::uint32_t q) { owner[q % ring.slots] = q; };
                    const auto read = [&](std::uint32_t q) -> std::string {
                        for (std::uint32_t x = first_read(q); x <= q; ++x)
                            if (owner[x % ring.slots] != x)
                                return std::string(ring.name) + ": layer " + std::to_string(layer) + " R " + std::to_string(R) +
                                       " kept " + std::to_string(kept) + " query " + std::to_string(q) + " key " +
                                       std::to_string(x) + " clobbered by " + std::to_string(owner[x % ring.slots]);
                        return {};
                    };
                    for (std::uint32_t q = 0; q < p0; ++q) write(q);
                    for (std::uint32_t r = 0; r < R; ++r) write(p0 + r);  // the pass stores every row first
                    for (std::uint32_t r = 0; r < R; ++r)
                        if (auto e = read(p0 + r); !e.empty()) return e;
                    for (std::uint32_t q = p0 + kept; q < p0 + R + 1U; ++q) {  // after commit: the plain redo
                        write(q);
                        if (auto e = read(q); !e.empty()) return e;
                    }
                }
    }
    return {};
}

// ---- shared speculative layer over a forward with the rows contract ------------------------------------------------
// Fwd: forward_rows(span tokens, bool logits) (synchronises), rows_greedy() -> span of R greedy ids, commit_rows(k),
// position(). EngineVerifyTarget needs Engine::Rows, Engine::LogitRows::all, forward_rows / read_top / commit / sync.
template <class Fwd>
class VerifyEngine {
public:
    struct Rows { std::uint32_t count = 0; };
    enum class LogitRows { all };
    explicit VerifyEngine(Fwd& f) : f_(f) {}
    Rows forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows) {
        f_.forward_rows(std::span<const std::uint32_t>(tokens, rows), false);
        return {rows};
    }
    void read_top(const Rows& out, std::uint32_t rows, std::uint32_t* host) const {
        const auto g = f_.rows_greedy();
        if (rows != out.count || g.size() < rows) throw std::logic_error("hy4 rows: read_top rows");
        std::copy(g.begin(), g.begin() + rows, host);
    }
    void commit(std::uint32_t k) { f_.commit_rows(k); }
    void sync() {}

private:
    Fwd& f_;
};

struct SpecOptions {
    std::uint32_t max_rows = kMaxVerifyRows;  // anchor + drafts per pass
    std::vector<double> verify_ms;            // seeds ms[R - 1], R = 1 .. max_rows (fixed costs: the Arbiter's model)
    std::uint32_t trace = 0;                  // first steps printed by the caller from SpecResult::steps
};
struct SpecResult {
    std::vector<std::uint32_t> ids;                   // emitted ids, ids[0] = the anchor
    std::vector<spec::StepRecord> steps;
    std::array<std::uint64_t, kMaxVerifyRows + 1> rows_hist{};  // passes by rows
    std::uint64_t proposed = 0, accepted = 0;
    double wall_ms = 0;
};

// Speculative greedy decode of n ids after a prompt already in `f` (position == prompt.size()); anchor = the greedy
// id after the prompt (ids[0]). Copy proposer only (no chain drafter yet); the Arbiter decides the rows of every pass.
// Output ids are those of plain greedy decode when forward_rows rows == step (the shared controller emits only the
// target's own greedy ids: accepted drafts equal them, the first mismatch is replaced by the target's id).
template <class Fwd>
SpecResult spec_decode(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o) {
    if (o.max_rows < 1U || o.max_rows > kMaxVerifyRows) throw std::invalid_argument("hy4 rows: spec max rows outside 1..8");
    spec::BlockConfig bc;
    bc.max_block = o.max_rows;
    bc.arbiter.windows.clear();
    for (std::uint32_t w = 1; w <= o.max_rows; ++w) bc.arbiter.windows.push_back(w);
    bc.copy = true;
    bc.chain = false;
    bc.hybrid = false;
    bc.fixed_costs = !o.verify_ms.empty();
    spec::BlockDrafter drafter(bc, nullptr);
    for (std::uint32_t r = 1; r <= o.max_rows && r <= o.verify_ms.size(); ++r) drafter.seed_verify_ms(r, o.verify_ms[r - 1U]);
    drafter.commit(prompt);
    SpecResult res;
    res.ids.push_back(anchor);
    drafter.commit(std::span<const std::uint32_t>(res.ids.data(), 1));
    VerifyEngine<Fwd> engine(f);
    spec::EngineVerifyTarget<VerifyEngine<Fwd>> target(engine);
    spec::BlockSpeculator sp(drafter, target);
    sp.start(anchor);
    const auto t0 = std::chrono::steady_clock::now();
    while (res.ids.size() < n) {
        spec::StepParams p;
        p.remaining = n - static_cast<std::uint32_t>(res.ids.size());
        p.max_rows = o.max_rows;
        p.context = f.position();
        const auto r = sp.step(res.ids, p);
        if (r.emitted == 0U) break;
        res.steps.push_back(r);
        ++res.rows_hist[std::min<std::uint32_t>(r.rows, kMaxVerifyRows)];
        res.proposed += r.proposed;
        res.accepted += r.accepted;
    }
    res.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (res.ids.size() > n) res.ids.resize(n);
    return res;
}

}  // namespace strix::hy4::hipbind::rows
