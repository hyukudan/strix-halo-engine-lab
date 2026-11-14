#pragma once

#include "strix/serve/short_first.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Which waiting short prompts share ONE device prefill pass.
//
// A short prompt's device prefill is mostly a fixed cost: one pass over the layers reads every weight matrix (and
// every routed expert it touches) whole however few rows it carries, plus one pair exchange per layer. N short
// prompts admitted together therefore pay that pass once instead of N times when their rows run as one group pass
// (rows concatenated, stateless work once over all rows, attention / caches / carries per member).
//
// The rule is pure scheduling over prompts that are ready now; it never holds a prompt to wait for company:
//   - the oldest waiting candidate is always in (a lone prompt runs exactly as today);
//   - the next candidates in arrival order join while the group stays within max_members, max_rows, the decode-gap
//     budget (a + b x summed rows <= gap_budget_ms) and the first member's route class; the first candidate that
//     does not fit ends the group (no skipping, so arrival order is kept);
//   - only short prompts (rows < short_rows) group; a longer first candidate runs alone.
// Exactness (R1): route_class is what the engine would select for that prompt ALONE (every row-count threshold
// evaluated on the member's own rows). The group pass runs with that class forced, never re-selected from the
// group's row total, so each member gets the kernels and bits of its solo run.
// The caller only offers candidates that may group: no prefix-cache resume, not mid-prefill.
namespace strix::serve {

struct GroupCandidate {
    std::uint64_t id = 0;            // the caller's handle (slot, request id)
    std::uint32_t rows = 0;          // prompt rows this prefill runs
    std::uint64_t arrival_seq = 0;   // admission order; candidates are passed oldest first
    std::uint32_t route_class = 0;   // engine route class for this prompt alone (R1)
};

struct PrefillGroupCost {
    double a_ms = 0.0;   // fixed ms of one pass
    double b_ms = 0.0;   // ms per row
    [[nodiscard]] constexpr double ms(std::uint64_t rows) const noexcept { return a_ms + b_ms * static_cast<double>(rows); }
};

// Physical limits supplied by the execution owner, after resolving its options.
// local_rows bounds either rank: every member restarts window ownership at rank 0.
struct PrefillGroupCapacity {
    std::uint64_t members = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t rows = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t member_rows = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t windows = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t local_rows = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t window_rows = 1;
    std::uint32_t split_class_mask = 0;
    std::uint32_t rejected_class_mask = 0;
    const char* reason = "eligible-routes-only";

    [[nodiscard]] bool accepts(std::uint32_t route) const noexcept {
        return members >= 2U && window_rows != 0U && (route & rejected_class_mask) == 0U;
    }
    [[nodiscard]] std::uint64_t window_count(std::uint32_t n) const noexcept {
        return window_rows == 0U ? 0U : (std::uint64_t{n} + window_rows - 1U) / window_rows;
    }
    [[nodiscard]] std::uint64_t local_count(std::uint32_t n, std::uint32_t route) const noexcept {
        if ((route & split_class_mask) == 0U || window_rows == 0U) return n;
        const std::uint64_t pair = 2ULL * window_rows;
        return n / pair * window_rows + std::min<std::uint64_t>(n % pair, window_rows);
    }
};

struct PrefillGroupPolicy {
    enum class Mode : std::uint8_t { automatic, off };
    static constexpr std::uint32_t kDefaultMaxMembers = 8;
    static constexpr std::uint32_t kDefaultMaxRows = 512;
    static constexpr std::uint32_t kDefaultShortRows = 64;

    Mode mode = Mode::automatic;
    std::uint32_t max_members = kDefaultMaxMembers;
    std::uint32_t max_rows = kDefaultMaxRows;      // summed rows of one group pass
    std::uint32_t short_rows = kDefaultShortRows;  // the family's short limit: only rows < short_rows group

    [[nodiscard]] constexpr bool groups() const noexcept { return mode == Mode::automatic && max_members > 1U && short_rows > 1U; }
    [[nodiscard]] constexpr const char* name() const noexcept { return mode == Mode::off ? "off" : "auto"; }
    [[nodiscard]] constexpr std::uint32_t storage_rows(std::uint32_t context) const noexcept {
        return groups() ? std::min({context, short_rows - 1U, max_rows}) : 0U;
    }
    [[nodiscard]] std::string describe() const {
        return std::string("prefill_group: ") + name() + " (members " + std::to_string(max_members) +
            ", rows " + std::to_string(max_rows) + ", short below " + std::to_string(short_rows) +
            "; " + (groups() ? "device eligibility/capacity pending" : "solo") + ")\n";
    }

    // STRIX_SERVE_PREFILL_GROUP = auto | 0 (off); STRIX_SERVE_PREFILL_GROUP_MEMBERS (2..64);
    // STRIX_SERVE_PREFILL_GROUP_ROWS (at least 1). `env(name)` returns the value or an empty string.
    // `family_default` is the mode when the switch is unset: a family keeps `off` until its receipts are measured.
    // short_rows is the family's limit and is set by the family, not by the environment.
    template <class Env>
    [[nodiscard]] static PrefillGroupPolicy from_env(Env env, Mode family_default = Mode::automatic,
                                                     std::uint32_t short_rows_limit = kDefaultShortRows) {
        PrefillGroupPolicy p;
        p.mode = family_default;
        p.short_rows = short_rows_limit;
        if (const std::string s = env("STRIX_SERVE_PREFILL_GROUP"); !s.empty()) {
            if (s == "auto" || s == "1" || s == "on") p.mode = Mode::automatic;
            else if (s == "0" || s == "off") p.mode = Mode::off;
            else throw std::invalid_argument("STRIX_SERVE_PREFILL_GROUP is auto or 0, not '" + s + "'");
        }
        if (const std::string s = env("STRIX_SERVE_PREFILL_GROUP_MEMBERS"); !s.empty()) {
            const long long v = std::stoll(s);
            if (v < 2 || v > 64) throw std::invalid_argument("STRIX_SERVE_PREFILL_GROUP_MEMBERS must be 2..64");
            p.max_members = static_cast<std::uint32_t>(v);
        }
        if (const std::string s = env("STRIX_SERVE_PREFILL_GROUP_ROWS"); !s.empty()) {
            const long long v = std::stoll(s);
            if (v < 1 || v > (1LL << 30)) throw std::invalid_argument("STRIX_SERVE_PREFILL_GROUP_ROWS must be at least 1");
            p.max_rows = static_cast<std::uint32_t>(v);
        }
        return p;
    }
};

// Members of the next prefill pass, as candidate ids in arrival order (empty only when there is no candidate).
//   candidates     prompts ready for their prefill now, oldest first (strictly increasing arrival_seq)
//   cost           the family's measured pass cost (a + b x rows)
//   gap_budget_ms  decode-gap budget left for this pass (the admit-first gap); +infinity when nothing decodes.
//                  It only limits company: the oldest prompt runs even when it alone exceeds the budget.
[[nodiscard]] inline std::vector<std::uint64_t> plan_prefill_group(std::span<const GroupCandidate> candidates,
                                                                   const PrefillGroupPolicy& policy,
                                                                   const PrefillGroupCost& cost, double gap_budget_ms,
                                                                   const PrefillGroupCapacity& capacity = {}) {
    std::vector<std::uint64_t> ids;
    if (candidates.empty()) return ids;
    for (std::size_t i = 1; i < candidates.size(); ++i)
        if (candidates[i].arrival_seq <= candidates[i - 1].arrival_seq)
            throw std::invalid_argument("plan_prefill_group: candidates must be oldest first");
    const GroupCandidate& first = candidates.front();
    ids.push_back(first.id);
    if (!policy.groups() || first.rows == 0U || first.rows >= policy.short_rows ||
        !capacity.accepts(first.route_class) || first.rows > capacity.member_rows) return ids;
    std::uint64_t rows = first.rows;
    std::uint64_t windows = capacity.window_count(first.rows);
    std::uint64_t local_rows = capacity.local_count(first.rows, first.route_class);
    for (std::size_t i = 1; i < candidates.size(); ++i) {
        const GroupCandidate& c = candidates[i];
        if (ids.size() >= std::min<std::uint64_t>(policy.max_members, capacity.members)) break;
        if (c.rows == 0U || c.rows >= policy.short_rows || c.rows > capacity.member_rows) break;
        if (c.route_class != first.route_class) break;
        const std::uint64_t next = rows + c.rows;
        if (next > std::min<std::uint64_t>(policy.max_rows, capacity.rows)) break;
        const auto next_windows = windows + capacity.window_count(c.rows);
        const auto next_local = local_rows + capacity.local_count(c.rows, c.route_class);
        if (next_windows > capacity.windows || next_local > capacity.local_rows) break;
        const double ms = cost.ms(next);
        if (!(ms <= gap_budget_ms)) break;  // NaN budget: no company
        rows = next;
        windows = next_windows;
        local_rows = next_local;
        ids.push_back(c.id);
    }
    return ids;
}

// ids are indices in the original arrival-ordered waiting queue. The selector
// already charged ids[0]; charge each actual companion, excluding selected owners.
template <class Container, class Skips>
void limit_prefill_overtakes(std::vector<std::uint64_t>& ids, Container& waiting, Skips skips) {
    for (std::size_t n = 1; n < ids.size(); ++n) {
        const auto selected = [&](std::size_t i) {
            return std::find(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), i) !=
                   ids.begin() + static_cast<std::ptrdiff_t>(n);
        };
        for (std::size_t i = 0; i < ids[n]; ++i)
            if (!selected(i) && skips(waiting[i]) >= kMaxQueueOvertakes) {
                ids.resize(n);
                return;
            }
        for (std::size_t i = 0; i < ids[n]; ++i)
            if (!selected(i)) ++skips(waiting[i]);
    }
}

}  // namespace strix::serve
