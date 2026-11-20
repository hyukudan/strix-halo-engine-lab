// Draft-chain segmentation for MTP / speculative drafters (shared host logic, any backend).
//
// A drafter that launches its whole chain of `depth` steps at once (one graph replay) wastes the steps after the first
// draft whose probability falls below the stop threshold: the verify discards them. Splitting the chain into two
// launches [0, a) and [a, depth), with the second issued only when the first did not stop, keeps every kept draft and
// every verified token identical (the steps that still run are the same launches on the same inputs) and saves
// depth - a steps whenever the chain stops inside the first segment, at the price of one host turn when it does not.
//
// DraftSplit keeps a histogram of how many drafts each chain kept and picks the segment ends 0 < b1 < ... < bm = depth
// that minimise the expected steps
//     b1 + sum_{i > 1} P(kept > b_{i-1}) * (b_i - b_{i-1} + host_turn)
// (host_turn = the extra sync + launch in draft-step units; a segment runs only when no draft before it stopped). With
// m <= 2 this is the single split a + P(kept > a) * (depth - a + host_turn). STRIX_MTP_DRAFT_SPLIT (or the backend's own
// variable): unset / "auto" = that choice, 0 = one segment (whole chain), N = fixed first segment N (then the rest).
// The segment cap (max_segments, 0 = depth) is the backend's STRIX_*_DRAFT_SEGMENTS: 2 = the single split.
// Context: two histograms, chosen by the caller per chain (the backends pass "the previous chain of this session kept its
// whole depth"): long-draft streams (JSON, copies) and short-draft streams (prose) then get their own ends from one
// process-wide chooser instead of the mix of both.
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "strix/env_flags.hpp"

namespace strix::mtp {

class DraftSplit {
public:
    static constexpr std::uint32_t kBins = 16;

    // mode: -1 auto, 0 off, N > 0 fixed first segment. host_turn in draft-step units.
    explicit DraftSplit(int mode = -1, double host_turn = 0.25, std::uint64_t min_samples = 8, std::uint32_t max_segments = 0)
        : mode_(mode), host_turn_(host_turn), min_samples_(min_samples), max_segments_(max_segments) {}

    // -1 / 0 / N from an environment variable (unset, empty or "auto" -> -1), recorded for the effective config.
    static int mode_from_env(const char* name) {
        const char* e = strix::envcfg::text(name);
        if (!e || !*e || std::string(e) == "auto") return -1;
        return std::atoi(e);
    }

    // Segment cap from an environment variable: unset / empty / "auto" -> 0 (no cap), N -> N (2 = the single split).
    static std::uint32_t segments_from_env(const char* name) {
        const char* e = strix::envcfg::text(name);
        if (!e || !*e || std::string(e) == "auto") return 0;
        const int n = std::atoi(e);
        return n > 0 ? static_cast<std::uint32_t>(n) : 0U;
    }

    // Segment ends for a chain of `depth` steps: ascending, last == depth (one entry: no split).
    std::vector<std::uint32_t> segment_ends(std::uint32_t depth, std::uint32_t context = 0) const {
        const auto& h = kept_[context & 1U];
        if (mode_ == 0 || depth < 2) return {depth};
        const std::uint32_t cap = max_segments_ == 0 || max_segments_ > depth ? depth : max_segments_;
        const std::uint32_t a = first_segment(depth, context);
        if (mode_ > 0 || cap <= 2 || a == depth || samples(depth, context) < min_samples_) {
            if (a == depth) return {depth};
            return {a, depth};
        }
        // g[c][s]: least expected steps from boundary s to the end with at most c segments (the chain alive at s)
        const double total = static_cast<double>(samples(depth, context));
        std::vector<double> alive(depth + 1U, 0.0);  // P(kept > s)
        for (std::uint32_t s = 0; s <= depth; ++s) {
            std::uint64_t more = 0;
            for (std::uint32_t n = s + 1; n <= depth && n < kBins; ++n) more += h[n];
            alive[s] = s == 0 ? 1.0 : static_cast<double>(more) / total;
        }
        const double inf = 1e300;
        std::vector<std::vector<double>> g(cap + 1U, std::vector<double>(depth + 1U, inf));
        std::vector<std::vector<std::uint32_t>> next(cap + 1U, std::vector<std::uint32_t>(depth + 1U, depth));
        for (std::uint32_t c = 0; c <= cap; ++c) g[c][depth] = 0.0;
        for (std::uint32_t c = 1; c <= cap; ++c)
            for (std::uint32_t s = depth; s-- > 0;) {
                const double turn = s > 0 ? host_turn_ : 0.0;
                double best = alive[s] * (depth - s + turn);  // one segment to the end (preferred on ties)
                std::uint32_t at = depth;
                for (std::uint32_t e = s + 1; e < depth; ++e) {
                    if (g[c - 1][e] >= inf) continue;
                    const double cost = alive[s] * (e - s + turn) + g[c - 1][e];
                    if (cost < best) best = cost, at = e;
                }
                g[c][s] = best;
                next[c][s] = at;
            }
        std::vector<std::uint32_t> ends;
        for (std::uint32_t s = 0, c = cap; s < depth; --c) {
            s = next[c][s];
            ends.push_back(s);
        }
        return ends;
    }

    // First segment length for a chain of `depth` steps (== depth: no split).
    std::uint32_t first_segment(std::uint32_t depth, std::uint32_t context = 0) const {
        const auto& h = kept_[context & 1U];
        if (mode_ == 0 || depth < 2) return depth;
        if (mode_ > 0) return static_cast<std::uint32_t>(mode_) < depth ? static_cast<std::uint32_t>(mode_) : depth;
        std::uint64_t total = 0;
        for (std::uint32_t n = 0; n <= depth && n < kBins; ++n) total += h[n];
        if (total < min_samples_) return depth < 2U ? depth : 2U;
        std::uint32_t best = depth;
        double best_cost = depth;
        for (std::uint32_t a = 1; a < depth; ++a) {
            std::uint64_t more = 0;
            for (std::uint32_t n = a + 1; n <= depth && n < kBins; ++n) more += h[n];
            const double cost = a + static_cast<double>(more) / static_cast<double>(total) * (depth - a + host_turn_);
            if (cost < best_cost) best_cost = cost, best = a;
        }
        return best;
    }

    // Record one finished chain that kept `kept` drafts (the stop rule keeps the low draft itself).
    void note(std::uint32_t kept, std::uint32_t context = 0) { ++kept_[context & 1U][kept < kBins ? kept : kBins - 1]; }

    // Receipt tally of finished chains: steps launched (all segments that ran), steps kept, segments after the first
    // (= host turns). launched - kept is the step work the verify discards.
    struct Tally {
        std::uint64_t chains = 0, launched = 0, kept = 0, turns = 0;
    };
    void note(std::uint32_t kept, std::uint32_t launched, std::uint32_t turns, std::uint32_t context = 0) {
        note(kept, context);
        ++tally_.chains;
        tally_.launched += launched;
        tally_.kept += kept;
        tally_.turns += turns;
    }
    const Tally& tally() const { return tally_; }

    int mode() const { return mode_; }
    std::uint64_t kept(std::uint32_t n) const { return n < kBins ? kept_[0][n] + kept_[1][n] : 0; }
    std::uint64_t kept(std::uint32_t n, std::uint32_t context) const { return n < kBins ? kept_[context & 1U][n] : 0; }

private:
    std::uint64_t samples(std::uint32_t depth, std::uint32_t context) const {
        std::uint64_t total = 0;
        for (std::uint32_t n = 0; n <= depth && n < kBins; ++n) total += kept_[context & 1U][n];
        return total;
    }

    int mode_;
    double host_turn_;
    std::uint64_t min_samples_;
    std::uint32_t max_segments_;
    std::array<std::array<std::uint64_t, kBins>, 2> kept_{};
    Tally tally_{};
};

}  // namespace strix::mtp
