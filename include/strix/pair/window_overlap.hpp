#pragma once
// Decode of a two-stage layer split with OVERLAPPED speculative verify windows (host only, model- and vendor-agnostic).
//
// A layer split decoding one verify window at a time is serial: draft, stage 0, hop, stage 1 (+ head and check), ids
// back. Its window costs the one-device window plus the hops, so two devices never decode faster than one
// (strix/pair/layer_pipeline.hpp decode, the layers form of the split planner). Overlap keeps two windows in flight:
// as soon as stage 0 has run window n it drafts window n+1 on the guess "n is accepted whole and its bonus token is the
// drafter's next token", snapshots its sequence state and runs n+1 while the last stage verifies n. If the guess holds,
// n+1 is half done when its turn comes; if not, stage 0 restores the snapshot and runs n+1 from the true tokens, so the
// committed tokens are the serial loop's (the bits are those of the serial split: same kernels, same rows).
// The drafter sits on stage 0 (its stage is free when the next chain starts); the target's last hidden rows the drafter
// reads cross back with the ids.
//
// Per window (a, b: stage compute at the cut, D: draft, s: snapshot, hop / back: link messages):
//   serial split (drafter on the last stage)  P_split = D1 + a + hop + b + back_ids
//   overlap serial path (drafter on stage 0)  P_ser   = D0 + a + hop + b + back_ids_and_hidden
//   hit   P_hit  = max(D0 + s + a, b, (D0 + s + a + hop + b + back) / 2)   (stage 0, last stage, two windows in a ring)
//   miss  P_miss = P_ser + s + w, w = max(0, (D0 + s + a) - (hop + b + back)): the doomed window is not abortable
//   E     = h P_hit + (1 - h) P_miss; the committed tokens per window are the serial loop's, so speedup = P_split / E.
// h is the share of windows whose speculative successor is on the path: every proposal accepted AND bonus == the
// drafter's next guess, counted from the acceptance histogram (never assumed). Launching a speculative window pays
// iff h >= h_min = (P_miss - P_ser) / ((P_ser - P_hit) + (P_miss - P_ser)): the gate is the pair's own arithmetic, not
// a fixed share. The cut minimises E: on the hit path it balances D0 + s + a against b, i.e. it moves layers off the
// drafter's device. A tensor-parallel form (both devices read every layer at once, joins per layer) is priced by the
// split planner and passed in as `tensor_ms`, the pattern of strix/pair/regime.hpp PipelineCost::tp_ms_per_row:
// the fastest admissible form wins, a more complex one only by `min_gain` (run noise).
// Links are strix::pair::chain::Link (one link type for every pair planner): USB4 nodes and two cards in one box are
// the same model with different latency and rates.
// Home: shared (host shared). The per-backend bodies it needs are each family's existing execution pieces: a stage-0
// state snapshot / restore and a second verify window workspace (HIP / CUDA / SYCL), no new kernel.
#include "strix/pair/chain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::pair::overlap {

// Drafter acceptance over windows of R rows (R - 1 proposals, then the bonus row).
struct Acceptance {
    std::vector<std::uint64_t> accepted;  // accepted[k]: windows with exactly k proposals accepted, k = 0 .. R - 1
    std::uint64_t bonus_hits = 0;         // of the accepted[R - 1] windows: the bonus token was the drafter's next guess
};
[[nodiscard]] inline std::uint64_t windows(const Acceptance& acc) {
    return std::accumulate(acc.accepted.begin(), acc.accepted.end(), std::uint64_t{0});
}
// Share of windows whose speculative successor is on the path; 0 with no windows (unknown is not a hit).
[[nodiscard]] inline double hit_rate(const Acceptance& acc) {
    const std::uint64_t n = windows(acc);
    if (n == 0U) return 0.0;
    if (acc.bonus_hits > acc.accepted.back()) throw std::invalid_argument("overlap: bonus hits exceed full-accept windows");
    return static_cast<double>(acc.bonus_hits) / static_cast<double>(n);
}
// Committed tokens per window (accepted proposals + the bonus); the same in every form.
[[nodiscard]] inline double tokens_per_window(const Acceptance& acc) {
    const std::uint64_t n = windows(acc);
    if (n == 0U) return 0.0;
    double sum = 0.0;
    for (std::size_t k = 0; k < acc.accepted.size(); ++k)
        sum += static_cast<double>(acc.accepted[k]) * static_cast<double>(k + 1U);
    return sum / static_cast<double>(n);
}

// One verify window of `rows` rows on the reference device (measured; a device's `speed` scales it).
struct Window {
    std::vector<double> layer_ms;          // per trunk layer
    double head_ms = 0.0;                  // final norm + head + verify check, on the last stage
    double draft_ms = 0.0;                 // drafter pass for the next window; 0: no drafter, nothing to overlap
    std::uint32_t rows = 1;                // R
    double hop_bytes_per_row = 0.0;        // residual row crossing the cut
    double id_bytes_per_row = 4.0;         // committed ids back to stage 0
    double hidden_bytes_per_row = 0.0;     // target last hidden the drafter reads (crosses back when it sits on stage 0)
    std::vector<double> state_bytes;       // per layer: sequence state a speculative window overwrites (recurrent /
                                           // convolution rows; appended KV rows need none). Empty: none.
    double snapshot_fixed_ms = 0.0;        // per snapshot (launch, sync)
    double copy_bytes_per_ms = 0.0;        // device-local copy rate of the snapshot
    std::uint32_t min_layers = 1;          // per stage
};
struct Pair {
    double speed0 = 1.0, speed1 = 1.0;     // time multiplier per device vs the reference (measured)
    chain::Link link;                      // device 0 <-> device 1 (forward = towards device 1)
    double free0 = std::numeric_limits<double>::infinity();  // bytes free on each device under the serial split
    double free1 = std::numeric_limits<double>::infinity();
};
struct Extra {                             // bytes the overlap form adds
    double window_bytes0 = 0.0, window_bytes1 = 0.0;  // second verify window workspace per stage
    double drafter_bytes = 0.0;            // moves from the last stage to stage 0
};
struct Policy {
    double min_gain = 0.03;                // a more complex form must win by this fraction of the simpler one
};

enum class Form : std::uint8_t { single, split, overlap, tensor };
[[nodiscard]] inline const char* form_name(Form f) noexcept {
    return f == Form::single ? "single" : f == Form::split ? "split" : f == Form::overlap ? "overlap" : "tensor";
}
struct Estimate {
    Form form = Form::single;
    bool admissible = false;
    std::uint32_t cut = 0;                 // split / overlap: first layer of stage 1; single: the device
    double ms = std::numeric_limits<double>::infinity();  // expected ms per window
    double p_hit = 0.0, p_miss = 0.0, p_serial = 0.0, h = 0.0, h_min = 0.0;  // overlap terms
    std::string reason;
};

namespace detail {
inline double sum(const std::vector<double>& v, std::size_t b, std::size_t e) {
    return std::accumulate(v.begin() + static_cast<std::ptrdiff_t>(b), v.begin() + static_cast<std::ptrdiff_t>(e), 0.0);
}
inline double message_ms(double latency_ms, double bytes, double bytes_per_ms) {
    if (bytes <= 0.0) return latency_ms;
    if (bytes_per_ms <= 0.0) throw std::invalid_argument("overlap: bytes cross a link without a declared rate");
    return latency_ms + bytes / bytes_per_ms;
}
inline void check(const Window& w, const Pair& p) {
    if (w.layer_ms.size() < 2U * std::max(w.min_layers, 1U)) throw std::invalid_argument("overlap: too few layers to cut");
    if (!w.state_bytes.empty() && w.state_bytes.size() != w.layer_ms.size())
        throw std::invalid_argument("overlap: state_bytes must have one entry per layer");
    if (w.rows == 0U || p.speed0 <= 0.0 || p.speed1 <= 0.0) throw std::invalid_argument("overlap: rows and speeds must be > 0");
}
inline double state_bytes(const Window& w, std::size_t cut) { return w.state_bytes.empty() ? 0.0 : sum(w.state_bytes, 0, cut); }
}  // namespace detail

// One device runs the whole window (draft included); the faster device wins, ties keep device 0.
[[nodiscard]] inline Estimate single(const Window& w, const Pair& p) {
    detail::check(w, p);
    const double base = detail::sum(w.layer_ms, 0, w.layer_ms.size()) + w.head_ms + w.draft_ms;
    Estimate e;
    e.form = Form::single;
    e.admissible = true;
    e.cut = p.speed1 < p.speed0 ? 1U : 0U;
    e.ms = base * std::min(p.speed0, p.speed1);
    return e;
}

// Serial split, drafter with the head on the last stage; the cut minimises the window, ties the balanced, then earliest.
[[nodiscard]] inline Estimate split(const Window& w, const Pair& p) {
    detail::check(w, p);
    const double rows = static_cast<double>(w.rows);
    const double hop = detail::message_ms(p.link.latency_ms, rows * w.hop_bytes_per_row, p.link.forward_bytes_per_ms);
    const double back = detail::message_ms(p.link.latency_ms, rows * w.id_bytes_per_row, p.link.reverse_bytes_per_ms);
    const std::size_t n = w.layer_ms.size(), lo = std::max(w.min_layers, 1U);
    Estimate e;
    e.form = Form::split;
    double best_gap = std::numeric_limits<double>::infinity();
    for (std::size_t cut = lo; cut + lo <= n; ++cut) {
        const double a = p.speed0 * detail::sum(w.layer_ms, 0, cut);
        const double b = p.speed1 * (detail::sum(w.layer_ms, cut, n) + w.head_ms + w.draft_ms);
        const double ms = a + hop + b + back;
        const double gap = std::abs(a - b);
        if (ms < e.ms || (ms == e.ms && gap < best_gap)) {
            e.ms = ms;
            best_gap = gap;
            e.cut = static_cast<std::uint32_t>(cut);
        }
    }
    e.p_serial = e.ms;
    e.admissible = true;
    return e;
}

// Overlapped windows (drafter on stage 0): the cut minimising the expected window at the measured hit rate.
// Inadmissible, with the reason, when there is no drafter, when h is below the break-even h_min at the best cut, or
// when a device cannot pay the extra bytes (second window, two stage-0 snapshots, the drafter on stage 0).
[[nodiscard]] inline Estimate overlapped(const Window& w, const Pair& p, const Acceptance& acc, const Extra& x = {}) {
    detail::check(w, p);
    Estimate e;
    e.form = Form::overlap;
    e.h = hit_rate(acc);
    if (w.draft_ms <= 0.0) { e.reason = "no drafter: nothing to guess the next window from"; return e; }
    const double rows = static_cast<double>(w.rows);
    const double hop = detail::message_ms(p.link.latency_ms, rows * w.hop_bytes_per_row, p.link.forward_bytes_per_ms);
    const double back = detail::message_ms(p.link.latency_ms, rows * (w.id_bytes_per_row + w.hidden_bytes_per_row),
                                           p.link.reverse_bytes_per_ms);
    const double d0 = p.speed0 * w.draft_ms;
    const std::size_t n = w.layer_ms.size(), lo = std::max(w.min_layers, 1U);
    bool memory_short = false;
    for (std::size_t cut = lo; cut + lo <= n; ++cut) {
        const double snap_bytes = detail::state_bytes(w, cut);
        if (2.0 * snap_bytes + x.window_bytes0 + x.drafter_bytes > p.free0 || x.window_bytes1 > p.free1 + x.drafter_bytes) {
            memory_short = true;
            continue;
        }
        double s = 0.0;
        if (snap_bytes > 0.0) {
            if (w.copy_bytes_per_ms <= 0.0) throw std::invalid_argument("overlap: state to snapshot without a copy rate");
            s = w.snapshot_fixed_ms + snap_bytes / w.copy_bytes_per_ms;
        }
        const double a = p.speed0 * detail::sum(w.layer_ms, 0, cut);
        const double b = p.speed1 * (detail::sum(w.layer_ms, cut, n) + w.head_ms);
        const double stage0 = d0 + s + a;
        const double p_ser = d0 + a + hop + b + back;
        const double p_hit = std::max({stage0, b, (stage0 + hop + b + back) / 2.0});
        const double p_miss = p_ser + s + std::max(0.0, stage0 - (hop + b + back));
        const double ms = e.h * p_hit + (1.0 - e.h) * p_miss;
        if (ms < e.ms) {
            e.ms = ms;
            e.cut = static_cast<std::uint32_t>(cut);
            e.p_hit = p_hit;
            e.p_miss = p_miss;
            e.p_serial = p_ser;
            const double loss = p_miss - p_ser, gain = p_ser - p_hit;
            e.h_min = gain + loss > 0.0 ? loss / (gain + loss) : 1.0;
        }
    }
    if (!std::isfinite(e.ms)) {
        e.reason = memory_short ? "memory: no cut leaves room for the second window, the snapshots and the drafter on stage 0"
                                : "no legal cut";
        return e;
    }
    if (e.h < e.h_min) {
        e.reason = "hit rate below break-even: speculative windows lose";
        return e;
    }
    e.admissible = true;
    return e;
}

struct Choice {
    Estimate chosen, single, split, overlap, tensor;
};
// The fastest admissible form; a later (more complex) form replaces the chosen one only by `min_gain`.
// `tensor_ms` is the tensor-parallel window priced by the split planner (0: no tensor form for this binding).
[[nodiscard]] inline Choice choose(const Window& w, const Pair& p, const Acceptance& acc, const Extra& x = {},
                                   double tensor_ms = 0.0, const Policy& policy = {}) {
    Choice c;
    c.single = single(w, p);
    c.split = split(w, p);
    c.overlap = overlapped(w, p, acc, x);
    c.tensor.form = Form::tensor;
    if (tensor_ms > 0.0) {
        c.tensor.admissible = true;
        c.tensor.ms = tensor_ms;
    } else {
        c.tensor.reason = "no tensor form declared";
    }
    c.chosen = c.single;
    for (const Estimate* cand : {&c.split, &c.overlap, &c.tensor})
        if (cand->admissible && cand->ms < c.chosen.ms * (1.0 - policy.min_gain)) c.chosen = *cand;
    return c;
}

}  // namespace strix::pair::overlap
