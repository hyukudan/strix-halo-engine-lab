#pragma once
// Shared e4b-eagle test helpers: checks, small configs, a synthetic committed-row scenario and drafter lockstep.
#include "gemma4/eagle/eagle.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle_test {
using namespace strix::gemma4;
using eagle::Config;
using eagle::Draft;
using eagle::Drafter;
using eagle::Rows;
using eagle::Weights;

inline int failures = 0;
#define EAGLE_CHECK(cond, ...)                                                                                     \
    do {                                                                                                           \
        if (!(cond)) {                                                                                             \
            ++eagle_test::failures;                                                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s ", __FILE__, __LINE__, #cond);                                    \
            std::fprintf(stderr, __VA_ARGS__);                                                                     \
            std::fprintf(stderr, "\n");                                                                            \
        }                                                                                                          \
    } while (0)
template <class F> bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }
    float uniform() { return float(double(next() >> 11U) * 0x1p-53 * 2.0 - 1.0); }
    std::uint32_t below(std::uint32_t n) { return std::uint32_t(next() % n); }
};

// Small but structurally real: head_dim 256 (target sliding table), 8q/2kv, window 512.
inline Config small_config(std::vector<std::int32_t> taps = {11, 23, eagle::final_tap}, bool concat = true, float scale = 0.0625F) {
    Config c;
    c.hidden = c.target_hidden = 256;
    c.ffn = 512;
    c.vocab = 512;
    c.target_vocab = 1024;
    c.taps = std::move(taps);
    c.concat = concat;
    c.attention_scale = scale;
    return c;
}

// Target embedding table [target_vocab][hidden] (Q8_0, like the deployed token_embd).
inline eagle::Matrix embedding_table(const Config& c, std::uint64_t seed) {
    Rng r{seed};
    std::vector<float> x(std::size_t(c.target_vocab) * c.hidden);
    for (auto& a : x) a = 0.05F * r.uniform();
    return eagle::pack(x, c.target_vocab, c.hidden, WeightFormat::q8_0);
}

// Synthetic committed target rows: taps [5][H] and final hidden [H] per position (deterministic per position).
struct Scenario {
    Config c;
    std::uint64_t seed = 1;
    std::uint32_t prompt = 37, until = 700;
    void row(std::uint64_t p, float* taps, float* hidden) const {
        Rng r{seed * 1000003ULL + p * 7919ULL};
        const auto H = c.hidden;
        for (std::uint32_t i = 0; i < eagle::engine_taps * H; ++i) taps[i] = 4.0F * r.uniform();
        for (std::uint32_t i = 0; i < H; ++i) hidden[i] = r.uniform();
    }
};

struct Staged {
    std::vector<float> taps, hidden;
    std::vector<std::uint32_t> next;
    std::vector<std::uint64_t> positions;
    Rows rows() const { return {taps, hidden, next, positions}; }
};
inline Staged make_rows(const Scenario& s, std::uint64_t first, std::span<const std::uint32_t> next) {
    Staged out;
    const auto H = s.c.hidden, n = std::uint32_t(next.size());
    out.taps.resize(std::size_t(n) * eagle::engine_taps * H);
    out.hidden.resize(std::size_t(n) * H);
    for (std::uint32_t r = 0; r < n; ++r) {
        s.row(first + r, out.taps.data() + std::size_t(r) * eagle::engine_taps * H, out.hidden.data() + std::size_t(r) * H);
        out.positions.push_back(first + r);
    }
    out.next.assign(next.begin(), next.end());
    if (!s.c.needs_final()) out.hidden.clear();
    return out;
}

// Drives all drafters in lockstep through prompt priming (chunks 1..8), cycles with k = 1..7 (and some k = 0),
// random acceptance (accepted drafts of drafter 0 become committed next tokens), stage/discard/restage, publish,
// past the 512 window. compare(reference, other, index, label) is called for every draft of drafters 1..n.
using Compare = std::function<void(const Draft&, const Draft&, std::size_t, const std::string&)>;
using StageCheck = std::function<void(std::span<const std::uint64_t>)>;
inline void lockstep(const Scenario& s, const std::vector<Drafter*>& drafters, const Compare& compare,
                     const StageCheck& after_publish = {}) {
    Rng r{s.seed ^ 0xabcdefULL};
    std::vector<std::uint32_t> tokens(s.prompt);
    for (auto& t : tokens) t = r.below(s.c.target_vocab);
    auto commit = [&](std::uint64_t first, std::span<const std::uint32_t> next, bool discard_once) {
        const auto rows = make_rows(s, first, next);
        for (auto* d : drafters) {
            if (discard_once) { d->stage(rows.rows()); d->discard(); }
            d->stage(rows.rows());
        }
        for (auto* d : drafters) d->publish();
        if (after_publish) after_publish(rows.positions);
    };
    std::uint64_t f = 0;
    std::uint32_t chunk = 1;
    while (f + 1 < s.prompt) {
        const auto n = std::min<std::uint64_t>(chunk, s.prompt - 1 - f);
        commit(f, std::span<const std::uint32_t>(tokens).subspan(f + 1, n), chunk == 3);
        f += n;
        chunk = chunk % 8 + 1;
    }
    std::uint32_t anchor = tokens.back();
    for (std::uint32_t cycle = 0; f < s.until; ++cycle) {
        const auto k = cycle % 9 == 8 ? 0U : 1U + cycle % 7;
        std::vector<std::uint32_t> next;
        if (k) {
            std::vector<Draft> drafts;
            for (auto* d : drafters) drafts.push_back(d->draft(anchor, f, k));
            const auto label = "pos " + std::to_string(f) + " k " + std::to_string(k);
            for (std::size_t i = 1; i < drafts.size(); ++i) compare(drafts[0], drafts[i], i, label);
            // Drafting is side-effect free: a repeat is identical.
            const auto again = drafters[0]->draft(anchor, f, k);
            for (std::uint32_t d = 0; d < k; ++d)
                EAGLE_CHECK(again.tokens[d] == drafts[0].tokens[d] && again.prob[d] == drafts[0].prob[d], "repeat draft %s", label.c_str());
            const auto a = r.below(k + 1);
            for (std::uint32_t d = 0; d < a; ++d) next.push_back(drafts[0].tokens[d]);
        }
        const auto bonus = r.below(s.c.target_vocab);
        next.push_back(bonus);
        commit(f, next, cycle % 11 == 5);
        f += next.size();
        anchor = bonus;
    }
}

} // namespace eagle_test
