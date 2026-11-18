// qwen35 HIP dense speculation work model (strix/qwen35/hip_dense_spec_costs.hpp) on the shared controller.
//   1. Model shape: a window is priced at its IU4 row bucket (1/2/4/8/16), non-decreasing in rows.
//   2. Config: fixed = cap 15 + fixed_costs (the CUDA sub15d mechanism), learned = the previous form (cap 7).
//   3. Decisions against a virtual node (walls from the HDD2-T1 receipts, a fake MTP chain with per-position agreement a):
//      - the learned form under the old partial-bucket core (rows 3 / 6 / 12 = 127 / 131 / 141 ms) and a one-off
//        first-draft spike collapses to shallow windows (receipt: code k ~ 1, E 1.90);
//      - the fixed model on the bucket core drafts deep (k 7) and is faster per token;
//      - emitted ids == serial greedy in every arm (exactness), and fixed-cost decisions do not depend on wall jitter.
#include "strix/qwen35/hip_dense_spec_costs.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace sp = strix::spec;
namespace q = strix::qwen35;

namespace {
#define HC_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

double g_clock = 0.0;  // virtual ms

struct Node {
    bool partial_core = false;   // old IU4 core: windows that do not fill a bucket pay the measured partial-bucket walls
    double flush = 8.0, step = 4.25, spike = 0.0, jitter = 0.0;
    std::uint64_t seed = 1;
    double verify(std::uint32_t r, std::uint64_t n) const {
        double v = q::hip_dense_verify_ms(r, true);
        if (partial_core && r != q::hip_dense_row_bucket(r))  // HDD2-T1 cost1.log: r3 127.1, r6 130.6, r12 140.7
            v = r < 4U ? 127.130 : r < 8U ? 130.589 : 140.732;
        return v + jitter * double(mix(n * 0x9e3779b9ULL + seed) % 1000) / 1000.0;
    }
};

struct Target final : sp::VerifyTarget {
    const std::vector<std::uint32_t>& response;
    const Node& node;
    std::uint32_t committed = 0;
    std::size_t last = 0;
    std::uint64_t n = 0;
    Target(const std::vector<std::uint32_t>& r, const Node& nd) : response(r), node(nd) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        HC_CHECK(tokens.size() >= 1 && tokens.size() <= 16, "rows %zu", tokens.size());
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::size_t pos = committed + i + 1;
            greedy.push_back(pos < response.size() ? response[pos] : 7U);
        }
        last = tokens.size();
        g_clock += node.verify(std::uint32_t(tokens.size()), ++n);
    }
    void commit(std::uint32_t k) override {
        HC_CHECK(k >= 1 && k <= last, "commit %u of %zu", k, last);
        committed += k;
    }
};

struct Chain final : sp::ChainDrafter {
    const std::vector<std::uint32_t>& response;
    const Node& node;
    double agreement;
    std::uint32_t stream = 1;  // response tokens emitted so far
    std::uint64_t calls = 0;
    Chain(const std::vector<std::uint32_t>& r, const Node& nd, double a) : response(r), node(nd), agreement(a) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream) + prefix.size() + j;
            const bool right = double(mix(pos * 977U + 5U) % 1000U) / 1000.0 < agreement;
            out.push_back(pos < response.size() && right ? response[pos] : 5U);
        }
        g_clock += node.flush + node.step * double(k > 0 ? k - 1 : 0) + (calls++ == 0 ? node.spike : 0.0);
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream += std::uint32_t(emitted.size()); }
};

struct Result {
    double E = 0, ms_per_token = 0, mean_k = 0;
    std::vector<std::uint32_t> ks;
};

Result run(bool fixed, const Node& node, double agreement, std::uint32_t n_tokens = 1200) {
    std::vector<std::uint32_t> prompt, response;
    for (std::uint32_t i = 0; i < 64; ++i) prompt.push_back(std::uint32_t(mix(i + 11) % 900U) + 10U);
    for (std::uint32_t i = 0; i < n_tokens + 32; ++i) response.push_back(std::uint32_t(mix(i * 31U + 7U) % 900U) + 10U);
    g_clock = 0.0;
    auto bc = q::hip_dense_spec_block_config(16U, true, fixed, 0.264);
    bc.copy = false;  // prose-like stream (no copy structure): the MTP chain is the only drafter
    bc.clock_ms = [] { return g_clock; };
    Chain chain(response, node, agreement);
    sp::BlockDrafter drafter(bc, &chain);
    q::hip_dense_seed_verify(drafter, bc.max_block, fixed, true);
    Target target(response, node);
    drafter.commit(prompt);
    std::vector<std::uint32_t> out{response[0]};
    drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
    sp::BlockSpeculator spec(drafter, target);
    spec.start(out[0]);
    Result res;
    std::uint64_t steps = 0, ksum = 0;
    while (out.size() < n_tokens) {
        sp::StepParams p;
        p.remaining = n_tokens - std::uint32_t(out.size());
        p.max_rows = 16U;
        p.context = prompt.size() + out.size();
        const auto rec = spec.step(out, p);
        HC_CHECK(rec.emitted >= 1, "no progress");
        ++steps;
        ksum += rec.proposed;
        res.ks.push_back(rec.proposed);
    }
    for (std::size_t i = 0; i < out.size(); ++i) HC_CHECK(out[i] == response[i], "ids differ at %zu", i);  // exactness
    res.E = double(out.size() - 1U) / double(steps);
    res.ms_per_token = g_clock / double(out.size() - 1U);
    res.mean_k = double(ksum) / double(steps);
    return res;
}
}  // namespace

int main() {
    // 1. model shape
    for (std::uint32_t r = 1; r <= 16; ++r) {
        const std::uint32_t b = q::hip_dense_row_bucket(r);
        HC_CHECK(b >= r && b <= 16U && (b & (b - 1U)) == 0U, "bucket(%u) = %u", r, b);
        HC_CHECK(q::hip_dense_verify_ms(r, true) == q::hip_dense_verify_ms(b, true), "r %u priced off its bucket", r);
        if (r > 1) HC_CHECK(q::hip_dense_verify_ms(r, true) >= q::hip_dense_verify_ms(r - 1, true), "not monotone at %u", r);
        if (r > 1) HC_CHECK(q::hip_dense_verify_ms(r, false) >= q::hip_dense_verify_ms(r - 1, false), "vector not monotone at %u", r);
    }
    HC_CHECK(q::hip_dense_verify_ms(3, true) == 76.526 && q::hip_dense_verify_ms(9, true) == 126.841, "receipt values");
    // 2. config
    const auto f = q::hip_dense_spec_block_config(16U, true, true, 0.264);
    HC_CHECK(f.fixed_costs && f.chain_max_k == 15U && f.max_block == 16U && f.chain && f.hybrid, "fixed config");
    HC_CHECK(f.chain_seed_flush_ms > f.chain_seed_step_ms && f.chain_seed_step_ms > 4.0 && f.chain_seed_step_ms < 4.5, "draft seeds %.3f",
             f.chain_seed_step_ms);
    const auto l = q::hip_dense_spec_block_config(16U, true, false, 0.264);
    HC_CHECK(!l.fixed_costs && l.chain_max_k == 7U && l.chain_seed_step_ms == 4.0, "learned config = previous form");
    const auto none = q::hip_dense_spec_block_config(16U, false, true, 1.0);
    HC_CHECK(!none.chain && !none.hybrid, "no MTP layer: no chain");

    // 3. decisions on a virtual node
    Node old_node;
    old_node.partial_core = true;
    old_node.spike = 60.0;  // first draft: subset head materialisation + first launches inside the timed wall
    Node new_node;
    new_node.spike = 60.0;
    for (const double a : {0.90, 0.82}) {
        const Result inc = run(false, old_node, a), fix = run(true, new_node, a), fix_old = run(true, old_node, a);
        std::printf("a %.2f: learned+old core E %.2f k %.2f %.1f ms/token | fixed+old core E %.2f k %.2f %.1f ms/token | "
                    "fixed+bucket core E %.2f k %.2f %.1f ms/token\n",
                    a, inc.E, inc.mean_k, inc.ms_per_token, fix_old.E, fix_old.mean_k, fix_old.ms_per_token, fix.E, fix.mean_k,
                    fix.ms_per_token);
        HC_CHECK(fix.mean_k >= 5.0, "fixed model stays shallow: k %.2f", fix.mean_k);
        HC_CHECK(fix.E >= inc.E + 1.0 && fix.ms_per_token < inc.ms_per_token * 0.8, "fixed model not better than learned");
        HC_CHECK(fix.ms_per_token <= fix_old.ms_per_token + 1e-9, "bucket core priced worse than the partial core");
    }
    // determinism: fixed-cost decisions under different wall jitter
    Node j1 = new_node, j2 = new_node;
    j1.jitter = 3.0; j1.seed = 11; j2.jitter = 3.0; j2.seed = 977; j2.spike = 0.0;
    const Result d0 = run(true, new_node, 0.86), d1 = run(true, j1, 0.86), d2 = run(true, j2, 0.86);
    HC_CHECK(d0.ks == d1.ks && d0.ks == d2.ks, "fixed-cost decisions depend on the wall clock");
    std::printf("hip dense spec costs: all checks passed\n");
    return 0;
}
