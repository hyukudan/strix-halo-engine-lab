// Host tests of strix/attention_split_plan.hpp (no GPU): every visible key lies in exactly one chunk, splits stay within
// the launch capacity, chunks are kUnit multiples, and the split + ascending merge (sink first) of the decode contract
// reproduces a serial softmax on random rows (CPU emulation of the CUDA schedule in FP32 against double).
#include "strix/attention_split_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace sp = strix::attn_split;

namespace {

int failures = 0;
void expect(bool ok, const char* what, unsigned a, unsigned b) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s (%u, %u)\n", what, a, b);
    }
}

void plan_cover(unsigned position, unsigned window, unsigned slots) {
    const auto p = sp::decode_split_plan(position, window);
    const unsigned lo = window && position >= window ? position - window + 1U : 0U;
    expect(p.first == lo, "first key", position, window);
    expect(p.keys == position - lo + 1U, "key count", position, window);
    expect(p.chunk % sp::kUnit == 0 && p.chunk >= sp::kUnit, "chunk unit", position, p.chunk);
    expect(p.splits >= 1 && p.splits <= sp::kMaxSplits, "split bound", position, p.splits);
    expect(p.splits <= sp::decode_split_capacity(slots, window), "capacity", position, p.splits);
    // the kernel finalises a single-chunk row in registers (no record, no ticket): exactly the rows with <= kUnit keys
    expect((p.splits == 1U) == (p.keys <= sp::kUnit), "single chunk iff keys <= unit", position, p.keys);
    // chunk c = [first + c chunk, min(first + (c + 1) chunk, position + 1)): non-empty, contiguous, covers the keys once
    unsigned covered = 0;
    for (unsigned c = 0; c < p.splits; ++c) {
        const unsigned b = p.first + c * p.chunk, e = std::min(b + p.chunk, position + 1U);
        expect(b < e, "non-empty chunk", position, c);
        covered += e - b;
    }
    expect(covered == p.keys, "cover", position, covered);
    expect(p.first + p.splits * p.chunk >= position + 1U, "last chunk reaches the row", position, p.splits);
}

// CPU emulation of one row and one head: chunks of the plan, passes of kUnit keys with the online max (log2 domain),
// then the merge sink first + chunks ascending; versus a serial double softmax.
double split_vs_serial(unsigned position, unsigned window, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const auto p = sp::decode_split_plan(position, window);
    std::vector<float> s(p.keys), v(p.keys);
    for (unsigned i = 0; i < p.keys; ++i) { s[i] = nd(rng); v[i] = nd(rng); }
    const float sink = nd(rng);
    constexpr float log2e = 1.4426950408889634f;
    std::vector<float> cm(p.splits), cl(p.splits), co(p.splits);
    for (unsigned c = 0; c < p.splits; ++c) {
        const unsigned b = c * p.chunk, e = std::min(b + p.chunk, p.keys);
        float m = -INFINITY, l = 0, o = 0;
        for (unsigned base = b; base < e; base += sp::kUnit) {
            const unsigned end = std::min(base + sp::kUnit, e);
            float bm = -INFINITY;
            for (unsigned k = base; k < end; ++k) bm = std::max(bm, s[k] * log2e);
            const float nm = std::max(m, bm), alpha = std::exp2(m - nm);
            float bs = 0, bo = 0;
            for (unsigned k = base; k < end; ++k) {
                const float pk = std::exp2(s[k] * log2e - nm);
                bs += pk;
                bo = std::fma(pk, v[k], bo);
            }
            l = std::fma(l, alpha, bs);
            o = o * alpha + bo;
            m = nm;
        }
        cm[c] = m; cl[c] = l; co[c] = o;
    }
    float mx = sink * log2e;
    for (unsigned c = 0; c < p.splits; ++c) mx = std::max(mx, cm[c]);
    float den = std::exp2(sink * log2e - mx), num = 0;
    for (unsigned c = 0; c < p.splits; ++c) {
        const float w = std::exp2(cm[c] - mx);
        den = std::fma(w, cl[c], den);
        num = std::fma(w, co[c], num);
    }
    double rm = sink, rden = 0, rnum = 0;
    for (unsigned i = 0; i < p.keys; ++i) rm = std::max(rm, static_cast<double>(s[i]));
    rden = std::exp(static_cast<double>(sink) - rm);
    for (unsigned i = 0; i < p.keys; ++i) {
        const double e = std::exp(static_cast<double>(s[i]) - rm);
        rden += e;
        rnum += e * v[i];
    }
    return std::fabs(static_cast<double>(num / den) - rnum / rden);
}

// Lanes form: a lane's chunk records live in its own slot of the block workspace. Every lane of a block writes the
// records of its plan (all lanes first, interleaved), then every lane merges its own records (sink first, chunks
// ascending). Each lane's result must equal, bit for bit, the same row run alone in a one-row workspace.
float lane_merge(const std::vector<float>& ws, unsigned long long base, const sp::SplitLaneLayout& lay, unsigned kvh,
                 unsigned g, unsigned splits, float sink2) {
    const auto rec = [&](unsigned c) { return base + ((1ULL * kvh * lay.capacity + c) * lay.group + g) * lay.record; };
    float mx = sink2;
    for (unsigned c = 0; c < splits; ++c) mx = std::max(mx, ws[rec(c)]);
    float den = std::exp2(sink2 - mx), num = 0;
    for (unsigned c = 0; c < splits; ++c) {
        const float w = std::exp2(ws[rec(c)] - mx);
        den = std::fma(w, ws[rec(c) + 1], den);
        num = std::fma(w, ws[rec(c) + 2], num);
    }
    return num / den;
}

void write_records(std::vector<float>& ws, unsigned long long base, const sp::SplitLaneLayout& lay, unsigned splits,
                   unsigned seed) {
    for (unsigned kvh = 0; kvh < lay.kv_heads; ++kvh)
        for (unsigned c = 0; c < splits; ++c)
            for (unsigned g = 0; g < lay.group; ++g) {
                const auto r = base + ((1ULL * kvh * lay.capacity + c) * lay.group + g) * lay.record;
                const unsigned h = (seed * 2654435761U) ^ (kvh * 97U + c * 13U + g);
                ws[r] = static_cast<float>(h % 1000U) * 0.01f;
                ws[r + 1] = 1.0f + static_cast<float>(h % 37U);
                ws[r + 2] = static_cast<float>(h % 101U) - 50.0f;
            }
}

void lanes_invariant(std::mt19937& rng) {
    const unsigned positions = 6000, window = 0;
    sp::SplitLaneLayout lay;
    lay.kv_heads = 8; lay.capacity = sp::decode_split_capacity(positions, window); lay.group = 3; lay.record = 130;
    lay.lane_slots = 16;
    expect(lay.ticket_words() % 4U == 0 && lay.ticket_words() >= lay.kv_heads * lay.lane_slots, "lane tickets", 0, 0);
    expect(lay.bytes() == 4ULL * lay.ticket_words() + 4ULL * lay.lane_slots * lay.lane_floats(), "lane bytes", 0, 0);
    for (unsigned l = 0; l + 1U < lay.lane_slots; ++l) {
        expect(lay.lane_ticket(l) + lay.kv_heads == lay.lane_ticket(l + 1U), "ticket slices disjoint", l, 0);
        expect(lay.lane_partials(l) + lay.lane_floats() == lay.lane_partials(l + 1U), "record slices disjoint", l, 0);
    }
    for (unsigned pos = 0; pos < positions; ++pos) expect(sp::lane_fits(pos, window, lay.capacity), "lane fits", pos, 0);
    std::uniform_int_distribution<unsigned> pick(0, positions - 1U), width(1, lay.lane_slots);
    for (int trial = 0; trial < 50; ++trial) {
        const unsigned count = width(rng);
        std::vector<unsigned> pos(count);
        for (auto& p : pos) p = pick(rng);
        std::vector<float> block(lay.lane_slots * lay.lane_floats(), -1.0f);
        for (unsigned l = 0; l < count; ++l)
            write_records(block, lay.lane_partials(l), lay, sp::decode_split_plan(pos[l], window).splits, pos[l]);
        for (unsigned l = 0; l < count; ++l) {
            const unsigned splits = sp::decode_split_plan(pos[l], window).splits;
            std::vector<float> alone(lay.lane_floats(), -2.0f);
            write_records(alone, 0, lay, splits, pos[l]);
            for (unsigned kvh = 0; kvh < lay.kv_heads; ++kvh)
                for (unsigned g = 0; g < lay.group; ++g) {
                    const float a = lane_merge(block, lay.lane_partials(l), lay, kvh, g, splits, 0.5f);
                    const float b = lane_merge(alone, 0, lay, kvh, g, splits, 0.5f);
                    expect(std::memcmp(&a, &b, sizeof a) == 0, "lane bits == one-row bits", pos[l], count);
                }
        }
    }
}

}  // namespace

int main() {
    static_assert(sp::decode_split_plan(0, 0).splits == 1 && sp::decode_split_plan(0, 0).keys == 1);
    static_assert(sp::decode_split_plan(255, 0).splits == 1 && sp::decode_split_plan(256, 0).splits == 2);
    static_assert(sp::decode_split_plan(2520, 1024).splits == 4 && sp::decode_split_plan(2520, 1024).first == 1497);
    static_assert(sp::decode_split_plan(2520, 0).splits == 10);
    static_assert(sp::decode_split_plan(16383, 0).splits == 64 && sp::decode_split_plan(16384, 0).chunk == 512);
    static_assert(sp::decode_split_capacity(1280, 1024) == 4 && sp::decode_split_capacity(16384, 0) == 64);
    static_assert(sp::decode_split_capacity(1U << 20U, 0) == 64 && sp::decode_split_capacity(100, 0) == 1);

    const unsigned slots_full = 1U << 20U;
    for (unsigned pos = 0; pos < 20000; ++pos) {
        plan_cover(pos, 0, slots_full);
        plan_cover(pos, 1024, 1280);
    }
    for (unsigned pos : {65535U, 65536U, 131071U, 262143U, 1048575U}) plan_cover(pos, 0, slots_full);

    std::mt19937 rng(7);
    double worst = 0;
    for (unsigned pos : {0U, 37U, 255U, 256U, 1023U, 2520U, 8191U, 16383U, 16384U, 40000U}) {
        worst = std::max(worst, split_vs_serial(pos, 0, rng));
        worst = std::max(worst, split_vs_serial(pos, 1024, rng));
    }
    lanes_invariant(rng);
    if (!(worst < 2e-4)) {
        ++failures;
        std::printf("FAIL split emulation vs serial softmax: %.3e\n", worst);
    }
    std::printf("attention split plan: %s (split vs serial max |err| %.3e)\n", failures ? "FAIL" : "PASS", worst);
    return failures ? 1 : 0;
}
