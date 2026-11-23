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
    // chunks are non-empty, contiguous, ascending and cover the keys once; passes of one chunk stay within kUnit lanes
    unsigned covered = 0, next = p.first;
    for (unsigned c = 0; c < p.splits; ++c) {
        const unsigned o = p.chunk_origin(c), b = p.chunk_begin(c), e = p.chunk_end(c);
        expect(b == next && b < e, "contiguous non-empty chunk", position, c);
        expect(o <= b && e - o <= p.chunk, "passes of a chunk fit the chunk", position, c);
        if (p.keys > sp::kUnit) {
            // position canonical: the chunk is the absolute aligned interval [a chunk, (a + 1) chunk) clipped to the row
            const unsigned aligned = (p.first_chunk + c) * p.chunk;
            expect(o == (c == 0U ? p.first / sp::kUnit * sp::kUnit : aligned), "absolute unit-aligned origin", position, c);
            expect(b == std::max(aligned, p.first) && e == std::min(aligned + p.chunk, position + 1U),
                   "clipped aligned chunk", position, c);
        } else {
            expect(o == p.first && e == position + 1U, "short row is one pass from first", position, c);
        }
        covered += e - b;
        next = e;
    }
    expect(covered == p.keys && next == position + 1U, "cover", position, covered);
}

// The uncapped unit shape is the absolute 256-key split of the global-max decode family: split index = key / 256.
void unit_shape(unsigned position, unsigned window) {
    const auto p = sp::decode_split_plan(position, window, sp::kUnitShape);
    const unsigned lo = window && position >= window ? position - window + 1U : 0U;
    expect(p.chunk == 256U && p.first_chunk == lo / 256U, "unit shape chunk", position, window);
    expect(p.splits == position / 256U - lo / 256U + 1U, "unit shape splits", position, window);
    for (unsigned c = 0; c < p.splits; ++c) {
        const unsigned s = lo / 256U + c;
        expect(p.chunk_begin(c) == std::max(lo, s * 256U) && p.chunk_end(c) == std::min(position + 1U, s * 256U + 256U),
               "unit shape split = key / 256", position, c);
    }
}

// Where the earlier origin-at-first plan was measured (window 0 up to 16384 keys, and every short row), the chunks are
// the same: first + c * chunk with chunk = 256.
void prior_domain(unsigned position, unsigned window) {
    const auto p = sp::decode_split_plan(position, window);
    const unsigned lo = window && position >= window ? position - window + 1U : 0U;
    const unsigned keys = position - lo + 1U;
    if (!((window == 0U && keys <= 16384U) || keys <= 256U)) return;
    expect(p.chunk == 256U && p.splits == (keys + 255U) / 256U, "prior domain splits", position, window);
    for (unsigned c = 0; c < p.splits; ++c)
        expect(p.chunk_origin(c) == lo + c * 256U && p.chunk_end(c) == std::min(lo + c * 256U + 256U, position + 1U),
               "prior domain chunk", position, c);
}

// Capacity covers every position below it (running maximum of splits), for each window.
void capacity_bound(unsigned limit, unsigned window) {
    unsigned worst = 0;
    for (unsigned pos = 0; pos < limit; ++pos) {
        worst = std::max(worst, sp::decode_split_plan(pos, window).splits);
        expect(worst <= sp::decode_split_capacity(pos + 1U, window), "capacity covers splits", pos, window);
    }
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
        // absolute passes from the chunk origin, keys before `first` masked (indices relative to first)
        float m = -INFINITY, l = 0, o = 0;
        for (unsigned abase = p.chunk_origin(c); abase < p.chunk_end(c); abase += sp::kUnit) {
            if (abase + sp::kUnit <= p.chunk_begin(c)) continue;  // fully masked pass: m, l, o unchanged
            const unsigned base = std::max(abase, p.chunk_begin(c)) - p.first;
            const unsigned end = std::min(abase + sp::kUnit, p.chunk_end(c)) - p.first;
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
    static_assert(sp::decode_split_plan(2520, 1024).splits == 5 && sp::decode_split_plan(2520, 1024).first == 1497);
    static_assert(sp::decode_split_plan(2559, 1024).splits == 4 && sp::decode_split_plan(300, 128).splits == 1);
    static_assert(sp::decode_split_plan(300, 128).chunk_origin(0) == 173 && sp::decode_split_plan(2520, 1024).origin == 1280 &&
                  sp::decode_split_plan(20000U, 15000U).origin == 4864U && sp::decode_split_plan(20000U, 15000U).chunk == 256U);
    static_assert(sp::decode_split_plan(2520, 0).splits == 10);
    static_assert(sp::decode_split_plan(16383, 0).splits == 64 && sp::decode_split_plan(16384, 0).chunk == 512);
    static_assert(sp::decode_split_capacity(1280, 1024) == 5 && sp::decode_split_capacity(16384, 0) == 64);
    static_assert(sp::decode_split_capacity(1U << 20U, 0, sp::kUnitShape) == 4096U);
    static_assert(sp::decode_split_capacity(1U << 20U, 0) == 64 && sp::decode_split_capacity(100, 0) == 1);

    const unsigned slots_full = 1U << 20U;
    for (unsigned pos = 0; pos < 20000; ++pos) {
        plan_cover(pos, 0, slots_full);
        plan_cover(pos, 1024, 1280);
    }
    for (unsigned pos : {65535U, 65536U, 131071U, 262143U, 1048575U}) plan_cover(pos, 0, slots_full);
    for (unsigned window : {0U, 1U, 100U, 255U, 256U, 257U, 1000U, 1024U, 4096U, 16128U, 16384U, 20000U, 70000U}) {
        for (unsigned pos = 0; pos < (1U << 17U); pos += window > 4096U ? 1U : 3U) {
            plan_cover(pos, window, slots_full);
            unit_shape(pos, window);
            prior_domain(pos, window);
        }
        capacity_bound(1U << 17U, window);
    }

    std::mt19937 rng(7);
    double worst = 0;
    for (unsigned pos : {0U, 37U, 255U, 256U, 1023U, 2520U, 8191U, 16383U, 16384U, 40000U}) {
        worst = std::max(worst, split_vs_serial(pos, 0, rng));
        worst = std::max(worst, split_vs_serial(pos, 1024, rng));
        worst = std::max(worst, split_vs_serial(pos, 200, rng));
        worst = std::max(worst, split_vs_serial(pos + 70000U, 20000, rng));
    }
    lanes_invariant(rng);
    if (!(worst < 2e-4)) {
        ++failures;
        std::printf("FAIL split emulation vs serial softmax: %.3e\n", worst);
    }
    std::printf("attention split plan: %s (split vs serial max |err| %.3e)\n", failures ? "FAIL" : "PASS", worst);
    return failures ? 1 : 0;
}
