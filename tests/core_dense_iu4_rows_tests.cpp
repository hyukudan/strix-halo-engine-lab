// Host test of the EXACT IU4 dense core's wide-window row plan (src/core/dense_iu4_rows.hpp, used by
// core/dense_linear_iu4.hpp kernel<..., Grid2> and run() for rows > 16).
// Invariant: one launch over a window of rows > 16 in kWideRows-row blocks stores every row exactly once and bitwise
// equal to the incumbent 16-row sweeps (dense_row_sweeps split; R16 bucket = two 8-row passes), pad rows are read only
// inside the caller's xh_rows(rows) scratch and never reach a stored row. The arithmetic is a CPU model of the contract's
// per-row structure (8 K ranges; per K64 group a chained pair-product sum, then fmaf by the group scale; ranges summed
// in ascending order); the row-block plan is the shared header itself.
#include "core/dense_iu4_rows.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
namespace plan = strix::core::dense_iu4_rows;

int failures = 0;
void check(bool ok, const char* what, unsigned rows) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL rows %u: %s\n", rows, what);
    }
}

constexpr unsigned kN = 8U, kRanges = 8U, kGroupsPerRange = 2U, kK = kRanges * kGroupsPerRange * 64U;

struct Model {
    std::vector<float> q;      // [kN][kK] integer codes -8..7
    std::vector<float> scale;  // [kN][kK / 64]
    // One output of one row: the contract's order (pairs in a group, groups in a range, ranges ascending).
    float dot(const float* x, unsigned n) const {
        float ranges[kRanges];
        for (unsigned w = 0; w < kRanges; ++w) {
            float acc = 0.0F;
            for (unsigned g = w * kGroupsPerRange; g < (w + 1U) * kGroupsPerRange; ++g) {
                float d = 0.0F;
                for (unsigned k = g * 64U; k < g * 64U + 64U; k += 2U)
                    d = std::fma(q[std::size_t(n) * kK + k + 1U], x[k + 1U], std::fma(q[std::size_t(n) * kK + k], x[k], d));
                acc = std::fma(scale[std::size_t(n) * (kK / 64U) + g], d, acc);
            }
            ranges[w] = acc;
        }
        float sum = ranges[0];
        for (unsigned w = 1; w < kRanges; ++w) sum += ranges[w];
        return sum;
    }
};

struct Scratch {
    std::vector<float> xh;       // [capacity rows][kK]
    std::vector<float> out;      // [rows][kN]
    std::vector<unsigned> hits;  // stores per row
    unsigned capacity = 0, rows = 0;
    bool read_past = false;
};

// One core pass over `pass` rows starting at window row `first`: computes every pass row (pad rows included, as the
// kernel does) and stores rows < first + stored only.
void core_pass(const Model& m, Scratch& s, unsigned first, unsigned pass, unsigned stored) {
    for (unsigned r = 0; r < pass; ++r) {
        const unsigned row = first + r;
        if (row >= s.capacity) { s.read_past = true; continue; }
        for (unsigned n = 0; n < kN; ++n) {
            const float v = m.dot(&s.xh[std::size_t(row) * kK], n);
            if (r < stored) s.out[std::size_t(row) * kN + n] = v;
        }
        if (r < stored) ++s.hits[row];
    }
}
// Incumbent: dense_row_sweeps (16-row sweeps, tail in powers of two); a run() of `count` rows takes bucket
// R = next power of two >= count, passes of min(R, 8) rows while a pass holds a stored row.
void sweeps16(const Model& m, Scratch& s, unsigned rows) {
    unsigned at = 0;
    for (unsigned tile = plan::kMaxRows; tile; tile /= 2U)
        while (rows - at >= tile) {
            const unsigned pass = tile > 8U ? 8U : tile;
            for (unsigned h = 0; h * pass < tile; ++h) core_pass(m, s, at + h * pass, pass, pass);
            at += tile;
        }
}
// Proposed: one launch, blocks of the shared plan (each block computes kWideRows rows, stores b.rows).
void wide(const Model& m, Scratch& s, unsigned rows) {
    for (unsigned b = 0; b < plan::blocks(rows); ++b) {
        const plan::Block blk = plan::block(b, rows);
        check(blk.rows >= 1U && blk.rows <= plan::kWideRows && blk.row0 + blk.rows <= rows, "block bounds", rows);
        core_pass(m, s, blk.row0, plan::kWideRows, blk.rows);
    }
}

Scratch make(const std::vector<float>& x, unsigned rows, unsigned capacity) {
    Scratch s;
    s.rows = rows;
    s.capacity = capacity;
    s.xh.assign(std::size_t(capacity) * kK, std::numeric_limits<float>::quiet_NaN());  // pad rows poison any leak
    std::memcpy(s.xh.data(), x.data(), std::size_t(rows) * kK * sizeof(float));
    s.out.assign(std::size_t(rows) * kN, std::numeric_limits<float>::infinity());
    s.hits.assign(capacity, 0U);
    return s;
}
}  // namespace

int main() {
    std::mt19937 rng(7U);
    std::uniform_int_distribution<int> code(-8, 7);
    std::uniform_real_distribution<float> sc(0.002F, 0.03F);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    Model m;
    m.q.resize(std::size_t(kN) * kK);
    for (auto& v : m.q) v = static_cast<float>(code(rng));
    m.scale.resize(std::size_t(kN) * (kK / 64U));
    for (auto& v : m.scale) v = sc(rng);
    constexpr unsigned kMaxWindow = 520U;
    std::vector<float> x(std::size_t(kMaxWindow) * kK);
    for (auto& v : x) v = nd(rng);

    // plan arithmetic
    check(!plan::wide(16U) && plan::wide(17U), "wide() threshold is rows > 16", 16U);
    check(plan::xh_rows(17U) == 24U && plan::xh_rows(512U) == 512U && plan::blocks(17U) == 3U, "xh_rows / blocks", 17U);
    unsigned checked = 0;
    for (unsigned rows = 17U; rows <= kMaxWindow; rows += (rows < 80U ? 1U : 37U)) {
        Scratch a = make(x, rows, unsigned(plan::xh_rows(rows)) + plan::kMaxRows);  // sweeps may read their bucket's pad rows
        Scratch b = make(x, rows, unsigned(plan::xh_rows(rows)));                   // wide: exactly the documented capacity
        sweeps16(m, a, rows);
        wide(m, b, rows);
        check(!b.read_past, "wide grid reads past xh_rows(rows)", rows);
        bool once = true;
        for (unsigned r = 0; r < b.capacity; ++r) once = once && b.hits[r] == (r < rows ? 1U : 0U);
        check(once, "every row stored exactly once, no store past the window", rows);
        check(std::memcmp(a.out.data(), b.out.data(), a.out.size() * sizeof(float)) == 0, "stored rows bitwise equal to 16-row sweeps", rows);
        bool finite = true;
        for (float v : b.out) finite = finite && std::isfinite(v);
        check(finite, "no pad row (NaN) reached a stored row", rows);
        ++checked;
    }
    if (failures) {
        std::fprintf(stderr, "dense iu4 rows: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dense iu4 rows: wide grid == 16-row sweeps bitwise, stored once, pads contained (%u windows 17..%u)\n", checked, kMaxWindow);
    return 0;
}
