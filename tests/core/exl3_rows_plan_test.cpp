// F1 rows EXL3 split-K job-list plan: the shared plan equals the frozen engine-private plan (every field, every
// throw) over a sweep of shapes, bit widths, row counts, job counts and column ranges.
#include "core/exl3_rows_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

namespace er = strix::core::exl3_rows;

// strix-trellis vector split policy for <= 8 rows (frozen copy of the decoder's two-argument rule).
struct VectorSplit {
    static constexpr unsigned splits(unsigned input, unsigned output) {
        return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
    }
};

constexpr unsigned kMaxJobs = 4, kMaxRows = 8, kLinCounters = 4096;

// ---- frozen private reference (engine plan before the bind; same operation order) ----
struct RefJob {
    std::uint32_t splits, cta_begin, block_begin;
    std::size_t xt_off, part_off;
    std::uint32_t prep_begin;
};
struct RefPlan {
    RefJob job[kMaxJobs]{};
    unsigned count = 0, rows = 0, xs_stride = 0, xs_stride5 = 0, prep_jobs = 0, args_prep_ctas = 0;
    unsigned col_begin = 0, col_count = 0;
    std::size_t counter_off = 0;
    unsigned ctas = 0, prep_ctas = 0;
    std::size_t bytes = 0;
};
constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
RefPlan ref_plan(const er::Matrix* jobs, unsigned count, unsigned rows) {
    if (count < 1 || count > kMaxJobs) throw std::invalid_argument("ref: jobs");
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("ref: rows");
    RefPlan p;
    p.count = count;
    p.rows = rows;
    std::size_t xt = 0, part = 0;
    unsigned blocks = 0, max_in = 0;
    for (unsigned i = 0; i < count; ++i) {
        const er::Matrix& w = jobs[i];
        if (w.input % 128 || w.output % 128 || w.bits < 3 || w.bits > 8 || w.bits == 7) throw std::invalid_argument("ref: shape");
        RefJob& j = p.job[i];
        j.splits = std::min(VectorSplit::splits(w.input, w.output), w.input / 16);
        j.cta_begin = p.ctas;
        j.block_begin = blocks;
        if (jobs[i].col_count != 0 || jobs[i].col_begin != 0) {
            if (count != 1 || jobs[i].col_count == 0 || jobs[i].col_begin % 128 || jobs[i].col_count % 128 ||
                jobs[i].col_begin + jobs[i].col_count > w.output)
                throw std::invalid_argument("ref: cols");
            p.col_begin = jobs[i].col_begin;
            p.col_count = jobs[i].col_count;
        }
        j.xt_off = xt;
        j.part_off = part;
        p.ctas += (p.col_count != 0 ? p.col_count / 64 : w.output / 64) * j.splits;
        blocks += w.output / 128;
        xt += (std::size_t{kMaxRows} * w.input + 127) & ~std::size_t{127};
        part += (std::size_t{j.splits} * kMaxRows * w.output + 63) & ~std::size_t{63};
        max_in = std::max(max_in, w.input);
        p.xs_stride = std::max(p.xs_stride, ((w.input / 16 + j.splits - 1) / j.splits) * 16U);
        const unsigned kt = w.input / 16;
        for (unsigned sp = 0; sp < j.splits; ++sp) {
            const unsigned k0 = kt * sp / j.splits, k1 = kt * (sp + 1) / j.splits;
            p.xs_stride5 = std::max(p.xs_stride5, ((k1 + 7) / 8 - k0 / 8) * 128U);
        }
        j.prep_begin = p.prep_jobs;
        p.prep_jobs += rows * (w.input / 128);
    }
    if (blocks > kLinCounters - 2) throw std::invalid_argument("ref: blocks");
    const std::size_t cnt_bytes = align256(std::size_t{kLinCounters} * 4);
    const std::size_t xt_bytes = align256(xt * 2);
    for (unsigned i = 0; i < count; ++i) {
        p.job[i].xt_off += cnt_bytes / 2;
        p.job[i].part_off += (cnt_bytes + xt_bytes) / 4;
    }
    p.counter_off = 0;
    p.bytes = cnt_bytes + xt_bytes + align256(part * 4);
    p.prep_ctas = std::max(1U, (rows * (max_in / 128) + 7) / 8);
    p.args_prep_ctas = std::min(64U, (p.prep_jobs + 3) / 4);
    return p;
}
// ---- end of frozen reference ----

unsigned long long g_cases = 0, g_throws = 0, g_fail = 0;

void compare(const er::Matrix* jobs, unsigned count, unsigned rows) {
    ++g_cases;
    bool ref_threw = false, new_threw = false;
    RefPlan r{};
    er::Layout<kMaxJobs> s{};
    try { r = ref_plan(jobs, count, rows); } catch (const std::invalid_argument&) { ref_threw = true; }
    try {
        s = er::plan<VectorSplit, kMaxJobs>(jobs, count, rows, er::Limits{kMaxJobs, kMaxRows, kLinCounters});
    } catch (const std::invalid_argument&) { new_threw = true; }
    if (ref_threw || new_threw) {
        ++g_throws;
        if (ref_threw != new_threw) ++g_fail;
        return;
    }
    bool ok = r.count == s.count && r.rows == s.rows && r.xs_stride == s.xs_stride && r.xs_stride5 == s.xs_stride5 &&
              r.prep_jobs == s.prep_jobs && r.args_prep_ctas == s.inline_prep_ctas && r.col_begin == s.col_begin &&
              r.col_count == s.col_count && r.counter_off == s.counter_off && r.ctas == s.ctas &&
              r.prep_ctas == s.prepare_grid && r.bytes == s.bytes;
    for (unsigned i = 0; i < kMaxJobs; ++i) {
        const RefJob& a = r.job[i];
        const er::JobLayout& b = s.job[i];
        ok = ok && a.splits == b.splits && a.cta_begin == b.cta_begin && a.block_begin == b.block_begin &&
             a.xt_off == b.xt_off && a.part_off == b.part_off && a.prep_begin == b.prep_begin;
    }
    if (!ok) {
        if (g_fail < 8) std::printf("MISMATCH count=%u rows=%u in0=%u out0=%u\n", count, rows, jobs[0].input, jobs[0].output);
        ++g_fail;
    }
}

}  // namespace

int main() {
    const std::uint32_t dims[] = {128, 256, 384, 512, 640, 1024, 1152, 1536, 2048, 3072, 4096, 4224, 6144, 8192, 12288, 100};
    const std::uint32_t bits[] = {2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<er::Matrix> pool;
    for (std::uint32_t in : dims)
        for (std::uint32_t out : dims)
            for (std::uint32_t b : bits) pool.push_back(er::Matrix{in, out, b, 0, 0});
    // single jobs, every row count (0..9 covers both throws), column ranges inside / outside / misaligned
    for (const er::Matrix& m : pool)
        for (unsigned rows = 0; rows <= kMaxRows + 1; ++rows) {
            compare(&m, 1, rows);
            const std::uint32_t cb[] = {0, 128, 256, 100}, cc[] = {0, 128, 384, 64};
            for (std::uint32_t b0 : cb)
                for (std::uint32_t c0 : cc) {
                    er::Matrix c = m;
                    c.col_begin = b0;
                    c.col_count = c0;
                    compare(&c, 1, rows);
                }
        }
    // groups of 2..4 (and 0 / 5) jobs over a strided walk of the pool, with a column range on a later job
    const std::size_t n = pool.size();
    for (std::size_t i = 0; i < n; i += 3)
        for (unsigned count = 0; count <= kMaxJobs + 1; ++count)
            for (unsigned rows = 1; rows <= kMaxRows; ++rows) {
                er::Matrix g[kMaxJobs + 1];
                for (unsigned k = 0; k <= kMaxJobs; ++k) g[k] = pool[(i + 37 * k + rows) % n];
                compare(g, count, rows);
                if (count >= 2) {
                    g[count - 1].col_count = 128;
                    compare(g, count, rows);
                }
            }
    // counter capacity: many 128-output blocks in one group
    {
        er::Matrix big[kMaxJobs];
        for (auto& m : big) m = er::Matrix{1024, 131072, 4, 0, 0};
        compare(big, 4, 8);
        compare(big, 3, 8);
    }
    // Invalid caller geometry must be rejected before division and without 32-bit range wrap.
    {
        const auto rejects = [](const er::Matrix& m) {
            try {
                (void)er::plan<VectorSplit, kMaxJobs>(&m, 1, 1, er::Limits{kMaxJobs, kMaxRows, kLinCounters});
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        const er::Matrix bad[] = {
            {0, 128, 4}, {128, 0, 4},
            {128, 128, 4, 0xffffff80U, 256}, // overflowed col_begin + col_count looked valid
            {256, 256, 4, 128, 0xffffff80U}, // in-range begin, overflowing col_count wraps the sum
            {128, 128, 4, 128, 128},         // range starts at output end
        };
        for (const auto& m : bad) {
            ++g_cases;
            if (!rejects(m)) { ++g_fail; std::printf("accepted invalid EXL3 matrix geometry\n"); }
            else ++g_throws;
        }
        struct ZeroSplit { static constexpr unsigned splits(unsigned, unsigned) { return 0U; } };
        const er::Matrix valid{128, 128, 4};
        ++g_cases;
        try {
            (void)er::plan<ZeroSplit, kMaxJobs>(&valid, 1, 1, er::Limits{kMaxJobs, kMaxRows, kLinCounters});
            ++g_fail;
            std::printf("accepted zero EXL3 split count\n");
        } catch (const std::invalid_argument&) { ++g_throws; }
    }
    // launch-shape rejection happens before any descriptor is read: a null table must throw, never be dereferenced
    for (const unsigned count : {0U, 5U, 9U}) {
        bool threw = false;
        try { (void)er::plan<VectorSplit, kMaxJobs>(nullptr, count, 8, er::Limits{kMaxJobs, kMaxRows, kLinCounters}); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { std::printf("null table, count %u: no throw\n", count); ++g_fail; }
    }
    for (const unsigned rows : {0U, 9U}) {
        bool threw = false;
        try { (void)er::plan<VectorSplit, kMaxJobs>(nullptr, 1, rows, er::Limits{kMaxJobs, kMaxRows, kLinCounters}); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { std::printf("null table, rows %u: no throw\n", rows); ++g_fail; }
    }
    std::printf("exl3 rows plan: %llu cases (%llu throwing), %llu mismatches\n", g_cases, g_throws, g_fail);
    return g_fail == 0 && g_cases > g_throws ? 0 : 1;
}
