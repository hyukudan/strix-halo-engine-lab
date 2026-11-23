#pragma once

// F1 `rows`, EXL3 split-K job-list form: the host plan of one launch group of dense EXL3 small-M linears (several
// matrices, each with its own input and output, one job table). Backend-neutral integers only: the CTA ranges of the
// jobs in the core grid, the per-job counter slots of the 128-output blocks, the workspace layout (counters, the
// transformed F16 inputs, the FP32 split partials) and the LDS row strides of the staged-input schedules.
//
// Numerics: the plan fixes the split count of every job (the SplitRule trait, clamped to one split per K16 tile) and
// therefore the chain order of every output; it is independent of the number of jobs in the group and of the row
// count (partials are laid out for the row capacity). Kernels consume the layout unchanged.
//
// SplitRule: `static constexpr unsigned splits(unsigned input, unsigned output)` (the engine binds the strix-trellis
// vector split policy of its decoder). Limits are data: job-table capacity, row capacity, counter words.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strix::core::exl3_rows {

struct Matrix {
    std::uint32_t input = 0, output = 0, bits = 0;
    // Output column range [col_begin, col_begin + col_count), multiples of 128; 0/0 = the whole output. Single job.
    std::uint32_t col_begin = 0, col_count = 0;
};

struct Limits {
    unsigned max_jobs = 0;  // <= MaxJobs of the layout
    unsigned max_rows = 0;  // row capacity: partial and input-transform buffers are sized for it
    unsigned counters = 0;  // u32 counter words at the workspace start; the last two are reserved
};

struct JobLayout {
    std::uint32_t splits = 0, cta_begin = 0, block_begin = 0;  // CTAs of this job in the core grid; counter base
    std::size_t xt_off = 0, part_off = 0;                      // halves / floats into the workspace
    std::uint32_t prep_begin = 0;                               // first (row, 128-block) input-transform item
};

template <unsigned MaxJobs> struct Layout {
    JobLayout job[MaxJobs]{};
    unsigned count = 0, rows = 0;
    unsigned xs_stride = 0;         // LDS row stride (halves) = the largest split of the group's jobs
    unsigned xs_stride5 = 0;        // LDS row stride (halves) = the largest 128-aligned span covering one split
    unsigned prep_jobs = 0;         // rows x input / 128 summed over the jobs
    unsigned inline_prep_ctas = 0;  // leading CTAs of a one-launch form that run the input transforms
    unsigned col_begin = 0, col_count = 0;
    unsigned ctas = 0;              // core grid
    unsigned prepare_grid = 0;      // CTAs (8 waves, one (row, block) item per wave) of a separate prepare launch
    std::size_t counter_off = 0, bytes = 0;
};

constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) & ~(a - 1); }

template <class SplitRule, unsigned MaxJobs>
Layout<MaxJobs> plan(const Matrix* jobs, unsigned count, unsigned rows, const Limits& lim) {
    if (lim.max_jobs > MaxJobs || lim.counters < 2) throw std::invalid_argument("exl3 rows plan: limits exceed the layout");
    if (count < 1 || count > lim.max_jobs) throw std::invalid_argument("exl3 rows plan: job count outside 1..max_jobs");
    if (rows < 1 || rows > lim.max_rows) throw std::invalid_argument("exl3 rows plan: rows outside 1..max_rows");
    Layout<MaxJobs> p;
    p.count = count;
    p.rows = rows;
    std::size_t xt = 0, part = 0;
    unsigned blocks = 0, max_in = 0;
    for (unsigned i = 0; i < count; ++i) {
        const Matrix& w = jobs[i];
        if (w.input % 128 || w.output % 128 || w.bits < 3 || w.bits > 8 || w.bits == 7)
            throw std::invalid_argument("exl3 rows plan: input/output multiples of 128, bits 3, 4, 5, 6 or 8");
        JobLayout& j = p.job[i];
        j.splits = std::min(SplitRule::splits(w.input, w.output), w.input / 16);
        j.cta_begin = p.ctas;
        j.block_begin = blocks;
        if (w.col_count != 0 || w.col_begin != 0) {
            if (count != 1 || w.col_count == 0 || w.col_begin % 128 || w.col_count % 128 || w.col_begin + w.col_count > w.output)
                throw std::invalid_argument("exl3 rows plan: column range needs one job and 128-aligned columns inside the output");
            p.col_begin = w.col_begin;
            p.col_count = w.col_count;
        }
        j.xt_off = xt;
        j.part_off = part;
        p.ctas += (p.col_count != 0 ? p.col_count / 64 : w.output / 64) * j.splits;
        blocks += w.output / 128;
        xt = xt + align_up(std::size_t{lim.max_rows} * w.input, 128);
        part = part + align_up(std::size_t{j.splits} * lim.max_rows * w.output, 64);
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
    // Counters at a fixed offset (every call of a workspace finds them at 0 and leaves them at 0), then xt (halves),
    // then partials (floats).
    if (blocks > lim.counters - 2) throw std::invalid_argument("exl3 rows plan: too many output blocks");
    const std::size_t cnt_bytes = align_up(std::size_t{lim.counters} * 4, 256);
    const std::size_t xt_bytes = align_up(xt * 2, 256);
    for (unsigned i = 0; i < count; ++i) {
        p.job[i].xt_off += cnt_bytes / 2;
        p.job[i].part_off += (cnt_bytes + xt_bytes) / 4;
    }
    p.counter_off = 0;
    p.bytes = cnt_bytes + xt_bytes + align_up(part * 4, 256);
    p.prepare_grid = std::max(1U, (rows * (max_in / 128) + 7) / 8);
    p.inline_prep_ctas = std::min(64U, (p.prep_jobs + 3) / 4);
    return p;
}

}  // namespace strix::core::exl3_rows
