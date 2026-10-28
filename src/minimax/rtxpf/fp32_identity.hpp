#pragma once
// minimax-m3 FP32 prefill identity. Declared order and kernels in fp32_identity_device.inc,
// instantiated verbatim for gfx1151 (fp32_identity.hip) and the RTX PRO (cuda/fp32_identity.cu). This header is
// platform-neutral (streams are passed as void*: hipStream_t / cudaStream_t). Engine opt-in: STRIX_MM_PREFILL_IDENTITY=fp32.
//
// A grouped call runs several EXL3 matrices with the SAME input and output sizes (one dense site, or all the experts'
// gate / up / down slices of a MoE layer) over a common row space of `entries` rows:
//   xt[e]   = H_in(suh_j * x[xmap ? xmap[e] : e])       (job j = the job whose [row_begin, row_begin + rows) holds e)
//   raw[e]  = W_j * xt[e]                                (fmaf chain over k ascending)
//   y[ymap ? ymap[e] : e] = svh_j * H_out(raw[e])
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::minimax::rtxpf {

struct F32Job {  // same layout as the device struct in fp32_identity_device.inc
    const std::uint32_t* trellis;
    const std::uint16_t* suh;
    const std::uint16_t* svh;
    std::uint32_t input, output, bits;
    std::uint32_t rows;
    std::uint32_t row_begin;
    std::uint32_t tile_begin;
};
// Any EXL3 view with trellis / suh / svh / input / output / bits (e.g. moe::opt::Exl3).
template <class W> F32Job job_of(const W& w, std::uint32_t rows) {
    return {w.trellis, w.suh, w.svh, w.input, w.output, w.bits, rows, 0, 0};
}

struct GroupedCall {
    const F32Job* jobs_d = nullptr;  // device copy of the job table (laid out by layout_jobs)
    std::uint32_t jobs = 0;
    std::uint32_t tiles = 0;         // total core CTAs
    std::uint32_t tile_rows = 64;    // core CTA height given to layout_jobs (64 or 128); same bits either way
    std::uint32_t tail_jobs = 0;     // tile_rows 128 only: a 64-row core over jobs_d[jobs, jobs + tail_jobs) (split_tails)
    std::uint32_t tail_tiles = 0;
    std::uint32_t input = 0, output = 0, entries = 0;
    const float* x = nullptr;
    std::uint32_t x_stride = 0;
    const std::uint32_t* xmap = nullptr;  // device, entries (nullptr: identity)
    float* y = nullptr;
    std::uint32_t y_stride = 0;
    const std::uint32_t* ymap = nullptr;  // device, entries (nullptr: identity)
    // CUDA: 0 = environment/default (v3.1), 3 = v3, 31 = v3.1 (exact dp4a decode).
    // STRIX_MM_FP32_CORE=3 restores v3. Explicit selection overrides the environment. HIP ignores this field.
    std::uint32_t core_version = 0;
};

// Pure CUDA selector, also checked without a device. Unrecognized values use v3.1.
inline std::uint32_t resolve_core_version(std::uint32_t requested, const char* env) {
    if (requested != 0) return requested == 3 ? 3U : 31U;
    return env && env[0] == '3' && env[1] == 0 ? 3U : 31U;
}

struct Extent {
    std::uint32_t entries = 0, tiles = 0;
};
// Fills row_begin / tile_begin of `jobs` (rows and matrices already set); returns the row space and the CTA count.
Extent layout_jobs(std::vector<F32Job>& jobs, std::uint32_t tile_rows = 64);
// Mixed tile heights after layout_jobs(jobs, 128): a job's last rows % 128 rows, when <= 64, go to a 64-row tail core
// instead of a mostly empty 128-row CTA. jobs keep row_begin (transform / output find them by it); their rows shrink and
// tile_begin is recomputed; tail gets one job per moved remainder. Every output keeps its own fmaf chain: same bits.
// Returns the tail CTA count and sets tiles128.
inline std::uint32_t split_tails(std::vector<F32Job>& jobs, std::vector<F32Job>& tail, std::uint32_t& tiles128) {
    std::uint32_t t64 = 0;
    tiles128 = 0;
    tail.clear();
    for (auto& j : jobs) {
        const std::uint32_t rem = j.rows % 128, cols = j.output / 128;
        if (rem != 0 && rem <= 64) {
            tail.push_back(F32Job{j.trellis, j.suh, j.svh, j.input, j.output, j.bits, rem, j.row_begin + j.rows - rem, t64});
            t64 += cols;
            j.rows -= rem;
        }
        j.tile_begin = tiles128;
        tiles128 += (j.rows + 127) / 128 * cols;
    }
    return t64;
}
// The core CTA height for a call: 128 when the matrices average >= 96 rows, else 64.
inline std::uint32_t tile_rows_for(std::uint64_t entries, std::size_t jobs) { return jobs && entries >= 96 * jobs ? 128U : 64U; }

// xt: entries x input floats, raw: entries x output floats (device workspaces).
void enqueue_grouped(const GroupedCall& c, float* xt, float* raw, void* stream);

// The engine's MoE combine for one rank's partial: out[r] = (fmaf chain over the row's 4 routes in ascending expert id)
// + shared[r]. slot / weight: device, rows x 4 (slot = the route's row in ye).
void enqueue_combine(const float* ye, const float* shared, const std::uint32_t* slot, const float* weight, std::uint32_t rows,
                     float* out, void* stream);

// The identity's SwiGLU (portable exp): a[i] = swigluoai(g[i], u[i]) for i < n (a may alias g).
void enqueue_swiglu(const float* g, const float* u, float* a, std::size_t n, void* stream);

}  // namespace strix::minimax::rtxpf
