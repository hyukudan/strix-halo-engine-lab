#pragma once
// Decode / verify weight-read geometry (backend-neutral; the decode step plan's companion to
// strix/decode_projection_group.hpp). A window of `bucket` rows (1..16) multiplies every weight word by each row.
// The kernel reads each word once per SWEEP: rows that share a sweep share its bytes, its scale loads and its
// weight decode. The bound on rows per sweep is a register budget of the backend kernel, given as traits:
//   - accumulator_rows: rows whose accumulators one wave holds through a whole sweep without spilling;
//   - operand_rows: rows whose activation operands it holds at once (more rows take another operand load and
//     wait per k-tile inside the same sweep, no extra weight bytes).
// Rows are independent (same operands, same per-row op order whatever the grouping), so the geometry never
// changes bits; it only decides how many times the weights are streamed. Backends: HIP dense IU4
// (core/dense_linear_iu4.hpp dispatch_bucket, core/w4a16.hpp wave_rows); a CUDA tall GEMV binds the same functions
// with its own traits. The same header carries the read-ahead rule for the windows between weight GEMVs (below).
namespace strix::decode {

struct WeightSweepTraits {
    unsigned accumulator_rows;
    unsigned operand_rows;
};

constexpr unsigned sweep_rows(unsigned bucket, WeightSweepTraits t) {
    return bucket < t.accumulator_rows ? bucket : t.accumulator_rows;
}
constexpr unsigned weight_sweeps(unsigned bucket, WeightSweepTraits t) {
    const unsigned per = sweep_rows(bucket, t);
    return per == 0U ? 0U : (bucket + per - 1U) / per;
}
// Operand loads (and their waits) per k-tile inside one sweep.
constexpr unsigned operand_chunks(unsigned bucket, WeightSweepTraits t) {
    const unsigned rows = sweep_rows(bucket, t);
    return t.operand_rows == 0U ? 0U : (rows + t.operand_rows - 1U) / t.operand_rows;
}

// HIP W4A16 IU4 small-M core (gfx1151) register budgets:
//   scalar activations (UniformX, SGPR operands, 8 words per row): 8 operand rows = 64 SGPRs of the 106 limit;
//   16 accumulator rows (acc + group dot, 4 VGPRs per row) in that form: one stream of the weights instead of two
//   for a 16-row window; its VGPR / scratch readout on the node gates it.
//   vector activations (pair_activations in VGPRs): one 16-row sweep spilled (private segment 144 B), so 8.
inline constexpr WeightSweepTraits kHipIu4ScalarX{16U, 8U};
inline constexpr WeightSweepTraits kHipIu4VectorX{8U, 8U};

// ---- Cross-kernel weight read-ahead (the step plan's rule for the windows between weight GEMVs).
// Between two weight GEMVs of a decode / verify step run dependent small kernels (split-K reduce, norms, the recurrent
// window, attention) during which no weight byte is requested: DRAM idles. The next GEMV's weights are constant, so they
// may be requested into the last-level cache as soon as the previous GEMV completes; the GEMV then reads that head from
// the cache. DRAM bytes do not change (each byte still crosses once) and bits do not change (a prefetch has no
// architectural effect: same kernels, same operands, same order).
//   - what: the consumer's reads in its block DISPATCH order (linear block index, x fastest), so the head is what its
//     first resident wave reads first;
//   - how much: readahead_budget = min(consumer bytes, cache / 4); the rest of the cache keeps the window's activations,
//     partial sums and recurrent state;
//   - which blocks: the longest dispatch-order prefix of blocks whose bytes fit the budget (readahead_lead_blocks).
// The geometry is the split-K tiled GEMV of the decode cores: a job's weight is `tiles` row tiles of `chunks` records
// each (tile-major, records contiguous); a block owns `tiles_per_block` consecutive tiles of one job and one K slice;
// slice k of `ks` covers records [k * chunks / ks, (k + 1) * chunks / ks) (the decode GEMV's own formula).
// Backends: CUDA NVFP4 decode GEMV (cuda_core/dense/nvfp4.cu, a graph side node after the previous GEMV). HIP: the same
// functions once its IU4 decode GEMV exposes this geometry (follow-up; cache size from the device declaration).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_DECODE_HD __host__ __device__
#else
#define STRIX_DECODE_HD
#endif

inline constexpr unsigned kReadaheadMaxJobs = 4U;
inline constexpr unsigned long long kReadaheadCacheDivisor = 4ULL;

struct ReadaheadGrid {
    unsigned jobs = 0;                        // 1..kReadaheadMaxJobs
    unsigned tiles[kReadaheadMaxJobs] = {};   // row tiles of each job
    unsigned ks[kReadaheadMaxJobs] = {};      // K slices of each job (blocks with y >= ks are empty)
    unsigned planes[kReadaheadMaxJobs] = {};  // 1, or 2 when the job streams a second weight in z = 1
    unsigned chunks = 0;                      // records per tile (one K for every job of a call)
    unsigned tiles_per_block = 0;
    unsigned gx = 0, gy = 0, gz = 0;          // dispatched grid
};

struct ReadaheadPiece {
    unsigned job = 0, plane = 0, tile = 0, c0 = 0, c1 = 0;  // records [c0, c1) of `tile`; empty when c0 == c1
};

// Blocks of job j along x: ceil(tiles / tiles_per_block).
STRIX_DECODE_HD constexpr unsigned readahead_job_blocks(const ReadaheadGrid& g, unsigned j) {
    return g.tiles_per_block == 0U ? 0U : (g.tiles[j] + g.tiles_per_block - 1U) / g.tiles_per_block;
}

// Piece t (0 .. tiles_per_block - 1) of dispatched block `lb`: what that block's tile t reads.
STRIX_DECODE_HD constexpr ReadaheadPiece readahead_piece(const ReadaheadGrid& g, unsigned long long lb, unsigned t) {
    ReadaheadPiece p{};
    if (g.gx == 0U || g.gy == 0U) return p;
    const unsigned x = static_cast<unsigned>(lb % g.gx);
    const unsigned y = static_cast<unsigned>((lb / g.gx) % g.gy);
    const unsigned z = static_cast<unsigned>(lb / (static_cast<unsigned long long>(g.gx) * g.gy));
    unsigned j = 0U, x0 = 0U;
    while (j < g.jobs && x >= x0 + readahead_job_blocks(g, j)) {
        x0 += readahead_job_blocks(g, j);
        ++j;
    }
    if (j >= g.jobs || y >= g.ks[j] || z >= g.planes[j]) return p;
    const unsigned tile = (x - x0) * g.tiles_per_block + t;
    if (tile >= g.tiles[j]) return p;
    p.job = j;
    p.plane = z;
    p.tile = tile;
    p.c0 = static_cast<unsigned>(static_cast<unsigned long long>(y) * g.chunks / g.ks[j]);
    p.c1 = static_cast<unsigned>(static_cast<unsigned long long>(y + 1U) * g.chunks / g.ks[j]);
    return p;
}

STRIX_DECODE_HD constexpr unsigned long long readahead_block_records(const ReadaheadGrid& g, unsigned long long lb) {
    unsigned long long n = 0;
    for (unsigned t = 0; t < g.tiles_per_block; ++t) {
        const ReadaheadPiece p = readahead_piece(g, lb, t);
        n += p.c1 - p.c0;
    }
    return n;
}

// Whether a consumer GEMV of `rows` activation rows is read-ahead eligible. The read-ahead only pays while the GEMV is
// bound by its weight stream; once the row tile makes it bound by its row work, the head it would fetch early arrives
// in time anyway and the side node only competes with the window kernels. `stream_bound_rows` is the largest row count
// of the backend's weight-stream-bound tiles (CUDA NVFP4: kNvfp4ReadaheadMaxRows). Rows outside it get no side node.
constexpr bool readahead_rows_ok(unsigned rows, unsigned stream_bound_rows) {
    return rows >= 1U && rows <= stream_bound_rows;
}

constexpr unsigned long long readahead_budget(unsigned long long consumer_bytes, unsigned long long cache_bytes) {
    const unsigned long long cap = cache_bytes / kReadaheadCacheDivisor;
    return consumer_bytes < cap ? consumer_bytes : cap;
}

// Longest dispatch-order prefix of blocks whose records (x record_bytes) fit `budget_bytes`.
constexpr unsigned long long readahead_lead_blocks(const ReadaheadGrid& g, unsigned long long record_bytes,
                                                   unsigned long long budget_bytes) {
    const unsigned long long total = static_cast<unsigned long long>(g.gx) * g.gy * g.gz;
    unsigned long long sum = 0, n = 0;
    for (; n < total; ++n) {
        const unsigned long long b = readahead_block_records(g, n) * record_bytes;
        if (sum + b > budget_bytes) break;
        sum += b;
    }
    return n;
}

}  // namespace strix::decode
