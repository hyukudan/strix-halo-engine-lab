#pragma once
// Tall slab plan of the grouped BF16 expert projection (Home: shared, backend-neutral; host and device use the same maps).
// The CUDA body is mma::microscaled::detail::projection_tall (src/cuda/core/mma_tile.cuh), selected through the schedule
// trait grouped_microscaled::Schedule; tests/moe_core/grouped_tall_plan_test.cpp simulates every map and the ring schedule.
//
// Arithmetic (unchanged, EXACT schedule of the incumbent grouped_microscaled tile): per live (row, output) one FP32
// accumulator starts at +0 and takes one BF16 m16n8k16 MMA per K16 step, K16 steps ascending over the whole K, with A = the
// BF16 input row and B = the decoded weight column (PackedOperand::decode_bf16, or the bitwise decode_bf16_shift). Every MMA
// sees the row at fragment row (row % 16), the output at fragment column (output % 8) and input k at slot (k % 16), exactly
// as the incumbent forms do (tiles and CTAs start on multiples of 16 rows and 128 outputs). So each MMA receives identical
// operands at identical fragment positions and the outputs are the incumbent's bits; only staging, ring depth, CTA extent and
// barrier count differ.
//
// Schedule: a CTA covers `rows` routed rows of one tile x 128 outputs. Per ring stage of k_slab inputs it copies the activation
// slab (16-byte chunks; rows past the tile are zero-filled) and the raw weight codes (one chunk per K16 tile of codes)
// asynchronously into a ring of `stages` slots, decodes the stage's weight slab ONCE into one of two BF16 slabs shared by
// every M warp of the CTA, and runs the k_slab / 16 K16 steps; one barrier per stage (incumbent: one barrier per K16 step and
// a register lookahead of one step, each 64-row CTA decoding the same weights again). Iteration s, after its barrier, issues
// the copy of stage s + stages - 1, decodes stage s + 1 and multiplies stage s (warps of one scheduler in alternate orders:
// see decode_first); the ring schedule section states the wait counts and why no slot or slab is written while it is read.
// Wide cut (r256_k32_s3): where an expert holds on average kWideMeanRows rows or more (tile_cut(), host side, EXACT BF16 tall
// tile only) the runs are cut at 256 rows and a CTA covers 256 rows x 128 outputs with the same warp tile and K order, so each
// decoded weight slab feeds twice the rows; its scale bytes stream in 8-stage chunks (shared memory: 95,360 B of 99 KiB). The
// bits are the incumbent's (same operands at the same fragment positions, K16 steps ascending); a launch whose shape or
// shared budget the form does not fit runs the 256-row tiles as two row slices of r128_k32_s4.
// NVFP4 shift decode: the per-scale-code tables of PackedOperand::shift_table are built once per CTA (Layout::table), so a
// stage's decode is a table read plus byte permutes (the same function composition as decode_bf16_shift).
#include "moe_core/grouped_microscaled_rows.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_TALL_HD __host__ __device__ __forceinline__
#else
#define STRIX_TALL_HD inline
#endif

namespace strix::moe_core::grouped_tall {

inline constexpr unsigned kOutputs = 128U;   // N extent of a CTA (the incumbent tile form's)
inline constexpr unsigned kMaxRows = 128U;   // plan rows per tile of a backend with a tall body (the default cut)
inline constexpr unsigned kWideRows = 256U;  // rows per tile of the wide cut (r256_k32_s3); the largest tile any form takes
// Regime data (measured, no-model bench, NVFP4 shift, mean live rows per expert = ceil(assignments / experts)): the wide cut
// halves the weight decode work per MMA. In the NVFP4 sweep mean 128 loses and mean 256 wins.
inline constexpr unsigned kWideMeanRows = 256U;
inline constexpr unsigned kScaleChunkStages = 8U;  // chunked scales: ring stages per scale chunk
inline constexpr unsigned kPadBytes = 16U;   // per staged row: 16-byte row pitches with an odd count of 16-byte units
inline constexpr unsigned kChunkBytes = 16U; // activation copy chunk
inline constexpr unsigned kShiftCodes = 128U;  // NVFP4 block-scale codes with a possible exponent-shift table (0x00 .. 0x7f)

// Instantiated forms. Selection (resolve()) uses r128_k32_s4, its direct-scale twin r128_k32_s4_direct and r64_k32_s4;
// explicit wide descriptors select r256_k32_s3.
enum class FormId : std::uint8_t { r128_k32_s4, r128_k32_s3, r128_w16_k32_s4, r64_k32_s4, r128_n4_k32_s4, r128_k32_s4_direct, r256_k32_s3 };

struct Form {
    unsigned rows;     // routed rows per CTA (M)
    unsigned k_slab;   // inputs per ring stage
    unsigned stages;   // ring slots (>= 3)
    unsigned warps_m;  // warps along M
    unsigned warps_n;  // warps along N (128 outputs)
    // Block-scale source (bit-neutral): true = the CTA's scale bytes staged once in shared memory ("block scales" below),
    // false = one global byte load per decode job per stage, one stage ahead. Staging pays where a CTA spans 128 rows
    // (no-model bench, T = 2048 / 8192, NVFP4 shift: gate-up -9.1 % / -6.6 %); a 64-row CTA repays the copy over half the
    // multiplies (+1.8 % / +3.2 %), so r64 keeps the direct loads.
    bool staged_scales;
    // Staged scales in chunks (bit-neutral; needs staged_scales): instead of the whole K, the scale bytes of kScaleChunkStages
    // ring stages per output sit in two chunk buffers, refilled asynchronously one chunk ahead ("block scales" below). A 256-row
    // CTA has no shared memory for whole-K scales (95 KiB with chunks, 109 KiB with the whole K at K = 2304).
    bool chunked_scales = false;

    STRIX_TALL_HD constexpr unsigned threads() const { return 32U * warps_m * warps_n; }
    STRIX_TALL_HD constexpr unsigned warps() const { return warps_m * warps_n; }
    STRIX_TALL_HD constexpr unsigned warp_m_blocks() const { return rows / 16U / warps_m; }
    STRIX_TALL_HD constexpr unsigned warp_n_blocks() const { return kOutputs / 8U / warps_n; }
    STRIX_TALL_HD constexpr unsigned k16_steps() const { return k_slab / 16U; }
};

STRIX_TALL_HD constexpr Form form(FormId id) {
    switch (id) {
    // 8 warps, warp tile 32 rows x 64 outputs: a tile's last rows engage only the warp rows they reach.
    case FormId::r128_k32_s4: return {128U, 32U, 4U, 4U, 2U, true};
    case FormId::r128_k32_s3: return {128U, 32U, 3U, 4U, 2U, true};
    case FormId::r128_w16_k32_s4: return {128U, 32U, 4U, 4U, 4U, true};  // 16 warps, warp tile 32 x 32
    case FormId::r64_k32_s4: return {64U, 32U, 4U, 2U, 4U, false};        // 8 warps, warp tile 32 x 32
    case FormId::r128_n4_k32_s4: return {128U, 32U, 4U, 2U, 4U, true};   // 8 warps, warp tile 64 x 32
    case FormId::r128_k32_s4_direct: return {128U, 32U, 4U, 4U, 2U, false};  // r128_k32_s4 with direct scale loads
    // 16 warps (8 along M x 2 along N), warp tile 32 x 64 as r128_k32_s4: every decoded weight slab feeds 256 rows, so the
    // decode instructions per MMA drop by 42 %. 3 slots and chunked scales fit 99 KiB.
    case FormId::r256_k32_s3: return {256U, 32U, 3U, 8U, 2U, true, true};
    }
    return {128U, 32U, 4U, 4U, 2U, true};
}

STRIX_TALL_HD constexpr const char* name(FormId id) {
    switch (id) {
    case FormId::r128_k32_s4: return "r128.k32.s4";
    case FormId::r128_k32_s3: return "r128.k32.s3";
    case FormId::r128_w16_k32_s4: return "r128.w16.k32.s4";
    case FormId::r64_k32_s4: return "r64.k32.s4";
    case FormId::r128_n4_k32_s4: return "r128.n4.k32.s4";
    case FormId::r128_k32_s4_direct: return "r128.k32.s4.direct";
    case FormId::r256_k32_s3: return "r256.k32.s3";
    }
    return "?";
}

// Shared-memory layout in bytes: `stages` slots of [activation rows | raw code rows], then two decoded BF16 slabs.
struct Layout {
    unsigned act_pitch;    // bytes per staged activation row: k_slab BF16 + pad
    unsigned act_bytes;
    unsigned code_tile;    // raw code bytes of one K16 tile of one output (8 NVFP4, 16 MXFP8)
    unsigned code_row;     // raw code bytes per output per stage
    unsigned code_bytes;
    unsigned stage_bytes;  // one ring slot
    unsigned ring_bytes;
    unsigned slab_pitch;   // bytes per decoded output row: k_slab BF16 + pad
    unsigned slab_bytes;
    unsigned table;        // NVFP4: offset of the per-CTA exponent-shift tables (16 bytes per scale code, then one flag byte)
    unsigned table_bytes;  // 0 for MXFP8
    unsigned smem;         // total dynamic shared bytes
};

STRIX_TALL_HD constexpr Layout layout(Form f, bool nvfp4) {
    Layout l{};
    l.act_pitch = 2U * f.k_slab + kPadBytes;
    l.act_bytes = f.rows * l.act_pitch;
    l.code_tile = nvfp4 ? 8U : 16U;
    l.code_row = l.code_tile * f.k16_steps();
    l.code_bytes = kOutputs * l.code_row;
    l.stage_bytes = l.act_bytes + l.code_bytes;
    l.ring_bytes = f.stages * l.stage_bytes;
    l.slab_pitch = 2U * f.k_slab + kPadBytes;
    l.slab_bytes = kOutputs * l.slab_pitch;
    l.table = l.ring_bytes + 2U * l.slab_bytes;
    l.table_bytes = nvfp4 ? kShiftCodes * 16U + kShiftCodes : 0U;
    l.smem = l.table + l.table_bytes;
    return l;
}

// ---- copy plan of one stage (thread, j) -> chunk ----
STRIX_TALL_HD constexpr unsigned act_parts(Form f) { return 2U * f.k_slab / kChunkBytes; }  // chunks per activation row
STRIX_TALL_HD constexpr unsigned act_chunks(Form f) { return f.rows * act_parts(f); }
STRIX_TALL_HD constexpr unsigned act_chunks_per_thread(Form f) { return act_chunks(f) / f.threads(); }
struct ActChunk {
    unsigned row;
    unsigned part;  // 16-byte part: inputs 8 part .. 8 part + 7 of the stage
};
// Consecutive threads take consecutive 16-byte parts of a row (coalesced global reads).
STRIX_TALL_HD constexpr ActChunk act_chunk(Form f, unsigned thread, unsigned j) {
    const unsigned c = j * f.threads() + thread;
    return {c / act_parts(f), c % act_parts(f)};
}
STRIX_TALL_HD constexpr unsigned act_offset(const Layout& l, unsigned row, unsigned part) {
    return row * l.act_pitch + kChunkBytes * part;
}

STRIX_TALL_HD constexpr unsigned code_chunks(Form f) { return kOutputs * f.k16_steps(); }  // one per K16 code tile
STRIX_TALL_HD constexpr unsigned code_chunks_per_thread(Form f) {
    return (code_chunks(f) + f.threads() - 1U) / f.threads();
}
STRIX_TALL_HD constexpr bool code_chunk_live(Form f, unsigned thread, unsigned j) {
    return j * f.threads() + thread < code_chunks(f);
}
struct CodeChunk {
    unsigned output;
    unsigned tile;  // K16 tile of the stage
};
STRIX_TALL_HD constexpr CodeChunk code_chunk(Form f, unsigned thread, unsigned j) {
    const unsigned c = j * f.threads() + thread;
    return {c / f.k16_steps(), c % f.k16_steps()};
}
STRIX_TALL_HD constexpr unsigned code_offset(const Layout& l, unsigned output, unsigned tile) {
    return l.act_bytes + output * l.code_row + tile * l.code_tile;
}

// ---- decode plan: threads / 128 threads per output, each decodes `columns` consecutive inputs of the stage ----
struct DecodeJob {
    unsigned output;
    unsigned column;   // first stage-local input
    unsigned columns;  // 8 or 16 (never crosses a K16 tile)
};
STRIX_TALL_HD constexpr unsigned decode_columns(Form f) { return f.k_slab / (f.threads() / kOutputs); }
STRIX_TALL_HD constexpr DecodeJob decode_job(Form f, unsigned thread) {
    const unsigned per_output = f.threads() / kOutputs;
    return {thread / per_output, (thread % per_output) * decode_columns(f), decode_columns(f)};
}
// Raw code bytes the job reads (offset inside a slot) and the decoded slab bytes it writes.
STRIX_TALL_HD constexpr unsigned decode_code_offset(const Layout& l, const DecodeJob& job) {
    return code_offset(l, job.output, job.column / 16U) + (job.column % 16U) * l.code_tile / 16U;
}
STRIX_TALL_HD constexpr unsigned decode_code_bytes(const Layout& l, const DecodeJob& job) {
    return job.columns * l.code_tile / 16U;
}
STRIX_TALL_HD constexpr unsigned slab_offset(const Layout& l, unsigned output, unsigned column) {
    return output * l.slab_pitch + 2U * column;
}

// ---- block scales: staged once per CTA ----
// The scale bytes of the CTA's 128 outputs over the whole K are copied once, before the ring, into a region after the tables
// (4-byte chunks; consecutive threads take consecutive chunks of one output's scale row, so the global reads coalesce), and
// each decode job reads its stage's scale byte there. The bytes are the same ones a direct load would read, so the arithmetic
// is unchanged. A row whose byte count is not a whole number of 4-byte words is copied byte by byte.
STRIX_TALL_HD constexpr unsigned scale_blocks(unsigned inputs, bool nvfp4) { return inputs / (nvfp4 ? 16U : 32U); }
// Shared bytes per output: whole words, an odd word count (the byte reads of the 16 outputs of a warp hit distinct banks).
STRIX_TALL_HD constexpr unsigned scale_pitch(unsigned inputs, bool nvfp4) {
    const unsigned words = (scale_blocks(inputs, nvfp4) + 3U) / 4U;
    return 4U * (words % 2U == 1U ? words : words + 1U);
}
STRIX_TALL_HD constexpr unsigned scale_bytes(unsigned inputs, bool nvfp4) { return kOutputs * scale_pitch(inputs, nvfp4); }
// Offset of output `output`'s scale byte of block `block` (block = input / block columns).
STRIX_TALL_HD constexpr unsigned scale_offset(const Layout& l, unsigned pitch, unsigned output, unsigned block) {
    return l.smem + output * pitch + block;
}
// Chunked scales (Form::chunked_scales): chunk c holds the scale bytes of stages 8 c .. 8 c + 7 of every output (chunk_bytes
// per output: k_slab / block columns bytes per stage), in buffer c % 2; the pitch is an odd number of words so the byte reads of
// the decode jobs of a warp hit distinct banks. The chunk region follows the tables and holds both buffers, whatever K is.
STRIX_TALL_HD constexpr unsigned block_columns(bool nvfp4) { return nvfp4 ? 16U : 32U; }
STRIX_TALL_HD constexpr unsigned scale_stage_bytes(Form f, bool nvfp4) { return f.k_slab / block_columns(nvfp4); }
STRIX_TALL_HD constexpr unsigned chunk_bytes(Form f, bool nvfp4) { return kScaleChunkStages * scale_stage_bytes(f, nvfp4); }
STRIX_TALL_HD constexpr unsigned chunk_words(Form f, bool nvfp4) { return chunk_bytes(f, nvfp4) / 4U; }
STRIX_TALL_HD constexpr unsigned chunk_pitch(Form f, bool nvfp4) { return chunk_bytes(f, nvfp4) + 4U; }
STRIX_TALL_HD constexpr unsigned chunk_buffer_bytes(Form f, bool nvfp4) { return kOutputs * chunk_pitch(f, nvfp4); }
STRIX_TALL_HD constexpr unsigned chunk_count(Form f, unsigned inputs) {
    return (inputs / f.k_slab + kScaleChunkStages - 1U) / kScaleChunkStages;
}
// Offset of the scale byte of (output, byte in the chunk) in buffer `buffer`.
STRIX_TALL_HD constexpr unsigned chunk_offset(Form f, const Layout& l, bool nvfp4, unsigned buffer, unsigned output,
                                              unsigned byte) {
    return l.smem + buffer * chunk_buffer_bytes(f, nvfp4) + output * chunk_pitch(f, nvfp4) + byte;
}
// The byte a decode job reads at stage `stage`: block (stage k_slab + column) / block columns of its output.
STRIX_TALL_HD constexpr unsigned chunk_read_offset(Form f, const Layout& l, bool nvfp4, unsigned stage, unsigned output,
                                                   unsigned column) {
    return chunk_offset(f, l, nvfp4, (stage / kScaleChunkStages) % 2U, output,
                        (stage % kScaleChunkStages) * scale_stage_bytes(f, nvfp4) + column / block_columns(nvfp4));
}
// Chunk copy tasks (4-byte words): task t of chunk c is word t % chunk_words of output t / chunk_words; global byte
// 4 (c chunk_words + t % chunk_words) of the output's scale row.
STRIX_TALL_HD constexpr unsigned chunk_tasks(Form f, bool nvfp4) { return kOutputs * chunk_words(f, nvfp4); }
STRIX_TALL_HD constexpr unsigned chunk_tasks_per_thread(Form f, bool nvfp4) {
    return (chunk_tasks(f, nvfp4) + f.threads() - 1U) / f.threads();
}
// Dynamic shared bytes of a launch: the fixed layout plus, for a form with staged scales, the scale region (whole K, or the
// two chunk buffers).
STRIX_TALL_HD constexpr unsigned scale_region_bytes(Form f, unsigned inputs, bool nvfp4) {
    return !f.staged_scales ? 0U : f.chunked_scales ? 2U * chunk_buffer_bytes(f, nvfp4) : scale_bytes(inputs, nvfp4);
}
STRIX_TALL_HD constexpr unsigned shared_bytes(Form f, const Layout& l, unsigned inputs, bool nvfp4) {
    return l.smem + scale_region_bytes(f, inputs, nvfp4);
}
// Whether the scale rows copy as 4-byte chunks (else byte by byte): whole words per row on a word-aligned scale array.
STRIX_TALL_HD constexpr bool scale_words(unsigned inputs, bool nvfp4, std::uintptr_t scale_address) {
    return scale_blocks(inputs, nvfp4) % 4U == 0U && scale_address % 4U == 0U;
}

// ---- ring schedule ----
// Chunked scales: the prologue copies chunks 0 and 1 in the first commit group; iteration s (s > 0, s % 8 == 0) copies chunk
// s / 8 + 1 (if it exists) in its commit group, into the buffer chunk s / 8 - 1 held, which every decode of it (stages up to
// 8 (s / 8) - 1, done in iteration 8 (s / 8) - 2) has left behind the barrier of iteration s; the decode of the chunk's first
// stage is in iteration 8 (s / 8 + 1) - 1, after the wait of that iteration covers the copy.
STRIX_TALL_HD constexpr bool chunk_copy_at(unsigned iteration) {
    return iteration != 0U && iteration % kScaleChunkStages == 0U;
}
STRIX_TALL_HD constexpr unsigned chunk_copied_at(unsigned iteration) { return iteration / kScaleChunkStages + 1U; }
// Prologue: copies of stages 0 .. stages - 2 (one commit group each), wait until prologue_pending() groups remain, barrier,
// decode stage 0. Iteration s: wait until loop_pending() groups remain (stage s + 1 landed), barrier, copy stage
// copy_stage(s) into slot(copy_stage(s)) (one commit group, empty past the last stage), decode stage s + 1 into slab(s + 1),
// multiply stage s from slot(s) and slab(s).
STRIX_TALL_HD constexpr unsigned prologue_pending(Form f) { return f.stages - 2U; }
STRIX_TALL_HD constexpr unsigned loop_pending(Form f) { return f.stages - 3U; }
STRIX_TALL_HD constexpr unsigned slot(Form f, unsigned stage) { return stage % f.stages; }
STRIX_TALL_HD constexpr unsigned slab(unsigned stage) { return stage & 1U; }
STRIX_TALL_HD constexpr unsigned copy_stage(Form f, unsigned iteration) { return iteration + f.stages - 1U; }
// Inside an iteration the decode and the multiply are independent (different slabs and slots). Warps w and w + 4 share a
// scheduler (warp slots are dealt round-robin over 4); w / 4 even decodes first, odd multiplies first, so the warps of one
// scheduler overlap decode ALU work with MMAs instead of meeting the tensor pipe together after each barrier.
STRIX_TALL_HD constexpr bool decode_first(Form, unsigned warp) { return (warp / 4U) % 2U == 0U; }

// ---- fragments (m16n8k16): warp (warp / warps_n, warp % warps_n) owns warp_m_blocks x warp_n_blocks fragments ----
STRIX_TALL_HD constexpr unsigned fragment_row(Form f, unsigned warp, unsigned m) {
    return 16U * ((warp / f.warps_n) * f.warp_m_blocks() + m);
}
STRIX_TALL_HD constexpr unsigned fragment_column(Form f, unsigned warp, unsigned n) {
    return 8U * ((warp % f.warps_n) * f.warp_n_blocks() + n);
}

// ---- admission ----
STRIX_TALL_HD constexpr bool valid(Form f, bool nvfp4) {
    const Layout l = layout(f, nvfp4);
    const unsigned per_output = f.threads() / kOutputs;
    return f.stages >= 3U && (!f.chunked_scales || f.staged_scales) && f.k_slab % 32U == 0U && f.warps_m >= 1U && f.warps_n >= 1U &&
           f.rows % (16U * f.warps_m) == 0U && kOutputs % (8U * f.warps_n) == 0U && f.warp_n_blocks() % 2U == 0U &&
           f.threads() % kOutputs == 0U && per_output >= 1U && f.k_slab % per_output == 0U &&
           (decode_columns(f) == 8U || decode_columns(f) == 16U) && act_chunks(f) % f.threads() == 0U &&
           (l.act_pitch / 16U) % 2U == 1U && (l.slab_pitch / 16U) % 2U == 1U && l.act_pitch % 16U == 0U &&
           l.act_bytes % 16U == 0U && l.code_row % 16U == 0U && l.stage_bytes % 16U == 0U && l.slab_bytes % 16U == 0U &&
           l.table % 16U == 0U && l.smem % 16U == 0U && (!nvfp4 || f.threads() >= kShiftCodes);
}

// Shapes a form takes: every K16 step of a stage exists, whole 128-output CTAs.
// Chunked scales copy whole words: the scale row of an output is a whole number of 4-byte words (inputs % 64 NVFP4, % 128 MXFP8).
STRIX_TALL_HD constexpr bool fits_shape(Form f, unsigned inputs, unsigned outputs, bool nvfp4 = true) {
    return inputs >= f.k_slab && inputs % f.k_slab == 0U && outputs >= kOutputs && outputs % kOutputs == 0U &&
           (!f.chunked_scales || scale_blocks(inputs, nvfp4) % 4U == 0U);
}

// Row slices (grid y) of a launch whose tiles hold at most max_tile_rows rows (0 = unknown: kMaxRows).
// An unknown bound (0) means the default cut; a tile of up to 256 rows (the wide cut) runs as slices on a narrower form.
STRIX_TALL_HD constexpr unsigned row_slices(Form f, unsigned max_tile_rows) {
    return grouped_microscaled::tile_row_slices(max_tile_rows, max_tile_rows == 0U && f.rows <= kMaxRows ? kMaxRows : kWideRows,
                                                f.rows);
}

// Tile cut of a launch (host side, before the runs are cut into tiles): the wide cut once an expert holds on average
// kWideMeanRows rows (ceil(assignments / experts)), else the default cut. Only the BF16 EXACT tile reads a wide cut.
STRIX_TALL_HD constexpr unsigned tile_cut(unsigned mean_rows) { return mean_rows >= kWideMeanRows ? kWideRows : kMaxRows; }

// Form of a launch whose tiles hold at most max_tile_rows rows: a wide cut takes the 256-row CTA, else 128-row CTAs, 64-row
// CTAs when no tile can exceed 64 rows. Every form stores the same bits.
STRIX_TALL_HD constexpr FormId select(unsigned max_tile_rows) {
    return max_tile_rows > kMaxRows ? FormId::r256_k32_s3
           : max_tile_rows != 0U && max_tile_rows <= 64U ? FormId::r64_k32_s4 : FormId::r128_k32_s4;
}

// Whether the tall body takes a launch: shape, row bound, and a shared-memory budget the backend reads from the device.
inline bool fits(const grouped_microscaled::Args& args, FormId id, bool nvfp4, std::size_t shared_budget) {
    const Form f = form(id);
    return valid(f, nvfp4) && fits_shape(f, args.inputs, args.outputs, nvfp4) && args.max_tile_rows <= kWideRows &&
           shared_bytes(f, layout(f, nvfp4), args.inputs, nvfp4) <= shared_budget;
}

// The same form with direct scale loads (the scale source is a trait: bits and every other part of the schedule are
// unchanged); a form that already loads directly is its own twin.
STRIX_TALL_HD constexpr FormId direct_twin(FormId id) {
    return id == FormId::r128_k32_s4 ? FormId::r128_k32_s4_direct : id;
}

// Try the selected form, then a two-slice r128 fallback for r256. If r128 staging fails, try the same form
// with direct scale loads (staging never removes a shape the direct form admits); false when neither fits (automatic
// then keeps the staged tile schedule, an explicit tall request refuses).
inline bool resolve(const grouped_microscaled::Args& args, bool nvfp4, std::size_t shared_budget, FormId& out) {
    const FormId id = select(args.max_tile_rows);
    if (fits(args, id, nvfp4, shared_budget)) { out = id; return true; }
    const FormId fallback = id == FormId::r256_k32_s3 ? FormId::r128_k32_s4 : id;
    if (fallback != id && fits(args, fallback, nvfp4, shared_budget)) { out = fallback; return true; }
    const FormId twin = direct_twin(fallback);
    if (twin != fallback && fits(args, twin, nvfp4, shared_budget)) { out = twin; return true; }
    return false;
}

}  // namespace strix::moe_core::grouped_tall

#undef STRIX_TALL_HD
