// Host test of the tall slab plan of the grouped BF16 expert projection (moe_core/grouped_tall_plan.hpp). It simulates the
// maps the CUDA body (mma::microscaled::detail::projection_tall) uses, for every form and both weight formats:
//  1. layout: every form is valid, its staged rows have odd 16-byte pitches, and it fits 99 KiB of shared memory;
//  2. copy plan: activation chunks and code chunks cover their stage region exactly once, aligned to the copy size;
//  3. decode plan: jobs cover every (output, input) of a stage once, read the bytes the code copies wrote, write 16-byte units;
//  4. operands: through the PTX ldmatrix / m16n8k16 fragment maps, every MMA of every warp reads row r at fragment row r % 16
//     and input k at slot k % 16 (A), output o at fragment column o % 8 and input k at slot k % 16 (B), from a stage image
//     built by the copy and decode plans (padding rows are zero-filled and never read by a live fragment);
//  5. stores: each live (row, output) is written once, from fragment row row % 16 and column output % 8;
//  6. ring schedule: wait counts, slots and slabs give every read the stage it expects, after it landed, with no copy or decode
//     into a slot or slab another thread may still read in the same barrier interval, K16 steps ascending over the whole K,
//     for both decode / multiply orders inside an iteration (warps of one scheduler alternate them);
//  7. selection and admission (staged block scales never remove a shape the direct-scale twin admits);
//  8. block scales staged once per CTA: the copy covers every (output, block) byte once (4-byte chunks on whole-word rows,
//     else bytes), after the tables, inside the launch's shared bytes; every decode job of every stage reads the byte of its
//     own (output, block), the one a direct global load reads; the byte reads of one warp hit distinct banks. The chunked form
//     (r256) does the same per 8-stage chunk in two buffers (partial last chunk included) and its chunk refills are ordered by
//     the ring schedule (6: no chunk is read before it landed or overwritten while a decode may still read it);
//  9. the wide cut: runs of an expert cut at 256 rows, the cut chosen by the mean rows per expert (threshold, both sides), the
//     256-row form chosen for a wide cut and its 128-row fallback, admission of the 256-row form's shapes and shared bytes.
#include "core/f16_tile_gemm.hpp"
#include "cuda/core/mma_tile.cuh"
#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/grouped_tall_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace tall = strix::moe_core::grouped_tall;
namespace mma = strix::cuda::core::mma;
namespace native = strix::cuda::core::native_mma;
namespace view = strix::moe_core::grouped_microscaled;

int failures = 0;
void check(bool ok, const char* what, tall::FormId id, bool nvfp4) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s (%s, %s)\n", what, tall::name(id), nvfp4 ? "nvfp4" : "mxfp8");
}

constexpr tall::FormId kForms[] = {tall::FormId::r128_k32_s4, tall::FormId::r128_k32_s3, tall::FormId::r128_w16_k32_s4,
                                   tall::FormId::r64_k32_s4, tall::FormId::r128_n4_k32_s4,
                                   tall::FormId::r128_k32_s4_direct, tall::FormId::r256_k32_s3};
constexpr std::size_t kSm120OptinShared = 99U * 1024U;

static_assert(tall::valid(tall::form(tall::FormId::r128_k32_s4), true));
static_assert(tall::layout(tall::form(tall::FormId::r128_k32_s4), true).smem == 69632U + 2176U);
static_assert(tall::layout(tall::form(tall::FormId::r128_k32_s4), false).smem == 77824U);
// The 256-row form: 3 x 22,528 ring + 2 x 10,240 slabs + 2,176 tables = 90,240; two chunk buffers of 128 x 20 B join it (K does not
// matter); MXFP8: 3 x 24,576 + 20,480 = 94,208 + 2 x 128 x 12 B.
static_assert(tall::valid(tall::form(tall::FormId::r256_k32_s3), true) && tall::valid(tall::form(tall::FormId::r256_k32_s3), false));
static_assert(tall::layout(tall::form(tall::FormId::r256_k32_s3), true).smem == 90240U);
static_assert(tall::shared_bytes(tall::form(tall::FormId::r256_k32_s3), tall::layout(tall::form(tall::FormId::r256_k32_s3), true),
                                 2304U, true) == 95360U);
static_assert(tall::shared_bytes(tall::form(tall::FormId::r256_k32_s3), tall::layout(tall::form(tall::FormId::r256_k32_s3), true),
                                 64U, true) == 95360U);
static_assert(tall::shared_bytes(tall::form(tall::FormId::r256_k32_s3), tall::layout(tall::form(tall::FormId::r256_k32_s3), false),
                                 2304U, false) == 97280U);

// 1 + 2 + 3: layout, copy and decode coverage.
void plans(tall::FormId id, bool nvfp4) {
    const tall::Form f = tall::form(id);
    const tall::Layout l = tall::layout(f, nvfp4);
    check(tall::valid(f, nvfp4), "form valid", id, nvfp4);
    check(l.smem <= kSm120OptinShared, "shared bytes within 99 KiB", id, nvfp4);
    check(l.act_pitch % 16U == 0U && (l.act_pitch / 16U) % 2U == 1U, "activation pitch odd 16-byte units", id, nvfp4);
    check(l.slab_pitch % 16U == 0U && (l.slab_pitch / 16U) % 2U == 1U, "slab pitch odd 16-byte units", id, nvfp4);

    std::vector<unsigned> stage(l.stage_bytes, 0U);
    for (unsigned t = 0; t < f.threads(); ++t) {
        for (unsigned j = 0; j < tall::act_chunks_per_thread(f); ++j) {
            const auto c = tall::act_chunk(f, t, j);
            const unsigned o = tall::act_offset(l, c.row, c.part);
            if (c.row >= f.rows || c.part >= tall::act_parts(f) || o % 16U != 0U || o + 16U > l.act_bytes) {
                check(false, "activation chunk inside its region", id, nvfp4);
                return;
            }
            for (unsigned b = 0; b < 16U; ++b) ++stage[o + b];
        }
        for (unsigned j = 0; j < tall::code_chunks_per_thread(f); ++j) {
            if (!tall::code_chunk_live(f, t, j)) continue;
            const auto c = tall::code_chunk(f, t, j);
            const unsigned o = tall::code_offset(l, c.output, c.tile);
            if (c.output >= tall::kOutputs || c.tile >= f.k16_steps() || o % l.code_tile != 0U || o < l.act_bytes ||
                o + l.code_tile > l.stage_bytes) {
                check(false, "code chunk inside its region", id, nvfp4);
                return;
            }
            for (unsigned b = 0; b < l.code_tile; ++b) ++stage[o + b];
        }
    }
    bool once = true;
    for (unsigned row = 0; row < f.rows; ++row)
        for (unsigned b = 0; b < 2U * f.k_slab; ++b) once = once && stage[row * l.act_pitch + b] == 1U;
    for (unsigned b = l.act_bytes; b < l.stage_bytes; ++b) once = once && stage[b] == 1U;
    check(once, "copies cover activation rows and codes exactly once", id, nvfp4);

    std::vector<unsigned> decoded(std::size_t{tall::kOutputs} * f.k_slab, 0U), slab(l.slab_bytes, 0U);
    for (unsigned t = 0; t < f.threads(); ++t) {
        const auto job = tall::decode_job(f, t);
        const unsigned read = tall::decode_code_offset(l, job), bytes = tall::decode_code_bytes(l, job);
        const unsigned write = tall::slab_offset(l, job.output, job.column);
        const bool inside = job.output < tall::kOutputs && job.column + job.columns <= f.k_slab &&
                            job.column / 16U == (job.column + job.columns - 1U) / 16U;
        // The code bytes of (output, inputs column .. column + columns) as the copy plan placed them.
        const unsigned expected = tall::code_offset(l, job.output, job.column / 16U) + (job.column % 16U) * l.code_tile / 16U;
        if (!inside || read != expected || bytes != job.columns * l.code_tile / 16U || bytes % 4U != 0U || read % bytes != 0U ||
            write % 16U != 0U || write + 2U * job.columns > l.slab_bytes || (2U * job.columns) % 16U != 0U) {
            check(false, "decode job reads its codes and writes 16-byte units", id, nvfp4);
            return;
        }
        for (unsigned c = 0; c < job.columns; ++c) ++decoded[std::size_t{job.output} * f.k_slab + job.column + c];
        for (unsigned b = 0; b < 2U * job.columns; ++b) ++slab[write + b];
    }
    bool jobs = true;
    for (const unsigned n : decoded) jobs = jobs && n == 1U;
    for (unsigned o = 0; o < tall::kOutputs; ++o)
        for (unsigned b = 0; b < l.slab_pitch; ++b) jobs = jobs && slab[o * l.slab_pitch + b] == (b < 2U * f.k_slab ? 1U : 0U);
    check(jobs, "decode jobs cover every output and input of a stage once", id, nvfp4);
}

// 4: operands of every MMA through the ldmatrix maps, on a stage image tagged by the copy / decode plans.
void operands(tall::FormId id, bool nvfp4, unsigned live) {
    constexpr std::uint32_t kPad = 0xffffffffU, kZero = 0xfffffffeU;
    const tall::Form f = tall::form(id);
    const tall::Layout l = tall::layout(f, nvfp4);
    // Tag of an activation half: 1 + row * 4096 + k; of a decoded weight half: 1 + (1 << 24) + output * 4096 + k.
    std::vector<std::uint32_t> act(l.act_bytes / 2U, kPad), slab(l.slab_bytes / 2U, kPad);
    for (unsigned t = 0; t < f.threads(); ++t) {
        for (unsigned j = 0; j < tall::act_chunks_per_thread(f); ++j) {
            const auto c = tall::act_chunk(f, t, j);
            for (unsigned h = 0; h < 8U; ++h)
                act[tall::act_offset(l, c.row, c.part) / 2U + h] = c.row < live ? 1U + c.row * 4096U + 8U * c.part + h : kZero;
        }
        const auto job = tall::decode_job(f, t);
        for (unsigned c = 0; c < job.columns; ++c)
            slab[tall::slab_offset(l, job.output, job.column) / 2U + c] = 1U + (1U << 24U) + job.output * 4096U + job.column + c;
    }
    bool ok = true;
    for (unsigned warp = 0; warp < f.warps() && ok; ++warp) {
        for (unsigned kk = 0; kk < f.k16_steps(); ++kk) {
            for (unsigned lane = 0; lane < 32U; ++lane) {
                for (unsigned m = 0; m < f.warp_m_blocks(); ++m) {
                    if (tall::fragment_row(f, warp, 0U) >= live) continue;  // a dead warp issues no ldmatrix / MMA
                    for (unsigned reg = 0; reg < 4U; ++reg) {
                        const auto addr = mma::detail::ldsm_a_address(8U * reg + lane / 4U);
                        const unsigned byte = tall::fragment_row(f, warp, m) * l.act_pitch + 32U * kk +
                                              addr.row * l.act_pitch + 16U * addr.segment + 4U * (lane % 4U);
                        const unsigned row = tall::fragment_row(f, warp, m) + native::a_row(lane, reg);
                        for (unsigned e = 0; e < 2U; ++e) {
                            const std::uint32_t got = act[byte / 2U + e];
                            const unsigned k = 16U * kk + native::a_slot(lane, reg, e);
                            const std::uint32_t want = row < live ? 1U + row * 4096U + k : kZero;
                            ok = ok && got == want && row % 16U == native::a_row(lane, reg);
                        }
                    }
                }
                for (unsigned n = 0; n < f.warp_n_blocks(); ++n) {
                    for (unsigned reg = 0; reg < 2U; ++reg) {
                        // ldsm_b<4> over the pair (n - n % 2, n - n % 2 + 1): matrix 2 (n % 2) + reg.
                        const auto addr = mma::detail::ldsm_b_address<4U>(16U * (n % 2U) + 8U * reg + lane / 4U);
                        const unsigned byte = tall::fragment_column(f, warp, n - n % 2U) * l.slab_pitch + 32U * kk +
                                              addr.row * l.slab_pitch + 16U * addr.segment + 4U * (lane % 4U);
                        const unsigned output = tall::fragment_column(f, warp, n) + native::b_col(lane);
                        for (unsigned e = 0; e < 2U; ++e) {
                            const unsigned k = 16U * kk + native::b_slot(lane, reg, e);
                            ok = ok && slab[byte / 2U + e] == 1U + (1U << 24U) + output * 4096U + k &&
                                 output % 8U == native::b_col(lane);
                        }
                    }
                }
            }
        }
    }
    check(ok, "MMA operands: row r / output o at fragment position r % 16 / o % 8, input k at slot k % 16", id, nvfp4);
}

// 5: stores.
void stores(tall::FormId id, unsigned rows) {
    const tall::Form f = tall::form(id);
    std::vector<unsigned> writes(std::size_t{rows} * tall::kOutputs, 0U);
    bool ok = true;
    for (unsigned slice = 0; slice < tall::row_slices(f, rows); ++slice) {
        const unsigned first = slice * f.rows, live = rows - first < f.rows ? rows - first : f.rows;
        for (unsigned warp = 0; warp < f.warps(); ++warp)
            for (unsigned m = 0; m < f.warp_m_blocks(); ++m)
                for (unsigned n = 0; n < f.warp_n_blocks(); ++n)
                    for (unsigned lane = 0; lane < 32U; ++lane)
                        for (unsigned i = 0; i < 4U; ++i) {
                            const unsigned row = tall::fragment_row(f, warp, m) + native::c_row(lane, i);
                            const unsigned col = tall::fragment_column(f, warp, n) + native::c_col(lane, i);
                            ok = ok && col < tall::kOutputs && row % 16U == native::c_row(lane, i) &&
                                 col % 8U == native::c_col(lane, i);
                            if (row < live) ++writes[std::size_t{first + row} * tall::kOutputs + col];
                        }
    }
    for (const unsigned w : writes) ok = ok && w == 1U;
    check(ok, "stores: every live (row, output) once from its fragment position", id, true);
}

// 6: the ring schedule of the body, interval by interval (an interval = the code between two barriers; operations of
// different threads inside one interval are unordered, a copy lands at an unknown time before the wait that covers it).
bool schedule(tall::Form f, unsigned steps, bool decode_first) {
    struct Group {
        int stage;
        int slot;
        int chunks[2];  // chunked scales: chunks copied by this group (-1 = none)
    };
    std::vector<Group> groups;
    int chunk_in[2] = {-1, -1}, chunk_visible[2] = {1 << 30, 1 << 30}, chunk_read[2] = {-1, -1}, chunk_group[2] = {-1, -1};
    std::vector<int> slot_stage(f.stages, -1), slot_visible(f.stages, 1 << 30), slot_read(f.stages, -1);
    std::vector<int> slot_group(f.stages, -1);
    int slab_stage[2] = {-1, -1}, slab_written[2] = {-1, -1}, slab_read[2] = {-1, -1};
    std::vector<int> decoded(steps, 0), computed;
    unsigned completed = 0;  // groups [0, completed) landed
    int interval = 0;
    bool ok = true;
    const auto issue = [&](unsigned stage) {
        const int q = static_cast<int>(tall::slot(f, stage));
        ok = ok && slot_read[q] < interval;  // nobody may still read the slot in this interval
        slot_stage[q] = static_cast<int>(stage);
        slot_visible[q] = 1 << 30;
        slot_group[q] = static_cast<int>(groups.size());
        groups.push_back({static_cast<int>(stage), q, {-1, -1}});
    };
    const auto commit_empty = [&] { groups.push_back({-1, -1, {-1, -1}}); };
    // Chunk c joins the group just pushed (the kernel copies it between the stage copy and the commit).
    const auto copy_chunk = [&](unsigned chunk) {
        const unsigned b = chunk % 2U;
        ok = ok && !groups.empty() && chunk_read[b] < interval;  // nobody may still read the buffer in this interval
        chunk_in[b] = static_cast<int>(chunk);
        chunk_visible[b] = 1 << 30;
        chunk_group[b] = static_cast<int>(groups.size()) - 1;
        Group& g = groups.back();
        (g.chunks[0] < 0 ? g.chunks[0] : g.chunks[1]) = static_cast<int>(chunk);
    };
    const auto wait_barrier = [&](unsigned pending) {
        const unsigned done = groups.size() > pending ? static_cast<unsigned>(groups.size()) - pending : 0U;
        for (unsigned g = completed; g < done; ++g) {
            if (groups[g].stage >= 0 && slot_group[groups[g].slot] == static_cast<int>(g))
                slot_visible[groups[g].slot] = interval + 1;
            for (const int chunk : groups[g].chunks)
                if (chunk >= 0 && chunk_group[chunk % 2] == static_cast<int>(g) && chunk_in[chunk % 2] == chunk)
                    chunk_visible[chunk % 2] = interval + 1;
        }
        if (done > completed) completed = done;
        ++interval;
    };
    const auto read_slot = [&](unsigned stage) {
        const int q = static_cast<int>(tall::slot(f, stage));
        ok = ok && slot_stage[q] == static_cast<int>(stage) && slot_visible[q] <= interval;
        slot_read[q] = interval;
    };
    const auto decode = [&](unsigned stage) {
        read_slot(stage);
        if (f.chunked_scales) {  // the stage's scale bytes: chunk stage / 8, landed, in its buffer
            const unsigned cb = (stage / tall::kScaleChunkStages) % 2U;
            ok = ok && chunk_in[cb] == static_cast<int>(stage / tall::kScaleChunkStages) && chunk_visible[cb] <= interval;
            chunk_read[cb] = interval;
        }
        const unsigned b = tall::slab(stage);
        const bool consumed = slab_stage[b] < 0 ||
                              (computed.size() > static_cast<std::size_t>(slab_stage[b]) &&
                               computed[static_cast<std::size_t>(slab_stage[b])] == slab_stage[b]);
        ok = ok && slab_read[b] < interval && consumed && decoded[stage] == 0;
        slab_stage[b] = static_cast<int>(stage);
        slab_written[b] = interval;
        decoded[stage] = 1;
    };
    const auto compute = [&](unsigned stage) {
        read_slot(stage);
        const unsigned b = tall::slab(stage);
        ok = ok && slab_stage[b] == static_cast<int>(stage) && slab_written[b] < interval &&
             computed.size() == stage;  // ascending, each once
        slab_read[b] = interval;
        computed.push_back(static_cast<int>(stage));
    };
    const unsigned chunks = f.chunked_scales ? tall::chunk_count(f, steps * f.k_slab) : 0U;
    for (unsigned s = 0; s + 1U < f.stages; ++s) {
        if (s < steps) issue(s);
        else commit_empty();
        if (s == 0U && chunks > 0U) {  // the prologue's chunks 0 and 1 join the first group
            copy_chunk(0U);
            if (chunks > 1U) copy_chunk(1U);
        }
    }
    wait_barrier(tall::prologue_pending(f));
    decode(0U);
    for (unsigned s = 0; s < steps; ++s) {
        wait_barrier(tall::loop_pending(f));
        if (tall::copy_stage(f, s) < steps) issue(tall::copy_stage(f, s));
        else commit_empty();
        if (chunks > 0U && tall::chunk_copy_at(s) && tall::chunk_copied_at(s) < chunks) copy_chunk(tall::chunk_copied_at(s));
        if (decode_first && s + 1U < steps) decode(s + 1U);
        compute(s);
        if (!decode_first && s + 1U < steps) decode(s + 1U);
    }
    for (unsigned s = 0; s < steps; ++s) ok = ok && decoded[s] == 1;
    return ok && computed.size() == steps;
}

// 7: selection and admission.
void selection() {
    const tall::FormId none = tall::FormId::r128_k32_s4;
    bool ok = tall::select(0U) == tall::FormId::r128_k32_s4 && tall::select(128U) == tall::FormId::r128_k32_s4 &&
              tall::select(65U) == tall::FormId::r128_k32_s4 && tall::select(64U) == tall::FormId::r64_k32_s4 &&
              tall::select(17U) == tall::FormId::r64_k32_s4;
    check(ok, "select: 128-row CTAs, 64-row CTAs when no tile exceeds 64 rows", none, true);
    ok = tall::row_slices(tall::form(tall::FormId::r64_k32_s4), 128U) == 2U &&
         tall::row_slices(tall::form(tall::FormId::r128_k32_s4), 0U) == 1U &&
         tall::row_slices(tall::form(tall::FormId::r64_k32_s4), 64U) == 1U;
    check(ok, "row slices", none, true);
    namespace cuda_tile = strix::cuda::core::mma::microscaled;
    // One backend-neutral row plan: a 129-row live bound at 128 rows per CTA takes 2 slices (also with an unknown bound
    // and a 256-row descriptor capacity), the CUDA tile and tall plans and the HIP tile launch bind the same helper.
    ok = view::tile_row_slices(129U, 256U, 128U) == 2U && view::tile_row_slices(0U, 256U, 128U) == 2U &&
         view::tile_row_slices(128U, 256U, 128U) == 1U && view::tile_row_slices(300U, 256U, 128U) == 2U &&
         view::tile_row_slices(1U, 256U, 256U) == 1U && view::tile_row_slices(129U, 256U, 0U) == 0U &&
         cuda_tile::row_slices(129U, 128U) == view::tile_row_slices(129U, cuda_tile::kRowsPerTile, 128U) &&
         tall::row_slices(tall::form(tall::FormId::r64_k32_s4), 129U) == view::tile_row_slices(129U, tall::kWideRows, 64U);
    check(ok, "shared row plan: 129 rows at 128 per CTA = 2 slices", none, true);
    // Schedule resolution (backend-neutral): automatic takes the narrow body for a known bound of 1..narrow rows and a
    // fitting stage, tall above it or with an unknown bound, staged when the stage does not fit; an explicit request is
    // kept; a backend without a narrow body (0 rows) never resolves to it. 48 rows at 32 per CTA take 2 slices.
    using S = view::Schedule;
    ok = view::resolve_schedule(S::automatic, 1U, 32U, true) == S::narrow &&
         view::resolve_schedule(S::automatic, 32U, 32U, true) == S::narrow &&
         view::resolve_schedule(S::automatic, 33U, 32U, true) == S::tall &&
         view::resolve_schedule(S::automatic, 0U, 32U, true) == S::tall &&
         view::resolve_schedule(S::automatic, 8U, 32U, false) == S::staged &&
         view::resolve_schedule(S::automatic, 8U, 0U, true) == S::tall &&
         view::resolve_schedule(S::staged, 8U, 32U, true) == S::staged &&
         view::resolve_schedule(S::tall, 8U, 32U, true) == S::tall &&
         view::resolve_schedule(S::narrow, 200U, 32U, true) == S::narrow &&
         view::resolve_schedule(S::automatic, 256U, 32U, true, 32U) == S::narrow &&
         view::resolve_schedule(S::automatic, 256U, 32U, true, 33U) == S::tall &&
         view::resolve_schedule(S::automatic, 0U, 32U, true, 4U) == S::narrow &&
         view::resolve_schedule(S::automatic, 256U, 32U, false, 4U) == S::staged &&
         view::resolve_schedule(S::automatic, 256U, 0U, true, 4U) == S::tall &&
         view::resolve_schedule(S::tall, 256U, 32U, true, 4U) == S::tall &&
         view::tile_row_slices(48U, 256U, 32U) == 2U && view::tile_row_slices(32U, 256U, 32U) == 1U;
    check(ok, "schedule resolution: narrow for 1..32 known rows, tall above, staged without the K64 stage", none, true);
    // The HIP direct packed launch admits only shapes whose every stage exists: K = 96 is refused under the K64 tall
    // schedule and taken by the K32 incumbent (staged fallback).
    namespace gemm = strix::core::f16_tile_gemm;
    using Tall = gemm::TileSchedule<2, 8, 64>;
    ok = !Tall::admits(96U, 128U) && Tall::admits(128U, 128U) && gemm::IncumbentSchedule::admits(96U, 128U) &&
         !gemm::IncumbentSchedule::admits(16U, 128U) && !Tall::admits(128U, 192U) && !Tall::admits(0U, 128U);
    check(ok, "packed launch admission: K = 96 refused under K64", none, true);
    view::Args args{};
    args.inputs = 2304U;
    args.outputs = 2304U;
    args.max_tile_rows = 128U;
    ok = tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared) &&
         tall::fits(args, tall::FormId::r128_k32_s4, false, kSm120OptinShared);
    args.inputs = 1152U;
    ok = ok && tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);
    check(ok, "Inkling gate-up 2304 -> 2304 and down 1152 -> 2304 admitted", none, true);
    args.inputs = 2320U;  // inputs % 32 != 0
    ok = !tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);
    args.inputs = 2304U;
    args.outputs = 2304U + 64U;  // outputs % 128 != 0
    ok = ok && !tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);
    args.outputs = 2304U;
    args.max_tile_rows = 257U;
    ok = ok && !tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared) &&
         !tall::fits(args, tall::FormId::r256_k32_s3, true, kSm120OptinShared);
    args.max_tile_rows = 128U;
    ok = ok && !tall::fits(args, tall::FormId::r128_k32_s4, true, 48U * 1024U);  // budget below the form's shared bytes
    check(ok, "ragged inputs / outputs, oversize tiles and a small shared budget refused", none, true);
    // Budget thresholds (K = 4096, N = 2048, NVFP4 and MXFP8): staged r128 at or above its bytes, the direct twin (the
    // incumbent r128 kernel) from its own bytes up to one byte below the staged ones, nothing below (automatic keeps the
    // staged tile schedule, an explicit tall request refuses); r64 has no twin.
    for (const bool nvfp4 : {true, false}) {
        view::Args big{};
        big.inputs = 4096U;
        big.outputs = 2048U;
        big.max_tile_rows = 128U;
        const tall::Form fs = tall::form(tall::FormId::r128_k32_s4), fd = tall::form(tall::FormId::r128_k32_s4_direct);
        const std::size_t staged = tall::shared_bytes(fs, tall::layout(fs, nvfp4), big.inputs, nvfp4);
        const std::size_t direct = tall::shared_bytes(fd, tall::layout(fd, nvfp4), big.inputs, nvfp4);
        tall::FormId got = tall::FormId::r128_k32_s3;
        ok = direct < staged && tall::layout(fs, nvfp4).smem == tall::layout(fd, nvfp4).smem &&
             tall::direct_twin(tall::FormId::r128_k32_s4) == tall::FormId::r128_k32_s4_direct &&
             tall::direct_twin(tall::FormId::r64_k32_s4) == tall::FormId::r64_k32_s4;
        ok = ok && tall::resolve(big, nvfp4, staged, got) && got == tall::FormId::r128_k32_s4;
        ok = ok && tall::resolve(big, nvfp4, staged - 1U, got) && got == tall::FormId::r128_k32_s4_direct;
        ok = ok && tall::resolve(big, nvfp4, direct, got) && got == tall::FormId::r128_k32_s4_direct;
        got = tall::FormId::r128_k32_s3;
        ok = ok && !tall::resolve(big, nvfp4, direct - 1U, got) && got == tall::FormId::r128_k32_s3;
        if (nvfp4)  // NVFP4 at a 96 KiB budget: the incumbent admits r128, staging alone would not -> the direct twin
            ok = ok && direct == 71808U && staged == 105088U && tall::resolve(big, nvfp4, 96U * 1024U, got) &&
                 got == tall::FormId::r128_k32_s4_direct;
        big.max_tile_rows = 64U;
        const tall::Form f64 = tall::form(tall::FormId::r64_k32_s4);
        const std::size_t b64 = tall::shared_bytes(f64, tall::layout(f64, nvfp4), big.inputs, nvfp4);
        ok = ok && tall::resolve(big, nvfp4, b64, got) && got == tall::FormId::r64_k32_s4 &&
             !tall::resolve(big, nvfp4, b64 - 1U, got);
        check(ok, "budget thresholds: staged r128 >= its bytes, direct twin below, none under the direct bytes", none, nvfp4);
    }
    ok = mma::microscaled::kRowsPerTile == tall::kMaxRows;
    check(ok, "plan rows per tile = the tall CTA extent", none, true);
}

// 8b. Chunked scales (r256): the CUDA body's chunk copies (prologue chunks 0 and 1, then one per 8 stages) and its decode reads.
void chunked_scales(tall::FormId id, bool nvfp4, unsigned inputs, std::uintptr_t scale_address) {
    const tall::Form f = tall::form(id);
    const tall::Layout l = tall::layout(f, nvfp4);
    const unsigned bc = tall::block_columns(nvfp4), stride = tall::scale_stage_bytes(f, nvfp4);
    const unsigned steps = inputs / f.k_slab, words = tall::chunk_words(f, nvfp4), pitch = tall::chunk_pitch(f, nvfp4);
    const unsigned total = tall::shared_bytes(f, l, inputs, nvfp4);
    check(tall::fits_shape(f, inputs, tall::kOutputs, nvfp4) == (tall::scale_blocks(inputs, nvfp4) % 4U == 0U && inputs >= f.k_slab &&
                                                                  inputs % f.k_slab == 0U),
          "chunked form admits whole-word scale rows only", id, nvfp4);
    if (tall::scale_blocks(inputs, nvfp4) % 4U != 0U || inputs % f.k_slab != 0U) return;  // refused shapes
    bool ok = (pitch / 4U) % 2U == 1U && tall::chunk_bytes(f, nvfp4) == 8U * stride &&
              total == l.smem + 2U * tall::kOutputs * pitch && l.smem >= l.table + l.table_bytes &&
              tall::chunk_tasks(f, nvfp4) == tall::kOutputs * words &&
              tall::chunk_tasks_per_thread(f, nvfp4) * f.threads() >= tall::chunk_tasks(f, nvfp4);
    check(ok, "chunk region after the tables, odd word pitch, whole-word tasks", id, nvfp4);

    // Copy map: chunk c, task (thread, j): word w of output o lands at chunk_offset(c % 2, o, 4 w); the global bytes are the
    // (output, block 4 (c words + w) + i) bytes, each exactly once over all chunks; images[c] tags the buffer after chunk c.
    const unsigned blocks = tall::scale_blocks(inputs, nvfp4), chunks = tall::chunk_count(f, inputs);
    std::vector<unsigned> hits(std::size_t{tall::kOutputs} * blocks, 0U);
    std::vector<std::vector<std::uint32_t>> images(chunks, std::vector<std::uint32_t>(total, ~0U));
    ok = true;
    for (unsigned c = 0; c < chunks; ++c)
        for (unsigned t = 0; t < f.threads(); ++t)
            for (unsigned j = 0; j < tall::chunk_tasks_per_thread(f, nvfp4); ++j) {
                const unsigned task = j * f.threads() + t;
                if (task >= tall::chunk_tasks(f, nvfp4)) continue;
                const unsigned o = task / words, w = task % words;
                if (4U * (c * words + w) >= blocks) continue;  // past the last chunk's words
                const unsigned dst = tall::chunk_offset(f, l, nvfp4, c % 2U, o, 4U * w);
                ok = ok && o < tall::kOutputs && dst % 4U == 0U && dst + 4U <= total && dst >= l.smem &&
                     // word copies on a word-aligned array (rows are whole words), else the body stores the bytes
                     (scale_address % 4U != 0U || (scale_address + std::uintptr_t{o} * blocks + 4U * (c * words + w)) % 4U == 0U);
                for (unsigned i = 0; i < 4U; ++i) {
                    const unsigned block = 4U * (c * words + w) + i;
                    ok = ok && block < blocks;
                    if (block < blocks) ++hits[std::size_t{o} * blocks + block];
                    images[c][dst + i] = o * 4096U + block;
                }
            }
    for (const unsigned h : hits) ok = ok && h == 1U;
    check(ok, "chunk copies cover every (output, block) once", id, nvfp4);

    // Decode reads: job (output, column) of stage s reads, in the buffer of chunk s / 8, the byte of block
    // (s k_slab + column) / block columns of its output (what a direct global load reads).
    ok = true;
    for (unsigned s = 0; s < steps; ++s)
        for (unsigned t = 0; t < f.threads(); ++t) {
            const tall::DecodeJob job = tall::decode_job(f, t);
            const unsigned read = tall::chunk_read_offset(f, l, nvfp4, s, job.output, job.column);
            const unsigned block = (s * f.k_slab + job.column) / bc;
            ok = ok && read < total && images[s / tall::kScaleChunkStages][read] == job.output * 4096U + block &&
                 (s * f.k_slab + job.column + job.columns - 1U) / bc == block;
        }
    check(ok, "decode jobs read their own (output, block) chunk byte", id, nvfp4);

    // Banks: within a warp, two jobs reading different words never share a bank (every stage of the chunk).
    ok = true;
    for (unsigned s = 0; s < tall::kScaleChunkStages && s < steps; ++s)
        for (unsigned w0 = 0; w0 < f.threads(); w0 += 32U) {
            unsigned word_of_bank[32];
            for (unsigned& b : word_of_bank) b = ~0U;
            for (unsigned t = w0; t < w0 + 32U; ++t) {
                const tall::DecodeJob job = tall::decode_job(f, t);
                const unsigned word = tall::chunk_read_offset(f, l, nvfp4, s, job.output, job.column) / 4U;
                unsigned& seen = word_of_bank[word % 32U];
                ok = ok && (seen == ~0U || seen == word);
                seen = word;
            }
        }
    check(ok, "chunk byte reads of a warp conflict-free", id, nvfp4);
}

// 8. Block scales staged once per CTA (the CUDA body's prologue copy and its decode reads).
void scales(tall::FormId id, bool nvfp4, unsigned inputs, std::uintptr_t scale_address) {
    const tall::Form f = tall::form(id);
    const tall::Layout l = tall::layout(f, nvfp4);
    const unsigned blocks = tall::scale_blocks(inputs, nvfp4), pitch = tall::scale_pitch(inputs, nvfp4);
    const unsigned block_columns = nvfp4 ? 16U : 32U, stride = f.k_slab / block_columns;
    if (f.chunked_scales) {
        chunked_scales(id, nvfp4, inputs, scale_address);
        return;
    }
    if (!f.staged_scales) {  // direct loads: no region, the incumbent shared bytes
        check(tall::shared_bytes(f, l, inputs, nvfp4) == l.smem, "direct-scale form adds no shared bytes", id, nvfp4);
        return;
    }
    bool ok = blocks * block_columns == inputs && pitch % 4U == 0U && (pitch / 4U) % 2U == 1U && pitch >= blocks &&
              tall::scale_bytes(inputs, nvfp4) == tall::kOutputs * pitch &&
              tall::shared_bytes(f, l, inputs, nvfp4) == l.smem + tall::scale_bytes(inputs, nvfp4) &&
              l.smem >= l.table + l.table_bytes;
    check(ok, "scale region after the tables, odd word pitch", id, nvfp4);

    // Copy map: global byte (output o, block b) lands at scale_offset(o, b), each exactly once.
    std::vector<unsigned> hits(tall::kOutputs * blocks, 0U);
    const auto land = [&](unsigned o, unsigned b, unsigned dst) {
        if (o >= tall::kOutputs || b >= blocks || dst != tall::scale_offset(l, pitch, o, b) ||
            dst >= tall::shared_bytes(f, l, inputs, nvfp4))
            return false;
        ++hits[o * blocks + b];
        return true;
    };
    ok = true;
    if (tall::scale_words(inputs, nvfp4, scale_address)) {
        const unsigned words = blocks / 4U;
        for (unsigned t = 0; t < f.threads(); ++t)
            for (unsigned c = t; c < tall::kOutputs * words; c += f.threads()) {
                const unsigned o = c / words, w = c % words, dst = tall::scale_offset(l, pitch, o, 4U * w);
                ok = ok && dst % 4U == 0U && (scale_address + std::uintptr_t{o} * blocks + 4U * w) % 4U == 0U;
                for (unsigned i = 0; i < 4U; ++i) ok = ok && land(o, 4U * w + i, dst + i);
            }
    } else {
        for (unsigned t = 0; t < f.threads(); ++t)
            for (unsigned c = t; c < tall::kOutputs * blocks; c += f.threads())
                ok = ok && land(c / blocks, c % blocks, tall::scale_offset(l, pitch, c / blocks, c % blocks));
    }
    for (const unsigned h : hits) ok = ok && h == 1U;
    check(ok, "scale copy covers every (output, block) once", id, nvfp4);

    // Decode reads: job (output, column) of stage s reads the byte of block (s k_slab + column) / block columns, the same
    // block every input of the job belongs to (a direct load reads tile16_address(output, s k_slab + column).scale_code).
    ok = true;
    for (unsigned s = 0; s < inputs / f.k_slab; ++s)
        for (unsigned t = 0; t < f.threads(); ++t) {
            const tall::DecodeJob job = tall::decode_job(f, t);
            const unsigned first = tall::scale_offset(l, pitch, job.output, job.column / block_columns);
            const unsigned block = (s * f.k_slab + job.column) / block_columns;
            ok = ok && first + s * stride == tall::scale_offset(l, pitch, job.output, block) &&
                 (s * f.k_slab + job.column + job.columns - 1U) / block_columns == block;
        }
    check(ok, "decode jobs read their own (output, block) scale byte", id, nvfp4);

    // Banks: within a warp, two jobs reading different words never share a bank.
    ok = true;
    for (unsigned w0 = 0; w0 < f.threads(); w0 += 32U) {
        unsigned word_of_bank[32];
        for (unsigned& b : word_of_bank) b = ~0U;
        for (unsigned t = w0; t < w0 + 32U; ++t) {
            const tall::DecodeJob job = tall::decode_job(f, t);
            const unsigned word = tall::scale_offset(l, pitch, job.output, job.column / block_columns) / 4U;
            unsigned& seen = word_of_bank[word % 32U];
            ok = ok && (seen == ~0U || seen == word);
            seen = word;
        }
    }
    check(ok, "scale byte reads of a warp conflict-free", id, nvfp4);
}

// 9. The wide cut: expert runs cut at 256 rows, the cut by regime, the form of a cut.
void wide_cut() {
    const tall::FormId none = tall::FormId::r256_k32_s3;
    // Runs of 600, 129, 1 and 0 rows cut at 128 and at 256 rows (the planners cut `remaining` into tiles of at most the cut):
    // every row once, in order, tile counts within the launch descriptor bound of the 128-row cut (a wide cut never needs more).
    const unsigned runs[4] = {600U, 129U, 1U, 0U};
    for (const unsigned cut : {tall::kMaxRows, tall::kWideRows}) {
        unsigned tiles = 0U, covered = 0U;
        bool ok = true;
        for (const unsigned run : runs)
            for (unsigned first = 0U; first < run; first += cut) {
                const unsigned rows = run - first < cut ? run - first : cut;
                ok = ok && rows >= 1U && rows <= cut;
                covered += rows;
                ++tiles;
            }
        const unsigned want = cut == tall::kWideRows ? 3U + 1U + 1U : 5U + 2U + 1U;
        ok = ok && covered == 730U && tiles == want &&
             tiles <= strix::moe_core::device_grouped::tile_bound(730U, 1U, 4U, tall::kMaxRows) + 4U;
        check(ok, "runs cut at 128 / 256 rows cover every assignment once", none, true);
    }
    bool ok = true;
    for (const unsigned rows : {1U, 2048U, 4096U, 8192U, 12288U})
        ok = ok && strix::moe_core::device_grouped::tile_bound(rows, 8U, 128U, tall::kWideRows) <=
                       strix::moe_core::device_grouped::tile_bound(rows, 8U, 128U, tall::kMaxRows);
    check(ok, "descriptor capacity of the 128-row cut holds the wide cut", none, true);
    // The cut by regime (mean rows per expert): one threshold, both sides; automatic selection of a launch from its bound.
    ok = tall::tile_cut(0U) == tall::kMaxRows && tall::tile_cut(tall::kWideMeanRows - 1U) == tall::kMaxRows &&
              tall::tile_cut(tall::kWideMeanRows) == tall::kWideRows && tall::tile_cut(768U) == tall::kWideRows &&
              strix::cuda::core::mma::microscaled::tile_cut(tall::kWideMeanRows - 1U) == tall::kMaxRows &&
              strix::cuda::core::mma::microscaled::tile_cut(tall::kWideMeanRows) == tall::kWideRows &&
              strix::cuda::core::mma::microscaled::kRowsPerTile == tall::kMaxRows;
    check(ok, "tile cut: default below the mean-rows threshold, wide at and above it", none, true);
    ok = tall::select(129U) == tall::FormId::r256_k32_s3 && tall::select(256U) == tall::FormId::r256_k32_s3 &&
         tall::select(128U) == tall::FormId::r128_k32_s4 && tall::select(0U) == tall::FormId::r128_k32_s4 &&
         tall::select(64U) == tall::FormId::r64_k32_s4;
    check(ok, "select: a wide cut takes the 256-row form, narrower cuts as before", none, true);
    // The 256-row form: 1 slice per tile, the 128-row fallback 2; unknown bound = the default cut.
    const tall::Form wide = tall::form(tall::FormId::r256_k32_s3), narrow = tall::form(tall::FormId::r128_k32_s4);
    ok = tall::row_slices(wide, 256U) == 1U && tall::row_slices(wide, 129U) == 1U && tall::row_slices(wide, 0U) == 1U &&
         tall::row_slices(narrow, 256U) == 2U && tall::row_slices(narrow, 129U) == 2U && tall::row_slices(narrow, 128U) == 1U &&
         tall::row_slices(narrow, 0U) == 1U && tall::row_slices(tall::form(tall::FormId::r64_k32_s4), 256U) == 4U;
    check(ok, "row slices of the wide cut", none, true);
    // Admission: shape (whole-word scale rows), shared bytes against the device budget, tile bound.
    view::Args args{};
    args.inputs = 2304U;
    args.outputs = 2304U;
    args.max_tile_rows = 256U;
    ok = tall::fits(args, tall::FormId::r256_k32_s3, true, kSm120OptinShared) &&
         tall::fits(args, tall::FormId::r256_k32_s3, false, kSm120OptinShared) &&
         tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);  // the fallback takes a 256-row tile
    args.inputs = 1152U;
    ok = ok && tall::fits(args, tall::FormId::r256_k32_s3, true, kSm120OptinShared);
    args.inputs = 96U;  // 6 scale blocks: not whole words
    ok = ok && !tall::fits(args, tall::FormId::r256_k32_s3, true, kSm120OptinShared) &&
         tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);
    args.inputs = 192U;  // NVFP4 12 blocks fit, MXFP8 6 blocks do not
    ok = ok && tall::fits(args, tall::FormId::r256_k32_s3, true, kSm120OptinShared) &&
         !tall::fits(args, tall::FormId::r256_k32_s3, false, kSm120OptinShared);
    args.inputs = 2304U;
    ok = ok && !tall::fits(args, tall::FormId::r256_k32_s3, true, 90U * 1024U);  // 95,360 B do not fit 90 KiB
    check(ok, "256-row form admission: whole-word scale rows, shared budget", none, true);
    constexpr auto staged = tall::FormId::r128_k32_s4;
    constexpr auto direct = tall::FormId::r128_k32_s4_direct;
    tall::FormId chosen{};
    check(tall::resolve(args, true, kSm120OptinShared, chosen) && chosen == tall::FormId::r256_k32_s3,
          "wide descriptor resolves to r256 when it fits", none, true);
    const auto staged_bytes = tall::shared_bytes(narrow, tall::layout(narrow, true), args.inputs, true);
    check(tall::resolve(args, true, staged_bytes, chosen) && chosen == staged &&
          tall::resolve(args, true, staged_bytes - 1U, chosen) && chosen == direct,
          "wide budget fallback retains the staged and direct two-slice forms", none, true);
}

}  // namespace

int main() {
    for (const auto id : kForms) {
        for (const bool nvfp4 : {true, false}) {
            plans(id, nvfp4);
            for (const unsigned inputs : {32U, 64U, 96U, 160U, 1152U, 2304U})
                for (const std::uintptr_t address : {std::uintptr_t{0x1000}, std::uintptr_t{0x1002}})
                    scales(id, nvfp4, inputs, address);
            for (const unsigned live : {1U, 15U, 16U, 17U, 33U, 63U, 64U, 65U, 100U, 127U, 128U, 129U, 200U, 255U, 256U})
                if (live <= tall::form(id).rows) operands(id, nvfp4, live);
        }
        for (unsigned rows = 1; rows <= tall::kWideRows; ++rows) stores(id, rows);  // 129..256: row slices of a narrower form
        for (const unsigned steps : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 16U, 17U, 24U, 25U, 36U, 72U})
            for (const bool order : {true, false})
                if (!schedule(tall::form(id), steps, order)) check(false, "ring schedule (both orders)", id, true);
    }
    selection();
    wide_cut();
    if (failures != 0) {
        std::printf("grouped_tall_plan_test: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("grouped_tall_plan_test: PASS (7 forms x 2 formats: layout, copies, decode, operands, stores, ring schedule, "
                "chunked scales, wide cut)\n");
    return 0;
}
