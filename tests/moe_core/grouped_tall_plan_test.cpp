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
//     for both decode / multiply orders inside an iteration (warp rows alternate them);
//  7. selection and admission.
#include "cuda/core/mma_tile.cuh"
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
                                   tall::FormId::r64_k32_s4};
constexpr std::size_t kSm120OptinShared = 99U * 1024U;

static_assert(tall::valid(tall::form(tall::FormId::r128_k32_s4), true));
static_assert(tall::layout(tall::form(tall::FormId::r128_k32_s4), true).smem == 69632U + 2176U);
static_assert(tall::layout(tall::form(tall::FormId::r128_k32_s4), false).smem == 77824U);

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
    };
    std::vector<Group> groups;
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
        groups.push_back({static_cast<int>(stage), q});
    };
    const auto commit_empty = [&] { groups.push_back({-1, -1}); };
    const auto wait_barrier = [&](unsigned pending) {
        const unsigned done = groups.size() > pending ? static_cast<unsigned>(groups.size()) - pending : 0U;
        for (unsigned g = completed; g < done; ++g)
            if (groups[g].stage >= 0 && slot_group[groups[g].slot] == static_cast<int>(g))
                slot_visible[groups[g].slot] = interval + 1;
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
    for (unsigned s = 0; s + 1U < f.stages; ++s) {
        if (s < steps) issue(s);
        else commit_empty();
    }
    wait_barrier(tall::prologue_pending(f));
    decode(0U);
    for (unsigned s = 0; s < steps; ++s) {
        wait_barrier(tall::loop_pending(f));
        if (tall::copy_stage(f, s) < steps) issue(tall::copy_stage(f, s));
        else commit_empty();
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
    args.max_tile_rows = 129U;
    ok = ok && !tall::fits(args, tall::FormId::r128_k32_s4, true, kSm120OptinShared);
    args.max_tile_rows = 128U;
    ok = ok && !tall::fits(args, tall::FormId::r128_k32_s4, true, 48U * 1024U);  // budget below the form's shared bytes
    check(ok, "ragged inputs / outputs, oversize tiles and a small shared budget refused", none, true);
    ok = mma::microscaled::kRowsPerTile == tall::kMaxRows;
    check(ok, "plan rows per tile = the tall CTA extent", none, true);
}

}  // namespace

int main() {
    for (const auto id : kForms) {
        for (const bool nvfp4 : {true, false}) {
            plans(id, nvfp4);
            for (const unsigned live : {1U, 15U, 16U, 17U, 33U, 63U, 64U, 65U, 100U, 127U, 128U})
                if (live <= tall::form(id).rows) operands(id, nvfp4, live);
        }
        for (unsigned rows = 1; rows <= tall::kMaxRows; ++rows) stores(id, rows);
        for (const unsigned steps : {1U, 2U, 3U, 4U, 5U, 7U, 36U, 72U})
            for (const bool order : {true, false})
                if (!schedule(tall::form(id), steps, order)) check(false, "ring schedule (both orders)", id, true);
    }
    selection();
    if (failures != 0) {
        std::printf("grouped_tall_plan_test: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("grouped_tall_plan_test: PASS (4 forms x 2 formats: layout, copies, decode, operands, stores, ring schedule)\n");
    return 0;
}
