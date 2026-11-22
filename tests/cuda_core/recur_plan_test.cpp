#include "recur_cases.hpp"

#include "strix/state/txn.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

namespace rec = strix::cuda::recur;
namespace state = strix::state;
using state::Rollback;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void mapping() {
    const auto d = rec::test::qwen27;
    auto decls = rec::test::declarations(d, Rollback::snapshot, Rollback::replay);
    const auto speculative = rec::make_plan(d, decls[0], decls[1], rec::Execution::speculative);
    check(speculative && speculative->matrix.write == rec::Write::ring_slot &&
          speculative->history.write == rec::Write::replay_input, "mixed S1 write forms");
    check(speculative->qk_width == 2048 && speculative->value_width == 6144 && speculative->channels == 10240 &&
          speculative->matrix.image_floats == 786432 && speculative->history.image_floats == 30720 &&
          speculative->matrix.replay_floats == 0 && speculative->history.replay_floats == 10240, "Qwen geometry");
    const auto committed = rec::make_plan(d, decls[0], decls[1], rec::Execution::committed);
    check(committed && committed->matrix.write == rec::Write::in_place && committed->history.write == rec::Write::in_place,
          "committed execution uses published image");

    auto bad = d;
    bad.kind = rec::Kind::kda;
    check(!rec::make_plan(bad, decls[0], decls[1], rec::Execution::speculative), "KDA admitted as GDN");
    bad = d;
    bad.value_dim = 127;
    check(!rec::make_plan(bad, decls[0], decls[1], rec::Execution::speculative), "unsupported warp width");
    bad = d;
    bad.key_heads = 0xffffffffU;
    check(!rec::make_plan(bad, decls[0], decls[1], rec::Execution::speculative), "overflowing channel count");
    --decls[1].replay_input_bytes;
    check(!rec::make_plan(d, decls[0], decls[1], rec::Execution::speculative), "wrong replay row byte count");
    ++decls[1].replay_input_bytes;
    decls[1].checkpoint_interval = 4;
    check(!rec::make_plan(d, decls[0], decls[1], rec::Execution::speculative), "checkpointed replay admitted");
    decls[1].checkpoint_interval = 0;
    decls[0].dims[0] = 64;
    check(!rec::make_plan(d, decls[0], decls[1], rec::Execution::speculative), "wrong matrix geometry");
    decls[0].dims[0] = d.value_dim;
    decls[0].rollback = Rollback::rewind;
    check(!rec::make_plan(d, decls[0], decls[1], rec::Execution::committed), "recurrent rewind admitted");
}

void binding() {
    const auto d = rec::test::other_dims;
    const auto decls = rec::test::declarations(d, Rollback::replay, Rollback::snapshot);
    const auto plan = rec::make_plan(d, decls[0], decls[1], rec::Execution::speculative);
    check(plan && plan->channels == 544 && plan->matrix.replay_floats == 550 && plan->history.image_floats == 2176,
          "independent key/value dimensions and tap width");
    constexpr std::uint32_t pending = 7;
    const auto layout = rec::test::layout(decls, pending);
    const auto& matrix = layout.instances[0];
    const auto& taps = layout.instances[1];
    std::vector<float> pool(layout.pool_bytes / sizeof(float));
    const auto m = rec::bind_state(plan->matrix, matrix, pool.data(), 2, pending);
    const auto t = rec::bind_state(plan->history, taps, pool.data() + taps.pool_offset / sizeof(float), 6, pending);
    check(m && m->slots == 3 && m->stride == matrix.row_stride / sizeof(float) &&
          m->input == pool.data() + matrix.input_offset / sizeof(float) && m->input_rows == pending, "replay input view");
    check(t && t->slots == 8 && !t->input && t->stride > plan->history.image_floats &&
          rec::image_slot(rec::Write::ring_slot, *t, 1) == 0 &&
          rec::image_slot(rec::Write::ring_slot, *t, 6) == 5, "wrapped ring view");
    check(rec::image_slot(rec::Write::in_place, *m, 6) == 2 && rec::image_slot(rec::Write::replay_input, *m, 1) == 1 &&
          rec::image_slot(rec::Write::replay_input, *m, 6) == 0, "working and published slot mapping");
    check(!rec::bind_state(plan->matrix, matrix, pool.data(), 3, pending), "published index outside layout");
    check(!rec::bind_state(plan->matrix, matrix, pool.data(), 0, pending + 1), "alignment padding treated as a replay row");
    check(!rec::bind_state(plan->history, taps, pool.data(), 0, pending + 1), "missing ring image");

    // Compare the indices against actual publication by the S1 CPU reference,
    // using a partial accept and another transaction across the ring boundary.
    const state::Update zero = [](std::uint32_t, state::Row, state::Row, std::span<std::byte> out) {
        std::fill(out.begin(), out.end(), std::byte{});
    };
    const state::Update updates[]{zero, zero};
    state::CpuState cpu(layout, updates);
    state::Txn txn(cpu, 0);
    const std::vector<std::byte> row(matrix.replay_input_bytes);
    const state::Row inputs[]{row, {}};
    for (const std::uint32_t accepted : {6U, 2U, 7U}) {
        auto mv = *m;
        auto tv = *t;
        mv.published = static_cast<std::uint32_t>((cpu.entry(0, 0).base - matrix.pool_offset) / matrix.row_stride);
        tv.published = static_cast<std::uint32_t>((cpu.entry(1, 0).base - taps.pool_offset) / taps.row_stride);
        txn.begin(pending);
        for (std::uint32_t i = 0; i < pending; ++i) txn.append(inputs);
        txn.commit(accepted);
        check(cpu.entry(0, 0).base == matrix.pool_offset + matrix.row_stride *
              rec::image_slot(plan->matrix.write, mv, accepted - 1), "S1 replay publication");
        check(cpu.entry(1, 0).base == taps.pool_offset + taps.row_stride *
              rec::image_slot(plan->history.write, tv, accepted - 1), "S1 ring publication");
    }
}

}  // namespace

int main() {
    try {
        mapping();
        binding();
        std::puts("k9c plan PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "k9c plan FAIL: %s\n", error.what());
        return 1;
    }
}
