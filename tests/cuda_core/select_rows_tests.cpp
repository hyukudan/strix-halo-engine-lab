#include "cuda/core/select_rows.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {
namespace sel = strix::cuda::core::select;
namespace exec = strix::exec;

static_assert(std::is_trivially_copyable_v<sel::AffineRows>);
static_assert(std::is_trivially_copyable_v<sel::SlotRows>);
static_assert(std::is_trivially_copyable_v<sel::SlotRowBase>);

void require(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "K5A row producer FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}
} // namespace

int main() {
    constexpr sel::RowStrides stride{11, 7, 5};
    std::array<float, 90> query{}, scores{}, keys{};
    std::array<unsigned, 50> selected{};
    const sel::RowBases base{query.data() + 1, keys.data() + 3, scores.data() + 2, selected.data() + 4};
    const sel::AffineRows affine{base, stride, 2048};
    sel::detail::VisibleRow input{};
    for (unsigned i = 0; i < 4; ++i) {
        require(affine.load(i, input), "affine row is live");
        const auto score = input.score(4);
        const auto pick = input.pick(4);
        require(score.query == base.query + i * stride.query && score.keys == base.keys, "query and shared keys");
        require(score.scores == base.scores + i * stride.scores && pick.scores == score.scores &&
                pick.mask_scores == score.scores && pick.selected == base.selected + i * stride.selected,
                "workspace pitches and score alias");
        require(score.visible == (i == 3 ? 513U : 512U) && pick.groups == score.visible &&
                pick.tail == (i + 1U) % 4U, "complete groups and partial tail across top-k");
    }

    exec::StepBlock step{};
    step.h.rows = 5;
    step.h.valid_mask = 0b111011U;
    step.row[0] = {0, 2050, 1, 0, 0, exec::step_valid, 0, 0};
    step.row[1] = {0, 64, 0, 0, 0, exec::step_valid, 0, 0};
    step.row[2] = step.row[0];  // No mask bit.
    step.row[3] = step.row[0];
    step.row[3].flags = 0;
    step.row[4] = step.row[0];
    step.row[4].slot = 2;
    step.row[5] = step.row[0];  // Outside header width, even with both validity bits.
    const std::array table{sel::SlotRowBase{{nullptr, nullptr, nullptr, selected.data()}, 64},
                           sel::SlotRowBase{base, 2048}};
    const sel::SlotRows slots{&step, table.data(), static_cast<unsigned>(table.size()), stride};
    require(slots.load(0, input), "slot resolves first row");
    const auto slot = input.score(4);
    require(slot.query == base.query + 2 * stride.query && slot.keys == base.keys &&
            slot.scores == base.scores + 2 * stride.scores && slot.visible == 512 &&
            input.pick(4).tail == 3, "slot position controls scratch offset and visibility");
    require(slots.load(1, input) && input.pick(4).selected == selected.data() &&
            input.pick(4).scores == nullptr && input.score(4).query == nullptr,
            "other slot supplies its own bases, including absent score storage");
    for (unsigned i = 2; i < 6; ++i) require(!slots.load(i, input), "invalid row is skipped");
    require(!slots.load(exec::kStepMaxRows, input), "row outside the descriptor is skipped");
    step.row[1].position = 63;
    require(!slots.load(1, input), "position before slot origin is skipped");
    std::puts("K5A CPU affine and X3 slot row producers PASS");
}
