#include "attn_position_split_cases.hpp"

int main() {
    namespace core = strix::attn_core;
    using namespace b6_fixture;

    const core::PositionSplitPlan plan{1027, 270};
    require(plan.split_count() == 5 && core::PositionSplitPlan{0, 0}.split_count() == 0, "split count");
    require(plan.row_keys(0).begin == 0 && plan.row_keys(0).end == 1, "first token");
    require(plan.row_keys(1026).begin == 757 && plan.row_keys(1026).end == 1027, "clipped window");
    require(plan.row_keys(1027).end == 0, "out-of-range row");
    require(core::position_split(255) == 0 && core::position_split(256) == 1 &&
            core::position_split(511) == 1 && core::position_split(512) == 2 &&
            core::position_split_begin(4) == 1024, "absolute boundaries");
    std::puts("b6 host absolute-chunks PASS");

    const Case cancellation{"cancellation", 1, 1, 1, 259, 0, 259, Layout::slot, {}};
    Data data{std::vector<float>(1), std::vector<float>(259), std::vector<float>(259)};
    data.v[255] = 16777216.0F;
    data.v[256] = 1.0F;
    data.v[257] = 1.0F;
    data.v[258] = -16777216.0F;
    const Row row{258, 0, 258, 0};
    const core::PositionSplitPlan cancellation_plan{259, 0};
    compare("merge order", {2.0F / 259.0F}, serial_row(cancellation, data, row, &cancellation_plan));
    std::puts("b6 host canonical-merge PASS");

    for (const auto& c : cases) {
        const auto input = make_data(c, true);
        const core::PositionSplitPlan batch_plan{c.context, c.window};
        for (const auto& r : c.rows) {
            const core::PositionSplitPlan single_plan{r.pos + 1U, c.window};
            const auto oracle = serial_row(c, input, r);
            compare(c.name, oracle, serial_row(c, input, r, &batch_plan));
            compare(c.name, oracle, serial_row(c, input, r, &single_plan));
        }
        std::printf("b6 host %s serial-bits PASS\n", c.name);
    }
}
