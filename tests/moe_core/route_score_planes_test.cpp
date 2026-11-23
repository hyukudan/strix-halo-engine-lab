#include "mmsel_cases.hpp"

#include <memory>

int main() {
    using namespace mmsel;
    try {
        const auto in = std::make_unique<Inputs>();
        fill_inputs(*in);
        Outputs expected{}, paired{}, split{};
        for (const unsigned rows : {0U, 1U, kRows}) {
            expected.reset();
            paired.reset();
            split.reset();
            for (unsigned row = 0U; row < rows; ++row)
                frozen::router_top4_row(in->interleaved, row, expected.ids + 1U, expected.weights + 1U);
            moe::route_score_planes_ref<Policy>(in->paired(), {paired.ids + 1U, rows * kTop},
                                               {paired.weights + 1U, rows * kTop});
            moe::route_score_planes_ref<Policy>(in->split(), {split.ids + 1U, rows * kTop},
                                               {split.weights + 1U, rows * kTop});
            check_known_rows(expected, rows);
            compare(expected, expected, rows, "frozen guards");
            compare(paired, expected, rows, "interleaved");
            compare(split, expected, rows, "split");
        }
        moe::route_score_planes_ref<Policy>(moe::RouteScorePlanes<1>{}, {}, {});
        bool rejected = false;
        try {
            moe::route_score_planes_ref<Policy>(in->paired(), {paired.ids, 1U}, {paired.weights, 1U});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("partial row was accepted");
        std::puts("mmsel host ID and weight bytes PASS rows=1,259 layouts=interleaved,split");
        std::puts("mmsel host ties, NaN/Inf, output guards and zero rows PASS");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mmsel host FAIL: %s\n", error.what());
        return 1;
    }
}
