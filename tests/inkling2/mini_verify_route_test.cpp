#include "attn_core/prepared_gqa_attention.hpp"
#include "core/gemv_rows.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"

#include <array>
#include <iostream>

namespace {

using strix::inkling2::MiniAppendKind;
using strix::inkling2::mini_resident_route;

constexpr bool check_routes() {
    for (bool packed : {false, true}) {
        for (bool graph : {false, true}) {
            for (unsigned rows = 1; rows <= strix::inkling2::kMiniMaxVerifyRows; ++rows) {
                const auto decode = mini_resident_route(MiniAppendKind::decode, 1U, packed, graph);
                const auto verify = mini_resident_route(MiniAppendKind::verify, rows, packed, graph);
                const auto prefill = mini_resident_route(MiniAppendKind::prefill, rows, packed, graph);
                if (decode.attention_rows != 1U || decode.packed_moe != packed || decode.step_graph != graph ||
                    verify.attention_rows != decode.attention_rows || verify.packed_moe != decode.packed_moe ||
                    verify.step_graph != (rows == 1U && graph) || prefill.attention_rows != rows ||
                    prefill.packed_moe != (rows == 1U && packed) || prefill.step_graph)
                    return false;
            }
            const auto prefill = mini_resident_route(MiniAppendKind::prefill, 128U, packed, graph);
            if (prefill.attention_rows != 128U || prefill.packed_moe || prefill.step_graph) return false;
            // Cohort decode (one row per lane, wider than a verify window): the experts take the same packed form
            // as a lone token, so a lane's bits do not depend on how many lanes share the step.
            for (unsigned rows = 2; rows <= 4U * strix::inkling2::kMiniMaxVerifyRows + 1U; ++rows) {
                const auto cohort = mini_resident_route(MiniAppendKind::decode, rows, packed, graph);
                if (cohort.packed_moe != packed || cohort.attention_rows != rows || cohort.step_graph) return false;
            }
        }
    }
    return true;
}

static_assert(check_routes());

// Cohort decode passes (the C17 served lanes defect: one 17-row pass put the dense rows past the HIP one-row cap):
// for every cohort size up to 512 lanes the passes tile the lanes in order, no pass exceeds the cap, a cohort within
// the cap is one pass, and a cohort above it is split evenly with no single-row pass.
constexpr bool check_cohort_passes() {
    constexpr unsigned cap = strix::core::gemv_rows::max_rows;
    static_assert(cap == 16U);
    for (unsigned lanes = 1U; lanes <= 512U; ++lanes) {
        const auto passes = strix::inkling2::mini_cohort_passes(lanes, cap);
        if (passes.count != (lanes + cap - 1U) / cap) return false;
        unsigned next = 0U, smallest = cap, largest = 0U;
        for (unsigned pass = 0; pass < passes.count; ++pass) {
            const auto rows = passes.rows(pass);
            if (passes.first(pass) != next || rows == 0U || rows > cap) return false;
            next += rows;
            smallest = rows < smallest ? rows : smallest;
            largest = rows > largest ? rows : largest;
        }
        if (next != lanes || largest - smallest > 1U || (lanes > cap && smallest < 2U)) return false;
    }
    const auto c17 = strix::inkling2::mini_cohort_passes(17U, cap);
    const auto c24 = strix::inkling2::mini_cohort_passes(24U, cap);
    const auto c32 = strix::inkling2::mini_cohort_passes(32U, cap);
    const auto c33 = strix::inkling2::mini_cohort_passes(33U, cap);
    const auto c16 = strix::inkling2::mini_cohort_passes(16U, cap);
    return c17.count == 2U && c17.rows(0) == 9U && c17.rows(1) == 8U && c17.first(1) == 9U &&
           c24.count == 2U && c24.rows(0) == 12U && c24.rows(1) == 12U &&
           c32.count == 2U && c32.rows(0) == 16U && c32.rows(1) == 16U &&
           c33.count == 3U && c33.rows(2) == 11U && c33.first(2) == 22U &&
           c16.count == 1U && c16.rows(0) == 16U &&
           strix::inkling2::mini_cohort_passes(0U, cap).count == 0U;
}

static_assert(check_cohort_passes());

struct Shape {
    static constexpr unsigned kQueryWidth = 4U;
    static constexpr unsigned kKvWidth = 2U;
    static constexpr unsigned kQueryHeads = 2U;
    static constexpr unsigned kRelativeExtent = 3U;
};

bool check_windows() {
    constexpr unsigned count = strix::inkling2::kMiniMaxVerifyRows;
    constexpr unsigned position = 1023U;
    std::array<float, count * Shape::kQueryWidth> query{}, output{};
    std::array<float, count * Shape::kKvWidth> keys{}, values{};
    std::array<float, count * Shape::kQueryHeads * Shape::kRelativeExtent> relative{};
    std::array<float, count * Shape::kQueryHeads> sinks{};
    const strix::attn_core::PreparedGqaAttentionRows input{
        query.data(), keys.data(), values.data(), relative.data(), sinks.data(), output.data(), position, count,
    };
    for (auto kind : {MiniAppendKind::prefill, MiniAppendKind::verify}) {
        const auto route = mini_resident_route(kind, input.count, true, true);
        unsigned visited = 0U;
        const auto status = strix::attn_core::for_each_prepared_gqa_window<Shape>(input, route.attention_rows, 0,
            [&](const auto& window) {
                const auto first = visited;
                visited += window.count;
                return window.count == route.attention_rows && window.first_position == position + first &&
                    window.queries == query.data() + first * Shape::kQueryWidth &&
                    window.output == output.data() + first * Shape::kQueryWidth &&
                    window.keys == keys.data() + first * Shape::kKvWidth &&
                    window.values == values.data() + first * Shape::kKvWidth &&
                    window.relative_logits == relative.data() + first * Shape::kQueryHeads * Shape::kRelativeExtent &&
                    window.sinks == sinks.data() + first * Shape::kQueryHeads ? 0 : 1;
            });
        if (status != 0 || visited != input.count) return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!check_windows()) {
        std::cerr << "FAIL Inkling Mini verify attention row views\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini verify route\n";
    return 0;
}
