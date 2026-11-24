#include "attn_core/prepared_gqa_attention.hpp"
#include "core/gemv_rows.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/serve/cohort_passes.hpp"

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

// Cohort decode pass cap (bound to the shared serve::plan_cohort_passes): packed decode takes the backend one-row cap
// (the C17 served lanes defect: one 17-row pass put the HIP dense rows past gemv_rows::max_rows onto WMMA); grouped
// decode keeps one pass of every lane, as before packed cohorts existed.
constexpr bool check_cohort_pass_cap() {
    constexpr unsigned cap = strix::core::gemv_rows::max_rows;
    static_assert(cap == 16U);
    using strix::inkling2::mini_cohort_pass_cap;
    using strix::serve::plan_cohort_passes;
    for (unsigned lanes = 1U; lanes <= 512U; ++lanes) {
        const auto grouped = plan_cohort_passes(lanes, mini_cohort_pass_cap(false, cap));
        if (grouped.count != 1U || grouped.rows(0U) != lanes) return false;
        const auto packed = plan_cohort_passes(lanes, mini_cohort_pass_cap(true, cap));
        if (packed.count != (lanes + cap - 1U) / cap || packed.rows(0U) > cap) return false;
    }
    const auto c17 = plan_cohort_passes(17U, mini_cohort_pass_cap(true, cap));
    const auto c34 = plan_cohort_passes(34U, mini_cohort_pass_cap(true, cap));
    return c17.count == 2U && c17.rows(0U) == 9U && c17.rows(1U) == 8U && c17.first(1U) == 9U &&
           c34.count == 3U && c34.rows(0U) == 12U && c34.rows(2U) == 11U && c34.first(2U) == 23U;
}

static_assert(check_cohort_pass_cap());

// One decode expert form resolver: the CLI keeps packed, strix-serve keeps grouped, a switch wins on either.
static_assert(strix::inkling2::mini_packed_decode(strix::inkling2::MiniDecodeEntry::cli, std::nullopt));
static_assert(!strix::inkling2::mini_packed_decode(strix::inkling2::MiniDecodeEntry::serve, std::nullopt));
static_assert(strix::inkling2::mini_packed_decode(strix::inkling2::MiniDecodeEntry::serve, true));
static_assert(!strix::inkling2::mini_packed_decode(strix::inkling2::MiniDecodeEntry::cli, false));

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
