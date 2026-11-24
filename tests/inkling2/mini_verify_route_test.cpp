#include "attn_core/prepared_gqa_attention.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/serve/cohort_passes.hpp"

#include <array>
#include <iostream>
#include <optional>

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
        }
    }
    return true;
}

static_assert(check_routes());

// Packed decode (the one default of the command line and strix-serve): every lane of a cohort of any width runs the
// packed expert form, in passes no wider than the one-row cap of the dense rows family (16 on HIP), exactly as
// MiniResidentDecoder::decode_slots splits and routes them. A lone lane is the width-1 case.
constexpr bool check_cohort_lanes() {
    using strix::inkling2::mini_cohort_pass_cap;
    if (!strix::inkling2::mini_packed_decode(std::nullopt) || !strix::inkling2::mini_packed_decode(true) ||
        strix::inkling2::mini_packed_decode(false))
        return false;
    constexpr unsigned one_row_cap = 16U;
    for (unsigned lanes = 1U; lanes <= 64U; ++lanes) {
        const auto passes = strix::serve::plan_cohort_passes(lanes, mini_cohort_pass_cap(true, one_row_cap));
        unsigned covered = 0U;
        for (unsigned pass = 0; pass < passes.count; ++pass) {
            const auto rows = passes.rows(pass);
            if (rows == 0U || rows > one_row_cap || passes.first(pass) != covered) return false;
            covered += rows;
            const bool verify = rows <= strix::inkling2::kMiniMaxVerifyRows;
            const auto kind = verify ? MiniAppendKind::verify : MiniAppendKind::decode;
            if (!mini_resident_route(kind, rows, true, false).packed_moe) return false;
        }
        if (covered != lanes) return false;
        // Grouped decode keeps one pass of every lane (no batch-invariance contract).
        if (strix::serve::plan_cohort_passes(lanes, mini_cohort_pass_cap(false, one_row_cap)).count != 1U) return false;
    }
    return true;
}

static_assert(check_cohort_lanes());

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
