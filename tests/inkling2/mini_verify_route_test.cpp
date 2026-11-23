#include "attn_core/prepared_gqa_attention.hpp"
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

// The cohort head covers every lane exactly once, in order, with at most kMiniHeadRows rows per launch (the shared
// tight-row bucket that keeps one-row bits), and its contiguous rows fit the pitched logits buffer of the decoder.
constexpr bool check_head_chunks() {
    constexpr unsigned vocabulary = 200058U, pitch = 200060U;
    for (unsigned rows = 1; rows <= 4U * strix::inkling2::kMiniHeadRows + 3U; ++rows) {
        unsigned next = 0U, launches = 0U;
        for (unsigned first = 0; first < rows;) {
            const auto chunk = strix::inkling2::mini_head_chunk(rows, first);
            if (chunk.first != next || chunk.rows == 0U || chunk.rows > strix::inkling2::kMiniHeadRows) return false;
            if (std::size_t{chunk.first + chunk.rows} * vocabulary > std::size_t{rows - 1U} * pitch + vocabulary)
                return false;
            next = first = chunk.first + chunk.rows;
            ++launches;
        }
        if (next != rows || launches != (rows + strix::inkling2::kMiniHeadRows - 1U) / strix::inkling2::kMiniHeadRows)
            return false;
    }
    return true;
}

static_assert(check_head_chunks());

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
