#include "attn_core/prepared_gqa_attention.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/serve/cohort_passes.hpp"

#include <array>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

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
    if (!strix::inkling2::kMiniPackedDecodeDefault || !strix::inkling2::mini_packed_decode(false, false) ||
        strix::inkling2::mini_packed_decode(false, true))
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
            if (!mini_resident_route(kind, rows, true, false, true).packed_moe) return false;
            // Without packed cohort rows (no served route lever) a multi-row decode keeps the incumbent grouped form.
            if (kind == MiniAppendKind::decode && mini_resident_route(kind, rows, true, false).packed_moe) return false;
        }
        if (covered != lanes) return false;
        // Grouped decode keeps the incumbent served schedule: the whole cohort in one batched forward.
        const auto grouped = strix::serve::plan_cohort_passes(lanes, mini_cohort_pass_cap(false, one_row_cap));
        if (grouped.count != 1U || grouped.rows(0U) != lanes || grouped.first(0U) != 0U) return false;
        if (mini_resident_route(MiniAppendKind::verify, 1U, false, false).packed_moe) return false;
    }
    return true;
}

static_assert(check_cohort_lanes());

// The transaction MiniResidentDecoder::decode_slots runs (serve::run_cohort_passes, the production call): 17 lanes at
// the HIP one-row cap 16 are passes of 9 + 8 rows; with ids = buf[0..17) and top = buf[1..18) the first pass's picks
// would overwrite the first input of the second pass without the snapshot. A second pass that throws must leave every
// output unchanged and retire the cohort once.
bool check_cohort_transaction() {
    constexpr unsigned lanes = 17U;
    for (const bool fail_second : {false, true}) {
        std::array<unsigned, lanes + 1U> buf{};
        for (unsigned i = 0; i <= lanes; ++i) buf[i] = 100U + i;
        const auto before = buf;
        const auto ids = std::span<const unsigned>(buf).first(lanes);
        const auto top = std::span<unsigned>(buf).subspan(1U, lanes);
        std::vector<unsigned> snapshot, picks;
        unsigned passes = 0U, retired = 0U;
        std::array<bool, lanes> lane_retired{};
        bool threw = false;
        try {
            strix::serve::run_cohort_passes(ids, top, strix::inkling2::mini_cohort_pass_cap(true, 16U), snapshot, picks,
                [&](unsigned first, unsigned rows, std::span<const unsigned> pass_ids, std::span<unsigned> out) {
                    if (passes++ == 1U && fail_second) throw std::runtime_error("pass two failed");
                    if (first != (passes == 1U ? 0U : 9U) || rows != (passes == 1U ? 9U : 8U)) throw std::logic_error("split");
                    for (unsigned r = 0; r < rows; ++r) out[r] = pass_ids[r] + 1000U;
                    // A writer racing the outputs, on the successful arm only (the snapshot check): the failure arm
                    // must leave the whole buffer untouched.
                    if (!fail_second)
                        for (unsigned r = 0; r < rows; ++r) buf[1U + first + r] = 7U;
                }, [&] {
                    ++retired;
                    lane_retired.fill(true);  // the production retire poisons every lane of the cohort (decode_slots_served)
                });
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (fail_second) {
            // Nothing published: the whole buffer is unchanged (pass one's picks too), every lane retired once.
            if (!threw || retired != 1U || passes != 2U || buf != before) return false;
            for (const bool lane : lane_retired) if (!lane) return false;
            continue;
        }
        if (threw || retired != 0U || passes != 2U) return false;
        for (unsigned i = 0; i < lanes; ++i) if (buf[1U + i] != before[i] + 1000U) return false;  // inputs read unchanged
    }
    // Grouped decode: the whole cohort in one pass (the incumbent served schedule).
    std::array<unsigned, 17> ids{}, top{};
    std::vector<unsigned> snapshot, picks;
    unsigned calls = 0U;
    strix::serve::run_cohort_passes(ids, top, strix::inkling2::mini_cohort_pass_cap(false, 16U), snapshot, picks,
        [&](unsigned first, unsigned rows, std::span<const unsigned>, std::span<unsigned>) {
            calls += first == 0U && rows == 17U ? 1U : 100U;
        }, [] {});
    return calls == 1U;
}

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
    if (!check_cohort_transaction()) {
        std::cerr << "FAIL Inkling Mini cohort pass transaction (snapshot, publish after the last pass, retire)\n";
        return 1;
    }
    if (!check_windows()) {
        std::cerr << "FAIL Inkling Mini verify attention row views\n";
        return 1;
    }
    std::cout << "PASS Inkling Mini verify route\n";
    return 0;
}
