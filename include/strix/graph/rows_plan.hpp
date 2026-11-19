#pragma once
// Rows planner (host only, backend neutral): the stage graph over R token rows of one pass.
// rows = 1 is the step, 2..16 verify(R), more a prompt window: ONE form with rows as a parameter. Per stage it states the
// cross-row rule (independent rows; block-causal attention: row r sees keys up to p0 + r; per-row selection lists;
// history: a row's side input uses the window's own earlier tokens) and the weight bytes / MACs the R rows cost.
// Rows share every weight read except distinct stacked items: with independent routing of `read` of `items` per row,
// the expected distinct items are items x (1 - (1 - read / items)^R) (a DERIVED expectation, not a measurement).
// Home: shared (host shared / plan).
#include "strix/graph/stage_graph.hpp"

#include <cstdint>
#include <vector>

namespace strix::graph {

enum class RowsForm : std::uint8_t { step, verify, window };
enum class RowRule : std::uint8_t { independent, block_causal, per_row_select, history };

inline constexpr std::uint32_t kVerifyRowsMax = 16U;

struct RowsStage {
    std::uint32_t instance = 0;
    RowRule rule = RowRule::independent;
    double bytes = 0;  // expected weight bytes for the R rows
    std::uint64_t macs = 0;
};
struct RowsPlan {
    RowsForm form = RowsForm::step;
    std::uint32_t rows = 1;
    std::vector<Instance> instances = {};
    std::vector<RowsStage> stages = {};
    double bytes = 0;
    std::uint64_t macs = 0;
};

[[nodiscard]] RowsForm rows_form(std::uint32_t rows);
// Expected distinct items read by `rows` rows that each read `read` of `items` independently.
[[nodiscard]] double distinct_items(std::uint64_t items, std::uint64_t read, std::uint32_t rows);
// slot = position of the first row (the regime of the window's first row; split windows at regime edges upstream).
[[nodiscard]] RowsPlan rows_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows);

}  // namespace strix::graph
