#include "strix/graph/rows_plan.hpp"

#include <cmath>
#include <stdexcept>

namespace strix::graph {

RowsForm rows_form(std::uint32_t rows) {
    if (rows == 0U) throw std::invalid_argument("rows plan: rows must be >= 1");
    return rows == 1U ? RowsForm::step : rows <= kVerifyRowsMax ? RowsForm::verify : RowsForm::window;
}

double distinct_items(std::uint64_t items, std::uint64_t read, std::uint32_t rows) {
    if (items == 0U || read > items) throw std::invalid_argument("rows plan: read exceeds items");
    if (read == items) return static_cast<double>(items);
    const double miss = 1.0 - static_cast<double>(read) / static_cast<double>(items);
    return static_cast<double>(items) * (1.0 - std::pow(miss, static_cast<double>(rows)));
}

namespace {
RowRule rule_of(const StageDecl& s) {
    if (s.rows == Rows::per_row) return RowRule::independent;
    switch (s.op) {
        case Op::attend: return RowRule::block_causal;
        case Op::select: return RowRule::per_row_select;
        default: return RowRule::history;
    }
}
}  // namespace

RowsPlan rows_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows) {
    RowsPlan p;
    p.form = rows_form(rows);
    p.rows = rows;
    p.instances = instances(g, slot);
    p.stages.reserve(p.instances.size());
    for (std::uint32_t i = 0; i < p.instances.size(); ++i) {
        const auto& in = p.instances[i];
        const auto& o = in.decl->operand;
        RowsStage r;
        r.instance = i;
        r.rule = rule_of(*in.decl);
        if (!o.tensor.empty()) {
            const double per_item = static_cast<double>(row_bytes(in.format, o.cols) * o.rows);
            r.bytes = per_item * (o.read == 1U && o.items == 1U ? 1.0 : distinct_items(o.items, o.read, rows));
        }
        r.macs = macs(in, rows);
        p.bytes += r.bytes;
        p.macs += r.macs;
        p.stages.push_back(r);
    }
    return p;
}

}  // namespace strix::graph
