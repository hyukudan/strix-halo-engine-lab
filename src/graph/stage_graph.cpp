#include "strix/graph/stage_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix::graph {

std::uint64_t row_bytes(Format f, std::uint64_t width) {
    const auto need = [&](std::uint64_t block) {
        if (width % block != 0U) throw std::invalid_argument("stage graph: row width not a multiple of the format block");
    };
    switch (f) {
        case Format::f32: return 4U * width;
        case Format::f16:
        case Format::bf16: return 2U * width;
        case Format::q8_0: need(32U); return width / 32U * 34U;
        case Format::q4_k: need(256U); return width / 256U * 144U;
        case Format::q6_k: need(256U); return width / 256U * 210U;
        case Format::iu4: need(64U); return width / 2U + width / 64U * 2U;  // IU4 g64: packed nibbles + FP16 scale per 64
        case Format::none: break;
    }
    throw std::invalid_argument("stage graph: operand without a format");
}

bool active(const StageGraph& g, const StageDecl& s, std::int32_t layer, std::uint32_t slot) {
    const Traits t = layer < 0 ? Traits{0} : g.layers.at(static_cast<std::size_t>(layer));
    if ((t & s.when_all) != s.when_all || (t & s.when_none) != 0U) return false;
    switch (s.pos) {
        case Pos::any: return true;
        case Pos::prefix_dense: return g.select_after != 0U && slot < g.select_after;
        case Pos::prefix_select: return g.select_after != 0U && slot >= g.select_after;
        case Pos::group_close: return g.group != 0U && (slot + 1U) % g.group == 0U;
    }
    return false;
}

Format format_of(const StageGraph& g, std::string_view tensor, std::int32_t layer) {
    Format f = Format::none;
    for (const auto& r : g.formats) {
        if (!r.tensor.empty() && r.tensor != tensor) continue;
        if (!r.layers.empty() &&
            (layer < 0 || std::find(r.layers.begin(), r.layers.end(), static_cast<std::uint32_t>(layer)) == r.layers.end()))
            continue;
        f = r.format;
    }
    return f;
}

std::vector<Instance> instances(const StageGraph& g, std::uint32_t slot) {
    std::vector<Instance> out;
    const auto add = [&](std::span<const StageDecl> table, std::int32_t layer) {
        for (const auto& s : table) {
            if (!active(g, s, layer, slot)) continue;
            out.push_back({&s, layer, s.operand.tensor.empty() ? Format::none : format_of(g, s.operand.tensor, layer)});
        }
    };
    add(g.before, -1);
    for (std::size_t l = 0; l < g.layers.size(); ++l) add(g.layer, static_cast<std::int32_t>(l));
    add(g.after, -1);
    return out;
}

std::uint64_t bytes(const Instance& s) {
    const auto& o = s.decl->operand;
    if (o.tensor.empty()) return 0U;
    return row_bytes(s.format, o.cols) * o.rows * o.read;
}

std::uint64_t macs(const Instance& s, std::uint32_t rows) {
    const auto& o = s.decl->operand;
    switch (s.decl->op) {
        case Op::linear:
        case Op::head:
        case Op::attend: return o.rows * o.cols * o.read * rows;
        default: return 0U;
    }
}

std::vector<Traits> layer_traits(std::uint32_t layers, std::span<const TraitList> lists) {
    std::vector<Traits> w(layers, Traits{0});
    for (const auto& t : lists) {
        for (const auto l : t.layers)
            if (l >= layers) throw std::invalid_argument("stage graph: trait list names a layer past the trunk");
        for (std::uint32_t l = 0; l < layers; ++l) {
            const bool listed = std::find(t.layers.begin(), t.layers.end(), l) != t.layers.end();
            if (listed != t.except) w[l] |= t.trait;
        }
    }
    return w;
}

namespace {
void check_table(const StageGraph& g, std::span<const StageDecl> table, bool trunk) {
    for (std::size_t i = 0; i < table.size(); ++i) {
        const auto& s = table[i];
        const std::string id(s.id);
        if (s.id.empty() || s.out.empty()) throw std::invalid_argument("stage graph: stage without id or output: " + id);
        for (std::size_t j = 0; j < i; ++j)
            if (table[j].id == s.id) throw std::invalid_argument("stage graph: duplicate stage id " + id);
        if ((s.op == Op::linear || s.op == Op::head) &&
            (s.operand.tensor.empty() || s.operand.rows == 0U || s.operand.cols == 0U || s.operand.read == 0U))
            throw std::invalid_argument("stage graph: projection without operand: " + id);
        if (s.leaves != 0U && s.leaves != 4U) throw std::invalid_argument("stage graph: leaves must be 0 or 4: " + id);
        if (s.operand.read > s.operand.items) throw std::invalid_argument("stage graph: reads more items than stacked: " + id);
        bool live = !trunk;
        if (!trunk && (s.when_all != 0U || s.when_none != 0U))
            throw std::invalid_argument("stage graph: edge stage with a layer guard: " + id);
        for (std::size_t l = 0; trunk && l < g.layers.size(); ++l) {
            const Traits t = g.layers[l];
            if ((t & s.when_all) != s.when_all || (t & s.when_none) != 0U) continue;
            live = true;
            if (!s.operand.tensor.empty())
                (void)row_bytes(format_of(g, s.operand.tensor, static_cast<std::int32_t>(l)), s.operand.cols);
        }
        if (!trunk && !s.operand.tensor.empty()) (void)row_bytes(format_of(g, s.operand.tensor, -1), s.operand.cols);
        if (!live) throw std::invalid_argument("stage graph: stage active on no layer: " + id);
    }
}
}  // namespace

void validate(const StageGraph& g) {
    check_table(g, g.before, false);
    check_table(g, g.layer, true);
    check_table(g, g.after, false);
}

}  // namespace strix::graph
