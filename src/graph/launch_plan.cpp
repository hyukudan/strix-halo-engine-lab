#include "strix/graph/launch_plan.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace strix::graph {

std::uint32_t LaunchPlan::kernel_count() const {
    std::uint32_t n = 0;
    for (const auto& l : launches) n += l.kernels;
    return n;
}
std::uint32_t LaunchPlan::kernel_count(std::int32_t layer) const {
    std::uint32_t n = 0;
    for (const auto& l : launches)
        if (l.layer == layer) n += l.kernels;
    return n;
}

namespace {

constexpr int kNone = -1;     // no producer in this step (weights, caches, host state)
constexpr int kPending = -2;  // a reduction applied as the prologue of its consumers

bool reads(const StageDecl& s, std::string_view b) {
    return std::any_of(std::begin(s.in), std::end(s.in), [&](std::string_view x) { return !x.empty() && x == b; });
}
bool kernel_op(Op op) { return op != Op::upload && op != Op::download && op != Op::clear; }
bool seam_head(const StageDecl& s) { return s.op == Op::combine && s.form == "hc_expand"; }
template <class T> bool listed(std::span<const T> list, const T& v) { return std::find(list.begin(), list.end(), v) != list.end(); }

class Planner {
public:
    Planner(LaunchPlan& p, std::uint32_t rows, const FusionPolicy& f) : p_(p), rows_(rows), f_(f), owner_(p.instances.size(), kNone) {}

    void run() {
        for (std::uint32_t i = 0; i < p_.instances.size(); ++i) {
            if (f_.mode == Fusion::today) today(i);
            else fused(i);
            writer_[p_.instances[i].decl->out] = static_cast<int>(i);
        }
    }

private:
    LaunchPlan& p_;
    std::uint32_t rows_;
    const FusionPolicy& f_;
    std::vector<int> owner_;
    std::vector<int> head_src_;  // per launch: the writer of its head's first input when the launch opened
    std::map<std::string_view, int> writer_;

    const StageDecl& d(std::uint32_t i) const { return *p_.instances[i].decl; }
    int back() const { return static_cast<int>(p_.launches.size()) - 1; }
    int producer(std::string_view b) const {
        const auto it = writer_.find(b);
        return it == writer_.end() ? kNone : owner_[static_cast<std::size_t>(it->second)];
    }
    int writer(std::string_view b) const {
        const auto it = writer_.find(b);
        return it == writer_.end() ? kNone : it->second;
    }
    void open(std::uint32_t i) {
        Launch l;
        l.entry = d(i).entry;
        l.stages = {i};
        l.kernels = f_.mode == Fusion::today ? std::max<std::uint32_t>(1U, d(i).kernels) : 1U;
        l.layer = p_.instances[i].layer;
        l.rows = rows_;
        p_.launches.push_back(std::move(l));
        head_src_.push_back(writer(d(i).in[0]));
        owner_[i] = back();
    }
    void join(std::uint32_t i, int launch, std::vector<std::uint32_t> Launch::*part) {
        (p_.launches[static_cast<std::size_t>(launch)].*part).push_back(i);
        owner_[i] = launch;
    }
    // Prologues of pending reductions this stage reads, attached to the launch that runs it.
    void prologues(std::uint32_t i) {
        auto& l = p_.launches[static_cast<std::size_t>(owner_[i])];
        for (const auto b : d(i).in) {
            if (b.empty() || producer(b) != kPending) continue;
            const auto w = static_cast<std::uint32_t>(writer(b));
            if (!listed<std::uint32_t>(l.prologue, w)) l.prologue.push_back(w);
        }
    }

    // ---- check mode: today's launch list ----
    void today(std::uint32_t i) {
        const auto& s = d(i);
        const auto& in = p_.instances[i];
        const bool prev = i > 0U && owner_[i - 1U] == back() && p_.instances[i - 1U].layer == in.layer;
        if (prev && !s.call.empty() && d(i - 1U).call == s.call) {
            join(i, back(), &Launch::stages);
            p_.launches.back().kernels += s.kernels;
            return;
        }
        if (prev && s.op == Op::linear && d(i - 1U).op == Op::linear && d(i - 1U).in[0] == s.in[0] &&
            p_.instances[i - 1U].format == in.format && listed(f_.group_formats, in.format) &&
            (f_.group_sites.empty() || listed(f_.group_sites, s.site)) && d(i - 1U).site == s.site) {
            join(i, back(), &Launch::stages);
            return;
        }
        if (s.op == Op::elementwise && back() >= 0 && producer(s.in[0]) == back()) {
            const auto entry = p_.launches.back().entry;
            const bool cap = std::any_of(f_.epilogues.begin(), f_.epilogues.end(),
                                         [&](const EpilogueCap& c) { return c.entry == entry && c.form == s.form; });
            if (cap) return join(i, back(), &Launch::epilogue);
        }
        if (s.kernels == 0U) throw std::invalid_argument("launch plan: a 0-kernel stage outside its call");
        open(i);
    }

    // ---- fused rules ----
    bool prologue_capable(std::uint32_t i) const {
        const auto b = d(i).out;
        bool any = false;
        bool clobbered = false;
        for (std::size_t j = i + 1U; j < p_.instances.size(); ++j) {
            const auto& c = *p_.instances[j].decl;
            if (reads(c, b)) {
                // A prologue re-reads the reduction's inputs at its consumer: refuse when one was rewritten in between.
                if (clobbered) return false;
                any = true;
                if (!(c.op == Op::linear || c.op == Op::head || seam_head(c))) return false;
            }
            if (c.out == b) break;
            if (reads(d(i), c.out)) clobbered = true;
        }
        return any;
    }
    // Value b is readable by a stage placed in launch k: as a main stage (producer strictly earlier) or as an epilogue
    // (k's own stages may produce it). A pending reduction runs as k's prologue: its inputs must come from before k.
    bool available(std::string_view b, int k, bool epilogue) const {
        if (b.empty()) return true;
        const int p = producer(b);
        if (p == kPending) {
            for (const auto x : d(static_cast<std::uint32_t>(writer(b))).in)
                if (!available(x, k, false)) return false;
            return true;
        }
        return epilogue ? p <= k : p < k;
    }
    bool available_all(std::uint32_t i, int k, bool epilogue) const {
        return std::all_of(std::begin(d(i).in), std::end(d(i).in), [&](std::string_view b) { return available(b, k, epilogue); });
    }
    // Placing stage i in launch k moves its write ahead of every stage already placed in k or later (pending reductions
    // run later, as prologues): refuse when one of them reads or writes the same name.
    bool out_free(std::uint32_t i, int k) const {
        const auto b = d(i).out;
        if (b.empty()) return true;
        for (std::uint32_t j = 0; j < i; ++j) {
            const int o = owner_[j];
            if ((o == kPending || o >= k) && (d(j).out == b || reads(d(j), b))) return false;
        }
        return true;
    }
    // Inputs written by two different linear members of one grouped launch: not tile-local.
    bool split_inputs(std::uint32_t i) const {
        int seen = kNone;
        for (const auto b : d(i).in) {
            if (b.empty()) continue;
            const int w = writer(b);
            if (w < 0 || d(static_cast<std::uint32_t>(w)).op != Op::linear) continue;
            const int l = owner_[static_cast<std::size_t>(w)];
            if (l < 0) continue;
            if (seen != kNone && seen == l) return true;
            seen = l;
        }
        return false;
    }
    void fused(std::uint32_t i) {
        const auto& s = d(i);
        const auto& in = p_.instances[i];
        if (s.op == Op::clear) {  // rule 5
            if (back() >= 0 && kernel_op(d(p_.launches.back().stages.front()).op)) return join(i, back(), &Launch::epilogue);
            return open(i);
        }
        if (s.op == Op::linear) {  // rule 1
            for (int k = back(); k >= 0 && p_.launches[static_cast<std::size_t>(k)].layer == in.layer; --k) {
                const auto h = p_.launches[static_cast<std::size_t>(k)].stages.front();
                if (d(h).op != Op::linear || d(h).in[0] != s.in[0] || p_.instances[h].format != in.format ||
                    head_src_[static_cast<std::size_t>(k)] != writer(s.in[0]))
                    continue;  // a different value in the same buffer name is a different input
                if (!available_all(i, k, false) || !out_free(i, k)) continue;
                join(i, k, &Launch::stages);
                return prologues(i);
            }
            open(i);
            return prologues(i);
        }
        const bool foldable = s.op == Op::elementwise || s.op == Op::store || s.op == Op::copy || (s.op == Op::combine && !seam_head(s));
        if (s.op == Op::norm || (foldable && (s.reduce || split_inputs(i)))) {  // rule 3
            if (prologue_capable(i)) {
                owner_[i] = kPending;
                return;
            }
            const int k = back();
            if (s.op == Op::norm && k >= 0 && d(p_.launches[static_cast<std::size_t>(k)].stages.front()).op == Op::norm &&
                p_.launches[static_cast<std::size_t>(k)].layer == in.layer && available_all(i, k, false) && out_free(i, k)) {
                join(i, k, &Launch::stages);  // adjacent independent vector chains: one launch
                return prologues(i);
            }
            open(i);
            return prologues(i);
        }
        if (foldable) {  // rule 2
            int last = kNone;
            for (const auto b : s.in)
                if (!b.empty()) last = std::max(last, producer(b));
            if (last >= 0 && kernel_op(d(p_.launches[static_cast<std::size_t>(last)].stages.front()).op) &&
                available_all(i, last, true) && out_free(i, last)) {
                join(i, last, &Launch::epilogue);
                return prologues(i);
            }
            open(i);
            return prologues(i);
        }
        if (s.op == Op::mix && back() >= 0 && seam_head(d(p_.launches.back().stages.front())) &&
            producer(s.in[0]) == back()) {  // rule 4
            join(i, back(), &Launch::stages);
            return prologues(i);
        }
        open(i);
        prologues(i);
    }
};

}  // namespace

LaunchPlan launch_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows, const FusionPolicy& policy) {
    if (rows == 0U) throw std::invalid_argument("launch plan: rows must be >= 1");
    LaunchPlan p;
    p.instances = instances(g, slot);
    Planner(p, rows, policy).run();
    return p;
}

}  // namespace strix::graph
