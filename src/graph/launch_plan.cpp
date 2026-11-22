#include "strix/graph/launch_plan.hpp"
#include "strix/device/profiles.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

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

using StageIndex = std::uint32_t;
using Sources = std::array<int, 4>;

std::vector<Sources> sources(std::span<const Instance> instances) {
    std::map<std::string_view, int> writers;
    std::vector<Sources> result;
    result.reserve(instances.size());
    for (const auto& instance : instances) {
        const auto& s = *instance.decl;
        Sources row{};
        for (std::size_t i = 0; i < row.size(); ++i) {
            const auto it = writers.find(s.in[i]);
            row[i] = it == writers.end() ? kNone : it->second;
        }
        if (!s.out.empty()) writers[s.out] = static_cast<int>(result.size());
        result.push_back(row);
    }
    return result;
}

bool state_form(Form form) {
    return form == Form::kv_append || form == Form::conv_tap ||
           form == Form::recur_commit || form == Form::index_pool_append;
}
bool store_form(Form form) {
    return (form >= Form::store_f32 && form <= Form::store_e4m3) || form == Form::store_iu4prep;
}
bool transform_form(Form form) { return form == Form::rotate_hadamard || form == Form::load_round_bf16; }
bool wait_form(Form form) { return form == Form::xch_join || form == Form::wait_host; }
bool emit_form(Form form) { return form == Form::emit_argmax || form == Form::emit_top2 || form == Form::emit_topn; }

Link stage_form(const StageDecl& stage, const PlanStage& binding) {
    if (binding.form.form != Form::none) return binding.form;
    if (const auto form = parse_form(stage.form)) return {*form, 0};
    if (stage.op == Op::norm) return {Form::norm_rms, 0};
    if (stage.form == "silu_mul") return {Form::act_silu_mul, 0};
    if (seam_head(stage)) return {Form::mix_expand, 0};
    return {};
}

bool append(Chain& chain, Link link) {
    if (link.form == Form::none) return true;
    if (chain.count == chain.links.size()) return false;
    chain.links[chain.count++] = link;
    return true;
}

struct StageEffects {
    const PlanStage* binding = nullptr;
    Link form = {}, input = {}, output = {};
    Chain stores = {};
};

const PlanStage& binding_for(const PlanRequest& request, const StageDecl& stage) {
    static constexpr PlanStage default_binding{};
    const auto it = std::find_if(request.stages.begin(), request.stages.end(),
                                 [&](const PlanStage& b) { return b.stage == &stage; });
    return it == request.stages.end() ? default_binding : *it;
}

class RequestStages {
public:
    RequestStages(LaunchPlan& plan, const PlanRequest& request) {
        if (request.phase == Phase::state_only) prune(plan, request);
        inputs = sources(plan.instances);
        effects.reserve(plan.instances.size());
        users.resize(plan.instances.size());
        for (StageIndex i = 0; i < plan.instances.size(); ++i) {
            const auto& s = *plan.instances[i].decl;
            const auto& b = binding_for(request, s);
            effects.push_back({&b, stage_form(s, b), b.input_transform});
            for (const int producer : inputs[i]) {
                if (producer == kNone) continue;
                auto& use = users[static_cast<std::size_t>(producer)];
                if (use.empty() || use.back() != i) use.push_back(i);
            }
        }
        for (std::size_t i = 0; i < users.size(); ++i) {
            const auto& s = *plan.instances[i].decl;
            const Form form = effects[i].form.form;
            // Keep the canonical row when a user does not declare an activation class.
            if (!kernel_op(s.op) || state_form(form) || wait_form(form) || form == Form::xch_publish ||
                s.op == Op::select || s.op == Op::norm || s.reduce || users[i].empty()) continue;
            auto& e = effects[i];
            const auto required = effects[users[i].front()].input;
            bool shared = required.form != Form::none;
            std::array<bool, static_cast<std::size_t>(Form::store_iu4prep) + 1U> stores{};
            for (const auto user : users[i]) {
                const auto& in = inputs[user];
                const bool primary_only = in[0] == static_cast<int>(i) &&
                    std::find(in.begin() + 1, in.end(), static_cast<int>(i)) == in.end();
                const auto store = effects[user].binding->operand_store;
                if (primary_only && store != Form::none) stores[static_cast<std::size_t>(store)] = true;
                shared = shared && primary_only && effects[user].input == required;
            }
            if (shared) {  // rule 14
                e.output = required;
                for (const auto user : users[i]) effects[user].input = {};
            }
            for (std::size_t id = 0; id < stores.size(); ++id)
                if (stores[id]) (void)append(e.stores, {static_cast<Form>(id), 0});
            if (e.stores.count < 2) e.stores = {};  // rule 12
        }
    }

    std::vector<Sources> inputs;
    std::vector<StageEffects> effects;
    std::vector<std::vector<StageIndex>> users;

private:
    static void prune(LaunchPlan& plan, const PlanRequest& request) {  // rule 13
        const auto in = sources(plan.instances);
        std::vector<bool> live(in.size());
        for (std::size_t i = in.size(); i-- > 0;) {
            const auto& s = *plan.instances[i].decl;
            if (!live[i] && !state_form(stage_form(s, binding_for(request, s)).form)) continue;
            live[i] = true;
            for (const auto source : in[i])
                if (source != kNone) live[static_cast<std::size_t>(source)] = true;
        }
        std::size_t i = 0;
        std::erase_if(plan.instances, [&](const Instance&) { return !live[i++]; });
    }
};

struct JobChains {
    Chain prologue = {}, epilogue = {}, stores = {};
    Link emit = {};
    StageIndex value = no_launch;
    bool exact_store = false;
};

class Planner {
public:
    Planner(LaunchPlan& p, std::uint32_t rows, const FusionPolicy& f, const PlanRequest* request = nullptr)
        : p_(p), rows_(rows), f_(f), request_(request), owner_(p.instances.size(), kNone) {
        if (request_) {
            stages_.emplace(p_, *request_);
            owner_.resize(p_.instances.size());
            job_.resize(p_.instances.size());
        }
    }

    void run() {
        for (std::uint32_t i = 0; i < p_.instances.size(); ++i) {
            if (f_.mode == Fusion::today) today(i);
            else fused(i);
            auto& writer = writer_[d(i).out];
            writer = {static_cast<int>(i), writer.current};
        }
        if (request_) finish_jobs();
    }

private:
    LaunchPlan& p_;
    std::uint32_t rows_;
    const FusionPolicy& f_;
    const PlanRequest* request_;
    std::vector<int> owner_;
    std::vector<int> head_src_;  // per launch: the writer of its head's first input when the launch opened
    struct Writer {
        int current = kNone;
        int previous = kNone;
    };
    std::map<std::string_view, Writer> writer_;
    std::optional<RequestStages> stages_;
    std::vector<std::uint8_t> job_;
    std::vector<std::array<JobChains, 8>> chains_;

    const StageEffects& effects(StageIndex i) const { return stages_->effects[i]; }
    const PlanStage& binding(StageIndex i) const { return *effects(i).binding; }
    std::string_view key(StageIndex i) const { return binding(i).key.empty() ? d(i).entry : binding(i).key; }

    const StageDecl& d(std::uint32_t i) const { return *p_.instances[i].decl; }
    int back() const { return static_cast<int>(p_.launches.size()) - 1; }
    int producer(std::string_view b) const {
        return launch_of(writer(b));
    }
    int writer(std::string_view b) const {
        const auto it = writer_.find(b);
        return it == writer_.end() ? kNone : it->second.current;
    }
    int launch_of(int stage) const { return stage == kNone ? kNone : owner_[static_cast<std::size_t>(stage)]; }
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
        if (request_) {
            auto& launch = p_.launches.back();
            launch.entry = key(i);
            launch.regime = {request_->phase, d(i).rows, request_->pos};
            chains_.emplace_back();
            add_job(i, back(), 0);
        }
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

    bool add_prologue(StageIndex i, Chain& chain) const {
        Sources seen{};
        seen.fill(kNone);
        std::size_t count = 0;
        for (const auto b : d(i).in) {
            if (b.empty() || producer(b) != kPending) continue;
            const int source = writer(b);
            if (std::find(seen.begin(), seen.end(), source) != seen.end()) continue;
            seen[count++] = source;
            if (!append(chain, effects(static_cast<StageIndex>(source)).form)) return false;
        }
        return append(chain, effects(i).input);
    }

    bool add_output(StageIndex i, JobChains& chain, Link form = {}) const {
        const auto& effect = effects(i);
        const bool side_effect = state_form(form.form) || form.form == Form::xch_publish;
        if (!side_effect && chain.stores.count != 0) return false;
        if (!append(chain.epilogue, form) || !append(chain.epilogue, effect.output)) return false;
        if (!side_effect) {
            chain.value = i;
            chain.stores = effect.stores;
        }
        if (effect.binding->head_emit.form != Form::none) chain.emit = effect.binding->head_emit;
        const auto links = static_cast<std::size_t>(chain.epilogue.count) + chain.stores.count;
        return links + (chain.emit.form != Form::none ? 1U : 0U) <= chain.epilogue.links.size();
    }

    void add_job(StageIndex i, int k, std::uint8_t job) {
        auto& launch = p_.launches[static_cast<std::size_t>(k)];
        auto& chain = chains_[static_cast<std::size_t>(k)][job];
        if (!add_prologue(i, chain.prologue) || !add_output(i, chain))
            throw std::invalid_argument("launch plan: standalone job exceeds chain capacity");
        launch.job_keys[job] = key(i);
        if (effects(i).form.form == Form::xch_publish) launch.publish_frames[job] = binding(i).publish_frames;
        job_[i] = job;
    }

    void finish_jobs() {
        for (std::size_t k = 0; k < p_.launches.size(); ++k) {
            auto& l = p_.launches[k];
            for (std::size_t j = 0; j < l.jobs; ++j) {
                auto& c = chains_[k][j];
                (void)append(c.epilogue, c.emit);
                for (std::size_t n = 0; n < c.stores.count; ++n) (void)append(c.epilogue, c.stores.links[n]);
                l.job_prologue_chain[j] = p_.chains.intern(c.prologue);
                l.epilogue_chain[j] = p_.chains.intern(c.epilogue);
            }
            if (std::all_of(l.job_prologue_chain.begin(), l.job_prologue_chain.begin() + l.jobs,
                            [&](ChainId id) { return id == l.job_prologue_chain[0]; }))
                l.prologue_chain = l.job_prologue_chain[0];

            const auto head = l.stages.front();
            const auto& b = binding(head);
            if (l.jobs == 1 && !chains_[k][0].exact_store && d(head).op == Op::linear &&
                !fixed_job(head) && request_->device && b.tiles != 0 &&
                b.tiles < request_->device->units && !b.split_k_key.empty())
                l.entry = l.job_keys[0] = b.split_k_key;  // rule 6, ordered split-K
        }
    }

    bool fixed_job(StageIndex i) const {
        return std::any_of(request_->packs.begin(), request_->packs.end(),
                           [&](const PlanPack& pack) { return listed(pack.jobs, key(i)); });
    }

    bool group_v2(StageIndex i, int k, bool same_input) {
        auto& l = p_.launches[static_cast<std::size_t>(k)];
        const auto head = l.stages.front();
        if (l.jobs == l.job_keys.size() || l.regime.rows != d(i).rows || fixed_job(head) || fixed_job(i) ||
            chains_[static_cast<std::size_t>(k)][0].exact_store)
            return false;
        const auto& b = binding(i);
        const auto wave_key = binding(head).wave_key;
        if (!same_input || l.pack == Pack::wave) {  // rule 6
            if (!request_->device || request_->device->units == 0 || wave_key.empty() ||
                wave_key != b.wave_key || b.tiles == 0) return false;
            std::uint64_t tiles = 0;
            for (const auto stage : l.stages) {
                const auto& member = binding(stage);
                if (member.wave_key != wave_key || member.tiles == 0) return false;
                tiles += member.tiles;
            }
            const auto target = static_cast<std::uint64_t>(request_->waves) * request_->device->units;
            if (tiles >= target) return false;
            l.pack = Pack::wave;
            l.entry = wave_key;
        }
        add_job(i, k, l.jobs++);
        return true;
    }

    bool attach_epilogue(StageIndex i, int k) {
        auto& l = p_.launches[static_cast<std::size_t>(k)];
        if (l.layer != p_.instances[i].layer) return false;
        const auto form = effects(i).form;
        if (form.form == Form::none) return false;
        for (const auto input : d(i).in)
            if (!input.empty() && producer(input) == kPending) return false;
        const auto source = stages_->inputs[i][0];
        if (source < 0 || owner_[static_cast<std::size_t>(source)] != k) return false;
        const auto source_job = job_[static_cast<std::size_t>(source)];
        auto& chain = chains_[static_cast<std::size_t>(k)][source_job];
        if (chain.value != static_cast<StageIndex>(source) || chain.exact_store) return false;
        const auto& tail = binding(i);
        const bool exact_store = tail.require_exact_epilogue;
        if (exact_store &&
            (tail.exact_epilogue_entry.empty() || rows_ != 1 || source_job != 0 || l.jobs != 1 ||
             l.pack != Pack::none || l.entry != tail.exact_epilogue_head || !l.epilogue.empty() ||
             !l.prologue.empty() || chain.prologue.count != 0 || chain.epilogue.count != 0 ||
             chain.emit.form != Form::none || fixed_job(l.stages.front()) ||
             (form.form != Form::store_bf16 && form.form != Form::softcap) ||
             std::any_of(std::begin(d(i).in) + 1, std::end(d(i).in), [](auto name) { return !name.empty(); })))
            return false;
        const bool side_effect = state_form(form.form) || form.form == Form::xch_publish;
        for (const int input : stages_->inputs[i]) {
            if (input < 0 || owner_[static_cast<std::size_t>(input)] != k ||
                (input != source && job_[static_cast<std::size_t>(input)] == source_job)) continue;
            if (!side_effect) {
                for (const auto user : stages_->users[static_cast<std::size_t>(input)]) {
                    const auto user_form = effects(user).form.form;
                    if (user != i && !(user < i && owner_[user] == k &&
                        (state_form(user_form) || user_form == Form::xch_publish))) return false;
                }
            }
            if (input == source) continue;
            const auto first = l.stages[source_job];
            const auto other = l.stages[job_[static_cast<std::size_t>(input)]];
            if (!binding(i).paired_tile || l.pack != Pack::none || d(first).op != Op::linear ||
                d(other).op != Op::linear || d(first).in[0] != d(other).in[0] ||
                stages_->inputs[first][0] != stages_->inputs[other][0] ||
                d(first).operand.rows != d(other).operand.rows ||
                chains_[static_cast<std::size_t>(k)][job_[static_cast<std::size_t>(input)]].value !=
                    static_cast<StageIndex>(input)) return false;
        }
        if (form.form == Form::xch_publish && l.publish_frames[source_job] != 0) return false;
        auto next = chain;
        if (!append(next.epilogue, effects(i).input) || !add_output(i, next, form)) return false;
        if (exact_store) {
            if (next.epilogue.count != 1 || next.stores.count != 0 || next.emit.form != Form::none) return false;
            next.exact_store = true;
            l.entry = l.job_keys[0] = tail.exact_epilogue_entry;
        }
        chain = next;
        if (form.form == Form::xch_publish) l.publish_frames[source_job] = binding(i).publish_frames;
        job_[i] = source_job;
        return true;
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
        if (request_) {
            if (b.empty()) return false;
            // A pending link belongs to one consumer's input chain. Do not nest virtual producers.
            for (const auto input : d(i).in)
                if (!input.empty() && producer(input) == kPending) return false;
        }
        bool any = false;
        bool clobbered = false;
        for (std::size_t j = i + 1U; j < p_.instances.size(); ++j) {
            const auto& c = *p_.instances[j].decl;
            if (reads(c, b)) {
                // A prologue re-reads the reduction's inputs at its consumer: refuse when one was rewritten in between.
                if (clobbered) return false;
                if (request_ && any && (d(i).op == Op::select || wait_form(effects(i).form.form))) return false;
                any = true;
                if (request_ && d(i).op == Op::select) {
                    if (c.op != Op::attend) return false;
                } else if (request_ && wait_form(effects(i).form.form)) {
                    if (!kernel_op(c.op) || wait_form(effects(static_cast<StageIndex>(j)).form.form)) return false;
                } else if (!(c.op == Op::linear || c.op == Op::head || seam_head(c))) {
                    return false;
                }
                if (request_ && p_.instances[i].layer != p_.instances[j].layer) return false;
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
        const auto w = writer_.find(b);
        if (w == writer_.end()) return true;
        const int p = launch_of(w->second.current);
        if (p == kPending) {
            for (const auto x : d(static_cast<StageIndex>(w->second.current)).in) {
                // An in-place reduction reads the value it replaced.
                if (x == b ? launch_of(w->second.previous) >= k : !available(x, k, false)) return false;
            }
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
        if (request_) {
            const auto form = effects(i).form.form;
            if ((wait_form(form) || (s.op == Op::select && binding(i).pool_keys != 0 &&
                                     binding(i).pool_keys <= 4096U)) && prologue_capable(i)) {
                owner_[i] = kPending;  // rules 8 and 15
                return;
            }
            if (wait_form(form)) return open(i);  // a join or wait is never an epilogue
            const int k = producer(s.in[0]);
            if (binding(i).sample && k == back() && k >= 0 &&
                d(p_.launches.back().stages.front()).op == Op::argmax &&
                p_.launches.back().stages.size() == 1 && p_.launches.back().layer == in.layer &&
                available_all(i, k, true) && out_free(i, k)) {
                join(i, k, &Launch::stages);  // rule 9: sampler runs in the pick launch
                return prologues(i);
            }
            if (state_form(form) || form == Form::xch_publish) {  // rules 7 and 15
                if (k >= 0 && kernel_op(d(p_.launches[static_cast<std::size_t>(k)].stages.front()).op) &&
                    available_all(i, k, true) && out_free(i, k) && attach_epilogue(i, k))
                    join(i, k, &Launch::epilogue);
                else open(i);
                return prologues(i);
            }
        }
        if (s.op == Op::clear) {  // rule 5
            if (back() >= 0 && kernel_op(d(p_.launches.back().stages.front()).op) &&
                (!request_ || (!chains_.back()[0].exact_store &&
                               p_.launches.back().layer == in.layer && available_all(i, back(), true) &&
                               out_free(i, back()))))
                return join(i, back(), &Launch::epilogue);
            return open(i);
        }
        if (linear_job(i)) {  // rules 1 and 6
            for (int k = back(); k >= 0 && p_.launches[static_cast<std::size_t>(k)].layer == in.layer; --k) {
                const auto h = p_.launches[static_cast<std::size_t>(k)].stages.front();
                if (request_ && !kernel_op(d(h).op)) break;
                if (!linear_job(h)) continue;
                const bool compatible = !request_ || (d(h).site == s.site &&
                    binding(h).operand_store == binding(i).operand_store && effects(h).input == effects(i).input);
                const bool same_input = d(h).op == Op::linear && s.op == Op::linear &&
                    d(h).in[0] == s.in[0] && p_.instances[h].format == in.format &&
                    head_src_[static_cast<std::size_t>(k)] == writer(s.in[0]) && compatible;
                if (!request_ && !same_input) continue;
                if (!available_all(i, k, false) || !out_free(i, k)) continue;
                if (request_ && !group_v2(i, k, same_input)) continue;
                join(i, k, &Launch::stages);
                return prologues(i);
            }
            open(i);
            return prologues(i);
        }
        const bool foldable = s.op == Op::elementwise || s.op == Op::store || s.op == Op::copy || (s.op == Op::combine && !seam_head(s));
        if (s.op == Op::norm || (foldable && (s.reduce || (split_inputs(i) && (!request_ || !binding(i).paired_tile))))) {
            // Rule 3
            if (prologue_capable(i)) {
                owner_[i] = kPending;
                return;
            }
            const int k = back();
            if (s.op == Op::norm && k >= 0 && d(p_.launches[static_cast<std::size_t>(k)].stages.front()).op == Op::norm &&
                p_.launches[static_cast<std::size_t>(k)].layer == in.layer && available_all(i, k, false) && out_free(i, k) &&
                (!request_ || (p_.launches.back().jobs < p_.launches.back().job_keys.size() &&
                               p_.launches.back().regime.rows == s.rows))) {
                if (request_) add_job(i, k, p_.launches.back().jobs++);
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
                available_all(i, last, true) && out_free(i, last) && (!request_ || attach_epilogue(i, last))) {
                join(i, last, &Launch::epilogue);
                return prologues(i);
            }
            open(i);
            return prologues(i);
        }
        if (s.op == Op::mix && back() >= 0 && seam_head(d(p_.launches.back().stages.front())) &&
            producer(s.in[0]) == back() && (!request_ || (available_all(i, back(), true) && out_free(i, back())))) {
            if (request_) {
                auto next = chains_.back()[0];
                if (!add_prologue(i, next.prologue) || !add_output(i, next)) {
                    open(i);
                    return prologues(i);
                }
                chains_.back()[0] = next;
            }
            join(i, back(), &Launch::stages);
            return prologues(i);
        }
        open(i);
        prologues(i);
    }

    bool linear_job(StageIndex i) const {
        return d(i).op == Op::linear || (request_ && d(i).op == Op::route && !binding(i).wave_key.empty());
    }
};

void merge_fixed_packs(LaunchPlan& plan, const PlanRequest& request) {
    if (request.packs.empty()) return;
    const auto inputs = sources(plan.instances);
    auto& launches = plan.launches;
    for (std::size_t first = 0; first < launches.size(); ++first) {
        for (const auto& pack : request.packs) {
            if (pack.jobs.size() > launches.size() - first || pack.jobs.front() != launches[first].entry) continue;
            auto group = std::span(launches).subspan(first, pack.jobs.size());
            const auto& base = group.front();
            bool legal = true;
            for (std::size_t j = 0; j < group.size() && legal; ++j) {
                const auto& job = group[j];
                legal = job.entry == pack.jobs[j] && job.layer == base.layer && job.regime.rows == base.regime.rows &&
                        job.stages.size() == 1 && job.jobs == 1;
                if (!legal || j == 0) continue;
                const auto stage = job.stages.front();
                const auto& prev = group[j - 1U];
                if (pack.pack == Pack::sum) {
                    const auto out = plan.instances[stage].decl->out;
                    legal = !out.empty() && out == plan.instances[prev.stages.front()].decl->out &&
                            !reads(*plan.instances[stage].decl, out) && !reads(*plan.instances[prev.stages.front()].decl, out);
                } else {
                    const auto source = static_cast<int>(prev.stages.front());
                    legal = std::find(inputs[stage].begin(), inputs[stage].end(), source) != inputs[stage].end();
                }
                // Prologues run before the pack. Only an explicitly piped job may read a job's output.
                for (std::size_t earlier = 0; earlier < j && legal; ++earlier) {
                    const auto& producer = group[earlier];
                    for (const auto p : job.prologue)
                        legal = legal && !reads(*plan.instances[p].decl, plan.instances[producer.stages.front()].decl->out);
                    if (pack.pack == Pack::sum) {
                        const auto source = static_cast<int>(producer.stages.front());
                        legal = legal && std::find(inputs[stage].begin(), inputs[stage].end(), source) == inputs[stage].end();
                    }
                    // The pack has no declarations for interleaving a member's epilogue with a later member.
                    for (const auto ep : producer.epilogue) {
                        const auto out = plan.instances[ep].decl->out;
                        legal = legal && !reads(*plan.instances[stage].decl, out);
                        for (const auto p : job.prologue) legal = legal && !reads(*plan.instances[p].decl, out);
                    }
                }
            }
            if (!legal) continue;
            if (pack.key.empty()) {
                if (request.device && request.device->vendor == "nvidia")
                    for (auto& job : group.subspan(1)) job.dependent = true;  // separate PDL launches
                break;
            }
            auto& fused = group.front();
            fused.entry = pack.key;
            fused.pack = pack.pack;
            for (const auto& job : group.subspan(1)) {
                const auto member = fused.jobs++;
                fused.stages.push_back(job.stages.front());
                for (const auto p : job.prologue)
                    if (!listed<StageIndex>(fused.prologue, p)) fused.prologue.push_back(p);
                fused.epilogue.insert(fused.epilogue.end(), job.epilogue.begin(), job.epilogue.end());
                fused.job_keys[member] = job.job_keys[0];
                fused.job_prologue_chain[member] = job.job_prologue_chain[0];
                fused.epilogue_chain[member] = job.epilogue_chain[0];
                fused.publish_frames[member] = job.publish_frames[0];
                if (fused.prologue_chain != job.prologue_chain) fused.prologue_chain = 0;
            }
            launches.erase(launches.begin() + static_cast<std::ptrdiff_t>(first + 1U),
                           launches.begin() + static_cast<std::ptrdiff_t>(first + pack.jobs.size()));
            break;
        }
    }
}

bool has_entry(const Launch& l, std::string_view key) {
    return l.entry == key || (key != "" &&
           std::find(l.job_keys.begin(), l.job_keys.begin() + l.jobs, key) != l.job_keys.begin() + l.jobs);
}

void insert_sidecars(LaunchPlan& plan, const PlanRequest& request) {  // rule 10
    if (request.sidecars.empty()) return;
    struct Attached {
        const PlanSidecar* spec;
        std::uint32_t parent;
    };
    const auto& original = plan.launches;
    std::vector<std::vector<Attached>> before(original.size() + 1U);
    for (std::uint32_t i = 0; i < original.size(); ++i) {
        const auto& parent = original[i];
        for (const auto& spec : request.sidecars) {
            if (!has_entry(parent, spec.parent_key)) continue;
            if (spec.during.empty()) {
                before[i + 1U].push_back({&spec, i});
                continue;
            }
            const auto trigger = std::find_if(original.begin(), original.end(), [&](const Launch& l) {
                if (l.layer != parent.layer) return false;
                const auto matches = [&](StageIndex i_stage) {
                    const auto& s = *plan.instances[i_stage].decl;
                    return s.id == spec.during && wait_form(stage_form(s, binding_for(request, s)).form);
                };
                return std::any_of(l.prologue.begin(), l.prologue.end(), matches) ||
                       std::any_of(l.stages.begin(), l.stages.end(), matches);
            });
            if (trigger == original.end()) continue;
            const auto at = static_cast<std::size_t>(trigger - original.begin());
            before[at].push_back({&spec, i < at ? i : no_launch});
        }
    }
    std::vector<Launch> result;
    std::vector<std::uint32_t> remap(original.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        for (const auto& [spec, parent] : before[i]) {
            const auto& owner = parent != no_launch ? result[remap[parent]] : original[i];
            Launch side;
            side.entry = side.job_keys[0] = spec->key;
            side.stream = spec->stream;
            side.during = spec->during;
            side.layer = owner.layer;
            side.rows = owner.rows;
            side.regime = owner.regime;
            side.sidecar = true;
            if (parent != no_launch) side.wait_for.push_back(remap[parent]);
            result.push_back(std::move(side));
        }
        if (i == original.size()) break;
        remap[i] = static_cast<std::uint32_t>(result.size());
        result.push_back(std::move(plan.launches[i]));
    }
    plan.launches = std::move(result);
}

using Predecessors = std::vector<std::vector<std::uint32_t>>;

Predecessors predecessors(const LaunchPlan& plan) {
    const auto in = sources(plan.instances);
    std::vector<std::uint32_t> placed(in.size(), no_launch);
    struct Access {
        std::uint32_t writer = no_launch;
        std::vector<std::uint32_t> readers;
    };
    std::map<std::string_view, Access> access;
    Predecessors result(plan.launches.size());
    for (std::uint32_t k = 0; k < plan.launches.size(); ++k) {
        const auto& l = plan.launches[k];
        auto& edges = result[k];
        edges = l.wait_for;
        const auto add = [&](std::uint32_t index) {
            if (index != no_launch && index != k) edges.push_back(index);
        };
        const auto stage = [&](StageIndex i, bool prologue) {
            const auto& s = *plan.instances.at(i).decl;
            for (std::size_t j = 0; j < in[i].size(); ++j) {
                const auto name = s.in[j];
                if (name.empty()) continue;
                auto& a = access[name];
                if (in[i][j] != kNone) {
                    const auto source = static_cast<std::size_t>(in[i][j]);
                    if (placed[source] == no_launch)
                        throw std::invalid_argument("launch plan: input scheduled after its consumer");
                    if (placed[source] == k && listed<StageIndex>(l.prologue, static_cast<StageIndex>(source)))
                        continue;  // launch-local recomputation has no materialized output
                    if (placed[source] != a.writer)
                        throw std::invalid_argument("launch plan: input overwritten before its consumer");
                    add(placed[source]);
                } else if (a.writer != no_launch) {
                    throw std::invalid_argument("launch plan: incoming value overwritten before its consumer");
                }
                add(a.writer);
                if (a.readers.empty() || a.readers.back() != k) a.readers.push_back(k);
            }
            if (!prologue && !s.out.empty()) {
                auto& a = access[s.out];
                add(a.writer);
                for (const auto reader : a.readers) add(reader);
                a.writer = k;
                a.readers.clear();
            }
            placed[i] = k;
        };
        for (const auto i : l.prologue) stage(i, true);
        for (const auto i : l.stages) stage(i, false);
        for (const auto i : l.epilogue) stage(i, false);
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }
    return result;
}

void assign_streams(LaunchPlan& plan, const PlanRequest& request) {
    const auto dependencies = predecessors(plan);
    auto& launches = plan.launches;
    for (std::uint32_t k = 1; k < launches.size(); ++k) {  // rule 16
        auto& branch = launches[k];
        const auto& main = launches[k - 1U];
        if (!request.device || main.stream != 0 || branch.sidecar || branch.dependent || main.layer != branch.layer ||
            !kernel_op(plan.instances[main.stages.front()].decl->op) ||
            !kernel_op(plan.instances[branch.stages.front()].decl->op) ||
            listed<std::uint32_t>(dependencies[k], k - 1U))
            continue;  // adjacent launches cannot have an indirect path without a direct edge
        const auto successor = std::find_if(launches.begin() + k + 1U, launches.end(),
                                            [](const Launch& l) { return !l.sidecar; });
        if (successor != launches.end() && successor->dependent) continue;
        const auto measured = std::find_if(request.overlap.begin(), request.overlap.end(), [&](const PlanOverlap& cost) {
            return main.entry == cost.main_key && branch.entry == cost.branch_key && cost.concurrent_us < cost.serial_us;
        });
        if (measured != request.overlap.end()) branch.stream = measured->stream;
    }
    std::array<std::uint32_t, 256> last{};
    last.fill(no_launch);
    for (std::uint32_t k = 0; k < launches.size(); ++k) {
        auto& l = launches[k];
        l.after = std::exchange(last[l.stream], k);
        l.wait_for.clear();
        for (const auto pred : dependencies[k])
            if (launches[pred].stream != l.stream) l.wait_for.push_back(pred);
    }
}

}  // namespace

LaunchPlan launch_plan(const StageGraph& g, std::uint32_t slot, std::uint32_t rows, const FusionPolicy& policy) {
    if (rows == 0U) throw std::invalid_argument("launch plan: rows must be >= 1");
    LaunchPlan p;
    p.instances = instances(g, slot);
    Planner(p, rows, policy).run();
    return p;
}

LaunchPlan launch_plan(const StageGraph& g, const PlanRequest& request) {
    if (request.rows == 0U || request.waves < 2U)
        throw std::invalid_argument("launch plan: rows must be positive and wave packing requires at least two waves");
    for (const auto& b : request.stages) {
        const auto valid_form = [](Link link) { return link.form == Form::none || !form_name(link.form).empty(); };
        if (!b.stage || !valid_form(b.form) || (b.input_transform.form != Form::none &&
            !transform_form(b.input_transform.form)) || (b.operand_store != Form::none && !store_form(b.operand_store)) ||
            (b.head_emit.form != Form::none && (b.stage->op != Op::head || !emit_form(b.head_emit.form))) ||
            (b.require_exact_epilogue && ((b.stage->op != Op::elementwise && b.stage->op != Op::store) ||
             b.stage->reduce || b.exact_epilogue_head.empty() != b.exact_epilogue_entry.empty())) ||
            b.publish_frames == 0 || b.publish_frames > 2)
            throw std::invalid_argument("launch plan: invalid stage binding");
    }
    for (const auto& pack : request.packs)
        if (pack.jobs.size() < 2U || pack.jobs.size() > Chain{}.links.size() ||
            (pack.pack != Pack::sum && pack.pack != Pack::pipe) || (pack.pack == Pack::sum && pack.key.empty()) ||
            std::any_of(pack.jobs.begin(), pack.jobs.end(), [](std::string_view key) { return key.empty(); }))
            throw std::invalid_argument("launch plan: invalid fixed pack binding");
    for (const auto& sidecar : request.sidecars)
        if (sidecar.parent_key.empty() || sidecar.key.empty() || sidecar.stream == 0)
            throw std::invalid_argument("launch plan: invalid sidecar binding");
    for (const auto& cost : request.overlap)
        if (cost.main_key.empty() || cost.branch_key.empty() || cost.stream == 0 || !std::isfinite(cost.serial_us) ||
            !std::isfinite(cost.concurrent_us) || cost.serial_us <= 0 || cost.concurrent_us <= 0)
            throw std::invalid_argument("launch plan: invalid concurrency measurement");

    std::uint32_t slot = 0;
    if (request.slot) slot = *request.slot;
    else if (request.pos == Pos::prefix_select) slot = g.select_after;
    else if (request.pos == Pos::group_close && g.group != 0) slot = g.group - 1U;
    const auto at_close = g.group != 0 && slot % g.group == g.group - 1U;
    const auto prefix = g.select_after == 0 ? Pos::any :
                        (slot < g.select_after ? Pos::prefix_dense : Pos::prefix_select);
    if (request.pos != Pos::any && request.pos != prefix && !(request.pos == Pos::group_close && at_close))
        throw std::invalid_argument("launch plan: position and slot disagree");
    auto resolved = request;
    if (resolved.pos == Pos::any) resolved.pos = at_close ? Pos::group_close : prefix;

    LaunchPlan plan;
    plan.instances = instances(g, slot);
    constexpr FusionPolicy fused{.mode = Fusion::fused};
    Planner(plan, request.rows, fused, &resolved).run();
    merge_fixed_packs(plan, resolved);
    insert_sidecars(plan, resolved);
    assign_streams(plan, resolved);
    validate_plan(plan);
    return plan;
}

void validate_plan(const LaunchPlan& plan) {
    std::vector<bool> covered(plan.instances.size());
    std::vector<bool> materialized(plan.instances.size());
    const auto inputs = sources(plan.instances);
    for (std::size_t i = 0; i < plan.launches.size(); ++i) {
        const auto& l = plan.launches[i];
        if (l.jobs == 0 || l.jobs > l.job_keys.size() || l.rows == 0 || l.kernels == 0 ||
            (l.sidecar ? (l.stream == 0 || l.entry.empty() || !l.stages.empty()) : l.stages.size() < l.jobs) ||
            (l.pack != Pack::none && l.jobs < 2) || (l.dependent && l.after == no_launch))
            throw std::invalid_argument("launch plan: invalid launch geometry");
        if (l.after != no_launch && (l.after >= i || plan.launches[l.after].stream != l.stream))
            throw std::invalid_argument("launch plan: invalid stream predecessor");
        for (const auto wait : l.wait_for)
            if (wait >= i || plan.launches[wait].stream == l.stream)
                throw std::invalid_argument("launch plan: invalid cross-stream predecessor");
        for (std::size_t j = 0; j < l.jobs; ++j) {
            (void)plan.chains.at(l.job_prologue_chain[j]);
            (void)plan.chains.at(l.epilogue_chain[j]);
            if (l.prologue_chain != 0 && l.prologue_chain != l.job_prologue_chain[j])
                throw std::invalid_argument("launch plan: inconsistent shared prologue");
            if (l.jobs > 1 && l.pack != Pack::pipe) {
                const auto group = std::span(l.stages).first(l.jobs);
                for (const auto source : inputs.at(l.stages[j]))
                    if (source != kNone && listed<StageIndex>(group, static_cast<StageIndex>(source)))
                        throw std::invalid_argument("launch plan: grouped job reads another job's output");
            }
        }
        for (const auto stage : l.prologue) covered.at(stage) = true;
        const auto write = [&](StageIndex stage) {
            if (materialized.at(stage)) throw std::invalid_argument("launch plan: repeated materialized stage");
            materialized[stage] = true;
            covered[stage] = true;
        };
        for (const auto stage : l.stages) write(stage);
        for (const auto stage : l.epilogue) write(stage);
    }
    if (std::find(covered.begin(), covered.end(), false) != covered.end())
        throw std::invalid_argument("launch plan: missing stage");
}

std::uint64_t plan_identity(const LaunchPlan& plan) {
    // Stable description digest. Fields are length-delimited and encoded little-endian.
    std::uint64_t hash = 14695981039346656037ULL;
    const auto byte = [&](std::uint8_t b) { hash = (hash ^ b) * 1099511628211ULL; };
    const auto number = [&](std::uint64_t n) {
        for (unsigned i = 0; i < 8; ++i) { byte(static_cast<std::uint8_t>(n)); n >>= 8U; }
    };
    const auto name = [&](std::string_view value) {
        number(value.size());
        for (const char c : value) byte(static_cast<std::uint8_t>(c));
    };
    const auto stages = [&](std::span<const StageIndex> indices) {
        number(indices.size());
        for (const auto index : indices) {
            const auto& stage = plan.instances.at(index);
            name(stage.decl->id);
            name(stage.decl->form);
            name(stage.decl->numeric);
            byte(static_cast<std::uint8_t>(stage.decl->op));
            for (const auto in : stage.decl->in) name(in);
            name(stage.decl->out);
            name(stage.decl->operand.tensor);
            number(static_cast<std::uint32_t>(stage.layer));
            byte(static_cast<std::uint8_t>(stage.format));
        }
    };
    number(plan.launches.size());
    for (const auto& l : plan.launches) {
        name(l.entry);
        number(static_cast<std::uint32_t>(l.layer));
        number(l.rows);
        number(l.kernels);
        byte(static_cast<std::uint8_t>(l.regime.phase));
        byte(static_cast<std::uint8_t>(l.regime.rows));
        byte(static_cast<std::uint8_t>(l.regime.pos));
        byte(static_cast<std::uint8_t>(l.pack));
        byte(l.jobs);
        for (std::size_t j = 0; j < l.jobs; ++j) {
            name(l.job_keys[j]);
            number(plan.chains.identity(l.job_prologue_chain[j]));
            number(plan.chains.identity(l.epilogue_chain[j]));
            byte(l.publish_frames[j]);
        }
        stages(l.stages);
        stages(l.prologue);
        stages(l.epilogue);
        byte(l.stream);
        number(l.after);
        number(l.wait_for.size());
        for (const auto wait : l.wait_for) number(wait);
        name(l.during);
        byte(l.sidecar);
        byte(l.dependent);
    }
    return hash;
}

}  // namespace strix::graph
