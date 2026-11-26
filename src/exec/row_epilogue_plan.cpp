#include "row_epilogue_plan.hpp"

#include "strix/engine_profile.hpp"

#include <map>
#include <stdexcept>
#include <utility>

namespace strix::exec {
namespace {

using namespace graph;

struct Use {
    std::uint32_t reader = no_launch;
    bool shared = false;
};

// Track versions by declaration order, so a read/write of the same named
// value consumes the previous writer. A second input port also counts as a use.
std::vector<Use> uses(std::span<const Instance> stages) {
    std::map<std::string_view, std::uint32_t> writers;
    std::vector<Use> result(stages.size());
    for (std::uint32_t i = 0; i < stages.size(); ++i) {
        const auto& stage = *stages[i].decl;
        for (const auto input : stage.in) {
            const auto writer = writers.find(input);
            if (writer == writers.end()) continue;
            auto& use = result[writer->second];
            if (use.reader != no_launch) use.shared = true;
            use.reader = i;
        }
        if (!stage.out.empty()) writers[stage.out] = i;
    }
    return result;
}

bool single_job(const Launch& launch) {
    return launch.kernels == 1 && launch.stages.size() == 1 && launch.epilogue.empty();
}

bool matches(const StageDecl& stage, ChainStageContract contract) {
    return !contract.entry.empty() && stage.entry == contract.entry && stage.numeric == contract.numeric;
}

bool row_tail(Form form) {
    switch (form) {
    case Form::store_f32:
    case Form::store_bf16:
    case Form::store_f16:
    case Form::residual_add:
    case Form::residual_add_round_bf16:
    case Form::scale_const:
    case Form::scale_layer:
    case Form::bias_add:
    case Form::softcap:
    case Form::quant_a8g32:
        return true;
    default:
        return false;
    }
}

// Derive the ordered tail from graph edges before consulting numerical
// capabilities. Adjacent incumbent leaves keep every auxiliary input ready.
std::size_t candidate_tails(const LaunchPlan& plan, std::span<const Use> readers,
                            std::span<const Launch> launches, const StageDecl& head) {
    auto source = launches.front().stages.front();
    std::size_t count = 0;
    for (const auto& tail : launches.subspan(1)) {
        if (count == Chain{}.links.size() || !single_job(tail) || tail.layer != launches.front().layer) break;
        const auto index = tail.stages.front();
        const auto& stage = *plan.instances[index].decl;
        if (index != source + 1U || readers[source].shared || readers[source].reader != index ||
            stage.in[0].empty() || stage.in[0] != plan.instances[source].decl->out ||
            (stage.op != Op::elementwise && stage.op != Op::combine) || stage.reduce || stage.leaves != 0 ||
            stage.site != head.site || stage.rows != head.rows || stage.split != head.split ||
            stage.pos != head.pos || stage.out.empty() || !row_tail(parse_form(stage.form).value_or(Form::none))) break;
        source = index;
        ++count;
    }
    return count;
}

bool matches(const RowEpilogueVariant& variant, const LaunchPlan& plan, std::span<const Launch> launches) {
    const auto& instance = plan.instances[launches.front().stages.front()];
    const auto& head = *instance.decl;
    if (variant.rows != launches.front().rows || variant.head_format != instance.format ||
        variant.output_width != head.operand.rows || variant.input_width != head.operand.cols ||
        variant.key.op != head.op || !matches(head, variant.head)) return false;
    for (std::size_t i = 0; i < variant.steps.size(); ++i) {
        const auto& step = variant.steps[i];
        const auto& tail = *plan.instances[launches[i + 1U].stages.front()].decl;
        if (!matches(tail, step.incumbent) || tail.form != form_name(step.link.form) || step.link.arg >= 4)
            return false;
    }
    return true;
}

struct Choice {
    const RowEpilogueVariant* variant = nullptr;
    const KernelEntry* entry = nullptr;
    std::size_t count = 0;
};

Choice choose(const Catalog& catalog, const RowEpilogueRequest& request, const LaunchPlan& plan,
              std::span<const Use> readers, std::span<const Launch> launches) {
    const auto& launch = launches.front();
    if (launch.layer < 0 || !single_job(launch)) return {};
    const auto& head = *plan.instances[launch.stages.front()].decl;
    if ((head.op != Op::linear && head.op != Op::combine) || head.rows != Rows::per_row ||
        head.split != Split::none || head.leaves != 0 || head.reduce ||
        head.operand.items != 1 || head.operand.read != 1) return {};
    const auto tails = candidate_tails(plan, readers, launches, head);
    const bool allow_fast = request.tier == profile::Name::fast || request.tier == profile::Name::turbo;
    Choice best;
    for (const auto& variant : request.variants) {
        const auto count = variant.steps.size();
        if (count == 0 || count > tails || count < best.count || variant.arch != request.arch ||
            !matches(variant, plan, launches)) continue;
        const auto found = catalog.find(variant.key, request.arch);
        if (found.size() != 1 || found.front()->launcher->max_rows < request.rows ||
            found.front()->pack != exec::Pack::none) continue;
        const auto* entry = found.front();
        const bool exact = entry->exact == lever::Exactness::bitwise;
        if (exact ? variant.exact_fixture.empty() : (!allow_fast || !variant.exact_fixture.empty())) continue;
        if (count == best.count && (!exact || best.entry->exact == lever::Exactness::bitwise)) continue;
        best = {&variant, entry, count};
    }
    return best;
}

}  // namespace

RowEpiloguePlan plan_row_epilogues(const graph::StageGraph& graph, const Catalog& catalog,
                                  const RowEpilogueRequest& request) {
    if (request.incumbent.mode != Fusion::today) throw std::invalid_argument("row epilogue requires check mode");
    RowEpiloguePlan result{launch_plan(graph, request.slot, request.rows, request.incumbent), {}};
    auto incumbent = std::move(result.graph.launches);
    result.graph.launches.clear();
    result.graph.launches.reserve(incumbent.size());
    result.selections.reserve(incumbent.size());
    const auto readers = uses(result.graph.instances);
    for (std::size_t i = 0; i < incumbent.size();) {
        const auto remaining = std::span(incumbent).subspan(i);
        const auto choice = choose(catalog, request, result.graph, readers, remaining);
        auto launch = std::move(incumbent[i]);
        RowEpilogueSelection selection;
        if (choice.entry) {
            Chain chain{};
            for (std::size_t n = 0; n < choice.count; ++n) {
                chain.links[chain.count++] = choice.variant->steps[n].link;
                launch.epilogue.push_back(remaining[n + 1U].stages.front());
            }
            launch.entry = launch.job_keys[0] = choice.entry->identity;
            launch.epilogue_chain[0] = result.graph.chains.intern(chain);
            selection = {choice.entry, choice.variant->exact_fixture};
        }
        result.graph.launches.push_back(std::move(launch));
        result.selections.push_back(selection);
        i += 1U + choice.count;
    }
    validate_plan(result.graph);
    return result;
}

}  // namespace strix::exec
