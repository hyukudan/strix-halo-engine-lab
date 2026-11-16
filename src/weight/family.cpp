#include <strix/weight/family.hpp>
#include <sstream>
namespace strix::weight {
Status resolve_family_weights(const FamilyPlanInputs& in, FamilyPlans& out) {
    if (!in.registry || in.arithmetic.size()!=in.weights.size() || in.incumbents.size()!=in.weights.size())
        return {Error::malformed,"one declared incumbent and arithmetic required per admitted tensor"};
    FamilyPlans result; result.weights=in.weights;
    auto valid=validate_map(*in.registry,result.weights); if (!valid) return valid;
    std::ostringstream text; Status overall{Error::ok,{}};
    for (std::size_t i=0; i<result.weights.size(); ++i) {
        ResolveRequest r{}; r.tensor=&result.weights[i]; r.hardware=in.hardware; r.workload=in.workload;
        r.required_arithmetic=in.arithmetic[i]; r.incumbent=in.incumbents[i]; r.maximum_permission=in.permission; r.allow_fast=in.allow_fast;
        apply_decision_override(r,in.measured_decisions_override);
        Plan plan{}; auto status=resolve(*in.registry,r,in.candidates,in.qualifications,plan);
        result.plans.push_back(plan); result.status.push_back(status);
        if (status) text << effective_config(WeightMap{result.weights[i]},std::span(&result.plans.back(),1));
        else { text << "weight tensor=" << result.weights[i].binding.tensor << " unsupported=" << static_cast<unsigned>(status.code) << " reason=" << status.detail << '\n'; if (overall) overall=status; }
    }
    result.description=text.str(); out=std::move(result); return overall;
}
}
