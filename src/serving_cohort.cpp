#include "strix/serving_cohort.hpp"
#include <stdexcept>
namespace strix {
void validate_serving_cohort_targets(std::span<const ServingCohortTarget> targets) {
    if (targets.size() != 2U && targets.size() != 4U)
        throw std::invalid_argument("serving tensor cohort requires two or four independent targets");
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        if (!target.backend || !target.input.ticket || !target.input.vocab_size ||
            target.input.token >= target.input.vocab_size || target.selected)
            throw std::invalid_argument("serving cohort target is incomplete or already delivered");
        for (std::size_t j = 0; j < i; ++j)
            if (targets[j].backend == target.backend)
                throw std::invalid_argument("serving cohort repeats a backend session");
    }
}
void execute_serving_target_fallback(ServingCohortTarget& target) {
    if (!target.backend || !target.input.ticket || !target.input.vocab_size || target.selected)
        throw std::invalid_argument("serving C1 fallback target is invalid");
    if (target.input.state_only) target.backend->execute_state(target.input.token, target.input.position);
    else target.selected = target.backend->execute_greedy(target.input.token, target.input.position, target.input.vocab_size);
}
} // namespace strix
