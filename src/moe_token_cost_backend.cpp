#include "strix/moe_token_cost_backend.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
using Clock = std::chrono::steady_clock;
ObservedSpeculationPolicy controller_policy(const ObservedMoeTokenCostPolicy& p) {
    if (!p.context_bucket_tokens) throw std::invalid_argument("MoE cost context bucket must be positive");
    return {p.bootstrap_samples_per_arm, p.reexplore_interval, p.switch_hysteresis_fraction};
}
bool same_key(const MoeTokenCostContext& a, const MoeTokenCostContext& b) noexcept {
    return a.model_content == b.model_content && a.execution_identity == b.execution_identity &&
        a.kernel_plan_identity == b.kernel_plan_identity &&
        a.runtime_compatibility_identity == b.runtime_compatibility_identity &&
        a.resolved_profile_identity == b.resolved_profile_identity &&
        a.context_capacity == b.context_capacity && a.vocab_size == b.vocab_size && a.eos_token_id == b.eos_token_id;
}
void increment(std::uint64_t& value, bool& saturated) noexcept {
    if (value == std::numeric_limits<std::uint64_t>::max()) saturated = true;
    else ++value;
}
}
MoeTokenCostBackend::MoeTokenCostBackend(SerialGreedyStepBackend& inner, ContextProvider provider,
    ModeSink sink, ObservedMoeTokenCostPolicy policy)
    : inner_(inner), context_(std::move(provider)), sink_(std::move(sink)), policy_(policy),
      intents_{IntentState(controller_policy(policy)), IntentState(controller_policy(policy)),
               IntentState(controller_policy(policy))} {
    if (!context_ || !sink_) throw std::invalid_argument("MoE cost policy requires actual context and mode sink");
}
MoeTokenCostBackend::Observation MoeTokenCostBackend::begin(std::size_t intent, std::uint32_t token,
    std::uint64_t position, std::uint64_t vocab) {
    if (active_) throw std::logic_error("MoE cost target already active");
    const auto c = context_();
    if (!c.ready || !c.model_content.valid() || !c.execution_identity || !c.kernel_plan_identity ||
        !c.runtime_compatibility_identity || !c.resolved_profile_identity || !c.vocab_size ||
        c.eos_token_id >= c.vocab_size || c.position != position || position >= c.context_capacity ||
        c.committed_tokens == std::numeric_limits<std::uint64_t>::max() || token >= c.vocab_size ||
        (vocab && vocab != c.vocab_size))
        throw std::invalid_argument("MoE cost target context/input differs from live session");
    auto& state = intents_[intent];
    const auto bucket = position / policy_.context_bucket_tokens;
    if (!state.initialized || !same_key(c,state.key) || bucket != state.bucket) {
        state.controller.reset(); state.key = c; state.bucket = bucket; state.initialized = true;
        increment(stats_.key_resets,stats_.saturated);
    }
    const auto arm = state.controller.select(1);
    const auto mode = arm == 0U ? MoeTokenSchedule::serial_expanded : MoeTokenSchedule::fork_join;
    try { sink_(mode); } catch (...) { state.controller.cancel_observation(); throw; }
    increment(stats_.arms[arm].selected,stats_.saturated);
    if (have_selection_ && stats_.last_selected != mode) increment(stats_.selection_changes,stats_.saturated);
    have_selection_ = true; stats_.last_selected = mode; active_ = true;
    return {c,intent,arm};
}
void MoeTokenCostBackend::censor(const Observation& observation) noexcept {
    intents_[observation.intent].controller.cancel_observation();
    increment(stats_.arms[observation.arm].censored,stats_.saturated); active_ = false;
}
void MoeTokenCostBackend::finish(const Observation& observation, double seconds,
    std::optional<std::uint32_t> selected) noexcept {
    // Observation must not throw after successful model publication/delivery.
    try {
        const auto after = context_();
        const auto& before = observation.before;
        if (!after.ready || before.cancelled || after.cancelled || !same_key(before,after) ||
            after.position != before.position + 1U || after.committed_tokens != before.committed_tokens + 1U ||
            after.position >= after.context_capacity ||
            after.position / policy_.context_bucket_tokens != before.position / policy_.context_bucket_tokens ||
            (selected && (*selected >= before.vocab_size || *selected == before.eos_token_id)) ||
            !intents_[observation.intent].controller.observe(observation.arm,seconds,1U)) {
            censor(observation); return;
        }
        auto& arm = stats_.arms[observation.arm]; increment(arm.observed,stats_.saturated);
        const double sum = arm.complete_wall_seconds + seconds;
        if (!std::isfinite(sum)) { arm.complete_wall_seconds = std::numeric_limits<double>::max(); stats_.saturated = true; }
        else arm.complete_wall_seconds = sum;
        active_ = false;
    } catch (...) { censor(observation); }
}
std::vector<float> MoeTokenCostBackend::execute(std::uint32_t token, std::uint64_t position) {
    const auto start = Clock::now(); const auto observation = begin(0,token,position,0);
    try {
        auto logits = inner_.execute(token,position);
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        finish(observation,seconds,std::nullopt); return logits;
    } catch (...) { censor(observation); throw; }
}
std::uint32_t MoeTokenCostBackend::execute_greedy(std::uint32_t token, std::uint64_t position, std::uint64_t vocab) {
    const auto start = Clock::now(); const auto observation = begin(1,token,position,vocab);
    try {
        const auto selected = inner_.execute_greedy(token,position,vocab);
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        finish(observation,seconds,selected); return selected;
    } catch (...) { censor(observation); throw; }
}
void MoeTokenCostBackend::execute_state(std::uint32_t token, std::uint64_t position) {
    const auto start = Clock::now(); const auto observation = begin(2,token,position,0);
    try {
        inner_.execute_state(token,position);
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        finish(observation,seconds,std::nullopt);
    } catch (...) { censor(observation); throw; }
}
std::optional<GreedyPrefixRestore> MoeTokenCostBackend::try_restore_greedy_prefix(const SerialGreedyRequest& request) {
    if (active_) throw std::logic_error("prefix restore during MoE cost target");
    return inner_.try_restore_greedy_prefix(request);
}
void MoeTokenCostBackend::capture_greedy_prefix(std::span<const std::uint32_t> prefix,
    std::uint32_t pending, std::uint64_t vocab) {
    if (active_) throw std::logic_error("prefix capture during MoE cost target");
    inner_.capture_greedy_prefix(prefix,pending,vocab);
}
} // namespace strix
