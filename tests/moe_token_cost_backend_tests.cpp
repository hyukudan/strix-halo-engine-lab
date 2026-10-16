#include "strix/moe_token_cost_backend.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
template<class Work> void rejects(Work&& work) {
    bool caught = false;
    try { work(); } catch (const std::exception&) { caught = true; }
    require(caught,"invalid cost-policy call accepted");
}
strix::MoeTokenCostContext context() {
    strix::MoeTokenCostContext c;
    c.model_content.bytes[0] = 1;
    c.execution_identity = 2; c.kernel_plan_identity = 3;
    c.runtime_compatibility_identity = 4; c.resolved_profile_identity = 5;
    c.context_capacity = 128; c.vocab_size = 4; c.eos_token_id = 3;
    return c;
}
struct Backend final : strix::SerialGreedyStepBackend {
    strix::MoeTokenCostContext& c;
    std::uint64_t full = 0, greedy = 0, state = 0, captures = 0, restores = 0;
    std::uint32_t selected = 1;
    bool fail = false, cancel_after = false, commit = true;
    explicit Backend(strix::MoeTokenCostContext& actual) : c(actual) {}
    void consume() {
        if (fail) throw std::runtime_error("injected target failure");
        if (commit) { ++c.position; ++c.committed_tokens; }
        c.cancelled = cancel_after;
    }
    std::vector<float> execute(std::uint32_t, std::uint64_t) override {
        ++full; consume(); return {0.0F,2.0F,-1.0F,0.0F};
    }
    std::uint32_t execute_greedy(std::uint32_t, std::uint64_t, std::uint64_t) override {
        ++greedy; consume(); return selected;
    }
    void execute_state(std::uint32_t, std::uint64_t) override { ++state; consume(); }
    std::optional<strix::GreedyPrefixRestore> try_restore_greedy_prefix(const strix::SerialGreedyRequest&) override {
        ++restores; return strix::GreedyPrefixRestore{2,1};
    }
    void capture_greedy_prefix(std::span<const std::uint32_t>, std::uint32_t, std::uint64_t) override { ++captures; }
};
void preserve_intents_and_bootstrap() {
    auto c = context(); Backend inner(c);
    std::vector<strix::MoeTokenSchedule> modes;
    strix::MoeTokenCostBackend cost(inner,[&] { return c; },[&](auto mode) { modes.push_back(mode); });
    for (unsigned i = 0; i < 4U; ++i)
        require(cost.execute_greedy(0,c.position,4) == 1U,"selection changed by scheduling");
    require(modes == std::vector<strix::MoeTokenSchedule>{strix::MoeTokenSchedule::serial_expanded,
        strix::MoeTokenSchedule::serial_expanded,strix::MoeTokenSchedule::fork_join,strix::MoeTokenSchedule::fork_join},
        "both arms require their own complete bootstrap observations");
    require(inner.greedy == 4U && inner.full == 0U && inner.state == 0U,"wrapper executed extra target work");
    const auto logits = cost.execute(0,c.position);
    require(logits == std::vector<float>{0,2,-1,0} && modes.back() == strix::MoeTokenSchedule::serial_expanded,
        "full output must remain full with an independent intent controller");
    cost.execute_state(0,c.position);
    require(modes.back() == strix::MoeTokenSchedule::serial_expanded && inner.state == 1U,
        "state-only must not request logits or borrow greedy observations");
    const auto stats = cost.stats();
    require(stats.arms[0].selected == 4U && stats.arms[1].selected == 2U &&
        stats.arms[0].observed == 4U && stats.arms[1].observed == 2U && stats.key_resets == 3U,
        "observations/intent keys do not describe actual completed calls");
    const std::array<std::uint32_t,2> prefix{0,1};
    const strix::SerialGreedyRequest request{prefix,4,128,1,3};
    const auto restored = cost.try_restore_greedy_prefix(request);
    cost.capture_greedy_prefix(prefix,1,4);
    require(restored && restored->consumed_prefix == 2U && restored->pending_token == 1U &&
        inner.restores == 1U && inner.captures == 1U && cost.stats().arms[0].selected == 4U,
        "cache hooks must be forwarded without invented model samples");
}
void censored_and_new_keys() {
    auto c = context(); Backend inner(c);
    strix::ObservedMoeTokenCostPolicy policy; policy.bootstrap_samples_per_arm = 1;
    strix::MoeTokenCostBackend cost(inner,[&] { return c; },[](auto) {},policy);
    inner.selected = 3; require(cost.execute_greedy(0,c.position,4) == 3U,"EOS output changed");
    inner.selected = 1; inner.cancel_after = true;
    (void)cost.execute_greedy(0,c.position,4);
    c.cancelled = false; inner.cancel_after = false; inner.commit = false;
    (void)cost.execute_greedy(0,c.position,4);
    inner.commit = true; inner.fail = true;
    rejects([&] { (void)cost.execute_greedy(0,c.position,4); });
    inner.fail = false;
    require(cost.stats().arms[0].censored == 4U && cost.stats().arms[0].observed == 0U,
        "EOS, cancellation, no commit and failure must not train the cost controller");
    // EOS as INPUT is ordinary teacher/prefix content, not a censor condition.
    (void)cost.execute_greedy(3,c.position,4);
    require(cost.stats().arms[0].observed == 1U,"input EOS incorrectly censored");
    ++c.resolved_profile_identity;
    (void)cost.execute_greedy(0,c.position,4);
    require(cost.stats().last_selected == strix::MoeTokenSchedule::serial_expanded && cost.stats().key_resets == 2U,
        "changed resolved profile must restart bootstrap rather than reuse costs");
    c.position = c.context_capacity - 1U;
    (void)cost.execute_greedy(0,c.position,4);
    require(cost.stats().arms[1].censored == 1U,"capacity-ending target entered steady cost totals");
}
void zero_dispatch_admission_and_observer_failure() {
    auto c = context(); Backend inner(c); unsigned sink_calls = 0;
    bool fail_after = false;
    strix::MoeTokenCostBackend cost(inner,[&] {
        if (fail_after && c.position != 0U) throw std::runtime_error("cost observer failed after real commit");
        return c;
    },[&](auto) { ++sink_calls; });
    rejects([&] { (void)cost.execute_greedy(4,0,4); });
    rejects([&] { (void)cost.execute_greedy(0,1,4); });
    rejects([&] { (void)cost.execute_greedy(0,0,5); });
    c.ready = false; rejects([&] { cost.execute_state(0,0); }); c.ready = true;
    require(sink_calls == 0U && inner.greedy == 0U && inner.state == 0U,"bad context dispatched a target");
    fail_after = true;
    require(cost.execute_greedy(0,0,4) == 1U && c.committed_tokens == 1U &&
        cost.stats().arms[0].censored == 1U,"post-commit telemetry error changed successful inference");
    auto d = context(); Backend second(d);
    strix::MoeTokenCostBackend bad_sink(second,[&] { return d; },[](auto) { throw std::runtime_error("sink failed"); });
    rejects([&] { bad_sink.execute_state(0,0); });
    require(second.state == 0U && d.position == 0U,"failed mode selection consumed target");
    auto invalid_policy = strix::ObservedMoeTokenCostPolicy{}; invalid_policy.context_bucket_tokens = 0;
    rejects([&] { strix::MoeTokenCostBackend invalid(second,[&] { return d; },[](auto) {},invalid_policy); });
}
void bucket_crossing() {
    auto c = context(); Backend inner(c);
    strix::ObservedMoeTokenCostPolicy policy; policy.context_bucket_tokens = 2;
    strix::MoeTokenCostBackend cost(inner,[&] { return c; },[](auto) {},policy);
    cost.execute_state(0,0); cost.execute_state(0,1); cost.execute_state(0,2);
    require(cost.stats().key_resets == 2U && cost.stats().arms[0].observed == 2U &&
        cost.stats().arms[0].censored == 1U,"bucket crossing must be censored and next bucket reset");
}
}
int main() try {
    preserve_intents_and_bootstrap(); censored_and_new_keys();
    zero_dispatch_admission_and_observer_failure(); bucket_crossing();
    std::cout << "MoE complete-token cost source fixtures completed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
