#include "strix/mtp_cohort_partition.hpp"
#include "strix/serve/step_planner.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace serve = strix::serve;
using strix::mtp_cohort_partition;
static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
namespace strix {
enum class EnvFlagMatch { exact };
namespace envcfg {
int value = -1;
bool flag(const char*, bool fallback, EnvFlagMatch) { return value < 0 ? fallback : value != 0; }
}
}
namespace strix::serve { struct ConstraintError : std::runtime_error { using runtime_error::runtime_error; }; }
struct KnownPrefillCancelled {};
struct Profiles { int prefill = 0, decode = 0; };
struct Options { bool baseline = false; Profiles profiles; };
[[maybe_unused]] static Profiles numerical_profile_pair(const Options& o) { return o.profiles; }
struct ForkAdapter {
    Options options_;
    explicit ForkAdapter(Options o) : options_(o) {}
#include "fork.inc"
};

struct Controller { std::size_t members = 0; };
struct HipMtpGreedyCohortMember {
    std::uint32_t maximum_candidates = 0, slot = 0;
    bool transaction = false;
};
struct HipMtpGreedyCycleResult { std::uint32_t slot = 0, drafts = 0, emitted = 1; };
struct CohortStats {};
// Boundary stand-in reproduces the two audited joint constraints. Ordinary
// execution honors the cap; policy controllers have a hard eight-member limit.
static std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(
    std::span<const HipMtpGreedyCohortMember> members, CohortStats*, Controller* joint, Controller* policy) {
    const auto policies = std::count_if(members.begin(), members.end(), [](const auto& c) { return c.transaction; });
    if (joint && policy && policies > 8) throw std::runtime_error("controller exceeds eight policy members");
    std::vector<HipMtpGreedyCycleResult> out;
    for (const auto& c : members) {
        const auto drafts = joint ? std::max(1U, c.maximum_candidates) : c.maximum_candidates;
        if (!c.maximum_candidates && drafts) throw std::runtime_error("sampled candidate lacks distribution");
        if (joint) (c.transaction ? policy : joint)->members++;
        out.push_back({c.slot, drafts, 1});
    }
    return out;
}
struct CohortAdapter {
    struct { CohortStats cohort; } stats_;
    Controller joint_, policy_joint_;
    std::size_t joint_group_start_ = 0;
    auto run(const std::vector<HipMtpGreedyCohortMember>& members, bool joint = true) {
#include "cohort.inc"
        return results;
    }
};

struct TestClock {
    using duration = std::chrono::duration<double, std::milli>;
    using time_point = std::chrono::time_point<TestClock, duration>;
    inline static double ms = 0;
    static time_point now() { return time_point(duration(ms)); }
};
struct Prompt {
    enum class Kind { serial, alignment, tiled, window };
    struct Block { std::uint32_t rows; };
    Kind kind;
    std::uint64_t pos = 0, end = 20000, tiled_count = 0;
    bool small = false;
    double duration_ms = 16;
    explicit Prompt(Kind k) : kind(k) {}
    std::uint64_t position() const { return pos; }
    std::uint64_t tiled_units() const { return tiled_count; }
    std::uint64_t remaining() const { return end - pos; }
    bool done() const { return pos == end; }
    void set_small_blocks(bool on) { small = on; }
    std::optional<Block> next_block() const {
        if (kind != Kind::tiled) return {};
        return Block{small ? 1024U : 8192U};
    }
    void step(std::uint32_t serial_rows) {
        const std::uint32_t rows = kind == Kind::alignment ? 1U :
            kind == Kind::tiled ? next_block()->rows : serial_rows;
        pos += std::min<std::uint64_t>(rows, remaining());
        if (kind == Kind::tiled) ++tiled_count;
        TestClock::ms += duration_ms;
    }
    void finish_block(Block b, int, bool, void*) { pos += b.rows; }
    void finish_window(std::uint32_t rows, bool) { pos += rows; }
};
struct Slot {
    std::unique_ptr<Prompt> prefill;
    unsigned overtakes = 0;
    bool finished = false, cancelled = false, prefilling = true;
};
static bool batch_prompt_small(const Options&) { return false; }
struct SchedulerAdapter {
    Options options_;
    struct { std::uint64_t prompt_steps = 0; double service_ms_suspended = 0; } stats_;
    std::vector<std::unique_ptr<Slot>> slots_;
    serve::StepPlanner step_planner_, tiled_step_planner_;
    double step_decode_gap_ms_ = 100, prompt_share_ = 0, prompt_debt_ms_ = 0;
    double step_prompt_ms_ = 0, step_decode_ms_ = 0, step_suspended_ms_ = 0;
    std::uint64_t step_prompt_rows_ = 0;
    std::uint32_t serial_rows_per_step_ = 16;
    bool step_prompt_ran_ = false, group_enabled = false, window_enabled = false;
    bool cancel_requested(const Slot& s) const { return s.cancelled; }
    void cancel_prefill(Slot& s, bool) { s.prefill.reset(); }
    void constraint_failed(Slot& s, const serve::ConstraintError&) { s.prefill.reset(); }
    void complete_prompt(Slot& s) { s.finished = true; s.prefill.reset(); }
    void add(Prompt::Kind kind, std::uint64_t end = 20000) {
        auto s = std::make_unique<Slot>();
        s->prefill = std::make_unique<Prompt>(kind);
        s->prefill->end = end;
        slots_.push_back(std::move(s));
    }
    bool run_prompt_group(Slot&) {
        if (!group_enabled) return false;
        struct Member { Slot* slot; Prompt::Block block; };
        std::vector<Member> members;
        for (auto& s : slots_) if (s->prefill) members.push_back({s.get(), {1024}});
        std::vector<int> logits(members.size());
        struct Owner { void* get() const { return nullptr; } };
        struct { bool group = false; Owner group_owner, block_owner; } lead;
        TestClock::ms += 2048;
#include "group.inc"
        return true;
    }
    bool run_prompt_windows(Slot&) {
        if (!window_enabled) return false;
        std::vector<Slot*> members;
        for (auto& s : slots_) if (s->prefill) members.push_back(s.get());
        std::vector<std::uint32_t> tickets(members.size(), 8);
        TestClock::ms += 16;
#include "windows.inc"
        return true;
    }
    std::uint32_t run() {
        using clock = TestClock;
        const bool planner_on = true, decoding = true;
        [[maybe_unused]] bool solo_tiled = false;
        [[maybe_unused]] bool plan_gap = false;
        std::uint32_t plan_rows = 0, plan_small_unit = 1024, plan_block_unit = 8192;
        const std::uint64_t short_rows = 2048;
        Slot* prompt = nullptr;
        step_prompt_ms_ = 0; step_suspended_ms_ = 0; step_prompt_rows_ = 0;
        std::vector<serve::StepLane> lanes;
        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i]->prefill) lanes.push_back({static_cast<std::uint32_t>(i), true, slots_[i]->prefill->remaining(), i});
        lanes.push_back({99, false});
#include "plan.inc"
#include "execute.inc"
#include "account.inc"
        return plan_rows;
    }
};

static void zero_draft() {
    CohortAdapter a;
    std::vector<HipMtpGreedyCohortMember> members{{0, 7, false}, {0, 19, false}};
    std::uint32_t completed = 0;
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        auto results = a.run(members);
        for (const auto& r : results) {
            check(r.drafts == 0, "sampling-off must never draft");
            completed += r.emitted;
        }
    }
    check(completed == 6, "both requests complete three output tokens");
    members.push_back({3, 21, false});
    check(a.run(members)[0].drafts == 0, "mixed zero/nonzero demand preserves zero");
}
static void policy_cohorts() {
    for (const auto count : {9U, 17U, 25U}) {
        CohortAdapter a;
        std::vector<HipMtpGreedyCohortMember> members;
        for (unsigned i = 0; i < count; ++i) members.push_back({1, i * 7U + 10U, true});
        for (unsigned round = 0; round < 3; ++round) {
            auto results = a.run(members);
            for (std::size_t i = 0; i < members.size(); ++i)
                check(results[i].slot == members[i].slot, "partition preserves member/result indices");
        }
        check(a.joint_.members == 0 && a.policy_joint_.members == 3U * count, "policy history stays separate");
    }
    CohortAdapter a;
    std::vector<HipMtpGreedyCohortMember> mixed;
    for (unsigned i = 0; i < 24; ++i) mixed.push_back({1, i + 100, i % 2 == 0});
    auto results = a.run(mixed);
    for (std::size_t i = 0; i < mixed.size(); ++i)
        check(results[i].slot == mixed[i].slot, "mixed order preserved");
    check(a.joint_.members == 12 && a.policy_joint_.members == 12, "mixed histories stay separate");
}
static void prefix_profiles() {
    strix::envcfg::value = -1;
    check(!ForkAdapter({false, {0, 0}}).fork_on_, "unqualified reuse is not automatic");
    strix::envcfg::value = 1;
    check(!ForkAdapter({false, {0, 1}}).fork_on_, "split profiles decline explicit reuse");
    check(ForkAdapter({false, {0, 0}}).fork_on_, "same-profile explicit experiment remains available");
    check(!ForkAdapter({true, {0, 0}}).fork_on_, "baseline declines reuse");
    strix::envcfg::value = 0;
    check(!ForkAdapter({false, {0, 0}}).fork_on_, "off declines reuse");
}
static void prompt_accounting() {
    for (const auto kind : {Prompt::Kind::serial, Prompt::Kind::alignment}) {
        SchedulerAdapter a;
        a.add(kind);
        check(a.run() == 1024, "initial plan buys small tile");
        const auto rows = kind == Prompt::Kind::alignment ? 1U : 16U;
        check(a.step_planner_.prompt_row_ms() == 16.0 / rows, "account actual serial/alignment delta");
        check(a.tiled_step_planner_.prompt_row_ms() == 0, "non-tiled work cannot train tile cost");
        a.slots_[0]->prefill->kind = Prompt::Kind::tiled;
        a.slots_[0]->prefill->duration_ms = 2048;
        check(a.run() == 1024, "serial observations cannot buy full T8192");
        check(a.tiled_step_planner_.prompt_row_ms() == 2, "actual tiled rows train tile cost");
        check(a.run() == 1024, "costly tile stays small");
    }
    for (bool windows : {false, true}) {
        SchedulerAdapter a;
        a.step_decode_gap_ms_ = 0;
        a.group_enabled = !windows; a.window_enabled = windows;
        const auto rows = windows ? 8U : 1024U;
        a.add(windows ? Prompt::Kind::window : Prompt::Kind::tiled, rows);
        a.add(windows ? Prompt::Kind::window : Prompt::Kind::tiled, rows);
        (void)a.run();
        check(a.step_prompt_rows_ == 2U * rows, "count all grouped/window lanes before releasing prompts");
        check(a.slots_[0]->finished && a.slots_[1]->finished, "grouped/window completion releases both prompts");
        check(a.step_planner_.prompt_row_ms() == 1, "grouped/window time uses actual total rows");
        check(a.tiled_step_planner_.prompt_row_ms() == 0, "group throughput cannot train single tile cost");
    }
    SchedulerAdapter a;
    a.add(Prompt::Kind::serial);
    a.slots_[0]->cancelled = true;
    (void)a.run();
    check(a.step_planner_.prompt_row_ms() == 0, "cancelled zero work has no planned-row credit");
    SchedulerAdapter bounded;
    bounded.add(Prompt::Kind::tiled);
    bounded.add(Prompt::Kind::tiled);
    bounded.group_enabled = true;
    (void)bounded.run();
    check(bounded.step_prompt_rows_ == 1024, "gap-bound tiled execution stays a single unit");
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "expected regression name");
        const std::string name = argv[1];
        if (name == "zero_draft") zero_draft();
        else if (name == "policy_cohorts") policy_cohorts();
        else if (name == "prefix_profiles") prefix_profiles();
        else if (name == "prompt_accounting") prompt_accounting();
        else throw std::runtime_error("unknown regression");
        std::cout << "PASS " << name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}



