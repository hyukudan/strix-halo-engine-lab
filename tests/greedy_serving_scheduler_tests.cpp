#include "rotary_mock.hpp"
#include "strix/greedy_serving_scheduler.hpp"
#include "strix/session/kv_capacity.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Rendezvous {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned arrived = 0;
    unsigned target;
    explicit Rendezvous(unsigned parties) : target(parties) {}
    void enter() {
        std::unique_lock lock(mutex);
        ++arrived; cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(10), [&] { return arrived == target; }))
            throw std::runtime_error("independent workers did not reach concurrent backend calls");
    }
};
struct Backend final : strix::SerialGreedyStepBackend {
    std::shared_ptr<std::atomic<unsigned>> calls;
    Rendezvous* barrier = nullptr;
    std::vector<unsigned>* events = nullptr;
    unsigned identity = 0;
    explicit Backend(std::shared_ptr<std::atomic<unsigned>> value, Rendezvous* rendezvous = nullptr)
        : calls(std::move(value)), barrier(rendezvous) {}
    std::vector<float> execute(std::uint32_t token, std::uint64_t position) override {
        std::vector<float> logits(16, -1);
        logits[execute_greedy(token, position, 16)] = 1;
        return logits;
    }
    std::uint32_t execute_greedy(std::uint32_t token, std::uint64_t, std::uint64_t) override {
        ++*calls;
        if (events) events->push_back(identity);
        if (barrier) barrier->enter();
        return (token + 1U) % 16U;
    }
    void execute_state(std::uint32_t, std::uint64_t) override { ++*calls; if (events) events->push_back(identity); }
};
struct FailingBackend final : strix::SerialGreedyStepBackend {
    std::vector<float> execute(std::uint32_t, std::uint64_t) override {
        throw std::runtime_error("contained backend failure");
    }
};
strix::ServingCompatibility compatibility() {
    strix::ServingCompatibility key;
    key.model_content.bytes.front() = 1;
    key.numerical_identity = 2;
    return key;
}
strix::GreedyServingPolicy policy(unsigned concurrency = 1) {
    strix::GreedyServingPolicy value;
    value.concurrency = concurrency; value.max_retained_requests = 2;
    value.reservation_budget_bytes = 4096; value.shared_reservation_bytes = 64;
    if (concurrency == 4) { value.max_retained_requests = 4; value.reservation_budget_bytes = 8192; }
    return value;
}
void check_page_leases() {
    using Capacity = strix::session::KvCapacity;
    Capacity capacity(112, 0, 2);
    const strix::session::KvPageLayout layout{4, 8, 20};
    auto result = capacity.reserve(layout, 5, 2, 0);
    require(result.admission == Capacity::Admission::accepted && capacity.reserved_bytes() == 72,
            "two contexts need four KV pages and two complete slot states");
    require(capacity.reserve(layout, 1, 1, 0).admission == Capacity::Admission::slot_limit,
            "unused KV bytes bypassed the slot limit");
    {
        auto fork = result.lease.execution->kv.fork();
        require(fork.size() == 4 && capacity.reserved_bytes() == 72, "a fork charged shared pages twice");
        fork.write(0, [](auto&) {});
        require(capacity.reserved_bytes() == 80 && fork.owner(1) == result.lease.execution->kv.owner(1),
                "first write failed to charge exactly one private page");
        fork.write(0, [](auto&) {});
        require(capacity.reserved_bytes() == 80, "a private write charged another page");
        result.lease.execution.reset();
        require(capacity.reserved_bytes() == 32, "whole-slot state survived its request or shared pages lost credit");
    }
    require(capacity.reserved_bytes() == 0, "last fork retained page credits");
    require(capacity.reserve({1, UINT64_MAX, 1}, 1, 1, 0).admission == Capacity::Admission::byte_limit &&
            capacity.reserve({1, 0, UINT64_MAX}, 1, 2, 0).admission == Capacity::Admission::byte_limit,
            "overflow admitted an undersized extent");
    auto full = capacity.reserve({1, 112, 0}, 1, 1, 0);
    auto fork = full.lease.execution->kv.fork();
    bool refused = false;
    try { fork.write(0, [](auto&) {}); } catch (const std::length_error&) { refused = true; }
    require(refused && fork.owner(0) == full.lease.execution->kv.owner(0) && capacity.reserved_bytes() == 112,
            "denied COW changed ownership or capacity");
}

void check_context_admission() {
    auto config = policy();
    config.reservation_budget_bytes = 512;
    auto owner = std::make_shared<int>(7);
    strix::GreedyServingScheduler scheduler(config, compatibility(), owner, {}, {4, 32, 128});
    const std::vector<std::uint32_t> prefix{1};
    const auto calls = std::make_shared<std::atomic<unsigned>>(0);
    unsigned factories = 0;
    const auto factory = [&] {
        ++factories;
        return std::make_unique<Backend>(calls);
    };
    std::uint64_t scope = 0;
    scheduler.with_optional_cache([&](auto& cache) { scope = cache.new_scope(); });
    const auto store = [&](std::size_t bytes) {
        bool stored = false;
        scheduler.with_optional_cache([&](auto& cache) {
            auto credit = cache.reserve(bytes);
            if (!credit) return;
            auto entry = std::make_shared<strix::PrefixCacheCheckpoint>();
            entry->retention = {scope, 512, 1, false, false};
            entry->name = entry->replacement = "prefix";
            stored = cache.retain_encoded(std::move(entry), std::vector<std::uint8_t>(bytes), &credit);
        });
        return stored;
    };
    require(store(256), "could not fill idle cache capacity");
    // Two full pages and a whole slot cost 192, retained results cost 64.
    auto first = scheduler.submit({prefix, 16, 8, 1, 15}, compatibility(), 1, 1, 64, owner, factory);
    require(first.request && factories == 1 && scheduler.reserved_bytes() == 320, "request failed to reclaim the idle cache before setup");
    scheduler.with_optional_cache([&](auto& cache) {
        require(cache.entry_count(scope) == 0 && cache.scope_evictions(scope) == 1,
                "pressured admission left the optional checkpoint resident");
    });
    require(!store(193) && store(192), "cache crossed the reserved request boundary");
    auto blocked = scheduler.submit({prefix, 16, 9, 1, 15}, compatibility(), 2, 1, 64, owner, factory);
    require(blocked.admission == strix::ServingAdmission::reservation_budget && factories == 1,
            "oversubscribed context reached the factory");
    scheduler.with_optional_cache([&](auto& cache) {
        require(cache.entry_count(scope) == 1, "impossible request discarded a useful cache entry");
    });
    require(first.request.cancel(), "queued cancellation failed");
    scheduler.run_until_idle();
    require(first.request.state() == strix::ServingRequestState::cancelled &&
            scheduler.reserved_bytes() == 128 && *calls == 0 && calls.use_count() == 1, "cancellation leaked execution credits");
    first.request = {};
    require(scheduler.reserved_bytes() == 64, "retained request credit survived the last handle");

    bool rolled_back = false;
    try {
        (void)scheduler.submit({prefix, 16, 9, 1, 15}, compatibility(), 2, 1, 64, owner,
            []() -> std::unique_ptr<strix::SerialGreedyStepBackend> { throw std::runtime_error("factory"); });
    } catch (const std::runtime_error&) { rolled_back = true; }
    require(rolled_back && scheduler.reserved_bytes() == 64, "partial setup retained page or slot credits");
    auto retry = scheduler.submit({prefix, 16, 9, 1, 15}, compatibility(), 2, 1, 64, owner, factory);
    require(retry.request && scheduler.reserved_bytes() == 352, "retry lost one full slot or KV page");
    scheduler.run_until_idle();
    require(retry.request.result().generated_tokens == std::vector<std::uint32_t>{2} &&
            scheduler.reserved_bytes() == 128 && calls.use_count() == 1, "completion failed to release execution capacity");
}

}

int main() try {
    check_page_leases();
    check_context_admission();
    auto owner = std::make_shared<int>(1);
    auto calls = std::make_shared<std::atomic<unsigned>>(0);
    const auto factory = [&] { return std::make_unique<Backend>(calls); };
    std::vector<std::uint32_t> prefix{1, 2, 3};
    for (const auto chunk : {1U, 2U, 100U}) {
        Backend backend(calls);
        strix::SerialGreedyTokenCursor cursor({prefix, 16, 10, 3, 15});
        while (!cursor.done()) cursor.advance(backend, chunk);
        const auto result = cursor.take_result();
        require(result.generated_tokens == std::vector<std::uint32_t>({4, 5, 6}) &&
            result.steps_executed == 5 && result.final_position == 5,
            "chunk boundaries changed greedy tokens or the last-emitted/unconsumed frontier");
    }
    for (auto factor : {2U, 4U}) {
        auto key = compatibility(); key.rotary = mock_rotary(factor)->identity();
        require(key != compatibility(), "same-shape serving numerical compatibility binds rotary mode");
        auto other = compatibility(); other.rotary = mock_rotary(factor == 2U ? 4U : 2U)->identity();
        require(key != other, "same-shape factor-2/factor-4 cohorts differ");
        strix::GreedyServingScheduler scheduler(policy(),key,owner);
        for(const auto& mismatch : {compatibility(),other}) {
            unsigned setups=0;
            auto rejected=scheduler.submit({prefix,16,10,3,15},mismatch,99,1,1024,owner,[&] {
                ++setups;return std::make_unique<Backend>(calls);
            });
            require(rejected.admission==strix::ServingAdmission::incompatible && !setups,
                "rotary serving mismatch refuses before backend allocation or publication");
        }
    }

    calls->store(0);
    // Stop boundaries retain the final emitted token without consuming it.
    for (const auto budget : {0U, 1U, 5U}) {
        Backend backend(calls);
        std::vector<std::uint32_t> one{1};
        strix::SerialGreedyTokenCursor cursor({one, 16, 1, budget, 15});
        while (!cursor.done()) cursor.advance(backend);
        require(cursor.result().steps_executed == 1 && cursor.result().final_position == 1 &&
            cursor.result().generated_tokens.size() == (budget ? 1U : 0U),
            "capacity/budget boundary consumed a final emitted token");
    }
    {
        Backend backend(calls);
        std::vector<std::uint32_t> one{1};
        strix::SerialGreedyTokenCursor cursor({one, 16, 10, 5, 2});
        cursor.advance(backend);
        require(cursor.done() && cursor.result().stop_reason == strix::SerialRequestStopReason::eos &&
            cursor.result().steps_executed == 1, "EOS token was consumed instead of stopping");
    }
    calls->store(0);
    {
        strix::GreedyServingScheduler scheduler(policy(), compatibility(), owner);
        auto first = scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 1, 1, 1024, owner, factory);
        auto second = scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 2, 1, 1024, owner, factory);
        require(first.admission == strix::ServingAdmission::accepted && second.request.cancel(), "admission/cancel failed");
        auto duplicate = scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 1, 1, 1024, owner, factory);
        require(duplicate.admission == strix::ServingAdmission::duplicate_session, "live session was admitted twice");
        duplicate = scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 1, 2, 1024, owner, factory);
        require(duplicate.admission == strix::ServingAdmission::duplicate_session, "a new epoch reused a still-retained session object");
        prefix.back() = 9; // admission privately owns the immutable prefix
        scheduler.run_until_idle();
        require(first.request.state() == strix::ServingRequestState::completed &&
            second.request.state() == strix::ServingRequestState::cancelled && calls->load() == 5,
            "queued cancellation affected another session or submitted work");
        require(first.request.result().generated_tokens == std::vector<std::uint32_t>({4, 5, 6}),
            "caller mutation changed the admitted request");
        const auto metrics = first.request.metrics();
        std::uint64_t intervals = 0;
        for (const auto bin : metrics.inter_token_ns_histogram) intervals += bin;
        require(metrics.emitted_tokens == 3 && metrics.inter_token_intervals == 2 && intervals == 2 &&
            metrics.time_to_first_token_ns.has_value(), "host token delivery metrics lost intervals");
        require(second.request.metrics().emitted_tokens == 0 &&
            !second.request.metrics().time_to_first_token_ns, "queued cancellation invented token timing");
        require(scheduler.reserved_bytes() == 2112, "completed handles prematurely released their resource reservations");
        auto blocked = scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 3, 1, 1024, owner, factory);
        require(blocked.admission == strix::ServingAdmission::request_limit, "retained handles bypassed admission limits");
        first.request = {}; second.request = {};
        require(scheduler.reserved_bytes() == 64, "finished handles did not release reservations after owners");
        bool setup_failed = false;
        try {
            static_cast<void>(scheduler.submit({prefix, 16, 10, 3, 15}, compatibility(), 4, 1, 1024, owner,
                []() -> std::unique_ptr<strix::SerialGreedyStepBackend> { throw std::runtime_error("factory failure"); }));
        } catch (const std::runtime_error&) { setup_failed = true; }
        scheduler.run_until_idle();
        require(setup_failed && scheduler.reserved_bytes() == 64, "failed setup leaked admission or blocked idle");
    }
    {
        auto fair_policy = policy(); fair_policy.prefill_chunk_rows = 1; fair_policy.max_decode_burst = 1;
        fair_policy.aging_ns = std::numeric_limits<std::uint64_t>::max();
        strix::GreedyServingScheduler scheduler(fair_policy, compatibility(), owner);
        std::vector<unsigned> events;
        const auto make_recording = [&](unsigned id) {
            auto backend = std::make_unique<Backend>(calls); backend->events = &events; backend->identity = id;
            return backend;
        };
        std::vector<std::uint32_t> long_prefix{1, 1, 1, 1}, short_prefix{2};
        auto a = scheduler.submit({long_prefix, 16, 10, 1, 15}, compatibility(), 20, 1, 1024, owner,
            [&] { return make_recording(1); });
        auto b = scheduler.submit({short_prefix, 16, 10, 4, 15}, compatibility(), 21, 1, 1024, owner,
            [&] { return make_recording(2); });
        scheduler.run_until_idle();
        require(events.size() == 8 && events[0] == 1 && events[1] == 2 && events[2] == 2 && events[3] == 1,
            "decode priority starved a waiting prefill instead of honoring the bounded burst");
        require(a.request.result().generated_tokens == std::vector<std::uint32_t>({2}) &&
            b.request.result().generated_tokens == std::vector<std::uint32_t>({3, 4, 5, 6}), "fair scheduling changed greedy outputs");
    }
    {
        strix::GreedyServingScheduler scheduler(policy(), compatibility(), owner);
        std::vector<std::uint32_t> single{1};
        auto bad = scheduler.submit({single, 16, 10, 1, 15}, compatibility(), 30, 1, 1024, owner,
            [] { return std::make_unique<FailingBackend>(); });
        auto good = scheduler.submit({single, 16, 10, 1, 15}, compatibility(), 31, 1, 1024, owner, factory);
        scheduler.run_until_idle();
        require(bad.request.state() == strix::ServingRequestState::failed && bad.request.failure() &&
            bad.request.result().steps_executed == 0, "backend failure published a successful step");
        require(good.request.state() == strix::ServingRequestState::completed &&
            good.request.result().generated_tokens == std::vector<std::uint32_t>({2}),
            "failed session prevented an independent request from completing");
        require(scheduler.reserved_bytes() == 2112, "failed handle released resources before its owner");
    }
    for (const auto concurrency : {2U, 4U}) {
        Rendezvous rendezvous(concurrency);
        strix::GreedyServingScheduler scheduler(policy(concurrency), compatibility(), owner);
        std::vector<std::uint32_t> single{1};
        const auto concurrent_factory = [&] { return std::make_unique<Backend>(calls, &rendezvous); };
        std::vector<strix::ServingSubmission> requests;
        for (unsigned i = 0; i < concurrency; ++i)
            requests.push_back(scheduler.submit({single, 16, 10, 1, 15}, compatibility(), 10U + i, 1, 1024, owner, concurrent_factory));
        scheduler.run_until_idle();
        for (const auto& request : requests) {
            require(request.request.state() == strix::ServingRequestState::completed, "independent sessions did not execute concurrently");
            require(request.request.result().generated_tokens == std::vector<std::uint32_t>({2}), "concurrent session outputs differ");
            require(request.request.metrics().time_to_first_token_ns.has_value(), "completed request has no host TTFT observation");
        }
    }
    std::cout << "greedy serving scheduler tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
