#include "strix/serve/cohort_session.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sv = strix::serve;
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// A broken contract inside the fake backend ends the test: thrown, the session would turn it into a request error.
void backend_contract(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "fake backend contract broken: " << message << '\n';
        std::abort();
    }
}

// The fake owns real per-lane contexts; the session, queue, leases and
// publication path are the production implementations under test.
class Backend final : public sv::CohortBackend {
public:
    std::uint32_t lanes() const override { return 2; }
    std::uint32_t rows_per_forward() const override { return 2; }
    std::uint32_t max_positions() const override { return 32; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> ids, bool first) override {
        backend_contract(pending.empty(), "prefill overlaps an uncommitted batch");
        auto& context = contexts.at(lane);
        if (first) { context.clear(); ++starts; }
        context.insert(context.end(), ids.begin(), ids.end());
        prefill_rows += ids.size();
    }
    void verify(std::span<const sv::CohortGroup> groups, std::span<const std::uint32_t> ids,
                std::span<std::uint32_t> top) override {
        backend_contract(pending.empty(), "verify overlaps an uncommitted batch");
        max_groups = std::max(max_groups, groups.size());
        std::size_t offset = 0;
        for (const auto& group : groups) {
            backend_contract(group.rows == 1, "greedy fixture unexpectedly drafts");
            auto context = contexts.at(group.lane);
            context.push_back(ids[offset]);
            top[offset] = next(context);
            pending.push_back({group.lane, ids[offset]});
            ++offset;
        }
        backend_contract(offset == ids.size() && ids.size() == top.size(), "batch shape mismatch");
    }
    void commit(std::span<const std::uint32_t> keep) override {
        backend_contract(keep.size() == pending.size(), "commit shape mismatch");
        for (std::size_t i = 0; i < keep.size(); ++i) {
            backend_contract(keep[i] == 1, "greedy fixture did not commit its anchor");
            contexts.at(pending[i].first).push_back(pending[i].second);
        }
        pending.clear();
    }
    bool supports_final_logits() const override { return sampled; }
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        backend_contract(sampled && std::ranges::any_of(pending, [lane](const auto& row) { return row.first == lane; }),
              "logits read outside a sampled lane");
        ++logit_reads;
        return logits;
    }
    void release(std::uint32_t lane) override { contexts.at(lane).clear(); ++releases; }
    void bind_lane(std::uint32_t, std::uint64_t, std::span<const std::uint32_t> words) override {
        bound.emplace_back(words.begin(), words.end());
    }
    std::vector<std::vector<std::uint32_t>> bound;
    static std::uint32_t next(std::span<const std::uint32_t> context) {
        std::uint32_t value = 7;
        for (const auto id : context) value = (value * 5U + id) % 97U;
        return value;
    }
    std::array<std::vector<std::uint32_t>, 2> contexts;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pending;
    std::size_t max_groups = 0, starts = 0, releases = 0, prefill_rows = 0, logit_reads = 0;
    bool sampled = false;
    std::array<float, 3> logits{-4.0F, 3.0F, 2.0F};
};

sv::GenerationJob job(std::vector<std::uint32_t> prompt, std::uint32_t count = 4) {
    sv::GenerationJob result;
    result.prompt = std::move(prompt);
    result.max_tokens = count;
    return result;
}
std::vector<std::uint32_t> solo(const sv::GenerationJob& request) {
    auto context = request.prompt;
    std::vector<std::uint32_t> result;
    for (std::uint32_t i = 0; i < request.max_tokens; ++i) {
        const auto id = Backend::next(context);
        result.push_back(id);
        context.push_back(id);
        if (std::ranges::find(request.stop_tokens, id) != request.stop_tokens.end()) break;
    }
    return result;
}
void drain_owner(sv::SessionRuntime& runtime, sv::GenerationQueue& queue) {
    for (unsigned step = 0; step < 100; ++step) {
        const auto snapshot = queue.snapshot();
        if (snapshot.running == 0 && snapshot.queued == 0) return;
        runtime.step(false);
    }
    throw std::runtime_error("cohort session failed to finish bounded host work");
}
std::vector<std::uint32_t> drain(const std::shared_ptr<sv::GenerationTicket>& ticket) {
    bool done = false;
    auto ids = ticket->wait(std::chrono::milliseconds(0), done, false);
    check(done, "ticket was not completed");
    check(ticket->error().empty(), "unexpected request error");
    return ids;
}

void exercise_session() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
    config.context_draft = false;
    config.prompt_slice = 32;
    sv::CohortSession session(queue, backend, config);
    auto& runtime = session.runtime();

    auto a_job = job({11, 12, 13}, 8);
    auto b_job = job({21, 22, 23}, 5);
    std::vector<std::uint32_t> a_stream;
    const auto a_expected = solo(a_job), b_expected = solo(b_job);
    auto a = queue.submit(a_job), b = queue.submit(b_job);
    check(a && b, "queue refused two requests within capacity");
    runtime.step(false);
    check(a->wait_admission(std::chrono::milliseconds(0)) &&
          b->wait_admission(std::chrono::milliseconds(0)) && runtime.live() == 2,
          "session did not admit concurrent requests");
    for (unsigned step = 0; (a_stream.empty() || backend.max_groups < 2) && step < 10; ++step) {
        bool done = false;
        const auto chunk = a->wait(std::chrono::milliseconds(0), done, false);
        a_stream.insert(a_stream.end(), chunk.begin(), chunk.end());
        if (done) {
            std::cerr << "step " << step << ": error '" << a->error() << "', finish " << a->stats().finish << ", streamed "
                      << a_stream.size() << ", max_groups " << backend.max_groups << '\n';
        }
        check(!done, "stream finished before cancellation could be exercised");
        if (a_stream.empty() || backend.max_groups < 2) runtime.step(false);
    }
    check(!a_stream.empty() && a_stream.size() < a_expected.size(), "stream did not publish before completion");
    a->cancel();
    drain_owner(runtime, queue);
    auto a_ids = a_stream;
    const auto rest = drain(a);
    a_ids.insert(a_ids.end(), rest.begin(), rest.end());
    const auto b_ids = drain(b);
    check(a_ids.size() >= a_stream.size() && a_ids.size() <= a_expected.size() &&
          std::equal(a_stream.begin(), a_stream.end(), a_ids.begin()) &&
          std::equal(a_ids.begin(), a_ids.end(), a_expected.begin()),
          "cancelled stream differs from its solo prefix");
    check(a->stats().finish == "cancelled" && a->stats().generated_tokens == a_ids.size(),
          "cancelled ticket did not retain its emitted prefix/accounting");
    check(b_ids == b_expected && b->stats().finish == "length", "cancellation changed its concurrent peer");
    check(backend.max_groups == 2 && backend.releases == 2, "requests did not batch or release their lanes");
    check(runtime.live() == 0 && runtime.cache_entries() == 0, "session retained a request or prefix");

    // Same prompt after retirement must be fully prefetched, not restored from
    // a retained prefix or stale physical lane. A different prompt uses it too.
    const auto old_rows = backend.prefill_rows;
    auto repeat_job = job({11, 12, 13}, 3);
    repeat_job.stable_prefix = 2;
    auto different_job = job({41, 42}, 3);
    auto repeat = queue.submit(repeat_job), different = queue.submit(different_job);
    drain_owner(runtime, queue);
    check(drain(repeat) == solo(repeat_job) && drain(different) == solo(different_job), "lane reuse leaked context");
    check(backend.prefill_rows - old_rows == repeat_job.prompt.size() + different_job.prompt.size() - 2U,
          "reused requests skipped prompt rows");
    check(repeat->stats().cached_tokens == 0 && different->stats().cached_tokens == 0 &&
          runtime.cache_entries() == 0 && runtime.cache_stats().hits == 0,
          "request-only session advertised prefix reuse");
    check(backend.starts == 4 && backend.releases == 4, "reused lanes did not reset and retire");

    auto stop_job = job({31, 32}, 7);
    stop_job.stop_tokens = {Backend::next(stop_job.prompt)};
    auto stop = queue.submit(stop_job);
    drain_owner(runtime, queue);
    check(drain(stop) == solo(stop_job) && stop->stats().finish == "stop", "stop termination was lost");
    // Receipts carry the lane's service split on the decoder clock (they read 0 before): prefill_ms is part of the
    // first-token time, decode_ms is non-negative.
    const auto split = stop->stats();
    check(split.first_token_ms && split.prefill_ms >= 0.0 && split.prefill_ms <= *split.first_token_ms + 1e-9 &&
          split.decode_ms >= 0.0, "cohort receipt prefill_ms / decode_ms");

    // Admission rejects sampled work before it reaches the greedy backend;
    // refusal must release the reservation so a later valid job can run.
    auto sampled = job({3, 4});
    sampled.sampling.temperature = 1;
    const auto before_reject = backend.starts;
    auto refused = queue.submit(sampled);
    drain_owner(runtime, queue);
    bool done = false;
    const auto refused_ids = refused->wait(std::chrono::milliseconds(0), done, false);
    check(refused_ids.empty() && done &&
          refused->error().find("temperature") != std::string::npos,
          "unsupported sampling was not refused as an admission error");
    check(backend.starts == before_reject, "refused sampled job touched physical state");
    auto valid_job = job({3, 4}, 2);
    auto valid = queue.submit(valid_job);
    drain_owner(runtime, queue);
    check(drain(valid) == solo(valid_job) && runtime.live() == 0, "admission refusal leaked a lease");
}
// The receipt split on a controlled decoder clock: every clock read advances 1 ms. The decoder's own event times
// (submit, first id, done) reach the finished hook; the receipt must report prefill_ms = first - submit and
// decode_ms = done - first, summing to done - submit. The prompt is read in one-row slices and decodes few ids, so
// the two intervals differ and a swapped or duplicated split fails.
void exercise_receipt_split() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
    config.context_draft = false;
    config.prompt_slice = 1;
    double submit = -1.0, first = -1.0, done = -1.0;
    sv::CohortSessionHooks hooks;
    hooks.finished = [&](const sv::GenerationJob&, const sv::CohortResult& result) {
        submit = result.submit_ms;
        first = result.first_ms;
        done = result.done_ms;
    };
    sv::CohortSession session(queue, backend, config, std::move(hooks));
    session.set_fake_clock(1.0);
    auto& runtime = session.runtime();
    auto work = job({41, 42, 43, 44, 45, 46, 47, 48}, 2);
    auto ticket = queue.submit(work);
    drain_owner(runtime, queue);
    check(drain(ticket) == solo(work), "fake-clock request changed its ids");
    const auto stats = ticket->stats();
    check(submit >= 0.0 && first > submit && done > first, "controlled clock: event times not captured in order");
    check(first - submit != done - first, "controlled clock: test setup needs unequal prefill and decode intervals");
    check(stats.prefill_ms == first - submit, "controlled clock: prefill_ms != first - submit");
    check(stats.decode_ms == done - first, "controlled clock: decode_ms != done - first");
    check(stats.prefill_ms + stats.decode_ms == done - submit, "controlled clock: split does not sum to done - submit");
}

void exercise_sampling() {
    Backend backend;
    backend.sampled = true;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
    config.context_draft = false;
    sv::CohortSession session(queue, backend, config);
    auto& runtime = session.runtime();
    sv::ApiConfig api;
    session.configure_api(api);
    check(api.engine_sampling_filters.temperature && api.engine_sampling_filters.seed &&
          api.engine_sampling_filters.top_k && api.engine_sampling_filters.top_p && api.engine_sampling_filters.min_p,
          "sampling backend did not advertise its executed filters");

    auto request = job({3, 4}, 5);
    request.sampling.temperature = 1.0;
    request.sampling.top_k = 1;
    request.sampling.top_p = 0.8;
    request.sampling.min_p = 0.1;
    request.sampling.seed = 544;
    request.sampling.seeded = true;
    const auto lone = queue.submit(request);
    drain_owner(runtime, queue);
    const auto expected = drain(lone);
    check(expected == std::vector<std::uint32_t>(request.max_tokens, 1U) && expected != solo(request),
          "sampled request took the greedy route or ignored top-k");
    const auto lone_stats = lone->stats();
    const auto receipt = std::ranges::find_if(lone_stats.engine_extra,
        [](const auto& pair) { return pair.first == "sampling_receipt"; });
    check(lone_stats.sampling == "sampling-v1" && receipt != lone_stats.engine_extra.end() &&
          receipt->second.find("\"seed\":544") != std::string::npos &&
          receipt->second.find("\"drafts\":0") != std::string::npos,
          "sampled result omitted the effective policy receipt");

    request.max_tokens = 24;
    request.sampling.top_k = 2;
    request.sampling.top_p = 1.0;
    const auto stochastic_start = backend.logit_reads;
    const auto stochastic = queue.submit(request);
    drain_owner(runtime, queue);
    const auto seeded = drain(stochastic);
    check(seeded.size() == request.max_tokens && std::ranges::find(seeded, 1U) != seeded.end() &&
          std::ranges::find(seeded, 2U) != seeded.end(), "seeded fixture did not draw both supported tokens");

    const auto greedy_job = job({9, 7}, request.max_tokens);
    const auto greedy = queue.submit(greedy_job);
    const auto mixed = queue.submit(request);
    drain_owner(runtime, queue);
    check(drain(mixed) == seeded && drain(greedy) == solo(greedy_job) &&
          greedy->stats().sampling == "greedy" && backend.max_groups == 2 &&
          backend.logit_reads - stochastic_start == 2U * request.max_tokens,
          "seeded sampling changed by cohort width or read the greedy lane");

    ++request.sampling.seed;
    const auto other_seed = queue.submit(request);
    drain_owner(runtime, queue);
    check(drain(other_seed) != seeded, "a different seed did not change the stochastic continuation");

    request.sampling.temperature = 1e-100;
    const auto refused = queue.submit(request);
    drain_owner(runtime, queue);
    bool done = false;
    check(refused->wait(std::chrono::milliseconds(0), done, false).empty() && done &&
          refused->error().find("temperature") != std::string::npos,
          "positive temperature rounded to zero and ran greedy");
    check(refused->stats().error_status == 400, "a sampling refusal did not carry HTTP 400");
}
void exercise_hooks() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
    config.context_draft = false;
    sv::CohortSessionHooks hooks;
    std::size_t prepared = 0, finished = 0;
    std::vector<std::string> finishes;
    std::uint32_t seen_identity = 0;
    hooks.prepare = [&](const sv::GenerationJob& request_job, sv::CohortRequest& request) {
        ++prepared;
        if (request_job.prompt.front() == 99U) throw std::invalid_argument("engine refused the prompt");
        if (request_job.prompt.front() == 31U) request.stop.push_back(Backend::next(request_job.prompt));
        if (request_job.prompt.front() == 5U) request.context_tokens = 4U;
        request.sampler_options.request_identity = 7U;
        seen_identity = static_cast<std::uint32_t>(request.sampler_options.request_identity);
        request.traits = {request_job.prompt.front(), 2U};
    };
    hooks.finished = [&](const sv::GenerationJob& request_job, const sv::CohortResult& result) {
        ++finished;
        check(result.ids.size() <= request_job.max_tokens, "finished hook saw more ids than requested");
        finishes.emplace_back(result.finish);
    };
    sv::CohortSession session(queue, backend, config, hooks);
    auto& runtime = session.runtime();

    // An engine default stop id added by prepare ends the request and reports stop.
    auto stop_job = job({31, 32}, 7);
    const auto stop = queue.submit(stop_job);
    drain_owner(runtime, queue);
    auto stop_expected = stop_job;
    stop_expected.stop_tokens = {Backend::next(stop_job.prompt)};
    check(drain(stop) == solo(stop_expected) && stop->stats().finish == "stop" && seen_identity == 7U,
          "prepare did not reach the request");
    check(!backend.bound.empty() && backend.bound.back() == std::vector<std::uint32_t>({31U, 2U}),
          "prepare traits did not reach the backend");

    // The context ceiling ends the request as context; a tie with max_tokens is length.
    auto context_job = job({5, 6}, 10);
    const auto context = queue.submit(context_job);
    drain_owner(runtime, queue);
    auto context_expected = context_job;
    context_expected.max_tokens = 4;
    check(drain(context) == solo(context_expected) && context->stats().finish == "context",
          "context ceiling did not finish as context");
    context_job.max_tokens = 4;
    const auto tie = queue.submit(context_job);
    drain_owner(runtime, queue);
    check(drain(tie) == solo(context_job) && tie->stats().finish == "length", "a tie with max_tokens was not length");

    // prepare refusing a job refuses it like validate and leaks nothing.
    const auto starts = backend.starts;
    const auto refused = queue.submit(job({99, 1}));
    drain_owner(runtime, queue);
    bool done = false;
    check(refused->wait(std::chrono::milliseconds(0), done, false).empty() && done &&
          refused->error().find("engine refused") != std::string::npos && backend.starts == starts,
          "prepare refusal did not refuse the job");
    // a prepare refusal is a request error: HTTP 400 with the invalid_request code, not the execution-failure default
    check(refused->stats().error_status == 400 && refused->stats().error_code == "invalid_request",
          "prepare refusal did not carry HTTP 400");
    const auto valid_job = job({3, 4}, 2);
    const auto valid = queue.submit(valid_job);
    drain_owner(runtime, queue);
    check(drain(valid) == solo(valid_job) && runtime.live() == 0, "prepare refusal leaked a lease");

    check(prepared == 5U && finished == 4U && finishes == std::vector<std::string>({"stop", "context", "length", "length"}),
          "finished hook was not called once per finished request");
}
void exercise_idle_and_annotate() {
    {   // an idle callback never makes step(false) block: two consecutive calls on an empty queue return promptly
        Backend backend;
        sv::GenerationQueue queue;
        sv::CohortConfig config;
        config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
        config.context_draft = false;
        sv::CohortSessionHooks hooks;
        int beats = 0;
        hooks.idle = [&] { ++beats; };
        hooks.idle_interval = std::chrono::milliseconds(1000);
        sv::CohortSession session(queue, backend, config, hooks);
        for (int i = 0; i < 2; ++i) {
            const auto start = std::chrono::steady_clock::now();
            session.runtime().step(false);
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            check(ms < 200.0, "step(false) with an idle callback blocked on an empty queue");
        }
        check(beats >= 1, "idle callback was not called");
    }
    {   // annotate sees the sampling receipt and populated stats; the published stats keep the receipt
        Backend backend;
        backend.sampled = true;
        sv::GenerationQueue queue;
        sv::CohortConfig config;
        config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
        config.context_draft = false;
        sv::CohortSessionHooks hooks;
        bool called = false, saw = false;
        hooks.annotate = [&](const sv::GenerationJob&, const sv::CohortResult& result, sv::GenerationStats& stats) {
            called = true;
            saw = !result.sampling_receipt.empty() && stats.generated_tokens == result.ids.size() &&
                  stats.generated_tokens != 0U && !stats.sampling.empty() &&
                  std::ranges::any_of(stats.engine_extra, [&](const auto& pair) {
                      return pair.first == "sampling_receipt" && pair.second == result.sampling_receipt; });
        };
        sv::CohortSession session(queue, backend, config, hooks);
        auto request = job({3, 4}, 5);
        request.sampling.temperature = 1.0;
        request.sampling.top_k = 1;
        request.sampling.seed = 544;
        request.sampling.seeded = true;
        const auto ticket = queue.submit(request);
        drain_owner(session.runtime(), queue);
        drain(ticket);
        check(called && saw, "annotate did not see the sampling receipt alongside populated stats");
    }
}

// Source compatibility of the round-2 header: a braced fourth argument selects the hooks overload (not ambiguous), a
// validate callable alone still binds, and positional CohortResult initialisers keep their field order.
void exercise_compatibility() {
    Backend backend;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.prompt_share = 0.0;  // prompts never wait on decode debt (no wall-clock input): the grouping is deterministic
    config.context_draft = false;
    sv::CohortSession braced(queue, backend, config, {});
    std::size_t validated = 0;
    Backend other;
    sv::GenerationQueue other_queue;
    sv::CohortSession with_validate(other_queue, other, config,
                                    [&validated](const sv::GenerationJob&) { ++validated; });
    const auto ran_job = job({4, 5}, 2);
    const auto ran = other_queue.submit(ran_job);
    drain_owner(with_validate.runtime(), other_queue);
    check(drain(ran) == solo(ran_job) && validated == 1U, "a validate callable alone did not bind");
    // every leading field positionally (the -Wextra build requires the full list); finish keeps its default
    const sv::CohortResult positional{9U, {1U, 2U}, "", 3U, true, 0.0, 0.0, 0.0, 0U, 0U, 0U, false, "greedy", ""};
    check(positional.restored == 3U && positional.first_from_prefill && positional.finish == "error",
          "positional CohortResult initialiser changed meaning");
    (void)braced;
}
} // namespace

int main() {
    try {
        exercise_session();
        exercise_receipt_split();
        exercise_sampling();
        exercise_hooks();
        exercise_idle_and_annotate();
        exercise_compatibility();
        std::cout << "cohort session: concurrent streams, cancellation, reuse, stop and no-prefix policy PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
