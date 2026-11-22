#include "strix/serve/cohort_session.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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

// The fake owns real per-lane contexts; the session, queue, leases and
// publication path are the production implementations under test.
class Backend final : public sv::CohortBackend {
public:
    std::uint32_t lanes() const override { return 2; }
    std::uint32_t rows_per_forward() const override { return 2; }
    std::uint32_t max_positions() const override { return 32; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> ids, bool first) override {
        check(pending.empty(), "prefill overlaps an uncommitted batch");
        auto& context = contexts.at(lane);
        if (first) { context.clear(); ++starts; }
        context.insert(context.end(), ids.begin(), ids.end());
        prefill_rows += ids.size();
    }
    void verify(std::span<const sv::CohortGroup> groups, std::span<const std::uint32_t> ids,
                std::span<std::uint32_t> top) override {
        check(pending.empty(), "verify overlaps an uncommitted batch");
        max_groups = std::max(max_groups, groups.size());
        std::size_t offset = 0;
        for (const auto& group : groups) {
            check(group.rows == 1, "greedy fixture unexpectedly drafts");
            auto context = contexts.at(group.lane);
            context.push_back(ids[offset]);
            top[offset] = next(context);
            pending.push_back({group.lane, ids[offset]});
            ++offset;
        }
        check(offset == ids.size() && ids.size() == top.size(), "batch shape mismatch");
    }
    void commit(std::span<const std::uint32_t> keep) override {
        check(keep.size() == pending.size(), "commit shape mismatch");
        for (std::size_t i = 0; i < keep.size(); ++i) {
            check(keep[i] == 1, "greedy fixture did not commit its anchor");
            contexts.at(pending[i].first).push_back(pending[i].second);
        }
        pending.clear();
    }
    bool supports_final_logits() const override { return sampled; }
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        check(sampled && std::ranges::any_of(pending, [lane](const auto& row) { return row.first == lane; }),
              "logits read outside a sampled lane");
        ++logit_reads;
        return logits;
    }
    void release(std::uint32_t lane) override { contexts.at(lane).clear(); ++releases; }
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
void exercise_sampling() {
    Backend backend;
    backend.sampled = true;
    sv::GenerationQueue queue;
    sv::CohortConfig config;
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

    const auto greedy_job = job({9, 7}, 5);
    const auto greedy = queue.submit(greedy_job);
    const auto mixed = queue.submit(request);
    drain_owner(runtime, queue);
    check(drain(mixed) == expected && drain(greedy) == solo(greedy_job) &&
          greedy->stats().sampling == "greedy" && backend.max_groups == 2 &&
          backend.logit_reads == 2U * request.max_tokens,
          "sampling changed by cohort width or read the greedy lane");

    request.sampling.temperature = 1e-100;
    const auto refused = queue.submit(request);
    drain_owner(runtime, queue);
    bool done = false;
    check(refused->wait(std::chrono::milliseconds(0), done, false).empty() && done &&
          refused->error().find("temperature") != std::string::npos,
          "positive temperature rounded to zero and ran greedy");
}
} // namespace

int main() {
    try {
        exercise_session();
        exercise_sampling();
        std::cout << "cohort session: concurrent streams, cancellation, reuse, stop and no-prefix policy PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
