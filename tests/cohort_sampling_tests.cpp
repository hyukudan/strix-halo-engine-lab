#include "strix/serve/cohort_decode.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace strix::serve;

int failures = 0;
void check(bool ok, std::string_view message) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::array<float, 5> final_row(std::uint32_t fed, std::uint64_t position) {
    std::array<float, 5> logits{0.9F, 0.85F, 0.95F, 0.91F, 0.87F};
    logits[(fed + position) % logits.size()] = 1.0F;
    return logits;
}

class FakeBackend final : public CohortBackend {
public:
    explicit FakeBackend(bool sampling = true) : sampling_(sampling) {}

    std::uint32_t lanes() const override { return 3; }
    std::uint32_t rows_per_forward() const override { return 12; }
    std::uint32_t max_positions() const override { return 512; }
    bool supports_final_logits() const override { return sampling_; }

    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) {
            positions_.at(lane) = 0;
            markers_.at(lane) = tokens.front();
        }
        positions_.at(lane) += tokens.size();
    }

    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        check(pending_.empty(), "verify starts outside a pending window");
        pending_.assign(groups.begin(), groups.end());
        std::size_t base = 0;
        bool greedy = false, sampled = false;
        for (const auto& g : groups) {
            const auto lane = g.lane;
            // the driver names exactly the rows whose final logits a sampled lane will read
            if (g.reads_scores != (markers_.at(lane) == 2U)) ++score_flag_errors;
            if (markers_.at(lane) == 2U) {
                sampled = true;
                check(g.rows == 1U, "sampled lane never verifies a context draft");
            } else {
                greedy = true;
            }
            for (std::size_t i = 0; i < g.rows; ++i) top[base + i] = tokens[base + i];
            const auto last = base + g.rows - 1U;
            logits_.at(lane) = final_row(tokens[last], positions_.at(lane) + g.rows);
            base += g.rows;
        }
        mixed_window = mixed_window || (greedy && sampled);
    }

    std::span<const float> read_final_logits(std::uint32_t lane) override {
        check(!pending_.empty() && markers_.at(lane) == 2U, "only sampled lanes read pending logits");
        ++reads;
        return logits_.at(lane);
    }

    void commit(std::span<const std::uint32_t> accept) override {
        for (std::size_t i = 0; i < accept.size(); ++i) positions_.at(pending_.at(i).lane) += accept[i];
        pending_.clear();
    }

    void release(std::uint32_t) override {}

    unsigned reads = 0;
    unsigned score_flag_errors = 0;
    bool mixed_window = false;

private:
    bool sampling_;
    std::array<std::uint64_t, 3> positions_{};
    std::array<std::uint32_t, 3> markers_{};
    std::array<std::array<float, 5>, 3> logits_{};
    std::vector<CohortGroup> pending_;
};

std::map<std::uint64_t, CohortResult> run(FakeBackend& backend, const std::vector<CohortRequest>& requests) {
    CohortConfig config;
    config.prior = {10.0, 0.01, 0.0, 0.0};
    config.learn_cost = false;
    CohortDecoder decoder(backend, config);
    decoder.set_fake_clock(0.01);
    for (const auto& request : requests) decoder.submit(request);
    decoder.run();
    std::map<std::uint64_t, CohortResult> results;
    for (auto& result : decoder.take_finished()) results.emplace(result.id, std::move(result));
    return results;
}

std::vector<std::uint32_t> reference(const CohortRequest& request) {
    strix::TokenSampler sampler(request.sampling, strix::TokenSamplerOptions{});
    auto token = request.prompt.back();
    std::vector<std::uint32_t> ids;
    for (std::uint64_t i = 0; i < request.max_tokens; ++i) {
        const auto position = request.prompt.size() + i;
        token = sampler.decide(final_row(token, position), position, nullptr).token;
        ids.push_back(token);
    }
    return ids;
}
}  // namespace

int main() {
    const CohortRequest greedy{1, std::vector<std::uint32_t>(32, 1), 64, {}};
    auto zero = greedy;
    zero.sampling = {0.0F, 3U, 0.875F, 0.125F, 51U};
    const CohortRequest sampled{2, std::vector<std::uint32_t>(32, 2), 64, {}, {0.75F, 4U, 0.875F, 0.125F, 51U}};
    const CohortRequest other_seed{3, sampled.prompt, sampled.max_tokens, {}, {0.75F, 4U, 0.875F, 0.125F, 91U}};

    FakeBackend greedy_backend(false);
    const auto old = run(greedy_backend, {greedy}).at(greedy.id);
    const auto temp_zero = run(greedy_backend, {zero}).at(zero.id);
    check(old.ids == std::vector<std::uint32_t>(64, 1U) && temp_zero.ids == old.ids,
          "temperature zero produces the existing greedy ids");
    check(old.drafted > 0U && temp_zero.drafted == old.drafted && temp_zero.accepted == old.accepted &&
          temp_zero.steps == old.steps && greedy_backend.reads == 0U && temp_zero.sampling == "greedy",
          "temperature zero preserves greedy draft scheduling and device picks");
    const auto refused = run(greedy_backend, {sampled}).at(sampled.id);
    check(refused.ids.empty() && refused.error.starts_with("temperature"), "backend without readback refuses sampling");

    FakeBackend mixed, reordered, solo;
    const auto first = run(mixed, {zero, sampled, other_seed});
    const auto second = run(reordered, {other_seed, sampled, zero});
    const auto alone = run(solo, {sampled}).at(sampled.id);
    check(mixed.mixed_window && reordered.mixed_window, "both lane orders exercise greedy and sampled rows together");
    check(first.at(greedy.id).ids == old.ids && second.at(greedy.id).ids == old.ids,
          "sampled neighbors preserve greedy ids");
    for (const auto* request : {&sampled, &other_seed}) {
        const auto& a = first.at(request->id);
        const auto& b = second.at(request->id);
        check(a.error.empty() && a.ids == b.ids && a.ids == reference(*request),
              "lane order does not affect decisions from the final row and absolute position");
        check(a.drafted == 0U && b.drafted == 0U && a.accepted == 0U && b.accepted == 0U &&
              a.sampling == "sampling-v1" && b.sampling == "sampling-v1", "sampled receipts report no drafts");
    }
    check(alone.ids == first.at(sampled.id).ids && alone.ids != first.at(other_seed.id).ids,
          "same seed replays solo, different seeds differ");
    check(mixed.reads == 128U && reordered.reads == 128U && solo.reads == 64U, "one readback per sampled decision");
    check(greedy_backend.score_flag_errors == 0U && mixed.score_flag_errors == 0U && reordered.score_flag_errors == 0U &&
              solo.score_flag_errors == 0U, "verify groups flag exactly the lanes that read their final logits");
    return failures == 0 ? 0 : 1;
}
