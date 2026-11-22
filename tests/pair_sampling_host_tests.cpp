#include "strix/serve/pair_sampling.hpp"

#include <array>
#include <cstdio>

namespace sv = strix::serve;

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void test_selection() {
    sv::GenerationJob job;
    job.sampling = {0.0, 0.98, 4, 0.05, 33, true};
    const std::array<float, 2> local{1.0F, 1.1F};
    std::vector<float> remote(16386, -1000.0F);
    remote[2] = 1.3F;
    remote[16384] = 1.2F;
    remote.back() = 100.0F;  // beyond the tokenizer's vocabulary
    const std::size_t vocab = local.size() + remote.size() - 1;
    std::vector<float> full(local.begin(), local.end());
    full.insert(full.end(), remote.begin(), remote.end() - 1);
    const auto greedy = static_cast<std::uint32_t>(std::max_element(full.begin(), full.end()) - full.begin());
    std::array<std::uint32_t, 2> ids{greedy, greedy};
    const auto no_read = [](std::size_t) -> std::span<const float> {
        throw std::runtime_error("greedy route read logits");
    };
    const auto no_exchange = [](std::span<const float>, std::span<float>) {
        throw std::runtime_error("greedy route exchanged logits");
    };
    check(!sv::make_pair_sampler(job), "temperature zero must retain the engine's greedy choice");
    sv::sample_pair_rows(0, true, remote.size(), vocab, {}, ids, no_read, no_exchange);
    check(ids[0] == greedy && ids[1] == greedy, "temperature zero changed the engine's greedy IDs");

    job.sampling.temperature = 0.8;
    auto first = sv::make_pair_sampler(job);
    auto repeat = sv::make_pair_sampler(job);
    ++job.sampling.seed;
    auto different = sv::make_pair_sampler(job);
    check(first && repeat && different, "positive temperature must construct a shared sampler");
    const auto& p = first->policy();
    check(p.temperature == 0.8F && p.top_k == 4 && p.top_p == 0.98F && p.min_p == 0.05F && p.seed == 33,
          "the adapter must forward every sampling parameter");

    std::vector<std::vector<float>> packets;
    sv::sample_pair_rows(2, false, remote.size(), 0, {}, {},
        [&](std::size_t row) { check(row == 1, "follower read a greedy row"); return std::span(remote); },
        [&](auto send, auto receive) {
            packets.emplace_back(send.begin(), send.end());
            std::fill(receive.begin(), receive.end(), 0.0F);
        });
    check(packets.size() == 2 && packets[0].size() == 16384 && packets[1].size() == 2,
          "follower must exchange its complete slice in bounded frames");

    const auto run = [&](strix::TokenSampler& sampler, std::uint64_t position) {
        std::size_t packet = 0;
        const std::array<sv::PairSamplingRow, 2> choices{{{}, {&sampler, position}}};
        sv::sample_pair_rows(2, true, remote.size(), vocab, choices, ids,
            [&](std::size_t row) { check(row == 1, "lead read a greedy row"); return std::span(local); },
            [&](auto send, auto receive) {
                const auto& wire = packets.at(packet++);
                check(wire.size() == send.size(), "ranks disagree on frame width");
                std::copy(wire.begin(), wire.end(), receive.begin());
            });
        check(ids[0] == greedy && packet == packets.size(), "sampled row changed a greedy cohort member");
        check(ids[1] < vocab, "sampling selected padding");
        return ids[1];
    };
    strix::TokenSampler oracle(p, {});
    bool diverged = false;
    for (std::uint64_t pos = 32; pos < 64; ++pos) {
        const auto token = run(*first, pos);
        check(token == oracle.decide(full, pos, nullptr).token, "the lead did not sample the assembled row at its absolute position");
        check(token == run(*repeat, pos), "identical seeds must reproduce the same sequence");
        diverged |= token != run(*different, pos);
    }
    check(diverged, "different seeds must diverge on this fixture");
}
}  // namespace

int main() try {
    test_selection();
    std::puts("pair sampling: greedy IDs preserved, policy forwarded, same seeds matched, different seeds diverged");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
