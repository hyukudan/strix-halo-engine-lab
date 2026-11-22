#include "../apps/cuda_ds41_replica_serve.hpp"
#include "../apps/cuda_qwen_serve_policy.hpp"
#include "strix/cuda/qwen_cohort.hpp"
#include "strix/head_argmax.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using cuda_qwen_serve::decide_on_host;
using cuda_qwen_serve::make_job_sampler;

constexpr std::array<float, 6> logits{-2.0F, 1.0F, 1.0F, 0.9F, 0.8F, 0.7F};
constexpr std::uint64_t prompt_tokens = 7;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void test_greedy() {
    strix::serve::GenerationJob job;
    job.sampling = {0.0, 0.5, 3, 0.125, 51, true};
    auto sampler = make_job_sampler(job);
    expect(!sampler, "temperature 0 bypasses sampling even when filters are present");
    const auto greedy = strix::head_argmax::reference(logits).index;
    expect(greedy == 1U, "device argmax contract chooses the lowest ID on a tie");
    const auto id = sampler ? decide_on_host(*sampler, logits, prompt_tokens) : greedy;
    expect(id == greedy, "temperature 0 yields the greedy ID on the same row");
}

void test_sampled() {
    strix::serve::GenerationJob first;
    first.sampling = {0.75, 0.875, 4, 0.125, 51, true};
    auto second = first;
    second.sampling.seed = 91;
    auto a = make_job_sampler(first);
    auto b = make_job_sampler(second);
    auto b_reordered = make_job_sampler(second);
    auto a_reordered = make_job_sampler(first);
    if (!a || !b || !b_reordered || !a_reordered) {
        expect(false, "positive temperature constructs one sampler per request");
        return;
    }
    const auto& policy = a->policy();
    expect(policy.temperature == 0.75F && policy.top_k == 4U && policy.top_p == 0.875F &&
               policy.min_p == 0.125F && policy.seed == 51U && b->policy().seed == 91U,
           "job fields are forwarded to the shared sampler");
    strix::TokenSampler reference({0.75F, 4U, 0.875F, 0.125F, 51U}, strix::TokenSamplerOptions{});
    bool differ = false;
    for (std::uint64_t pos = prompt_tokens; pos < prompt_tokens + 64U; ++pos) {
        const auto id_a = decide_on_host(*a, logits, pos);
        const auto id_b = decide_on_host(*b, logits, pos);
        expect(id_a == reference.decide(logits, pos, nullptr).token,
               "the final row and absolute position drive the shared decision");
        expect(id_b == decide_on_host(*b_reordered, logits, pos) &&
                   id_a == decide_on_host(*a_reordered, logits, pos),
               "repeated seeds reproduce independently of admission and decision order");
        differ = differ || id_a != id_b;
    }
    expect(differ, "distinct seeds produce distinct streams");
}

void test_replica_sampling() {
    namespace replica = strix::ds41::replica_serve;
    const std::vector<std::string> plain{"--phases", "chats0", "--chat-steps", "1"};
    auto zero = plain;
    zero.insert(zero.end(), {"--temperature", "0", "--top-k", "4", "--seed", "51"});
    expect(!replica::take_sampling_args(zero) && zero == plain, "replica temperature zero uses the unchanged greedy request");
    replica::check_request_args(zero);

    auto args = plain;
    args.insert(args.end(), {"--temperature", "0.75", "--top_k=4", "--top-p", "0.875", "--min_p", "0.125", "--seed", "51"});
    auto repeated_args = args;
    auto different_args = args;
    different_args.back() = "91";
    auto sampler = replica::take_sampling_args(args);
    auto repeated = replica::take_sampling_args(repeated_args);
    auto different = replica::take_sampling_args(different_args);
    expect(args == plain && repeated_args == plain && different_args == plain,
           "sampling flags are separated from the remaining spool args");
    if (!sampler || !repeated || !different) {
        expect(false, "the spool creates one shared sampler per sampled request");
        return;
    }
    const auto& policy = sampler->policy();
    expect(policy.temperature == 0.75F && policy.top_k == 4U && policy.top_p == 0.875F &&
               policy.min_p == 0.125F && policy.seed == 51U, "spool args reach the shared sampler policy");
    expect(replica::sampling_receipt_fields(policy) ==
               ",\"sampling\":\"sampling-v1\",\"sampling_policy\":{\"temperature\":0.75,\"top_k\":4,\"top_p\":0.875,\"min_p\":0.125,\"seed\":51}",
           "replica records the actual policy and identity");

    std::array<float, logits.size() * 2U> window{100.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::copy(logits.begin(), logits.end(), window.begin() + logits.size());
    strix::TokenSampler reference(policy, strix::TokenSamplerOptions{});
    bool differ = false;
    for (std::uint64_t pos = prompt_tokens; pos < prompt_tokens + 64U; ++pos) {
        const auto token = replica::sample_final_row(*sampler, window, 2, pos);
        expect(token == reference.decide(logits, pos, nullptr).token, "replica selects on the final row, at the absolute position");
        expect(token == replica::sample_final_row(*repeated, window, 2, pos), "replica replays the same seed");
        differ = replica::sample_final_row(*different, window, 2, pos) != token || differ;
    }
    expect(differ, "replica different seeds differ");
    expect(replica::sampled_chat_phase("schats0b2f3w") == "chats0b2" && replica::sampled_chat_phase("dchats0") == "dchats0",
           "sampled chat phases use plain decode");
    try {
        (void)replica::sampled_chat_phase("spec0");
        expect(false, "greedy qualification phases must be refused");
    } catch (const replica::SamplingReject& ex) {
        expect(std::string_view(ex.what()).starts_with("--phases"), "unsupported phases name --phases");
    }
    try {
        replica::check_request_args({"--window", "fixed"}, true);
        expect(false, "sampled requests refuse explicit draft controls");
    } catch (const replica::SamplingReject& ex) {
        expect(std::string_view(ex.what()).starts_with("--window"), "unsupported control names --window");
    }
    for (const std::string field : {"--temperature", "--top-k", "--top-p", "--min-p", "--seed"}) {
        std::vector<std::string> malformed{field, "1oops"};
        try {
            (void)replica::take_sampling_args(malformed);
            expect(false, "replica rejects a malformed sampling field");
        } catch (const replica::SamplingReject& ex) {
            expect(std::string_view(ex.what()).starts_with(field), "replica reports the malformed field");
        }
    }
}

}  // namespace

int main() {
    test_greedy();
    test_sampled();
    test_replica_sampling();
    return failures == 0 ? 0 : 1;
}
