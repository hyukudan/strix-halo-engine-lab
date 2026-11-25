#include "strix/serial_request.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/token_sampling.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace strix {
namespace {

void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

using serve::make_job_sampler;

constexpr std::array logits{0.75F, 1.0F, 1.0F, 0.875F, 0.8F};

std::array<std::uint32_t, 64> decisions(strix::TokenSampler* sampler) {
    std::array<std::uint32_t, 64> ids{};
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        ids[pos] = sampler ? sampler->decide(logits, pos, nullptr).token
                           : strix::select_greedy_token(logits, static_cast<std::uint32_t>(logits.size()));
    }
    return ids;
}

} // namespace
} // namespace strix

int main(int argc, char** argv) try {
    using namespace strix;
    if (argc == 2 && std::string_view(argv[1]) == "counter") {
        // The arrival-order identity is retired: the server must refuse it, never draw a queue-dependent stream.
        bool refused = false;
        try { (void)strix::next_token_sampling_request_identity(); } catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "STRIX_SAMPLING_REQUEST_IDENTITY=counter is refused");
        std::cout << "Qwen job sampling: counter identity refused\n";
        return 0;
    }
    const auto initial_identity = strix::next_token_sampling_request_identity();
    expect(initial_identity == 1, "initial request identity");

    strix::serve::GenerationJob job;
    job.sampling.top_k = 4;
    job.sampling.top_p = 0.9;
    job.sampling.min_p = 0.125;
    job.sampling.seed = 42;
    job.sampling.seeded = true;
    std::string error = "stale";
    const auto greedy = make_job_sampler(job.sampling, error, SampledAcceptance::coupled);
    expect(!greedy && error.empty(), "temperature zero bypasses sampler creation");
    const auto greedy_ids = decisions(nullptr);
    expect(greedy_ids.front() == 1U && decisions(greedy.get()) == greedy_ids, "temperature zero matches greedy IDs");

    job.sampling.temperature = 0.75;
    auto invalid = job.sampling;
    invalid.top_p = 0.0;
    expect(!make_job_sampler(invalid, error, SampledAcceptance::coupled) && error.find("top-p") != std::string::npos,
           "invalid policy produces a request-local error");

    const auto sampler = make_job_sampler(job.sampling, error, SampledAcceptance::coupled);
    expect(sampler && error.empty(), "valid request produces a sampler after a refusal");
    const auto& policy = sampler->policy();
    expect(policy.temperature == static_cast<float>(job.sampling.temperature) && policy.top_k == job.sampling.top_k &&
           policy.top_p == static_cast<float>(job.sampling.top_p) && policy.min_p == static_cast<float>(job.sampling.min_p) &&
           policy.seed == job.sampling.seed, "adapter forwards the complete job sampling policy");
    expect(sampler->coupled(), "predictor drafts preserve the target draw");
    expect(sampler->options().request_identity == 1U, "adapter obtains the shared request identity");

    const auto repeat = make_job_sampler(job.sampling, error, SampledAcceptance::coupled);
    expect(repeat && error.empty(), "repeat request produces an independent sampler");
    expect(decisions(sampler.get()) == decisions(repeat.get()), "same seed and policy reproduce the complete IDs");
    ++job.sampling.seed;
    const auto different = make_job_sampler(job.sampling, error, SampledAcceptance::coupled);
    expect(different && error.empty() && decisions(sampler.get()) != decisions(different.get()),
           "different seeds produce different IDs");
    std::cout << "Qwen job sampling: greedy parity, policy and seeded IDs passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Qwen job sampling: " << e.what() << '\n';
    return 1;
}
