// Device sampling-v1 against the host transform: for random logit rows, policies, drafts and uniforms the
// device decision must equal decide_token(make_token_distribution(...)) (the device sums in a fixed parallel
// order and uses the device exp, so only a uniform within a few double ulps of a cumulative boundary could
// differ; none is expected over these draws).
#include "strix/hip_token_sampling_device.hpp"
#include "strix/token_sampling.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;

namespace {
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

// Logits shaped like a language model row: a few strong tokens, a long tail, some exact ties.
std::vector<float> make_logits(std::mt19937_64& rng, std::uint32_t vocab, float spread) {
    std::normal_distribution<float> tail(-4.0F, spread);
    std::vector<float> out(vocab);
    for (auto& v : out) v = tail(rng);
    std::uniform_int_distribution<std::uint32_t> pick(0, vocab - 1U);
    for (unsigned k = 0; k < 24U; ++k) out[pick(rng)] = 12.0F + static_cast<float>(k % 6U) * 0.75F;
    for (unsigned k = 0; k < 64U; ++k) out[pick(rng)] = out[pick(rng)]; // ties
    return out;
}
} // namespace

int main(int argc, char** argv) try {
    const unsigned cases = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 400U;
    std::mt19937_64 rng(0x5eedULL);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const TokenSamplingPolicy policies[] = {
        {1.0F, 0U, 1.0F, 0.0F, 0U}, {0.7F, 0U, 1.0F, 0.0F, 0U}, {0.6F, 0U, 1.0F, 0.1F, 0U}, {0.7F, 40U, 1.0F, 0.0F, 0U},
        {1.0F, 5U, 1.0F, 0.0F, 0U}, {1.0F, 0U, 0.8F, 0.0F, 0U}, {0.7F, 40U, 0.9F, 0.05F, 0U}, {2.0F, 0U, 0.95F, 0.0F, 0U},
        {0.3F, 0U, 1.0F, 0.0F, 0U}, {1.0F, 3000U, 1.0F, 0.0F, 0U}, {1.5F, 0U, 0.5F, 0.0F, 0U},
    };
    unsigned decided = 0, drafted = 0, accepted = 0, residual = 0, fallbacks = 0;
    for (unsigned c = 0; c < cases; ++c) {
        const auto& policy = policies[c % std::size(policies)];
        const std::uint32_t vocab = c % 3U == 0U ? 248320U : c % 3U == 1U ? 151936U : 4099U;
        const auto logits = make_logits(rng, vocab, c % 2U ? 2.0F : 3.5F);
        // Draft: the same transform over a perturbed row restricted to a token subset (like the draft vocabulary).
        TokenDraft draft;
        const bool with_draft = c % 4U != 0U;
        if (with_draft) {
            std::vector<std::uint32_t> ids;
            std::vector<float> values;
            std::normal_distribution<float> noise(0.0F, 0.8F);
            for (std::uint32_t t = 0; t < vocab; ++t)
                if (t % 3U != 1U) { ids.push_back(t); values.push_back(logits[t] + noise(rng)); }
            draft.position = 7U;
            draft.distribution = make_token_distribution(values, ids, policy);
            draft.token = draft.distribution.sample(uniform(rng));
        }
        const double u_plain = uniform(rng), u_accept = uniform(rng), u_residual = uniform(rng);
        const auto host = decide_token(make_token_distribution(logits, {}, policy), with_draft ? &draft : nullptr,
            u_plain, with_draft ? u_accept : 0.0, with_draft ? u_residual : 0.0);
        bool fallback = false;
        const auto device = hip_decide_token_for_testing(logits, policy, with_draft ? &draft : nullptr, u_plain,
            with_draft ? u_accept : 0.0, with_draft ? u_residual : 0.0, &fallback);
        fallbacks += fallback ? 1U : 0U;
        require(device.token == host.token && device.drafted == host.drafted && device.accepted == host.accepted &&
                device.residual == host.residual,
            "device decision differs from the host transform in case " + std::to_string(c) + ": token " +
                std::to_string(device.token) + " vs " + std::to_string(host.token));
        ++decided; drafted += host.drafted; accepted += host.accepted; residual += host.residual;
    }
    // A non-finite logit is rejected as on the host.
    {
        std::vector<float> bad(4099U, 0.0F);
        bad[17] = NAN;
        bool rejected = false;
        try { (void)hip_decide_token_for_testing(bad, policies[0], nullptr, 0.5, 0.0, 0.0); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "device sampling accepted a non-finite logit");
    }
    std::cout << "device sampling-v1: " << decided << " decisions equal the host (" << drafted << " drafted, " << accepted
              << " accepted, " << residual << " residual, " << fallbacks << " host top-p fallbacks)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "device sampling test failed: " << error.what() << '\n';
    return 1;
}
