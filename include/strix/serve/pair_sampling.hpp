#pragma once

#include "strix/serve/engine.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::serve {

// The follower replays the lead's seeds only while every request shares identity 1 (the shared mode switch).
inline bool pair_sampling_replays_seeds() { return !token_sampling_request_identity_counts(); }

// A per-request sampler held by value in the pair schedulers (the shared translation, strix/serve/job_sampling.hpp).
inline std::optional<TokenSampler> make_pair_sampler(const GenerationJob& job, const TokenSamplerOptions& options = {}) {
    const auto shared = make_job_sampler(job.sampling, options);
    if (!shared) return std::nullopt;
    return std::optional<TokenSampler>(std::in_place, shared->policy(), shared->options());
}

struct PairSamplingRow {
    TokenSampler* sampler = nullptr;
    std::uint64_t position = 0;
};

// The follower sends its vocabulary slice; only the lead assembles the row and samples it.
// The full-duplex control transports need equal message lengths, so the lead sends padding.
// A zero follower width is the one-node route. A zero mask leaves the greedy picks in place.
template<class Read, class Exchange>
void sample_pair_rows(std::uint32_t mask, bool lead, std::size_t follower_width, std::size_t vocabulary,
                      std::span<const PairSamplingRow> choices, std::span<std::uint32_t> tokens,
                      Read&& read, Exchange&& exchange) {
    if (mask == 0) return;
    constexpr std::size_t chunk_floats = 16384;  // 64 KiB control-frame limit
    std::vector<float> scratch(std::min(chunk_floats, follower_width));
    std::vector<float> full;
    for (std::size_t row = 0; mask != 0; ++row, mask >>= 1) {
        if ((mask & 1U) == 0) continue;
        const std::span<const float> local = read(row);
        if (lead) {
            if (row >= choices.size() || row >= tokens.size() || !choices[row].sampler)
                throw std::logic_error("pair sampling: missing lead sampler");
            full.resize(local.size() + follower_width);
            std::copy(local.begin(), local.end(), full.begin());
        } else if (local.size() != follower_width) {
            throw std::logic_error("pair sampling: wrong follower vocabulary slice");
        }
        for (std::size_t begin = 0; begin < follower_width; begin += chunk_floats) {
            const std::size_t count = std::min(chunk_floats, follower_width - begin);
            auto buffer = std::span(scratch).first(count);
            if (lead) exchange(std::span<const float>(buffer), std::span(full).subspan(local.size() + begin, count));
            else exchange(local.subspan(begin, count), buffer);
        }
        if (lead) {
            const auto logits = std::span<const float>(full).first(std::min(vocabulary, full.size()));
            tokens[row] = choices[row].sampler->decide(logits, choices[row].position, nullptr).token;
        }
    }
}

}  // namespace strix::serve
