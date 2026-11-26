#pragma once

#include <optional>
#include <span>

namespace strix::exec {

// Sampling a single logits row and verifying a window under a sampling rule
// are separate contracts. A tapped step can maintain a drafter's context
// without implementing verification itself.
struct DecodeCapabilities {
    bool verifier{};
    bool taps{};
    bool sampling{};
    bool sampled_verifier{};

    [[nodiscard]] constexpr bool provides(DecodeCapabilities required) const {
        return (!required.verifier || verifier) && (!required.taps || taps) &&
               (!required.sampling || sampling) &&
               (!required.sampled_verifier || (verifier && sampling && sampled_verifier));
    }
};

template <class Route>
struct DecodeCandidate {
    Route route;
    DecodeCapabilities capabilities;
};

// The binding supplies candidates in qualified preference order. Failure is
// explicit so a forced request cannot silently lose part of its contract.
template <class Route, std::size_t Extent>
[[nodiscard]] constexpr std::optional<Route> resolve_decode_route(
    std::span<const DecodeCandidate<Route>, Extent> candidates, DecodeCapabilities required) {
    for (const auto& candidate : candidates)
        if (candidate.capabilities.provides(required)) return candidate.route;
    return std::nullopt;
}

}  // namespace strix::exec
