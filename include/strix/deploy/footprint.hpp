#pragma once

#include "strix/deploy/spec.hpp"
#include <utility>
#include <vector>

namespace strix::deploy {

struct Footprint {
    std::uint64_t weights = 0, kv_per_token = 0, kv_total = 0;
    std::uint64_t workspace = 0, staging = 0;
    std::vector<std::pair<std::string, std::uint64_t>> extra;
    // kv_total is the complete reservation when nonzero; otherwise context * slots * kv_per_token.
};
struct Candidate {
    std::string topology;
    Format format;
    std::uint32_t context = 0, slots = 0, ranks = 0;
};

// A provider advertises only implemented choices, ordered by preference. It must
// account for every live and peak allocation without loading model weights or devices.
class FootprintProvider {
public:
    virtual ~FootprintProvider() = default;
    [[nodiscard]] virtual std::string model() const = 0;
    [[nodiscard]] virtual std::vector<std::string> topologies() const = 0;
    [[nodiscard]] virtual std::vector<Format> formats(const std::string& topology) const = 0;
    [[nodiscard]] virtual std::uint32_t max_slots() const { return 1; }
    [[nodiscard]] virtual Footprint footprint(const Spec&, const Candidate&, std::uint32_t rank) const = 0;
};

} // namespace strix::deploy
