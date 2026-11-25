#pragma once

#include "strix/engine_profile.hpp"
#include <charconv>

namespace strix::spec {
inline constexpr std::uint32_t kCandidateBlockRows = 8;
inline constexpr std::string_view kBlockDraftLeverName = "spec.block-candidate@1";
inline constexpr profile::Lever kBlockDraftLever{
    kBlockDraftLeverName, "STRIX_BLOCK_DRAFTER", "--block-drafter", {"", "", "", ""}, "off",
    profile::Kind::exact, {}, {}, "", "off",
    "host reference only; native verify identity, model fidelity and complete workload receipts pending",
    false, {}, true};
struct BlockDraftSelection {
    std::string drafter;
    std::uint32_t block{};
    bool sampled{};
    bool enabled() const { return !drafter.empty(); }
};
// Used after the ONE profile resolver, by CLI and server. Values:
// off | dflash2:8:greedy | dflash2:8:sampled. No device is enabled by this parser.
inline BlockDraftSelection block_draft_selection(const profile::Resolution& resolution) {
    const auto* entry = resolution.find(kBlockDraftLeverName);
    if (!entry || entry->value == "off") return {};
    const std::string_view value(entry->value);
    const auto first = value.find(':');
    const auto second = first == value.npos ? value.npos : value.find(':', first + 1);
    if (first == value.npos || second == value.npos || value.substr(0, first) != "dflash2")
        throw std::invalid_argument("--block-drafter expects off or dflash2:8:greedy|sampled");
    BlockDraftSelection out;
    out.drafter = value.substr(0, first);
    const auto digits = value.substr(first + 1, second-first-1);
    const auto parsed = std::from_chars(digits.data(), digits.data()+digits.size(), out.block);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data()+digits.size() || out.block != kCandidateBlockRows)
        throw std::invalid_argument("block drafter: this checkpoint contract requires block 8");
    const auto rule = value.substr(second+1);
    if (rule != "greedy" && rule != "sampled") throw std::invalid_argument("block drafter: unknown verification rule");
    out.sampled = rule == "sampled";
    if (out.sampled && resolution.profile != profile::Name::fast && resolution.profile != profile::Name::turbo)
        throw std::invalid_argument("block drafter: EXACT permits only greedy-identical verification");
    return out;
}
// A pending sampled contract is not executable, even under FAST. A name in a
// profile receipt is never permission to silently use the old row sampler.
inline void require_block_draft_execution(const BlockDraftSelection& selection, bool native_bound = false) {
    if (!selection.enabled()) return;
    if (selection.sampled) throw std::invalid_argument("block drafter: sampled rejection hook unqualified and unbound");
    if (!native_bound) throw std::invalid_argument("block drafter: native forward and verify identity are not bound");
}
} // namespace strix::spec
