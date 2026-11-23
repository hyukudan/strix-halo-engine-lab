#pragma once

#include "strix/engine_profile.hpp"
#include "strix/lever.hpp"

#include <cmath>
#include <optional>

namespace strix::profile {

enum class GateVerdict { pass, unresolved, fail, hold };

struct ClassVerdict {
    std::string_view content_class, band;
    GateVerdict verdict;
    std::uint32_t rows;
};

struct GateRecord {
    std::string_view lever, model, gate_set;
    GateVerdict verdict;
    std::span<const ClassVerdict> classes;
    Name tier = Name::fast;
};

struct GateScope {
    std::string_view model, gate_set, content_class, band;
};

// A scope without a content label needs a final verdict for every declared class and band.
[[nodiscard]] inline std::optional<GateVerdict> gate_verdict(std::string_view lever, const GateScope& scope,
                                                            Name tier, std::span<const GateRecord> gates) {
    const GateRecord* match = nullptr;
    for (const auto& record : gates) {
        if (record.lever != lever || record.model != scope.model || record.gate_set != scope.gate_set ||
            record.tier != tier) continue;
        if (match) throw std::invalid_argument("duplicate gate record for " + std::string(lever));
        match = &record;
    }
    if (!match) return std::nullopt;
    if (match->verdict != GateVerdict::pass) return match->verdict;

    const bool pooled_scope = scope.content_class.empty() && scope.band.empty();
    bool found = false;
    auto verdict = GateVerdict::pass;
    for (const auto& row : match->classes) {
        if (!pooled_scope && (row.content_class != scope.content_class || row.band != scope.band)) continue;
        if (found && !pooled_scope)
            throw std::invalid_argument("duplicate content class and band for " + std::string(lever));
        found = true;
        if (row.verdict == GateVerdict::hold || row.verdict == GateVerdict::fail) verdict = row.verdict;
        if (verdict == GateVerdict::pass && row.rows < 800U) verdict = GateVerdict::unresolved;
        if (verdict == GateVerdict::pass && row.verdict == GateVerdict::unresolved) verdict = GateVerdict::unresolved;
    }
    return found ? verdict : GateVerdict::unresolved;
}

struct RegimeRecord {
    std::string_view lever, device;
    lever::Regime regime;
    double gain; // Relative to EXACT for FAST, relative to FAST for TURBO.
    std::string_view receipt;
};

[[nodiscard]] inline const RegimeRecord* measured_regime(std::string_view name, std::string_view device,
                                                        const lever::Inputs& inputs,
                                                        std::span<const RegimeRecord> regimes) {
    for (const auto& record : regimes) {
        if (record.lever != name || record.device != device || record.receipt.empty() ||
            !std::isfinite(record.gain) || record.gain <= 0) continue;
        const lever::Decl decl{.name = name, .default_on = true, .regime = record.regime};
        if (lever::automatic(decl, inputs, true).on) return &record;
    }
    return nullptr;
}

} // namespace strix::profile
