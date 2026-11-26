#pragma once
// Charged re-probe episodes of the shared arbiter (ArbiterConfig::probe_loss): which engine runs them is data.
// A row exists only where a device receipt measured the episodes against the fixed re-probe interval with equal ids;
// every other engine and backend keeps the fixed interval (0), the policy before the episodes, bit for bit.
// STRIX_SPEC_PROBE_LOSS (0..1, 0 = the fixed interval) overrides every engine; command line and server construct
// their drafters through this one resolver.

#include "strix/env_flags.hpp"
#include "strix/spec/drafting.hpp"

#include <string>
#include <string_view>

namespace strix::spec {

inline constexpr double kMeasuredProbeLoss = 0.015;
inline constexpr const char* kProbeLossSwitch = "STRIX_SPEC_PROBE_LOSS";

struct ProbeLossDefault {
    std::string_view engine;   // model registry key
    std::string_view backend;  // cuda | hip | cpu
    double probe_loss = 0.0;
    std::string_view receipt;
};

inline constexpr ProbeLossDefault probe_loss_defaults[] = {
    {"inkling2-mini", "cuda", kMeasuredProbeLoss,
     "RTX CLI EXACT, 256 tokens, ids equal: code 130.2 / 131.4 tok/s vs 116.0 with the fixed interval (90.8 with "
     "speculation off); prose 91.3 / 91.2 vs 94.8 with speculation off (0.963)"},
};

[[nodiscard]] constexpr double table_probe_loss(std::string_view engine, std::string_view backend) noexcept {
    for (const auto& row : probe_loss_defaults)
        if (row.engine == engine && row.backend == backend) return row.probe_loss;
    return 0.0;
}

struct ProbeLossChoice {
    double value = 0.0;
    std::string source;  // the switch, the receipt row, or the fixed interval
};

[[nodiscard]] inline ProbeLossChoice resolve_probe_loss(std::string_view engine, std::string_view backend) {
    const double table = table_probe_loss(engine, backend);
    const double value = envcfg::number<double>(kProbeLossSwitch, table, 0.0, 1.0);
    if (value != table) return {value, kProbeLossSwitch};
    return {value, table > 0.0 ? "measured default" : "fixed interval"};
}

inline ProbeLossChoice apply_probe_loss(ArbiterConfig& config, std::string_view engine, std::string_view backend) {
    auto choice = resolve_probe_loss(engine, backend);
    config.probe_loss = choice.value;
    return choice;
}

}  // namespace strix::spec
