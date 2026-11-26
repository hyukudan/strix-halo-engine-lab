#pragma once
// Charged re-probe episodes of the shared arbiter (ArbiterConfig::probe_loss): which engine runs them is model data.
// A model declares a value per backend in its TOML ([speculation.probe_loss], for example `cuda = 0.015`) only where a
// device receipt measured the episodes against the fixed re-probe interval with equal ids; the generated header
// (strix/model_data/<model>.hpp) carries it to the entry point, which passes it here. Every model and backend without
// a row keeps the fixed interval (0), the policy before the episodes, bit for bit.
// STRIX_SPEC_PROBE_LOSS (a value in [0, 1), 0 = the fixed interval) overrides every engine; command line and server
// construct their drafters through this one resolver.

#include "strix/env_flags.hpp"
#include "strix/spec/drafting.hpp"

#include <stdexcept>
#include <string>

namespace strix::spec {

inline constexpr const char* kProbeLossSwitch = "STRIX_SPEC_PROBE_LOSS";
// The value an entry point passes when its model's TOML holds no [speculation.probe_loss] row for the backend.
inline constexpr double kFixedReprobeInterval = 0.0;

struct ProbeLossChoice {
    double value = 0.0;
    std::string source;  // the switch, the model's measured row, or the fixed interval
};

// `model_default`: the model TOML's [speculation.probe_loss] value for this backend, or kFixedReprobeInterval.
[[nodiscard]] inline ProbeLossChoice resolve_probe_loss(double model_default) {
    if (!(model_default >= 0.0 && model_default < 1.0))
        throw std::invalid_argument("[speculation.probe_loss] must lie in [0, 1)");
    const double value = envcfg::number<double>(kProbeLossSwitch, model_default, 0.0, 1.0);
    // The arbiter takes [0, 1): refuse 1 here, naming the switch, instead of at drafter construction.
    if (!(value < 1.0)) throw std::invalid_argument(std::string(kProbeLossSwitch) + " must lie in [0, 1)");
    if (value != model_default) return {value, kProbeLossSwitch};
    return {value, model_default > 0.0 ? "measured default" : "fixed interval"};
}

inline ProbeLossChoice apply_probe_loss(ArbiterConfig& config, double model_default) {
    auto choice = resolve_probe_loss(model_default);
    config.probe_loss = choice.value;
    return choice;
}

}  // namespace strix::spec
