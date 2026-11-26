#pragma once

#include <cmath>
#include <limits>

namespace strix::pair::planner {

// Compare the same workload and numerical class. A positive infinity baseline is
// reserved for a workload that cannot fit on one device. Zero denotes missing evidence.
struct DeviceGain {
    double single_ms = 0.0;
    double multi_ms = 0.0;
    double multi_uncertainty_ms = 0.0;

    [[nodiscard]] double ratio() const noexcept {
        return std::isfinite(single_ms) && single_ms > 0.0 && std::isfinite(multi_ms) && multi_ms > 0.0
            ? single_ms / multi_ms : 0.0;
    }

    [[nodiscard]] bool clears(double margin = 0.0) const noexcept {
        const double upper = multi_ms + multi_uncertainty_ms;
        if (!std::isfinite(upper) || multi_ms <= 0.0 || multi_uncertainty_ms < 0.0 ||
            !std::isfinite(margin) || margin < 0.0 || margin >= 1.0)
            return false;
        if (single_ms == std::numeric_limits<double>::infinity()) return true;
        return std::isfinite(single_ms) && single_ms > 0.0 && upper < single_ms * (1.0 - margin);
    }
};

}  // namespace strix::pair::planner
