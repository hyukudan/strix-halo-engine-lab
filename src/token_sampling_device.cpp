#include "strix/token_sampling_device.hpp"
#include "strix/token_sampling_timing.hpp"
#include "strix/env_flags.hpp"

#include <cstdio>
#include <mutex>

namespace strix {

TokenSamplingPath token_sampling_path(const TokenSamplingPolicy& policy) noexcept {
    // -1 auto, else the forced path.
    static const int forced = [] {
        // unknown STRIX_SAMPLE_PATH throws (std::terminate here: noexcept); 2 = auto (also unset)
        const std::size_t path = strix::env_choice("STRIX_SAMPLE_PATH", 2, {"device", "host", "auto"});
        if (strix::env_flag_default_off("STRIX_SAMPLE_HOST") || path == 1) return 1;
        if (path == 0) return 0;
        return -1;
    }();
    if (forced >= 0) return forced ? TokenSamplingPath::host : TokenSamplingPath::device;
    static_cast<void>(policy);
    return TokenSamplingPath::device;
}
bool device_draft_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {
    static const bool draft_host = strix::env_flag_default_off("STRIX_SAMPLE_DRAFT_HOST");
    return !draft_host && device_token_sampling_enabled(policy);
}

namespace {
struct SampleTimingTotals {
    std::mutex mutex;
    // [kind][rows 0..8]: calls, decided rows, readback seconds, decide seconds
    std::uint64_t calls[5][9]{}, decided[5][9]{};
    double readback[5][9]{}, decide[5][9]{};
    ~SampleTimingTotals() {
        static const char* names[5] = {"window", "serial", "draft", "kernels", "draft-kernels"};
        for (unsigned k = 0; k < 5U; ++k) for (unsigned r = 0; r < 9U; ++r) if (calls[k][r])
            std::fprintf(stderr, "sample-timing %s rows=%u: %llu calls, %llu decided rows, readback %.1f us/call, decide %.1f us/call\n",
                names[k], r, static_cast<unsigned long long>(calls[k][r]), static_cast<unsigned long long>(decided[k][r]),
                readback[k][r] * 1e6 / static_cast<double>(calls[k][r]), decide[k][r] * 1e6 / static_cast<double>(calls[k][r]));
    }
};
SampleTimingTotals& sample_timing_totals() { static SampleTimingTotals totals; return totals; }
} // namespace
bool sample_timing_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_SAMPLE_TIMING", false);
    return on;
}
void record_sample_timing(SampleTimingKind kind,std::uint32_t rows,double readback_seconds,double decide_seconds,
    std::uint32_t decided_rows) {
    auto& t = sample_timing_totals();
    const auto k = static_cast<unsigned>(kind), r = rows < 9U ? rows : 0U;
    const std::lock_guard lock(t.mutex);
    ++t.calls[k][r]; t.decided[k][r] += decided_rows; t.readback[k][r] += readback_seconds; t.decide[k][r] += decide_seconds;
}
} // namespace strix
