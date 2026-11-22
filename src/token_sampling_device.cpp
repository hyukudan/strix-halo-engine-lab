#include "strix/token_sampling_device.hpp"
#include "strix/token_sampling_timing.hpp"
#include "strix/env_flags.hpp"

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

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

DeviceSamplePlan plan_device_sample_enqueues(std::span<const DeviceSampleGroup> groups, std::uint32_t max_rows) {
    const auto fail = [](const char* why) { throw std::invalid_argument(std::string("device token sampling: ") + why); };
    if (max_rows == 0U) fail("zero rows per sequence");
    DeviceSamplePlan plan;
    for (const auto& g : groups) {
        if (g.sampler == nullptr) fail("group without a sampler");
        if (g.rows.empty() || g.rows.size() > max_rows) fail("group row count is invalid");
        plan.rows += static_cast<std::uint32_t>(g.rows.size());
        if (plan.rows > device_sample_max_rows) fail("too many rows in one call");
    }
    const auto n = static_cast<std::uint32_t>(groups.size());
    std::vector<bool> placed(n, false);
    plan.order.reserve(n);
    for (std::uint32_t g = 0; g < n; ++g) {
        if (placed[g]) continue;
        const auto& policy = groups[g].sampler->policy();
        for (std::uint32_t h = g; h < n; ++h)
            if (!placed[h] && same_token_transform(policy, groups[h].sampler->policy())) { placed[h] = true; plan.order.push_back(h); }
    }
    std::uint32_t offset = 0;
    for (std::uint32_t at = 0; at < n; ++at) {
        const auto g = plan.order[at];
        const auto rows = static_cast<std::uint32_t>(groups[g].rows.size());
        auto* open = plan.enqueues.empty() ? nullptr : &plan.enqueues.back();
        if (open == nullptr || open->rows + rows > max_rows ||
            !same_token_transform(groups[plan.order[open->first]].sampler->policy(), groups[g].sampler->policy())) {
            plan.enqueues.push_back({at, 0U, offset, 0U});
            open = &plan.enqueues.back();
        }
        ++open->groups;
        open->rows += rows;
        offset += rows;
    }
    return plan;
}

namespace {
struct SampleTimingTotals {
    std::mutex mutex;
    // [kind][rows 0..16, 0 = more]: calls, decided rows, readback seconds, decide seconds
    std::uint64_t calls[5][17]{}, decided[5][17]{};
    double readback[5][17]{}, decide[5][17]{};
    ~SampleTimingTotals() {
        static const char* names[5] = {"window", "serial", "draft", "kernels", "draft-kernels"};
        for (unsigned k = 0; k < 5U; ++k) for (unsigned r = 0; r < 17U; ++r) if (calls[k][r])
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
    const auto k = static_cast<unsigned>(kind), r = rows < 17U ? rows : 0U;
    const std::lock_guard lock(t.mutex);
    ++t.calls[k][r]; t.decided[k][r] += decided_rows; t.readback[k][r] += readback_seconds; t.decide[k][r] += decide_seconds;
}
} // namespace strix
