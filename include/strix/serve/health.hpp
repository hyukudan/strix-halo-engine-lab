#pragma once

#include "strix/decode_numerical_profile.hpp"
#include "strix/preflight.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/json.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>

namespace strix::serve {

enum class BackendState { starting, responsive, probe_pending, unresponsive, failed, draining, unchecked };
enum class BackendProof { none, control_only, work_completion, idle_scratch_event };
struct BackendHealth {
    BackendState state = BackendState::unchecked;
    BackendProof proof = BackendProof::none;
    std::optional<double> probe_age_seconds, last_completion_age_seconds;
    std::optional<std::uint64_t> required_ranks, responsive_ranks;
};
// Providers must return bounded cached evidence, without runner locks or device calls.
// A control-only ACK never establishes device readiness.
class BackendProbe {
public:
    virtual ~BackendProbe() = default;
    [[nodiscard]] virtual BackendHealth snapshot() const = 0;
};
// Kept generic so CPU tests can spy on the exact no-request owner path.
namespace detail {
template<class Counter>
inline std::uint64_t probe_generation(const Counter& requested, std::uint64_t seen) noexcept {
    const auto generation = requested.load(std::memory_order_relaxed);
    return generation != seen ? generation : 0;
}
} // namespace detail
// The sole backend owner calls boundary(). Readers never wait on an owner lock.
// One coalesced generation may be outstanding; all evidence storage is fixed size.
// Generations never wrap: exhaustion latches failure before zero can be published.
class BackendProbeSlot final : public BackendProbe {
public:
    using Clock = std::function<std::int64_t()>; // steady nanoseconds; injectable for CPU tests
    using Pause = std::function<void()>;
    explicit BackendProbeSlot(std::function<void()> wake = {}, Clock clock = {}, Pause pause = {});
    [[nodiscard]] BackendHealth snapshot() const override;
    void boundary(BackendProof proof) noexcept {
        if (const auto generation = pending_generation()) acknowledge(generation, proof);
    }
    // Batch owner samples before step(), then publishes after it returns. A request
    // arriving during an unrequested step waits for the next completed step.
    [[nodiscard]] std::uint64_t pending_generation() const noexcept {
        return detail::probe_generation(requested_, owner_generation_);
    }
    void complete(std::uint64_t generation, BackendProof proof) noexcept { acknowledge(generation, proof); }
    void fail() noexcept { failed_.store(true, std::memory_order_release); }
private:
    void acknowledge(std::uint64_t generation, BackendProof proof) noexcept;
    [[nodiscard]] BackendHealth cached(std::int64_t now) const;
    std::function<void()> wake_;
    Clock clock_;
    Pause pause_;
    mutable std::atomic<std::uint64_t> requested_{0};
    std::atomic<std::uint64_t> acknowledged_{0};
    // Timestamp and proof share one atomic word, so readers cannot mix publications.
    std::atomic<std::uint64_t> stamp_{0};
    mutable std::atomic<bool> failed_{false};
    std::uint64_t owner_generation_ = 0; // sole owner only
};
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
using HealthProvider = std::function<BackendHealth()>;

// Construct once at startup, then publish through shared_ptr<const ServeHealthRecord>.
// JSON inputs below are reserialized through explicit field/value allowlists.
struct ServeHealthRecord {
    Json build_info, asset_verification;
    std::string build_info_sha256;
    std::optional<std::uint64_t> context_tokens, slots, prefill_tile_rows;
    std::optional<DecodeNumericalProfile> decode_profile, prefill_profile;
    std::string draft_profile;
    bool prefix_enabled = false, prefix_disk_enabled = false;
    std::optional<std::uint64_t> prefix_host_bytes, prefix_disk_bytes;
    std::optional<PreflightObservation> preflight;
    std::optional<PreflightAssessment> preflight_assessment;
    bool backend_probe_enabled = true;
    // Preserve HTTP reachability success while backend proof is pending.
    bool interim_unchecked_http_success = false;
};
struct HealthResponse { int status; Json body; };
[[nodiscard]] bool serve_health_record_enabled() noexcept;
[[nodiscard]] HealthResponse render_health(const ServeHealthRecord&, const BackendHealth&,
    std::string_view service_state, std::string_view model, const QueueSnapshot&, std::string_view capabilities_id = {});
} // namespace strix::serve
