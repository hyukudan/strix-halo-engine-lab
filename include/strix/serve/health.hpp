#pragma once

#include "strix/decode_numerical_profile.hpp"
#include "strix/preflight.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/json.hpp"

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
    // Phase 1c only: preserve legacy HTTP success while real device proof is absent.
    bool interim_unchecked_http_success = false;
};
struct HealthResponse { int status; Json body; };
[[nodiscard]] bool serve_health_record_enabled() noexcept;
[[nodiscard]] HealthResponse render_health(const ServeHealthRecord&, const BackendHealth&,
    std::string_view service_state, std::string_view model, const QueueSnapshot&, std::string_view capabilities_id = {});
} // namespace strix::serve
