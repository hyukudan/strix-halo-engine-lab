#pragma once

#include "strix/model_runtime_binding.hpp"
#include "strix/serial_request.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>

namespace strix {
namespace detail { struct GreedyServingJob; }

struct ServingCompatibility {
    ModelContentIdentity model_content;
    std::uint64_t numerical_identity = 0;
    int device_ordinal = 0;
    friend bool operator==(const ServingCompatibility&, const ServingCompatibility&) = default;
};
struct GreedyServingPolicy {
    // Independent C1 sessions on owned streams, not a cross-request tensor batch.
    std::uint32_t concurrency = 1;
    std::size_t max_retained_requests = 64;
    std::uint64_t reservation_budget_bytes = 0;
    std::uint64_t shared_reservation_bytes = 0;
    std::uint64_t prefill_chunk_rows = 2;
    std::uint32_t max_decode_burst = 8;
    std::uint64_t aging_ns = 1000000;
    std::uint64_t queue_timeout_ns = 0; // zero disables the queue timeout
};
enum class ServingRequestState { queued, running, completed, cancelled, failed, timed_out };
enum class ServingAdmission { accepted, request_limit, reservation_budget, duplicate_session, incompatible };
struct ServingRequestMetrics {
    // Host steady-clock intervals include backend completion/selection delivery.
    // Service durations across requests overlap; they are not GPU busy time.
    std::uint64_t queue_wait_ns = 0, service_ns = 0, wall_ns = 0;
    std::uint64_t chunks = 0;
    std::optional<std::uint64_t> time_to_first_token_ns;
    bool saturated = false;
};
class GreedyServingScheduler;
class ServingRequestHandle final {
public:
    ServingRequestHandle() = default;
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(job_); }
    [[nodiscard]] ServingRequestState state() const;
    [[nodiscard]] bool cancel() const noexcept;
    // Available only after a terminal state, while this handle retains owners.
    [[nodiscard]] const SerialGreedyRequestResult& result() const;
    [[nodiscard]] ServingRequestMetrics metrics() const;
    [[nodiscard]] std::exception_ptr failure() const;
private:
    explicit ServingRequestHandle(std::shared_ptr<detail::GreedyServingJob> job);
    std::shared_ptr<detail::GreedyServingJob> job_;
    friend class GreedyServingScheduler;
};
struct ServingSubmission {
    ServingAdmission admission = ServingAdmission::request_limit;
    ServingRequestHandle request;
};

// Serialized chunks within each session; independent sessions may overlap on
// two/four worker threads and their owned GPU streams. Backend calls must fence,
// validate and commit their own step before returning. A failed/cancelled request
// never releases an in-flight backend. Keep actual_owner alive for every borrowed
// session/arena/backend reference; the shared model owner is retained separately.
// Reservation bytes are declared maximum additional per-request allocations,
// including response growth/snapshots/scratch, not measured physical residency.
// Completed handles continue charging their reservation until released.
class GreedyServingScheduler final {
public:
    using BackendFactory = std::function<std::unique_ptr<SerialGreedyStepBackend>()>;
    GreedyServingScheduler(GreedyServingPolicy policy, ServingCompatibility compatibility,
        std::shared_ptr<const void> shared_model_owner);
    ~GreedyServingScheduler();
    GreedyServingScheduler(const GreedyServingScheduler&) = delete;
    GreedyServingScheduler& operator=(const GreedyServingScheduler&) = delete;
    [[nodiscard]] ServingSubmission submit(SerialGreedyRequest request,
        ServingCompatibility compatibility, std::uintptr_t session_identity,
        std::uint64_t allocation_epoch, std::uint64_t reservation_bytes,
        std::shared_ptr<const void> actual_owner, BackendFactory factory);
    // C1 executes inline without a worker or batch-fill wait. C2/C4 wait for
    // queued/setup/running work to finish; holding completed handles is allowed.
    // Never call this blocking method from a backend/factory callback. Destruction
    // must not race client method calls; it cancels and joins its own workers.
    void run_until_idle();
    [[nodiscard]] std::uint64_t reserved_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
