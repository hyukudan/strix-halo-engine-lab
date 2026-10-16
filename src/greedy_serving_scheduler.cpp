#include "strix/greedy_serving_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace strix {
namespace {
using Clock = std::chrono::steady_clock;
bool terminal(ServingRequestState state) noexcept {
    return state != ServingRequestState::queued && state != ServingRequestState::running;
}
std::uint64_t elapsed(Clock::time_point from, Clock::time_point to) noexcept {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count();
    return ns > 0 ? static_cast<std::uint64_t>(ns) : 0;
}
void accumulate(std::uint64_t& total, std::uint64_t value, bool& saturated) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) {
        total = std::numeric_limits<std::uint64_t>::max(); saturated = true;
    } else total += value;
}
struct ReservationBudget {
    std::mutex mutex;
    std::uint64_t reserved = 0, limit = 0;
    std::size_t requests = 0, max_requests = 0;
};
struct Reservation {
    std::shared_ptr<ReservationBudget> budget;
    std::uint64_t bytes = 0;
    ServingAdmission admission = ServingAdmission::reservation_budget;
    Reservation(std::shared_ptr<ReservationBudget> value, std::uint64_t amount) : budget(std::move(value)) {
        std::lock_guard lock(budget->mutex);
        if (budget->requests >= budget->max_requests) admission = ServingAdmission::request_limit;
        else if (amount <= budget->limit - budget->reserved) {
            bytes = amount; budget->reserved += bytes; ++budget->requests;
            admission = ServingAdmission::accepted;
        }
    }
    Reservation(Reservation&& other) noexcept : budget(std::move(other.budget)), bytes(other.bytes), admission(other.admission) {
        other.bytes = 0; other.admission = ServingAdmission::reservation_budget;
    }
    ~Reservation() {
        if (admission != ServingAdmission::accepted) return;
        std::lock_guard lock(budget->mutex);
        budget->reserved -= bytes; --budget->requests;
    }
};
}
namespace detail {
struct GreedyServingJob {
    // Reverse destruction releases backend/session leases before reservation.
    Reservation reservation;
    std::shared_ptr<const void> model_owner, actual_owner;
    std::unique_ptr<SerialGreedyStepBackend> backend;
    std::vector<std::uint32_t> prefix;
    SerialGreedyTokenCursor cursor;
    const std::uintptr_t session_identity;
    const std::uint64_t allocation_epoch;
    std::atomic<ServingRequestState> state{ServingRequestState::queued};
    std::atomic<bool> cancel_requested{false};
    Clock::time_point admitted = Clock::now(), queued = admitted;
    ServingRequestMetrics metrics;
    std::exception_ptr error;
    GreedyServingJob(Reservation lease, std::shared_ptr<const void> model,
        std::shared_ptr<const void> owner, SerialGreedyRequest request,
        std::uintptr_t identity, std::uint64_t epoch)
        : reservation(std::move(lease)), model_owner(std::move(model)), actual_owner(std::move(owner)),
          prefix(request.prefix_tokens.begin(), request.prefix_tokens.end()),
          cursor({prefix, request.vocab_size, request.context_capacity, request.max_new_tokens, request.eos_token_id}),
          session_identity(identity), allocation_epoch(epoch) {}
};
}

ServingRequestHandle::ServingRequestHandle(std::shared_ptr<detail::GreedyServingJob> job) : job_(std::move(job)) {}
ServingRequestState ServingRequestHandle::state() const {
    if (!job_) throw std::logic_error("empty serving request handle");
    return job_->state.load(std::memory_order_acquire);
}
bool ServingRequestHandle::cancel() const noexcept {
    if (!job_ || terminal(job_->state.load(std::memory_order_acquire))) return false;
    job_->cancel_requested.store(true, std::memory_order_release); return true;
}
const SerialGreedyRequestResult& ServingRequestHandle::result() const {
    if (!terminal(state())) throw std::logic_error("serving result is still in flight");
    return job_->cursor.result();
}
ServingRequestMetrics ServingRequestHandle::metrics() const {
    if (!terminal(state())) throw std::logic_error("serving metrics are still in flight");
    return job_->metrics;
}
std::exception_ptr ServingRequestHandle::failure() const {
    if (!terminal(state())) throw std::logic_error("serving request is still in flight");
    return job_->error;
}

struct GreedyServingScheduler::Impl {
    const GreedyServingPolicy policy;
    const ServingCompatibility compatibility;
    const std::shared_ptr<const void> model_owner;
    const std::shared_ptr<ReservationBudget> budget = std::make_shared<ReservationBudget>();
    std::mutex mutex;
    std::mutex inline_execution;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<detail::GreedyServingJob>> queue;
    std::vector<std::weak_ptr<detail::GreedyServingJob>> retained;
    std::size_t active = 0, setting_up = 0;
    std::uint32_t decode_burst = 0;
    // Last member: partially constructed pools stop/join workers before state.
    std::vector<std::jthread> workers;

    Impl(GreedyServingPolicy value, ServingCompatibility key, std::shared_ptr<const void> owner)
        : policy(value), compatibility(std::move(key)), model_owner(std::move(owner)) {
        if ((policy.concurrency != 1 && policy.concurrency != 2 && policy.concurrency != 4) ||
            !policy.max_retained_requests || !policy.prefill_chunk_rows || !policy.max_decode_burst ||
            !policy.reservation_budget_bytes || policy.shared_reservation_bytes > policy.reservation_budget_bytes ||
            !compatibility.model_content.valid() || !compatibility.numerical_identity ||
            compatibility.device_ordinal < 0 || !model_owner)
            throw std::invalid_argument("serving requires current owners, compatible identity and bounded resources");
        budget->reserved = policy.shared_reservation_bytes;
        budget->limit = policy.reservation_budget_bytes;
        budget->max_requests = policy.max_retained_requests;
        retained.reserve(policy.max_retained_requests);
        if (policy.concurrency != 1)
            for (std::uint32_t i = 0; i < policy.concurrency; ++i)
                workers.emplace_back([this](std::stop_token stop) { worker(stop, false); });
    }
    ~Impl() {
        for (auto& worker_thread : workers) worker_thread.request_stop();
        cv.notify_all();
        for (auto& worker_thread : workers) if (worker_thread.joinable()) worker_thread.join();
        for (const auto& job : queue) finish(job, ServingRequestState::cancelled);
    }
    void finish(const std::shared_ptr<detail::GreedyServingJob>& job, ServingRequestState state) noexcept {
        if (state == ServingRequestState::cancelled || state == ServingRequestState::timed_out) job->cursor.cancel();
        job->metrics.wall_ns = elapsed(job->admitted, Clock::now());
        job->state.store(state, std::memory_order_release);
    }
    std::shared_ptr<detail::GreedyServingJob> pick() {
        const auto now = Clock::now();
        auto selected = queue.begin();
        const auto aged = std::find_if(queue.begin(), queue.end(), [&](const auto& job) {
            return elapsed(job->queued, now) >= policy.aging_ns;
        });
        if (aged != queue.end()) selected = aged;
        else if (decode_burst < policy.max_decode_burst) {
            const auto decoding = std::find_if(queue.begin(), queue.end(), [](const auto& job) { return !job->cursor.prefilling(); });
            if (decoding != queue.end()) selected = decoding;
        }
        auto job = std::move(*selected); queue.erase(selected);
        if (job->cursor.prefilling()) decode_burst = 0;
        else if (decode_burst < policy.max_decode_burst) ++decode_burst;
        else decode_burst = 0;
        accumulate(job->metrics.queue_wait_ns, elapsed(job->queued, now), job->metrics.saturated);
        return job;
    }
    void worker(std::stop_token stop, bool inline_worker) {
        for (;;) {
            std::shared_ptr<detail::GreedyServingJob> job;
            {
                std::unique_lock lock(mutex);
                if (inline_worker && queue.empty()) return;
                if (!cv.wait(lock, stop, [&] { return !queue.empty(); })) return;
                job = pick(); ++active;
                job->state.store(ServingRequestState::running, std::memory_order_release);
            }
            auto state = ServingRequestState::queued;
            const auto started = Clock::now();
            if (job->cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) state = ServingRequestState::cancelled;
            else if (policy.queue_timeout_ns && elapsed(job->queued, started) >= policy.queue_timeout_ns) state = ServingRequestState::timed_out;
            else {
                try {
                    job->cursor.advance(*job->backend, policy.prefill_chunk_rows);
                    accumulate(job->metrics.chunks, 1, job->metrics.saturated);
                    if (!job->metrics.time_to_first_token_ns && !job->cursor.result().generated_tokens.empty())
                        job->metrics.time_to_first_token_ns = elapsed(job->admitted, Clock::now());
                    if (job->cursor.done()) state = ServingRequestState::completed;
                    else if (job->cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) state = ServingRequestState::cancelled;
                } catch (...) { job->error = std::current_exception(); state = ServingRequestState::failed; }
            }
            accumulate(job->metrics.service_ns, elapsed(started, Clock::now()), job->metrics.saturated);
            {
                std::lock_guard lock(mutex);
                --active;
                if (state == ServingRequestState::queued) {
                    try {
                        job->queued = Clock::now(); queue.push_back(job);
                        job->state.store(ServingRequestState::queued, std::memory_order_release);
                    } catch (...) { job->error = std::current_exception(); finish(job, ServingRequestState::failed); }
                } else finish(job, state);
            }
            cv.notify_all();
            if (stop.stop_requested()) return;
        }
    }
};

GreedyServingScheduler::GreedyServingScheduler(GreedyServingPolicy policy, ServingCompatibility compatibility,
    std::shared_ptr<const void> owner) : impl_(std::make_unique<Impl>(policy, std::move(compatibility), std::move(owner))) {}
GreedyServingScheduler::~GreedyServingScheduler() = default;

ServingSubmission GreedyServingScheduler::submit(SerialGreedyRequest request,
    ServingCompatibility compatibility, std::uintptr_t session, std::uint64_t epoch,
    std::uint64_t bytes, std::shared_ptr<const void> owner, BackendFactory factory) {
    if (compatibility != impl_->compatibility) return {ServingAdmission::incompatible, {}};
    SerialGreedyTokenCursor validate(request);
    if (!session || !epoch || !bytes || !owner || !factory)
        throw std::invalid_argument("serving admission requires actual session epoch, owner and backend factory");
    const auto emissions = std::min(request.max_new_tokens, request.context_capacity - request.prefix_tokens.size() + 1U);
    if (emissions > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t) ||
        request.prefix_tokens.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t) - emissions ||
        (emissions + request.prefix_tokens.size()) * sizeof(std::uint32_t) > bytes)
        return {ServingAdmission::reservation_budget, {}};
    std::shared_ptr<detail::GreedyServingJob> job;
    {
        std::lock_guard lock(impl_->mutex);
        std::erase_if(impl_->retained, [](const auto& weak) { return weak.expired(); });
        for (const auto& weak : impl_->retained) if (const auto live = weak.lock())
            if (live->session_identity == session)
                return {ServingAdmission::duplicate_session, {}};
        Reservation reservation(impl_->budget, bytes);
        if (reservation.admission != ServingAdmission::accepted) return {reservation.admission, {}};
        job = std::make_shared<detail::GreedyServingJob>(std::move(reservation), impl_->model_owner,
            std::move(owner), request, session, epoch);
        impl_->retained.push_back(job); ++impl_->setting_up;
    }
    try {
        job->backend = factory();
        if (!job->backend) throw std::invalid_argument("serving factory returned no backend");
        std::lock_guard lock(impl_->mutex);
        job->queued = Clock::now(); impl_->queue.push_back(job); --impl_->setting_up;
    } catch (...) {
        { std::lock_guard lock(impl_->mutex); --impl_->setting_up; }
        impl_->cv.notify_all(); throw;
    }
    impl_->cv.notify_all();
    return {ServingAdmission::accepted, ServingRequestHandle(std::move(job))};
}

void GreedyServingScheduler::run_until_idle() {
    if (impl_->policy.concurrency == 1) {
        std::lock_guard execution(impl_->inline_execution);
        for (;;) {
            {
                std::unique_lock lock(impl_->mutex);
                impl_->cv.wait(lock, [&] { return !impl_->queue.empty() || !impl_->setting_up; });
                if (impl_->queue.empty()) return;
            }
            impl_->worker(std::stop_token{}, true);
        }
    }
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] { return impl_->queue.empty() && !impl_->active && !impl_->setting_up; });
}
std::uint64_t GreedyServingScheduler::reserved_bytes() const {
    std::lock_guard lock(impl_->budget->mutex); return impl_->budget->reserved;
}
} // namespace strix
