#include "strix/greedy_serving_scheduler.hpp"
#include "strix/session/kv_capacity.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
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

}
namespace detail {
struct GreedyServingJob {
    // Reverse destruction releases backend/session leases before reservation.
    session::KvCapacity::Lease reservation;
    std::shared_ptr<const void> model_owner, actual_owner;
    std::unique_ptr<SerialGreedyStepBackend> backend;
    std::vector<std::uint32_t> prefix;
    SerialGreedyTokenCursor cursor;
    const std::uintptr_t session_identity;
    const std::uint64_t allocation_epoch;
    std::atomic<ServingRequestState> state{ServingRequestState::queued};
    std::atomic<bool> cancel_requested{false};
    Clock::time_point admitted = Clock::now(), queued = admitted;
    std::optional<Clock::time_point> last_token;
    ServingRequestMetrics metrics;
    std::exception_ptr error;
    GreedyServingJob(session::KvCapacity::Lease lease, std::shared_ptr<const void> model,
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
    const std::shared_ptr<ServingCohortExecutor> cohort;
    const session::KvPageLayout execution_state;
    session::KvCapacity capacity;
    std::mutex mutex;
    std::mutex inline_execution;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<detail::GreedyServingJob>> queue;
    std::vector<std::weak_ptr<detail::GreedyServingJob>> retained;
    std::size_t active = 0, setting_up = 0;
    std::uint32_t decode_burst = 0;
    // Last member: partially constructed pools stop/join workers before state.
    std::vector<std::jthread> workers;

    Impl(GreedyServingPolicy value, ServingCompatibility key, std::shared_ptr<const void> owner,
         std::shared_ptr<ServingCohortExecutor> executor, session::KvPageLayout layout)
        : policy(value), compatibility(std::move(key)), model_owner(std::move(owner)),cohort(std::move(executor)),
          execution_state(layout), capacity(policy.reservation_budget_bytes, policy.shared_reservation_bytes, policy.max_retained_requests) {
        if ((policy.concurrency != 1 && policy.concurrency != 2 && policy.concurrency != 4) ||
            !execution_state.page_tokens || !policy.max_retained_requests || !policy.prefill_chunk_rows || !policy.max_decode_burst ||
            !policy.reservation_budget_bytes || policy.shared_reservation_bytes > policy.reservation_budget_bytes ||
            !compatibility.model_content.valid() || !compatibility.numerical_identity ||
            compatibility.device_ordinal < 0 || !model_owner)
            throw std::invalid_argument("serving requires current owners, compatible identity and bounded resources");
        if ((policy.tensor_cohort_rows!=1U && policy.tensor_cohort_rows!=2U && policy.tensor_cohort_rows!=4U) ||
            policy.tensor_cohort_rows>policy.concurrency || (policy.tensor_cohort_rows>1U && !cohort))
            throw std::invalid_argument("serving tensor cohort requires explicit executor and bounded C2/C4 capacity");
        retained.reserve(policy.max_retained_requests);
        if (policy.tensor_cohort_rows>1U)
            workers.emplace_back([this](std::stop_token stop) { cohort_worker(stop); });
        else if (policy.concurrency != 1)
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
        if (execution_state.page_bytes || execution_state.whole_slot_bytes)
            job->backend.reset();
        job->reservation.execution.reset();
        job->metrics.wall_ns = elapsed(job->admitted, Clock::now());
        job->state.store(state, std::memory_order_release);
    }
    void end_chunk(const std::shared_ptr<detail::GreedyServingJob>& job, ServingRequestState state) {
        if (state == ServingRequestState::queued) {
            try {
                std::lock_guard lock(mutex);
                job->queued = Clock::now();
                queue.push_back(job);
                job->state.store(state, std::memory_order_release);
                --active;
                return;
            } catch (...) {
                job->error = std::current_exception();
                state = ServingRequestState::failed;
            }
        }
        // Backend cleanup can call application code. Keep the job active until
        // teardown completes but do not hold the queue lock across it.
        finish(job, state);
        std::lock_guard lock(mutex);
        --active;
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
    void cohort_delivery_metrics(const std::shared_ptr<detail::GreedyServingJob>& job) noexcept {
        const auto emitted=job->cursor.result().generated_tokens.size();
        if(emitted<=job->metrics.emitted_tokens) return;
        const auto delivered=Clock::now();
        if(!job->last_token) job->metrics.time_to_first_token_ns=elapsed(job->admitted,delivered);
        else {
            const auto interval=elapsed(*job->last_token,delivered);
            accumulate(job->metrics.inter_token_intervals,1,job->metrics.saturated);
            accumulate(job->metrics.inter_token_ns_total,interval,job->metrics.saturated);
            job->metrics.inter_token_ns_max=std::max(job->metrics.inter_token_ns_max,interval);
            const auto bucket=interval>1?static_cast<unsigned>(std::bit_width(interval))-1U:0U;
            accumulate(job->metrics.inter_token_ns_histogram[bucket],1,job->metrics.saturated);
        }
        job->last_token=delivered; job->metrics.emitted_tokens=emitted;
    }
    void cohort_worker(std::stop_token stop) {
        for(;;) {
            std::array<std::shared_ptr<detail::GreedyServingJob>,4> jobs;
            std::array<ServingRequestState,4> states{};
            std::array<ServingCohortTarget,4> targets;
            std::array<std::size_t,4> target_job{};
            std::size_t count=0,target_count=0;
            {
                std::unique_lock lock(mutex);
                if(!cv.wait(lock,stop,[&]{return !queue.empty();})) return;
                // Only ready work: no timer, barrier, or speculative fill wait.
                while(count<policy.tensor_cohort_rows && !queue.empty()) {
                    jobs[count]=pick(); jobs[count]->state.store(ServingRequestState::running,std::memory_order_release);
                    states[count]=ServingRequestState::queued; ++count; ++active;
                }
            }
            const auto started=Clock::now();
            for(std::size_t i=0;i<count;++i) {
                auto& job=*jobs[i];
                if(job.cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) {
                    states[i]=ServingRequestState::cancelled; continue;
                }
                if(policy.queue_timeout_ns && elapsed(job.queued,started)>=policy.queue_timeout_ns) {
                    states[i]=ServingRequestState::timed_out; continue;
                }
                try {
                    const auto input=job.cursor.prepare_target(*job.backend);
                    accumulate(job.metrics.chunks,1,job.metrics.saturated);
                    if(input) {
                        if(job.cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) {
                            job.cursor.cancel_prepared_target(*input); states[i]=ServingRequestState::cancelled;
                        } else { targets[target_count]={job.backend.get(),*input,{}}; target_job[target_count++]=i; }
                    } else if(job.cursor.done()) states[i]=ServingRequestState::completed;
                } catch(...) { job.error=std::current_exception(); states[i]=ServingRequestState::failed; }
            }
            std::size_t first=0;
            while(first<target_count) {
                // Recheck cancel immediately before submission, without
                // attributing a fabricated target call to a cancelled lease.
                const auto job_index=target_job[first]; auto& first_job=*jobs[job_index];
                if(first_job.cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) {
                    first_job.cursor.cancel_prepared_target(targets[first].input);
                    states[job_index]=ServingRequestState::cancelled; ++first; continue;
                }
                std::size_t width=target_count-first>=4U?4U:(target_count-first>=2U?2U:1U);
                std::array<SerialGreedyStepBackend*,4> backends{};
                for(std::size_t row=0;row<width;++row) backends[row]=targets[first+row].backend;
                if(width>1U && !cohort->compatible(std::span<SerialGreedyStepBackend* const>(backends.data(),width))) {
                    width=2U;
                    if(!cohort->compatible(std::span<SerialGreedyStepBackend* const>(backends.data(),width))) width=1U;
                }
                // A late cancellation removes only that member before any
                // submit. Unaffected prepared leases remain ready for C1/2/4.
                bool cancelled_member=false;
                for(std::size_t row=1;row<width;++row) {
                    const auto index=first+row; auto& job=*jobs[target_job[index]];
                    if(job.cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) {
                        job.cursor.cancel_prepared_target(targets[index].input);
                        states[target_job[index]]=ServingRequestState::cancelled;
                        for(std::size_t move=index+1U;move<target_count;++move) {
                            targets[move-1U]=targets[move]; target_job[move-1U]=target_job[move];
                        }
                        --target_count; cancelled_member=true; break;
                    }
                }
                if(cancelled_member) continue;
                try {
                    if(width==1U) execute_serving_target_fallback(targets[first]);
                    else cohort->execute(std::span<ServingCohortTarget>(targets.data()+first,width));
                    for(std::size_t row=0;row<width;++row) {
                        const auto index=first+row; auto& job=*jobs[target_job[index]];
                        try {
                            if(job.cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) job.cursor.cancel();
                            job.cursor.accept_target(*job.backend,targets[index].input,targets[index].selected);
                            if(job.cursor.cancelled()) states[target_job[index]]=ServingRequestState::cancelled;
                            else if(job.cursor.done()) states[target_job[index]]=ServingRequestState::completed;
                        } catch(...) { job.error=std::current_exception(); states[target_job[index]]=ServingRequestState::failed; }
                    }
                } catch(...) {
                    const auto error=std::current_exception();
                    for(std::size_t row=0;row<width;++row) {
                        auto& job=*jobs[target_job[first+row]];
                        job.cursor.fail_target(); job.error=error; states[target_job[first+row]]=ServingRequestState::failed;
                    }
                }
                first+=width;
            }
            for(std::size_t i=0;i<count;++i) {
                auto& job=jobs[i]; cohort_delivery_metrics(job);
                accumulate(job->metrics.service_ns,elapsed(started,Clock::now()),job->metrics.saturated);
                end_chunk(job, states[i]);
            }
            cv.notify_all();
            if(stop.stop_requested()) return;
        }
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
                    const auto emitted = job->cursor.result().generated_tokens.size();
                    if (emitted > job->metrics.emitted_tokens) {
                        const auto delivered = Clock::now();
                        if (!job->last_token) job->metrics.time_to_first_token_ns = elapsed(job->admitted, delivered);
                        else {
                            const auto interval = elapsed(*job->last_token, delivered);
                            accumulate(job->metrics.inter_token_intervals, 1, job->metrics.saturated);
                            accumulate(job->metrics.inter_token_ns_total, interval, job->metrics.saturated);
                            job->metrics.inter_token_ns_max = std::max(job->metrics.inter_token_ns_max, interval);
                            const auto bucket = interval > 1 ? static_cast<unsigned>(std::bit_width(interval)) - 1U : 0U;
                            accumulate(job->metrics.inter_token_ns_histogram[bucket], 1, job->metrics.saturated);
                        }
                        job->last_token = delivered;
                        job->metrics.emitted_tokens = emitted;
                    }
                    if (job->cursor.done()) state = ServingRequestState::completed;
                    else if (job->cancel_requested.load(std::memory_order_acquire) || stop.stop_requested()) state = ServingRequestState::cancelled;
                } catch (...) { job->error = std::current_exception(); state = ServingRequestState::failed; }
            }
            accumulate(job->metrics.service_ns, elapsed(started, Clock::now()), job->metrics.saturated);
            end_chunk(job, state);
            cv.notify_all();
            if (stop.stop_requested()) return;
        }
    }
};

GreedyServingScheduler::GreedyServingScheduler(GreedyServingPolicy policy, ServingCompatibility compatibility,
    std::shared_ptr<const void> owner, std::shared_ptr<ServingCohortExecutor> cohort)
    : GreedyServingScheduler(policy, std::move(compatibility), std::move(owner), std::move(cohort), {}) {}
GreedyServingScheduler::GreedyServingScheduler(GreedyServingPolicy policy, ServingCompatibility compatibility,
    std::shared_ptr<const void> owner, std::shared_ptr<ServingCohortExecutor> cohort, session::KvPageLayout layout)
    : impl_(std::make_unique<Impl>(policy, std::move(compatibility), std::move(owner), std::move(cohort), layout)) {}
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
        if (impl_->retained.size() >= impl_->policy.max_retained_requests)
            return {ServingAdmission::request_limit, {}};
        auto reservation = impl_->capacity.reserve(impl_->execution_state, request.context_capacity, 1, bytes);
        if (reservation.admission == session::KvCapacity::Admission::slot_limit)
            return {ServingAdmission::request_limit, {}};
        if (reservation.admission != session::KvCapacity::Admission::accepted)
            return {ServingAdmission::reservation_budget, {}};
        job = std::make_shared<detail::GreedyServingJob>(std::move(reservation.lease), impl_->model_owner,
            std::move(owner), request, session, epoch);
        impl_->retained.push_back(job); ++impl_->setting_up;
    }
    try {
        job->backend = factory();
        if (!job->backend) throw std::invalid_argument("serving factory returned no backend");
        std::lock_guard lock(impl_->mutex);
        job->queued = Clock::now(); impl_->queue.push_back(job); --impl_->setting_up;
    } catch (...) {
        job.reset();
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
    return impl_->capacity.reserved_bytes();
}
void GreedyServingScheduler::with_optional_cache(const std::function<void(PrefixStateCache&)>& visit) const {
    impl_->capacity.with_optional_cache(visit);
}
} // namespace strix
