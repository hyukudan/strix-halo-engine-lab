#include "strix/serve/engine.hpp"

#include <algorithm>
#include <exception>
#include <thread>

namespace strix::serve {

namespace {
using SteadyClock = std::chrono::steady_clock;
constexpr auto kNever = SteadyClock::time_point::max();

// The process-wide arming state (HTTP side and the expiry thread only), on its own cache line (written on every arming
// request, unlike the period word the engine threads read). One lock orders the arming requests and the expiry, so an
// arming request never loses its period to an expiry decided before it. The expiry thread is started by the first
// arming request with a finite expiry and lives until the process ends (it does not belong to any ApiService).
class alignas(64) Arming {
public:
    ~Arming() {
        {
            const std::lock_guard lock(mutex_);
            if (!thread_.joinable()) return;
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }
    bool arm(SteadyClock::duration expiry) {
        const std::lock_guard lock(mutex_);
        const auto now = SteadyClock::now();
        const auto until = expiry > SteadyClock::duration::zero() ? now + expiry : kNever;
        const auto period = observe_period();
        if ((period & 1U) != 0U) {  // armed: the word is not written again; the deadline only moves later
            if (until > deadline_) deadline_ = until;
            return false;
        }
        deadline_ = until;
        serve_observe_period_.store(period + 1U, std::memory_order_relaxed);
        if (until != kNever && !thread_.joinable()) thread_ = std::thread([this] { run(); });
        wake_.notify_one();  // a new period: the thread waits for one (or for its deadline)
        return true;
    }
    void disarm() {
        const std::lock_guard lock(mutex_);
        end_locked();
    }

private:
    void end_locked() {
        const auto period = observe_period();
        if ((period & 1U) != 0U) serve_observe_period_.store(period + 1U, std::memory_order_relaxed);
    }
    void run() {
        std::unique_lock lock(mutex_);
        while (!stop_) {
            const auto period = observe_period();
            if ((period & 1U) == 0U || deadline_ == kNever) {  // unarmed, or armed without an expiry: until the period changes
                wake_.wait(lock, [&] { return stop_ || observe_period() != period; });
                continue;
            }
            if (SteadyClock::now() >= deadline_) {
                end_locked();
                continue;
            }
            // within a period the deadline only moves later: it is read again when this one passes
            wake_.wait_until(lock, deadline_, [&] { return stop_ || observe_period() != period; });
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    SteadyClock::time_point deadline_{};  // under mutex_: the armed period ends then (kNever: it does not expire)
    bool stop_ = false;
    std::thread thread_;
};
Arming& arming() {
    static Arming state;  // destroyed (its thread joined) when the process ends
    return state;
}
}  // namespace

bool observe_arm(double expiry_seconds) {
    const auto expiry = expiry_seconds > 0.0
        ? std::chrono::duration_cast<SteadyClock::duration>(std::chrono::duration<double>(std::min(expiry_seconds, 1e9)))
        : SteadyClock::duration::zero();
    return arming().arm(expiry);
}

void observe_disarm() { arming().disarm(); }

void GenerationTicket::notify() {
    const std::lock_guard lock(mutex_);
    changed_.notify_all();
}

std::vector<std::uint32_t> GenerationTicket::wait(std::chrono::milliseconds timeout, bool& done) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, timeout, [&] { return done_ || !pending_.empty() || cancelled_.load(); });
    std::vector<std::uint32_t> out;
    out.swap(pending_);
    done = done_;
    return out;
}

GenerationStats GenerationTicket::stats() const {
    const std::lock_guard lock(mutex_);
    return stats_;
}
std::string GenerationTicket::error() const {
    const std::lock_guard lock(mutex_);
    return error_;
}

bool GenerationTicket::publish(std::span<const std::uint32_t> tokens) {
    {
        const std::lock_guard lock(mutex_);
        pending_.insert(pending_.end(), tokens.begin(), tokens.end());
        changed_.notify_all();
    }
    return !cancelled_.load();
}

void GenerationTicket::finish(GenerationStats stats, std::string error) {
    const std::lock_guard lock(mutex_);
    stats_ = std::move(stats);
    error_ = std::move(error);
    done_ = true;
    changed_.notify_all();
}

std::shared_ptr<GenerationTicket> GenerationQueue::submit(GenerationJob job) {
    auto ticket = std::make_shared<GenerationTicket>();
    const std::lock_guard lock(mutex_);
    if (stopping_ || entries_.size() >= max_queued_) return nullptr;
    ticket->queue_position_.store(entries_.size() + totals_.running);
    ticket->period_ = observe_period();  // one read: armed (odd) and the period still_observed() compares
    job.observed = ticket->observed_ = (ticket->period_ & 1U) != 0U;
    job.observed_period = ticket->period_;
    entries_.push_back({std::move(job), ticket, std::chrono::steady_clock::now()});
    ++totals_.queued;
    ready_.notify_one();
    return ticket;
}

void GenerationQueue::run(GenerationEngine& engine, const std::function<void(const GenerationStats&)>& observe) {
    for (;;) {
        Entry entry;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || !entries_.empty(); });
            if (entries_.empty()) return;
            entry = std::move(entries_.front());
            entries_.pop_front();
            --totals_.queued;
            ++totals_.running;
            std::uint64_t position = 0;
            for (auto& waiting : entries_) waiting.ticket->queue_position_.store(++position);
        }
        auto ticket = entry.ticket;
        GenerationStats stats;
        std::string error;
        const auto started = std::chrono::steady_clock::now();
        if (ticket->cancelled()) stats.finish = "cancelled";
        else {
            entry.job.on_tokens = [ticket](std::span<const std::uint32_t> tokens) { return ticket->publish(tokens); };
            try { stats = engine.generate(entry.job); }
            catch (const std::exception& e) { error = e.what(); }
            catch (...) { error = "engine failure"; }
            if (ticket->cancelled() && error.empty()) stats.finish = "cancelled";
        }
        stats.queue_ms = std::chrono::duration<double, std::milli>(started - entry.submitted).count();
        const auto busy = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const bool observed = still_observed(*ticket);
        ObserveDelta delta;
        if (observed) {  // computed before the lock, which only adds it
            if (stats.first_token_ms) *stats.first_token_ms += stats.queue_ms;  // the engine measured from generate()
            delta = observe_delta(stats, error);
        }
        {
            const std::lock_guard lock(mutex_);
            --totals_.running;
            if (!error.empty()) ++totals_.failed;
            else if (stats.finish == "cancelled") ++totals_.cancelled;
            else ++totals_.completed;
            totals_.prompt_tokens += stats.prompt_tokens;
            totals_.cached_tokens += stats.cached_tokens;
            totals_.generated_tokens += stats.generated_tokens;
            totals_.busy_seconds += busy;
            if (observed) observe_locked(delta);
        }
        if (observe) observe(stats);
        ticket->finish(std::move(stats), std::move(error));
    }
}

std::optional<TakenJob> GenerationQueue::take(bool wait) {
    for (;;) {
        Entry entry;
        {
            std::unique_lock lock(mutex_);
            if (wait) ready_.wait(lock, [&] { return stopping_ || !entries_.empty(); });
            if (entries_.empty()) return std::nullopt;
            entry = std::move(entries_.front());
            entries_.pop_front();
            --totals_.queued;
            ++totals_.running;
            std::uint64_t position = 0;
            for (auto& waiting : entries_) waiting.ticket->queue_position_.store(++position);
        }
        TakenJob taken{std::move(entry.job), entry.ticket, entry.submitted, std::chrono::steady_clock::now()};
        if (taken.ticket->cancelled()) {
            GenerationStats stats;
            stats.finish = "cancelled";
            finish(taken, std::move(stats));
            continue;
        }
        auto ticket = taken.ticket;
        taken.job.on_tokens = [ticket](std::span<const std::uint32_t> tokens) { return ticket->publish(tokens); };
        return taken;
    }
}

void GenerationQueue::finish(TakenJob& job, GenerationStats stats, std::string error,
    const std::function<void(const GenerationStats&)>& observe) {
    if (job.ticket->cancelled() && error.empty()) stats.finish = "cancelled";
    stats.queue_ms = std::chrono::duration<double, std::milli>(job.started - job.submitted).count();
    // Batched jobs overlap: busy time adds each job's own service time.
    const auto busy = std::chrono::duration<double>(std::chrono::steady_clock::now() - job.started).count();
    const bool observed = still_observed(*job.ticket);
    ObserveDelta delta;
    if (observed) delta = observe_delta(stats, error);  // before the lock, which only adds it
    {
        const std::lock_guard lock(mutex_);
        --totals_.running;
        if (!error.empty()) ++totals_.failed;
        else if (stats.finish == "cancelled") ++totals_.cancelled;
        else ++totals_.completed;
        totals_.prompt_tokens += stats.prompt_tokens;
        totals_.cached_tokens += stats.cached_tokens;
        totals_.generated_tokens += stats.generated_tokens;
        totals_.busy_seconds += busy;
        if (observed) observe_locked(delta);
    }
    if (observe) observe(stats);
    job.ticket->finish(std::move(stats), std::move(error));
}

// An observed request whose armed period expired before it finished is not observed: its per-step measurements stopped
// at the expiry. Runs once per finished request on the finishing thread, before the ticket is finished (the HTTP side
// reads observed() only after that).
bool GenerationQueue::still_observed(GenerationTicket& ticket) noexcept {
    if (!ticket.observed_) return false;
    if (observe_period() == ticket.period_) return true;
    ticket.observed_ = false;
    return false;
}

ObserveDelta GenerationQueue::observe_delta(const GenerationStats& stats, const std::string& error) noexcept {
    ObserveDelta d{};  // every member zero, then no histogram observation
    d.ttft_bucket = ttft_seconds_bounds.size() + 1;
    d.decode_bucket = decode_tok_s_bounds.size() + 1;
    if (!error.empty()) return d;
    d.requests = 1;
    if (stats.generated_tokens != 0U) {  // the prompt was read (a request cancelled before its first token computed none)
        d.prompt_tokens = stats.prompt_tokens;
        d.cached_tokens = stats.cached_tokens;
        d.imported_tokens = stats.imported_tokens;
        const auto restored = stats.cached_tokens + stats.imported_tokens;
        d.computed_tokens = stats.prompt_tokens > restored ? stats.prompt_tokens - restored : 0U;
    }
    if (stats.first_token_ms) {
        d.ttft_seconds = *stats.first_token_ms / 1000.0;
        d.ttft_bucket = decltype(ObserveSnapshot::ttft_seconds)::bucket(d.ttft_seconds, ttft_seconds_bounds);
    }
    if (stats.decode_ms > 0.0 && stats.generated_tokens >= 2U) {
        d.decode_tokens_per_second = 1000.0 * static_cast<double>(stats.generated_tokens) / stats.decode_ms;
        d.decode_bucket = decltype(ObserveSnapshot::decode_tokens_per_second)::bucket(d.decode_tokens_per_second, decode_tok_s_bounds);
    }
    return d;
}

void GenerationQueue::observe_locked(const ObserveDelta& d) noexcept {
    auto& o = observed_;
    o.requests += d.requests;
    o.prompt_tokens += d.prompt_tokens;
    o.cached_tokens += d.cached_tokens;
    o.imported_tokens += d.imported_tokens;
    o.computed_tokens += d.computed_tokens;
    o.ttft_seconds.add(d.ttft_bucket, d.ttft_seconds);
    o.decode_tokens_per_second.add(d.decode_bucket, d.decode_tokens_per_second);
}

bool GenerationQueue::stopping() const {
    const std::lock_guard lock(mutex_);
    return stopping_;
}

void GenerationQueue::shutdown() {
    std::deque<Entry> dropped;
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
        dropped.swap(entries_);
        totals_.queued = 0;
        ready_.notify_all();
    }
    for (auto& entry : dropped) {
        GenerationStats stats;
        stats.finish = "cancelled";
        entry.ticket->finish(stats, "server shutting down");
    }
}

QueueSnapshot GenerationQueue::snapshot() const {
    const std::lock_guard lock(mutex_);
    return totals_;
}

ObserveSnapshot GenerationQueue::observed() const {
    const std::lock_guard lock(mutex_);
    return observed_;
}

}  // namespace strix::serve
