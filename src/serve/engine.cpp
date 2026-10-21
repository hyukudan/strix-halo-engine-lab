#include "strix/serve/engine.hpp"

#include <exception>

namespace strix::serve {

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
    }
    if (observe) observe(stats);
    job.ticket->finish(std::move(stats), std::move(error));
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

}  // namespace strix::serve
