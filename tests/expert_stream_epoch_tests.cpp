// CPU tests of the streamed-expert run epochs (strix/expert_stream_epoch.hpp), the lifecycle both CUDA DS41 expert
// providers bind: run counters take only their own run's publications and freeze at end_run; the idle preload has its
// own counters; a late publication is dropped. The threaded part models the providers' loader (one worker, jobs
// captured under the mutex, job-local counters published under it) with a controllable preload I/O that is held
// while the caller reads the stats and begins the next run; build it with -fsanitize=thread to check the protocol.
#include "strix/expert_stream_epoch.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
bool same(const strix::ExpertStreamCounters& a, const strix::ExpertStreamCounters& b) {
    return a.bytes == b.bytes && a.file_bytes == b.file_bytes && a.io_seconds == b.io_seconds;
}

void unit() {
    using strix::ExpertStreamCounters;
    strix::ExpertStreamEpochs e;
    check(!e.is_open() && e.run().bytes == 0U && !e.preload_followed(0), "fresh state");
    e.open(1);
    const auto r1 = e.run_job();
    check(e.publish(r1, {100, 100, 0.5}), "current run may publish per-layer timing and byte metadata");
    e.publish(r1, {50, 0, 0.0});
    check(e.run().bytes == 150U && e.run().file_bytes == 100U && e.run().io_seconds == 0.5, "run accumulates its own jobs");
    e.close();
    const ExpertStreamCounters closed = e.run();
    check(!e.publish(r1, {7, 7, 1.0}), "an ended run cannot publish per-layer metadata");
    check(same(e.run(), closed) && e.dropped() == 1U, "closed run is frozen, late run job dropped");
    const auto p1 = e.start_preload();
    e.publish(p1, {200, 200, 2.0});
    check(same(e.run(), closed), "preload never enters the run counters");
    check(e.preload().bytes == 200U && e.preload_followed(1) && !e.preload_followed(0), "preload counters tagged with the run it followed");
    e.open(2);
    check(e.run().bytes == 0U && e.preload().bytes == 200U, "open zeroes the run, keeps the preload");
    check(!e.publish(r1, {9, 9, 9.0}), "an old job cannot mark the new run's timing origin");
    check(e.run().bytes == 0U && e.dropped() == 2U, "another run's job is dropped");
    e.publish(e.run_job(), {30, 0, 0.0});
    e.close();
    const auto p2 = e.start_preload();
    e.publish(p1, {5, 5, 5.0});  // the superseded preload
    check(e.preload().bytes == 0U && e.preload_followed(2) && e.dropped() == 3U, "a new preload replaces the old one, stale preload dropped");
    e.publish(p2, {40, 40, 0.25});
    check(e.preload().bytes == 40U && e.run().bytes == 30U, "second preload counted apart from run 2");
}

// Model of the providers' loader thread (the same lock discipline, no CUDA). Each run use reads 100 bytes in 1 s of
// "I/O"; the idle preload reads `preload_layers` layers, each held until the test grants a permit.
class Loader {
public:
    explicit Loader(std::uint32_t preload_layers) : preload_layers_(preload_layers), worker_([this] { loop(); }) {}
    ~Loader() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
        }
        {
            std::lock_guard<std::mutex> g(io_m_);
            permits_ += 1000;
        }
        wake_.notify_all();
        gate_.notify_all();
        worker_.join();
    }
    void begin_run(std::uint32_t uses) {
        std::lock_guard<std::mutex> g(m_);
        ++run_id_;
        epochs_.open(run_id_);
        run_ = true;
        uses_ = uses;
        served_ = 0;
        wake_.notify_all();
    }
    void end_run() {
        std::lock_guard<std::mutex> g(m_);
        epochs_.close();
        run_ = false;
        wake_.notify_all();
    }
    strix::ExpertStreamCounters stats() const {
        std::lock_guard<std::mutex> g(m_);
        return epochs_.run();
    }
    strix::ExpertStreamCounters preload() const {
        std::lock_guard<std::mutex> g(m_);
        return epochs_.preload();
    }
    bool preload_followed(std::uint64_t run) const {
        std::lock_guard<std::mutex> g(m_);
        return epochs_.preload_followed(run);
    }
    void grant(std::uint32_t n) {
        {
            std::lock_guard<std::mutex> g(io_m_);
            permits_ += n;
        }
        gate_.notify_all();
    }
    // waits (bounded) for: all uses served / the preload blocked in its layer `layer` / the preload done
    bool wait_served() { return wait(m_, done_, [&] { return served_ == uses_; }); }
    bool wait_preload_at(std::uint32_t layer) { return wait(io_m_, gate_, [&] { return preload_waiting_ == layer; }); }
    bool wait_preload_done() { return wait(m_, done_, [&] { return preload_done_; }); }
    // spins (bounded) until n preload reads finished, with no synchronisation with the reading thread
    bool wait_reads_relaxed(std::uint32_t n) const {
        const auto t0 = std::chrono::steady_clock::now();
        while (reads_done_.load(std::memory_order_relaxed) < n)
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) return false;
            else std::this_thread::yield();
        return true;
    }

private:
    template <class P> static bool wait(std::mutex& m, std::condition_variable& cv, P pred) {
        std::unique_lock<std::mutex> lock(m);
        return cv.wait_for(lock, std::chrono::seconds(20), pred);
    }
    void loop() {
        bool idle_done = true;  // no preload before the first run
        std::uint64_t run_seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(m_);
            wake_.wait(lock, [&] { return stop_ || (run_ && (run_id_ != run_seen || served_ < uses_)) || (!run_ && !idle_done); });
            if (stop_) return;
            if (run_ && run_id_ != run_seen) {
                run_seen = run_id_;
                idle_done = false;
            }
            if (!run_) {
                idle_done = true;
                preload_done_ = false;
                const auto job = epochs_.start_preload();
                lock.unlock();
                strix::ExpertStreamCounters pre{};  // job-local: nobody else touches it
                for (std::uint32_t l = 0; l < preload_layers_; ++l) {
                    {  // the I/O hold has its own lock: like a real read, it never takes m_
                        std::unique_lock<std::mutex> g(io_m_);
                        preload_waiting_ = l;
                        gate_.notify_all();
                        gate_.wait(g, [&] { return permits_ > 0U; });
                        --permits_;
                        preload_waiting_ = ~0U;
                    }
                    pre.bytes += 100;  // the held "read"
                    pre.file_bytes += 100;
                    pre.io_seconds += 1.0;
                    reads_done_.fetch_add(1, std::memory_order_relaxed);  // observation only: no happens-before
                }
                lock.lock();
                epochs_.publish(job, pre);
                preload_done_ = true;
                lock.unlock();
                done_.notify_all();
                continue;
            }
            if (served_ >= uses_) continue;
            const auto job = epochs_.run_job();
            lock.unlock();
            const strix::ExpertStreamCounters c{100, 100, 1.0};  // the use's "read", job-local
            lock.lock();
            epochs_.publish(job, c);
            ++served_;
            lock.unlock();
            done_.notify_all();
        }
    }

    const std::uint32_t preload_layers_;
    mutable std::mutex m_;  // the provider's mutex (epochs, run state)
    std::mutex io_m_;       // the test's I/O hold only (permits_, preload_waiting_)
    std::condition_variable wake_, done_, gate_;
    std::atomic<std::uint32_t> reads_done_{0};  // preload reads finished (relaxed: the test sees it, no ordering)
    bool stop_ = false, run_ = false, preload_done_ = false;
    std::uint64_t run_id_ = 0;
    std::uint32_t uses_ = 0, served_ = 0, permits_ = 0, preload_waiting_ = ~0U;
    strix::ExpertStreamEpochs epochs_;
    std::thread worker_;  // last: starts after every member above exists
};

void lifecycle() {
    Loader ld(2);
    ld.begin_run(3);
    check(ld.wait_served(), "run 1 served");
    ld.end_run();
    const auto closed = ld.stats();
    check(closed.bytes == 300U && closed.io_seconds == 3.0, "run 1 counters");
    check(ld.wait_preload_at(0), "preload holds its first read");
    check(same(ld.stats(), closed), "stats during the held preload I/O = the closed snapshot");
    ld.grant(1);
    // after the preload's first read finished, with nothing ordering it before this stats(): under TSan an unlocked
    // shared counter written by that read (the defect this protocol removes) is reported here
    check(ld.wait_reads_relaxed(1), "preload finished its first read");
    check(same(ld.stats(), closed), "stats unordered with a preload read = the closed snapshot");
    check(ld.wait_preload_at(1), "preload holds its second read");
    check(same(ld.stats(), closed), "stats after one preload read = the closed snapshot");
    ld.begin_run(2);  // the preload is still reading (it serves this run's first slots)
    check(ld.stats().bytes == 0U && ld.stats().io_seconds == 0.0, "begin_run during the preload: zero, preload not counted");
    ld.grant(1);
    check(ld.wait_preload_done(), "preload done");
    check(ld.wait_served(), "run 2 served");
    check(ld.preload().bytes == 200U && ld.preload().io_seconds == 2.0 && ld.preload_followed(1), "preload counted apart, tagged run 1");
    ld.end_run();
    const auto r2 = ld.stats();
    check(r2.bytes == 200U && r2.file_bytes == 200U && r2.io_seconds == 2.0, "run 2 = its own two uses only");
    check(ld.wait_preload_at(0), "second preload holds");
    check(same(ld.stats(), r2), "run 2 frozen during the second preload");
    ld.grant(2);
    check(ld.wait_preload_done(), "second preload done");
    check(same(ld.stats(), r2) && ld.preload_followed(2), "run 2 unchanged after the second preload published");
}

}  // namespace

int main() {
    unit();
    lifecycle();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return EXIT_FAILURE;
    }
    std::cout << "expert stream epoch tests: OK\n";
    return EXIT_SUCCESS;
}
