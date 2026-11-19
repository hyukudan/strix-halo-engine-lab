#pragma once
// Run epochs of a streamed-expert loader (backend-neutral: the CUDA DS41 streamer and its whole-layer provider bind it).
//
// A loader thread serves two kinds of job: the uses of a run (between begin_run and end_run) and the idle preload that
// follows a run end (the next run's first slots). The preload is not joined by end_run, so it can still be reading
// while the caller ends the run, reads its stats or begins the next run. The rule that keeps the receipts exact and
// race-free: a job captures what it belongs to (ExpertStreamJob) under the owner's mutex when it starts, accumulates
// its counters locally (no shared writes during the I/O) and publishes them under the same mutex when it is done.
// A run's counters take only publications of that run's own jobs, so they stop changing when the run ends (the closed
// snapshot) and a preload never enters them; the preload has its own counters, tagged with the run it followed.
//
// Not synchronised by itself: every member is called with the owner's mutex held.
#include <cstdint>

namespace strix {

struct ExpertStreamCounters {
    std::uint64_t bytes = 0;       // bytes enqueued into slots
    std::uint64_t file_bytes = 0;  // part of `bytes` read from the file tier
    double io_seconds = 0.0;       // host time inside file reads
    ExpertStreamCounters& operator+=(const ExpertStreamCounters& o) noexcept {
        bytes += o.bytes;
        file_bytes += o.file_bytes;
        io_seconds += o.io_seconds;
        return *this;
    }
};

// What a loader job belongs to, captured under the owner's mutex when the job starts.
struct ExpertStreamJob {
    std::uint64_t epoch = 0;  // run job: the run it serves; preload job: the run it followed
    bool run = false;         // false: the idle preload between runs
};

class ExpertStreamEpochs {
public:
    // begin_run: `epoch` (the owner's run id, increasing) is open and its counters start at zero.
    void open(std::uint64_t epoch) noexcept {
        epoch_ = epoch;
        open_ = true;
        run_ = {};
    }
    // end_run: the epoch's counters are final from here on.
    void close() noexcept { open_ = false; }
    [[nodiscard]] bool is_open() const noexcept { return open_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
    [[nodiscard]] ExpertStreamJob run_job() const noexcept { return {epoch_, true}; }
    // A new idle preload (it follows the current epoch): the previous preload's counters are gone from here on.
    [[nodiscard]] ExpertStreamJob start_preload() noexcept {
        preload_ = {};
        preload_epoch_ = epoch_;
        preload_valid_ = true;
        return {epoch_, false};
    }

    // A run job counts only for its own epoch while that epoch is open; a preload job only for the latest preload.
    // Anything else is a late publication (a run ended early, e.g. on an error path, or a superseded preload) and is
    // dropped: it must not enter another job's receipt.
    // Returns whether other job-local receipt fields may be published under this same lock.
    bool publish(const ExpertStreamJob& job, const ExpertStreamCounters& c) noexcept {
        const bool current = job.run ? (open_ && job.epoch == epoch_) : (preload_valid_ && job.epoch == preload_epoch_);
        if (current) (job.run ? run_ : preload_) += c;
        else ++dropped_;
        return current;
    }

    // The open epoch's counters so far, or the last closed epoch's final counters.
    [[nodiscard]] const ExpertStreamCounters& run() const noexcept { return run_; }
    // The last preload's counters and the run it followed (valid: a preload published at least once).
    [[nodiscard]] const ExpertStreamCounters& preload() const noexcept { return preload_; }
    [[nodiscard]] bool preload_followed(std::uint64_t epoch) const noexcept { return preload_valid_ && preload_epoch_ == epoch; }
    [[nodiscard]] std::uint64_t dropped() const noexcept { return dropped_; }  // run publications that came too late

private:
    std::uint64_t epoch_ = 0, preload_epoch_ = 0, dropped_ = 0;
    bool open_ = false, preload_valid_ = false;
    ExpertStreamCounters run_{}, preload_{};
};

}  // namespace strix
