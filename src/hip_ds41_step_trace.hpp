#pragma once
// STRIX_DS41_STEP_TRACE: host/device timeline of DeviceForward::step_window, one CSV per process (default off).
//   =1  host steady_clock segments (inputs, Engram hash/read/pack/upload, every sync, every plan step's enqueue with its
//       host time in the communicator, the end sync, op-timing collection) plus one device event after every plan step
//       (the op-timing events; created here when op_timing is off).
//   =2  the same host segments, but only two device events (step start / end): measures the cost of the per-step events.
// STRIX_DS41_STEP_TRACE_OUT=FILE (default ds41-step-trace-r<rank>.csv in the working directory).
// Rows: step,rows,kind,name,layer,site,host_t0_us,host_t1_us,comm_us,gpu_t_us,step_wall_us,events_us
//   host times relative to step_window's entry; gpu_t_us = device time of the row's end event relative to the first
//   event, shifted by that event's host record time (the device is idle there: it follows a stream synchronisation).
// Nothing here changes what is enqueued or computed; with the variable unset no StepTrace exists.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace strix::ds41 {

class StepTrace {
public:
    using clock = std::chrono::steady_clock;
    static int env_level() {
        const char* v = std::getenv("STRIX_DS41_STEP_TRACE");
        return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
    }
    StepTrace(int level, std::uint32_t rank) : level_(level) {
        const char* o = std::getenv("STRIX_DS41_STEP_TRACE_OUT");
        const std::string path = o != nullptr && *o != 0 ? std::string(o) : "ds41-step-trace-r" + std::to_string(rank) + ".csv";
        out_ = std::fopen(path.c_str(), "w");
        if (out_ != nullptr) std::fprintf(out_, "step,rows,kind,name,layer,site,host_t0_us,host_t1_us,comm_us,gpu_t_us,step_wall_us,events_us\n");
        (void)hipEventCreate(&begin_);
        (void)hipEventCreate(&end_);
    }
    ~StepTrace() {
        if (out_ != nullptr) std::fclose(out_);
        (void)hipEventDestroy(begin_);
        (void)hipEventDestroy(end_);
    }
    StepTrace(const StepTrace&) = delete;
    StepTrace& operator=(const StepTrace&) = delete;
    [[nodiscard]] int level() const noexcept { return level_; }

    void begin(std::uint32_t rows) {
        rows_ = rows;
        origin_ = last_ = clock::now();
        segs_.clear();
        ev0_us_ = -1.0;
    }
    // host segment from the previous mark to now
    void mark(const char* name) {
        const auto t = clock::now();
        segs_.push_back({"host", name, -1, {}, us(last_), us(t), 0.0, -1});
        last_ = t;
    }
    // the first device event was just recorded (call after hipEventRecord)
    void first_event(hipStream_t st) {
        if (level_ == 2) (void)hipEventRecord(begin_, st);
        mark("ev0_record");
        ev0_us_ = segs_.back().t1;
    }
    // a plan step's enqueue, from the previous mark; event = index of its end event (level 1), comm in seconds
    void op(std::string_view name, int layer, const std::string& site, double comm_seconds, int event) {
        const auto t = clock::now();
        segs_.push_back({"op", std::string(name), layer, site, us(last_), us(t), comm_seconds * 1e6, level_ == 1 ? event : -1});
        last_ = t;
    }
    // a blocking synchronisation inside a plan step (its time is also inside that step's op row)
    void sync(const char* name, clock::time_point t0, int layer) {
        segs_.push_back({"sync", name, layer, {}, us(t0), us(clock::now()), 0.0, -1});
    }
    void last_event(hipStream_t st) {
        if (level_ == 2) (void)hipEventRecord(end_, st);
    }
    // after the step's wall time is taken: device times, then the rows of this step
    void finish(const std::vector<hipEvent_t>& events, double step_seconds) {
        double events_us = 0.0;
        std::vector<double> gpu(segs_.size(), -1.0);
        for (std::size_t i = 0; i < segs_.size(); ++i) {
            if (segs_[i].event <= 0 || static_cast<std::size_t>(segs_[i].event) >= events.size()) continue;
            float ms = 0.0F;
            if (hipEventElapsedTime(&ms, events[0], events[static_cast<std::size_t>(segs_[i].event)]) == hipSuccess)
                gpu[i] = ev0_us_ + ms * 1e3;
            events_us = ms * 1e3;
        }
        if (level_ == 2) {
            float ms = 0.0F;
            if (hipEventElapsedTime(&ms, begin_, end_) == hipSuccess) events_us = ms * 1e3;
        }
        if (out_ != nullptr) {
            for (std::size_t i = 0; i < segs_.size(); ++i) {
                const auto& s = segs_[i];
                std::fprintf(out_, "%llu,%u,%s,%s,%d,%s,%.1f,%.1f,%.1f,%.1f,,\n", static_cast<unsigned long long>(step_), rows_, s.kind,
                             s.name.c_str(), s.layer, s.site.c_str(), s.t0, s.t1, s.comm, gpu[i]);
            }
            std::fprintf(out_, "%llu,%u,step,step,-1,,0,%.1f,0,%.1f,%.1f,%.1f\n", static_cast<unsigned long long>(step_), rows_,
                         us(clock::now()), ev0_us_, step_seconds * 1e6, events_us);
        }
        ++step_;
    }

private:
    struct Seg {
        const char* kind;
        std::string name;
        int layer;
        std::string site;
        double t0, t1, comm;
        int event;
    };
    double us(clock::time_point t) const { return std::chrono::duration<double, std::micro>(t - origin_).count(); }
    int level_ = 0;
    std::FILE* out_ = nullptr;
    hipEvent_t begin_ = nullptr, end_ = nullptr;
    std::uint32_t rows_ = 0;
    std::uint64_t step_ = 0;
    clock::time_point origin_{}, last_{};
    double ev0_us_ = -1.0;
    std::vector<Seg> segs_;
};

}  // namespace strix::ds41
