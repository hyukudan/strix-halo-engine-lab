// Energy and power as a shared measurement: host only, model agnostic, no device runtime.
//
// A Sampler owns a list of Sources and a background thread that reads them at a fixed period into a fixed ring of
// Samples. Nothing here needs elevated rights or changes a system setting: a source reads what the machine already
// exposes, and a source that is missing or unreadable reports "unavailable" with the reason and is skipped. A run never
// fails because of this file.
//
// Receipt: summarize(t0, t1, tokens) turns a time window of the ring into energy_j, avg_w, peak_w, tok_per_j, the
// median GPU clock and the maximum temperature. Energy comes from a package energy counter when one is readable
// (counter difference, wrap handled by the source), otherwise from the trapezoid integral of the power samples. The
// window edges are interpolated between the neighbouring samples, so a caller only needs the timestamps it already has.
//
// Process sampler: start_process_sampler() starts one sampler per process (the device gate calls it, so an engine does
// not). request_block() / block_json() build the one shared receipt block "strix-power-receipt-v1" that every CLI and
// the server print. Switch: STRIX_POWER_SAMPLER=0 turns the sampler and every receipt off; STRIX_POWER_SAMPLE_MS sets
// the period (default 100, range 10..10000).
//
// A new device (a discrete card, an accelerator, another OS) is one Source subclass plus one register_source() call;
// no caller changes. See docs/engine/power-receipt.md.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strix::power {

inline constexpr double kAbsent = std::numeric_limits<double>::quiet_NaN();

// One reading of every channel. A channel nobody filled stays NaN.
struct Sample {
    std::int64_t t_ns = 0;        // steady clock
    double energy_j = kAbsent;    // cumulative package energy since the source's first read (sum over packages)
    double power_w = kAbsent;     // instantaneous power, summed over the sources that report a power domain
    double gpu_clock_mhz = kAbsent;
    double temp_c = kAbsent;      // maximum over the sources that report one
    double cpu_mhz = kAbsent;     // mean over the CPU frequency policies
    double gpu_busy_pct = kAbsent;
};

struct SourceStatus {
    std::string name;      // "amdgpu-hwmon", "rapl", "cpufreq", ...
    bool available = false;
    std::string detail;    // available: what is read (channels); unavailable: the reason
};

// A device or counter. probe() runs once before the first read and must not throw; read() adds what it has to the
// sample (energy and power are added to what is there, temperature takes the maximum, the clocks are set when absent)
// and must be cheap: it runs on the sampler thread every period.
class Source {
public:
    virtual ~Source() = default;
    virtual SourceStatus probe() = 0;
    virtual void read(Sample& sample) noexcept = 0;
};

// root is the filesystem prefix of the system tree: empty means the real machine ("/"); a test passes a directory
// holding a fake sys/ tree.
using SourceFactory = std::function<std::unique_ptr<Source>(const std::string& root)>;
// Adds a source kind for every sampler made afterwards. The built-in kinds are registered already.
void register_source(std::string kind, SourceFactory factory);
[[nodiscard]] std::vector<std::unique_ptr<Source>> make_sources(const std::string& root = {});

struct Receipt {
    bool available = false;
    std::string reason;            // when not available
    double duration_ms = 0.0, energy_j = 0.0, avg_w = 0.0, peak_w = 0.0;
    std::uint64_t tokens = 0;
    double tok_per_j = 0.0;        // 0 when tokens or energy is 0
    double gpu_clock_mhz_median = 0.0, temp_c_max = 0.0, cpu_mhz_median = 0.0, gpu_busy_pct_mean = 0.0;
    std::size_t samples = 0;       // samples inside the window
    std::string energy_source;     // "counter" | "power-integral"
};

struct Options {
    int period_ms = 100;
    std::size_t capacity = 1U << 16;  // ring size in samples
};

class Sampler {
public:
    explicit Sampler(std::vector<std::unique_ptr<Source>> sources, Options options = {});
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // True when at least one source is available.
    [[nodiscard]] bool available() const noexcept { return usable_; }
    [[nodiscard]] const std::vector<SourceStatus>& sources() const noexcept { return status_; }

    void start();  // background thread; no-op when nothing is available or already started
    void stop();
    // Reads every source now on the calling thread and stores the sample (used at window edges and by tests, which pass
    // the timestamp).
    void sample_now();
    void sample_at(std::int64_t t_ns);
    [[nodiscard]] Sample latest() const;
    [[nodiscard]] std::size_t sample_count() const;
    // Energy and time over every sample taken so far (not limited by the ring).
    void lifetime(double& joules, double& seconds) const;
    [[nodiscard]] Receipt summarize(std::int64_t t0_ns, std::int64_t t1_ns, std::uint64_t tokens) const;

    [[nodiscard]] static std::int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    [[nodiscard]] static std::int64_t to_ns(std::chrono::steady_clock::time_point t) noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    }

private:
    void run();
    [[nodiscard]] std::vector<Sample> window_locked(std::int64_t t0_ns, std::int64_t t1_ns, std::size_t& inside, bool& lead) const;

    std::vector<std::unique_ptr<Source>> sources_;
    std::vector<SourceStatus> status_;
    std::vector<Source*> live_;
    Options options_;
    bool usable_ = false;
    mutable std::mutex mutex_;       // ring
    std::mutex read_mutex_;          // one reader of the sources at a time
    std::vector<Sample> ring_;
    std::size_t head_ = 0, size_ = 0;
    double lifetime_j_ = 0.0, lifetime_s_ = 0.0;
    std::thread thread_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false, started_ = false;
};

// ---- the shared receipt block -------------------------------------------------------------------------------------

struct Phase {
    double ms = 0.0;           // wall time of the phase
    std::uint64_t tokens = 0;  // tokens the phase produced (prefill: prompt tokens computed; decode: tokens emitted)
};

struct Block {
    bool enabled = false;      // false: STRIX_POWER_SAMPLER=0
    std::vector<SourceStatus> sources;
    Receipt request, prefill, decode;
};

// The process sampler, or null when it is off, not started, or no source is available.
[[nodiscard]] Sampler* process_sampler() noexcept;
// Idempotent. Honors STRIX_POWER_SAMPLER / STRIX_POWER_SAMPLE_MS. Never throws.
void start_process_sampler() noexcept;
void stop_process_sampler() noexcept;
[[nodiscard]] bool sampler_enabled() noexcept;
// Source list of the process sampler (probed even when nothing is available, so a receipt can say why).
[[nodiscard]] std::vector<SourceStatus> process_sources();

// First token of the current request (set once until reset_first_token()); the shared greedy loop calls it.
void mark_first_token() noexcept;
void reset_first_token() noexcept;
[[nodiscard]] std::int64_t first_token_ns() noexcept;  // 0 when not marked

// Block for a request that ended at end_ns: decode is the last decode.ms, prefill the prefill.ms before it.
[[nodiscard]] Block request_block(std::int64_t end_ns, Phase prefill, Phase decode);
// Block for a request [start, stop]: split at the first-token mark when it lies inside (prefill includes the first
// token, as the request timing does; decode is the remaining generated - 1 tokens), otherwise one prefill phase.
[[nodiscard]] Block request_block(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop,
                                  std::uint64_t prompt_tokens, std::uint64_t generated_tokens);
// {"schema":"strix-power-receipt-v1",...}; one line, no trailing newline.
[[nodiscard]] std::string block_json(const Block& block);
// Process view for a status page: sources, the latest reading and the totals since the sampler started.
[[nodiscard]] std::string status_json();

}  // namespace strix::power
