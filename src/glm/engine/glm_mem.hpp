#pragma once
// GLM engine: host memory accounting for the rank load (design notes). Host-only (no HIP): /proc + amdgpu sysfs
// snapshots, the exact allocation plan of a load (LoadPlan, filled by Lib's dry run), the physical cost model of one
// hipMalloc (AllocModel, calibrated at load time), the admission check that fails closed BEFORE allocating, and a
// monitor thread that writes a memory trace line every second and aborts the process when the node nears the OOM line.
//
// Why: load measurement (2026-10-25 07:24) drove both nodes into the global OOM with a 77 GiB logical rank slice: every expert linear is
// one 2,107,392 B hipMalloc and the kernel's gpu_active reached 107.8 GiB on node 1 (no other tenant), i.e. each one
// cost ~4 MiB. Nothing failed cleanly because GTT pages come out of system RAM: the kernel swaps and OOM-kills first.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::glm::engine::mem {

inline constexpr std::uint64_t kKiB = 1024, kMiB = kKiB * 1024, kGiB = kMiB * 1024;

inline std::uint64_t round_up(std::uint64_t b, std::uint64_t g) { return g ? (b + g - 1) / g * g : b; }
inline double gib(std::uint64_t b) { return static_cast<double>(b) / static_cast<double>(kGiB); }

// One snapshot of the node and of this process (bytes). gpu == false when no amdgpu mem_info_* files were found;
// meminfo_ok / cma_free_ok / zoneinfo_ok say whether MemAvailable, CmaFree and the zone CMA layout were read (missing
// values are 0); gpu requires gtt_total, gtt_used AND vram_used (the combined device measurement needs both counters).
struct Snapshot {
    double t = 0;  // seconds since the Probe was created
    bool meminfo_ok = false, cma_free_ok = false, zoneinfo_ok = false;
    std::uint64_t mem_total = 0, mem_available = 0, mem_free = 0, cached = 0, file_lru = 0, cma_free = 0, shmem = 0;
    std::uint64_t cma_total = 0;    // MIGRATE_CMA pageblocks (/proc/zoneinfo 'cma'): KHO scratch here, ~13.16 GiB, CmaTotal 0
    std::uint64_t gpu_active = 0;   // /proc/meminfo GPUActive (this kernel): GPU-backed pages, all devices
    std::uint64_t swap_total = 0, swap_free = 0;
    double psi_full10 = 0;          // /proc/pressure/memory 'full avg10' (% of time every task stalled on memory)
    std::uint64_t rss = 0, rss_anon = 0, rss_file = 0, rss_shmem = 0, hwm = 0;
    bool gpu = false;
    std::uint64_t gtt_total = 0, gtt_used = 0, vram_total = 0, vram_used = 0;
    // RAM that can still back unmovable pages (TTM / GTT pool pages are never __GFP_MOVABLE): MemAvailable minus the free
    // CMA pageblocks, minus the reclaimable file pages that may sit inside CMA pageblocks (reclaiming them frees CMA only;
    // without the zone layout all file pages are assumed to sit there). 0 without /proc/meminfo.
    // Without CmaFree but with CMA in the zone layout, the whole CMA area is subtracted (conservative); with neither,
    // the headroom is 0 (admission refuses, the watchdog fires).
    std::uint64_t ram_headroom() const {
        if (!meminfo_ok || (!cma_free_ok && !zoneinfo_ok)) return 0;  // free CMA unknowable: nothing is trusted
        const std::uint64_t cfree = cma_free_ok ? cma_free : zoneinfo_ok ? cma_total : 0;
        const std::uint64_t cma_used = !zoneinfo_ok ? file_lru : cma_total > cfree ? cma_total - cfree : 0;
        const std::uint64_t off = cfree + std::min(cma_used, file_lru);
        return mem_available > off ? mem_available - off : 0;
    }
    std::uint64_t gtt_headroom() const { return gtt_total > gtt_used ? gtt_total - gtt_used : 0; }
    // Device memory in use as the GPU driver counts it: GTT + the VRAM carve-out (hipMalloc may land in either; the
    // carve-out is 512 MiB here, ~160 MiB used at idle). Without amdgpu counters: the kernel's GPUActive.
    std::uint64_t dev_used() const { return gpu ? gtt_used + vram_used : gpu_active; }
    bool dev_counters() const { return gpu || gpu_active > 0; }
    std::uint64_t swap_used() const { return swap_total > swap_free ? swap_total - swap_free : 0; }
};

struct Paths {
    std::string meminfo = "/proc/meminfo";
    std::string status = "/proc/self/status";
    std::string drm_device;  // directory holding mem_info_gtt_total etc.; empty = $STRIX_GLM_DRM_DEVICE or the first card
    std::string zoneinfo = "/proc/zoneinfo";
    std::string psi = "/proc/pressure/memory";
};

class Probe {
public:
    explicit Probe(Paths p = {});
    Snapshot read() const;
    const Paths& paths() const { return p_; }

private:
    Paths p_;
    double t0_ = 0;
};

class BudgetError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A planned consumer: gtt = GPU allocations (GTT; on this APU they are system RAM too), host = host-only RAM.
struct Term {
    std::string name;
    std::uint64_t gtt = 0, host = 0;
};
struct Margins {
    std::uint64_t ram = 8 * kGiB;  // RAM left for the OS, sshd, the peer link, other users (STRIX_GLM_MEM_MARGIN_GIB)
    std::uint64_t gtt = 2 * kGiB;  // GTT left under the ttm limit (STRIX_GLM_GTT_MARGIN_GIB)
    static Margins from_env();
};
// Throws BudgetError (every term + both headrooms in the message) unless
//   sum(gtt) + margins.gtt <= gtt_headroom (when the GPU counters exist) and sum(gtt + host) + margins.ram <= ram_headroom.
void admit(const std::string& what, const std::vector<Term>& terms, const Snapshot& s, const Margins& m);
std::string describe(const Snapshot& s);

// Physical cost of one device allocation: round_up(bytes, granule), or bytes * ratio when no granule explains the
// calibration (then rounded to 4 KiB).
struct AllocModel {
    std::uint64_t granule = 4096;
    double ratio = 0;  // > 0: ratio model
    bool calibrated = false;
    std::uint64_t physical(std::uint64_t bytes) const;
    std::string str() const;
};
// From a calibration: `measured` bytes per allocation of `size` (delta of gtt_used over K touched allocations / K).
// Picks the smallest granule in {4 KiB, 64 KiB, 2 MiB} whose rounding covers `measured` (0.5 % noise allowed); above
// the 2 MiB rounding it switches to a ratio model; below 0.9 x size the measurement is not trusted (2 MiB granule).
AllocModel infer_alloc_model(std::uint64_t size, double measured);
// From retained calibration batches: per-batch measured bytes per allocation (each batch's delta / its count). The
// batches agree when (max - min) <= 10 % of max + 64 KiB; the model is inferred from the LARGEST batch (an allocator that
// reuses a freed fragment in one batch can only make that batch look cheaper). `consistent` = false when they disagree:
// the caller repeats once, then uses this function's conservative answer (the worse of max and the 2 MiB granule).
AllocModel model_from_batches(std::uint64_t size, const std::vector<double>& per_batch, bool& consistent);

// Size of the single allocation strix_weights_prepare_parts makes for one linear (strix-trellis prepare_regions:
// trellis, suh, svh, bias regions, each start aligned to STRIX_TRELLIS_WORKSPACE_ALIGNMENT = 256).
std::uint64_t prepared_bytes(std::uint64_t tr, std::uint64_t suh, std::uint64_t svh, std::uint64_t bias = 0, std::uint64_t align = 256);

// Every device allocation of one load, in load order, filled by Lib's dry run (no file read, nothing allocated).
struct LoadPlan {
    std::vector<std::uint64_t> device;  // requested bytes of every hipMalloc (arena chunks, prepared linears, state, ...)
    std::vector<std::size_t> mark;      // device.size() after each load step (step_name[i])
    std::vector<std::string> step_name;
    std::uint64_t logical = 0;          // sum of device
    std::uint64_t bytes_read = 0;       // file bytes the load will read
    std::uint64_t stage_max = 0;        // largest pinned staging request
    std::uint64_t whole_max = 0;        // largest pageable whole-tensor read (output slices of a non-full range)
    std::uint64_t bounce = 0;           // O_DIRECT bounce buffer
    void add(std::uint64_t bytes) {
        device.push_back(bytes);
        logical += bytes;
    }
    void step(std::string name) {
        mark.push_back(device.size());
        step_name.push_back(std::move(name));
    }
    std::uint64_t dominant_size() const;  // the most frequent allocation size (the expert linear on GLM)
    std::uint64_t physical(const AllocModel& m, std::size_t upto = std::numeric_limits<std::size_t>::max()) const;
    std::uint64_t host_bytes(std::uint64_t pinned_stage) const { return pinned_stage + whole_max + bounce; }
    std::string summary(const AllocModel& m) const;
};

// Watchdog thresholds (bytes; 0 = off). Defaults: abort when the RAM headroom (Snapshot::ram_headroom) falls under
// 4 GiB (GTT fills at >= 2 GiB/s during a load: implementation), swap use grows 2 GiB over its value at start,
// this process's RSS passes 8 GiB, memory PSI 'full avg10' reaches 25 % (thrash), or the device memory used (GTT +
// VRAM) passes the limit the engine sets for the load only (use at the load start + plan physical + 2 GiB).
struct Watchdog {
    bool enabled = true;                         // STRIX_GLM_WATCHDOG=0 turns it off
    std::uint64_t min_ram_headroom = 4 * kGiB;  // STRIX_GLM_WD_MIN_FREE_GIB
    std::uint64_t max_swap_growth = 2 * kGiB;   // STRIX_GLM_WD_SWAP_GIB
    std::uint64_t max_rss = 8 * kGiB;           // STRIX_GLM_WD_RSS_GIB
    std::uint64_t max_gtt_used = 0;             // STRIX_GLM_WD_GTT_GIB (absolute, device = GTT + VRAM); else the engine's load limit
    double max_psi_full10 = 25;                  // STRIX_GLM_WD_PSI_FULL (percent; 0 = off)
    static Watchdog from_env();
};

// Two threads. The sampler takes a snapshot + watchdog check every `sample_ms` and queues one trace line every
// `trace_ms`; it never does I/O beyond reading /proc and sysfs. A writer thread drains the bounded queue (256 lines;
// overflow is counted, never blocks the sampler) with one write(2) per line on an O_APPEND fd and reports write failures
// on stderr once. On a breach the sampler queues the reason (trace + stderr), waits at most `abort_grace_ms` for the
// writer, then calls on_abort(reason) (default: _Exit(86)) - a blocked trace file or stderr cannot stop the abort.
// Telemetry loss is a breach: /proc/meminfo unreadable, or the amdgpu counters gone after being present at start.
// The kernel OOM killer never picks a GPU process on its own (GTT is invisible in RSS: load measurement lost sshd and the desktop
// first). oom_score_adj 1000 makes this process the first victim. Returns false when /proc refused the write.
bool set_oom_score_adj(int v);

class Monitor {
public:
    using AbortFn = std::function<void(const std::string&)>;
    Monitor(Probe probe, std::string trace_path, Watchdog wd, AbortFn on_abort = {}, unsigned sample_ms = 100, unsigned trace_ms = 1000,
            unsigned abort_grace_ms = 300);
    ~Monitor();
    Monitor(const Monitor&) = delete;
    Monitor& operator=(const Monitor&) = delete;
    void phase(const std::string& p);
    void counters(std::uint64_t device_logical, std::uint64_t bytes_read) {
        dev_.store(device_logical);
        read_.store(bytes_read);
    }
    void set_max_gtt_used(std::uint64_t b) { max_gtt_.store(b); }
    bool tripped() const { return tripped_.load(); }
    std::uint64_t lines() const { return lines_.load(); }  // trace lines queued
    std::uint64_t write_errors() const;
    std::uint64_t dropped() const;
    struct Sink;  // writer-thread state, shared with that thread (it may outlive the Monitor when its fd is stuck)
    const Snapshot& start() const { return start_; }
    static std::string header();
    std::string line(const Snapshot& s) const;

private:
    void run();
    bool push(int fd, std::string s);  // false = dropped (queue full)
    void stop_writer();               // stop + join the writer, or detach it when its write is stuck (<= 2 s)
    std::string check(const Snapshot& s) const;  // empty = fine, else the breach
    Probe probe_;
    std::string trace_path_;
    Watchdog wd_;
    AbortFn on_abort_;
    unsigned sample_ms_, trace_ms_, grace_ms_;
    Snapshot start_;
    std::shared_ptr<Sink> sink_;
    std::thread writer_;
    mutable std::mutex mu_;
    std::string phase_ = "init";
    std::atomic<std::uint64_t> dev_{0}, read_{0}, max_gtt_{0}, lines_{0};
    std::atomic<bool> stop_{false}, tripped_{false};
    std::thread th_;
};

}  // namespace strix::glm::engine::mem
