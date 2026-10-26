// GLM-5.3-Flash TP2 exchange module: the engine (see glm_xfer.hpp; copied from minimax/xfer 3851376c, names only).
#include "glm_xfer.hpp"
#include "glm_xfer_device.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>

namespace strix::glm::xfer {
namespace {

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glmxfer: ") + what + ": " + hipGetErrorString(e));
}

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

constexpr std::size_t kTxFlag = 0, kRxFlag = 8;  // u64 index: separate 64-byte lines

__global__ void publish_kernel(std::uint64_t* flag, std::uint64_t sequence) {
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __hip_atomic_store(flag, sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

constexpr unsigned kCopyThreads = 256;
__global__ __launch_bounds__(kCopyThreads) void copy_publish_kernel(std::byte* dst, const std::byte* src, std::size_t bytes,
                                                                    ProducerTarget target) {
    const std::size_t vec = bytes / 16, stride = static_cast<std::size_t>(gridDim.x) * kCopyThreads;
    auto* d = reinterpret_cast<uint4*>(dst);
    const auto* s = reinterpret_cast<const uint4*>(src);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * kCopyThreads + threadIdx.x; i < vec; i += stride) d[i] = s[i];
    if (blockIdx.x == 0)
        for (std::size_t i = vec * 16 + threadIdx.x; i < bytes; i += kCopyThreads) dst[i] = src[i];
    glmx_publish(target, gridDim.x);
}

constexpr unsigned kArgThreads = 1024;
__device__ __forceinline__ bool better(float v, std::uint32_t i, float bv, std::uint32_t bi) { return v > bv || (v == bv && i < bi); }

__global__ __launch_bounds__(kArgThreads) void argmax_candidate_kernel(const float* logits, unsigned vn, unsigned ld, std::uint32_t id0,
                                                                       std::uint32_t* tx, ProducerTarget target) {
    __shared__ float sv[kArgThreads];
    __shared__ std::uint32_t si[kArgThreads];
    __shared__ unsigned bad;
    const unsigned t = threadIdx.x;
    if (t == 0) bad = 0;
    __syncthreads();
    const float* l = logits + static_cast<std::size_t>(blockIdx.x) * ld;
    float bv = -__builtin_inff();
    std::uint32_t bi = 0xFFFFFFFFU;
    bool finite = true;
    for (unsigned j = t; j < vn; j += kArgThreads) {
        const float v = l[j];
        finite &= __builtin_isfinite(v);
        if (better(v, j, bv, bi)) {
            bv = v;
            bi = j;
        }
    }
    if (!finite) atomicOr(&bad, 1U);
    sv[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (unsigned s = kArgThreads / 2; s >= 1; s >>= 1U) {
        if (t < s && better(sv[t + s], si[t + s], sv[t], si[t])) {
            sv[t] = sv[t + s];
            si[t] = si[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        tx[2 * blockIdx.x] = __float_as_uint(sv[0]);
        tx[2 * blockIdx.x + 1] = bad != 0 || si[0] == 0xFFFFFFFFU ? kNonFinite : id0 + si[0];
    }
    glmx_publish(target, gridDim.x);
}

__global__ void argmax_pick_kernel(const std::uint32_t* c0, const std::uint32_t* c1, unsigned rows, std::uint32_t* out) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const std::uint32_t i0 = c0[2 * r + 1], i1 = c1[2 * r + 1];
    const float v0 = __uint_as_float(c0[2 * r]), v1 = __uint_as_float(c1[2 * r]);
    out[r] = i0 == kNonFinite || i1 == kNonFinite ? kNonFinite : v1 > v0 ? i1 : i0;  // tie: rank 0 (lower ids)
}

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::stoull(v) : fallback;
}

}  // namespace

// One thread computing CRC-32C of payloads in submission order (deferred CRCs). It spins while work may come and naps
// after an idle period; submit() and wait() are lock-free (one producer: the progress thread).
class Exchanger::CrcWorker {
public:
    explicit CrcWorker(int cpu) : thread_([this] { run(); }) {
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
        }
    }
    ~CrcWorker() {
        stop_ = true;
        thread_.join();
    }
    std::uint64_t submit(const std::byte* p, std::size_t n) {
        const std::uint64_t t = submitted_.load(std::memory_order_relaxed) + 1;
        while (t - completed_.load(std::memory_order_acquire) > kRing) _mm_pause();
        Task& k = ring_[t % kRing];
        k.p = p;
        k.n = n;
        submitted_.store(t, std::memory_order_release);
        return t;
    }
    std::uint32_t wait(std::uint64_t ticket) {
        while (completed_.load(std::memory_order_acquire) < ticket) _mm_pause();
        return ring_[ticket % kRing].crc;
    }

private:
    static constexpr std::uint64_t kRing = 8;
    struct Task {
        const std::byte* p = nullptr;
        std::size_t n = 0;
        std::uint32_t crc = 0;
    };
    void run() {
        auto last = Clock::now();
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::uint64_t c = completed_.load(std::memory_order_relaxed);
            if (c < submitted_.load(std::memory_order_acquire)) {
                Task& k = ring_[(c + 1) % kRing];
                k.crc = dist::crc32c(std::span<const std::byte>(k.p, k.n));
                completed_.store(c + 1, std::memory_order_release);
                last = Clock::now();
            } else if (Clock::now() - last < std::chrono::milliseconds(3)) {
                _mm_pause();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }
    Task ring_[kRing];
    std::atomic<std::uint64_t> submitted_{0}, completed_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

void Options::from_env() {
    depth = static_cast<unsigned>(env_u64("STRIX_GLMX_DEPTH", depth));
    if (const char* v = std::getenv("STRIX_GLMX_PUBLISH"); v != nullptr && *v != '\0') {
        const std::string s(v);
        if (s == "kernel") publish = Publish::kernel;
        else if (s == "write_value") publish = Publish::write_value;
        else throw std::invalid_argument("STRIX_GLMX_PUBLISH must be kernel or write_value");
    }
    if (const char* v = std::getenv("STRIX_GLMX_CRC"); v != nullptr && *v != '\0') {
        const std::string c(v);
        if (c == "0") crc = false;
        else if (c == "1") crc = true, crc_deferred = false;
        else if (c == "deferred") crc = true, crc_deferred = true;
        else throw std::invalid_argument("STRIX_GLMX_CRC must be 0, 1 or deferred");
    }
    auto memory = [](const char* name, Memory fallback) {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return fallback;
        const std::string m(v);
        if (m == "host") return Memory::host_coherent;
        if (m == "device") return Memory::device_fine;
        if (m == "uncached") return Memory::device_uncached;
        throw std::invalid_argument(std::string(name) + " must be host or device");
    };
    tx_memory = memory("STRIX_GLMX_TX_MEM", tx_memory);
    rx_memory = memory("STRIX_GLMX_RX_MEM", rx_memory);
    if (const char* v = std::getenv("STRIX_GLMX_PROGRESS_CPU"); v != nullptr && *v != '\0') progress_cpu = std::atoi(v);
    if (const char* v = std::getenv("STRIX_GLMX_CRC_CPU"); v != nullptr && *v != '\0') crc_cpu = std::atoi(v);
    test_corrupt_sequence = env_u64("STRIX_GLMX_TEST_CORRUPT", test_corrupt_sequence);
    idle_spin_us = static_cast<unsigned>(env_u64("STRIX_GLMX_IDLE_SPIN_US", idle_spin_us));
    timeout = std::chrono::milliseconds(env_u64("STRIX_GLMX_TIMEOUT_MS", static_cast<std::uint64_t>(timeout.count())));
}

Exchanger::Exchanger(std::unique_ptr<dist::Transport> owned, Options options)
    : owned_(std::move(owned)), transport_(owned_.get()), options_(options) {
    init();
}

Exchanger::Exchanger(dist::Transport& borrowed, Options options) : transport_(&borrowed), options_(options) { init(); }

void Exchanger::init() {
    if (transport_ == nullptr) throw std::invalid_argument("glmxfer: no transport");
    if (options_.rank > 1) throw std::invalid_argument("glmxfer: rank must be 0 or 1");
    if (options_.capacity == 0 || options_.capacity > 0xFFFFFFFFULL) throw std::invalid_argument("glmxfer: capacity");
    if (options_.depth < 2 || options_.depth > 64) throw std::invalid_argument("glmxfer: depth must be 2..64");
    stride_ = (options_.capacity + sizeof(Trailer) + 4095) & ~std::size_t{4095};
    const std::size_t total = stride_ * options_.depth;
    const unsigned flags = hipHostMallocMapped | hipHostMallocCoherent;
    auto slots = [&](Memory m, std::byte** host, std::byte** dev, void** alloc, const char* what) {
        if (m != Memory::host_coherent) {
            hip_check(hipExtMallocWithFlags(alloc, total, m == Memory::device_fine ? hipDeviceMallocFinegrained : hipDeviceMallocUncached), what);
            *host = *dev = static_cast<std::byte*>(*alloc);
        } else {
            hip_check(hipHostMalloc(alloc, total, flags), what);
            *host = static_cast<std::byte*>(*alloc);
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(dev), *host, 0), what);
        }
    };
    slots(options_.tx_memory, &tx_host_, &tx_dev_, &tx_alloc_, "TX slots");
    slots(options_.rx_memory, &rx_host_, &rx_dev_, &rx_alloc_, "RX slots");
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&flags_host_), 4096, flags), "flags");
    hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&flags_dev_), flags_host_, 0), "flags");
    std::memset(tx_host_, 0, total);
    std::memset(rx_host_, 0, total);
    std::memset(flags_host_, 0, 4096);
    hip_check(hipMalloc(reinterpret_cast<void**>(&counter_dev_), 64), "counter");
    hip_check(hipMemset(counter_dev_, 0, 64), "counter");
    hip_check(hipMalloc(reinterpret_cast<void**>(&gate_dev_), 64), "gate");
    hip_check(hipMemset(gate_dev_, 0, 64), "gate");
    if (options_.crc && options_.crc_deferred) crc_worker_ = std::make_unique<CrcWorker>(options_.crc_cpu);
    thread_ = std::thread([this] { run(); });
    if (options_.progress_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(options_.progress_cpu, &set);
        pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
    }
}

Exchanger::~Exchanger() {
    if (done_.load() < posted_.load() && transport_ != nullptr) transport_->abort("glmxfer: closing with exchanges in flight");
    stop_ = true;
    {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    if (flags_host_ != nullptr) {
        __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);  // release any waiting stream
        (void)hipStreamSynchronize(options_.stream);
    }
    crc_worker_.reset();
    (void)hipFree(counter_dev_);
    (void)hipFree(gate_dev_);
    (void)hipHostFree(flags_host_);
    if (options_.rx_memory != Memory::host_coherent) (void)hipFree(rx_alloc_);
    else (void)hipHostFree(rx_alloc_);
    if (options_.tx_memory != Memory::host_coherent) (void)hipFree(tx_alloc_);
    else (void)hipHostFree(tx_alloc_);
}

std::byte* Exchanger::tx_slot(std::uint64_t sequence) const { return tx_host_ + ((sequence - 1) % options_.depth) * stride_; }
std::byte* Exchanger::rx_slot(std::uint64_t sequence) const { return rx_host_ + ((sequence - 1) % options_.depth) * stride_; }

void* Exchanger::tx() {
    const std::uint64_t s = next_ + 1;
    if (s > received_ + options_.depth) throw std::logic_error("glmxfer: tx() of a sequence whose slot is still in use (recv() of s - depth missing)");
    tx_taken_ = s;
    return tx_dev_ + (tx_slot(s) - tx_host_);
}

ProducerTarget Exchanger::producer_target() {
    return ProducerTarget{flags_dev_ + kTxFlag, counter_dev_, next_ + 1};
}

void Exchanger::send(std::size_t bytes, const Tag& tag, Publish how) {
    check();
    if (bytes == 0 || bytes > options_.capacity) throw std::invalid_argument("glmxfer: message size outside 1..capacity");
    const std::uint64_t s = next_ + 1;
    if (s > received_ + options_.depth) throw std::logic_error("glmxfer: send() of a sequence whose slot is still in use");
    if (how == Publish::kernel) {
        hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(32), 0, options_.stream, flags_dev_ + kTxFlag, s);
        hip_check(hipGetLastError(), "publish kernel");
    } else if (how == Publish::write_value) {
        hip_check(hipStreamWriteValue64(options_.stream, flags_dev_ + kTxFlag, s, 0), "publish write");
    }
    next_ = s;
    tail_deferred_ = options_.crc && options_.crc_deferred && bytes > kInlineCrcMax;
    post(Job{s, bytes, tag, false});
}

void Exchanger::post(const Job& job) {
    while (jobs_posted_ - done_.load(std::memory_order_acquire) >= kQueue) {
        check();
        _mm_pause();
    }
    jobs_[jobs_posted_ % kQueue] = job;
    ++jobs_posted_;
    posted_.store(jobs_posted_, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_one();
    }
}

void Exchanger::send_copy(const void* local, std::size_t bytes, const Tag& tag) {
    auto* dst = static_cast<std::byte*>(tx());
    const unsigned blocks = static_cast<unsigned>(std::clamp<std::size_t>((bytes + 16 * kCopyThreads - 1) / (16 * kCopyThreads), 1, 64));
    hipLaunchKernelGGL(copy_publish_kernel, dim3(blocks), dim3(kCopyThreads), 0, options_.stream, dst,
                       static_cast<const std::byte*>(local), bytes, producer_target());
    hip_check(hipGetLastError(), "copy kernel");
    send(bytes, tag, Publish::producer);
}

void argmax_exchange(Exchanger& x, const float* logits, unsigned rows, unsigned vn, unsigned ld, std::uint32_t id0, std::uint64_t step,
                     std::uint32_t* out_ids) {
    if (rows == 0 || vn == 0 || ld < vn) throw std::invalid_argument("glmxfer argmax: bad shape");
    auto* tx = static_cast<std::uint32_t*>(x.tx());
    const hipStream_t s = x.options().stream;
    hipLaunchKernelGGL(argmax_candidate_kernel, dim3(rows), dim3(kArgThreads), 0, s, logits, vn, ld, id0, tx, x.producer_target());
    hip_check(hipGetLastError(), "argmax candidate kernel");
    x.send(std::size_t{rows} * 8, Tag{Kind::argmax, 60, step}, Publish::producer);
    const auto* peer = static_cast<const std::uint32_t*>(x.recv());
    const auto* local = static_cast<const std::uint32_t*>(x.last_local());
    const bool r0 = x.options().rank == 0;
    hipLaunchKernelGGL(argmax_pick_kernel, dim3((rows + 63) / 64), dim3(64), 0, s, r0 ? local : peer, r0 ? peer : local, rows, out_ids);
    hip_check(hipGetLastError(), "argmax pick kernel");
}

const void* Exchanger::recv() {
    const std::uint64_t s = received_ + 1;
    if (s > next_) throw std::logic_error("glmxfer: recv() without a matching send()");
    hip_check(hipStreamWaitValue64(options_.stream, flags_dev_ + kRxFlag, s, hipStreamWaitValueGte, ~std::uint64_t{0}),
              "wait for the peer's message");
    received_ = s;
    last_local_ = tx_dev_ + (tx_slot(s) - tx_host_);
    return rx_dev_ + (rx_slot(s) - rx_host_);
}

RxView Exchanger::recv_view() {
    const std::uint64_t s = received_ + 1;
    if (s > next_) throw std::logic_error("glmxfer: recv_view() without a matching send()");
    received_ = s;
    last_local_ = tx_dev_ + (tx_slot(s) - tx_host_);
    return RxView{rx_dev_ + (rx_slot(s) - rx_host_), flags_dev_ + kRxFlag, s, gate_dev_};
}

void Exchanger::drain() {
    if (tail_deferred_) {  // the last payload's CRC has not travelled yet: one host-only seal message (both ranks)
        tail_deferred_ = false;
        post(Job{next_, 0, Tag{Kind::control, 0xFFFFFFFFU, 0}, true});
    }
    const auto deadline = Clock::now() + options_.timeout;
    while (done_.load(std::memory_order_acquire) < posted_.load(std::memory_order_acquire) && !failed_.load()) {
        if (Clock::now() > deadline) {
            transport_->abort("glmxfer: drain deadline");
            throw std::runtime_error("glmxfer: drain: exchanges still in flight after the deadline");
        }
        _mm_pause();
    }
    check();
}

void Exchanger::check() const {
    if (!failed_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(failure_mutex_);
    throw std::runtime_error("glmxfer: exchange failed: " + failure_);
}

Stats Exchanger::stats() const { return stats_; }

std::string Exchanger::name() const {
    const char* pub = options_.publish == Publish::kernel ? "kernel" : options_.publish == Publish::write_value ? "write_value" : "producer";
    return "glmxfer(" + transport_->name() + ", depth " + std::to_string(options_.depth) + ", publish " + pub +
           ", crc " + (options_.crc ? "on" : "off") + ")";
}

void Exchanger::fail(const std::string& why) {
    {
        std::lock_guard<std::mutex> g(failure_mutex_);
        if (failure_.empty()) failure_ = why;
    }
    failed_.store(true, std::memory_order_release);
    __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
    transport_->abort(why);
    throw std::runtime_error(why);
}

void Exchanger::process(const Job& job) {
    const auto t0 = Clock::now();
    const auto deadline = t0 + options_.timeout;
    unsigned n = 0;
    while (!job.seal && __atomic_load_n(&flags_host_[kTxFlag], __ATOMIC_ACQUIRE) < job.sequence) {
        if (stop_.load(std::memory_order_relaxed)) fail("glmxfer: stopped while waiting for a producer");
        if ((++n & 1023U) == 0 && Clock::now() > deadline)
            fail("glmxfer: producer of sequence " + std::to_string(job.sequence) + " not published within the timeout");
        _mm_pause();
    }
    const auto t1 = Clock::now();
    std::byte* tx = job.seal ? seal_tx_ : tx_slot(job.sequence);
    std::byte* rx = job.seal ? seal_rx_ : rx_slot(job.sequence);
    const bool deferred = options_.crc && options_.crc_deferred && job.bytes > kInlineCrcMax;
    std::uint64_t tx_ticket = 0;
    if (deferred) tx_ticket = crc_worker_->submit(tx, job.bytes);  // overlaps the wire below
    Trailer t;
    t.epoch = options_.epoch;
    t.plan = options_.plan;
    t.sequence = job.sequence;
    t.step = job.tag.step;
    t.bytes = static_cast<std::uint32_t>(job.bytes);
    t.kind = static_cast<std::uint16_t>(job.tag.kind);
    t.layer = job.tag.layer;
    t.rank = options_.rank;
    if (deferred) t.flags |= kTrailerDeferred;
    else t.crc32c = options_.crc ? dist::crc32c(std::span<const std::byte>(tx, job.bytes)) : 0U;
    if (prev_deferred_) {
        t.flags |= kTrailerPrev;
        t.prev_crc32c = prev_tx_crc_;
    }
    std::memcpy(tx + job.bytes, &t, sizeof(t));
    try {
        transport_->exchange(std::span<const std::byte>(tx, job.bytes + sizeof(Trailer)),
                             std::span<std::byte>(rx, job.bytes + sizeof(Trailer)));
    } catch (const std::exception& e) {
        fail(std::string("glmxfer: transport: ") + e.what());
    }
    const auto t2 = Clock::now();
    if (!job.seal && job.sequence == options_.test_corrupt_sequence && job.bytes > 0) rx[0] ^= std::byte{1};
    Trailer p;
    std::memcpy(&p, rx + job.bytes, sizeof(p));
    if (p.magic != kTrailerMagic || p.epoch != t.epoch || p.plan != t.plan || p.sequence != t.sequence || p.step != t.step ||
        p.bytes != t.bytes || p.kind != t.kind || p.layer != t.layer || p.rank != 1U - options_.rank || p.flags != t.flags)
        fail("glmxfer: peer message identity differs at sequence " + std::to_string(job.sequence) + " (peer seq " +
             std::to_string(p.sequence) + " kind " + std::to_string(p.kind) + " layer " + std::to_string(p.layer) + " bytes " +
             std::to_string(p.bytes) + ")");
    if (prev_deferred_ && p.prev_crc32c != crc_worker_->wait(prev_rx_ticket_))
        fail("glmxfer: payload CRC mismatch at sequence " + std::to_string(prev_sequence_) + " (deferred check)");
    if (options_.crc && !deferred && p.crc32c != dist::crc32c(std::span<const std::byte>(rx, job.bytes)))
        fail("glmxfer: payload CRC mismatch at sequence " + std::to_string(job.sequence));
    if (deferred) prev_tx_crc_ = crc_worker_->wait(tx_ticket);  // TX slot fully read before the stream may reuse it
    prev_deferred_ = deferred;
    if (job.seal) return;
    __atomic_store_n(&flags_host_[kRxFlag], job.sequence, __ATOMIC_RELEASE);
    if (deferred) {
        prev_rx_ticket_ = crc_worker_->submit(rx, job.bytes);  // after the release: off the critical path
        prev_sequence_ = job.sequence;
    }
    const auto t3 = Clock::now();
    stats_.exchanges += 1;
    stats_.bytes += job.bytes;
    stats_.producer_wait += seconds(t0, t1);
    stats_.wire += seconds(t1, t2);
    stats_.check += seconds(t2, t3);
}

void Exchanger::run() {
    auto last = Clock::now();
    const auto idle = std::chrono::microseconds(options_.idle_spin_us);
    try {
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::uint64_t d = done_.load(std::memory_order_relaxed);
            if (d < posted_.load(std::memory_order_acquire)) {
                process(jobs_[d % kQueue]);
                done_.store(d + 1, std::memory_order_release);
                last = Clock::now();
                continue;
            }
            if (Clock::now() - last < idle) {
                _mm_pause();
                continue;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            sleeping_.store(true, std::memory_order_seq_cst);
            wake_.wait(lock, [&] { return stop_.load() || done_.load() < posted_.load(std::memory_order_seq_cst); });
            sleeping_.store(false, std::memory_order_seq_cst);
            last = Clock::now();
        }
    } catch (const std::exception& e) {
        if (!failed_.load()) {
            std::lock_guard<std::mutex> g(failure_mutex_);
            if (failure_.empty()) failure_ = e.what();
            failed_.store(true);
        }
        __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
    }
}

}  // namespace strix::glm::xfer
