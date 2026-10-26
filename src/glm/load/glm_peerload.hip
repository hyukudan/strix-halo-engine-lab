// GLM TP2 load paths for node 1 (design notes §5): how fast can rank 1
// get its 77.5 GiB of weight bytes into GPU memory, exactly? Throughput proxies on stand-in files (MiniMax EXL3 shards,
// identical on both nodes): read / link / H2D rates and their composition, not a qualification of GLM packing.
//
//   glm-peerload selftest
//       CPU only: the checksum (strix::dist::detail::crc32c_parallel, standard CRC-32C) equals the serial
//       strix::dist::crc32c on random buffers, and detects planted corruptions (review finding two-bit case, random
//       1- and 2-bit flips) in 8 MiB chunks.
//   glm-peerload local TAG GIB FILE...
//       This node alone: O_DIRECT reads of 8 MiB chunks (4 reader threads) into pinned slots, CRC-32C per chunk,
//       hipMemcpyAsync into a device arena (1 GiB, cycled). = the local loader's read path.
//   glm-peerload stride GIB THREADS PIECE GAP FILE...
//       This node alone: O_DIRECT reads of PIECE bytes every PIECE + GAP bytes (expert gate/up rank slices are 8 KiB
//       every 16 KiB) by THREADS threads until GIB of pieces are read; host memory only.
//   glm-peerload pair RANK MODE GIB FILE...      (both ranks, the same FILE list)
//       peer  : rank 0 reads GIB from its disk and streams one 8 MiB + 64 B message per chunk over the USB4 host path
//               (CRC-32C, magic and sequence in the trailer); rank 1 checks every trailer and copies to its device arena.
//       both  : rank 0 reads 2 x GIB ONCE; each 16 MiB read is sliced into 8 KiB pieces, even pieces gathered for
//               itself (H2D to its device), odd pieces gathered for rank 1 (sent) - the read-once, slice-for-both-ranks
//               loader with the expert gate/up interleave; rank 1 receives GIB.
//       local2: both ranks load GIB from their own disks at the same time (the TP2 load with a node 1 copy).
//       After the timed phase (VALIDATION, untimed): the device bytes of the last chunks (<= 1 GiB) are copied back and
//       their CRC compared with the received / read ones; in peer mode rank 0 re-reads its source chunks and rank 1
//       re-reads the same chunks from its own copy of the files, and every chunk CRC is compared.
#include "strix/dist/communicator.hpp"
#include "dist_crc32c_parallel.hpp"

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace dist = strix::dist;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::size_t kChunk = std::size_t{8} << 20U, kTrailer = 64, kSlots = 16, kPiece = 8192;
constexpr unsigned kReaders = 4;
constexpr std::uint64_t kMagic = 0x44414f4c52454550ULL;  // "PEERLOAD"

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
double s_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
std::uint32_t crc(const std::uint8_t* p, std::size_t n) {
    return strix::dist::detail::crc32c_parallel(std::span<const std::byte>(reinterpret_cast<const std::byte*>(p), n));
}

// First exception of any worker; rethrown by the owner after joining.
class Errors {
public:
    void set(std::exception_ptr e) {
        std::lock_guard lk(m_);
        if (!e_) e_ = e;
    }
    void rethrow() {
        std::lock_guard lk(m_);
        if (e_) std::rethrow_exception(e_);
    }
    [[nodiscard]] bool any() {
        std::lock_guard lk(m_);
        return static_cast<bool>(e_);
    }

private:
    std::mutex m_;
    std::exception_ptr e_;
};

struct Chunk {
    unsigned file;
    std::uint64_t off;
};
struct Files {
    std::vector<int> fds;
    std::vector<Chunk> chunks;
    Files(const std::vector<std::string>& paths, double gib) {
        const auto want = static_cast<std::uint64_t>(gib * 1073741824.0) / kChunk;
        for (unsigned f = 0; f < paths.size() && chunks.size() < want; ++f) {
            const int fd = ::open(paths[f].c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (fd < 0) throw std::runtime_error("open " + paths[f]);
            fds.push_back(fd);
            struct stat st{};
            if (::fstat(fd, &st) != 0) throw std::runtime_error("stat " + paths[f]);
            for (std::uint64_t o = 0; o + kChunk <= static_cast<std::uint64_t>(st.st_size) && chunks.size() < want; o += kChunk)
                chunks.push_back({f, o});
        }
        if (chunks.size() < want) throw std::runtime_error("files too small for the requested GiB");
    }
    ~Files() {
        for (int fd : fds) ::close(fd);
    }
    Files(const Files&) = delete;
    Files& operator=(const Files&) = delete;
    void read(std::size_t k, std::uint8_t* dst) const {
        std::size_t done = 0;
        while (done < kChunk) {
            const ssize_t n = ::pread(fds[chunks[k].file], dst + done, kChunk - done, static_cast<off_t>(chunks[k].off + done));
            if (n <= 0) throw std::runtime_error("pread chunk " + std::to_string(k));
            done += static_cast<std::size_t>(n);
        }
    }
};

struct Trailer {
    std::uint64_t magic = kMagic;
    std::uint64_t seq = 0;
    std::uint32_t crc = 0;
    std::uint32_t pad[11]{};
};
static_assert(sizeof(Trailer) == kTrailer);

// A ring of pinned slots [payload kChunk][trailer 64 B]: producers fill sequence numbers, one in-order consumer.
class Ring {
public:
    Ring() {
        for (auto& s : slot_) hip_ok(hipHostMalloc(reinterpret_cast<void**>(&s), kChunk + 4096, hipHostMallocDefault), "pinned slot");
    }
    ~Ring() {
        for (auto* s : slot_) (void)hipHostFree(s);
    }
    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;
    std::uint8_t* slot(std::size_t seq) { return slot_[seq % kSlots]; }
    bool wait_free(std::size_t seq) {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return seq < released_ + kSlots || failed_; });
        return !failed_;
    }
    void filled(std::size_t seq) {
        std::lock_guard lk(m_);
        ready_.push_back(seq);
        cv_.notify_all();
    }
    bool wait_ready(std::size_t seq) {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return std::find(ready_.begin(), ready_.end(), seq) != ready_.end() || failed_; });
        return !failed_;
    }
    void release(std::size_t seq) {
        std::lock_guard lk(m_);
        ready_.erase(std::find(ready_.begin(), ready_.end(), seq));
        released_ = seq + 1;
        cv_.notify_all();
    }
    void fail() {
        std::lock_guard lk(m_);
        failed_ = true;
        cv_.notify_all();
    }

private:
    std::uint8_t* slot_[kSlots]{};
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::size_t> ready_;
    std::size_t released_ = 0;
    bool failed_ = false;
};

// Readers of chunks [0, n) into the ring, CRC + magic + sequence into the trailer. Joined by the destructor.
class Readers {
public:
    Readers(const Files& files, std::size_t n, Ring& ring, Errors& err) {
        for (unsigned i = 0; i < kReaders; ++i)
            t_.emplace_back([&files, n, &ring, &err, this] {
                try {
                    for (;;) {
                        const std::size_t k = next_.fetch_add(1);
                        if (k >= n || !ring.wait_free(k)) return;
                        std::uint8_t* s = ring.slot(k);
                        files.read(k, s);
                        Trailer tr;
                        tr.seq = k;
                        tr.crc = crc(s, kChunk);
                        std::memcpy(s + kChunk, &tr, kTrailer);
                        ring.filled(k);
                    }
                } catch (...) {
                    err.set(std::current_exception());
                    ring.fail();
                }
            });
    }
    ~Readers() {
        for (auto& t : t_) t.join();
    }

private:
    std::atomic<std::size_t> next_{0};
    std::vector<std::thread> t_;
};

// Device sink: in-order H2D copies into a cycled 1 GiB arena; events say when a slot may be reused.
struct DeviceSink {
    static constexpr std::size_t kArena = std::size_t{1} << 30U;
    hipStream_t stream = nullptr;
    std::uint8_t* arena = nullptr;
    hipEvent_t ev[kSlots]{};
    DeviceSink() {
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        hip_ok(hipMalloc(reinterpret_cast<void**>(&arena), kArena), "arena");
        for (auto& e : ev) hip_ok(hipEventCreateWithFlags(&e, hipEventDisableTiming), "event");
    }
    ~DeviceSink() {
        (void)hipStreamSynchronize(stream);
        for (auto& e : ev) (void)hipEventDestroy(e);
        (void)hipFree(arena);
        (void)hipStreamDestroy(stream);
    }
    DeviceSink(const DeviceSink&) = delete;
    DeviceSink& operator=(const DeviceSink&) = delete;
    void copy(const std::uint8_t* src, std::size_t k) {
        hip_ok(hipMemcpyAsync(arena + (k * kChunk) % kArena, src, kChunk, hipMemcpyHostToDevice, stream), "h2d");
        hip_ok(hipEventRecord(ev[k % kSlots], stream), "record");
    }
    void finish(std::size_t k) { hip_ok(hipEventSynchronize(ev[k % kSlots]), "sync"); }
    // VALIDATION: the CRC of the device bytes of chunk k (only the last kArena / kChunk chunks are still resident).
    std::uint32_t device_crc(std::size_t k) {
        static std::vector<std::uint8_t> h(kChunk);
        hip_ok(hipStreamSynchronize(stream), "sync");
        hip_ok(hipMemcpy(h.data(), arena + (k * kChunk) % kArena, kChunk, hipMemcpyDeviceToHost), "d2h");
        return crc(h.data(), kChunk);
    }
};

struct Result {
    double gib = 0, s = 0;
    std::size_t bad = 0;  // trailer failures (magic, sequence or CRC)
    std::vector<std::uint32_t> crcs;
};

// Device validation of the last chunks still in the arena: their device CRC == the recorded one.
std::size_t validate_device(DeviceSink& sink, const std::vector<std::uint32_t>& crcs) {
    const std::size_t n = crcs.size(), keep = std::min<std::size_t>(n, DeviceSink::kArena / kChunk);
    std::size_t bad = 0;
    for (std::size_t k = n - keep; k < n; ++k) bad += sink.device_crc(k) != crcs[k] ? 1 : 0;
    std::printf("VALIDATION device bytes: last %zu of %zu chunks re-read from device memory, %zu CRC differ\n", keep, n, bad);
    return bad;
}

// Local load: read -> pinned -> device.
Result load_local(const Files& files, std::size_t n, bool validate) {
    Ring ring;
    DeviceSink sink;
    Errors err;
    Result r;
    r.crcs.resize(n);
    constexpr std::size_t lag = kSlots / 2;
    const auto t0 = Clock::now();
    {
        Readers readers(files, n, ring, err);  // joined at the end of this scope, after ring.fail() on any error
        try {
            for (std::size_t k = 0; k < n; ++k) {
                if (!ring.wait_ready(k)) break;
                Trailer tr;
                std::memcpy(&tr, ring.slot(k) + kChunk, kTrailer);
                r.crcs[k] = tr.crc;
                sink.copy(ring.slot(k), k);
                if (k >= lag) {
                    sink.finish(k - lag);
                    ring.release(k - lag);
                }
            }
        } catch (...) {  // a consumer (HIP) failure: stop the readers blocked on a full ring before they are joined
            err.set(std::current_exception());
        }
        if (err.any()) ring.fail();
    }
    err.rethrow();
    hip_ok(hipStreamSynchronize(sink.stream), "sync");
    r.s = s_since(t0);
    r.gib = static_cast<double>(n * kChunk) / 1073741824.0;
    if (validate) r.bad += validate_device(sink, r.crcs);
    return r;
}

// Rank 0, peer mode: read -> transport (one message per chunk, trailer last).
Result send_peer(const Files& files, std::size_t n, dist::Transport& t) {
    Ring ring;
    Errors err;
    Result r;
    const auto t0 = Clock::now();
    {
        Readers readers(files, n, ring, err);
        try {
            for (std::size_t k = 0; k < n; ++k) {
                if (!ring.wait_ready(k)) break;
                t.send(std::span<const std::byte>(reinterpret_cast<const std::byte*>(ring.slot(k)), kChunk + kTrailer));
                ring.release(k);
            }
        } catch (...) {
            err.set(std::current_exception());
            ring.fail();
        }
    }
    err.rethrow();
    r.s = s_since(t0);
    r.gib = static_cast<double>(n * kChunk) / 1073741824.0;
    return r;
}

// Rank 0, both mode: reads 2n chunks once; each pair of chunks (16 MiB) is sliced into 8 KiB pieces, even pieces
// gathered into its own 8 MiB message (H2D), odd pieces into rank 1's (sent). Returns (own, sent).
std::pair<Result, Result> both_rank0(const Files& files, std::size_t n, dist::Transport& t) {
    Ring ring;
    DeviceSink sink;
    Errors err;
    Result own, sent;
    own.crcs.resize(n);
    struct Gather {  // [own|peer][double buffer], pinned; freed on every path
        std::uint8_t* p[2][2] = {};
        Gather() {
            try {
                for (auto& g : p)
                    for (auto*& b : g) hip_ok(hipHostMalloc(reinterpret_cast<void**>(&b), kChunk + 4096, hipHostMallocDefault), "gather");
            } catch (...) {  // roll back the buffers already allocated: the destructor does not run for a failed constructor
                release();
                throw;
            }
        }
        ~Gather() { release(); }
        void release() noexcept {
            for (auto& g : p)
                for (auto*& b : g)
                    if (b != nullptr) {
                        (void)hipHostFree(b);
                        b = nullptr;
                    }
        }
        Gather(const Gather&) = delete;
        Gather& operator=(const Gather&) = delete;
    } gb;
    auto& gather = gb.p;
    const auto t0 = Clock::now();
    {
        Readers readers(files, 2 * n, ring, err);
        try {
            for (std::size_t m = 0; m < n; ++m) {
                if (!ring.wait_ready(2 * m) || !ring.wait_ready(2 * m + 1)) break;
                std::uint8_t* own_buf = gather[0][m % 2];
                std::uint8_t* peer_buf = gather[1][m % 2];
                if (m >= 2) sink.finish(m - 2);  // own_buf of m - 2 copied to the device
                for (std::size_t half = 0; half < 2; ++half) {
                    const std::uint8_t* src = ring.slot(2 * m + half);
                    for (std::size_t p = 0; p < kChunk / kPiece; p += 2) {
                        std::memcpy(own_buf + (half * kChunk / 2) + (p / 2) * kPiece, src + p * kPiece, kPiece);
                        std::memcpy(peer_buf + (half * kChunk / 2) + (p / 2) * kPiece, src + (p + 1) * kPiece, kPiece);
                    }
                }
                ring.release(2 * m);
                ring.release(2 * m + 1);
                own.crcs[m] = crc(own_buf, kChunk);
                sink.copy(own_buf, m);
                Trailer tr;
                tr.seq = m;
                tr.crc = crc(peer_buf, kChunk);
                std::memcpy(peer_buf + kChunk, &tr, kTrailer);
                t.send(std::span<const std::byte>(reinterpret_cast<const std::byte*>(peer_buf), kChunk + kTrailer));
                sent.crcs.push_back(tr.crc);
            }
        } catch (...) {
            err.set(std::current_exception());
            ring.fail();
        }
    }
    err.rethrow();
    hip_ok(hipStreamSynchronize(sink.stream), "sync");
    own.s = sent.s = s_since(t0);
    own.gib = sent.gib = static_cast<double>(n * kChunk) / 1073741824.0;
    own.bad = validate_device(sink, own.crcs);
    return {own, sent};
}

// Rank 1 (peer and both): transport -> pinned -> trailer check -> device.
Result recv_peer(std::size_t n, dist::Transport& t) {
    Ring ring;
    DeviceSink sink;
    Errors err;
    Result r;
    r.crcs.resize(n);
    constexpr std::size_t lag = kSlots / 2;
    std::size_t bad = 0;
    const auto t0 = Clock::now();
    {
        std::thread rx([&] {
            try {
                for (std::size_t k = 0; k < n; ++k) {
                    if (!ring.wait_free(k)) return;
                    t.receive(std::span<std::byte>(reinterpret_cast<std::byte*>(ring.slot(k)), kChunk + kTrailer));
                    ring.filled(k);
                }
            } catch (...) {
                err.set(std::current_exception());
                ring.fail();
            }
        });
        try {
            for (std::size_t k = 0; k < n; ++k) {
                if (!ring.wait_ready(k)) break;
                Trailer tr;
                std::memcpy(&tr, ring.slot(k) + kChunk, kTrailer);
                const std::uint32_t c = crc(ring.slot(k), kChunk);
                if (tr.magic != kMagic || tr.seq != k || tr.crc != c) ++bad;
                r.crcs[k] = c;
                sink.copy(ring.slot(k), k);
                if (k >= lag) {
                    sink.finish(k - lag);
                    ring.release(k - lag);
                }
            }
        } catch (...) {  // consumer failure: unblock the receiver (ring + transport) before joining it
            err.set(std::current_exception());
            ring.fail();
            t.abort("glm-peerload: receiver consumer failed");
        }
        rx.join();
    }
    err.rethrow();
    hip_ok(hipStreamSynchronize(sink.stream), "sync");
    r.s = s_since(t0);
    r.gib = static_cast<double>(n * kChunk) / 1073741824.0;
    r.bad = bad + validate_device(sink, r.crcs);
    return r;
}

// Strided O_DIRECT pieces over the files (no GPU).
Result load_stride(const std::vector<std::string>& paths, double gib, unsigned threads, std::size_t piece, std::size_t gap) {
    std::vector<int> fds;
    std::vector<std::uint64_t> sizes;
    for (const auto& p : paths) {
        const int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("open " + p);
        struct stat st{};
        ::fstat(fd, &st);
        fds.push_back(fd);
        sizes.push_back(static_cast<std::uint64_t>(st.st_size));
    }
    const std::size_t stride = piece + gap;
    std::vector<std::pair<unsigned, std::uint64_t>> at;
    const auto want = static_cast<std::size_t>(gib * 1073741824.0 / static_cast<double>(piece));
    for (unsigned f = 0; f < fds.size() && at.size() < want; ++f)
        for (std::uint64_t o = 0; o + piece <= sizes[f] && at.size() < want; o += stride) at.emplace_back(f, o);
    if (at.size() < want) throw std::runtime_error("files too small for the stride test");
    std::atomic<std::size_t> next{0};
    std::atomic<bool> bad{false};
    const auto t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (unsigned i = 0; i < threads; ++i)
            pool.emplace_back([&] {
                void* buf = nullptr;
                if (::posix_memalign(&buf, 4096, piece) != 0) {
                    bad = true;
                    return;
                }
                for (;;) {
                    const std::size_t k = next.fetch_add(1);
                    if (k >= at.size()) break;
                    if (::pread(fds[at[k].first], buf, piece, static_cast<off_t>(at[k].second)) != static_cast<ssize_t>(piece)) bad = true;
                }
                std::free(buf);
            });
        for (auto& t : pool) t.join();
    }
    Result r;
    r.s = s_since(t0);
    r.gib = static_cast<double>(at.size() * piece) / 1073741824.0;
    r.bad = bad ? 1 : 0;
    for (int fd : fds) ::close(fd);
    return r;
}

void print(const char* what, unsigned rank, const Result& r) {
    std::printf("RESULT %-22s rank %u: %.2f GiB in %.2f s = %.2f GB/s (timed), trailer/validation failures %zu\n", what, rank, r.gib, r.s,
                r.gib * 1.073741824 / r.s, r.bad);
}

int selftest() {
    int fails = 0;
    std::mt19937_64 rng(42);
    std::vector<std::uint8_t> buf(kChunk + 4096);
    for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{4096}, std::size_t{12288}, std::size_t{12289}, std::size_t{1} << 20U, kChunk}) {
        for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<std::uint8_t>(rng());
        const auto a = crc(buf.data(), n);
        const auto b = dist::crc32c(std::span<const std::byte>(reinterpret_cast<const std::byte*>(buf.data()), n));
        if (a != b) {
            std::printf("crc parallel != serial at %zu bytes  FAIL\n", n);
            ++fails;
        }
    }
    // review finding: zero 8 MiB chunk, bit 0 flipped at byte offsets 2796204 and 5592400 must be detected.
    std::fill(buf.begin(), buf.end(), 0);
    const auto zero = crc(buf.data(), kChunk);
    buf[2796204] ^= 1U;
    buf[5592400] ^= 1U;
    if (crc(buf.data(), kChunk) == zero) {
        std::printf("two-bit corruption (regression case) NOT detected  FAIL\n");
        ++fails;
    }
    // random single- and two-bit flips on random data
    for (std::size_t i = 0; i < kChunk; ++i) buf[i] = static_cast<std::uint8_t>(rng());
    const auto good = crc(buf.data(), kChunk);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t a = rng() % kChunk, b = rng() % kChunk;
        const unsigned ba = static_cast<unsigned>(rng() % 8), bb = static_cast<unsigned>(rng() % 8);
        buf[a] ^= static_cast<std::uint8_t>(1U << ba);
        if (trial % 2 == 1 && !(a == b && ba == bb)) buf[b] ^= static_cast<std::uint8_t>(1U << bb);
        if (crc(buf.data(), kChunk) == good) {
            std::printf("flip trial %d NOT detected  FAIL\n", trial);
            ++fails;
        }
        buf[a] ^= static_cast<std::uint8_t>(1U << ba);
        if (trial % 2 == 1 && !(a == b && ba == bb)) buf[b] ^= static_cast<std::uint8_t>(1U << bb);
    }
    if (crc(buf.data(), kChunk) != good) ++fails;
    std::printf("peerload selftest (crc32c_parallel == dist::crc32c at 7 sizes, two-bit regression case, 200 random 1/2-bit flips): %s\n",
                fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

std::unique_ptr<dist::Transport> open_pair(unsigned rank) {
    setenv("STRIX_DIST_USB4_SPIN", "0", 1);
    dist::Usb4AutoOptions o;
    o.rank = rank;
    o.timeout = std::chrono::milliseconds(30000);
    dist::Usb4Topology topo;
    auto t = dist::usb4_host_auto(o, &topo);
    std::printf("PAIRING: %s\n", topo.receipt().c_str());
    return t;
}

void barrier(dist::Transport& t) {
    std::uint64_t a = 1, b = 0;
    t.exchange(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&a), 8), std::span<std::byte>(reinterpret_cast<std::byte*>(&b), 8));
}

int pair(unsigned rank, const std::string& m, double gib, const std::vector<std::string>& paths) {
    auto t = open_pair(rank);
    barrier(*t);
    int rc = 0;
    const std::size_t n = static_cast<std::size_t>(gib * 1073741824.0) / kChunk;
    if (m == "local2") {
        Files files(paths, gib);
        const Result r = load_local(files, n, true);
        print("local2", rank, r);
        rc = r.bad == 0 ? 0 : 1;
    } else if (m == "peer" && rank == 0) {
        Files files(paths, gib);
        print("peer-send", 0, send_peer(files, n, *t));
        const Result again = load_local(files, n, false);  // VALIDATION: the source bytes' CRCs
        t->send(std::span<const std::byte>(reinterpret_cast<const std::byte*>(again.crcs.data()), n * 4));
    } else if (m == "both" && rank == 0) {
        Files files(paths, 2 * gib);
        const auto [own, sent] = both_rank0(files, n, *t);
        print("both-own", 0, own);
        print("both-send", 0, sent);
        t->send(std::span<const std::byte>(reinterpret_cast<const std::byte*>(sent.crcs.data()), n * 4));
        rc = own.bad == 0 ? 0 : 1;
    } else if ((m == "peer" || m == "both") && rank == 1) {
        const Result got = recv_peer(n, *t);
        print(m == "both" ? "both-recv" : "peer-recv", 1, got);
        std::vector<std::uint32_t> src(n);
        t->receive(std::span<std::byte>(reinterpret_cast<std::byte*>(src.data()), n * 4));
        std::size_t diff_src = 0, diff_local = 0;
        for (std::size_t k = 0; k < n; ++k) diff_src += src[k] != got.crcs[k] ? 1 : 0;
        std::string local_note = "not applicable (sliced stream)";
        if (m == "peer") {  // VALIDATION: the same bytes from rank 1's own copy of the files
            Files mine(paths, gib);
            const Result lr = load_local(mine, n, false);
            for (std::size_t k = 0; k < n; ++k) diff_local += lr.crcs[k] != got.crcs[k] ? 1 : 0;
            local_note = std::to_string(diff_local) + " chunks differ";
        }
        std::printf("EXACT %s rank 1: %zu chunks; trailer/device failures %zu; received vs sender's source CRCs %zu differ; received vs own file copy: %s\n",
                    m.c_str(), n, got.bad, diff_src, local_note.c_str());
        rc = got.bad == 0 && diff_src == 0 && diff_local == 0 ? 0 : 1;
    } else {
        throw std::invalid_argument("mode must be peer, both or local2");
    }
    barrier(*t);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "selftest") return selftest();
        if (mode == "local" && argc >= 5) {
            Files files(std::vector<std::string>(argv + 4, argv + argc), std::stod(argv[3]));
            const Result r = load_local(files, files.chunks.size(), true);
            print((std::string("local-") + argv[2]).c_str(), 0, r);
            return r.bad == 0 ? 0 : 1;
        }
        if (mode == "stride" && argc >= 7) {
            const unsigned threads = static_cast<unsigned>(std::stoul(argv[3]));
            const std::size_t piece = std::stoull(argv[4]), gap = std::stoull(argv[5]);
            const Result r = load_stride(std::vector<std::string>(argv + 6, argv + argc), std::stod(argv[2]), threads, piece, gap);
            const std::string tag = "stride-" + std::to_string(piece) + "+" + std::to_string(gap) + "-t" + std::to_string(threads);
            print(tag.c_str(), 0, r);
            return r.bad == 0 ? 0 : 1;
        }
        if (mode == "pair" && argc >= 6)
            return pair(static_cast<unsigned>(std::stoul(argv[2])), argv[3], std::stod(argv[4]), std::vector<std::string>(argv + 5, argv + argc));
        std::fprintf(stderr, "usage: %s selftest | local TAG GIB FILE... | stride GIB THREADS PIECE GAP FILE... | pair RANK peer|both|local2 GIB FILE...\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm-peerload error: %s\n", e.what());
        return 3;
    }
}
