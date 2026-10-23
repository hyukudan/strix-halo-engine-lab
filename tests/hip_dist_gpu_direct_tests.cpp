// GPU-direct exchange tests on one GPU, against a fake of the driver's GPU-queue ioctls (QUERY, QUEUE, IMPORT, ARM,
// PREPARE_SLOT, SUBMIT, EVENT, RETIRE, STOP, RELEASE) that connects two in-process ranks:
//  - a TX lands in the oldest posted RX of the peer, in physical order (as the ring; with end-to-end flow
//    control), as a CPU copy between the two ranks' host-mapped arenas; EVENT batches carry the exact
//    frame geometry the queue adapter checks (payload frames, then the terminal frame);
//  - a TX whose payload size differs from the peer's RX fails both like the kernel's terminal check.
// Checks: many exchanges of 1 B .. the full capacity (and chunked messages above it), byte-exact both ways, with work
// enqueued between start and finish, slots reused many times, through the Communicator. Failures end with the
// receiving stream trapping before anything consumes the data, so each failure case runs in a child process:
// ranks that disagree on a size (terminal check), on a tag (in-band identity), a silent peer (deadline).

#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

constexpr std::uint32_t ring_frames = 256, depth = 4;

// ---- fake driver ---------------------------------------------------------------------------------------------
struct FakeOp {
    sgu_prepare_slot prep{};
    bool submitted = false, complete = false, failed = false;
};
struct FakeSide;
struct FakeWire {
    std::mutex mutex;
    FakeSide* side[2] = {nullptr, nullptr};
};
struct FakeSide {
    FakeWire* wire = nullptr;
    int rank = 0;
    std::array<std::uint8_t, 16> local{}, peer{};
    sgu_fence fence{};
    // Arenas in host-mapped coherent memory: the fake moves bytes with the CPU, like a NIC, never through a GPU
    // queue (a GPU copy could sit behind a stream that waits for this very delivery). The ranks' kernels reach
    // them through their device pointers.
    void* tx = nullptr;
    void* rx = nullptr;
    void* tx_device = nullptr;
    void* rx_device = nullptr;
    std::map<std::uint64_t, FakeOp> ops; // by handle
    std::deque<sgu_event> events;
    std::uint64_t generation = 0;
    bool silent = false; // a peer that never delivers (models a stalled or dead rank)

    FakeSide& other() { return *wire->side[1 - rank]; }
    static std::uint32_t frames_of(const sgu_prepare& p) { return static_cast<std::uint32_t>((p.payload_bytes + 4095U) / 4096U) + 1U; }
    void complete(FakeOp& op, bool failed) {
        op.complete = true;
        op.failed = failed;
        const auto& p = op.prep.operation;
        const std::uint32_t total = frames_of(p);
        for (std::uint32_t first = 0; first < total; first += SGU_EVENT_BATCH) {
            sgu_event e{};
            e.fence = fence;
            e.handle = p.handle;
            e.sequence = p.sequence;
            e.payload_bytes = p.payload_bytes;
            e.monotonic_ns = 12345U + p.handle;
            e.direction = p.direction;
            e.status = failed ? SGU_EVENT_FAILED : SGU_EVENT_COMPLETE;
            e.first_frame = first;
            e.count = std::min<std::uint32_t>(SGU_EVENT_BATCH, total - first);
            e.total_frames = total;
            for (std::uint32_t n = 0; n < e.count; ++n) {
                const std::uint32_t logical = first + n;
                const bool terminal = logical + 1U == total;
                const std::uint64_t offset = terminal ? std::uint64_t{ring_frames - depth + op.prep.slot} * 4096U : p.pool_offset + std::uint64_t{logical} * 4096U;
                const std::uint32_t bytes = terminal ? p.terminal_bytes : static_cast<std::uint32_t>(std::min<std::uint64_t>(4096U, p.payload_bytes - std::uint64_t{logical} * 4096U));
                auto& f = e.frames[n];
                f.pool_offset = offset;
                f.capacity = 4096;
                f.expected_bytes = bytes;
                f.actual_bytes = bytes;
                f.physical_slot = static_cast<std::uint32_t>(offset / 4096U);
                f.error_flags = failed && terminal ? SGU_FRAME_ERROR_TERMINAL : 0U;
            }
            events.push_back(e);
        }
    }
    // Physical order, as the ring: the oldest submitted TX of `from` lands in the oldest posted RX of `to` (no lookup
    // by sequence). The receiver's kernel rejects a terminal that does not match that RX (sequence, size, slot).
    static void deliver(FakeSide& from, FakeSide& to) {
        if (from.silent || to.silent) return;
        while (true) {
            FakeOp* tx = nullptr;
            FakeOp* rx = nullptr;
            for (auto& [h, op] : from.ops)
                if (op.prep.operation.direction == SGU_DIRECTION_TX && op.submitted && !op.complete) { tx = &op; break; }
            for (auto& [h, op] : to.ops)
                if (op.prep.operation.direction == SGU_DIRECTION_RX && op.submitted && !op.complete) { rx = &op; break; }
            if (!tx || !rx) return;
            const bool same = rx->prep.operation.sequence == tx->prep.operation.sequence &&
                              rx->prep.operation.payload_bytes == tx->prep.operation.payload_bytes && rx->prep.slot == tx->prep.slot;
            if (same)
                std::memcpy(static_cast<std::byte*>(to.rx) + rx->prep.operation.pool_offset,
                            static_cast<const std::byte*>(from.tx) + tx->prep.operation.pool_offset, tx->prep.operation.payload_bytes);
            from.complete(*tx, false);
            to.complete(*rx, !same);
        }
    }
    strix::transport::GpuQueueSyscall ioctl(unsigned long command, void* argument) {
        std::lock_guard lock(wire->mutex);
        const auto ok = strix::transport::GpuQueueSyscall{0, 0};
        switch (command) {
        case SGU_IOC_QUERY: {
            auto& q = *static_cast<sgu_query*>(argument);
            q.fence.abi_cookie = SGU_ABI_COOKIE;
            q.fence.device_instance[0] = static_cast<std::uint8_t>(1 + rank);
            std::memcpy(q.fence.local_uuid, local.data(), 16);
            std::memcpy(q.fence.peer_uuid, peer.data(), 16);
            q.state = SGU_STATE_OPEN;
            q.frame_bytes = SGU_FRAME_BYTES;
            q.ring_frames = ring_frames;
            q.maximum_sg_entries = SGU_MAX_SG_ENTRIES;
            q.maximum_window_bytes = std::uint64_t{ring_frames} * SGU_FRAME_BYTES;
            q.maximum_imported_bytes = SGU_MAX_IMPORTED_BYTES;
            q.maximum_event_batch = SGU_EVENT_BATCH;
            fence = q.fence;
            return ok;
        }
        case SGU_IOC_QUEUE: {
            auto& c = *static_cast<sgu_queue_config*>(argument);
            c.maximum_depth = SGU_MAX_QUEUE_DEPTH;
            c.features = SGU_CAP_MULTI_OP | SGU_CAP_NONBLOCK_SUBMIT | SGU_CAP_SESSION_RECOVERY | SGU_CAP_TERMINAL_HEADER;
            return ok;
        }
        case SGU_IOC_IMPORT: {
            auto& v = *static_cast<sgu_import*>(argument);
            v.fence.import_generation = ++generation;
            fence = v.fence;
            return ok;
        }
        case SGU_IOC_ARM: case SGU_IOC_STOP: case SGU_IOC_RELEASE: {
            auto& a = *static_cast<sgu_action*>(argument);
            if (command == SGU_IOC_STOP) a.result = SGU_RESULT_CONTROLLER_STOPPED;
            return ok;
        }
        case SGU_IOC_PREPARE_SLOT: {
            const auto& p = *static_cast<sgu_prepare_slot*>(argument);
            if (p.operation.flags != SGU_PREPARE_TERMINAL_HEADER || p.operation.terminal_bytes != SGU_TERMINAL_HEADER_BYTES)
                return {-1, EINVAL};
            ops[p.operation.handle].prep = p;
            return ok;
        }
        case SGU_IOC_SUBMIT: {
            auto& a = *static_cast<sgu_action*>(argument);
            auto it = ops.find(a.handle);
            if (it == ops.end()) return {-1, EINVAL};
            it->second.submitted = true;
            a.queued_frames = frames_of(it->second.prep.operation);
            deliver(*this, other());
            deliver(other(), *this);
            return ok;
        }
        case SGU_IOC_EVENT: {
            if (events.empty()) return {-1, EAGAIN};
            auto& e = *static_cast<sgu_event*>(argument);
            e = events.front();
            events.pop_front();
            return ok;
        }
        case SGU_IOC_RETIRE: {
            ops.erase(static_cast<sgu_action*>(argument)->handle);
            return ok;
        }
        default:
            return {-1, ENOTTY};
        }
    }
};

struct Rank {
    std::vector<std::unique_ptr<FakeSide>> sides;
    std::vector<void*> host_memory;
    ~Rank() { for (auto* m : host_memory) (void)hipHostFree(m); }
};
std::array<std::uint8_t, 16> uuid(std::uint8_t v) { std::array<std::uint8_t, 16> u{}; u.fill(v); return u; }

// Two ranks over `streams` fake streams; returns each rank's GPU-direct exchange.
// `rank1_order`: rank 1's links in this order of the physical wires (a crossed cabling), as a real rank would
// list them before its bootstrap.
std::pair<std::unique_ptr<strix::dist::GpuDirectLinks>, std::unique_ptr<strix::dist::GpuDirectLinks>>
make_pair(std::vector<std::unique_ptr<FakeWire>>& wires, Rank& r0, Rank& r1, std::size_t streams, std::chrono::milliseconds timeout,
          std::vector<FakeSide*>* rank1_sides = nullptr, const std::vector<std::size_t>* rank1_order = nullptr) {
    std::vector<strix::dist::GpuDirectLink> links[2];
    Rank* ranks[2] = {&r0, &r1};
    for (std::size_t k = 0; k < streams; ++k) {
        wires.push_back(std::make_unique<FakeWire>());
        for (int r = 0; r < 2; ++r) {
            auto side = std::make_unique<FakeSide>();
            side->wire = wires.back().get();
            side->rank = r;
            side->local = uuid(static_cast<std::uint8_t>(0x10 + r));
            side->peer = uuid(static_cast<std::uint8_t>(0x11 - r));
            const std::size_t window = std::size_t{ring_frames} * 4096U;
            check(hipHostMalloc(&side->tx, window, hipHostMallocMapped | hipHostMallocCoherent), "tx arena");
            check(hipHostMalloc(&side->rx, window, hipHostMallocMapped | hipHostMallocCoherent), "rx arena");
            check(hipHostGetDevicePointer(&side->tx_device, side->tx, 0), "tx arena device pointer");
            check(hipHostGetDevicePointer(&side->rx_device, side->rx, 0), "rx arena device pointer");
            ranks[r]->host_memory.push_back(side->tx);
            ranks[r]->host_memory.push_back(side->rx);
            wires.back()->side[r] = side.get();
            if (r == 1 && rank1_sides) rank1_sides->push_back(side.get());
            strix::dist::GpuDirectLink link;
            link.name = "fake" + std::to_string(k);
            FakeSide* raw = side.get();
            link.io.lifetime = std::make_shared<int>(0);
            link.io.ioctl_call = [raw](unsigned long c, void* a) { return raw->ioctl(c, a); };
            link.io.require_current = [] {};
            link.expected.local = raw->local;
            link.expected.peer = raw->peer;
            link.expected.ring_frames = ring_frames;
            link.expected.depth = depth;
            link.expected.terminal = strix::transport::GpuQueueTerminal::kernel_header;
            link.arena.tx = raw->tx_device;
            link.arena.rx = raw->rx_device;
            auto& pool = link.arena.pool;
            const auto buffer = [&](int fd, std::uint64_t inode) {
                sgu_buffer b{};
                b.fd = fd; b.full_bytes = window; b.window_offset = 0; b.window_bytes = window; b.allocation_generation = 1;
                b.fd_device = 7; b.fd_inode = inode;
                return b;
            };
            pool.tx = buffer(100 + static_cast<int>(k) * 2, 1000 + k * 2);
            pool.rx = buffer(101 + static_cast<int>(k) * 2, 1001 + k * 2);
            pool.lifetime = std::make_shared<int>(0);
            pool.require_current = [] {};
            pool.gpu_quiescent = [] { return true; };
            links[r].push_back(std::move(link));
            ranks[r]->sides.push_back(std::move(side));
        }
    }
    if (rank1_order) {
        std::vector<strix::dist::GpuDirectLink> crossed;
        for (auto k : *rank1_order) crossed.push_back(std::move(links[1][k]));
        links[1] = std::move(crossed);
    }
    strix::dist::GpuDirectOptions o;
    o.epoch = 7; o.pool_identity = 9; o.timeout = timeout;
    return {strix::dist::gpu_direct_exchange(std::move(links[0]), o), strix::dist::gpu_direct_exchange(std::move(links[1]), o)};
}

std::vector<std::byte> pattern(std::size_t n, unsigned seed) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::byte>((i * 2654435761U + seed * 40503U) >> 11U);
    return v;
}

// Both ranks run the same exchanges (sizes in the same order) through a Communicator each.
void test_exchanges() {
    std::vector<std::unique_ptr<FakeWire>> wires;
    Rank r0, r1;
    auto [d0, d1] = make_pair(wires, r0, r1, 4, std::chrono::milliseconds(20000));
    const std::size_t cap = d0->capacity();
    std::cout << "pair ready: capacity " << cap << " B" << std::endl;
    auto [t0, t1] = strix::dist::memory_pair();
    strix::dist::Communicator c0({0, 2, 1, 1}, std::move(t0), std::move(d0), 3U * cap), c1({1, 2, 1, 1}, std::move(t1), std::move(d1), 3U * cap);
    // Sizes that are not whole 4-byte words included: the last word travels padded and nothing past `n` is written.
    std::vector<std::size_t> sizes{1, 3, 12, 13, 64, 4096, 4097, 20480, 20481, 20544, 122880, 131072 + 12, cap / 2U, cap - 1U, cap,
                                   2U * cap + 4096U, 2U * cap + 4099U};
    // Slot edges: payloads whose equal split puts a part at, just below or just above k slots.
    const std::size_t slot = (cap + 64U) / 4U;
    for (std::size_t k = 1; k <= 4U; ++k)
        for (std::size_t d : {68U, 65U, 64U, 4U, 1U}) sizes.push_back(k * slot - d);
    std::mt19937 rng(5);
    std::vector<std::size_t> order = sizes;  // every size once, then a random sequence
    for (int i = 0; i < 160; ++i) order.push_back(sizes[rng() % sizes.size()]);
    std::atomic<std::size_t> bad{0}, checked{0};
    const auto run = [&](strix::dist::Communicator& c, unsigned rank) {
        hipStream_t s{};
        check(hipStreamCreate(&s), "stream");
        void *local = nullptr, *remote = nullptr, *other = nullptr;
        check(hipMalloc(&local, 3U * cap), "local");
        check(hipMalloc(&remote, 3U * cap), "remote");
        check(hipMalloc(&other, 1U << 20U), "other");
        for (std::size_t i = 0; i < order.size(); ++i) {
            const auto n = order[i];
            const auto mine = pattern(n, static_cast<unsigned>(i * 2U + rank)), theirs = pattern(n, static_cast<unsigned>(i * 2U + 1U - rank));
            check(hipMemcpyAsync(local, mine.data(), n, hipMemcpyHostToDevice, s), "fill");
            check(hipMemsetAsync(remote, 0xA5, n + 8U, s), "guard");
            check(hipStreamSynchronize(s), "fill"); // the pageable source is a temporary
            if (i % 2U == 0U) {
                c.start(s, local, n, strix::dist::Tag{1U, static_cast<std::uint32_t>(i), 0U});
                check(hipMemsetAsync(other, static_cast<int>(i & 0xffU), 1U << 20U, s), "work between start and finish");
                c.finish(s, remote);
            } else {
                c.exchange(s, local, remote, n, strix::dist::Tag{1U, static_cast<std::uint32_t>(i), 0U});
            }
            std::vector<std::byte> got(n + 8U);
            check(hipMemcpyAsync(got.data(), remote, n + 8U, hipMemcpyDeviceToHost, s), "read back");
            check(hipStreamSynchronize(s), "read back");
            bool same = std::equal(theirs.begin(), theirs.end(), got.begin());
            for (std::size_t g = n; g < n + 8U; ++g) same = same && got[g] == std::byte{0xA5};
            if (!same) ++bad;
            ++checked;
            if (rank == 0U && (i < 3U || i % 40U == 0U)) std::cout << "  exchange " << i << " (" << n << " B) done\n" << std::flush;
        }
        // A run of one size (the early receives match), quiesce (consumes the receive posted for a guessed next
        // exchange), then another size: all byte-exact.
        for (std::size_t i = 0; i < 22U; ++i) {
            const std::size_t n = i < 20U ? 20480U : 4096U;
            if (i == 20U) c.quiesce(s);
            const auto mine = pattern(n, static_cast<unsigned>(1000U + i * 2U + rank)), theirs = pattern(n, static_cast<unsigned>(1000U + i * 2U + 1U - rank));
            check(hipMemcpyAsync(local, mine.data(), n, hipMemcpyHostToDevice, s), "fill");
            check(hipStreamSynchronize(s), "fill");
            c.exchange(s, local, remote, n, strix::dist::Tag{1U, static_cast<std::uint32_t>(1000U + i), 0U});
            std::vector<std::byte> got(n);
            check(hipMemcpyAsync(got.data(), remote, n, hipMemcpyDeviceToHost, s), "read back");
            check(hipStreamSynchronize(s), "read back");
            if (!std::equal(theirs.begin(), theirs.end(), got.begin())) ++bad;
            ++checked;
        }
        (void)hipFree(local); (void)hipFree(remote); (void)hipFree(other);
        (void)hipStreamDestroy(s);
    };
    std::exception_ptr e1;
    std::thread peer([&] { try { run(c1, 1); } catch (...) { e1 = std::current_exception(); } });
    run(c0, 0);
    peer.join();
    if (e1) std::rethrow_exception(e1);
    expect(bad == 0U, "GPU-direct exchanges byte-exact (" + std::to_string(bad.load()) + " of " + std::to_string(checked.load()) + " differ)");
    {
        const auto st = c0.device_exchange()->stats();
        const char* pp = std::getenv("STRIX_DIST_GPU_PREPOST");
        if (!(pp && pp[0] == '0'))
            expect(st.preposted > 0U && st.prepost_misses > 0U, "early receives used (" + std::to_string(st.preposted) + ") and wrong guesses recovered (" +
                                                                     std::to_string(st.prepost_misses) + ")");
        std::cout << "early receives: " << st.preposted << " used, " << st.prepost_misses << " wrong guesses (internal exchanges)" << std::endl;
    }
    std::cout << "gpu-direct (fake driver, 4 streams, depth " << depth << "): " << checked.load() << " exchanges of 1 B .. " << 2U * cap + 4099U
              << " B byte-exact both ways, through the communicator (" << c0.transport_name() << ")" << std::endl;
}


// The bootstrap without host-mode streams: rank 1 lists its rings in a crossed order; the in-band HELLO puts them in
// rank 0's order (exchanges are byte-exact only if every part lands on the matching ring), then both apply rank
// 0's latency ranking. A renumbered stream (dataN differs at the two ends) fails with that reason.
// renumber: rank 1 names one ring's stream differently (fine); foreign: rank 1's ring 1 reports another peer.
std::vector<strix::dist::GpuDirectRing> fake_rings(unsigned rank, const std::vector<std::size_t>& physical, bool renumber, bool foreign) {
    std::vector<strix::dist::GpuDirectRing> out;
    const std::string me(32, rank ? '1' : '0'), other(32, rank ? '0' : '1');
    for (auto k : physical) {
        std::string stream = "data" + std::to_string(k % 2U);
        if (renumber && rank == 1U && k == 1U) stream = "data0";
        out.push_back({"r" + std::to_string(rank) + "/" + std::to_string(k) + " fake" + std::to_string(k), stream, me,
                       foreign && rank == 1U && k == 1U ? std::string(32, '7') : other});
    }
    return out;
}
std::string run_bootstrap(bool renumber, bool foreign = false) {
    std::vector<std::unique_ptr<FakeWire>> wires;
    Rank r0, r1;
    const std::vector<std::size_t> crossed{2, 0, 3, 1}, straight{0, 1, 2, 3};
    auto [d0, d1] = make_pair(wires, r0, r1, 4, std::chrono::milliseconds(5000), nullptr, &crossed);
    auto [t0, t1] = strix::dist::memory_pair();
    // The fake's endpoint identities are rank-wide (0x10 / 0x11 for every ring): the stream names and the HELLO
    // indices are what tells the rings apart here.
    std::string receipt0, error1;
    std::thread peer([&] {
        try { (void)strix::dist::gpu_direct_bootstrap(*d1, 1, *t1, fake_rings(1, crossed, renumber, foreign), 42); }
        catch (const std::exception& e) { error1 = e.what(); }
    });
    std::string error0;
    try { receipt0 = strix::dist::gpu_direct_bootstrap(*d0, 0, *t0, fake_rings(0, straight, renumber, foreign), 42); }
    catch (const std::exception& e) { error0 = e.what(); }
    if (!error0.empty() || !error1.empty()) {
        peer.join(); // both ranks refuse at the HELLO check, before the next control message
        return "FAILED: " + error0 + " | " + error1;
    }
    peer.join();
    // Both ranks keep the first three rings of the agreed order (the way max_streams applies after the bootstrap).
    d0->keep(3);
    d1->keep(3);
    strix::dist::Communicator c0({0, 2, 1, 1}, std::move(t0), std::move(d0), 8U << 20U), c1({1, 2, 1, 1}, std::move(t1), std::move(d1), 8U << 20U);
    std::atomic<std::size_t> bad{0};
    const std::vector<std::size_t> sizes{5, 64, 20480, 300000, 3000000};
    const auto run = [&](strix::dist::Communicator& c, unsigned rank) {
        hipStream_t st{};
        check(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        void *local = nullptr, *remote = nullptr;
        check(hipMalloc(&local, 3000000), "local");
        check(hipMalloc(&remote, 3000000), "remote");
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            const auto n = sizes[i];
            const auto mine = pattern(n, static_cast<unsigned>(i * 2U + rank)), theirs = pattern(n, static_cast<unsigned>(i * 2U + 1U - rank));
            // Never the null stream: it would wait for the other rank's stream, which waits for this rank's send.
            check(hipMemcpyAsync(local, mine.data(), n, hipMemcpyHostToDevice, st), "fill");
            check(hipStreamSynchronize(st), "fill");
            c.exchange(st, local, remote, n, strix::dist::Tag{1U, static_cast<std::uint32_t>(i), 0U});
            std::vector<std::byte> got(n);
            check(hipMemcpyAsync(got.data(), remote, n, hipMemcpyDeviceToHost, st), "read back");
            check(hipStreamSynchronize(st), "read back");
            if (got != theirs) ++bad;
        }
        (void)hipFree(local); (void)hipFree(remote); (void)hipStreamDestroy(st);
    };
    std::exception_ptr e1;
    std::thread other([&] { try { run(c1, 1); } catch (...) { e1 = std::current_exception(); } });
    run(c0, 0);
    other.join();
    if (e1) std::rethrow_exception(e1);
    expect(bad == 0U, "exchanges after the bootstrap byte-exact (" + std::to_string(bad.load()) + " differ)");
    return receipt0;
}
void test_bootstrap() {
    const auto ok = run_bootstrap(false);
    expect(ok.rfind("FAILED", 0) != 0, "bootstrap over crossed rings: " + ok);
    std::cout << "bootstrap (fake driver, rank 1's rings crossed): byte-exact exchanges after HELLO + reorder; rank 0 receipt:" << ok << std::endl;
    const auto renumbered = run_bootstrap(true);
    expect(renumbered.rfind("FAILED", 0) != 0, "a stream named differently at the two ends is still the same ring: " + renumbered);
    std::cout << "bootstrap with a stream numbered differently at the two ends: accepted, byte-exact" << std::endl;
    const auto bad = run_bootstrap(false, true);
    expect(bad.find("endpoints do not match") != std::string::npos, "a foreign endpoint must fail the HELLO: " + bad);
    std::cout << "bootstrap with a foreign endpoint refused: " << bad.substr(0, 200) << std::endl;
}

// The size router: both ranks calibrate at construction (host path vs direct on 1..4 rings over a size grid), rank 0's
// table is applied by both; then exchanges of every size are byte-exact whatever path each size took.
void test_router() {
    std::vector<std::unique_ptr<FakeWire>> wires;
    Rank r0, r1;
    auto [d0, d1] = make_pair(wires, r0, r1, 4, std::chrono::milliseconds(20000));
    auto [t0, t1] = strix::dist::memory_pair();
    ::setenv("STRIX_DIST_ROUTE", "auto", 1);
    std::unique_ptr<strix::dist::Communicator> c0, c1;
    std::exception_ptr e1;
    std::thread peer([&] {
        try { c1 = std::make_unique<strix::dist::Communicator>(strix::dist::Group{1, 2, 1, 1}, std::move(t1), std::move(d1), 4U << 20U); }
        catch (...) { e1 = std::current_exception(); }
    });
    std::string error0;
    try { c0 = std::make_unique<strix::dist::Communicator>(strix::dist::Group{0, 2, 1, 1}, std::move(t0), std::move(d0), 4U << 20U); }
    catch (const std::exception& e) { error0 = e.what(); }
    peer.join();
    ::setenv("STRIX_DIST_ROUTE", "gpu", 1);
    std::string error1;
    if (e1) try { std::rethrow_exception(e1); } catch (const std::exception& e) { error1 = e.what(); }
    if (!error0.empty() || !error1.empty()) {
        expect(false, "router calibration: rank 0: " + error0 + " | rank 1: " + error1);
        return;
    }
    expect(c0->route_text() == c1->route_text(), "both ranks apply rank 0's route table (" + c0->route_text() + " / " + c1->route_text() + ")");
    std::atomic<std::size_t> bad{0};
    const std::vector<std::size_t> sizes{3, 64, 1000, 4096, 20480, 65537, 163840, 600000, 3000000};
    const auto run = [&](strix::dist::Communicator& c, unsigned rank) {
        hipStream_t st{};
        check(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
        void *local = nullptr, *remote = nullptr;
        check(hipMalloc(&local, 3000000), "local");
        check(hipMalloc(&remote, 3000000), "remote");
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            const auto n = sizes[i];
            const auto mine = pattern(n, static_cast<unsigned>(i * 2U + rank)), theirs = pattern(n, static_cast<unsigned>(i * 2U + 1U - rank));
            check(hipMemcpyAsync(local, mine.data(), n, hipMemcpyHostToDevice, st), "fill");
            check(hipStreamSynchronize(st), "fill");
            c.exchange(st, local, remote, n, strix::dist::Tag{1U, static_cast<std::uint32_t>(i), 0U});
            std::vector<std::byte> got(n);
            check(hipMemcpyAsync(got.data(), remote, n, hipMemcpyDeviceToHost, st), "read back");
            check(hipStreamSynchronize(st), "read back");
            if (got != theirs) ++bad;
        }
        (void)hipFree(local); (void)hipFree(remote); (void)hipStreamDestroy(st);
    };
    std::exception_ptr e2;
    std::thread other([&] { try { run(*c1, 1); } catch (...) { e2 = std::current_exception(); } });
    run(*c0, 0);
    other.join();
    if (e2) std::rethrow_exception(e2);
    expect(bad == 0U, "routed exchanges byte-exact (" + std::to_string(bad.load()) + " differ)");
    std::cout << "router (fake driver + memory transport): calibrated table " << c0->route_text() << "; exchanges of 3 B .. 3 MB byte-exact" << std::endl;
    // The default (no STRIX_DIST_ROUTE): the measured table, decided locally (no calibration traffic).
    {
        std::vector<std::unique_ptr<FakeWire>> w2;
        Rank s0, s1;
        auto [x0, x1] = make_pair(w2, s0, s1, 4, std::chrono::milliseconds(20000));
        auto [m0, m1] = strix::dist::memory_pair();
        ::unsetenv("STRIX_DIST_ROUTE");
        strix::dist::Communicator a({0, 2, 1, 1}, std::move(m0), std::move(x0), 4U << 20U), b({1, 2, 1, 1}, std::move(m1), std::move(x1), 4U << 20U);
        ::setenv("STRIX_DIST_ROUTE", "gpu", 1);
        expect(a.route_text() == "65536:host,1048576:gpu:2,inf:mix:4:35" && a.route_text() == b.route_text(),
               "default route table with host streams (" + a.route_text() + ")");
        std::cout << "router default (host streams available): " << a.route_text() << std::endl;
    }
    // Mixed route: large messages split between the host path (head) and two rings (tail), at the same time.
    {
        std::vector<std::unique_ptr<FakeWire>> w3;
        Rank s0, s1;
        auto [x0, x1] = make_pair(w3, s0, s1, 4, std::chrono::milliseconds(20000));
        auto [m0, m1] = strix::dist::memory_pair();
        strix::dist::Communicator a({0, 2, 1, 1}, std::move(m0), std::move(x0), 4U << 20U), b({1, 2, 1, 1}, std::move(m1), std::move(x1), 4U << 20U);
        const std::string table = "65536:host,262144:gpu:2,inf:mix:2:30";
        a.set_routes(table);
        b.set_routes(table);
        expect(a.route_text() == table, "mixed route table round trip (" + a.route_text() + ")");
        bad = 0;
        std::exception_ptr e3;
        std::thread t([&] { try { run(b, 1); } catch (...) { e3 = std::current_exception(); } });
        run(a, 0);
        t.join();
        if (e3) std::rethrow_exception(e3);
        expect(bad == 0U, "mixed-route exchanges byte-exact (" + std::to_string(bad.load()) + " differ)");
        expect(a.stats().exchanges == sizes.size(), "a mixed exchange counts once (" + std::to_string(a.stats().exchanges) + ")");
        bool refused = false;
        try { a.set_routes("inf:mix:2:100"); } catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "mix host share must be 1..99");
        std::cout << "router mixed (" << table << "): exchanges of 3 B .. 3 MB byte-exact" << std::endl;
    }
}

// A producer that never completes (rank 0's stream waits on a flag nobody sets, so its device-to-host copy and event
// never complete): rank 0's bounded wait stops its communicator and closes the channel, so rank 1, already reading,
// ends at once instead of at its own (longer) transport deadline; rank 0 then refuses further exchanges. Host path over
// socket pairs (the USB4 host transport's code).
void test_stall_bound() {
    int a[2] = {-1, -1}, b[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, a) != 0 || ::socketpair(AF_UNIX, SOCK_STREAM, 0, b) != 0) {
        expect(false, "socketpair");
        return;
    }
    for (const int fd : {a[0], a[1], b[0], b[1]}) ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
    ::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "800", 1);
    auto t0 = strix::dist::usb4_host_descriptors({a[0], b[0]}, {"s0", "s1"}, {1, 1}, 0, std::chrono::milliseconds(4000));
    auto t1 = strix::dist::usb4_host_descriptors({a[1], b[1]}, {"s0", "s1"}, {1, 1}, 0, std::chrono::milliseconds(4000));
    std::string m0, m1, again;
    double dt0 = 0, dt1 = 0;
    std::uint64_t* flag = nullptr;
    check(hipExtMallocWithFlags(reinterpret_cast<void**>(&flag), 8, hipMallocSignalMemory), "flag");
    __atomic_store_n(flag, std::uint64_t{0}, __ATOMIC_RELEASE);
    hipStream_t s0{}, s1{};
    check(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking), "stream");
    check(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking), "stream");
    void *l0 = nullptr, *r0 = nullptr, *l1 = nullptr, *r1 = nullptr;
    for (void** q : {&l0, &r0, &l1, &r1}) check(hipMalloc(q, 4096), "buffer");
    {
        strix::dist::Communicator c0({0, 2, 1, 1}, std::move(t0), 1U << 16U), c1({1, 2, 1, 1}, std::move(t1), 1U << 16U);
        ::unsetenv("STRIX_DIST_WAIT_TIMEOUT_MS");
        check(hipStreamWaitValue64(s0, flag, 1, hipStreamWaitValueGte, ~std::uint64_t{0}), "block rank 0's stream");
        const auto run = [&](strix::dist::Communicator& c, hipStream_t st, void* l, void* r, std::string& msg, double& dt) {
            const auto t = std::chrono::steady_clock::now();
            try { c.exchange(st, l, r, 4096, strix::dist::Tag{1U, 7U, 3U}); msg = "completed"; } catch (const std::exception& e) { msg = e.what(); }
            dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        };
        std::thread peer([&] { run(c1, s1, l1, r1, m1, dt1); });
        run(c0, s0, l0, r0, m0, dt0);
        peer.join();
        try { c0.exchange(s1, l0, r0, 64, strix::dist::Tag{1U, 8U, 0U}); again = "completed"; } catch (const std::exception& e) { again = e.what(); }
        __atomic_store_n(flag, std::uint64_t{1}, __ATOMIC_RELEASE); // release rank 0's stream before teardown
        check(hipStreamSynchronize(s0), "release");
    }
    expect(dt0 < 2.5 && m0.find("producer") != std::string::npos && m0.find("exchange 0: 4096 B, tag 1, layer 7, request 3") != std::string::npos,
           "rank 0 stops on its producer within the bound (" + std::to_string(dt0) + " s: " + m0 + ")");
    expect(dt1 < 2.5 && m1 != "completed", "rank 1 ends when rank 0 closes, before its own 4 s deadline (" + std::to_string(dt1) + " s: " + m1 + ")");
    expect(again.find("stopped earlier") != std::string::npos, "a stopped communicator refuses further exchanges (" + again + ")");
    std::cout << "stall bound: rank 0 stopped after " << dt0 << " s (" << m0.substr(0, 90) << "...), rank 1 after " << dt1 << " s ("
              << m1.substr(0, 90) << "...)" << std::endl;
    for (void* q : {l0, r0, l1, r1}) (void)hipFree(q);
    (void)hipStreamDestroy(s0);
    (void)hipStreamDestroy(s1);
    (void)hipFree(flag);
}

// Child scenarios (run in their own process: a failure traps the receiving stream).
int child_mismatch(bool size) {
    std::vector<std::unique_ptr<FakeWire>> wires;
    Rank r0, r1;
    auto [d0, d1] = make_pair(wires, r0, r1, 2, std::chrono::milliseconds(3000));
    const auto run = [&](strix::dist::DeviceExchange& d, unsigned rank) {
        hipStream_t s{};
        check(hipStreamCreate(&s), "stream");
        void* b = nullptr;
        check(hipMalloc(&b, 1U << 20U), "buffer");
        const std::size_t bytes = size && rank == 1U ? 8192U : 12288U;
        const strix::dist::ExchangeIdentity id{1U, 0U, size ? 1U : 1U + rank, 5U};
        d.start(s, b, bytes, id);
        d.finish(s, b);
        if (hipStreamSynchronize(s) != hipSuccess) std::fprintf(stderr, "SYNC-FAILED rank %u\n", rank);
        const auto f = d.failure();
        if (!f.empty()) std::fprintf(stderr, "rank %u reason: %s\n", rank, f.c_str());
        d.start(s, b, bytes, id); // must refuse
        std::fprintf(stderr, "CHILD-SURVIVED\n");
    };
    std::thread peer([&] { try { run(*d1, 1); } catch (const std::exception& e) { std::fprintf(stderr, "rank 1: %s\n", e.what()); } });
    try { run(*d0, 0); } catch (const std::exception& e) { std::fprintf(stderr, "rank 0: %s\n", e.what()); }
    peer.join();
    std::_Exit(0); // the faulted HIP context is not torn down
}
int child_silent() {
    std::vector<std::unique_ptr<FakeWire>> wires;
    Rank r0, r1;
    std::vector<FakeSide*> rank1;
    auto [d0, d1] = make_pair(wires, r0, r1, 2, std::chrono::milliseconds(400), &rank1);
    for (auto* f : rank1) f->silent = true;
    hipStream_t s{};
    check(hipStreamCreate(&s), "stream");
    void* b = nullptr;
    check(hipMalloc(&b, 1U << 16U), "buffer");
    d0->start(s, b, 20480, {1U, 0U, 1U, 0U});
    d0->finish(s, b);
    // The gather traps: the stream never hands invalid data to later work, and its synchronisation fails.
    if (hipStreamSynchronize(s) != hipSuccess) std::fprintf(stderr, "SYNC-FAILED rank 0\n");
    std::fprintf(stderr, "rank 0 reason: %s\n", d0->failure().c_str());
    try { d0->start(s, b, 20480, {1U, 0U, 1U, 0U}); std::fprintf(stderr, "CHILD-SURVIVED\n"); }
    catch (const std::exception& e) { std::fprintf(stderr, "refused: %s\n", e.what()); }
    std::_Exit(0);
}

// Runs this binary with `scenario`; returns (seconds, exit status, captured stderr).
std::tuple<double, int, std::string> run_child(const char* self, const char* scenario) {
    char path[] = "/tmp/gpu-direct-child-XXXXXX";
    const int fd = ::mkstemp(path);
    const auto t0 = std::chrono::steady_clock::now();
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(fd, 2);
        ::dup2(fd, 1);
        ::execl(self, self, scenario, static_cast<char*>(nullptr));
        std::_Exit(127);
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string text;
    if (FILE* f = std::fopen(path, "r")) {
        char buffer[4096];
        std::size_t n;
        while ((n = std::fread(buffer, 1, sizeof(buffer), f)) > 0) text.append(buffer, n);
        std::fclose(f);
    }
    ::close(fd);
    ::unlink(path);
    return {seconds, status, text};
}
void test_failures(const char* self) {
    struct Case { const char* scenario; const char* expect; const char* label; };
    for (const Case c : {Case{"--child-size", "terminal", "ranks disagree on a size (terminal check)"},
                         Case{"--child-tag", "different collective (tag)", "ranks disagree on a tag (in-band identity)"},
                         Case{"--child-silent", "not complete within", "a silent peer (deadline)"}}) {
        const auto [seconds, status, text] = run_child(self, c.scenario);
        const bool stopped = text.find("SYNC-FAILED") != std::string::npos; // the receiving stream faulted
        const bool reason = text.find(c.expect) != std::string::npos;
        expect(stopped && reason && text.find("CHILD-SURVIVED") == std::string::npos && seconds < 20.0,
               std::string(c.label) + ": the stream stops and the next call names the reason (status " + std::to_string(status) + ", " +
                   std::to_string(seconds) + " s): " + text.substr(0, 900));
        std::string first;
        for (std::size_t at = 0; at < text.size();) {
            const auto end = text.find('\n', at);
            const auto line = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
            if (line.find("reason:") != std::string::npos || line.find("refused:") != std::string::npos) { first = line; break; }
            if (end == std::string::npos) break;
            at = end + 1U;
        }
        std::cout << c.label << ": stopped after " << seconds << " s: " << first.substr(0, 200) << '\n';
    }
}
} // namespace

// The mechanism the exchange relies on: a stream waits on signal memory that a CPU thread writes later.
void test_flag_wait() {
    std::uint64_t* flag = nullptr;
    check(hipExtMallocWithFlags(reinterpret_cast<void**>(&flag), 8, hipMallocSignalMemory), "signal memory");
    __atomic_store_n(flag, std::uint64_t{0}, __ATOMIC_RELEASE);
    hipStream_t s{};
    check(hipStreamCreate(&s), "stream");
    void* b = nullptr;
    check(hipMalloc(&b, 4096), "buffer");
    check(hipStreamWaitValue64(s, flag, 5U, hipStreamWaitValueGte, ~std::uint64_t{0}), "wait value");
    check(hipMemsetAsync(b, 1, 4096, s), "work after the wait");
    std::atomic<bool> done{false};
    std::thread writer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        __atomic_store_n(flag, std::uint64_t{5}, __ATOMIC_RELEASE);
    });
    std::thread watchdog([&] {
        for (int i = 0; i < 300 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!done.load()) { std::cerr << "FAIL: the stream never saw the flag (hipStreamWaitValue64 on signal memory)\n"; std::_Exit(3); }
    });
    const auto t0 = std::chrono::steady_clock::now();
    check(hipStreamSynchronize(s), "stream");
    done = true;
    writer.join();
    watchdog.join();
    std::cout << "flag wait: the stream continued " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()
              << " ms after enqueue (CPU write at 50 ms)\n" << std::flush;
    (void)hipFree(b);
    (void)hipFree(flag);
    (void)hipStreamDestroy(s);
}

int main(int argc, char** argv) try {
    // Both ranks live in this process: a stream waiting on its flag blocks the hardware queue it is on, so the two
    // ranks' streams must not share one (one rank per process in real use). More queues than streams here.
    ::setenv("GPU_MAX_HW_QUEUES", "16", 0);
    // The size router calibrates at construction, both ranks at once: the tests that build both communicators on
    // one thread use the fixed all-direct route; test_router builds them on two threads with the calibration.
    ::setenv("STRIX_DIST_ROUTE", "gpu", 1);
    if (argc > 1 && std::string(argv[1]) == "--child-size") return child_mismatch(true);
    if (argc > 1 && std::string(argv[1]) == "--child-tag") return child_mismatch(false);
    if (argc > 1 && std::string(argv[1]) == "--child-silent") return child_silent();
    if (argc > 1 && std::string(argv[1]) == "--bootstrap") { test_bootstrap(); return failures == 0 ? 0 : 1; }
    if (argc > 1 && std::string(argv[1]) == "--router") { test_router(); return failures == 0 ? 0 : 1; }
    test_failures(argv[0]); // before this process touches the GPU (the children fork from here)
    test_flag_wait();
    test_exchanges();
    test_bootstrap();
    test_router();
    test_stall_bound();
    if (failures == 0) std::cout << "gpu-direct tests passed" << std::endl;
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "gpu-direct tests: " << e.what() << '\n';
    return 2;
}
