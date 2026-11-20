// Host contract of dist::Communicator (CPU HIP stub: device memory is host memory, copies complete at once; an
// in-process transport pair and an in-process device exchange). Checks:
//  - exchange_host() and exchange() are interchangeable on the wire, also after ready-view exchanges whose CRC is
//    deferred: the deferred CRC is carried and checked by both calls, in both mixed orders (host/device, device/host),
//    host/host, host/host/device/device and staged/host/staged/device; a corrupted deferred payload fails the next
//    exchange_host() on the receiving rank (before: it was never checked by exchange_host());
//  - transfer() is one way on every route of the default table (host <= 64 KiB, device <= 1 MiB, mixed above):
//    the bytes arrive, nothing travels in the unused direction (host or device), no early receive is posted; a
//    deferred CRC rides one-way messages per direction (a corrupted one fails at the receiving rank);
//  - a device engine without a one-way form: transfer() refuses device and mixed routes on both ranks, keeps the host one.
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/dist/host_copy.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::dist {
// The stub's device memory is host memory: the mapped-host copy kernel is a plain copy.
void enqueue_host_d2h_copy(void* mapped_host, const void* device, std::size_t bytes, hipStream_t) {
    std::memcpy(mapped_host, device, bytes);
}
} // namespace strix::dist

namespace {
using namespace strix::dist;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}
std::byte pattern(unsigned rank, std::size_t i, unsigned salt) {
    return static_cast<std::byte>((i * 131U + rank * 71U + salt * 29U + (i >> 9U)) & 0xffU);
}

// Counts the bytes this rank puts on the host transport; optionally flips one payload bit of received message `at`.
class Counting final : public Transport {
    std::unique_ptr<Transport> inner_;
    long corrupt_at_;
    long seen_ = 0;
public:
    std::atomic<std::uint64_t> sent{0};
    Counting(std::unique_ptr<Transport> inner, long corrupt_at) : inner_(std::move(inner)), corrupt_at_(corrupt_at) {}
    void send(std::span<const std::byte> b) override { sent += b.size(); inner_->send(b); }
    void receive(std::span<std::byte> b) override { inner_->receive(b); flip(b); }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        sent += out.size();
        inner_->exchange(out, in);
        flip(in);
    }
    std::string name() const override { return "memory-counting"; }
private:
    void flip(std::span<std::byte> b) {
        if (seen_++ == corrupt_at_) b[(b.size() - sizeof(Trailer)) / 3U] ^= std::byte{0x10};
    }
};

// In-process device exchange: messages per direction, one-way when asked, counting bytes and early-receive hints.
struct DeviceWire {
    std::mutex mutex;
    std::condition_variable ready;
    std::array<std::deque<std::vector<std::byte>>, 2> to; // to[r]: messages for rank r
};
class FakeDirect final : public DeviceExchange {
    std::shared_ptr<DeviceWire> wire_;
    unsigned rank_;
    bool one_way_form_;
    ExchangeDirection next_ = ExchangeDirection::both, current_ = ExchangeDirection::both;
    std::size_t bytes_ = 0;
    DeviceExchangeStats stats_;
public:
    std::atomic<unsigned> hints{0};
    FakeDirect(std::shared_ptr<DeviceWire> wire, unsigned rank, bool one_way_form) : wire_(std::move(wire)), rank_(rank), one_way_form_(one_way_form) {}
    std::size_t capacity() const noexcept override { return std::size_t{32} << 20U; }
    unsigned rings() const noexcept override { return 2U; }
    bool use_direction(ExchangeDirection d) override {
        if (!one_way_form_) return false;
        next_ = d;
        return true;
    }
    void expect_next(std::size_t bytes, unsigned) override { if (bytes) ++hints; }
    void start(hipStream_t, const void* local, std::size_t bytes, const ExchangeIdentity&) override {
        current_ = next_;
        next_ = ExchangeDirection::both;
        bytes_ = bytes;
        ++stats_.exchanges;
        if (current_ != ExchangeDirection::both) ++stats_.one_way;
        if (current_ != ExchangeDirection::receive) {
            const auto* p = static_cast<const std::byte*>(local);
            {
                std::lock_guard lock(wire_->mutex);
                wire_->to[1U - rank_].emplace_back(p, p + bytes);
            }
            wire_->ready.notify_all();
            stats_.bytes_tx += bytes;
        }
    }
    void finish(hipStream_t, void* remote) override {
        if (current_ == ExchangeDirection::send) return;
        std::unique_lock lock(wire_->mutex);
        if (!wire_->ready.wait_for(lock, std::chrono::seconds(30), [&] { return !wire_->to[rank_].empty(); }))
            throw std::runtime_error("fake direct: receive timed out");
        auto message = std::move(wire_->to[rank_].front());
        wire_->to[rank_].pop_front();
        if (message.size() != bytes_) throw std::runtime_error("fake direct: size differs");
        if (remote) std::memcpy(remote, message.data(), message.size());
        stats_.bytes_rx += bytes_;
    }
    std::string name() const override { return "fake-direct"; }
    DeviceExchangeStats stats() const override { return stats_; }
    std::string failure() const override { return {}; }
};

struct Rank {
    Counting* host = nullptr;
    FakeDirect* direct = nullptr;
    std::unique_ptr<Communicator> comm;
};
struct PairOptions {
    bool direct = false, one_way_form = true;
    long corrupt_rank1_at = -1;
    std::size_t max_bytes = std::size_t{1} << 20U;
};
std::array<Rank, 2> make_pair(const PairOptions& o) {
    auto p = memory_pair();
    std::array<std::unique_ptr<Transport>, 2> t{std::move(p.first), std::move(p.second)};
    auto wire = std::make_shared<DeviceWire>();
    std::array<Rank, 2> ranks;
    for (unsigned r = 0; r < 2U; ++r) {
        auto host = std::make_unique<Counting>(std::move(t[r]), r == 1U ? o.corrupt_rank1_at : -1L);
        ranks[r].host = host.get();
        const Group group{r, 2U, 1U, 0x7465737400000001ULL};
        if (o.direct) {
            auto direct = std::make_unique<FakeDirect>(wire, r, o.one_way_form);
            ranks[r].direct = direct.get();
            ranks[r].comm = std::make_unique<Communicator>(group, std::move(host), std::move(direct), o.max_bytes);
        } else {
            ranks[r].comm = std::make_unique<Communicator>(group, std::move(host), o.max_bytes);
        }
    }
    return ranks;
}
// Runs one body per rank on two threads; returns each rank's error ("" = passed).
std::array<std::string, 2> run(std::array<Rank, 2>& ranks, const std::function<void(unsigned, Rank&)>& body) {
    std::array<std::string, 2> errors;
    auto one = [&](unsigned r) {
        try { body(r, ranks[r]); } catch (const std::exception& e) { errors[r] = e.what(); }
    };
    std::thread peer(one, 1U);
    one(0U);
    peer.join();
    return errors;
}

// One ready-view (deferred CRC) exchange of n bytes.
// compare = false: the payload is not compared (a corrupted view is delivered before its deferred CRC is known).
void staged(Communicator& c, unsigned rank, std::size_t n, unsigned salt, bool compare = true) {
    auto producer = c.prepare_staged_ready(nullptr, n);
    if (!producer.data) throw std::runtime_error("no ready view for this size");
    auto* data = static_cast<std::byte*>(producer.data);
    for (std::size_t i = 0; i < n; ++i) data[i] = pattern(rank, i, salt);
    __atomic_store_n(&producer.flag->done, producer.token, __ATOMIC_RELEASE);
    c.start_staged_ready(nullptr, n, Tag{static_cast<std::uint32_t>(TagKind::raw_slots), 0U, salt});
    auto consumer = c.finish_staged_ready();
    const auto* got = static_cast<const std::byte*>(consumer.data);
    for (std::size_t i = 0; compare && i < n; ++i)
        if (got[i] != pattern(1U - rank, i, salt)) throw std::runtime_error("ready view payload differs");
    __atomic_store_n(&consumer.flag->done, consumer.token, __ATOMIC_RELEASE);
    c.release_staged_ready(nullptr);
}
void host_control(Communicator& c, unsigned rank, unsigned salt) {
    std::array<std::byte, 512> out{}, in{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = pattern(rank, i, salt);
    c.exchange_host(out, in, Tag{static_cast<std::uint32_t>(TagKind::control), 0xC0U, salt});
    for (std::size_t i = 0; i < in.size(); ++i)
        if (in[i] != pattern(1U - rank, i, salt)) throw std::runtime_error("exchange_host payload differs");
}
void device_control(Communicator& c, unsigned rank, unsigned salt) {
    std::array<std::byte, 512> out{}, in{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = pattern(rank, i, salt);
    c.exchange(nullptr, out.data(), in.data(), out.size(), Tag{static_cast<std::uint32_t>(TagKind::control), 0xC0U, salt});
    for (std::size_t i = 0; i < in.size(); ++i)
        if (in[i] != pattern(1U - rank, i, salt)) throw std::runtime_error("exchange payload differs");
}
// One-way transfer of n bytes from rank `from`; returns transfer()'s answer and checks the payload on the receiver.
bool one_way(Communicator& c, unsigned rank, unsigned from, std::size_t n, unsigned salt) {
    std::vector<std::byte> buffer(n);
    const bool sending = rank == from;
    if (sending)
        for (std::size_t i = 0; i < n; ++i) buffer[i] = pattern(from, i, salt);
    const bool moved = c.transfer(nullptr, sending, buffer.data(), n, Tag{static_cast<std::uint32_t>(TagKind::raw_slots), 0U, salt});
    if (moved && !sending)
        for (std::size_t i = 0; i < n; ++i)
            if (buffer[i] != pattern(from, i, salt)) throw std::runtime_error("one-way payload differs at " + std::to_string(i));
    return moved;
}

void carry_interop() {
    using Step = std::function<void(Communicator&, unsigned)>;
    const Step staged_step = [](Communicator& c, unsigned r) { staged(c, r, 4096U, 1U); };
    const Step host = [](Communicator& c, unsigned r) { host_control(c, r, 2U); };
    const Step device = [](Communicator& c, unsigned r) { device_control(c, r, 2U); }; // the same collective as `host`
    struct Case { const char* name; std::vector<Step> rank0, rank1; };
    const Case cases[] = {
        {"hd", {staged_step, host}, {staged_step, device}},
        {"dh", {staged_step, device}, {staged_step, host}},
        {"hh", {staged_step, host, host}, {staged_step, host, host}},
        {"hhdd", {staged_step, host, host, device, device}, {staged_step, host, host, device, device}},
        {"staged-host-staged-device", {staged_step, host, staged_step, device}, {staged_step, host, staged_step, device}},
        {"staged-host-staged-mixed", {staged_step, host, staged_step, device}, {staged_step, device, staged_step, host}},
    };
    for (const auto& k : cases) {
        std::fprintf(stderr, "[dist-contract] carry_interop case=%s\n", k.name);
        auto ranks = make_pair({});
        const auto errors = run(ranks, [&](unsigned r, Rank& rank) {
            for (const auto& step : r == 0U ? k.rank0 : k.rank1) step(*rank.comm, r);
        });
        expect(errors[0].empty() && errors[1].empty(), std::string("carry ") + k.name + ": " + errors[0] + " | " + errors[1]);
    }
    // The ready view's payload is corrupted on rank 1: its deferred CRC is checked by the next exchange_host().
    // (The view is delivered before its CRC is known, so the payload is not compared here.)
    for (const bool host_next : {true, false}) {
        std::fprintf(stderr, "[dist-contract] carry_interop corrupted host_next=%d\n", host_next ? 1 : 0);
        auto fresh = make_pair({.corrupt_rank1_at = 0});
        const auto errs = run(fresh, [&](unsigned r, Rank& rank) {
            auto& c = *rank.comm;
            staged(c, r, 4096U, 1U, false);
            if (host_next) host_control(c, r, 2U);
            else device_control(c, r, 2U);
        });
        expect(errs[1].find("payload CRC mismatch (deferred") != std::string::npos,
               std::string("corrupted deferred payload, then ") + (host_next ? "exchange_host" : "exchange") +
                   ": rank 1 must fail on the deferred CRC, got '" + errs[1] + "'");
        expect(errs[0].empty(), "corrupted deferred payload: rank 0 (sound data) passes, got '" + errs[0] + "'");
    }
}

void one_way_routes() {
    const std::size_t sizes[] = {std::size_t{64} << 10U, std::size_t{1} << 20U, (std::size_t{1} << 20U) + 64U,
                                 std::size_t{16} << 20U, (std::size_t{5} << 20U) + 192U};
    for (const std::size_t n : sizes) {
        for (const unsigned from : {0U, 1U}) {
            std::fprintf(stderr, "[dist-contract] one_way_routes bytes=%zu from=%u\n", n, from);
            auto ranks = make_pair({.direct = true, .max_bytes = std::size_t{16} << 20U});
            const std::string route = ranks[0].comm->route_text();
            std::array<bool, 2> moved{};
            std::array<unsigned, 2> records{};
            const auto errors = run(ranks, [&](unsigned r, Rank& rank) {
                const auto before = stub_event_records;
                moved[r] = one_way(*rank.comm, r, from, n, 5U);
                records[r] = stub_event_records - before;
            });
            const std::string label = "one-way " + std::to_string(n) + " B from rank " + std::to_string(from) + " (" + route + ")";
            if (n == (std::size_t{64} << 10U) && !std::getenv("STRIX_DIST_TRACE"))
                expect(records[from] == 1U && records[1U - from] == 1U, label + ": one event per active host copy");
            expect(errors[0].empty() && errors[1].empty(), label + ": " + errors[0] + " | " + errors[1]);
            expect(moved[0] && moved[1], label + ": transfer() must move it on both ranks");
            const unsigned to = 1U - from;
            const auto rx = ranks[to].direct->stats(), tx = ranks[from].direct->stats();
            expect(ranks[to].host->sent.load() == 0U, label + ": the receiver sent " + std::to_string(ranks[to].host->sent.load()) + " host bytes");
            expect(rx.bytes_tx == 0U && tx.bytes_rx == 0U, label + ": device bytes in the unused direction");
            expect(ranks[0].direct->hints.load() == 0U && ranks[1].direct->hints.load() == 0U, label + ": an early receive was posted");
            const std::uint64_t host = ranks[from].host->sent.load();
            const bool device_only = n > (std::size_t{64} << 10U) && n <= (std::size_t{1} << 20U);
            const bool host_only = n <= (std::size_t{64} << 10U);
            if (host_only) expect(host == n + sizeof(Trailer) && tx.bytes_tx == 0U, label + ": host route");
            else if (device_only) expect(host == 0U && tx.bytes_tx == n, label + ": device route");
            else expect(host > sizeof(Trailer) && tx.bytes_tx > 0U && host - sizeof(Trailer) + tx.bytes_tx == n, label + ": mixed route split");
        }
    }
    // An engine without a one-way form: device and mixed routes refuse (nothing sent), the host route still goes one way.
    for (const std::size_t n : {std::size_t{1} << 20U, std::size_t{16} << 20U, std::size_t{64} << 10U}) {
        std::fprintf(stderr, "[dist-contract] one_way_routes unsupported bytes=%zu\n", n);
        auto ranks = make_pair({.direct = true, .one_way_form = false, .max_bytes = std::size_t{16} << 20U});
        std::array<bool, 2> moved{};
        const auto errors = run(ranks, [&](unsigned r, Rank& rank) { moved[r] = one_way(*rank.comm, r, 0U, n, 6U); });
        const bool host_route = n <= (std::size_t{64} << 10U);
        expect(errors[0].empty() && errors[1].empty(), "no one-way form " + std::to_string(n) + ": " + errors[0] + " | " + errors[1]);
        expect(moved[0] == host_route && moved[1] == host_route, "no one-way form " + std::to_string(n) + ": transfer() answer");
        if (!host_route) expect(ranks[0].host->sent.load() == 0U && ranks[1].host->sent.load() == 0U, "no one-way form: nothing sent");
    }
}

void one_way_carry() {
    // Deferred CRCs ride one-way messages per direction: rank 0 -> 1, then 1 -> 0, then a control round.
    {
        std::fputs("[dist-contract] one_way_carry normal\n", stderr);
        auto ranks = make_pair({});
        const auto errors = run(ranks, [&](unsigned r, Rank& rank) {
            staged(*rank.comm, r, 4096U, 1U);
            if (!one_way(*rank.comm, r, 0U, std::size_t{32} << 10U, 7U)) throw std::runtime_error("host one-way refused");
            if (!one_way(*rank.comm, r, 1U, std::size_t{32} << 10U, 8U)) throw std::runtime_error("host one-way refused");
            host_control(*rank.comm, r, 9U);
            device_control(*rank.comm, r, 10U);
        });
        expect(errors[0].empty() && errors[1].empty(), "one-way carry: " + errors[0] + " | " + errors[1]);
    }
    // A corrupted ready view on rank 1 fails at the first message rank 1 receives (rank 0's one-way transfer).
    {
        std::fputs("[dist-contract] one_way_carry corrupted\n", stderr);
        auto ranks = make_pair({.corrupt_rank1_at = 0});
        const auto errors = run(ranks, [&](unsigned r, Rank& rank) {
            auto& c = *rank.comm;
            staged(c, r, 4096U, 1U, false);
            static_cast<void>(one_way(c, r, 0U, std::size_t{32} << 10U, 7U));
        });
        expect(errors[1].find("payload CRC mismatch (deferred") != std::string::npos,
               "corrupted ready view then one-way 0->1: rank 1 must fail on the deferred CRC, got '" + errors[1] + "'");
        expect(errors[0].empty(), "corrupted ready view then one-way 0->1: rank 0 passes, got '" + errors[0] + "'");
    }
}
} // namespace

int main() {
    setenv("STRIX_HOST_ANON_PIN", "0", 1);
    // HIP's pinned allocations must support the alignas(64) staged readiness control lines. A CPU stub
    // based on calloc's max_align_t guarantee is insufficient even when most ordinary tests happen to pass.
    void* coherent = nullptr;
    expect(hipHostMalloc(&coherent, sizeof(StagedReady), hipHostMallocCoherent) == hipSuccess && coherent != nullptr,
           "HIP stub coherent allocation");
    if (coherent) {
        expect(reinterpret_cast<std::uintptr_t>(coherent) % alignof(StagedReady) == 0U,
               "HIP stub pinned memory must align StagedReady");
        (void)hipHostFree(coherent);
    }
    if (failures) return 1;
    std::fputs("[dist-contract] starting carry_interop\n", stderr);
    carry_interop();
    std::fputs("[dist-contract] starting one_way_routes\n", stderr);
    one_way_routes();
    std::fputs("[dist-contract] starting one_way_carry\n", stderr);
    one_way_carry();
    if (failures) {
        std::fprintf(stderr, "dist exchange contract: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dist exchange contract: all checks passed\n");
    return 0;
}
