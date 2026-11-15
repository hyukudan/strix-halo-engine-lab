// strix::handoff::Lanes over loopback: an urgent piece submitted behind a deep bulk queue reaches its receiver early
// (FIFO would deliver it last), every piece arrives once with a valid manifest, and abort releases queued bulk once.
#include "strix/handoff_lanes.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace ho = strix::handoff;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
constexpr std::uint32_t kUrgent = 1, kBulk = 2;
constexpr unsigned kBulkPieces = 24;
constexpr std::size_t kBulkBytes = 256 << 10, kUrgentBytes = 64 << 10;

struct Receiver {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;  // (kind, index) in delivery order
    std::exception_ptr error;
    std::thread thread;
    void run(ho::Listener& l, std::atomic<bool>* go) {
        thread = std::thread([this, &l, go] {
            try {
                ho::ReceiverOptions o;
                o.max_buffered_wire_bytes = 512 << 10;  // a slow consumer: the transport stops reading ahead
                o.socket_buffer = 64 << 10;
                auto s = l.accept(o);
                while (go && !go->load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                ho::Piece p;
                while (s->next(p)) {
                    order.emplace_back(p.header.kind, p.header.index);
                    for (std::size_t i = 0; i < p.plain.size(); i += 4099)
                        if (p.plain[i] != static_cast<std::uint8_t>(p.header.kind * 31 + p.header.index)) throw std::runtime_error("payload");
                }
                s->reply(0);
            } catch (...) { error = std::current_exception(); }
        });
    }
};
}  // namespace

int main() try {
    ho::Listener la(0, "127.0.0.1"), lb(0, "127.0.0.1");
    std::atomic<bool> go{false};
    Receiver ra, rb;
    ra.run(la, &go);
    rb.run(lb, nullptr);
    ho::SenderOptions so;
    so.codec = ho::Codec::identity;
    so.socket_buffer = 64 << 10;
    so.connect_timeout = std::chrono::seconds(20);
    auto a = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", la.port(), ""}, ho::Hello{7, "lanes-test", 0, 1, {}}, so);
    auto b = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", lb.port(), ""}, ho::Hello{8, "lanes-test", 0, 1, {}}, so);
    std::vector<std::vector<std::uint8_t>> data;
    for (unsigned i = 0; i < 2 * kBulkPieces; ++i)
        data.emplace_back(kBulkBytes, static_cast<std::uint8_t>(kBulk * 31 + i % kBulkPieces));
    std::vector<std::uint8_t> urgent(kUrgentBytes, static_cast<std::uint8_t>(kUrgent * 31 + 0));
    std::atomic<unsigned> released{0};
    {
        ho::Lanes lanes({a.get(), b.get()}, ho::Lanes::Options{kBulkBytes, true});
        for (unsigned i = 0; i < kBulkPieces; ++i)
            for (unsigned dst = 0; dst < 2; ++dst) {
                auto& d = data[dst * kBulkPieces + i];
                lanes.bulk(dst, {kBulk, i, 0, i, 1}, {strix::bp_codec::Region{d.data(), d.size(), 1}}, [&] { ++released; });
            }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the feeder fills what the slow receiver allows
        const strix::bp_codec::Region r{urgent.data(), urgent.size(), 1};
        lanes.urgent(0, {kUrgent, 0, 0, 0, 1}, {&r, 1}, [&] { ++released; });
        go = true;
        lanes.drain();
        require(a->finish().status == 0 && b->finish().status == 0, "replies");
        const auto st = lanes.stats();
        require(st.urgent_pieces == 1 && st.bulk_pieces == 2 * kBulkPieces && st.bulk_bytes == 2 * kBulkPieces * kBulkBytes, "stats");
    }
    ra.thread.join(); rb.thread.join();
    if (ra.error) std::rethrow_exception(ra.error);
    if (rb.error) std::rethrow_exception(rb.error);
    require(released.load() == 2 * kBulkPieces + 1, "every release ran once");
    require(ra.order.size() == kBulkPieces + 1 && rb.order.size() == kBulkPieces, "every piece arrived");
    std::size_t at = ra.order.size();
    for (std::size_t i = 0; i < ra.order.size(); ++i) if (ra.order[i].first == kUrgent) at = i;
    std::printf("urgent piece delivered at position %zu of %zu (FIFO: %u)\n", at, ra.order.size(), kBulkPieces);
    require(at < kBulkPieces / 2, "urgent piece must overtake the queued bulk");
    for (unsigned i = 0, k = 0; i < ra.order.size(); ++i)
        if (ra.order[i].first == kBulk) require(ra.order[i].second == k++, "bulk keeps its order");

    // abort: queued bulk is released exactly once, later bulk is refused (and released)
    {
        ho::Listener lc(0, "127.0.0.1");
        Receiver rc;
        std::atomic<bool> never{false};
        rc.run(lc, &never);
        auto c = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", lc.port(), ""}, ho::Hello{9, "lanes-test", 0, 1, {}}, so);
        std::atomic<unsigned> dropped{0};
        {
            ho::Lanes lanes({c.get()}, ho::Lanes::Options{kBulkBytes, true});
            for (unsigned i = 0; i < kBulkPieces; ++i)
                lanes.bulk(0, {kBulk, i, 0, i, 1}, {strix::bp_codec::Region{data[i].data(), data[i].size(), 1}}, [&] { ++dropped; });
            lanes.abort();
            bool refused = false;
            try { lanes.bulk(0, {kBulk, 0, 0, 0, 1}, {strix::bp_codec::Region{data[0].data(), 1, 1}}, [&] { ++dropped; }); }
            catch (const std::runtime_error&) { refused = true; }
            require(refused, "bulk after abort is refused");
        }
        c->abort("test");
        c.reset();  // the Sender releases what it was handed
        never = true;
        lc.shutdown();
        rc.thread.join();
        require(dropped.load() == kBulkPieces + 1, "abort releases every callback once");
    }
    std::puts("PASS: handoff lanes (urgent overtakes bulk, manifest, abort releases)");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
