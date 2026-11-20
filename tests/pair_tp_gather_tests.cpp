// Host tests of the shared tensor-parallel gather link (strix/pair/gather_link.hpp) and the placement inputs
// (strix/pair/tp_inputs.hpp). Invariants: the slot swap addresses (local = slot[rank], remote = slot[1 - rank]); the
// staged route calls its exchange directly (the Communicator orders its copy after the producer on the stream), no
// route waits for the stream on the host; the wire
// tag sequence is the call count and survives a counter restart; rates and trip costs come only from a flag, the
// declaration or the measurement (no compiled default).
#include "strix/pair/gather_link.hpp"
#include "strix/pair/tp_inputs.hpp"

#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
template<class F> bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

struct Event { char what; const void* local; void* remote; std::size_t bytes; std::uint64_t tag; };
std::vector<Event> events;  // 's' = host stream wait, 'l' = legacy exchange, 'f' = fast exchange

struct FakeOps {
    using Stream = int*;
    static Stream stream(void* s) noexcept { return static_cast<int*>(s); }
    static void sync(Stream) { events.push_back({'s', nullptr, nullptr, 0U, 0U}); }
    static std::uint64_t tag(std::uint64_t call) noexcept { return call; }
};
template<char C> struct FakeExchange {
    void exchange(int*, const void* local, void* remote, std::size_t bytes, std::uint64_t tag) {
        events.push_back({C, local, remote, bytes, tag});
    }
    int drains = 0;
    bool corrupt = false;  // the seal of the last message fails its integrity check
    void drain() {
        ++drains;
        if (corrupt) throw std::runtime_error("seal: deferred check of the last message failed");
    }
};
using Legacy = FakeExchange<'l'>;
using Fast = FakeExchange<'f'>;
using Link = strix::pair::GatherLink<FakeOps, Legacy, Fast>;

void test_routes() {
    using strix::pair::GatherRoute;
    Legacy legacy;
    Fast fast;
    int stream_object = 0;
    std::vector<std::uint8_t> slots(2U * 64U);
    for (const std::uint32_t rank : {0U, 1U}) {
        // staged: the exchange only, every call (no host stream wait before it)
        events.clear();
        Link st(rank, GatherRoute::staged, &legacy, nullptr);
        st.gather(&stream_object, slots.data(), 64U, 40U);
        Link::fn(&st, &stream_object, slots.data(), 64U, 64U);
        check(events.size() == 2U && events[0].what == 'l' && events[1].what == 'l', "staged: one exchange per gather, no host wait");
        check(events[0].local == slots.data() + rank * 64U && events[0].remote == slots.data() + (1U - rank) * 64U,
              "staged: local = slot[rank], remote = slot[1 - rank]");
        check(events[0].bytes == 40U && events[0].tag == 0U && events[1].tag == 1U, "staged: bytes and tag sequence");
        check(st.calls() == 2U && st.link_ms() >= 0.0, "staged: counters");
        // device direct: legacy exchange, no host wait
        events.clear();
        Link dd(rank, GatherRoute::device_direct, &legacy, nullptr);
        dd.gather(&stream_object, slots.data(), 64U, 8U);
        check(events.size() == 1U && events[0].what == 'l', "device direct: no host wait");
        // channel: fast exchange only, no host wait
        events.clear();
        Link ch(rank, GatherRoute::channel, &legacy, &fast);
        for (int i = 0; i < 3; ++i) ch.gather(&stream_object, slots.data(), 64U, 16U);
        bool fast_only = events.size() == 3U;
        for (const auto& e : events) fast_only = fast_only && e.what == 'f';
        check(fast_only, "channel: no host wait, never the legacy exchange");
        check(events.size() == 3U && events[2].tag == 2U && events[2].local == slots.data() + rank * 64U, "channel: tags and slots");
        // counter restart keeps the wire tag sequence
        ch.reset_counters();
        events.clear();
        ch.gather(&stream_object, slots.data(), 64U, 16U);
        check(ch.calls() == 1U && events.size() == 1U && events[0].tag == 3U, "reset_counters: calls restart, tags continue");
    }
    check(throws([&] { (void)Link(2U, GatherRoute::staged, &legacy, nullptr); }), "rank 2 refused");
    check(throws([&] { (void)Link(0U, GatherRoute::channel, &legacy, nullptr); }), "channel route without its exchange refused");
    check(throws([&] { (void)Link(0U, GatherRoute::staged, nullptr, &fast); }), "staged route without the communicator refused");
    Link st(0U, GatherRoute::staged, &legacy, nullptr);
    check(throws([&] { st.gather(&stream_object, slots.data(), 64U, 65U); }), "message larger than its slot refused");
    using Bare = strix::pair::GatherLink<FakeOps, Legacy, strix::pair::NoExchange>;
    check(throws([&] { (void)Bare(0U, GatherRoute::channel, &legacy, nullptr); }), "unbound channel route refused");
}

// L1: the channel's last message is checked by the drain's seal; finish() must surface that failure (both ranks call it)
// and the inline-checked routes have nothing to finish. Route policy: one backend-neutral function.
void test_finish_and_route() {
    using strix::pair::GatherRoute;
    Legacy legacy;
    Fast fast;
    int stream_object = 0;
    std::vector<std::uint8_t> slots(2U * 64U);
    Link ch(0U, GatherRoute::channel, &legacy, &fast);
    ch.gather(&stream_object, slots.data(), 64U, 64U);
    ch.finish();
    check(fast.drains == 1, "finish: the channel route drains (seal of the last message)");
    fast.corrupt = true;
    check(throws([&] { ch.finish(); }), "finish: a failed check of the last gather is reported, not swallowed");
    Link st(1U, GatherRoute::staged, &legacy, nullptr);
    st.finish();
    check(legacy.drains == 0, "finish: staged route checks inline, no drain");
    check(strix::pair::select_route(false, false) == GatherRoute::staged, "route: staged when nothing else");
    check(strix::pair::select_route(true, false) == GatherRoute::device_direct, "route: device direct when offered");
    check(strix::pair::select_route(true, true) == GatherRoute::channel && strix::pair::select_route(false, true) == GatherRoute::channel,
          "route: the channel when requested");
}

void test_inputs() {
    namespace tp = strix::pair::tp;
    const auto f = tp::rank_rates({200.0, 180.0}, {243.0, 243.0}, 2U);
    check(f.source == "flag" && f.gbytes_ps[1] == 180.0, "rates: the flag overrides the declaration");
    const auto d = tp::rank_rates({}, {243.0, 230.0}, 2U);
    check(d.source == "topology" && d.gbytes_ps[0] == 243.0 && d.gbytes_ps[1] == 230.0, "rates: the declaration");
    check(throws([] { (void)tp::rank_rates({}, {}, 2U); }), "rates: no flag and no declaration refused");
    check(throws([] { (void)tp::rank_rates({}, {243.0, 0.0}, 2U); }), "rates: an undeclared rank refused");
    check(throws([] { (void)tp::rank_rates({195.0}, {243.0, 243.0}, 2U); }), "rates: a flag with the wrong rank count refused");
    check(throws([] { (void)tp::rank_rates({195.0, -1.0}, {}, 2U); }), "rates: a non-positive flag refused");
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(throws([&] { (void)tp::rank_rates({inf, 180.0}, {}, 2U); }), "rates: infinite override refused at admission");
    check(throws([&] { (void)tp::rank_rates({}, {243.0, inf}, 2U); }), "rates: infinite topology rate refused");
    check(throws([&] { (void)tp::rank_rates({nan, 180.0}, {}, 2U); }), "rates: NaN override refused");

    strix::dist::LinkCost lc;
    lc.latency_us = 34.0;
    lc.us_per_byte = 1.0 / 3500.0;  // 3.5 GB/s
    const auto t = tp::trip_cost(lc);
    check(t.latency_ms == 0.034 && t.bytes_per_ms == lc.bytes_per_ms() && t.bytes_per_ms > 3.49e6 && t.bytes_per_ms < 3.51e6,
          "trip: measured latency and rate");
    const auto o = tp::trip_cost(lc, 12.0);
    check(o.latency_ms == 0.012 && o.bytes_per_ms == t.bytes_per_ms, "trip: the override replaces the latency only");
    check(throws([] { (void)tp::trip_cost(strix::dist::LinkCost{}); }), "trip: an invalid measurement refused");
    check(throws([&] { (void)tp::trip_cost(lc, -1.0); }), "trip: a negative override refused");
    check(throws([&] { (void)tp::trip_cost(lc, inf); }), "trip: infinite latency override refused");
    check(throws([&] { (void)tp::trip_cost(lc, nan); }), "trip: NaN latency override refused");
}

}  // namespace

// Device-direct size split (P796-A): the Communicator's table keeps small sizes on its host path, so those gathers are
// staged (host-waited) and the label says where the split is; the size-free routes ignore it.
void test_route_split() {
    namespace pr = strix::pair;
    struct R { std::size_t up_to; bool direct; unsigned rings, host_percent; };
    const std::vector<R> table{{65536U, false, 0U, 0U}, {1048576U, true, 2U, 0U}, {SIZE_MAX, true, 4U, 35U}};
    const std::vector<R> rings{{32768U, true, 1U, 0U}, {SIZE_MAX, true, 4U, 0U}};
    const std::vector<R> mixed_first{{65536U, true, 2U, 35U}, {SIZE_MAX, true, 4U, 0U}};
    check(pr::host_prefix_bytes(table) == 65536U, "host prefix of the measured table: 64 KiB");
    check(pr::host_prefix_bytes(rings) == 0U, "every size on the rings: no host prefix");
    check(pr::host_prefix_bytes(mixed_first) == 65536U, "a mixed entry still carries a host part: counted");
    check(pr::host_prefix_bytes(std::vector<R>{}) == 0U, "empty table: no host prefix");
    check(pr::effective_route(pr::GatherRoute::device_direct, 10240U, 65536U) == pr::GatherRoute::staged, "1-row gather: staged");
    check(pr::effective_route(pr::GatherRoute::device_direct, 65537U, 65536U) == pr::GatherRoute::device_direct, "above: device");
    check(pr::effective_route(pr::GatherRoute::channel, 10U, 65536U) == pr::GatherRoute::channel, "channel ignores the split");
    check(pr::effective_route(pr::GatherRoute::device_direct, 10U, 0U) == pr::GatherRoute::device_direct, "no split: device");
    check(pr::route_waits_on_host(pr::effective_route(pr::GatherRoute::device_direct, 10240U, 65536U)), "small gather waits on host");
    check(pr::gather_route_label(pr::GatherRoute::device_direct, 65536U) ==
              std::string("device direct (stream-ordered) above 65536 B; staged (host-staged exchange) up to 65536 B"), "split label");
    check(pr::gather_route_label(pr::GatherRoute::device_direct, 0U) == std::string("device direct (stream-ordered)"), "plain label");
    Legacy legacy;
    Fast fast;
    const Link d(0U, pr::GatherRoute::device_direct, &legacy, &fast, 65536U);
    check(d.route_for(10240U) == pr::GatherRoute::staged && d.route_for(1U << 20U) == pr::GatherRoute::device_direct &&
              d.host_max_bytes() == 65536U, "link: route per size");
    const Link st(0U, pr::GatherRoute::staged, &legacy, &fast, 65536U);
    check(st.host_max_bytes() == 0U && st.route_label() == std::string(pr::gather_route_name(pr::GatherRoute::staged)),
          "a split given to another route is dropped");
}

int main() {
    test_routes();
    test_finish_and_route();
    test_inputs();
    test_route_split();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("pair tp gather tests: all passed\n");
    return EXIT_SUCCESS;
}
