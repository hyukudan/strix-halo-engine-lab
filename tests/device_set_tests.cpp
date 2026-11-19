// device::Set: one resolved view of units and links from the topology declaration, measured trip costs per gap class,
// canonical digest. Synthetic topologies: a two-node USB4 pair, a same-box PCIe pair, an unknown link, one node.
#include "strix/device/set.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
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
using namespace strix::device;

TopologyNode node(const char* id, const char* host, std::uint32_t rank, double gbps) {
    TopologyNode n;
    n.id = id;
    n.host = host;
    n.rank = rank;
    n.vendor = "amd";
    n.arch = "gfx1151";
    n.memory_bytes = 100ULL << 30;
    n.memory_gbytes_ps = gbps;
    n.rate_source = gbps > 0.0 ? FactSource::declared : FactSource::unknown;
    return n;
}
TopologyLink tlink(const char* name, std::uint32_t a, std::uint32_t b, const char* cls, double gbps, bool measured, double unc) {
    TopologyLink l;
    l.name = name;
    l.a = a;
    l.b = b;
    l.link_class = cls;
    l.gbytes_ps = gbps;
    l.rtt_us = 30.0;
    l.uncertainty = unc;
    l.measured = measured;
    return l;
}
Topology usb4_pair() {
    Topology t;
    t.source = "<synthetic usb4 pair>";
    t.nodes = {node("a", "host-a", 0, 243.0), node("b", "host-b", 1, 243.0)};
    t.links = {tlink("ab", 0, 1, "usb4", 4.58, false, 0.30)};
    return t;
}

void test_usb4_pair() {
    const Set s = resolve_set(usb4_pair());
    check(s.units.size() == 2U && s.units[0].read_bytes_per_ms == 243.0e6 && s.units[1].rank == 1U, "units: rate GB/s -> bytes/ms, rank order");
    check(s.units[0].read_src == FactSource::declared && s.units[0].host != s.units[1].host, "units: source and host kept");
    const SetLink* l = s.link(1U, 0U);
    check(l != nullptr && l->a == 0U && l->b == 1U, "link lookup is undirected");
    check(l != nullptr && l->bytes_per_ms == 4.58e6 && l->src == RateSource::nominal && l->resource == "link-ab", "nominal class rate, default resource");
    check(l != nullptr && !l->cost_measured[0] && !l->cost_measured[1] && !l->cost_measured[2], "no trip cost until measured");
}

void test_measured_costs_and_digest() {
    Set s = resolve_set(usb4_pair());
    const std::uint64_t d0 = digest(s);
    strix::dist::LinkCost b2b;
    b2b.latency_us = 37.5;
    b2b.us_per_byte = 1000.0 / 3.4e6;
    set_measured(s, 1U, 0U, GapClass::back_to_back, b2b);
    check(s.links[0].cost_measured[0] && s.links[0].cost[0].latency_us == 37.5, "set_measured records the class");
    const std::string line = describe(s);
    check(line.find("set: units=2 links=1 usb4 b2b=37.5us in_token=- idle=-") == 0U, "describe line");
    const std::uint64_t d1 = digest(s);
    check(d1 != d0, "digest changes with a measured cost");
    // the same facts in another declaration order: same digest
    Topology t = usb4_pair();
    t.links.push_back(tlink("ab2", 0, 1, "usb4", 4.58, false, 0.30));
    Topology u = t;
    std::swap(u.links[0], u.links[1]);
    check(digest(resolve_set(t)) == digest(resolve_set(u)), "digest invariant under link order");
    b2b.latency_us = 37.6;
    Set s2 = resolve_set(usb4_pair());
    set_measured(s2, 0U, 1U, GapClass::back_to_back, b2b);
    check(digest(s2) != d1, "digest differs when a measured cost differs");
    check(throws([&] { set_measured(s2, 0U, 2U, GapClass::idle, b2b); }), "no link between the units: refused");
    check(gap_class(40.0) == GapClass::back_to_back && gap_class(250.0) == GapClass::in_token && gap_class(30000.0) == GapClass::idle,
          "gap classes by idle time");
}

void test_same_box_and_unknown() {
    Topology t;
    t.source = "<synthetic same box>";
    t.nodes = {node("gpu0", "box", 0, 960.0), node("gpu1", "box", 1, 960.0), node("far", "other", 2, 0.0)};
    t.links = {tlink("pcie", 0, 1, "pcie4x16", 25.0, true, 0.0), tlink("net", 1, 2, "", 0.0, false, 0.0)};
    const Set s = resolve_set(t);
    check(s.units[0].host == s.units[1].host, "same-box units share the host");
    check(s.link(0U, 1U) != nullptr && s.link(0U, 1U)->src == RateSource::measured && s.link(0U, 1U)->bytes_per_ms == 25.0e6, "declared measured rate");
    check(s.link(1U, 2U) != nullptr && s.link(1U, 2U)->bytes_per_ms == 0.0 && s.link(1U, 2U)->src == RateSource::unknown, "unknown link stays 0 / unknown");
    check(s.units[2].read_bytes_per_ms == 0.0 && s.units[2].read_src == FactSource::unknown, "unknown memory rate stays 0");
    check(s.link(0U, 2U) == nullptr, "no link: none invented");
    Topology bad = t;
    bad.links.push_back(tlink("self", 1, 1, "usb4", 4.58, false, 0.3));
    check(throws([&] { (void)resolve_set(bad); }), "a link joining no two nodes is refused");
}

void test_one_node() {
    Topology t;
    t.source = "single";
    t.nodes = {node("solo", "", 0, 243.0)};
    const Set s = resolve_set(t);
    check(s.units.size() == 1U && s.links.empty(), "one node: one unit, no link");
}
}  // namespace

int main() {
    test_usb4_pair();
    test_measured_costs_and_digest();
    test_same_box_and_unknown();
    test_one_node();
    if (failures != 0) {
        std::fprintf(stderr, "device set tests: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("device set tests: all passed");
    return 0;
}
