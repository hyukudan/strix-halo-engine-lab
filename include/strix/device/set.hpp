#pragma once
// One resolved, immutable view of the devices and links a multi-device plan may use (host only, vendor-agnostic).
//
// The facts come from the topology declaration (strix-topology-v1, topology.hpp: nodes with their memory and memory
// rate, links with class / rate / round trip / shared port) or its legacy two-node shape; this header only joins them
// into one view and adds what the declaration cannot hold: the MEASURED trip cost of a link per idle-gap class. The
// same cable costs far less back to back than after an idle gap, so a planner pricing decode trips with the
// back-to-back number underprices them; the class a trip belongs to follows from the compute time before it.
// Unknown stays unknown: a zero rate is never traversed, and a planner that needs a missing number refuses and names it.
// Every participant resolves the same declaration and agrees the measured costs, so digest() is equal on all of them.
// Host test: tests/device_set_tests.cpp.
// Home: shared (host shared). Consumers: the multi-device planner (strix/pair/planner.hpp).
#include "strix/device/topology.hpp"
#include "strix/dist/link_cost.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::device {

// Idle gap before a trip: back to back (< 100 us of compute since the previous trip), inside a token (< 10 ms), idle.
enum class GapClass : std::uint8_t { back_to_back = 0, in_token = 1, idle = 2 };
inline constexpr std::size_t kGapClasses = 3;
[[nodiscard]] constexpr const char* gap_class_name(GapClass g) noexcept {
    return g == GapClass::back_to_back ? "back_to_back" : g == GapClass::in_token ? "in_token" : "idle";
}
[[nodiscard]] constexpr GapClass gap_class(double gap_us) noexcept {
    return gap_us < 100.0 ? GapClass::back_to_back : gap_us < 10000.0 ? GapClass::in_token : GapClass::idle;
}

enum class RateSource : std::uint8_t { measured, nominal, unknown };
[[nodiscard]] constexpr const char* rate_source_name(RateSource s) noexcept {
    return s == RateSource::measured ? "measured" : s == RateSource::nominal ? "nominal" : "unknown";
}

struct Unit {                       // one GPU process endpoint (one rank)
    std::string id, host;           // topology node id; host groups the units of one machine ("" = not declared)
    std::string vendor, arch;       // "" = unknown
    std::uint32_t rank = 0;
    std::uint64_t memory_bytes = 0;     // usable device memory; 0 = unknown
    double read_bytes_per_ms = 0.0;     // memory read rate; 0 = unknown
    FactSource read_src = FactSource::unknown;
    std::vector<std::string> roles;     // empty = every multi-node role
};

struct SetLink {                    // undirected pair of units (declaration.hpp Link is the per-host attach record)
    std::string name;               // topology link name
    std::uint32_t a = 0, b = 0;     // unit indices, a < b
    std::string kind;               // link class (usb4, pcie4x16, 2.5gbe, ...); "" = unknown
    std::string resource;           // shared port: transfers on one resource do not overlap
    double bytes_per_ms = 0.0;      // useful payload rate, one direction; 0 = unknown
    double rtt_us = 0.0;            // declared / nominal round trip of one small message; 0 = unknown
    RateSource src = RateSource::unknown;
    double uncertainty = 0.0;       // relative half-width of the rate (nominal class or declared)
    std::array<dist::LinkCost, kGapClasses> cost{};   // measured trip cost per gap class
    std::array<bool, kGapClasses> cost_measured{};    // false = that class was not measured
};

struct Set {
    std::string source;             // the topology's source ("legacy", a path, ...)
    std::vector<Unit> units;        // index == rank
    std::vector<SetLink> links;      // sorted by (a, b, name)

    [[nodiscard]] const SetLink* link(std::uint32_t a, std::uint32_t b) const noexcept {
        if (a > b) std::swap(a, b);
        for (const auto& l : links) if (l.a == a && l.b == b) return &l;
        return nullptr;
    }
};

// Joins the topology's nodes and links. Rates: GB/s -> bytes/ms; a link with a measured rate is `measured`, one priced
// by its class `nominal`, one with neither `unknown` (rate 0). No trip cost is measured yet (set_measured adds them).
[[nodiscard]] inline Set resolve_set(const Topology& t) {
    Set s;
    s.source = t.source;
    for (std::uint32_t r = 0; r < t.size(); ++r) {
        const auto& n = t.nodes[r];
        Unit u;
        u.id = n.id;
        u.host = n.host;
        u.vendor = n.vendor;
        u.arch = n.arch;
        u.rank = r;
        u.memory_bytes = n.memory_bytes;
        u.read_bytes_per_ms = n.memory_gbytes_ps > 0.0 ? n.memory_gbytes_ps * 1e6 : 0.0;
        u.read_src = n.memory_gbytes_ps > 0.0 ? n.rate_source : FactSource::unknown;
        u.roles = n.roles;
        s.units.push_back(std::move(u));
    }
    for (const auto& tl : t.links) {
        if (tl.a >= t.size() || tl.b >= t.size() || tl.a == tl.b) throw std::invalid_argument("device set: link " + tl.name + " joins no two nodes");
        SetLink l;
        l.name = tl.name;
        l.a = std::min(tl.a, tl.b);
        l.b = std::max(tl.a, tl.b);
        l.kind = tl.link_class;
        l.resource = tl.port.empty() ? "link-" + tl.name : tl.port;
        l.bytes_per_ms = tl.gbytes_ps > 0.0 ? tl.gbytes_ps * 1e6 : 0.0;
        l.rtt_us = tl.rtt_us;
        l.src = tl.gbytes_ps <= 0.0 ? RateSource::unknown : tl.measured ? RateSource::measured : RateSource::nominal;
        l.uncertainty = tl.uncertainty;
        s.links.push_back(std::move(l));
    }
    std::sort(s.links.begin(), s.links.end(), [](const SetLink& x, const SetLink& y) {
        return x.a != y.a ? x.a < y.a : x.b != y.b ? x.b < y.b : x.name < y.name;
    });
    return s;
}

// Records a trip cost measured (and agreed by both ends) on the link joining units a and b for one gap class.
inline void set_measured(Set& s, std::uint32_t a, std::uint32_t b, GapClass g, const dist::LinkCost& c) {
    if (a > b) std::swap(a, b);
    for (auto& l : s.links) {
        if (l.a != a || l.b != b) continue;
        l.cost[static_cast<std::size_t>(g)] = c;
        l.cost_measured[static_cast<std::size_t>(g)] = true;
        return;
    }
    throw std::invalid_argument("device set: no link between units " + std::to_string(a) + " and " + std::to_string(b));
}

// Canonical 64-bit FNV-1a over every field in index / sorted order (doubles by bit pattern): equal on every participant
// that resolved the same facts and agreed the same measurements, different when any of them differs.
[[nodiscard]] inline std::uint64_t digest(const Set& s) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto bytes = [&h](const void* p, std::size_t n) {
        const auto* c = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ULL; }
    };
    const auto str = [&](std::string_view v) { const std::uint64_t n = v.size(); bytes(&n, sizeof n); bytes(v.data(), v.size()); };
    const auto num = [&](double v) { const auto b = std::bit_cast<std::uint64_t>(v); bytes(&b, sizeof b); };
    const auto u64 = [&](std::uint64_t v) { bytes(&v, sizeof v); };
    u64(s.units.size());
    for (const auto& u : s.units) {
        str(u.id); str(u.host); str(u.vendor); str(u.arch);
        u64(u.rank); u64(u.memory_bytes); num(u.read_bytes_per_ms); u64(static_cast<std::uint64_t>(u.read_src));
        u64(u.roles.size());
        for (const auto& r : u.roles) str(r);
    }
    std::vector<const SetLink*> order;
    for (const auto& l : s.links) order.push_back(&l);
    std::sort(order.begin(), order.end(), [](const SetLink* x, const SetLink* y) {
        return x->a != y->a ? x->a < y->a : x->b != y->b ? x->b < y->b : x->name < y->name;
    });
    u64(order.size());
    for (const SetLink* l : order) {
        str(l->name); u64(l->a); u64(l->b); str(l->kind); str(l->resource);
        num(l->bytes_per_ms); num(l->rtt_us); u64(static_cast<std::uint64_t>(l->src)); num(l->uncertainty);
        for (std::size_t g = 0; g < kGapClasses; ++g) {
            u64(l->cost_measured[g] ? 1U : 0U);
            if (l->cost_measured[g]) { num(l->cost[g].latency_us); num(l->cost[g].us_per_byte); num(l->cost[g].tail_us); }
        }
    }
    return h;
}

// One line: `set: units=2 links=1 usb4 b2b=37.5us in_token=- idle=- rate=4.58GB/s src=nominal`.
[[nodiscard]] inline std::string describe(const Set& s) {
    std::string out = "set: units=" + std::to_string(s.units.size()) + " links=" + std::to_string(s.links.size());
    char buf[96];
    for (const auto& l : s.links) {
        out += " " + (l.kind.empty() ? std::string("unknown") : l.kind);
        for (std::size_t g = 0; g < kGapClasses; ++g) {
            const char* name = g == 0 ? "b2b" : g == 1 ? "in_token" : "idle";
            if (l.cost_measured[g]) std::snprintf(buf, sizeof buf, " %s=%.1fus", name, l.cost[g].latency_us);
            else std::snprintf(buf, sizeof buf, " %s=-", name);
            out += buf;
        }
        std::snprintf(buf, sizeof buf, " rate=%.2fGB/s src=%s", l.bytes_per_ms / 1e6, rate_source_name(l.src));
        out += buf;
    }
    return out;
}

}  // namespace strix::device
