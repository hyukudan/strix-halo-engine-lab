#include "strix/device/topology.hpp"

#include "strix/device/link_class.hpp"
#include "strix/device/profiles.hpp"
#include "strix/model_description.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strix::device {
namespace {
using Value = model_description::Value;
using Table = Value::Table;

constexpr std::uint32_t kMaxNodes = 64;
constexpr double kGiB = 1073741824.0;

[[noreturn]] void bad(const std::string& key, const std::string& what) { throw std::invalid_argument(key + ": " + what); }

// The returned reference aliases `v` only. `key` is a view (not a reference bound to a temporary std::string) so a
// literal key at the call site creates no temporary that the result could appear to outlive.
const Table& table(const Value& v, std::string_view key) {
    if (const auto* t = std::get_if<Table>(&v.data)) return *t;
    bad(std::string(key), "expected a table");
}
std::string string(const Value& v, const std::string& key) {
    if (const auto* s = std::get_if<std::string>(&v.data)) return *s;
    bad(key, "expected a string");
}
std::uint64_t integer(const Value& v, const std::string& key) {
    if (const auto* n = std::get_if<std::int64_t>(&v.data); n && *n >= 0) return static_cast<std::uint64_t>(*n);
    bad(key, "expected a nonnegative integer");
}
double positive(const Value& v, const std::string& key) {
    double out = -1.0;
    if (const auto* d = std::get_if<double>(&v.data)) out = *d;
    else if (const auto* n = std::get_if<std::int64_t>(&v.data)) out = static_cast<double>(*n);
    if (!std::isfinite(out) || out <= 0.0) bad(key, "expected a positive number");
    return out;
}
std::vector<std::string> strings(const Value& v, const std::string& key) {
    const auto* a = std::get_if<Value::Array>(&v.data);
    if (!a) bad(key, "expected an array of strings");
    std::vector<std::string> out;
    for (const auto& item : *a) out.push_back(string(item, key));
    return out;
}
void only(const Table& t, const std::set<std::string_view>& allowed, const std::string& prefix) {
    for (const auto& entry : t)
        if (!allowed.contains(entry.first)) bad(prefix + entry.first, "unknown key");
}
const Value* find(const Table& t, std::string_view key) {
    const auto it = t.find(key);
    return it == t.end() ? nullptr : &it->second;
}

TopologyNode parse_node(const std::string& id, const Value& v) {
    const std::string key = "nodes." + id;
    if (id.empty()) bad("nodes", "empty node id");
    const auto& t = table(v, key);
    only(t, {"rank", "host", "profile", "memory_gib", "bandwidth_gbytes_ps", "fp16_tflops", "units", "roles"}, key + ".");
    TopologyNode n;
    n.id = id;
    const auto* rank = find(t, "rank");
    if (!rank) bad(key + ".rank", "required");
    const auto r = integer(*rank, key + ".rank");
    if (r >= kMaxNodes) bad(key + ".rank", "at most " + std::to_string(kMaxNodes) + " nodes");
    n.rank = static_cast<std::uint32_t>(r);
    if (const auto* x = find(t, "host")) n.host = string(*x, key + ".host");
    const Profile* profile = nullptr;
    if (const auto* x = find(t, "profile")) {
        n.profile = string(*x, key + ".profile");
        profile = find_profile(n.profile);
        if (!profile) bad(key + ".profile", "unknown profile " + n.profile + "; known: " + profile_names());
        if (profile->memory_bytes) { n.memory_bytes = profile->memory_bytes; n.memory_source = FactSource::profile; }
        if (profile->memory_gbytes_ps > 0) { n.memory_gbytes_ps = profile->memory_gbytes_ps; n.rate_source = FactSource::profile; }
        if (profile->peaks.fp16 > 0) { n.fp16_tflops = profile->peaks.fp16; n.compute_source = FactSource::profile; }
        n.units = profile->units;
        n.vendor = std::string(profile->vendor);
        n.arch = std::string(profile->arch);
    }
    if (const auto* x = find(t, "memory_gib")) {
        const double gib = positive(*x, key + ".memory_gib");
        // Not capped by the profile: a unified-memory profile's size depends on the node's GTT setting. The observation
        // (observe()) refuses a declared size above what the node reports.
        // Range check before the conversion: llround of >= 2^63 is undefined, and a size that rounds to 0 bytes would read as
        // "no limit" downstream.
        const double bytes = gib * kGiB;
        if (!(bytes >= 0.5 && bytes < 9.0e18)) bad(key + ".memory_gib", "out of range (1 byte .. 8 EiB)");
        n.memory_bytes = static_cast<std::uint64_t>(std::llround(bytes));
        n.memory_source = FactSource::declared;
    }
    if (profile && profile->name == "custom" && !find(t, "memory_gib")) bad(key + ".memory_gib", "required with profile custom");
    if (const auto* x = find(t, "bandwidth_gbytes_ps")) {
        n.memory_gbytes_ps = positive(*x, key + ".bandwidth_gbytes_ps");
        n.rate_source = FactSource::declared;
    }
    if (const auto* x = find(t, "fp16_tflops")) {
        n.fp16_tflops = positive(*x, key + ".fp16_tflops");
        n.compute_source = FactSource::declared;
    }
    if (const auto* x = find(t, "units")) {
        const auto u = integer(*x, key + ".units");
        if (u == 0 || u > 100000) bad(key + ".units", "expected 1..100000");
        n.units = static_cast<std::uint32_t>(u);
    }
    if (const auto* x = find(t, "roles")) {
        n.roles = strings(*x, key + ".roles");
        std::set<std::string> seen;
        for (const auto& role : n.roles) {
            if (!known_role(role) || !role_allowed(Kind::gpu, role)) bad(key + ".roles", "unknown GPU role " + role);
            if (!seen.insert(role).second) bad(key + ".roles", "repeated role " + role);
        }
    }
    return n;
}

TopologyLink parse_link(const std::string& name, const Value& v, const std::map<std::string, std::uint32_t>& ranks) {
    const std::string key = "links." + name;
    const auto& t = table(v, key);
    only(t, {"between", "class", "gbps_measured", "rtt_us_measured", "port"}, key + ".");
    TopologyLink l;
    l.name = name;
    const auto* between = find(t, "between");
    if (!between) bad(key + ".between", "required (two node ids)");
    const auto ends = strings(*between, key + ".between");
    if (ends.size() != 2) bad(key + ".between", "expected two node ids");
    const auto ia = ranks.find(ends[0]), ib = ranks.find(ends[1]);
    if (ia == ranks.end()) bad(key + ".between", "unknown node " + ends[0]);
    if (ib == ranks.end()) bad(key + ".between", "unknown node " + ends[1]);
    if (ia->second == ib->second) bad(key + ".between", "a link joins two different nodes");
    l.a = std::min(ia->second, ib->second);
    l.b = std::max(ia->second, ib->second);
    const auto* cls = find(t, "class");
    if (!cls) bad(key + ".class", "required");
    l.link_class = string(*cls, key + ".class");
    const auto* c = find_link_class(l.link_class);
    if (!c) bad(key + ".class", "unknown link class " + l.link_class);
    l.gbytes_ps = c->gbytes_ps;
    l.rtt_us = c->rtt_us;
    l.uncertainty = c->uncertainty;
    if (const auto* x = find(t, "gbps_measured")) {
        l.gbytes_ps = positive(*x, key + ".gbps_measured") / 8.0;
        l.measured = true;
        l.uncertainty = 0.10;
    }
    if (const auto* x = find(t, "rtt_us_measured")) {
        l.rtt_us = positive(*x, key + ".rtt_us_measured");
        l.rtt_measured = true;
    }
    l.port = "link-" + name;
    if (const auto* x = find(t, "port")) l.port = string(*x, key + ".port");
    return l;
}

void check_connected(const Topology& t) {
    if (t.size() <= 1) return;
    std::vector<bool> seen(t.size(), false);
    std::vector<std::uint32_t> stack{0};
    seen[0] = true;
    while (!stack.empty()) {
        const auto r = stack.back();
        stack.pop_back();
        for (const auto& l : t.links) {
            const std::uint32_t other = l.a == r ? l.b : (l.b == r ? l.a : r);
            if (other != r && !seen[other]) { seen[other] = true; stack.push_back(other); }
        }
    }
    for (const auto& n : t.nodes)
        if (!seen[n.rank]) bad("nodes." + n.id, "no link reaches this node from rank 0");
}

std::string transport_class(std::string_view transport) {
    if (transport.rfind("usb4", 0) == 0) return "usb4"; // usb4auto, usb4-auto, usb4gpu, usb4-gpu, usb4:dev,...
    return {};
}

std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}
std::string gib_text(std::uint64_t bytes) { return bytes ? num(static_cast<double>(bytes) / kGiB) + "GiB" : "?GiB"; }
std::string quoted(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) { out += ' '; continue; }
        out += c;
    }
    return out + "\"";
}

// Canonical JSON body shared by topology_json and topology_id (no source, no local rank).
std::string body_json(const Topology& t) {
    std::ostringstream os;
    os << "\"name\":" << quoted(t.name) << ",\"ranks\":" << t.size() << ",\"uniform\":" << (t.uniform() ? "true" : "false")
       << ",\"nodes\":[";
    for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        const auto& n = t.nodes[i];
        os << (i ? "," : "") << "{\"id\":" << quoted(n.id) << ",\"rank\":" << n.rank << ",\"host\":" << quoted(n.host)
           << ",\"profile\":" << quoted(n.profile) << ",\"vendor\":" << quoted(n.vendor) << ",\"arch\":" << quoted(n.arch)
           << ",\"memory_bytes\":" << n.memory_bytes << ",\"memory_source\":" << quoted(fact_source_name(n.memory_source))
           << ",\"memory_gbytes_ps\":" << num(n.memory_gbytes_ps) << ",\"rate_source\":" << quoted(fact_source_name(n.rate_source))
           << ",\"fp16_tflops\":" << num(n.fp16_tflops) << ",\"units\":" << n.units
           << ",\"compute_source\":" << quoted(fact_source_name(n.compute_source)) << ",\"roles\":[";
        for (std::size_t r = 0; r < n.roles.size(); ++r) os << (r ? "," : "") << quoted(n.roles[r]);
        os << "]}";
    }
    os << "],\"links\":[";
    for (std::size_t i = 0; i < t.links.size(); ++i) {
        const auto& l = t.links[i];
        os << (i ? "," : "") << "{\"name\":" << quoted(l.name) << ",\"a\":" << l.a << ",\"b\":" << l.b
           << ",\"class\":" << quoted(l.link_class) << ",\"gbytes_ps\":" << num(l.gbytes_ps) << ",\"rtt_us\":" << num(l.rtt_us)
           << ",\"uncertainty\":" << num(l.uncertainty) << ",\"rate\":" << quoted(l.measured ? "measured" : (l.gbytes_ps > 0 ? "nominal" : "unknown"))
           << ",\"port\":" << quoted(l.port) << "}";
    }
    os << "]";
    return os.str();
}

std::string fnv1a64(std::string_view s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string env(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value) return {};
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
#endif
}

} // namespace

std::string_view fact_source_name(FactSource source) noexcept {
    switch (source) {
    case FactSource::declared: return "declared";
    case FactSource::profile: return "profile";
    case FactSource::observed: return "observed";
    case FactSource::legacy: return "legacy";
    case FactSource::unknown: break;
    }
    return "unknown";
}

std::vector<const TopologyLink*> Topology::links_between(std::uint32_t a, std::uint32_t b) const {
    std::vector<const TopologyLink*> out;
    const auto lo = std::min(a, b), hi = std::max(a, b);
    for (const auto& l : links)
        if (l.a == lo && l.b == hi) out.push_back(&l);
    return out;
}

double Topology::direct_gbytes_ps(std::uint32_t a, std::uint32_t b) const {
    double sum = 0;
    for (const auto* l : links_between(a, b)) {
        if (l->gbytes_ps <= 0) return 0; // one unknown link makes the total unknown
        sum += l->gbytes_ps;
    }
    return sum;
}

bool Topology::uniform() const noexcept {
    for (const auto& n : nodes)
        if (n.memory_bytes != nodes.front().memory_bytes || n.memory_gbytes_ps != nodes.front().memory_gbytes_ps ||
            n.fp16_tflops != nodes.front().fp16_tflops || n.units != nodes.front().units)
            return false;
    return true;
}

Topology parse_topology(std::string_view toml, std::string source) {
    const auto doc = model_description::parse(toml);
    const auto& root = doc.fields;
    only(root, {"schema", "name", "nodes", "links"}, "");
    const auto* schema = find(root, "schema");
    if (!schema) bad("schema", "required (strix-topology-v1)");
    if (string(*schema, "schema") != "strix-topology-v1") bad("schema", "expected strix-topology-v1");
    Topology t;
    t.source = std::move(source);
    if (const auto* x = find(root, "name")) t.name = string(*x, "name");
    const auto* nodes = find(root, "nodes");
    if (!nodes) bad("nodes", "required (at least one [nodes.<id>] table)");
    std::vector<TopologyNode> parsed;
    for (const auto& [id, value] : table(*nodes, "nodes")) parsed.push_back(parse_node(id, value));
    if (parsed.empty()) bad("nodes", "required (at least one [nodes.<id>] table)");
    if (parsed.size() > kMaxNodes) bad("nodes", "at most " + std::to_string(kMaxNodes) + " nodes");
    t.nodes.resize(parsed.size());
    std::vector<bool> filled(parsed.size(), false);
    std::map<std::string, std::uint32_t> ranks;
    for (auto& n : parsed) {
        if (n.rank >= parsed.size())
            bad("nodes." + n.id + ".rank", "ranks are 0.." + std::to_string(parsed.size() - 1) + " for " + std::to_string(parsed.size()) + " nodes");
        if (filled[n.rank]) bad("nodes." + n.id + ".rank", "rank " + std::to_string(n.rank) + " is declared twice");
        filled[n.rank] = true;
        ranks[n.id] = n.rank;
        t.nodes[n.rank] = std::move(n);
    }
    if (const auto* links = find(root, "links"))
        for (const auto& [name, value] : table(*links, "links")) t.links.push_back(parse_link(name, value, ranks));
    for (std::size_t i = 0; i < t.links.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (t.links[i].a == t.links[j].a && t.links[i].b == t.links[j].b && t.links[i].port == t.links[j].port)
                bad("links." + t.links[i].name + ".port", "parallel link to links." + t.links[j].name + " on the same port " + t.links[i].port);
    check_connected(t);
    return t;
}

Topology load_topology(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("topology: cannot read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    try {
        return parse_topology(text.str(), path);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("topology " + path + ": " + e.what());
    }
}

Topology legacy_pair_topology(std::uint32_t rank, std::string_view transport) {
    if (rank > 1) throw std::invalid_argument("--rank " + std::to_string(rank) + ": a pair route without --topology has ranks 0 and 1");
    Topology t;
    t.source = "legacy";
    t.name = "pair";
    for (std::uint32_t r = 0; r < 2; ++r) {
        TopologyNode n;
        n.id = "rank" + std::to_string(r);
        n.rank = r;
        n.memory_source = n.rate_source = n.compute_source = FactSource::legacy;
        t.nodes.push_back(std::move(n));
    }
    TopologyLink l;
    l.name = "pair";
    l.a = 0;
    l.b = 1;
    l.link_class = transport_class(transport);
    if (const auto* c = find_link_class(l.link_class)) {
        l.gbytes_ps = c->gbytes_ps;
        l.rtt_us = c->rtt_us;
        l.uncertainty = c->uncertainty;
    }
    l.port = "link-pair";
    t.links.push_back(std::move(l));
    t.local_rank = rank;
    t.local_known = true;
    return t;
}

Topology single_node_topology() {
    Topology t;
    t.source = "single";
    TopologyNode n;
    n.id = "local";
    t.nodes.push_back(std::move(n));
    t.local_known = true;
    return t;
}

Topology resolve_topology(const Topology* declared, const LocalIdentity& identity) {
    if (!declared) {
        if (!identity.node.empty())
            throw std::invalid_argument("--node " + identity.node + ": no topology declared (--topology FILE or STRIX_TOPOLOGY)");
        if (identity.rank >= 0) return legacy_pair_topology(static_cast<std::uint32_t>(identity.rank), identity.transport);
        return single_node_topology();
    }
    Topology t = *declared;
    const auto where = " (topology " + t.source + ")";
    if (!identity.node.empty()) {
        const auto it = std::find_if(t.nodes.begin(), t.nodes.end(), [&](const TopologyNode& n) { return n.id == identity.node; });
        if (it == t.nodes.end()) throw std::invalid_argument("--node " + identity.node + ": not a declared node" + where);
        if (identity.rank >= 0 && static_cast<std::uint32_t>(identity.rank) != it->rank)
            throw std::invalid_argument("--node " + identity.node + " is rank " + std::to_string(it->rank) + " but --rank " +
                                        std::to_string(identity.rank) + " was given" + where);
        t.local_rank = it->rank;
    } else if (identity.rank >= 0) {
        if (static_cast<std::uint32_t>(identity.rank) >= t.size())
            throw std::invalid_argument("--rank " + std::to_string(identity.rank) + ": the topology declares ranks 0.." +
                                        std::to_string(t.size() - 1) + where);
        t.local_rank = static_cast<std::uint32_t>(identity.rank);
    } else if (!identity.hostname.empty() &&
               std::any_of(t.nodes.begin(), t.nodes.end(), [&](const TopologyNode& n) { return n.host == identity.hostname; })) {
        int matches = 0;
        for (const auto& n : t.nodes)
            if (n.host == identity.hostname) { t.local_rank = n.rank; ++matches; }
        if (matches > 1)
            throw std::invalid_argument("host " + identity.hostname + " is declared by " + std::to_string(matches) +
                                        " nodes: pass --node ID" + where);
    } else if (t.size() == 1) {
        t.local_rank = 0;
    } else {
        throw std::invalid_argument("cannot tell which node this process is: pass --node ID (or STRIX_NODE), --rank R, or declare "
                                    "`host` for this machine" + where);
    }
    t.local_known = true;
    return t;
}

void observe(Topology& t, std::uint32_t rank, const ObservedNode& o) {
    if (rank >= t.size()) throw std::invalid_argument("observe: rank " + std::to_string(rank) + " is not declared");
    auto& n = t.nodes[rank];
    const std::string key = "nodes." + n.id;
    if (!o.arch.empty()) {
        if (!n.arch.empty() && n.arch != o.arch)
            throw std::invalid_argument(key + ": declared arch " + n.arch + " but this node reports " + o.arch);
        n.arch = o.arch;
    }
    if (!o.vendor.empty() && n.vendor.empty()) n.vendor = o.vendor;
    if (o.memory_total_bytes) {
        if (n.memory_source == FactSource::declared && n.memory_bytes > o.memory_total_bytes)
            throw std::invalid_argument(key + ".memory_gib: declares " + gib_text(n.memory_bytes) + " but the node reports " +
                                        gib_text(o.memory_total_bytes) + " usable");
        if (n.memory_source != FactSource::declared) {
            n.memory_bytes = o.memory_total_bytes;
            n.memory_source = FactSource::observed;
        }
    }
    if (o.units && n.compute_source != FactSource::declared && n.units == 0) n.units = o.units;
}

TopologyEnvironment read_topology_environment() { return {env("STRIX_TOPOLOGY"), env("STRIX_NODE")}; }

std::string local_hostname() {
#ifdef _WIN32
    char buf[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD size = sizeof buf;
    if (!GetComputerNameA(buf, &size)) return {};
    return std::string(buf, size);
#else
    char buf[256] = {};
    if (gethostname(buf, sizeof buf - 1) != 0) return {};
    return buf;
#endif
}

TopologyCommandLine take_topology_command_line(int& argc, char** argv) {
    TopologyCommandLine out;
    int w = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        std::string* target = a == "--topology" ? &out.path : (a == "--node" ? &out.node : nullptr);
        if (!target) {
            argv[w++] = argv[i];
            continue;
        }
        if (i + 1 >= argc || argv[i + 1][0] == '\0') throw std::invalid_argument(std::string(a) + ": expected a value");
        *target = argv[++i];
    }
    for (int i = w; i < argc; ++i) argv[i] = nullptr;
    argc = w;
    return out;
}

Topology start_topology(const TopologyCommandLine& cli, int legacy_rank, std::string_view legacy_transport) {
    const auto environment = read_topology_environment();
    const std::string path = !cli.path.empty() ? cli.path : environment.path;
    LocalIdentity identity;
    identity.node = !cli.node.empty() ? cli.node : environment.node;
    identity.rank = legacy_rank;
    identity.transport = std::string(legacy_transport);
    identity.hostname = local_hostname();
    if (path.empty()) return resolve_topology(nullptr, identity);
    const auto declared = load_topology(path);
    return resolve_topology(&declared, identity);
}

std::string topology_line(const Topology& t) {
    std::ostringstream os;
    os << "topology: source=" << t.source;
    if (!t.name.empty()) os << " name=" << t.name;
    os << " ranks=" << t.size() << " local=";
    if (t.local_known) os << t.local_rank << "(" << t.local().id << ")";
    else os << "?";
    os << " uniform=" << (t.uniform() ? "yes" : "no") << " nodes=";
    for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        const auto& n = t.nodes[i];
        os << (i ? "," : "") << n.id << "[r" << n.rank;
        if (!n.host.empty()) os << " host=" << n.host;
        if (!n.profile.empty()) os << " " << n.profile;
        if (!n.arch.empty()) os << " " << n.arch;
        os << " " << gib_text(n.memory_bytes) << "(" << fact_source_name(n.memory_source) << ") "
           << (n.memory_gbytes_ps > 0 ? num(n.memory_gbytes_ps) : "?") << "GB/s(" << fact_source_name(n.rate_source) << ") "
           << (n.fp16_tflops > 0 ? num(n.fp16_tflops) : "?") << "TF(" << fact_source_name(n.compute_source) << ")]";
    }
    os << " links=";
    if (t.links.empty()) os << "-";
    for (std::size_t i = 0; i < t.links.size(); ++i) {
        const auto& l = t.links[i];
        os << (i ? "," : "") << l.name << "[" << t.nodes[l.a].id << "-" << t.nodes[l.b].id << " "
           << (l.link_class.empty() ? "?" : l.link_class) << " " << (l.gbytes_ps > 0 ? num(l.gbytes_ps) : "?") << "GB/s "
           << (l.measured ? "measured" : (l.gbytes_ps > 0 ? "nominal" : "unknown")) << " rtt=" << num(l.rtt_us) << "us port=" << l.port
           << "]";
    }
    os << " id=" << topology_id(t);
    return os.str();
}

std::string topology_json(const Topology& t) {
    std::ostringstream os;
    os << "{\"schema\":\"strix-topology-v1\",\"source\":" << quoted(t.source) << ",\"local_rank\":";
    if (t.local_known) os << t.local_rank;
    else os << "null";
    os << "," << body_json(t) << ",\"topology_id\":\"" << topology_id(t) << "\"}";
    return os.str();
}

std::string topology_id(const Topology& t) { return fnv1a64("{" + body_json(t) + "}"); }

Declaration to_declaration(const Topology& t) {
    Declaration d;
    d.source = t.source;
    d.machine = t.name;
    for (const auto& n : t.nodes) {
        Descriptor g;
        g.id = n.id;
        g.kind = Kind::gpu;
        if (!n.vendor.empty()) g.vendor = parse_vendor(n.vendor);
        g.arch = n.arch;
        g.memory_bytes = n.memory_bytes;
        // attach left empty: a cluster has no declaring machine, every node is reached only through [links].
        g.roles = n.roles.empty() ? std::vector<std::string>{"pair", "decode", "prefill"} : n.roles;
        g.profile = n.profile;
        g.budget_bytes = n.memory_source == FactSource::declared ? n.memory_bytes : 0;
        g.bandwidth_gbytes_ps = n.rate_source == FactSource::declared ? n.memory_gbytes_ps : 0;
        g.units = n.units;
        g.planned = true; // no identity across machines: planners only, the gate never binds it
        d.devices.push_back(std::move(g));
    }
    for (const auto& l : t.links) {
        Link k;
        k.name = l.name;
        k.a = t.nodes[l.a].id;
        k.b = t.nodes[l.b].id;
        k.link = l.link_class;
        k.gbps_measured = l.measured ? l.gbytes_ps * 8.0 : 0.0;
        k.rtt_us_measured = l.rtt_measured ? static_cast<std::uint64_t>(std::llround(l.rtt_us)) : 0;
        k.port = l.port;
        d.links.push_back(std::move(k));
    }
    return d;
}

} // namespace strix::device
