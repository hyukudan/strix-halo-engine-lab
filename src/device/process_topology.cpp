#include "strix/device/process_topology.hpp"

#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace strix::device {
namespace {
struct State {
    std::mutex mutex;
    TopologyCommandLine cli;
    std::optional<Topology> resolved;
    int resolved_rank = -1;  // the legacy rank the topology was resolved with (-1 = none)
};
State& state() {
    static State s;
    return s;
}
void export_env(const char* key, const std::string& value) {
#ifdef _WIN32
    if (_putenv_s(key, value.c_str()) != 0) throw std::runtime_error(std::string("topology: cannot export ") + key);
#else
    if (setenv(key, value.c_str(), 1) != 0) throw std::runtime_error(std::string("topology: cannot export ") + key);
#endif
}
bool env_set(const char* key) {
    const char* v = std::getenv(key);
    return v && *v;
}
// A legacy result that a caller with more facts may replace.
bool upgradable(const Topology& t, int rank, std::string_view transport) {
    if (t.source == "single") return rank >= 0;
    if (t.source == "legacy") return !transport.empty() && !t.links.empty() && t.links.front().link_class.empty();
    return false;
}
} // namespace

void keep_topology_command_line(const TopologyCommandLine& cli) {
    auto& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (!cli.path.empty()) {
        s.cli.path = cli.path;
        export_env("STRIX_TOPOLOGY", cli.path);
    }
    if (!cli.node.empty()) {
        s.cli.node = cli.node;
        export_env("STRIX_NODE", cli.node);
    }
}

TopologyCommandLine kept_topology_command_line() {
    auto& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    return s.cli;
}

bool topology_declared() { return !kept_topology_command_line().path.empty() || env_set("STRIX_TOPOLOGY"); }

const Topology& process_topology(int legacy_rank, std::string_view legacy_transport) {
    auto& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.resolved && !upgradable(*s.resolved, legacy_rank, legacy_transport)) {
        if (legacy_rank >= 0 && static_cast<std::uint32_t>(legacy_rank) != s.resolved->local_rank)
            throw std::invalid_argument("topology: rank " + std::to_string(legacy_rank) + " contradicts the resolved local rank " +
                                        std::to_string(s.resolved->local_rank) + " (topology " + s.resolved->source + ")");
        return *s.resolved;
    }
    const int rank = legacy_rank >= 0 ? legacy_rank : s.resolved_rank;
    s.resolved = start_topology(s.cli, rank, legacy_transport);
    s.resolved_rank = rank;
    return *s.resolved;
}

const Topology* resolved_process_topology() noexcept {
    auto& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    return s.resolved ? &*s.resolved : nullptr;
}

std::string topology_config_field() {
    const auto* t = resolved_process_topology();
    return t ? "\"topology\":" + topology_json(*t) : std::string();
}

std::string with_topology_field(std::string json_object) {
    const auto field = topology_config_field();
    if (field.empty() || json_object.size() < 2 || json_object.front() != '{' || json_object.back() != '}') return json_object;
    json_object.insert(json_object.size() - 1, (json_object.size() > 2 ? "," : "") + field);
    return json_object;
}

std::string gate_topology_line(int argc, char** argv) {
    int rank = -1;
    std::string transport;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string_view a = argv[i];
        const std::string_view v = argv[i + 1];
        if (a == "--rank" && !v.empty() && v.size() <= 3 && v.find_first_not_of("0123456789") == std::string_view::npos) rank = std::stoi(std::string(v));
        if (a == "--transport") transport = std::string(v);
    }
    if (topology_declared()) return topology_line(process_topology(rank, transport));
    if (rank < 0) return {};
    try {
        return topology_line(process_topology(rank, transport));
    } catch (const std::exception&) {
        return {};  // the legacy shape cannot describe this argv; the family's own parser decides
    }
}

const TopologyLink* local_pair_link(const Topology& t) noexcept {
    for (const auto& l : t.links)
        if (l.a == t.local_rank || l.b == t.local_rank) return &l;
    return nullptr;
}

double link_rate_mbytes_ps(LinkUse use, const TopologyLink* link) noexcept {
    const std::string_view legacy = use == LinkUse::host_egress ? legacy_host_link_class : legacy_pair_link_class;
    // host_egress is the prefill host's own port, not a node-to-node link: strix-topology-v1 does not declare it yet.
    if (use == LinkUse::host_egress || !link || (link->link_class.empty() && !link->measured))
        return link_rate_mbytes_ps(use, legacy);
    return link_rate_mbytes_ps(use, link->link_class, link->measured ? link->gbytes_ps : 0.0);
}

void reset_process_topology_for_tests() {
    auto& s = state();
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.cli = {};
    s.resolved.reset();
    s.resolved_rank = -1;
}

} // namespace strix::device
