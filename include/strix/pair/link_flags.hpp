#pragma once
// Pair-route command line shared by every engine's two-node route (Qwen, Gemma 4): --rank 0|1, --peer HOST and
// --transport usb4auto|usb4gpu|tcp are removed from argv and exported as <PREFIX>_RANK, <PREFIX>_PEER and
// <PREFIX>_TRANSPORT, so an engine's option parser never sees them and a re-exec'd child inherits them.
// Node topology (strix-topology-v1, device/topology.hpp): --topology FILE and --node ID (or STRIX_TOPOLOGY /
// STRIX_NODE, or flags the device gate already took) declare the nodes; the rank, the peer host and the transport the
// flags leave open then come from the declaration (resolve_pair_flags). Without a declaration the legacy path below
// is the one that ran before: same exports, same error strings.
#include "strix/device/link_class.hpp"
#include "strix/device/process_topology.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::pair {
inline void set_env(const std::string& key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), value) != 0) throw std::runtime_error("pair prefill environment update failed");
#else
    if (setenv(key.c_str(), value, 1) != 0) throw std::runtime_error("pair prefill environment update failed");
#endif
}

// Raw pair flags ("" = not given on the command line).
struct PairFlags {
    std::string rank, peer, transport;
};

// The transport a declared link class implies for the pair route: "" keeps the family default (usb4 classes: the
// family's usb4auto / usb4gpu choice; unknown class: as legacy), "tcp" for Ethernet classes. A class with no pair
// transport between two nodes (PCIe / OCuLink attach a device to a host) is refused with the reason.
[[nodiscard]] inline std::string pair_transport_for_class(std::string_view link_class, std::string_view link_name) {
    if (link_class.empty() || link_class.rfind("usb4", 0) == 0) return {};
    if (link_class.size() > 3 && link_class.substr(link_class.size() - 3) == "gbe") return "tcp";
    throw std::invalid_argument("pair prefill: link " + std::string(link_name) + " is class " + std::string(link_class) +
                                ", which has no pair transport between nodes: pass --transport");
}

// Fills what the flags leave open from a resolved topology. The pair route is a two-party pipe: a topology of N != 2
// ranks is refused here (the N-stage pipeline is QUEUE-topology item 4). Given flags always win (operator override);
// a given --rank was already checked against the declaration by resolve_topology.
[[nodiscard]] inline PairFlags resolve_pair_flags(PairFlags given, const device::Topology& topology) {
    if (topology.size() != 2)
        throw std::invalid_argument("pair prefill: the pair route has two ranks; topology " + topology.source + " declares " +
                                    std::to_string(topology.size()));
    const auto local = topology.local_rank;
    if (given.rank.empty()) given.rank = std::to_string(local);
    // The peer is the node joined to this one by a declared link (N-form: the one link of a two-rank topology).
    for (const auto& link : topology.links) {
        if (link.a != local && link.b != local) continue;
        const auto& peer = topology.nodes.at(link.a == local ? link.b : link.a);
        if (given.peer.empty()) given.peer = peer.host;
        if (given.transport.empty()) given.transport = pair_transport_for_class(link.link_class, link.name);
        break;
    }
    return given;
}

// Returns the remaining arguments (argv[0] first, nullptr-terminated); argc/argv then point into it.
// require_rank = false: a missing rank is not an error (Qwen's deploy check runs without the pair pipe).
inline std::vector<char*> strip_flags(int& argc, char**& argv, std::string_view prefix, bool require_rank = true) {
    std::vector<char*> kept{argv[0]};
    const std::string base(prefix);
    bool rank_seen = false, peer_seen = false, transport_seen = false;
    PairFlags given;
    device::TopologyCommandLine topology_cli;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--rank" || argument == "--peer" || argument == "--transport" || argument == "--topology" ||
            argument == "--node") {
            if (++index == argc) throw std::invalid_argument("pair prefill pair flag requires a value");
            const char* value = argv[index];
            if (argument == "--rank") {
                if (rank_seen || (std::string_view(value) != "0" && std::string_view(value) != "1"))
                    throw std::invalid_argument("pair prefill requires one --rank 0|1");
                rank_seen = true;
                given.rank = value;
            } else if (argument == "--peer") given.peer = value, peer_seen = true;
            else if (argument == "--transport") given.transport = value, transport_seen = true;
            else if (argument == "--topology") topology_cli.path = value;
            else topology_cli.node = value;
        } else kept.push_back(argv[index]);
    }
    device::keep_topology_command_line(topology_cli);
    if (device::topology_declared() || !device::kept_topology_command_line().node.empty()) {
        const auto& topology = device::process_topology(rank_seen ? std::stoi(given.rank) : -1, given.transport);
        given = resolve_pair_flags(std::move(given), topology);
        rank_seen = true;
    }
    if (!given.rank.empty()) set_env(base + "_RANK", given.rank.c_str());
    if (peer_seen || !given.peer.empty()) set_env(base + "_PEER", given.peer.c_str());
    if (transport_seen || !given.transport.empty()) set_env(base + "_TRANSPORT", given.transport.c_str());
    if (!rank_seen && require_rank) {
        const char* inherited = std::getenv((base + "_RANK").c_str());
        if (!inherited || (std::string_view(inherited) != "0" && std::string_view(inherited) != "1"))
            throw std::invalid_argument("pair prefill requires --rank 0|1");
    }
    argc = static_cast<int>(kept.size());
    kept.push_back(nullptr);
    argv = kept.data();
    return kept;
}
inline unsigned env_rank(std::string_view prefix) {
    const char* value = std::getenv((std::string(prefix) + "_RANK").c_str());
    if (!value || (std::string_view(value) != "0" && std::string_view(value) != "1"))
        throw std::invalid_argument("pair prefill requires --rank 0|1");
    return value[0] == '0' ? 0U : 1U;
}
} // namespace strix::pair
