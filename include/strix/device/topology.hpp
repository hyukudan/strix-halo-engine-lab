#pragma once

#include "strix/device/declaration.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// strix-topology-v1: the node topology every multi-node binary reads, the multi-node counterpart of the device
// declaration (strix/device/declaration.hpp). One file describes the whole cluster and is identical on every node.
//
//   schema = "strix-topology-v1"
//   name = "two-node-usb4"                       # optional, display only
//
//   [nodes.a]                                    # the table key is the node id
//   rank = 0                                     # required: ranks are 0..N-1, unique, contiguous
//   host = "node-a"                              # optional: host name or address; matched against this machine's
//                                                # host name when neither --node nor --rank says who we are
//   profile = "strix-halo-128"                   # known device model (profiles.hpp), or "custom" with memory_gib
//   memory_gib = 124                             # optional usable device memory for placement (over the profile)
//   bandwidth_gbytes_ps = 243                    # optional measured memory rate (over the profile)
//   fp16_tflops = 59.4                           # optional dense FP16 rate (over the profile)
//   units = 40                                   # optional CU / SM count (over the profile)
//   roles = ["prefill", "decode"]                # optional; absent = every multi-node role
//
//   [nodes.b]
//   rank = 1
//   profile = "strix-halo-128"
//
//   [links.ab]                                   # a link between two nodes
//   between = ["a", "b"]
//   class = "usb4"                               # device/link_class.hpp classes; measured keys override the class
//   gbps_measured = 36.6                         # optional useful payload Gbit/s
//   rtt_us_measured = 30                         # optional
//   port = "usb4-0"                              # optional shared resource; parallel links need distinct ports
//
// One node entry is one rank: one GPU process. A machine with two GPUs is two entries with the same `host` joined by
// a pcie link. Unknown keys, unknown profiles or classes, a gap or repeat in the ranks, a link to an unknown node or
// to itself, two links between the same nodes on one port, and a node no link reaches are errors naming the key.
//
// Which node is this process: --node ID > STRIX_NODE > the legacy --rank R (must name a declared rank) > the node
// whose `host` equals this machine's host name > the only node. Otherwise the start is refused.
//
// Without a file: a legacy rank flag (`--rank R` with a pair transport) resolves to the two-node shape every pair
// route assumes today (source "legacy"), and no rank at all resolves to one node. Nothing here changes placement:
// routes read the topology; the planner (not this file) turns it into a plan.
namespace strix::device {

// Where a value came from, printed beside it.
enum class FactSource { declared, profile, observed, legacy, unknown };
[[nodiscard]] std::string_view fact_source_name(FactSource source) noexcept;

struct TopologyNode {
    std::string id;
    std::uint32_t rank = 0;
    std::string host;                    // "" = not declared
    std::string profile;                 // "" = none (legacy / observed only)
    std::string vendor, arch;            // from the profile or the observation; "" = unknown
    std::uint64_t memory_bytes = 0;      // usable device memory for placement; 0 = unknown
    double memory_gbytes_ps = 0;         // memory rate; 0 = unknown
    double fp16_tflops = 0;              // dense FP16 compute rate; 0 = unknown
    std::uint32_t units = 0;             // CU / SM count; 0 = unknown
    std::vector<std::string> roles;      // empty = every multi-node role
    FactSource memory_source = FactSource::unknown;
    FactSource rate_source = FactSource::unknown;   // memory_gbytes_ps
    FactSource compute_source = FactSource::unknown; // fp16_tflops / units
};

struct TopologyLink {
    std::string name;
    std::uint32_t a = 0, b = 0;          // ranks, a < b
    std::string link_class;              // "" = unknown (legacy tcp)
    double gbytes_ps = 0;                // useful payload GB/s, one direction; 0 = unknown
    double rtt_us = 0;
    double uncertainty = 0;              // relative half-width of gbytes_ps
    std::string port;                    // shared resource; default "link-<name>"
    bool measured = false;               // gbps_measured declared (else nominal from the class)
    bool rtt_measured = false;           // rtt_us_measured declared
};

struct Topology {
    std::string source;                  // path, "<inline>", "legacy" or "single"
    std::string name;
    std::vector<TopologyNode> nodes;     // index == rank
    std::vector<TopologyLink> links;     // declaration order of [links] (name order)
    std::uint32_t local_rank = 0;
    bool local_known = false;            // set by resolve_topology / with_local

    [[nodiscard]] std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(nodes.size()); }
    [[nodiscard]] const TopologyNode& local() const { return nodes.at(local_rank); }
    // Links joining two ranks (both directions), in declaration order; empty = no direct link.
    [[nodiscard]] std::vector<const TopologyLink*> links_between(std::uint32_t a, std::uint32_t b) const;
    // Sum of the direct links' rates (parallel links on distinct ports add); 0 = no direct link or unknown rate.
    [[nodiscard]] double direct_gbytes_ps(std::uint32_t a, std::uint32_t b) const;
    // True when every node states the same memory, memory rate and compute (the shape every pair route assumes).
    [[nodiscard]] bool uniform() const noexcept;
};

// Throws std::invalid_argument naming the offending key. The result has local_known = false.
[[nodiscard]] Topology parse_topology(std::string_view toml, std::string source = "<inline>");
// Additionally throws std::runtime_error when the file cannot be read.
[[nodiscard]] Topology load_topology(const std::string& path);

// The shape a pair route assumes today when started with `--rank R` and no topology file: two nodes with no declared
// facts, rank R local, one link whose class comes from the transport spelling (usb4auto / usb4-auto / usb4gpu /
// usb4-gpu / usb4[:...] -> usb4; tcp-* and anything else -> unknown rate). Throws std::invalid_argument for R > 1.
[[nodiscard]] Topology legacy_pair_topology(std::uint32_t rank, std::string_view transport);
// One node, rank 0, nothing declared.
[[nodiscard]] Topology single_node_topology();

// Who this process is, from the command line and the environment (resolve_topology applies the precedence above).
struct LocalIdentity {
    std::string node;                    // --node ID / STRIX_NODE; "" = not given
    int rank = -1;                       // legacy --rank R; -1 = not given
    std::string transport;               // legacy transport spelling, for the legacy shape; "" = not given
    std::string hostname;                // this machine's host name; "" = unknown
};

// Pure decision. `declared` null = no topology file: legacy pair shape when identity.rank >= 0, else one node.
// Throws std::invalid_argument (a --node or --rank that the file does not declare, two nodes matching the host name,
// no way to tell which node this is).
[[nodiscard]] Topology resolve_topology(const Topology* declared, const LocalIdentity& identity);

// Discovery hook: what this process (or a peer's handshake) observed on node `rank`. Fills the facts the file left
// unknown (source "observed"); refuses a declared arch that differs, or a declared memory above the observed total.
struct ObservedNode {
    std::string vendor, arch;
    std::uint64_t memory_total_bytes = 0; // device-usable bytes; 0 = not observed
    std::uint32_t units = 0;
};
void observe(Topology& topology, std::uint32_t rank, const ObservedNode& observed); // throws std::invalid_argument

// Environment and command line, mirroring the device gate.
struct TopologyEnvironment {
    std::string path;                    // STRIX_TOPOLOGY
    std::string node;                    // STRIX_NODE
};
[[nodiscard]] TopologyEnvironment read_topology_environment();
[[nodiscard]] std::string local_hostname(); // "" when unknown
// Removes `--topology FILE` and `--node ID` from an argument vector (in place) and returns them. Throws
// std::invalid_argument for a flag without a value. Does not touch --rank (the routes keep parsing it).
struct TopologyCommandLine {
    std::string path;
    std::string node;
};
[[nodiscard]] TopologyCommandLine take_topology_command_line(int& argc, char** argv);
// Loads (explicit path > STRIX_TOPOLOGY > none) and resolves with this machine's host name. Throws
// std::invalid_argument / std::runtime_error with the one line to show the operator.
[[nodiscard]] Topology start_topology(const TopologyCommandLine& cli, int legacy_rank = -1,
                                      std::string_view legacy_transport = {});

// "topology: source=FILE name=x ranks=2 local=0(a) uniform=yes nodes=a[r0 strix-halo-128 124GiB(declared)
//  243GB/s(declared) 59.4TF(profile)],b[...] links=ab[a-b usb4 4.58GB/s nominal rtt=30us port=link-ab]"
[[nodiscard]] std::string topology_line(const Topology& topology);
// {"schema":"strix-topology-v1","source":...,"ranks":N,"local_rank":R,"nodes":[...],"links":[...],"topology_id":"..."}
[[nodiscard]] std::string topology_json(const Topology& topology);
// FNV-1a-64 of the canonical JSON without source/local fields: two nodes of one cluster print the same id.
[[nodiscard]] std::string topology_id(const Topology& topology);

// The same nodes and links as a device declaration (one remote GPU entry per node, id = node id, profile and
// overrides carried, links as [links]), so the shared link graph and route planner (deploy/link.hpp, route.hpp)
// read the cluster without a second link model.
[[nodiscard]] Declaration to_declaration(const Topology& topology);

} // namespace strix::device
