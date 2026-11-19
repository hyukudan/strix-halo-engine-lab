#pragma once
// The process's node topology, held once like the bound device (hip_gate.hpp): the command line is taken by the
// device gate (or by the first pair-flag strip, whichever runs first), the topology is resolved ONCE with the family's
// legacy rank / transport, and every receipt and --print-effective-config printer reads the same value, so the CLI
// and the server of a family print the same topology for the same argv (one resolver).
// Legacy invariant: a process with no --topology / --node / STRIX_TOPOLOGY / STRIX_NODE resolves exactly the legacy
// shape (legacy_pair_topology for a --rank, single_node_topology otherwise) and nothing else changes.
#include "strix/device/link_class.hpp"
#include "strix/device/topology.hpp"

#include <string>
#include <string_view>

namespace strix::device {

// Keeps the --topology / --node values (a non-empty field replaces the kept one) and exports them as STRIX_TOPOLOGY /
// STRIX_NODE, so a re-exec'd or spawned child of this process resolves the same node.
void keep_topology_command_line(const TopologyCommandLine& cli);
[[nodiscard]] TopologyCommandLine kept_topology_command_line();
// True when a declaration is in force: a kept --topology path or STRIX_TOPOLOGY.
[[nodiscard]] bool topology_declared();

// Resolves (first call) and returns the process topology. A later call that knows more replaces a legacy result
// (a rank for a "single" shape, a transport class for a legacy pair without one); a later call whose rank contradicts
// the resolved local rank throws std::invalid_argument naming both.
[[nodiscard]] const Topology& process_topology(int legacy_rank = -1, std::string_view legacy_transport = {});
// The resolved topology, or nullptr when no caller resolved it yet.
[[nodiscard]] const Topology* resolved_process_topology() noexcept;

// `"topology":{...}` (topology_json of the resolved topology) for a JSON effective config; "" when unresolved.
[[nodiscard]] std::string topology_config_field();
// `json_object` with topology_config_field() added as its last member; unchanged when unresolved or not an object.
// Used at --print-effective-config print sites only: plan / identity JSON exchanged between ranks must not carry it
// (local_rank differs per node).
[[nodiscard]] std::string with_topology_field(std::string json_object);
// `topology: ...` for the gate receipt beside `device:`. Peeks --rank R / --transport T in argv (does not remove them).
// Declared topology: resolved and printed; any refusal throws (the gate exits with the reason). Legacy with a --rank:
// the legacy pair shape, printed; a legacy argv the shape cannot describe prints nothing and never fails (the family
// owns its rank). Legacy without a --rank: "" (the family resolves it when it knows its rank).
[[nodiscard]] std::string gate_topology_line(int argc, char** argv);

// ONE link-rate source for two-party consumers (link_class.hpp rule). local_pair_link: the first declared link touching
// the local rank (nullptr when none). link_rate_mbytes_ps: a declared measurement > the use's calibration for the
// link class > the class nominal; a null link or a link of unknown class without a measurement (no declaration, the
// legacy tcp shape) reads the use's legacy class, i.e. exactly the constant the consumer used before.
[[nodiscard]] const TopologyLink* local_pair_link(const Topology& topology) noexcept;
[[nodiscard]] double link_rate_mbytes_ps(LinkUse use, const TopologyLink* link) noexcept;

// Host tests only: forget the kept command line and the resolved topology (environment untouched).
void reset_process_topology_for_tests();

} // namespace strix::device
