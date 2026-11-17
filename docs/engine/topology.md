# Node topology declaration

*[Español](topology.es.md)*

A multi-node run is described the same way a device is: declared in a file or discovered, resolved by one shared
function, printed in the effective configuration, and never assumed by a model. The device declaration
(`strix-devices-v1`, [deployment](../deployment.md)) says which GPUs a machine may use; the topology
(`strix-topology-v1`, `include/strix/device/topology.hpp`) says which nodes take part in one run, how much each can
hold and how fast it is, and what joins them. One single node and today's two-node pair are just two instances.

## The file

One file describes the whole cluster and is identical on every node. One node entry is one rank, that is one GPU
process; a machine with two GPUs is two entries with the same `host`, joined by a PCIe link.

```toml
schema = "strix-topology-v1"
name = "two-node-usb4"                 # optional, display only

[nodes.a]                              # the table key is the node id
rank = 0                               # ranks are 0..N-1, unique, contiguous
host = "node-a"                        # optional: matched against this machine's host name
profile = "strix-halo-128"             # a device model of the catalogue (strix-device-plan --list), or "custom"
memory_gib = 124                       # optional usable device memory for placement
bandwidth_gbytes_ps = 243              # optional measured memory rate
fp16_tflops = 59.4                     # optional dense FP16 rate
units = 40                             # optional CU / SM count
roles = ["prefill", "decode"]          # optional; absent = every multi-node role

[nodes.b]
rank = 1
profile = "strix-halo-128"

[links.ab]
between = ["a", "b"]
class = "usb4"                         # same classes as the device declaration
gbps_measured = 36.6                   # optional measured useful Gbit/s; replaces the nominal class rate
rtt_us_measured = 30                   # optional
port = "usb4-0"                        # optional shared resource; two links between the same nodes need two ports
```

A node may differ from the others in every fact: memory, memory rate, compute and links. Each value is printed with
where it came from: `declared`, `profile`, `observed` (discovery), `legacy` or `unknown`. Unknown is not a guess:
a planner that needs it refuses with the reason.

Refused with the key named: an unknown key, profile, class or role; a rank missing, repeated or out of range; a
`custom` profile without `memory_gib`; a link with one end, to an unknown node or to itself; two links between the
same nodes on the same port; a node no link reaches.

## Which node is this process

`--node ID` > `STRIX_NODE` > the legacy `--rank R` (it must name a declared rank) > the node whose `host` is this
machine's host name > the only node. Anything else refuses the start: "cannot tell which node this process is".
`--topology FILE` > `STRIX_TOPOLOGY` selects the file; `take_topology_command_line` removes both new flags from the
argument vector so any binary accepts them, like `--devices`.

## Without a file (backward compatibility)

- `--rank R` with a pair transport (`usb4auto`, `usb4-auto`, `usb4gpu`, `usb4-gpu`, `usb4:...`, `tcp-*`) resolves to
  the shape every pair route assumes today: two nodes, rank R local, one link of class `usb4` (or unknown rate for
  tcp), `source=legacy`. Nothing about placement changes: routes keep their own split until they read the topology.
- No rank and no file is one node (`source=single`).

## Discovery

`observe(topology, rank, ObservedNode)` takes what a node measured (arch, usable memory, units), from the local probe
or a peer's handshake. It fills facts the file left unknown and refuses a declared arch that differs or a declared
memory above what the node reports.

## Printing and consumers

`topology_line()` prints one line beside the `device:` receipt:

```text
topology: source=two.toml name=two-node-usb4 ranks=2 local=1(b) uniform=yes nodes=a[r0 host=node-a strix-halo-128
gfx1151 124GiB(declared) 256GB/s(profile) 59.4TF(profile)],b[...] links=ab[a-b usb4 4.58GB/s nominal rtt=30us
port=link-ab] id=...
```

`topology_json()` is the `strix-topology-v1` object for effective-config JSON; `topology_id` hashes the cluster
without the local rank, so every node of one run prints the same id. `to_declaration()` gives the same nodes and
links as a device declaration, so the shared link graph and route planner (`include/strix/deploy/link.hpp`,
`route.hpp`) read the cluster with the one link model of `include/strix/device/link_class.hpp`.

The topology carries no model knowledge and makes no placement decision. The shared planner turns
(topology, model cost table, tier) into a plan. Family routes that still assume two equal ranks keep doing so until
each one reads the topology; today's two-node result is the invariant of every such change.

Tests: `tests/device_topology_tests.cpp` (1, 2 equal, 2 unequal, 3 and 4 nodes, invalid files, legacy shape,
identity, discovery, the link-graph bridge), target `strix-device-topology-tests`.

## Bound today

- Device gate: every HIP binary's `bind_hip_main` takes `--topology FILE` / `--node ID` and prints `topology:` beside
  `device:` when a file is declared (or a legacy `--rank R` is on the command line). A declared file that cannot place
  this process stops the start with the reason.
- Pair routes (Qwen, Gemma 4; `include/strix/pair/link_flags.hpp`): with a file, the rank, the peer host and the
  transport that the flags leave open come from it (Ethernet classes -> `tcp`, `usb4` classes -> the family default,
  PCIe/OCuLink refused); explicit `--rank/--peer/--transport` still win. The pair route has two ranks: a file with
  another node count is refused with the reason. Without a file the legacy flags behave exactly as before.
- Qwen `--print-effective-config` carries the `"topology"` member when the process resolved one.
- One link-rate source: `link_class.hpp` is the only class table (`interconnects()` is a view of it). Decisions tuned
  on a protocol-level rate keep it as a named calibration row there (relay host path 4,400 MB/s; layer-cut handoff
  118 / 290 / 1,180 / 4,600 MB/s; host egress 125 MB/s). Rule: declared measurement > calibration of the class > class
  nominal. No file, or the lab's two-node `usb4` file without a measurement, gives exactly the former numbers.
- One placement planner for layer chains (`include/strix/pair/chain.hpp`), bound to the declaration by
  `include/strix/pair/chain_topology.hpp`: ranks in rank order on a line, memory per node, prefill rate scaled by each
  node's `fp16_tflops` over the fastest one, line links at the declared rate. With every node's memory declared, a
  layer-owned pipeline (stages keep their layers' state for decode) only accepts plans whose stages fit, so a smaller
  node gets fewer layers instead of a failed start. Bound by the Qwen dense pair route (any number of ranks).
- Stage balance over N ranks (`include/strix/pair/balance.hpp` `balance_chain`, one cost vector per rank); the
  two-rank `balance_split` is its N = 2 form with the same cut bit for bit. The Qwen MoE pair cut (measured cost
  file) scales each rank's costs by the declared speed (`stage_cost_scale`); equal or undeclared nodes scale by 1.0.
- Host collectives over N ranks (`include/strix/dist/mesh.hpp`: ordered all-gather / all-reduce over pairwise
  exchanges) and the deploy facts exchange over N (`exchange_facts`; `exchange_pair_facts` is its N = 2 form, same
  wire record).

Tests: `tests/device_topology_bindings_tests.cpp`, target `strix-device-topology-bindings-tests`;
`tests/pair_topology_plan_tests.cpp`, target `strix-pair-topology-plan-tests` (1-4 ranks, unequal memory, a slow rank,
a slow link, the lab plan unchanged).
