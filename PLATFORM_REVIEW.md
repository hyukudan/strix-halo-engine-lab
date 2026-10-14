# Platform qualification plan

Detailed external platform comparisons are private. Native qualification must establish
actual UMA residency, page faults, mapping limits, compiler/library support, state
traffic and completed-operation transport costs on the selected hosts.

Measure independent controls and preserve negative outcomes. Do not promote an external
constant, mechanism or throughput figure into a default without project-owned evidence.
The authoritative requirements are in the [architecture guide](docs/README.md)
and [qualification ledger](docs/qualification-ledger.md). Start practical
placement tests in the [UMA residency lab](lab/uma_residency/README.md), paired
communication controls in the [transport lab](lab/transport/README.md), and
direct GPU-owned transport qualification in the
[commissioning guide](lab/direct_gpu_stream/COMMISSIONING.md). Keep exact host
inventories and run artifacts in private storage outside Git.
