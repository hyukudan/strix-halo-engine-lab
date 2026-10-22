# deepseek41 hot-expert placement (reader and layer split)

Status: host reader of the placement file and the pure split of a layer's routed rows into local and remote
work; not yet wired into a device forward. Files: `include/strix/ds41_hot_placement.hpp`,
`src/ds41_hot_placement.cpp`, tests in `tests/ds41_hot_placement_tests.cpp` (`strix-ds41-hot-placement-tests`).

## File format `tensormeld/hot-expert-placement-v1`

One UTF-8 JSON object per placement:

```json
{
  "format": "tensormeld/hot-expert-placement-v1",
  "model": {"family": "deepseek-v4.1-flash", "file": "...gguf", "model_identity": "<16 hex or null>",
            "layers": 40, "experts": 384, "top_k": 6, "expert_bytes": 9953280, "canonical_split": 192},
  "dense": {"device": "rtx-pro", "layers": "all"},
  "devices": [
    {"id": "rtx-pro",  "kind": "cuda", "role": "parts-host",   "budget_gib": 80.0, "reserve_gib": 9.5, "expert_gib": 70.5},
    {"id": "rtx-4090", "kind": "cuda", "role": "expert-tier",  "budget_gib": 20.0, "reserve_gib": 0.0, "expert_gib": 20.0},
    {"id": "pc-ram",   "kind": "host", "role": "expert-cache", "budget_gib": 0.0},
    {"id": "amd",      "kind": "hip",  "role": "expert-server",
     "servers": [{"node": "node1", "experts": [0, 192]}, {"node": "node2", "experts": [192, 384]}]}
  ],
  "layers": [{"layer": 0, "rtx-pro": [1, 5, 9], "rtx-4090": [], "pc-ram": []}, "... one entry per layer"],
  "combine": {"identity": "ds41-canonical-halves-v1", "remote_return": "..."},
  "fit": {"...": "provenance of the fit"}, "expected": {"...": "predicted figures"}
}
```

Device ids: `rtx-pro` (the parts-host GPU: dense layers, resident experts, the combine), `rtx-4090` (a second
CUDA device of the same host; returns vectors), `pc-ram` (pinned host memory of the parts-host; copied to the
parts-host GPU on demand and computed there), `amd` (expert servers holding every expert; each server holds a
contiguous id range). `layers[l][dev]` lists the experts resident on `dev` for layer `l`; `amd` is implicit:
every expert not listed elsewhere.

`parse_hot_placement(text, expect)` / `load_hot_placement(path, expect)` reject:

- an unknown `format` or `combine.identity` (additive keys are allowed within v1);
- geometry (`layers`, `experts`, `top_k`, `expert_bytes`, `canonical_split`) different from the caller's;
- a non-null `model_identity` different from the caller's (a null identity is accepted);
- unknown device ids, a repeated device, a missing `amd` entry, AMD server ranges that do not cover
  `[0, experts)` contiguously;
- layers not `0..layers-1` in order, an explicit `amd` list, an unknown array-valued key in a layer entry
  (it would silently move experts to `amd`), ids out of range, not strictly ascending, or on two devices;
- a device over its memory: `resident experts * expert_bytes > capacity`, capacity = `expert_gib` if given,
  else `budget_gib - reserve_gib`, times 2^30 (not checked for `amd`).

Result: per layer and expert the device (`HotDevice::rtx_pro | rtx_4090 | pc_ram | amd`) and the **weight slot**,
the index of the expert in its device's ascending list for that layer (dense numbering per device and layer, the
order of a device's weight upload; for `amd` the index in the complement list). `amd_server_of(id)` gives the
server of an AMD expert.

## Canonical combine

`y = (shared + low) + high`; `low` folds the row's experts with id < `canonical_split` in ascending id starting
from 0, `high` the rest likewise; each step is one FP32 product (weight times unweighted raw vector) then one
FP32 add (`ds41-moe.md`). Adding a remote partial and a local partial of the same half changes the bits in
general, so a remote target returns:

- the raw (unweighted) FP32 vector of each (row, expert) it computes, which the parts-host places in its
  canonical position; or
- for the leading run of one half of a row's ascending order on one target (at least two positions), one
  **folded partial** of that prefix, computed as the leading steps of the same fold with the routing weights.
  The parts-host starts that half's accumulator from it and continues the fold. Positions of the same target
  later in the half (after a position elsewhere) are returned raw. A target with `n` positions in a half and a
  leading run of `run >= 1` returns `n - run + 1` vectors (`run = 1` is the raw case); a target that does not
  lead the half returns `n` raw vectors.

Nothing else is pre-aggregated. Targets: `rtx-4090` and each AMD server; `pc-ram` experts are computed on the
parts-host GPU and only add upload traffic.

## Layer split

`split_hot_layer(placement, layer, ids, weights, rows, sizes)` takes `rows` rows of `top_k` router ids and
weights in router order and returns `HotLayerSplit`:

- `order[row * top_k + p]`: canonical position `p` of the row (ascending id, the low half first) with expert,
  router slot `row * top_k + k`, weight slot, weight, device, AMD server and whether it is covered by a folded
  partial; `low_counts[row]`; `folded_prefix[row * 2 + half]` (0 = none);
- per device: distinct experts used and (row, expert) positions; `upload_bytes` = distinct `pc-ram` experts *
  `expert_bytes`;
- per target with work (`rtx-4090` first, then AMD servers ascending): the requests in (row, half, position)
  order (`raw` with its position, or `folded_prefix` with its length), distinct experts, rows sent,
  raw / folded vector counts, `input_bytes = rows sent * 5808` (A8 input with the row's ids and weights) and
  `output_bytes = vectors * 20480` (one FP32 hidden vector each); totals `network_input_bytes` /
  `network_output_bytes` over the AMD servers and `pcie_input_bytes` / `pcie_output_bytes` for `rtx-4090`.

It is a pure function of the placement and the routes; an offline simulator using the same accounting gets the
same traffic figures. The tests rebuild the combine from the requested vectors for random 8-row windows and
check it equals the canonical combine bit for bit, including halves with a folded prefix followed by raw
vectors of the same target.
