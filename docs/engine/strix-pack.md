# Convert and verify Strix packs

A Strix pack stores compact weight matrices in SP4, SP3 or SP2. Each format scales groups
of 32 weights and uses 68, 52 or 36 bytes per 128 weights respectively. Keep the source
checkpoint beside the pack. The runtime uses it for retained tensors and checks that it
matches the checkpoint identity recorded during conversion.

## Convert a checkpoint

Build the host tools or install the host tools package. For Inkling 2 Mini, start with the
[SP4 description](../../examples/inkling2-mini-pack.json). Installed descriptions are under
`share/strix/examples/` in the installation prefix.

```sh
strix-pack convert --model MODEL.safetensors \
  --description examples/inkling2-mini-pack.json --plan-only

strix-pack convert --model MODEL.safetensors \
  --description examples/inkling2-mini-pack.json --out MODEL.sp4.pack --quality

strix-pack verify --pack MODEL.sp4.pack --model MODEL.safetensors
```

`convert` accepts one safetensors file, a sharded safetensors index, or its checkpoint
directory. Rules select BF16, MXFP4 or block-scaled FP8 matrices. A rule can also split a
rank-3 NVFP4 or MXFP8 bank, including its scale and outer-scale planes, into individually
addressable expert matrices. The converter uses fixed tables and round-to-nearest codes
from one group maximum. `--quality` reports per-role weight SQNR against decoded source
weights. Model quality requires a separate gate.

The description is JSON with schema `strix-pack-convert/v1`. Set each entry in `roles` to
`sp4`, `sp3`, `sp2` or `keep`. In `rules`, give each rule its `role` and a tensor name
`prefix`, `suffix` or both. Each tensor can match one rule. A rank-3 expert rule declares
`layout: "expert-contiguous"` or `"expert-interleaved"` for gate/up halves interleaved in the
source. Unmatched tensors stay in safetensors. A `keep` rule can supply its own `reason`.
Use narrower selectors when a selected tensor's dtype or width is unsupported. For SP4,
SP3 and SP2 the logical column count must be a multiple of 128.

The plan prints byte totals by role, the retained tensors and their reasons, and a bound
on the largest matrix's conversion scratch. The node figures are resident weight estimates
for one full copy and an ideal evenly sharded pair, excluding KV, activations, allocator
reserve and replication. A replica uses the full copy per node. A pack's on-disk bytes
also include framing, alignment and the digest. Conversion holds one decoded matrix and
one packed payload at a time, and publishes a new path on a filesystem with hard-link
support. `--threads` limits encoder threads. `--scratch-mib` caps matrix scratch at 8,192 MiB by default.

For calibrated coding, use the [calibration tool](lowbit-calibration.md) and its
`--checked-role ROLE` option to write a v3 pack for that role. The calibration identity
records the fitting moments and damping. Standalone RTN has an all-zero calibration
identity. The calibration tool's default output is the existing v2 sidecar.

## Use the pack

```sh
strix-inkling2-mini-cpu MODEL.safetensors 1200 \
  --profile fast --host-pack MODEL.sp4.pack --speculation off --generate 1
```

Inkling 2 Mini's explicit CPU path binds present attention, controller down, head and
expert projections. It opens the shape and role table and binds the checkpoint digest
first. As the run requests a role, its packed bytes are mapped and checked. Each
physical expert projection has one cache shared by every visit to that layer and
loads each routed expert once. To warm a hot set before any decode, pass
`--host-pack-hot ROUTES.json` with a `routes` array of `[physical_layer, expert, count]`
triples for the experts to warm. Positive counts take priority from highest to lowest,
with ties in physical layer and expert order. Unpacked matrices
continue to come from the checkpoint. CPU projection consumes the packed codes and
scales directly with FP32 accumulation.

EXACT uses the incumbent weights. FAST and TURBO select their existing qualified rows
when invoked without a pack option. The v3 CPU path is explicitly lossy and does not
have a tier gate verdict. Current device options, including the fixed-table Inkling head,
have their own pack identity and measured bands. Check the
[Inkling profile documentation](lowbit-calibration.md#inkling-2-mini-rows-gfx1151-hip-71)
before selecting a device option.

## Integrity and load measurements

V3 stores a format revision, group size, scale form, role, source digest, coder, calibration
identity and payload digest for each tensor. Each payload starts on a 4 KiB boundary. The
final 32 bytes are the SHA-256 of every preceding file byte, including padding. A first
mapped access checks only that tensor's SHA-256. Keep mapped files immutable for their
entire lifetime. `verify` reads with bounded buffers, checks the complete image and every
tensor digest, and optionally binds the original checkpoint through `--model`. With
`--expect SHA256`, the file must match a separately recorded identity. V2 remains readable
with its historical header-and-payload identity. For a v2 pack, supply `--expect` to check
the current file against its original content. Unknown versions are refused.

```sh
strix-pack probe --pack MODEL.sp4.pack --tensor model.llm.unembed.weight
strix-pack probe --model MODEL.safetensors --tensor model.llm.unembed.weight
```

Each probe reports milliseconds from index open to a usable tensor and the process peak
resident bytes. The pack arm verifies the selected tensor. The checkpoint arm reads its
raw payload into memory. Add `--mmap` to the checkpoint arm to measure a raw mapped view and
first-byte touch. These are process startup probes. File-cache state depends on the host.
