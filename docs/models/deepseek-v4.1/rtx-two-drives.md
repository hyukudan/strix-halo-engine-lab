# DeepSeek V4.1 Flash on the RTX PRO: weights from two NVMe drives

The RTX prefill service streams the routed experts of every layer that is not resident on the device from the model
file for each request, and both RTX engines read Engram rows from it on every step. With one drive that stream is
bounded by the drive. Two drives can serve it together without a second full copy of the model.

## Set up

1. Write the byte ranges to place on the second drive (the routed-expert tensors and both Engram tables, interleaved
   chunk by chunk):

   ```sh
   python -I tools/ds41_stripe_ranges.py MODEL_DIR/DeepSeek-V4.1-Flash-1017-Q2.gguf > ranges.json
   ```

2. Build the stripe with tierstream's `ts-stripe` (one sequential read of the model, unbuffered writes, a full
   read-back check; nothing is published if any step fails):

   ```sh
   ts-stripe make MODEL_DIR/DeepSeek-V4.1-Flash-1017-Q2.gguf ranges.json STRIPE_DIR
   ts-stripe verify STRIPE_DIR MODEL_DIR/DeepSeek-V4.1-Flash-1017-Q2.gguf
   ```

   For the Q2 GGUF (340.6 GiB) the stripe is 166.4 GiB: every odd 16 MiB chunk of the 142.4 GiB of routed experts
   and of the two 94.4 GiB Engram tables. The model file itself is not changed.
   The helper finds Engram tables by tensor name on any layer. `--policy interleave` (default) selects odd chunks;
   `interleave-even` selects even chunks and `all` selects every chunk of each chosen tensor.

3. Pass both directories to the engine:

   ```sh
   strix-cuda-ds41-prefill-service MODEL_DIR/DeepSeek-V4.1-Flash-1017-Q2.gguf --weight-dirs "MODEL_DIR;STRIPE_DIR" ...
   ```

   The same flags work on `strix-cuda-ds41-replica` and `strix-cuda-ds41-decode`.

| Flag | Default | Meaning |
|---|---|---|
| `--weight-dirs LIST` | none | `;`-separated directories; the model's own directory and duplicates are ignored. With no other directory the engine reads the model file exactly as before. Environment `STRIX_WEIGHT_DIRS` when the flag is absent; an explicit empty value turns it off. |
| `--drive-readers N` | 4 | reader threads per drive, in 1..32 |
| `--weight-verify cached\|always` | cached | `cached`: each stripe chunk is verified once per stripe version, at its first open through the serving handle; the result is cached by file identity. `always`: every stripe chunk is verified through the serving handle at every open, plus inline whole-chunk re-checks (a qualification mode: inline hashing costs most of the gain). Environment `STRIX_WEIGHT_VERIFY` is used only when a secondary directory is effective. |

The replica passes the model digest from its admission receipt. For the prefill service and decode app, tierstream
hashes the model file when it finds a stripe manifest for it, and keeps the digest in a persistent cache by file
identity; a listed directory without a stripe for this model costs no hashing. Without a cached digest, the first
multi-drive start reads and hashes the whole model once; for a 340.6 GiB file this can take several minutes,
before stripe verification and engine loading. These startup costs are separate from request timings. A start
without secondary directories performs no additional hash work and emits no `weight_io` fields or events.

## Which reads use the second drive

Reads of 8 MiB or more (the whole-layer expert streams of prefill, startup loads) are split between the drives with
shares that follow each drive's measured speed; a slow drive gets little or nothing, and every 16th large read uses
the model drive alone as a live baseline. Decode range reads below 8 MiB stay on the model drive. Engram batches
of fewer than 64 records also stay there; queued batches of 64 or more records may split between drives.

## What is guaranteed

- The stripe is bound to the model file's SHA-256. With cached verification, each stripe chunk is verified once per
  stripe version at its first open, through the serving handle. With always verification, chunks are verified at
  every open plus inline whole-chunk re-checks. Partial-chunk reads rely on that open-time verification. A stripe
  that does not match, a missing drive or a read error makes the engine read those chunks from the model file instead.
  Open-time exclusions are reported at startup; read-time fallbacks are counted in the stats.
- Both files are opened read-only. On Windows they are also shared for reading only, so neither can be changed while
  the engine runs. Linux has no such exclusion (stripe data files are published read-only, 0444, but an owner or a
  privileged process can still change them): keep both files unchanged while the engine runs, and use
  `--weight-verify always` where the stripe directory is not owned by the engine's user alone.
- The service's stats JSON (`weight_io`) reports bytes, reads and busy time per drive, fallback bytes, demoted
  chunks, verification failures, which drives are read unbuffered and the effective verification mode. With
  `--buffered`, the direct decode reader has a separate session, reported as `decode_session` when active.

## Measurements

Prefill service, exact profile, `--drive-readers 4`, `--overlay c-u6 --host-gib 32 --positions 131072`;
model on a PCIe 4.0 NVMe, stripe on a second PCIe 4.0 NVMe; arms interleaved one drive / two drives / one drive with a 60 s rest between them; state
SHA-256, logits FNV and next token equal in every arm. Changes are relative to the mean of the two one-drive runs:

| prompt | one drive A / A2 | two drives B |
|---:|---:|---:|
| 8K | 13.27 / 13.12 s | 10.64 s (-19 %) |
| 32K | 19.39 / 18.98 s | 17.44 s (-9 %) |
| 64K | 29.30 / 28.97 s | 27.76 s (-5 %) |
| 128K | 50.19 / 49.66 s | 48.42 s (-3 %) |

Expert streaming per request: 11.0 -> 8.45 s (-23 %; 29 % of the bytes from the second drive). A third, slower drive
added no benefit. Long prompts are compute bound and gain less. Decode throughput was not measured here; decode
range reads below 8 MiB stay on the model drive, subject to the queued-batch rule above.
