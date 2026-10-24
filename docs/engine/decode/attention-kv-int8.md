# Int8 attention KV cache (kv-int8-tok-v1)

`--attention-kv-storage kv-int8-tok-v1` stores the QSA attention KV cache of the target model in int8 with
one FP32 scale per token, KV head and K/V half. The default, `--attention-kv-storage bf16` (or no flag), keeps
the canonical BF16 cache and every existing route, token, identity and receipt field unchanged.

It is a new named numerical identity, not an organization of the BF16 arithmetic: attention reads
dequantized values, so logits differ from the BF16 cache. It is opt-in and not qualified
(`attention_kv_storage_qualification: not_assessed`).

## Storage

The target model has 12 QSA layers, 2 KV heads and head dimension 256. One KV row (one token, one layer) is

| Part | BF16 cache | kv-int8-tok-v1 |
| --- | --- | --- |
| K | 512 BF16 (1024 B) | 512 int8 codes (512 B) |
| V | 512 BF16 (1024 B) | 512 int8 codes (512 B) |
| Scales | — | 4 FP32: K head 0, K head 1, V head 0, V head 1 (16 B) |
| Row | 2048 B | 1040 B |
| Per token, 12 layers | 24 KiB (24576 B) | 12480 B (12.19 KiB) |

`base_kv_bytes_per_token`, the session state layout, the indexed-attention state layout (`kv_row_bytes`,
`kv_scale_offset`, `kv_scale_bytes`), occupied-KV snapshots and MTP checkpoints/rollbacks all derive from
`ModelContract::attention_kv_storage`, so the KV allocation of a given context is about half (51%) of the BF16
one. Other state (GDN recurrent and convolution state, the sparse index keys, PLE convolution) is unchanged.
The MTP head keeps its own BF16 KV cache (one layer); the indexer reads its own raw index keys, never the KV
cache.

## Arithmetic

For each token, KV head h and half (K or V), with x the 256 values the BF16 cache would store (the FP32
key after norm/RoPE or the FP32 value, rounded to BF16):

- `s = max|x| / 127` (FP32 division, correctly rounded), `s = 0` for an all-zero head;
- `code = clamp(rint(x / s), -127, 127)` (round half to even), `0` when `s = 0`;
- a reader uses `float(code) * s` (one FP32 multiplication) exactly where the BF16 cache uses the BF16
  value.

The maximum is order-independent, so the decode writer, the window/prefill rows writer and every row store
the same bytes. Every score and value kernel keeps its products, reduction tree and value chain; only the
loaded element changes. The decode, window (2/4/8 rows) and prefill-group (16/32/64 rows) score/value kernels
therefore stay bit-identical to one another, as with the BF16 cache. The dequantization error of an element
is at most `s / 2` plus the FP32 roundings of `x / s` and `code * s`.

## Admitted routes

Admitted: eager C1 decode (explicit organizations and the automatic v5 organization), the known-prefix windows
and tiled prefill, teacher-forced scoring (serial and tiled), prefix-accepting MTP target windows and their
checkpoints. A split prefill/decode numerical profile pair is not rejected; it was not checked with this
identity.

Rejected at option parsing: device graph or integer head, GPU greedy token, prefix-cache roundtrip, prompt
lookup, serving, QSA GQA heads per block other than 1 and hierarchical QSA selection, RAW recording/reference
comparison, offline tuning, the fast prefill QSA component (it reads BF16 keys and values), cohort windows,
the prefix state cache and HTTP serving. The QSA GQA launcher also rejects a non-BF16 KV contract.

## Identities

- `ModelContract::attention_kv_storage` enters the compiled execution identity and the serial session layout
  identity (only when it is not BF16, so BF16 identities are unchanged). Prefix-cache and snapshot keys use those
  identities, so a BF16 entry never restores into an int8 session or the reverse.
- The indexed-attention launcher's implementation identity binds the storage (the step's kernel id is unchanged).
- MTP goldens bind the target prefill/decode profile pair together with the KV identity: an int8 run requires a
  version-2 golden dumped under kv-int8-tok-v1 (`--mtp-golden-dump` with the same flags). A BF16 golden, or a
  version-1 golden, is rejected.
- Receipts add `attention_kv_storage`, `attention_kv_bytes_per_token` and
  `attention_kv_storage_qualification` when the storage is not BF16.

## Validation

`strix-hip-attention-kv-int8-tests` checks, on the device, that the decode and rows writers store the host
model's bytes (including an all-zero head, an outlier head and exact half-step ties), that every element is
within half a step (plus FP32 rounding), and that the decode, 8-row window and 16-row prefill-group score/softmax/value kernels give
bitwise identical probabilities and outputs, close to an FP64 attention over the dequantized cache. Full-model
exactness and quality results are recorded in DECISIONS.md (D-046).
