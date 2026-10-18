# Explicit native MTP CLI

`--mtp-metadata-only 1` is an explicit diagnostic. It retains the same actual
asset, provenance, rounding and budget admission, but does not require golden
file/path pins. It loads the real owners and reports their plan, semantic and
native/head profile identities, content identities and full file SHA values.
It returns before any target or predictor forward and never activates or
qualifies MTP. The ordinary path still requires independent expected outputs;
the diagnostic cannot substitute for a model fixture invocation.

`strix-hip-decode-token MODEL.gguf TOKEN... --generate N --mtp` selects a separate eager target request. It requires at least three prompt tokens and externally supplied independent full-forward reference rows for every adjacent prompt pair. The default decoder route and optimization profile remain unchanged. Graph, prepared dispatch, overlap, islands, other request controllers and recording/export routes are rejected before native setup rather than silently disabled.

The request requires all of these values (each accepts `--name VALUE` or `--name=VALUE`):

| Options | Meaning |
| --- | --- |
| `--mtp-sidecar`, `--mtp-sidecar-logical-name`, `--mtp-sidecar-sha256`, `--mtp-sidecar-bytes` | Actual local file, its framed content namespace and expected complete file digest/size |
| `--mtp-target-sha256`, `--mtp-target-content` | Expected complete target file digest and existing framed target content identity; these are different hashes |
| `--mtp-source-repository`, `--mtp-source-revision`, `--mtp-source-path` | Declared source provenance, retained with the verified asset |
| `--mtp-converter-revision`, `--mtp-converter-sha256`, `--mtp-norm-converter-sha256` | Pinned conversion reference provenance |
| `--mtp-conversion effective-norm-embedding-first-v1` | Explicit admitted checkpoint interpretation |
| `--mtp-normalized-rounding fp32\|bf16-rne`, `--mtp-projection-rounding fp32\|bf16-rne` | Both native cast boundaries; they must match the independent golden profile |
| `--mtp-sidecar-residency register\|copy` | Explicit sidecar storage choice; the target's existing mapping/residency is retained |
| `--mtp-header-bytes`, `--mtp-sidecar-retained-bytes`, `--mtp-sidecar-setup-bytes` | Positive bounded header and sidecar backing reservations |
| `--mtp-golden`, `--mtp-golden-sha256`, `--mtp-golden-max-bytes` | Independent versioned full-value file, expected complete SHA256 and positive file budget; see [format](mtp-forward-golden-file.md) |
| `--mtp-replay-host-bytes`, `--mtp-runtime-bytes`, `--mtp-target-capture-bytes` | Positive complete replay payload, independent predictor pool and target capture reservations |
| `--mtp-policy-id`, `--mtp-candidate-depth` | Nonzero declared replay policy identity and maximum depth 1..8 |
| `--mtp-max-hidden-error`, `--mtp-max-logit-error` | Optional finite nonnegative declared tolerances, each defaulting to exact value agreement (zero); selected tokens must agree exactly |

The golden rows must start at source position zero, match all prompt source/next tokens, include every BF16 target input and predictor output wide value, and contain every finite FP32 logit. Full-file pinning binds supplied bytes; it does not establish that a reference was independently produced. The decoder never obtains expected values from its own outputs. Mismatching source/model/semantic/cast identities or replay policy prevent generation. Actual replay observations feed the existing qualification and activation functions; no readiness flag is forced.

### Self-generated golden (developer benchmarking)

`--mtp-golden-dump FILE` replaces `--mtp-golden`/`--mtp-golden-sha256` (it rejects them and `--mtp-metadata-only`). With every other MTP option unchanged, it consumes the prompt with the real target, runs the same native MTP forward that replay runs over each prompt pair, writes the complete version-2 golden (it records the target prefill/decode numerical profile pair) for exactly those rows (refusing to overwrite an existing file), prints a one-line JSON receipt with `mtp_golden_sha256`, and exits without replay, qualification or generation. Window options are accepted but no window is prepared. The file binds the actual target/sidecar content, structural plan, semantic contract and native numerical profile; its independent-source digest is SHA256 of `strix-mtp-golden:self-generated-native-forward:not-independent:v1` and its reference ID is `0x314e4547464c4553` (`SELFGEN1`). A run consuming such a golden reports `mtp_reference_independence: self-generated-native-forward-not-independent`.

Two-step use: (1) run with `--mtp-golden-dump prompt.golden` and the target prompt; (2) run the same command line with `--mtp-golden prompt.golden --mtp-golden-sha256 <printed sha256>` instead. Replay then matches exactly (zero tolerances) under the same profile and casts. Drafter values only change acceptance; every emitted token is still target verified, so the generated output is unaffected. This is not independent evidence and does not qualify the drafter. Size `--mtp-golden-max-bytes` for `184 + rows*(24 + 4*wide + 4*vocab)` (about 1.03 MB per row for the 2560x4-wide, 248320-vocab model; 64 MiB covers a 43-token prompt) and `--mtp-replay-host-bytes` slightly above the file size (64 MiB also covers the 43-token replay).

The prompt is consumed once by the real target hidden producer. Its snapshots drive independent replay, leaving that same predictor primed immediately before the final prompt row. The final prompt target selection is emitted once as an unconsumed pending token. Subsequent cycles consume pending inputs, return only new target-verified outputs, and respect EOS, context and remaining output budget. Serial target verification is the default for this explicit route. Target D1 uses its own real record/shadow on its owned stream when the resolved target catalog requests D1.

Optional closed R2/R4/R8 target verification requires `--mtp-target-window-rows 2|4|8` and all positive values `--mtp-window-arena-bytes`, `--mtp-window-capture-bytes`, `--mtp-window-checkpoint-bytes`, `--mtp-window-total-bytes`, `--mtp-window-history-intervals`. It also requires explicit paired wave8 GDN input/output with their existing virtual/prefetch prerequisites and resolved D1. The declared candidate depth must be at least `rows - 1`; for example, eight-row verification requires depth seven or eight. Remaining per-cycle budget, context or history can still choose serial verification. This prepares the existing actual all-head known-window owner, two capture pools and typed occupied-KV checkpoint. A nonaccepted whole window is restored before serial target verification. Whole-chunk tickets retain the actual final session, without invented intermediate sessions.

Reservations cover declared requested payloads, including target D1 capture/shadow requirements. Window total includes the window arena/completion record plus checkpoint/capture payloads. They exclude opaque driver bookkeeping and container capacity overhead, and do not assert physical RAM fit. Snapshot tails follow their declared logical policy. Receipt fields describe actual operations from a completed invocation, separate declared-policy replay activation from hardware/performance qualification, and label cycle-only host wall timing. Source implementation alone establishes no native or model validation result.

The verification receipt follows actual window attempts. A configured owner with zero attempts reports `serial-target-fallback` and separately records its requested rows/configuration. Pair-entry dispatch counts explicitly exclude wide R4/R8 dispatches; the wide count is `null` because the current verifier result does not expose it. No total is inferred from requested geometry. The base backing inventory keeps its original pre-optional-request boundary. A distinct `mtp_target_d1_inventory` describes the later actual target D1 presence as unaccounted storage, without inventing backing keys or adding it to the base allocation census.


Optional `--mtp-window-expert-union --mtp-window-expert-union-bytes N` connects
closed target verification to the existing routed-expert union. It requires a
configured target window, resolved A8 gate/up, grouped Q5_1 down, slot-parallel
down and deferred status. It does not select these prerequisites implicitly.
The requested union must contain admitted routed steps; otherwise setup rejects
it. Ordinary generation and serial MTP retain their existing behavior.

The native union workspace is allocated once with its own lease and retained
owner. Its actual size must fit the separate explicit union ceiling and the
existing window arena ceiling, together with the arena and eight-byte completion
record. The same charge is subtracted from the existing total window ceiling
before admitting both capture pools and the rollback checkpoint. Declared
ceilings are never enlarged. No union allocation occurs per cycle.

Receipts report the compiled eligible step count, actual workspace size and
generation, and the GPU-produced CSR summary from the last eligible layer of the
last completed target window. This includes provisionally completed windows that
were subsequently restored. That summary is not an aggregate reuse, traffic or
performance measurement. Checkpoint fixtures additionally expose actual union
and variant dispatch counts and require a nonzero union witness when requested.
Independent replay, complete state restoration and full-cycle cache comparisons
remain mandatory; this integration does not qualify speed or arbitrary models.
