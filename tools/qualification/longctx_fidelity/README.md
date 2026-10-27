# Native long-context symbolic fidelity kit

This Python 3 standard-library kit constructs retrieval and reasoning prompts, computes
independent symbolic ground truth, plans bounded turns and scores exact answers. CPU
generation, planning and tests do not load a GPU. Only an explicit `run_cli` invocation
executes the model. No engine code is changed.

Ground truth is the task's mathematical answer. A numerical reference would compare
logits or state under a declared profile and causal history; retrieval accuracy does
not establish numerical equivalence. No continuation, PPL, HTTP adapter, state capture,
checkpointing or rotary extension is implemented here. Native qualification remains
unclaimed until unchanged-native token/logit/seven-state exactness, model fidelity,
boundary, repeatability and complete-workload gates pass. Missing evidence is unmeasured.

## Tasks and determinism

`pcg32.py` implements PCG XSH RR 64/32 with documented constants and two-step seeding;
Python `random` and process-dependent hashes are never used. The task is a pure function
of family, depth, position and seed, with explicit answer variant and filler stratum.

* `niah_single`: 128-bit exact key, opaque 16-character value.
* `multikey` / `multikey32`: 8/32 relevant exact keys, near-prefix distractors,
  obsolete versions and 3-8 evidence blocks. Greatest numeric version wins, regardless
  of text order. Ordered JSON pairs are the explicit multi-answer exception to the
  24-character scalar answer limit; each value is eight characters. Reserve enough
  output tokens for the complete JSON (for example G=2048), reducing D accordingly.
* `vartrack` / `vartrack256`: 16 registers and 64/256 events. Increasing sequence
  number determines copy/swap/increment order; increments use mathematical modulo 97.
* `multihop2` / `multihop` / `multihop8`: 2/4/8 unique primary links with other-namespace
  distractors, no queried cycles, forward/reverse/interleaved text order by seed.
* `aggregate` / `aggregate128`: 32/128 signed bounded integer events over eight groups;
  filter exact group and `flag=keep`, then sum/count/min. Every event counts, including
  duplicate rows. Empty sum/count/min is explicitly zero.

Filler strata `prose`, `records`, `similar-records` are separately reported. Grammar
units have unique serials and PCG tags (32-bit in `prose`, 128-bit in the record strata);
no existing block is tiled. Filler
contains observations, never active instructions. Answer/key collision checks cover
opaque evidence. Entropy is synthetic, not an estimate of natural text difficulty.
Padding character count is recorded separately; PLE touched pages remain unknown.

For multi-block tasks the first block tracks the requested fractional anchor. Other
blocks occupy successive distinct decile centers. Final offsets include template and
instructions. Construction refuses overlapping/impossible placements rather than
moving or truncating facts. Normal scalar answers are at most 24 characters. `symbol`
uses A-H with an explicit codebook or modulo-8 rule. All eight symbols must encode as
one token both alone and appended to the final assistant prefix; otherwise build fails.
The codebook is a deliberate answer mapping, not an independent retrieval oracle.

## Generate and inspect (CPU)

Run from the repository root, using a private output directory outside Git. All commands
below use relative staged placeholders. `python3` may be named `python` on Windows.

```sh
python3 -m tools.qualification.longctx_fidelity.build --fake --family niah_single \
  --depth 8192 --position 0.05 --seed 104729 --out PRIVATE_CASES
python3 -m tools.qualification.longctx_fidelity.build --tokenizer bin/strix-tokenize \
  --model assets/MODEL.gguf --model-receipt assets/model-content.json \
  --family niah_single --depth 262143 --generate 1 --variant symbol \
  --position 0.95 --out PRIVATE_CASES
```

`FakeTokenizer` maps UTF-8 bytes to IDs and provides reproducible CPU fixtures. Its
lengths have no model-token interpretation; the runner refuses fake cases. Regenerate
with the real tokenizer before execution. `--manifest-only` writes no prompt or ID
files, for private dry-run matrices only.

`StrixTokenize` follows `apps/tokenize_main.cpp`: `encode` parses control tokens,
`plain` does not; each reads a complete UTF-8 file and returns comma-separated IDs.
`decode` reads a comma/space-separated ID file and returns raw UTF-8. Every encoding
subprocess handles one full candidate prompt, never a fragment. Token-boundary recovery
uses CPU prefix decoding. There is no implicit BOS. A frozen model content receipt
is mandatory; the kit hashes that receipt without rehashing a large model every call.
Verify the actual staged GGUF against the content receipt before acquiring a turn.

The supported chat is exactly one user message, no system/tools, `enable_thinking=false`,
with generation prompt. The server strips outer user whitespace and renders:

```text
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The template contract is in `src/serve/chat_template.cpp`; its placeholder rendering
SHA256 is recorded. This is not a claim of HTTP/CLI profile equality.

D is the consumed prompt-token count, last input position D-1, first generated token
position D. D+G must be <=262144. D=262143 requires G=1 and `symbol`. Rich-answer cells
need smaller D. The builder iterates whole-prompt encoding, adjusts only generated
filler, checks exact D and lossless roundtrip, and recovers covering token spans for
every immutable fact. Starts must be within eight tokens of recorded targets; BPE may
place a delimiter and fact in the same token. Drift or nonconvergence is a refusal.
Single-unit filler padding is always verified by full-prompt encoding. Construction
may require multiple CPU tokenizer loads; it is intentionally outside GPU turns.

Each case writes `.txt`, little-endian `.u32` and canonical `.json`: sorted keys,
compact separators, UTF-8, no NaN, trailing LF. Hashes bind text, IDs, ground truth,
Python sources, generator version/seed, tokenizer binary and model receipt. Fields
for resolved configuration, references and state lineage start null; no evidence is
invented. `write_case` publishes the manifest atomically after its assets.

## Plan and prepare a turn (CPU)

```sh
python3 -m tools.qualification.longctx_fidelity.plan PRIVATE_CASES/*.json --out PRIVATE_PLAN
bash tools/qualification/longctx_fidelity/mkturn.sh \
  --binary bin/strix-hip-decode-token --model assets/MODEL.gguf \
  --model-receipt assets/model-content.json --turn-list PRIVATE_PLAN/turn-001.txt \
  --receipts PRIVATE_RECEIPTS/turn-001 --script PRIVATE_TURNS/turn-001.sh
```

Turn lists contain one explicit case-manifest path per line, relative to the list's
parent directory. The planner writes these relative paths even when plan and cases
are in separate directories. Use explicit case manifests, excluding aggregate
matrix/turn manifests, in the planner's input glob. PowerShell does not expand native
command globs: pass the file list with `(Get-ChildItem PRIVATE_CASES -Filter *.json).FullName`.

`plan --matrix dimensions.json --out PRIVATE_PLAN` also expands a scheduling blueprint.
Its JSON has `modes`, `depths`, `families`, `seeds`, `positions` arrays and optional
`generate`, `stratum`, `variant`; only native cells within D+G are accepted. This mode
writes planned matrix metadata, not executable prompt assets. Generate matching real
cases before execution. Matrix expansion sorts `(mode,depth,family,seed,position)` and
never probes devices.
Turns keep one mode/depth/family; cells at 192K and above each get a separate turn.
Per-case predicted bounds: prefill 8K=4s, 32K=25s, 64K=50s, 128K=105s, 192K=160s,
262K=210s, plus 40s model load **for each fresh process**, plus G*0.03s decode.
Non-bucket depths round up to the next depth bucket. Default soft bound is 500s;
`--prediction-table TABLE.json` overrides prefill upper bounds in plan and runner
(and script generation). Its JSON maps depth upper bounds to seconds. Use measured
upper bounds for an actual campaign. These defaults are estimates, not measurements.

`mkturn.sh` only generates a bash script and a hashed file manifest. Stage paths
relative to the repository root. The generated script tests executable permission,
hashes binary/model receipt/list/case assets/Python sources, checks `ldd` for missing
libraries, runs `bash -n` and rejects CR/CRLF text. It invokes the runner, then archives
receipts even when the runner fails. Submit it using the authorized
site GPU-turn wrapper (one GPU job at a time, hard wall-clock watchdog). Script generation
does not request or consume a turn. The soft clock includes script preflight time.

## Run, decode, score

Within an authorized node turn, `run_cli` invokes the plain public command
`strix-hip-decode-token MODEL.gguf <ids...> --generate G`, first with
`--print-effective-config`. No legacy organization flags are supplied. The environment
contains only PATH/LD_LIBRARY_PATH/HOME and explicit `--allow-env NAME=VALUE` additions;
the complete environment is captured in private receipts. Overrides are deliberate
profile changes and require their own campaign identity. Secrets do not belong there.

All case files and argv sizes are checked before the first execution. Linux checks
`getconf ARG_MAX` including environment, pointer overhead and safety headroom, plus
per-string limits. Windows checks its UTF-16 command-line limit. Large prompts can
exceed argv admission: that is an explicit refusal; this kit adds no engine input-file
loader. Do not interpret a refused case as a quality measurement.

The runner stops starting cases when elapsed + predicted >500s. It bounds subprocesses
by remaining soft time, captures configuration/stdout/stderr and wall time, checks echoed
prompt IDs, and atomically writes receipts. Failure, partial JSON, abnormal stop, missing
IDs or deadline leaves INCOMPLETE; following cases are also unmeasured. EOS is a complete
engine result whose answer still has to pass scoring. Raw receipts remain immutable.

After the turn, decode on CPU and score:

```sh
python3 -m tools.qualification.longctx_fidelity.score decode \
  --tokenizer bin/strix-tokenize --model assets/MODEL.gguf \
  --model-receipt assets/model-content.json PRIVATE_RECEIPTS/*.receipt.json
python3 -m tools.qualification.longctx_fidelity.score score \
  --receipts PRIVATE_RECEIPTS --out PRIVATE_REPORT/scores.json PRIVATE_CASES/*.json
python3 -m tools.qualification.longctx_fidelity.score self-test
python3 -m unittest discover -s tools/qualification/longctx_fidelity/tests -v
```

CPU decoding writes a sidecar bound to the raw receipt hash and separates a declared
terminal protocol EOS (native ID 248044) from
answer content: `serial_request.cpp` includes it in generated IDs, and tokenizer decode
preserves controls. Answer content then ends at the first `<|im_end|>` (ID 248046; the model emits
`<|im_end|>` and a newline before the protocol EOS), or at the separated EOS when no `<|im_end|>` is
present. The sidecar retains the full raw text of every emitted ID and the separated EOS ID. Scoring accepts COMPLETE
outputs only when case/input/ground-truth/manifest hashes match. Missing output is never
credited, even if a receipt says COMPLETE. PLANNED means no receipt; INCOMPLETE means
missing or invalid result; COMPLETE is measurement availability, not an accuracy pass.
Normalize only surrounding whitespace and one final period. No case folding, reasoning
removal or substring acceptance. Multi-key ordered JSON pairs report per-value and
all-values exact match; duplicate keys fail. Tables show depth by position, measured
counts and Wilson 95% intervals; zero measurements yields NA. The swapped-value mutation
self-test must pass before reporting. Keep generated prompts, receipts and unpublished
campaign results outside Git, following the repository publication policy.

## First pilot (measured)

Plain default command on gfx1151 (Strix Halo, 128 GB), `niah_single`, `prose` filler, seed 104729,
one process per case:

| Consumed tokens | Positions | Exact | Wall per case, incl. model load |
|---:|---|---:|---:|
| 8,192 | .05 .15 .25 .35 .45 .55 .65 .75 .85 .95 | 10/10 | 9.3 s (44 s cold first case) |
| 32,768 | .05 .25 .50 .75 .95 | 5/5 | 27 s (61.5 s cold first case) |
| 131,072 | .05 .95 | 2/2 | 101 s |
| 262,112 | .50 | 1/1 | 244 s |
| 262,143 (one answer token, symbol set A-H) | .95 | 1/1 | 209 s |

Every case above was built by this version of the generator, and a repeated 8K case produced
identical generated token IDs. Every case used the same needle key/value
(one seed), so this establishes that retrieval works at each depth under the default resolution, not an
accuracy rate: the confirmatory matrix needs held-out seeds, the multi-key and reasoning families, the
other filler strata and continuation scoring before long context is called qualified.

Two practical details found on hardware: the model ends its answer with `<|im_end|>` and a newline
before the protocol EOS (the scorer cuts at `<|im_end|>`), and a 262,143-ID argv needs the larger stack
limit that the generated turn script sets.
