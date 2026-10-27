# Native long-context symbolic fidelity kit

This Python 3 standard-library kit constructs retrieval and reasoning prompts, computes
independent symbolic ground truth, plans bounded turns and scores exact answers. CPU
generation, planning and tests do not load a GPU. Only an explicit `run_cli` invocation
executes the model. No engine code is changed.

Ground truth is the task's mathematical answer. A numerical reference would compare
logits or state under a declared profile and causal history; retrieval accuracy does
not establish numerical equivalence. No continuation, PPL, HTTP adapter, state capture,
checkpointing or rotary implementation is provided here; explicit YaRN cases use the engine's
experimental extension (not yet in this repository's engine; see below). Native qualification remains
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
  output tokens for the complete JSON: default G=2048 for eight keys and G=8192 for
  32 keys, reducing D accordingly. The question requests `[[key,value],...]` pairs.
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

## Task answer protocol and short controls

Generator version `longctx-fidelity-v2` changes normal reasoning-family questions and
makes the normal multikey pair request explicit. `niah_single` prompt bytes and its
CPU golden ID hash are unchanged. Existing aggregate and multihop goldens changed;
vartrack and multikey now have additional goldens. Symbol questions are unchanged.

Normal `vartrack*`, `multihop*` and `aggregate*` questions end with:
`You may reason briefly. End your reply with one final line: Answer: <value>`.
Their builder and matrix default is G=1024 (multihop, aggregate), G=4096 (vartrack) and
G=8192 (vartrack256): the model reasons step by step even in non-thinking mode, and 64-event
vartrack did not finish within 1024 tokens even as a short control. The scorer takes the value on the LAST
line starting literally with `Answer:`. It trims outer whitespace, optional enclosing
backticks/quotes and one final period. Other reasoning text, including inline
`Answer:` decoys, does not supply an answer. Missing protocol lines in measured
outputs are wrong, not INCOMPLETE. Score rows record `extracted_answer` and
`answer_line_found`. Symbol variants retain their one-symbol protocol and existing
default G=32; explicit G=1 edge probes remain unchanged. `niah_single` keeps G=32.
Explicit `--generate` overrides defaults; D+G must fit the selected mode's cap.

Multikey scoring accepts exactly two homogeneous JSON array shapes:
`[["key","value"],...]` or `[{"key":"key","value":"value"},...]`.
One optional JSON code fence and surrounding text are accepted. Keys and values must
be strings; duplicate keys/fields, mixed shapes and truncated JSON fail. All-values
EM requires the complete values and requested key order. Per-value EM looks up each
expected key in a valid array, so reordered correct values retain per-value credit
but fail all-values EM. Invalid/truncated arrays receive no per-value credit; the
scorer never repairs JSON or rescues an inner array. EOS/turn handling still ends
answer content at the first `<|im_end|>` as described below.

For design 3.3 short controls, add `--short` to the same build command. Family, seed,
position, stratum and answer variant select the same facts, question and evidence
order as the long arm. The builder removes all filler/padding and encodes the entire
immutable task/template; this token count becomes actual D. `--depth` is retained as
`requested_depth` (and must satisfy the normal budget); span targets describe actual
short placements. Short case IDs end in `.short` and use actual D, so rebuilding the
same control with another requested depth produces the same prompt and ID. Generate
one short control per layout. Plan short controls from their built manifests so the
planner uses actual D, rather than a nominal matrix depth.

Reports group controls by `(family, seed, layout)`, where layout records position,
stratum and answer variant. `short_long_pairs` contains each measured long arm's
exact and per-value long-minus-short deltas against its unique measured short arm.
Unmeasured or ambiguous short controls are omitted, never treated as zero accuracy.
The text report prints paired exact deltas; heatmaps label short arms separately.
These deltas measure filler/context effects within the declared synthetic task and
protocol; they do not establish full model qualification.

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
position D. Native D+G must be <=262144. Native D=262143 requires G=1 and `symbol`. Rich-answer cells
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

## YaRN mode cases (experimental)

Select `build --mode yarn2` or `--mode yarn4` explicitly; the default remains `native`.
These cases need an engine build with the explicit `--rope-yarn` mode and `--prompt-ids`
input, which the engine in this repository does not provide yet; an engine without them refuses the arguments, and the case stays INCOMPLETE.
The caps are 524288 and 1048576 respectively, and D+G must fit the selected cap.
The requested context C defaults to `max(300000, round_up_512(D+G))`.
For D=299000 and G=32 this is 300000, since round_up_512(299032)=299520;
for D=300000 and G=32 it is 300032. Override with `--context C`.
The builder refuses C unless `262144 < C <= cap` and `D+G <= C`.
Native context stays automatic; `--context` requires a YaRN mode.

Case IDs start with the mode; YaRN manifests record `mode` and requested
`context`, `rope_mode`, and `limit`. Native prompt bytes, IDs and manifest fields
are preserved (source hashes track the current kit). Outputs are not comparable
to native token-for-token; each mode declares a different numerical identity.
These cases do not establish model fidelity or production qualification.

The runner writes whitespace-separated decimal IDs to `CASE.prompt-ids.txt` beside
the receipts and invokes `MODEL.gguf --prompt-ids FILE --rope-yarn F --context C --generate G`.
Configuration inspection uses the same arguments plus `--print-effective-config`.
Receipts record exact argv and the generation stderr line
`Qwen experimental YaRN: factor=... context=... T=...`; missing or mismatched
factor/context leaves INCOMPLETE. Turn preparation creates and hashes the prompt-ID
text files along with the other staged assets. Native positional IDs are unchanged.

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
`generate`, `stratum`, `variant`; modes are `native`, `yarn2`, `yarn4`, each within its D+G cap. This mode
writes planned matrix metadata, not executable prompt assets. Generate matching real
cases before execution. Matrix expansion sorts `(mode,depth,family,seed,position)` and
never probes devices.
Turns keep one mode/depth/family; cells at 192K and above each get a separate turn.
Per-case predicted bounds: prefill 8K=4s, 32K=25s, 64K=50s, 128K=105s, 192K=160s,
262K=210s, 393216=330s, 524288=470s, 786432=710s, 1048576=950s,
plus 40s model load **for each fresh process**, plus G*0.03s decode.
Non-bucket depths round up to the next depth bucket. Default soft bound is 500s;
`--prediction-table TABLE.json` overrides prefill upper bounds in plan and runner
(and script generation). Its JSON maps depth upper bounds to seconds. Use measured
upper bounds for an actual campaign. These defaults are estimates, not measurements.
Cases exceeding the soft bound are refused, including deep YaRN cases under default estimates.

`mkturn.sh` only generates a bash script and a hashed file manifest. Stage paths
relative to the repository root. The generated script tests executable permission,
hashes binary/model receipt/list/case assets/Python sources, checks `ldd` for missing
libraries, runs `bash -n` and rejects CR/CRLF text. It invokes the runner, then archives
receipts even when the runner fails. Submit it using the authorized
site GPU-turn wrapper (one GPU job at a time, hard wall-clock watchdog). Script generation
does not request or consume a turn. The soft clock includes script preflight time.

## Run, decode, score

Within an authorized node turn, `run_cli` invokes the plain public command
`strix-hip-decode-token MODEL.gguf <ids...> --generate G` for native cases, first with
`--print-effective-config`. No legacy organization flags are supplied. The environment
contains only PATH/LD_LIBRARY_PATH/HOME and explicit `--allow-env NAME=VALUE` additions;
the complete environment is captured in private receipts. Overrides are deliberate
profile changes and require their own campaign identity. Secrets do not belong there.

All case files and argv sizes are checked before the first execution. Linux checks
`getconf ARG_MAX` including environment, pointer overhead and safety headroom, plus
per-string limits. Windows checks its UTF-16 command-line limit. Large prompts can
exceed argv admission: that is an explicit refusal. YaRN cases use the engine's existing
`--prompt-ids` input-file contract. Do not interpret a refused case as a quality measurement.

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
Scalar retrieval/symbol scoring trims only surrounding whitespace and one final
period, without case folding or substring acceptance. Reasoning and multikey answers
follow the explicit task protocols above. Multi-key answers report per-value and
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

Every case above was built by generator v1 (v2 preserves these retrieval prompt bytes), and a repeated 8K case produced
identical generated token IDs. Every case used the same needle key/value
(one seed), so this establishes that retrieval works at each depth under the default resolution, not an
accuracy rate: the confirmatory matrix needs held-out seeds, the multi-key and reasoning families, the
other filler strata and continuation scoring before long context is called qualified.

Two practical details found on hardware: the model ends its answer with `<|im_end|>` and a newline
before the protocol EOS (the scorer cuts at `<|im_end|>`), and a 262,143-ID argv needs the larger stack
limit that the generated turn script sets.

## Reasoning and multi-key pilot (measured)

Seed 130363, `prose` filler, plain default command, positions .25 and .75 at 8,192 and 32,768 tokens
plus one short control per family (the same facts with minimal filler):

| Family | Exact | Notes |
|---|---:|---|
| `multikey` (8 keys) | 4/4 | 32/32 values; one 8K position cannot place its evidence blocks and is refused at build time |
| `multihop` (4 links) | 5/5 | |
| `aggregate` (32 events) | 5/5 | |
| `vartrack` (64 events) | 1/5 | the short control and one 8K case exhaust G=4096 tracing every event; both 32K cases answer a wrong final value |

The short-control failure shows that the 64-event variable-tracking task exceeds the output budget
for this model even without long context; it is a task-capacity limit, not a depth effect. Two kit
issues found by these runs are fixed: the scorer accepts the last complete fenced JSON block after
reasoning text with inline code, and the aggregate questions name count/sum/min unambiguously.
