# Inkling 2 Mini native bring-up

The text decoder can run a serial CPU reference directly from the original
checkpoint. The native HIP and CUDA text decoders compose all 48 visits
using resident weights, a device route plan, and a persistent state slice per
visit. CPU compilers and both device toolchains can build the bring-up
fixtures. Full-device behavior still needs qualification on hardware.

## Checkpoint inspection

The inspector checks the original single-shard text tensor schema at
[checkpoint revision ed40e7d](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7).
It reads tensor metadata and validates the optional index against the shard.
The vision tensors are counted. Their model-specific geometry and the separate
drafter checkpoint require their own admission checks.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strix-inkling2-mini-inspect
build/strix-inkling2-mini-inspect MODEL_DIR/model-00001-of-00001.safetensors \
    MODEL_DIR/model.safetensors.index.json
```

Expert matrices in physical layers 0â€“19 pack two E2M1 values in each byte.
Each block of 16 has an E4M3 scale and each expert has an outer scale of
`amax / outer_scale_denominator`. Layers 20â€“23 use E4M3 weights with E8M0
scales for blocks of 32. This schema validates their metadata. Scale values
and device weight decoding are separate numerical checks.

## CPU text reference

```sh
cmake --build build --target strix-inkling2-mini-cpu
build/strix-inkling2-mini-cpu MODEL_DIR/model-00001-of-00001.safetensors \
    TOKEN_ID TOKEN_ID --generate 2 --logits next-token.f32
```

Pass the input token IDs in sequence. The reference evaluates prompt rows
serially and projects logits only for the last prompt row and requested decode
steps. The optional logits file contains the next-token row as 200058
little-endian FP32 values. Generation uses greedy selection and stops on EOS.
The loader reads only selected expert matrices during each MoE call. The CPU
cache stores BF16 operands widened to FP32, so the BF16 cache budget below
applies to the planned device layout. CPU output still needs numerical
qualification against an independent reference for the full checkpoint.

## Native text entry points

```sh
cmake -S . -B build-hip -DSTRIX_ENABLE_HIP_RUNTIME=ON
cmake --build build-hip --target strix-inkling2-mini-hip

cmake -S . -B build-cuda -DSTRIX_CUDA_CORE=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target strix-inkling2-mini-cuda
```

Both commands accept token IDs, raw text via `--prompt`, or chat via
`--messages FILE`. The chat file can be a JSON message array or an object
with `messages` and optional `tools` and `reasoning_effort` fields. The
official `tokenizer.json` defaults to the checkpoint directory. Supply
`--tokenizer FILE` to select another copy of the same tokenizer. Use
`--tokenize FILE` to print raw-text token IDs without loading model weights.
Chat files may carry images as base64 `image_url` parts; see
[Inkling 2 Mini image input](../../engine/vision-adapters.md#inkling-2-mini).

Generation defaults to greedy selection and can emit one FP32 prompt logits row
with `--logits FILE`. The file appears only when the run succeeds (a failed run leaves none) and an existing file
is never replaced: a name that exists at the start, or that appears during the run, fails the run and keeps its
bytes. On Linux the output directory must be on a file system that supports `O_TMPFILE` (the dump has no name
until the run succeeds). Repeating the example below needs a new name (or removing the previous file). CPU and device generation accept `--temperature`,
`--top-k` and `--seed` through the shared sampler.

```sh
build-hip/strix-inkling2-mini-hip MODEL.safetensors \
    --prompt 'Write one line about Ibiza.' --generate 24
build-cuda/strix-inkling2-mini-cuda MODEL.safetensors \
    TOKEN_ID TOKEN_ID --prefill-chunk 128 --generate 24
```

Decode runs the experts on the one-row packed form by default on the command
line (both backends; `--grouped-decode` selects the grouped tile kernel there,
printed as `moe_decode` in `strix-decode-route/v1`). `strix-serve` holds
the last prompt token back and decodes it. On HIP it keeps its existing
route in every profile and takes no decode form flag: prompts run in 128-row
chunks (`--prefill-rows`, clamped to the context, which without `--slots`
also caps the lanes), and a cohort decodes with the grouped form in one
batched forward, whose lane bits depend on the cohort width as before. On
CUDA it runs the command line's decode form (every decode row takes the
packed one-row body at any cohort width, so every lane's bits equal its lone
run, and every prefill row the grouped tile; `--packed-decode` /
`--grouped-decode` as on the command line) and, without `--prefill-rows`, the command line's prompt chunk:
it keeps the largest of 8192, 2048 and 128 rows whose load leaves the slot
count of the 128-row chunk (startup line `prefill_chunk`), and a prompt
admitted beside a decoding lane runs in the shared long pieces of 1024 rows
(`serve::kMixedServiceLongPieces`, cut at 32-row steps), so a decoding lane
waits about one piece and one decode round (190 ms on an RTX PRO 6000); the
pieces run back to back with one decode round between them (prompt share 0,
`STRIX_SERVE_PROMPT_SHARE` replaces it); every prompt row,
a one-row chunk included, runs the prefill form, so the bits do not change
with the chunk or the pieces. Every server runs the command line's route by
default (`inkling2-mini-served-cli-route@1`, the build's behaviour; the switch
`STRIX_INK_SERVED_CLI_ROUTE=0` keeps the held-back route above): the final
prompt row is a prefill row and the first id comes from it, as on the command
line, so the served EXACT ids are the command line's (RTX PRO 6000: 8/8 prompts,
at the held-back route's speed and slot bytes). The chunk ladder still guards
the slot count: the command line's chunk runs this route, a smaller candidate
runs the held-back route (one startup line says so). On HIP the smaller
candidate is the held-back 128-row chunk, so the command line's chunk and tail
(a larger working memory per slot) never cost a session slot; gfx1151 served
EXACT prefill with the command line's chunk: 8K 601 -> 2,030 tok/s, 32K
578 -> 1,870, ids equal to the held-back route. The startup
line reports it as `served_route off`, `declined`, or `on qualification
qualified` with its form, chunk and tail; the command line prints
`strix-served-route/v1 ... effective n/a`. On that route decode is packed and
a served lane keeps the bits of its own lone decode at any cohort width:
cohorts run in passes of at most 16 lanes, the row limit of the dense one-row
form, with one head launch per pass; lanes default to 128 without `--slots`.
A cohort decode on that route checks every lane before any pass appends and
runs its passes through `serve::run_cohort_passes`: the input ids are copied
before the first pass, the picks are published after the last one, and every
lane of the cohort is retired if a pass fails. The packed route uses one kernel for gate/up with BF16
activation and one for down with ordered expert combine.

## Two nodes: replicas

The model fits one node, so two nodes serve independent requests as
replicas: one full `strix-serve` per node and `strix-router` in front. Each
request runs on one node with the single-node identity. The
`[multi_device]` section of `models/inkling2-mini.toml` declares this form;
`single` remains the default. Use `--placement replica` to select the declared two-node form.

Pin the same backend, profile, checkpoint and tokenizer on both workers: without `--backend` a build with HIP chooses HIP and a
CUDA-only build chooses CUDA, and the router checks the endpoint count and the requested tier, not the backend, so a
mixed pair would answer identical requests with different identities. `--backend` goes after `--` so it reaches the
Inkling adapter's own parser (the top-level `strix-serve --backend` is read by the shell and not forwarded to the
in-process arguments); `--profile` is a shell option and is forwarded to the adapter. Pass `--tokenizer PATH`
explicitly too: without it each worker loads the `tokenizer.json` next to its own checkpoint, the loader only checks
vocabulary size and EOS, and two tokenizers with different merges would tokenize the same request differently while
the router, which sees only health and capacity, could not tell.

```sh
# on node A and on node B: one worker each, listening on that node's private address
# (the listener defaults to 127.0.0.1, which the other node cannot reach); same --profile, --backend and checkpoint
strix-serve --model inkling2-mini --profile exact --bind NODE_A_ADDRESS --port 18196 -- --checkpoint MODEL.safetensors --tokenizer TOKENIZER.json --backend hip   # node A
strix-serve --model inkling2-mini --profile exact --bind NODE_B_ADDRESS --port 18196 -- --checkpoint MODEL.safetensors --tokenizer TOKENIZER.json --backend hip   # node B
# identity check, before any traffic: on each node, the checkpoint and tokenizer hashes must be equal across the two nodes
sha256sum MODEL.safetensors TOKENIZER.json
# and the effective identity of each worker (backend, tier/profile) must be the same on both nodes: print it with the worker's own
# command line plus --print-effective-config (the worker's listening line shows only its address; the checkpoint and
# tokenizer paths are the ones on its command line, pinned by the hashes above)
strix-serve --model inkling2-mini --profile exact --print-effective-config -- --checkpoint MODEL.safetensors --tokenizer TOKENIZER.json --backend hip
# only then check that both workers answer from the router's node
curl -fsS http://NODE_A_ADDRESS:18196/health && curl -fsS http://NODE_B_ADDRESS:18196/health
# on the node that takes the API traffic (the same tier the workers run)
strix-router --description models/inkling2-mini.toml --placement replica --profile exact \
  --replica http://NODE_A_ADDRESS:18196 --replica http://NODE_B_ADDRESS:18196 --listen 0.0.0.0:8090
# both endpoints must be listed (healthy, with capacity) before sending traffic
curl -fsS http://ROUTER_ADDRESS:8090/router/status
# then send independent requests through the router and check that both endpoints receive them
```

The router prints `effective-multi-device {...}` and refuses to start if
the model data does not admit replicas for that many endpoints (one
endpoint is refused, and so is the same `host:port` given twice: one worker
is not two replicas). Only the router executes the replica form: each
worker is a one-device launch, and `strix-serve --model inkling2-mini
--device-count 2 --placement replica -- --checkpoint MODEL.safetensors` is refused, since one
process would start one worker. Inkling has no native route in the `strix`
CLI (`strix run|serve inkling2-mini` with model arguments is not a
supported entry point; `strix inspect inkling2-mini` prints metadata
only), so the standalone entry point is `strix-serve`.
`strix-serve --model inkling2-mini --device-count 2 --placement replica
--print-effective-config` prints the same placement as the router. To roll
back, start the router without `--description`, or serve from one node.

Dos nodos: cada nodo ejecuta un `strix-serve` completo, escuchando en su
dirección privada (`--bind`; por defecto escucha en 127.0.0.1 y el otro
nodo no lo alcanza), y `strix-router` reparte las peticiones; cada petición
se resuelve en un solo nodo con la identidad de un nodo. Fija el mismo
backend, perfil, checkpoint y tokenizador en los dos workers (`--profile exact` y, tras
`--`, `--checkpoint ... --tokenizer RUTA --backend hip`; el `--backend` de nivel superior lo
consume el shell y no llega al adaptador), porque el router comprueba el
número de endpoints y el perfil, no el backend, y sin `--tokenizer` cada worker
carga el `tokenizer.json` junto a su propio checkpoint (el cargador solo
comprueba tamaño de vocabulario y EOS). Antes de enviar tráfico, comprueba que
`sha256sum` del checkpoint y del tokenizador coincide en los dos nodos y que la
identidad efectiva de cada worker (backend, perfil; `--print-effective-config`
con la misma línea de comandos) es la misma; solo después comprueba que
peticiones independientes llegan a ambos endpoints. Solo el router ejecuta la
forma `replica`; `strix-serve --model inkling2-mini --device-count 2 --placement replica --
--checkpoint MODEL.safetensors` se rechaza (la CLI `strix` no tiene ruta
nativa para Inkling) y el mismo `host:puerto` repetido no cuenta como dos
réplicas. Para volver atrás, arranca el router sin `--description` o sirve
desde un nodo.

Verify windows (lookup and drafter speculation on the command line) project
and attend each row with its own launches by default. The exact switch
`STRIX_INK_VERIFY_ROWS=1` (`inkling2-mini-verify-rows@1`, off in every profile
column, HIP only; CUDA declines it) projects a window of 2..8 rows through one
pitched rows head launch set, attends its rows in one lanes launch set (each
row at its own position with its one-row geometry) and records the window's
convolution inputs, so a partial commit rebuilds the histories by copies
instead of replaying the kept rows. Every row keeps its one-row arithmetic.
The command line prints `Inkling verify_rows on|off|declined`, the server's
startup line `verify_rows` (served lanes decode one row per step, so the
server's output does not change). It stays opt-in until device identity
against the per-row route.

`STRIX_INK_VERIFY_ROWS=2` adds the lanes pick: greedy picks whose scores
nobody reads (verify windows without a sampled or observing reader, served
cohorts of 2+ lanes whose groups declare no score read) go through the lanes
form of the EXACT screened head (BF16 head only; one read of the head plane
for all lanes, each lane with its one-row token). A later score read of a
served cohort lane recomputes the skipped rows through the per-row head, also
after another slot's verify window or prompt chunk (the rows are materialized
before their head inputs are overwritten); a verify window picked this way has
no later score reader. The tokens are those of the full head; it is a distinct
identity, and the startup lines print `on+lanes-pick`
(`on+lanes-pick-declined` without the BF16 screened head, and on a server with
one admitted slot or on the served CLI route, whose cohorts keep level 1).

On HIP, the command line prefills in chunks of 8192 rows (`--prefill-chunk`),
and a remainder of up to chunk / 8 rows (and more than 16) rides in the final
chunk (`STRIX_INKLING_PREFILL_TAIL=0` turns that off). A HIP server on the
command line's route uses the same schedule (`--prefill-rows` sets the chunk),
reads the whole prompt by prefill and takes the first id from its final row, as
the command line does. While another lane decodes it runs a prompt in 128-row
pieces cut inside those chunks only at 16-row wave tiles and never shorter than
17 rows, where every row keeps its route forms, so the answer does not depend
on other requests. The CUDA command line prefills in 8192-row chunks on the
EXACT profile and in 2048-row chunks on the others. An
explicit `--prefill-rows` without `--slots` also caps the served lanes at that
many rows, as before. The server's startup line prints `served_route` (`off`
or `declined` on the existing route; `on` adds `prefill_chunk`, `prefill_tail`
and `first_id prefill`); the command line prints `strix-prefill-route/v1`.

Under memory pressure on CUDA, a GPU that cannot hold the 8192-row workspaces
of the EXACT default loads with 2048-row chunks instead: the command line
prints `strix-prefill-chunk 8192 -> 2048 (device memory)` (a shorter prompt
names its own size in place of 8192) and the
`strix-prefill-route/v1` line names the 2048-row chunk. EXACT output does not
depend on the chunk, so the generated ids are the same. Only a refused device
allocation selects the smaller chunk; any other load error is reported as is.
An explicit `--prefill-chunk` is never changed, and `--prefill-chunk 2048`
selects the smaller chunk up front. The CUDA server is not affected: it keeps
its own 128-row prompt chunks (`--prefill-rows`).

The greedy head on CUDA is the screened head in every profile (RTX identity
receipts: greedy ids equal the full head, CLI and served); an explicit
`STRIX_INK_SCREENED_HEAD=0` (`inkling2-mini-screened-head@1`) keeps the full
head. CUDA greedy picks that return no score row run the shared EXACT screened head
(`cuda/core/head_screened.cuh`, an 8-bit plane of the BF16 head built at
load): the same token and status as the full head. Steps that return or read
scores (`--logits`, sampled lanes, verify rows, cohort programs) keep the
full head; a sampled served lane reads the scores of its own step, projected
on the first read. A compact head, a plane that cannot be built, or (served) a
plane that would cost a KV slot keeps the full head. HIP keeps its own
screened head (`STRIX_SCREENED_HEAD=0` turns it off) and ignores the switch.
The command line, the gate and the server print `inkling2 head` with
`screened lever`, `full lever-off`, `full build-refused`, `full kv-slot` or,
on HIP, `screened hip-binding` / `full hip-binding`; the server's startup
line carries the same `head` field. When screened, the first screened pick
prints `inkling2 head screened first pick`, the decoder prints `inkling2 head
screened picks N` when it closes, and the CUDA step program's install line
ends with `head=screened`.

## Execution and reuse

| Stage | Shared component | Mini binding |
| --- | --- | --- |
| Weights | `strix-weight` safetensors indexer and packed K16 NVFP4/MXFP8 operand view | 822 text tensor names, extents and dtypes, including quantization sidecars. The loader exposes contiguous code and scale ranges by physical layer and projection for device residency. CPU and device expert tiles decode compact weights into BF16 operands with the checkpoint's gate/up row order. |
| Tied layers | `src/core/looped_layers.hpp` | 24 physical layers, with layers 6â€“17 visited three times for 48 visits |
| Residual and cross-pass messages | `src/core/residual_reference.hpp` | Save the prelude output and block bank. Before each repeat, capture the previous pass's second block input and final output, then restore the prelude. Read four messages from the captured bank and apply rank-64 gates. |
| Attention | `src/attn_core` causal GQA reference, prepared BF16 cache binding, dense HIP prefill and CUDA Flash | 24 query heads, 8 KV heads and dimension 128. Both backends append prepared K/V together and dispatch attention with relative logits and dynamic sinks on the same stream. |
| Short convolutions | `src/core/short_convolution.hpp` and the shared HIP/CUDA device body | Four histories per visit for K, V, post-attention and post-MoE. Full attention uses dynamic K/V taps. Local K/V and both output convolutions use static depthwise taps. The device primitive updates history after the output rows have consumed the previous history. |
| MoE | `strix::moe::RoutePolicy`, activation traits, expert-major plans and grouped BF16 WMMA/MMA | Sigmoid plus correction bias for selection of 8 of 128 experts. Normalize unbiased scores with scale 8. Softcap the two linear factors in SwiGLU and evaluate sigmoid on the raw gate. Both backends select routes, build expert-major descriptors, gather BF16 rows, project around softcapped SwiGLU and combine into token order on a single stream. The original host plan remains a fixture reference. |
| Embeddings, tokenizer and sampling | Shared embedding rows, tokenizer and logit selection | Learned embedding RMS norm, separate output matrix, final tanh softcap 30 and output vocabulary of 200058 tokens |
| Sessions and placement | `include/strix/session`, `include/strix/pair` and `include/strix/dist` | Own the weight binding by physical layer and the persistent state by logical visit. Preserve the two message-bank rows as dependencies at pass boundaries. |
| Vision and drafter | Shared image ops, serving and speculative verification | Vision tensor map and projector. Bind drafter taps and qualify rollback across all visits before enabling speculation. |

The Mini expert layout binds the checkpoint's physical-layer format and
gate/up order. The common Mini MoE dispatch binds model dimensions, BF16
scratch and softcaps, with HIP and CUDA adapters selecting backend tile sizes.
Each attention adapter binds the Mini query/KV head counts, relative logits,
sinks and window to its backend. The session plan fixes cache and history
offsets by logical visit at the largest admitted chunk so single-token decode
uses the same layout as prefill.

The routed MoE stage reuses each of its two temporary buffers after its last
reader. The gather and activation share a BF16 buffer, and the gate/up and
down projections share an FP32 buffer. This requires 13,824 bytes per routed
row, or 108 KiB per token at top-8, plus routing metadata. Choose a bounded
prefill chunk when allocating scratch.

The cache plan uses eight full-context KV allocations and 40 sliding KV rings.
At 1,048,576 positions, BF16 payload is 32 GiB for full attention and 160 MiB
for the sliding rings with one-row writes. For prefill that writes keys before
attention, size each sliding ring to hold the preceding 1023 keys and the
entire chunk. A later staged implementation can keep a 1024-row persistent
ring and expose the new chunk separately. Test prefill/decode parity across
the ring wrap before changing the storage strategy.

The residual controller shares two normalized bank rows across the repeat.
The lag projection reads the original K-by-N checkpoint layout in one launch
and keeps its FP32 accumulator and BF16 result. CUDA also reuses one
large-M activation split for compatible same-input projections.


## Verification

```sh
cmake -S tests/inkling2 -B build/inkling2-host -DCMAKE_BUILD_TYPE=Release
cmake --build build/inkling2-host
ctest --test-dir build/inkling2-host --output-on-failure
```

The root host CTest also registers the Inkling fixtures and includes a test for
the checkpoint inspector's schema and shared reader. The standalone build
runs the header-only fixtures.

`strix-hip-inkling2-mini-attention-test` is built by HIP offline CI. On a
compatible device it exercises the optional relative bias with and without
Split-K, contiguous and wrapped K/V, and Mini's cache-append path.
`strix-hip-inkling2-mini-short-conv-test` checks static and dynamic taps,
including history continuity from prefill into decode.
`strix-hip-inkling2-mini-grouped-test` exercises both expert formats,
interleaved gate/up rows, uneven route groups and Mini's down projection.
The grouped-row and expert-activation fixtures isolate ordering and rounding.
`strix-hip-inkling2-mini-moe-test` compares the full routed stage with a scalar
calculation using synthetic packed weights in both Mini formats. The CUDA
fixture uses the same calculation and packed weights. The CUDA short
convolution fixture likewise shares its data with HIP. The CUDA attention test
covers prepared full/wrapped KV, bias and sinks with both Flash layouts.

```sh
cmake -S . -B build-cuda -DSTRIX_CUDA_CORE=ON -DBUILD_TESTING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target strix-cuda-inkling2-mini-attention-test \
    strix-cuda-inkling2-mini-short-conv-test strix-cuda-inkling2-mini-moe-test
ctest --test-dir build-cuda -L inkling2 -R '^strix-cuda'
```

- [x] Pin the checkpoint and verify the original text tensor inventory.
- [x] Exercise pass restarts, message gates, state identities, score and router policies in host fixtures.
- [x] Execute the full CPU token path, including greedy generation, with a synthetic checkpoint in the published layout.
- [x] Compare CPU next-token logits with an independent calculation on a single-token prompt using the original checkpoint.
- [ ] Expand CPU parity coverage across prompt lengths, decode positions and the sliding-window boundary.
- [x] Build native text entry points and share resident execution between HIP and CUDA.
- [ ] Compare native operators with pinned reference intermediate tensors on device.
- [ ] Validate text prefill, multi-token decode and logits on a single Strix Halo with the official weights.
- [ ] Validate cache rollover, state restore and rollback for all visits.
- [ ] Qualify the integrated HIP text, CLI image and CLI DSpark paths on the original weights.
- [ ] Measure TTFT, prefill, decode and peak memory at 1K, 8K, 32K and 128K. Check long-context admission and continuation.

The [HIP qualification runbook](hip-qualification.md) records the per-gate status, device commands,
profile checks and measurement table.

Tracking #909 for Mini and #908 for Small.
