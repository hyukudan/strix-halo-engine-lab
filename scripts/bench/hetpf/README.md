# hetpf: DeepSeek V4.1 CUDA prefill -> pair decode hand-off

One measurement of the mixed route against both devices alone, with the same prompt and the same request shape.

| arm | route | TTFT |
|---|---|---|
| A | `strix-cuda-ds41-prefill-service` prefills, exports the attention state and POSTs it to rank 0's `/admin/import-state` (rank 0 relays it to rank 1, both import); a streamed `/v1/completions` on the pair continues it | service wall (prefill, export, sha256, transfer, import) + pair first token |
| B | the pair prefills the same ids itself (`exact` or `fast` = `STRIX_DS41_FAST_PREFILL=1`, per serve process) | first streamed token |
| R | the CUDA service alone (its `next_token` is the first token) | queue + prefill |

Every request asks for 65 tokens (the first + 64 decode steps). Each serve process runs at least two requests of an arm:
the first is reported as cold, the last as warm, never mixed. Arms alternate inside a process (`A,B,A,B`, then `B,A,B,A`).

Files:
- `link_probe.py`: `serve` on the node, `probe` on the host: application-level RTT (1-byte TCP ping-pong) and useful
  upload/download throughput; `--state-bytes` predicts the state's transfer time.
- `pair_serve.sh OUT MODE HOLD`: one pair serve process inside a pair GPU ticket on rank 0 (rank 1 over ssh); writes
  `OUT/ready`, stops on `OUT/stop` or after HOLD s; samples GPU busy and sclk on both nodes.
- `het_ab.py`: the client (arms A/B/R), one JSON line per request; reuses `tools/ds41_hetero_import_driver.py`'s `Server`.
- `run-pc.sh OUT`: the whole run from the CUDA host (link probe, service, R, one ticket per `TICKETS` entry, summary).
  Hosts, paths, the ticket command and the device come from the environment (see its header); nothing is hard-coded.
- `summarize.py`: `OUT/summary.txt`: TTFT cold/warm per arm, A's phase breakdown, decode tok/s, Mclk/token (busy x
  clock x time / tokens) for the CUDA prefill, the pair prefill and the pair decode, continuation prefixes against the
  first exact B, state sha256 and next token.

Gate (declared before measuring): the pair checks the state's sha256 on import and rank 1 checks its digest; the A
continuation must start with the service's `next_token` and be taken without a pair prefill (`continued`); continuations
are compared with the pair's own exact and FAST ones. The CUDA exact prefill is not bitwise equal to the pair's, so A is
a mixed route: equality with the pair is never required, a KL/dNLL check against the pair exact reference is.
