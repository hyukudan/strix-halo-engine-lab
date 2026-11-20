# Hy4-flash gate kits (G0..G5)

Thin bindings of the shared runner (`scripts/bench/common`: ticket.sh, runlib.sh, run1.sh, receipt.py, abab.py) and the
shared FAST logit gate (`tools/strix-gate/fast_logit_gate.py`). No model path, host name or device is built in: every
path comes from arguments, the peer from `KIT_PEER`.

| rung | kit | runs where | pass rule |
|---|---|---|---|
| G0 | `g0-host.sh --build DIR [--wfmt-build DIR] [--pack P1] [--tokenizer BIN]` | host (no GPU) | every committed row of `tests/hy4_fixtures/g0-tests.tsv` PASS + fixtures PASS |
| G1 | `ticket.sh OUT hy4/g1-ops --bin OPS_BIN` | one GPU | every case of `tests/hy4_fixtures/g1-cases.tsv` PASS in its declared class |
| G2 | `ticket.sh OUT hy4/g2-identity --bin BIN --model P1 --prompts DIR --ref REF` | one node | 32/32 greedy ids per prompt; exact class also equal first-token logits FNV |
| G3 | `KIT_PEER=.. ticket.sh OUT hy4/g3-pair ...` (same arguments) | pair | same lines as G2 (one oracle for 1 and N ranks) |
| G4 | `fast_logit_gate.py seal/compare` on teacher-forced dumps | host reducer | KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01 (abs) |
| G5 | `ticket.sh OUT hy4/g5-prefill ...`, `common/serve-decode.sh` | pair / one node | receipts only (warm + cold, ABAB x3, clocks per token) |

`check_fixtures.py` checks the committed fixtures (`check`), exports the prompt ids in the engine formats (`export`)
and compares identity lines (`identity`); `check_fixtures.py self-test` covers its own invariants.
The engine output lines these kits parse are specified next to each kit's usage text.
