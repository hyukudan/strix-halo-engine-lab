# Hy4-flash gate fixtures

Data for the qualification ladder G0..G5 (kits in `scripts/bench/hy4`). Model data only: no code, no host or model paths.

| file | rung | content |
|---|---|---|
| `prompts.json` | G2, G3, G4 | gate prompts gate-en / gate-zh: original text, raw completion (no chat template, no BOS: the pack declares `add_bos_token = false`) |
| `<id>.ids` | G2, G3 | prompt ids, one decimal id per line, from the shared tokenizer (`strix-tokenize PART1.gguf encode FILE`) |
| `manifest.tsv` | all | every fixture: rung, file, producer, format, exactness class, status (committed / pending), sha256 |
| `g0-tests.tsv` | G0 | host tests of the shared contracts the model binds (committed / pending) |
| `g1-cases.tsv` | G1 | device op cases vs the host contract oracle, declared class (bitwise / ids-equal / tol:<contract>) |
| `ref/oracle-identity.txt` | G2, G3 | pending: the CPU oracle's identity lines (hy4-host-ref-v1) for the gate prompts |

`python3 -I scripts/bench/hy4/check_fixtures.py check [--tokenize-cmd 'strix-tokenize PART1.gguf encode']` verifies the ids
(length 1..32, inside the vocabulary of 120,832, sha256 equal to the manifest, optionally re-encoded from the text);
`export --out DIR` writes the `.u32` / `.csv` / `.ids` forms the engines read. The sha256 of an ids file is the sha256 of
its exact bytes (one id per line, final newline).
