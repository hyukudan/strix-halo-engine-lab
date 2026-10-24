# DeepSeek V4.1 Flash: model contract (host only)

`include/strix/deepseek41_model.hpp` admits the DeepSeek V4.1 Flash 1017 GGUF (`general.architecture =
deepseek41`, 1,046 tensors) and its speculative draft (`deepseek41-dspark`, 81 tensors). Nothing here executes.

- **Configuration.** The upstream configuration is embedded as JSON in `deepseek41.config`. It is parsed strictly
  (duplicate keys, trailing bytes, non-integer sizes are errors) and every dedicated `deepseek41.*` key must agree
  with it. `compress_ratios` lists the target layers in the metadata and the target layers plus the draft stages
  (which must be 0, sliding window only) in the JSON.
- **Attention organisation per layer** (`layer_attributes`): compress ratio (0 window only, 2 gated pooling, 1 plain
  projection), KV sources (own compressor and index keys), the source a layer reads, index sources and the layer whose
  top-k a layer reuses, the two-level candidate source and its users, Engram layers. All derived from the config.
- **Engram contract.** Each position hashes the (max_ngram_size − 1) n-grams ending there over n_heads heads; every
  (layer, n-gram size, head) owns a prime-sized bucket range, the next unused prime above `engram_vocab_size − 1`.
  The admitted file has 24 rows of 264 bytes (256 FP8 E4M3 + 8 E8M0 scales) per token and Engram layer. The checks:
  bucket primes follow that rule and sum to the table rows, multipliers are odd and bounded so that
  `token × multiplier` fits int64, the compressed-token map covers the vocabulary and every compressed id.
- **Inventory.** `expected_target_tensors` / `expected_draft_tensors` generate every name, GGUF shape and storage type
  from the configuration; a file must match it exactly (no missing, extra or mismatched tensor). Roles and residency
  go into a `ModelManifest`: the Engram tables are file-backed candidates, everything else resident.
- **Admission** is by the raw file's byte count and SHA-256 (`pinned_files`, `admit_file`), source revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- `strix-deepseek41-inspect TARGET.gguf [--draft DRAFT.gguf] [--hash]` prints the derived contract as JSON and, with
  `--hash`, hashes the files and applies the admission. `strix-deepseek41-model-tests` runs synthetic fixtures and,
  with `STRIX_DEEPSEEK41_GGUF` / `STRIX_DEEPSEEK41_DRAFT_GGUF` set, validates the real headers.
