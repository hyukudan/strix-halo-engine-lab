# Band-edge timing campaign

The [generated ladder inventory](../band-edges.md) lists the nominal and effective edges, the measured points, and the separate quality receipts. Generate a device plan from the same data.

```sh
bash scripts/bench/bench.sh campaign band-edges plan
bash scripts/bench/bench.sh campaign band-edges plan --row deepseek41-pair/d1_prefill/4096 --out band-edge-receipts
```

Each row gets its nominal lower edge, integer midpoint and upper edge. The fixed GLM key appears in the inventory for reference and has no band edges to sweep. Each point is a serialized ABBA block. FAST compares the candidate with EXACT. TURBO compares the candidate with FAST. Where the source measured an increment over a composition, the plan keeps that composition in both arms. DeepSeek replay compares D1 with D1 plus replay. Qwen cap9 and Inkling IU4 compare FAST with FAST plus the candidate.

On the device, put the checked-out executables on `PATH` and set `BAND_MODEL`, `BAND_BUILD_ID`, `BAND_WEIGHTS_ID`, `BAND_DEVICE_PRODUCT` and `BAND_POWER_STATE` to the model asset path, binary build or hash, weights or manifest hash, device product, and power state. Inkling also needs `BAND_TOKENIZER`. Use the same clean environment and model, power and topology for every arm. Use the matching tokenizer to create `prompts/N.ids` with exactly N plain-text token IDs in comma-separated or whitespace-separated form. Keep the exact same IDs for all four turns at a point. Use separate held-out texts for quality gates.

Run the printed resident commands in order. On pair engines, use a shell on each rank with the same settings and record both logs. Start rank 1 before rank 0, then check readiness on rank 0 before sampling. The client verifies the local resident command and its explicit overrides through `/proc` on Linux. Reuse the same resident process for consecutive B turns when convenient. The Qwen 27B HIP band exists in the native decode app and uses its one-token, MTP-off CLI. Use the exact same input and capacity on both arms.

Each sample has a 115-second deadline and up to five seconds for termination. Resident requests use the already-loaded server and fresh prefill. The client kills the local server on timeout, so terminate rank 1 before moving to the next turn. Startup and asset loading for resident servers are outside the timing window. Qwen 27B's historical EXACT 24,576-token control took over two minutes, so the bounded plan can report an incomplete point there.

```sh
bash scripts/bench/bench.sh campaign band-edges report --out band-edge-receipts > band-timing.json
```

The report re-parses raw outputs, checks prompt and artifact identities, and requires both paired ratios to exceed the row's noise margin. `KEEP` covers three winning points. `RAISE-LOWER-EDGE-TO n` identifies the first prompt of the winning suffix. An edge above the upper bound means the row should be disabled. A timed-out, refused, mismatched or missing result gives `INCOMPLETE` and a nonzero exit. These receipts are timing evidence. Any change to default admission still requires the applicable quality scope, recipe and implementation identity to match.

Regenerate the inventory after editing a ladder table.

```sh
cmake --build build --target strix-band-inventory
build/strix-band-inventory json > docs/band-edges.json
build/strix-band-inventory markdown > docs/band-edges.md
ctest --test-dir build -R strix-band-inventory-current --output-on-failure
```
