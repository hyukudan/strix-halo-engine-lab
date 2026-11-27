# Best on record

Generated from [the ledger](best_on_record.json). See the [integration guide](README.md).

| Row | tok/s | Prompt tokens | Generated tokens | Requests | Timing | Commit | Date | Receipt | Conditions |
|---|---:|---:|---:|---:|---|---|---|---|---|
| deepseek-v4.1/pair/exact/cli/aggregate-c8 | 113.9 | — | — | 8 | decode | 6a5604bf | 2026-10-26 | [Evidence](../../docs/models/deepseek-v4.1/README.md#measured-results-the-pair) | Eight independent decode rows, rate from step time. |
| deepseek-v4.1/pair/exact/cli/decode-b1 | 39.98 | 15–21 | 256 | 1 | decode | f42edd05e | 2026-11-21 | [Evidence](../../README.md#results) | Short chat, DSpark, earlier published overview. |
| deepseek-v4.1/pair/exact/cli/prefill-2k | 791.9 | 2048 | — | 1 | prefill | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | Earlier published 2K CLI snapshot, prompt class unspecified. |
| deepseek-v4.1/pair/exact/cli/prefill-32k | 1285 | 32768 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Engine prompt benchmark, warm. |
| deepseek-v4.1/pair/exact/cli/prefill-8k | 1165 | 8192 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Engine prompt benchmark, warm. |
| deepseek-v4.1/pair/exact/served/aggregate-c8 | 103.29 | 17–46 | 256 | 8 | wall | ebe642fb | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Whole-run median, grouped short-prompt admission. |
| deepseek-v4.1/pair/exact/served/decode-b1 | 57.7 | 344 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | DSpark, short chat. |
| deepseek-v4.1/pair/exact/served/prefill-2k | 782 | 1971 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/exact/served/prefill-32k | 1273 | 30975 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/exact/served/prefill-8k | 1111 | 7994 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/fast/cli/prefill-32k | 1981 | 32768 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/band-edges.md) | D1 plus R256 public band receipt, inherited in TURBO. |
| deepseek-v4.1/pair/fast/cli/prefill-8k | 1855 | 8192 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/band-edges.md) | D1 plus R256 public band receipt, inherited in TURBO. |
| deepseek-v4.1/pair/fast/served/decode-b1 | 53.7 | 344 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | DSpark, short chat. |
| deepseek-v4.1/pair/fast/served/prefill-2k | 969 | 1971 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/fast/served/prefill-32k | 1708 | 30975 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/fast/served/prefill-8k | 1588 | 7994 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/turbo/cli/prefill-32k | 1981 | 32768 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/band-edges.md) | D1 plus R256 public band receipt, inherited in TURBO. |
| deepseek-v4.1/pair/turbo/cli/prefill-8k | 1855 | 8192 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/band-edges.md) | D1 plus R256 public band receipt, inherited in TURBO. |
| deepseek-v4.1/pair/turbo/served/prefill-32k | 1710 | 30975 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| deepseek-v4.1/pair/turbo/served/prefill-8k | 1591 | 7994 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/exact/cli/aggregate-c8 | 71.3 | — | — | 8 | decode | ec76191f2 | 2026-11-04 | [Evidence](../../docs/models/glm-5.3/README.md#measured-results-the-pair) | R8 device batch, rate from step time. |
| glm-5.3/pair/exact/cli/decode-b1 | 37.1 | — | 64–128 | 1 | decode | 63dcea5e (receipt) | 2026-11-15 | [Evidence](../../docs/models/glm-5.3/README.md#measured-results-the-pair) | Chat, default MTP self-speculation. |
| glm-5.3/pair/exact/cli/prefill-32k | 642 | 32768 | — | 1 | prefill | 2c46c63f | 2026-11-15 | [Evidence](../../docs/models/glm-5.3/README.md#measured-results-the-pair) | Warm second prefill. |
| glm-5.3/pair/exact/cli/prefill-8k | 664 | 8192 | — | 1 | prefill | 2c46c63f | 2026-11-15 | [Evidence](../../docs/models/glm-5.3/README.md#measured-results-the-pair) | Warm second prefill. |
| glm-5.3/pair/exact/served/decode-b1 | 42.8 | 362 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/exact/served/prefill-2k | 531 | 2037 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/exact/served/prefill-32k | 578 | 31258 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/exact/served/prefill-8k | 629 | 8108 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/fast/cli/prefill-2k | 734.41 | 2049 | — | 1 | ttft | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/band-edges.md) | Qualified lower edge, 2,049 tokens in 2.79 seconds. |
| glm-5.3/pair/fast/cli/prefill-8k | 1000 | 8192 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#glm-pf-wmma-x) | Warm WMMA bundle receipt for the default band. Free-running gate re-score is pending a policy decision. |
| glm-5.3/pair/fast/served/decode-b1 | 41.1 | 362 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/fast/served/prefill-2k | 530 | 2037 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/fast/served/prefill-32k | 568 | 31258 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/fast/served/prefill-8k | 628 | 8108 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/turbo/cli/prefill-8k | 1000 | 8192 | — | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#glm-pf-wmma-x) | Warm WMMA bundle receipt for the default band. Free-running gate re-score is pending a policy decision. |
| glm-5.3/pair/turbo/served/decode-b1 | 41.3 | 362 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/turbo/served/prefill-2k | 718 | 2037 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/turbo/served/prefill-32k | 879 | 31258 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| glm-5.3/pair/turbo/served/prefill-8k | 949 | 8108 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| hy4-flash/one/exact/cli/decode-b1 | 15.75 | 2048 | 64 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | BASE, plain. |
| hy4-flash/one/exact/cli/prefill-2k | 15.2 | 2048 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | BASE position by position. |
| hy4-flash/one/fast/cli/decode-b1 | 17 | 2048 | 64 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Manual 2048-row prompt windows. |
| hy4-flash/one/fast/cli/prefill-2k | 810 | 2048 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | 8K default, 2K and 32K explicit 2048 rows. |
| hy4-flash/one/fast/cli/prefill-32k | 699 | 32500 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | 8K default, 2K and 32K explicit 2048 rows. |
| hy4-flash/one/fast/cli/prefill-8k | 782 | 8192 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | 8K default, 2K and 32K explicit 2048 rows. |
| hy4-flash/one/turbo/cli/decode-b1 | 15.65 | 8192 | 64 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| hy4-flash/one/turbo/cli/prefill-8k | 739 | 8192 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| hy4-flash/pair/exact/cli/decode-b1 | 22.42 | 1024 | 64 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Tensor parallel, plain. |
| hy4-flash/pair/fast/cli/decode-b1 | 22.64 | 1024 | 64 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Tensor parallel, plain. |
| hy4-flash/pair/fast/cli/prefill-32k | 1328 | 32500 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | USB4 layer split. |
| hy4-flash/pair/fast/cli/prefill-8k | 1387 | 8192 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | USB4 layer split. |
| inkling2-mini/one/exact/cli/prefill-32k | 1939 | 32760 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/exact/cli/prefill-8k | 2181 | 8236 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/exact/served/decode-b1 | 23.6 | 396 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Short chat, published build resolves to the same route. |
| inkling2-mini/one/exact/served/prefill-2k | 591 | 2280 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Published build resolves to the same route. |
| inkling2-mini/one/exact/served/prefill-32k | 1890 | 32758 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/exact/served/prefill-8k | 2037 | 8234 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/fast/served/decode-b1 | 23.6 | 396 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Short chat, published build resolves to the same route. |
| inkling2-mini/one/fast/served/prefill-2k | 591 | 2280 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Published build resolves to the same route. |
| inkling2-mini/one/fast/served/prefill-32k | 578 | 35170 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Published build resolves to the same route. |
| inkling2-mini/one/fast/served/prefill-8k | 601 | 9116 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Published build resolves to the same route. |
| inkling2-mini/one/turbo/cli/prefill-32k | 2072 | 32760 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/turbo/cli/prefill-8k | 2380 | 8236 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/turbo/served/decode-b1 | 23.6 | 396 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Short chat, published build resolves to the same route. |
| inkling2-mini/one/turbo/served/prefill-2k | 591 | 2280 | 64 | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Published build resolves to the same route. |
| inkling2-mini/one/turbo/served/prefill-32k | 2053 | 32758 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| inkling2-mini/one/turbo/served/prefill-8k | 2229 | 8234 | — | 1 | prefill | 1f0b9287 (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md#inkling2-moe-iu4-turbo) | Public catalog qualification medians for the integrated ladder; 32K is inside the ladder. |
| minimax-m3.1/pair/exact/cli/aggregate-c8 | 61.4 | 1024 | — | 8 | decode | ff3df9a6 (receipt) | 2026-10-26 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | Eight independent decode rows, rate from step time; public release revision. |
| minimax-m3.1/pair/exact/cli/decode-b1 | 23.28 | 2048 | 256 | 1 | decode | f42edd05e | 2026-11-21 | [Evidence](../../README.md#results) | Greedy, code. |
| minimax-m3.1/pair/exact/cli/prefill-32k | 644 | 32768 | — | 1 | prefill | 515ea90b | 2026-11-15 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | First session in a fresh process. |
| minimax-m3.1/pair/exact/cli/prefill-8k | 669 | 8192 | — | 1 | prefill | ae06f9466 | 2026-11-14 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | Warm second session. |
| minimax-m3.1/pair/exact/served/aggregate-c8 | 56.1 | 190 | 256 | 8 | wall | e6ce78571 | 2026-11-02 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | Best repeat of eight distinct prompts, length approximate. |
| minimax-m3.1/pair/exact/served/decode-b1 | 27 | 508 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/exact/served/prefill-32k | 590 | 30179 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/exact/served/prefill-8k | 667 | 7791 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/fast/cli/prefill-32k | 682 | 32768 | — | 1 | prefill | 515ea90b | 2026-11-15 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | Qualified layers 40–59, cold. |
| minimax-m3.1/pair/fast/cli/prefill-8k | 687 | 8192 | — | 1 | prefill | 515ea90b | 2026-11-15 | [Evidence](../../docs/models/minimax-m3.1/README.md#measured-results) | Qualified layers 40–59, cold. |
| minimax-m3.1/pair/fast/served/decode-b1 | 27.1 | 508 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/fast/served/prefill-32k | 640 | 30179 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/fast/served/prefill-8k | 709 | 7791 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/turbo/served/decode-b1 | 26.7 | 508 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/turbo/served/prefill-32k | 674 | 30179 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| minimax-m3.1/pair/turbo/served/prefill-8k | 709 | 7791 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| nemotron-3.5-bolt-e4b/one/exact/cli/aggregate-c8 | 237.9 | — | 256 | 8 | decode | cb207e9d (receipt) | 2026-11-02 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Plain benchmark decode, no drafting. |
| nemotron-3.5-bolt-e4b/one/exact/cli/decode-b1 | 91.9 | — | 256 | 1 | decode | cb207e9d (receipt) | 2026-11-02 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Shipped MTP plus lookup, benchmark. |
| nemotron-3.5-bolt-e4b/one/exact/cli/prefill-2k | 539 | 2048 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/exact/cli/prefill-32k | 420 | 32768 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/exact/cli/prefill-8k | 509 | 8192 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/exact/served/aggregate-c8 | 187.4 | — | 128 | 8 | wall | d7069831d | 2026-11-03 | [Evidence](../../docs/performance.md#current-snapshot) | Short chat, prefix cache off. |
| nemotron-3.5-bolt-e4b/one/exact/served/decode-b1 | 92.1 | — | 128 | 1 | wall | d7069831d | 2026-11-03 | [Evidence](../../docs/performance.md#current-snapshot) | End-to-end short chat, prefix cache off. |
| nemotron-3.5-bolt-e4b/one/fast/cli/aggregate-c8 | 340.6 | — | 256 | 8 | decode | cb207e9d (receipt) | 2026-11-02 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Plain benchmark decode, no drafting. |
| nemotron-3.5-bolt-e4b/one/fast/cli/decode-b1 | 99.5 | — | 256 | 1 | decode | cb207e9d (receipt) | 2026-11-02 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Shipped MTP plus lookup, benchmark. |
| nemotron-3.5-bolt-e4b/one/fast/cli/prefill-2k | 2880 | 2048 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/fast/cli/prefill-32k | 2336 | 32768 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/fast/cli/prefill-8k | 2780 | 8192 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/turbo/cli/decode-b1 | 45.34 | 8192 | — | 1 | decode | 1eba63b4 (receipt) | 2026-11-27 | [Evidence](../../docs/optimizations.md#g4-int8-global-kv-shadow) | Published TURBO attention band, drafting off. |
| nemotron-3.5-bolt-e4b/one/turbo/cli/prefill-2k | 2880 | 2048 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/turbo/cli/prefill-32k | 2336 | 32768 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/one/turbo/cli/prefill-8k | 2780 | 8192 | — | 1 | prefill | e80432ce (receipt) | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#measured-results) | Historical native prompt timing, Q4_K_M. |
| nemotron-3.5-bolt-e4b/pair/fast/cli/prefill-2k | 4132 | 2048 | — | 1 | prefill | 51b42906e | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#two-nodes-pipeline-prompt-prefill-opt-in) | Layer pipeline, fresh process, drafting off. |
| nemotron-3.5-bolt-e4b/pair/fast/cli/prefill-32k | 3965 | 32768 | — | 1 | prefill | 51b42906e | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#two-nodes-pipeline-prompt-prefill-opt-in) | Layer pipeline, fresh process, drafting off. |
| nemotron-3.5-bolt-e4b/pair/fast/cli/prefill-8k | 4804 | 8192 | — | 1 | prefill | 51b42906e | 2026-11-10 | [Evidence](../../docs/models/nemotron-3.5-bolt-e4b/README.md#two-nodes-pipeline-prompt-prefill-opt-in) | Layer pipeline, fresh process, drafting off. |
| qwen3.8-27b/one/exact/cli/decode-b1 | 35.3 | 2048 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | MTP default after 2K. |
| qwen3.8-27b/one/exact/cli/prefill-2k | 644 | 2048 | — | 1 | prefill | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | Earlier published CLI snapshot, plain decode; FAST explicitly used 512 rows. |
| qwen3.8-27b/one/exact/cli/prefill-32k | 517 | 32768 | — | 1 | prefill | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | Earlier published CLI snapshot, real text and plain decode. |
| qwen3.8-27b/one/exact/cli/prefill-8k | 606 | 8192 | — | 1 | prefill | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | Earlier published CLI snapshot, plain decode; FAST explicitly used 512 rows. |
| qwen3.8-27b/one/fast/cli/decode-b1 | 35.2 | 2048 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | MTP default after 2K. |
| qwen3.8-27b/one/fast/cli/prefill-2k | 728.83 | 2048 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-and-speculative-apps) | Qualified 512-row window: 2K/8K fastest profile arm, 32K explicit window. |
| qwen3.8-27b/one/fast/cli/prefill-32k | 532.81 | 32768 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/optimizations.md) | Published default band, 32,768 tokens in 61.5 seconds. |
| qwen3.8-27b/one/fast/cli/prefill-8k | 677.02 | 8192 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-and-speculative-apps) | Qualified 512-row window: 2K/8K fastest profile arm, 32K explicit window. |
| qwen3.8-27b/one/turbo/cli/prefill-2k | 728.83 | 2048 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-and-speculative-apps) | TURBO inherits the same default windows in these admitted bands. |
| qwen3.8-27b/one/turbo/cli/prefill-32k | 532.81 | 32768 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-and-speculative-apps) | TURBO inherits the same default windows in these admitted bands. |
| qwen3.8-27b/one/turbo/cli/prefill-8k | 677.02 | 8192 | 1 | 1 | prefill | 3d9c562e (receipt) | 2026-11-26 | [Evidence](../../docs/models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-and-speculative-apps) | TURBO inherits the same default windows in these admitted bands. |
| qwen3.8-27b/pair/exact/cli/decode-b1 | 45.4 | 2048 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Tensor parallel, MTP default. |
| qwen3.8-27b/pair/fast/cli/decode-b1 | 45.5 | 2048 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Tensor parallel, MTP default. |
| qwen3.8/one/exact/cli/decode-b1 | 51 | 32768 | — | 1 | decode | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | MTP with automatic prefill, earlier published overview. |
| qwen3.8/one/exact/cli/prefill-2k | 1435 | 2048 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| qwen3.8/one/exact/cli/prefill-32k | 1856 | 32768 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Warm page cache, real text. |
| qwen3.8/one/exact/cli/prefill-8k | 1885 | 8192 | — | 1 | prefill | ae06f9466 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Warm page cache, real text. |
| qwen3.8/one/exact/served/decode-b1 | 40.2 | — | — | 1 | decode | d3a89ae16 | 2026-11-21 | [Evidence](../../README.md#results) | MTP with automatic windows, prose, earlier published overview. |
| qwen3.8/one/fast/cli/decode-b1 | 21.9 | 2048–32500 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Plain decode, bundle absent. |
| qwen3.8/one/fast/cli/prefill-2k | 1673 | 2048 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| qwen3.8/one/fast/cli/prefill-32k | 1975 | 32768 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Warm page cache, real text. |
| qwen3.8/one/fast/cli/prefill-8k | 2008 | 8192 | — | 1 | prefill | ae06f9466 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | Warm page cache, real text. |
| qwen3.8/one/turbo/cli/decode-b1 | 21.8 | 2048–32500 | 128 | 1 | decode | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) | Plain decode, bundle absent. |
| qwen3.8/one/turbo/cli/prefill-2k | 1188 | 2048 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| qwen3.8/one/turbo/cli/prefill-32k | 1665 | 32500 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../README.md#results) |  |
| qwen3.8/one/turbo/cli/prefill-8k | 2080.05 | 8192 | — | 1 | prefill | ca2de6ae8 | 2026-11-25 | [Evidence](../../docs/band-edges.md) | Main cap9 re-measure, 8,192 tokens in 3.9374 seconds. |
| qwen3.8/pair/exact/cli/prefill-32k | 2968 | 32768 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | USB4 state handoff; 8K forced pipeline. |
| qwen3.8/pair/exact/cli/prefill-8k | 2233 | 8192 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | USB4 state handoff; 8K forced pipeline. |
| qwen3.8/pair/fast/cli/prefill-32k | 3191 | 32768 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | USB4 state handoff; 8K forced pipeline. |
| qwen3.8/pair/fast/cli/prefill-8k | 2389 | 8192 | — | 1 | prefill | e5886cb79 | 2026-11-14 | [Evidence](../../docs/performance.md#current-snapshot) | USB4 state handoff; 8K forced pipeline. |
