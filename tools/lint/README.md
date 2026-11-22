# Shared-core advisory lints

```sh
python -B tools/lint/check.py --report build/lint-report.txt
```

`layering` reports model-header, SDK and upward include edges, crossings between the two CUDA kernel homes, and model-local kernels and conventionally named resolvers, planners and loaders. `no_lab_constants` checks shared C++ source and headers for model references, fixed topology and link values, and literal host, path and network references. These are source-text checks over `src`, `include/strix` and `apps`. They do not expand the preprocessor or infer topology values from arbitrary expressions.

Findings carry a source line, rule, and baseline or new status. Reports redact literal machine values and remain advisory. The allow-list stores counts per source file, rule and finding fingerprint so line movement preserves baseline matches and duplicate findings consume separate entries.

To reproduce the allow-list from the original local Git revision, run `python -B tools/lint/check.py --baseline-ref 2303d1934`. The command reads source blobs at that revision.
