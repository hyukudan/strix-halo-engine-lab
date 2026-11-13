# Audit 560 host adapter regressions

Configure this small test project and build Release (no HIP or full engine build):

```sh
cmake -S tests/audit560_host -B build/audit560
cmake --build build/audit560 --config Release
ctest --test-dir build/audit560 -C Release --output-on-failure
```

The generator extracts the actual serving adapter's cohort call, prefix eligibility,
planning, execution and accounting sections. Tests compile those sections against
deterministic host stand-ins and link the production shared step planner and cohort
partitioner. Stand-ins model the runtime's minimum physical rows and eight-policy
controller limit; these tests do not execute HIP, sampling or HTTP.

While the fix is uncommitted, configure with `-DAUDIT560_ORIGINAL_CONTROL=ON` to
also compile the original HEAD sections. All four controls must fail their respective
regressions (CTest `WILL_FAIL`); the four fixed cases must pass. This option defaults
OFF so the normal tests remain useful after committing the fix.

Native follow-up qualification remains required: concurrent sampled HTTP completion,
policy selection, latency under mixed load, and prefix hits with restored-token witnesses
plus complete output/logit/state equality against OFF. Host success establishes none of
those numerical or performance claims.
