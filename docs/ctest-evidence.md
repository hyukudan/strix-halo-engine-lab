# Complete suite evidence

`tools/qualification/ctest_evidence.py` reads an existing CTest JUnit report and
an immutable JSON array of expected test names. It starts no build, test process,
model or probe. Freeze the expected names before the campaign; do not derive them
from the successful cases afterward.

Every reported case and its reported duration/diagnostic remains in the receipt.
Missing or extra names, any skipped case and any failed case prevent
`all_requested_tests_passed`. Duplicate names, inconsistent aggregate counts,
unknown statuses and unsupported/malformed XML are rejected. A suite with no
reported cases cannot pass a nonempty expected selection. Output is created
exclusively and cannot replace either source evidence or an existing receipt.

The command exits0 for complete reported success,3 for an incomplete/failed/skipped
suite and2 for malformed inputs. `--stage host|native|model` is a caller declaration,
not hardware observation. File hashes bind the exact parsed buffers. Qualification
and observed native execution stay null: hardware, executable/source identity,
model fidelity and performance require their own campaign evidence.

```text
python tools/qualification/ctest_evidence.py --expected-names campaign-tests.json --junit existing-ctest.xml --stage native --output fresh-ctest-receipt.json
```

This utility is source only until the implementation bundle reaches its scheduled
validation stage. Its fixture source covers skips, absent/extra/duplicate cases,
nested suite totals, contradictory metadata, diagnostic retention and output
preservation; writing those fixtures is not a test result.
