# Execution receipt v1

`strix::receipt::Receipt` holds the nine CORE-SPEC receipt sections. `write_json` validates the typed values and emits one UTF-8 JSON object. C++ callers link the SDK-free `strix_serve_json` library.

`null` records an unobserved section or measurement. An observed section can contain an empty list. Numerical identity reuses the existing eight-field identity and hash. An empty component records an unknown value. The route fingerprint records the effective route and has its own field. `map_legacy_identity` preserves identity and provenance from current engine and route records, including their existing line framing.

```sh
python -B tools/receipt-schema/check.py receipt.json
```

The checker reads the bundled JSON Schema and verifies the numerical hash and cross-field counts. Each valid input prints `PASS`. Invalid input gives a field diagnostic and exit status 1. Receipt producers supply the measurements and gate record IDs.
