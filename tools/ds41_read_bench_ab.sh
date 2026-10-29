#!/usr/bin/env bash
# Keep the output directory private: these are timing receipts, not publication inputs.
set -euo pipefail
: "${BENCH:?set BENCH to strix-ds41-read-bench}"
: "${MODEL:?set MODEL to the GGUF}"
: "${DIRS:?set DIRS to the secondary directory list}"
OUT=${OUT:-read-bench-results}
PYTHON=${PYTHON:-python}
mkdir -p "$OUT"
export STRIX_WEIGHT_VERIFY=always
for mode in prefill units engram; do
    for rep in 1 2 3; do
        for arm in A B; do
            dirs=""
            [[ $arm == A ]] || dirs=$DIRS
            "$BENCH" "$MODEL" "$@" --mode "$mode" --weight-dirs "$dirs" --weight-verify always \
                --preload-state "${PRELOAD_STATE:-uncontrolled}" --json "$OUT/$mode-$rep-$arm.json"
        done
    done
done
"$PYTHON" - "$OUT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
print('mode      arm  median GB/s  median seconds  direct masks  location bytes (median)')
for mode in ('prefill', 'units', 'engram'):
    for arm in ('A', 'B'):
        rows = [json.loads((root / f'{mode}-{rep}-{arm}.json').read_text()) for rep in (1, 2, 3)]
        locations = max(len(r['location_bytes']) for r in rows)
        split = [int(statistics.median(r['location_bytes'][i] if i < len(r['location_bytes']) else 0 for r in rows))
                 for i in range(locations)]
        print(f"{mode:9} {arm} {statistics.median(r['gb_s'] for r in rows):12.4f}"
              f" {statistics.median(r['seconds'] for r in rows):15.4f}"
              f" {str(sorted(set(r['direct_mask'] for r in rows))):13} {split}")
        if arm == 'B':
            for r in rows:
                s = r['session']
                if not r['multi'] or s is None or s['fallback_bytes'] or s['verify_fail'] or s['disabled_mask']:
                    raise SystemExit('B did not serve a healthy multi-location session; receipts are not qualified')
                if r['direct_mask'] != (1 << len(r['location_bytes'])) - 1:
                    raise SystemExit('buffered location: receipts are not qualified')
PY
