#!/usr/bin/env bash
# buckets.sh BUILD_DIR YARDSTICK OUT_DIR [ROWS...]: the shared grouped MoE expert core (F16 tile, Qwen MoE shapes:
# hidden 2560, expert width 640, 512 experts, top-10, synthetic Q4_K gate/up + Q8_0 down) per mean rows/expert bucket,
# against the vendor FP16 GEMM in strided-batched form (one GEMM per expert, 512 experts, tools/hipblaslt_gemm_yardstick.cpp
# with HB_BATCH). Per bucket: strix-hip-moe-wmma-prefill-tests bench2 under a rocprofv3 kernel trace (gate/up and down
# kernel times), then the attribution tunes 338 (no WMMA) and 594 (no dequantisation) at the two largest buckets
# (diagnostic, bits not exact by design). Rows default 8 16 32 64 128 256; T = rows x 512 / 10 rounded down to 8.
# Env: ROCPROF (default rocprofv3 on PATH, else $ROCM_PATH/bin); ATTRIB=0 skips the attribution tunes; HB_ARMS / HB_RAND
# pass to the yardstick (defaults 3 / 2: D FP32 and D FP16 on full-mantissa data). Writes OUT_DIR/summary.txt (<= 60 lines) and OUT_DIR/buckets.done. Run inside the node's GPU turn (~60 s).
set -u
B=${1:?BUILD_DIR}; Y=${2:?YARDSTICK}; OUT=${3:?OUT_DIR}; shift 3
ROWS=${*:-8 16 32 64 128 256}
mkdir -p "$OUT" || exit 1
rm -f "$OUT/buckets.done" || exit 1
S=$OUT/summary.txt; : > "$S" || exit 1
bin=$(find "$B" -name strix-hip-moe-wmma-prefill-tests -type f -perm -u+x | head -1)
[ -n "$bin" ] || { echo "MISSING strix-hip-moe-wmma-prefill-tests" >> "$S"; touch "$OUT/buckets.done"; exit 1; }
PROF=${ROCPROF:-$(command -v rocprofv3 || echo "${ROCM_PATH:-/opt/rocm}/bin/rocprofv3")}
[ -x "$PROF" ] || { echo "MISSING rocprofv3 (set ROCPROF)" >> "$S"; touch "$OUT/buckets.done"; exit 1; }
clock() { for f in /sys/class/drm/card*/device/pp_dpm_sclk; do [ -r "$f" ] && grep -m1 '\*' "$f" | awk '{print $2}' && return; done; echo "?"; }
fclock() { for f in /sys/class/drm/card*/device/pp_dpm_fclk; do [ -r "$f" ] && grep -m1 '\*' "$f" | awk '{print $2}' && return; done; echo "?"; }
kernels() {  # top MoE kernels of one trace: name, calls, mean us
  st=$(find "$1" -name '*kernel_stats.csv' | head -1)
  [ -n "$st" ] || { echo "  trace unreadable"; return; }
  python3 - "$st" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if 'prefill_f16x' in r['Name']]
for r in sorted(rows, key=lambda r: -float(r['TotalDurationNs']))[:2]:
    n = r['Name'].replace('(anonymous namespace)::', '').split('(')[0].split('::')[-1]
    print('  kernel %-40s calls %4s mean %9.1f us' % (n, r['Calls'], float(r['AverageNs']) / 1e3))
PY
}
run() {  # run LABEL ROWS [env...]
  local label=$1 r=$2; shift 2
  local T=$(( r * 512 / 10 / 8 * 8 )) d=$OUT/trace-$label
  rm -rf "$d"
  echo "== $label rows/expert $r T $T sclk $(clock) fclk $(fclock)" >> "$S"
  env "$@" timeout 30 "$PROF" --kernel-trace --stats -d "$d" -o run --output-format csv -- "$bin" bench2 10 "$T" 12 8 0 4 > "$OUT/$label.log" 2>&1
  grep -E '^bench2' "$OUT/$label.log" | sed -E 's/ \(min.*TOPS/ /' | cut -c1-120 >> "$S"
  kernels "$d" >> "$S"
}
for r in $ROWS; do run "r$r" "$r" BENCH_CORES=0; done
big=$(echo "$ROWS" | tr ' ' '\n' | sort -n | tail -2 | tr '\n' ' ')
[ "${ATTRIB:-1}" = 1 ] && for r in $big; do for tu in 338 594; do run "r$r-tune$tu" "$r" BENCH_CORES=0 BENCH_NOCHECK=1 STRIX_INTPREFILL_MOE_TUNE=$tu; done; done
# Vendor arms: HB_ARMS (default 3 = D FP32 then D FP16), data class HB_RAND (default 2 = full-mantissa, the class of the
# bench's random K-quant blocks; 0 = the old constant bytes, which raise the vendor's sustained clock).
echo "== vendor FP16 strided-batched, 512 experts, arms ${HB_ARMS:-3}, data class ${HB_RAND:-2}, sclk $(clock)" >> "$S"
for r in $ROWS; do
  HB_RAND=${HB_RAND:-2} HB_ARMS=${HB_ARMS:-3} HB_BATCH=512 timeout 20 "$Y" "$r" 2560:1280 640:2560 2>&1 | grep '^hb' | sed -E 's/ algos.*//' >> "$S"
done
touch "$OUT/buckets.done"
