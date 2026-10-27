#!/usr/bin/env bash
set -euo pipefail
# Builds and CPU-only admission gates; never runs GPU work. Invoke inside qprefill-rocm at /src.
mkdir -p build-ple-batch
CXX=/opt/rocm/bin/amdclang++
flags=(-O3 -std=gnu++20 --offload-arch=gfx1151 -Iinclude -Werror -Rpass-analysis=kernel-resource-usage)
"$CXX" "${flags[@]}" -x hip tests/ple_batch_fixture.hip -o build-ple-batch/ple_batch_fixture >build-ple-batch/fixture.log 2>&1
"$CXX" "${flags[@]}" -Wall -Wextra -Wpedantic -Wconversion -Wshadow -include include/strix/hip_runtime_telemetry.hpp -x hip src/hip_ple_launcher.hip -c -o build-ple-batch/engine.o >build-ple-batch/engine.log 2>&1

build-ple-batch/ple_batch_fixture --cpu-only
build-ple-batch/ple_batch_fixture --admission-only 1
for override in STRIX_Q8_ROWS_WIDE_KERNEL=1 STRIX_Q8_ROWS_WIDE_STAGE=1 STRIX_Q8_ROWS_LANE=0 STRIX_Q8_ROWS_LANE_MIN_OUTPUTS=20000 STRIX_Q8_ROWS_LANE_WAVES=8 STRIX_Q8_ROWS_LANE_STAGE=4 STRIX_Q8_ROWS_LANE_PREFETCH=0 STRIX_Q8_ROWS_LANE_EXP=1; do
    env "$override" build-ple-batch/ple_batch_fixture --admission-only 0
done
