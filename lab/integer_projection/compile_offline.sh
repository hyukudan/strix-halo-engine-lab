#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: compile_offline.sh PRIVATE_OUTPUT_DIRECTORY' >&2; exit 2; fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$source_dir/../.." && pwd)
output=$(realpath -m -- "$1")
case "$output/" in "$repo/"*) echo 'Compiler output must remain outside the source checkout.' >&2; exit 2;; esac
ancestor=$output
while true; do
    if [[ -e "$ancestor/.git" ]]; then echo 'Compiler output must remain outside every Git checkout.' >&2; exit 2; fi
    [[ $ancestor == / ]] && break
    ancestor=$(dirname -- "$ancestor")
done
if [[ -e $output || -L $output ]]; then echo 'Compiler output requires a new, unused directory.' >&2; exit 2; fi
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
compiler=${HIPCC:-/opt/rocm/bin/hipcc}
llvm=${ROCM_LLVM:-/opt/rocm/llvm/bin}
"$compiler" --version > "$output/compiler.txt"
sha256sum "$source_dir/integer_projection.hip" "$repo/include/strix/integer_projection.hpp" \
    "$repo/include/strix/integer_projection_execution.hpp" "$repo/include/strix/hip_integer_projection.hpp" \
    "$repo/include/strix/low_precision.hpp" "$repo/include/strix/iq4_xs.hpp" \
    "$repo/src/integer_projection.cpp" "$repo/src/integer_projection_execution.cpp" "$repo/src/hip_integer_projection.hip" \
    "$repo/src/low_precision.cpp" "$repo/src/iq4_xs.cpp" \
    "$repo/tests/integer_projection_tests.cpp" "$repo/tests/integer_projection_execution_tests.cpp" \
    "$source_dir/compile_offline.sh" > "$output/source.sha256"
args=(--offload-arch=gfx1151 -mno-wavefrontsize64 -std=c++20 -O2
    -Wall -Wextra -Werror -ffp-contract=off "-I$repo/include")
printf '%q ' "$compiler" "${args[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
"$compiler" "${args[@]}" --genco "$repo/src/hip_integer_projection.hip" \
    -o "$output/integer_projection.hsaco" > "$output/device.stdout" 2> "$output/device.stderr"
"$compiler" "${args[@]}" "$source_dir/integer_projection.hip" "$repo/src/hip_integer_projection.hip" \
    "$repo/src/integer_projection.cpp" "$repo/src/integer_projection_execution.cpp" \
    "$repo/src/low_precision.cpp" "$repo/src/iq4_xs.cpp" -o "$output/integer-projection-native" \
    > "$output/link.stdout" 2> "$output/link.stderr"
"$llvm/clang-offload-bundler" --list --type=o --input="$output/integer_projection.hsaco" > "$output/bundles.txt"
"$llvm/clang-offload-bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1151 \
    --input="$output/integer_projection.hsaco" --output="$output/integer_projection.elf"
"$llvm/llvm-objdump" -d --mcpu=gfx1151 "$output/integer_projection.elf" > "$output/disassembly.txt"
"$llvm/llvm-readelf" --notes "$output/integer_projection.elf" > "$output/notes.txt"
grep -q 'v_wmma_i32_16x16x16_iu8' "$output/disassembly.txt"
grep -q 'v_wmma_i32_16x16x16_iu4' "$output/disassembly.txt"
grep -q '.wavefront_size: 32' "$output/notes.txt"
echo 'Offline compile/link/instruction emission passed. No GPU execution or qualification.'
