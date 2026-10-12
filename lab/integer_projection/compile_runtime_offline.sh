#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: compile_runtime_offline.sh NEW_PRIVATE_OUTPUT_DIRECTORY PINNED_CORE_ARCHIVE' >&2; exit 2; fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$source_dir/../.." && pwd)
output=$(realpath -m -- "$1")
core=$(realpath -e -- "$2")
private_path() {
    local path=$1 ancestor=$1
    case "$path/" in "$repo/"*) echo 'Compiler artifacts must remain outside source.' >&2; exit 2;; esac
    while true; do
        if [[ -e "$ancestor/.git" ]]; then echo 'Compiler artifacts must remain outside every Git checkout.' >&2; exit 2; fi
        [[ $ancestor == / ]] && break
        ancestor=$(dirname -- "$ancestor")
    done
}
private_path "$output"
private_path "$core"
if [[ ! -f $core || -e $output || -L $output ]]; then echo 'Require existing private core archive and new unused output directory.' >&2; exit 2; fi
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
compiler=${HIPCC:-/opt/rocm/bin/hipcc}
llvm=${ROCM_LLVM:-/opt/rocm/llvm/bin}
"$compiler" --version > "$output/compiler.txt"
sha256sum "$core" > "$output/core-before.sha256"
sources=(include/strix/integer_projection.hpp include/strix/integer_projection_execution.hpp
    include/strix/integer_projection_runtime.hpp include/strix/hip_integer_projection.hpp
    include/strix/hip_integer_projection_runtime.hpp include/strix/hip_launch_context.hpp
    include/strix/hip_lm_head_launcher.hpp include/strix/low_precision.hpp include/strix/iq4_xs.hpp
    src/integer_projection.cpp src/integer_projection_execution.cpp src/integer_projection_runtime.cpp
    src/hip_integer_projection.hip src/hip_integer_projection_runtime.hip src/hip_lm_head_launcher.hip
    src/low_precision.cpp src/iq4_xs.cpp tests/integer_projection_runtime_fixture.hpp
    tests/integer_projection_runtime_tests.cpp tests/hip_integer_projection_runtime_tests.hip
    lab/integer_projection/compile_runtime_offline.sh)
for path in "${sources[@]}"; do sha256sum "$repo/$path"; done > "$output/source.sha256"
args=(--offload-arch=gfx1151 -mno-wavefrontsize64 -std=c++20 -O2
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -ffp-contract=off "-I$repo/include" "-I$repo/tests")
printf '%q ' "$compiler" "${args[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
for pair in 'runtime:hip_integer_projection_runtime' 'projection:hip_integer_projection'; do
    label=${pair%%:*}
    file=${pair#*:}
    "$compiler" "${args[@]}" --genco "$repo/src/$file.hip" -o "$output/$label.hsaco" \
        > "$output/$label.stdout" 2> "$output/$label.stderr"
    "$llvm/clang-offload-bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1151 \
        --input="$output/$label.hsaco" --output="$output/$label.elf"
    "$llvm/llvm-objdump" -d --mcpu=gfx1151 "$output/$label.elf" > "$output/$label.disassembly.txt"
    "$llvm/llvm-readelf" --notes "$output/$label.elf" > "$output/$label.notes.txt"
done
"$compiler" "${args[@]}" "$repo/tests/hip_integer_projection_runtime_tests.hip" \
    "$repo/src/hip_integer_projection_runtime.hip" "$repo/src/hip_integer_projection.hip" \
    "$repo/src/hip_lm_head_launcher.hip" "$repo/src/integer_projection_runtime.cpp" \
    -x none "$core" -o "$output/integer-runtime-native" > "$output/link.stdout" 2> "$output/link.stderr"
sha256sum "$core" > "$output/core-after.sha256"
cmp "$output/core-before.sha256" "$output/core-after.sha256"
sha256sum -c "$output/source.sha256" > "$output/source-check.txt"
grep -q 'v_wmma_i32_16x16x16_iu8' "$output/projection.disassembly.txt"
grep -q 'v_wmma_i32_16x16x16_iu4' "$output/projection.disassembly.txt"
grep -q '.wavefront_size: 32' "$output/projection.notes.txt"
grep -q '.wavefront_size: 32' "$output/runtime.notes.txt"
sha256sum "$output/integer-runtime-native" > "$output/binary.sha256"
echo 'Runtime compile/link/disassembly passed; no native execution or qualification. Core archive is a separately pinned build dependency.'
