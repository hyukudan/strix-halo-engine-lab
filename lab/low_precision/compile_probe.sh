#!/usr/bin/env bash
set -euo pipefail
# No downloads or device access. Output directory is explicit and never erased.
if [[ $# -ne 1 ]]; then
    echo 'usage: CLANG=/opt/rocm/llvm/bin/clang++ compile_probe.sh OUTPUT_DIRECTORY' >&2
    exit 2
fi
compiler=${CLANG:-/opt/rocm/llvm/bin/clang++}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=$1
mkdir -p -- "$output_dir"
"$compiler" --version > "$output_dir/compiler.txt"
sha256sum "$source_dir/intrinsic_probe.cpp" > "$output_dir/source.sha256"
arguments=(--target=amdgcn-amd-amdhsa -mcpu=gfx1151 -mno-wavefrontsize64
    -nogpulib -O3 -std=c++20 -Wall -Wextra -Werror)
printf '%q ' "$compiler" "${arguments[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
"$compiler" "${arguments[@]}" -S -emit-llvm "$source_dir/intrinsic_probe.cpp" \
    -o "$output_dir/probe.ll" 2> "$output_dir/compile.stderr"
"$compiler" "${arguments[@]}" -S "$source_dir/intrinsic_probe.cpp" \
    -o "$output_dir/probe.s" 2>> "$output_dir/compile.stderr"
"$compiler" "${arguments[@]}" -c "$source_dir/intrinsic_probe.cpp" \
    -o "$output_dir/probe.o" 2>> "$output_dir/compile.stderr"
objdump=${LLVM_OBJDUMP:-$(dirname -- "$compiler")/llvm-objdump}
"$objdump" -d --mcpu=gfx1151 "$output_dir/probe.o" > "$output_dir/disassembly.txt"
if ! grep -q 'v_wmma_i32_16x16x16_iu8' "$output_dir/disassembly.txt" ||
   ! grep -q 'v_wmma_i32_16x16x16_iu4' "$output_dir/disassembly.txt" ||
   ! grep -Eq 'amdhsa_wavefront_size32[[:space:]]+1' "$output_dir/probe.s"; then
    echo 'Expected IU8/IU4 instructions absent; retain compiler artifacts as failed gate.' >&2
    exit 1
fi
echo 'Offline instruction emission passed; no matrix-layout/native correctness/performance qualification.'
