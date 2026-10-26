#!/bin/bash
# implementation: build the spec CPU + HIP tests (CPU only, nice 19) with AMD clang (the gate-K binaries' compiler),
# plus the four S1 mutant builds (-DGLMSPEC_MUTANT=1..4) that must fail. Prints the full SHA-256 of every binary.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
CXX=/usr/bin/amdclang++
F="-std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Isrc"
HF="-x hip --offload-arch=gfx1151 -fno-gpu-flush-denormals-to-zero"
mkdir -p obj
nice -n 19 $CXX $F -c src/glm/host/glm_host.cpp -o obj/glm_host.o
nice -n 19 $CXX $F -c src/glm/spec/glm_spec_host.cpp -o obj/glm_spec_host.o
for f in tests/hip_glm_spec_tests src/glm/spec/glm_spec_dev src/glm/kernels/glm_tier_a src/glm/kernels/glm_dsa; do
    nice -n 19 $CXX $F $HF -save-temps=obj -c $f.hip -o obj/$(basename $f).o
done
LINK="obj/hip_glm_spec_tests.o obj/glm_tier_a.o obj/glm_dsa.o obj/glm_spec_host.o obj/glm_host.o"
nice -n 19 $CXX --hip-link --offload-arch=gfx1151 obj/glm_spec_dev.o $LINK -o hip_glm_spec_tests
for m in 1 2 3 4; do
    nice -n 19 $CXX $F $HF -DGLMSPEC_MUTANT=$m -c src/glm/spec/glm_spec_dev.hip -o obj/glm_spec_dev_m$m.o
    nice -n 19 $CXX --hip-link --offload-arch=gfx1151 obj/glm_spec_dev_m$m.o $LINK -o hip_glm_spec_tests_m$m
done
nice -n 19 $CXX $F tests/glm_spec_cpu_tests.cpp obj/glm_spec_host.o obj/glm_host.o -o glm_spec_cpu_tests
# host mutant 1: MTP extension steps select fresh instead of sharing (S0 must fail)
nice -n 19 $CXX $F -DGLMSPEC_HOST_MUTANT=1 tests/glm_spec_cpu_tests.cpp src/glm/spec/glm_spec_host.cpp obj/glm_host.o -o glm_spec_cpu_tests_hm1
sha256sum hip_glm_spec_tests hip_glm_spec_tests_m1 hip_glm_spec_tests_m2 hip_glm_spec_tests_m3 hip_glm_spec_tests_m4 glm_spec_cpu_tests     glm_spec_cpu_tests_hm1
