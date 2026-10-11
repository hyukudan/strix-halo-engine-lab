// Compiler/instruction probe only. Per-lane inputs deliberately do not implement
// a logical matrix layout. No lane/fragment correctness or native execution claim.
// Builtin declarations: https://clang.llvm.org/docs/AMDGPUBuiltinReference.html

#if !defined(__AMDGCN__) || !defined(__gfx1151__)
#error "This bounded probe requires gfx1151; Wave32 is checked in emitted metadata"
#endif

using int2 = int __attribute__((ext_vector_type(2)));
using int4 = int __attribute__((ext_vector_type(4)));
using int8 = int __attribute__((ext_vector_type(8)));

extern "C" __attribute__((amdgpu_kernel))
void probe_signed_iu8(int8* output, const int4* a, const int4* b) {
    const auto lane = __builtin_amdgcn_workitem_id_x();
    const int8 zero{};
    output[lane] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, a[lane], true, b[lane], zero, false);
}

extern "C" __attribute__((amdgpu_kernel))
void probe_signed_iu4(int8* output, const int2* a, const int2* b) {
    const auto lane = __builtin_amdgcn_workitem_id_x();
    const int8 zero{};
    output[lane] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        true, a[lane], true, b[lane], zero, false);
}

extern "C" __attribute__((amdgpu_kernel))
void probe_signed_unsigned_iu4(int8* output, const int2* a, const int2* b) {
    const auto lane = __builtin_amdgcn_workitem_id_x();
    const int8 zero{};
    output[lane] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        true, a[lane], false, b[lane], zero, false);
}
