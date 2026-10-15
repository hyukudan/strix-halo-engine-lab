#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx1151__) || !__has_builtin(__builtin_amdgcn_sudot4)
#error "gfx1151 native sudot4 compile qualification is mandatory"
#endif
#endif
extern "C" __global__ void strix_q4k_a8_native_dot4_compile_probe(const int* a,const unsigned* b,int* out) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    out[i]=__builtin_amdgcn_sudot4(true,a[i],false,b[i],0,false);
}
