#include <hip/hip_runtime.h>
#include "h3/gemm/dense.hpp"
#include <stdexcept>
namespace strix::h3::gemm {
namespace {
using Short16 = short __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
__global__ __launch_bounds__(128) void dense_kernel(const std::uint16_t* x, const std::uint16_t* w,
        unsigned m, unsigned n, unsigned k, std::uint16_t* y) {
    __shared__ short a[64*16], b[64*16];
    const unsigned tid=threadIdx.x, lane=tid%32, wave=tid/32;
    const unsigned r0=blockIdx.y*64, c0=blockIdx.x*64, wm=(wave/2)*32, wn=(wave%2)*32;
    Float8 acc[2][2]{};
    for (unsigned base=0; base<k; base+=16) {
        for (unsigned i=tid; i<64*16; i+=128) {
            const unsigned row=i/16, col=i%16;
            a[i]=(r0+row<m) ? static_cast<short>(x[static_cast<std::size_t>(r0+row)*k+base+col]) : 0;
            b[i]=(c0+row<n) ? static_cast<short>(w[static_cast<std::size_t>(c0+row)*k+base+col]) : 0;
        }
        __syncthreads();
        for (unsigned i=0; i<2; ++i) {
            const Short16 av=*reinterpret_cast<const Short16*>(a+(wm+i*16+lane%16)*16);
            for (unsigned j=0; j<2; ++j) {
                const Short16 bv=*reinterpret_cast<const Short16*>(b+(wn+j*16+lane%16)*16);
                acc[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av,bv,acc[i][j]);
            }
        }
        __syncthreads();
    }
    for (unsigned i=0; i<2; ++i) for (unsigned j=0; j<2; ++j) for (unsigned e=0; e<8; ++e) {
        const unsigned r=r0+wm+i*16+2*e+lane/16, c=c0+wn+j*16+lane%16;
        if (r<m && c<n) {
            unsigned u=__float_as_uint(acc[i][j][e]);
            if ((u&0x7fffffffU)>0x7f800000U) u|=0x00400000U;
            else u+=0x7fffU+((u>>16)&1U);
            y[static_cast<std::size_t>(r)*n+c]=static_cast<std::uint16_t>(u>>16);
        }
    }
}
}
void dense_bf16(const std::uint16_t* x, const std::uint16_t* w, std::uint32_t m,
                std::uint32_t n, std::uint32_t k, std::uint16_t* y, void* stream) {
    if (!m || !n) return;
    if (!k || k%16) throw std::invalid_argument("dense BF16: K must be a multiple of 16");
    hipLaunchKernelGGL(dense_kernel,dim3((n+63)/64,(m+63)/64),dim3(128),0,
                      static_cast<hipStream_t>(stream),x,w,m,n,k,y);
    if (hipGetLastError()!=hipSuccess) throw std::runtime_error("dense BF16: launch failed");
}
}
