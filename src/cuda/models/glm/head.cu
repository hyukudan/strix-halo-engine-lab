#include "head.cuh"
#include "producer/head_select.hpp"
#include "row_mma.cuh"
#include "../../core/head_argmax.cuh"
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <algorithm>

namespace strix::glm::rtxpf {
namespace {
#include "projection.inl"
struct EmitChoice {  // HeadChoice words per row: token, nonfinite count, logit bits
    U32* result;
    __device__ void operator()(std::uint32_t row,const ::strix::head_argmax::Pick& p) const {
        U32* r=result+4*row;r[0]=p.index;r[1]=p.nonfinite;r[2]=__float_as_uint(p.value);
    }
};
}
std::size_t head_decoded_bytes(){return std::size_t(head_hidden)*head_vocab*2;}
std::size_t head_workspace_bytes(unsigned rows){return ((std::size_t(rows)*head_hidden*2+255)&~std::size_t(255))+std::size_t(rows)*head_vocab*4;}
cudaError_t decode_head(Matrix w,U16* q,cudaStream_t s){
    if(!w.packed||!q||w.input!=head_hidden||w.output!=head_vocab||w.bits!=6)return cudaErrorInvalidValue;
    layer_decode<<<w.input/16*(w.output/16),256,0,s>>>(w,q);return cudaGetLastError();
}
cudaError_t enqueue_head(const HeadWeights& h,const float* x,float* logits,unsigned rows,void* ws,std::size_t bytes,cudaStream_t s){
    if(!h.decoded||!h.suh||!h.svh||!x||!logits||!rows||rows>64||!ws||reinterpret_cast<std::uintptr_t>(ws)%256||bytes<head_workspace_bytes(rows))return cudaErrorInvalidValue;
    auto* operand=static_cast<U16*>(ws);auto* raw=reinterpret_cast<float*>(static_cast<char*>(ws)+((std::size_t(rows)*head_hidden*2+255)&~std::size_t(255)));
    input_transform<<<(rows*(head_hidden/128)+7)/8,256,0,s>>>(x,operand,h.suh,rows,head_hidden);
    projection_reference<<<unsigned((std::size_t(rows)*head_vocab+255)/256),256,0,s>>>(nullptr,operand,h.decoded,raw,rows,head_hidden,head_vocab,false);
    output_transform<<<(rows*(head_vocab/128)+7)/8,256,0,s>>>(raw,logits,h.svh,rows,head_vocab);
    return cudaGetLastError();
}
std::size_t argmax_workspace_bytes(unsigned rows){return ::strix::head_argmax::workspace_bytes(rows);}
cudaError_t enqueue_argmax(const float* x,unsigned n,unsigned rows,U32* result,void* workspace,cudaStream_t s){
    if(!x||!n||!rows||rows>16||!result||!workspace)return cudaErrorInvalidValue;
    ::strix::cuda::core::head_argmax::View v;v.logits=x;v.stride=n;v.count=n;v.rows=rows;
    v.partial=static_cast<::strix::head_argmax::Pick*>(workspace);
    return ::strix::cuda::core::head_argmax::pick_rows(v,EmitChoice{result},s);
}
}
