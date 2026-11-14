#pragma once
#include "v14_core_policy.hpp"

namespace strix::core {
inline bool pipeline_k2_candidate(int input,int output,int bits,int codebook,
                                  int rows,int splits,bool aligned) {
    return aligned && bits==2 && codebook==2 && rows==1 && splits==4 &&
        ((input==8192 && output==5120) || (input==1280 && output==32768));
}
// Zero retains v9. Modes 1/2/3 are the frozen exact-row experiments;
// mode 4 retains v9's LDS body at a separately measured split count.
inline int exact_gemv_mode(int input,int output,int bits,int codebook,
                           int rows,int splits,bool aligned) {
    if(!aligned || codebook!=2 || rows<1 || rows>8) return 0;
    const int* choices=nullptr;
    static constexpr int qb[8]={2,2,2,2,1,1,3,0};
    static constexpr int wob[8]={2,2,2,3,3,3,3,0};
    static constexpr int shared[8]={2,2,2,2,2,3,3,0};
    static constexpr int inverse[8]={2,2,2,2,1,1,3,0};
    static constexpr int qa[8]={0,0,2,2,2,2,2,2};
    static constexpr int woa[8]={2,2,2,0,3,3,3,0};
    static constexpr int kv[8]={0,1,2,1,2,1,2,2};
    static constexpr int head[8]={0,0,2,2,2,3,3,0};
    if(bits==5) {
        if(splits==4 && input==1280 && output==32768) choices=qb;
        if(splits==4 && input==8192 && output==5120) choices=wob;
        if(splits==4 && input==2304 && output==5120) choices=inverse;
        if(input==5120 && output==2304) {
            if(splits==8 && rows==8) return 4;
            if(splits==16) choices=shared;
        }
        if(input==4096 && output==1024) {
            if(splits==16 && (rows==4 || rows==8)) return 4;
            if(splits==32) choices=woa;
        }
    } else if(bits==6) {
        if(splits==16 && input==5120 && output==1280) choices=qa;
        if(splits==32 && input==5120 && output==512) choices=kv;
        if(splits==4 && input==5120 && output==129280) choices=head;
    }
    return choices?choices[rows-1]:0;
}

// Do not combine the shared R4 exact-global S16 experiment with unmeasured S8.
// R8 retains precisely the static LDS body used in the split experiment.
inline int measured_vector_splits(int input,int output,int bits,int codebook,int rows) {
    if(bits==5 && codebook==2) {
        if(input==4096 && output==1024 && (rows==1 || rows==4 || rows==8)) return 16;
        if(input==5120 && output==2304 && rows==8) return 8;
    }
    return output<=1024?32:(output<=4096 && input>=4096?16:4);
}
enum class GemvKernel { k2_multirow, k2_pipeline, exact, lds, baseline };
struct GemvSelection {
    GemvKernel kernel=GemvKernel::baseline;
    int rows=1, load=1;
    bool static_rows=false, fast=false;
};
// One effective choice feeds both launch and receipts. Alignment is that of the
// transformed input, partials and packed weights, not the original FP32 input.
inline GemvSelection select_gemv(int input,int output,int bits,int cb,int rows,int total,int splits,
    bool a4,bool p4,bool b4) {
    if (k2_multirow_candidate(input,output,bits,cb,rows,total,splits,a4,p4,b4))
        return {GemvKernel::k2_multirow,rows,1,false,false};
    if (pipeline_k2_candidate(input,output,bits,cb,rows,splits,a4))
        return {GemvKernel::k2_pipeline,rows,1,false,false};
    const int exact=exact_gemv_mode(input,output,bits,cb,rows,splits,a4);
    if (exact && exact!=4) return {GemvKernel::exact,rows,exact==3?2:1,true,exact!=1};
    if (exact==4) return {GemvKernel::lds,rows==4?4:8,2,rows!=4,true};
    const bool measured_rows=rows==1 || rows==2 || rows==4 || rows==8;
    const bool measured_shape=
        (bits==5 && splits==4 && ((input==1280 && output==32768) ||
         (input==8192 && output==5120) || (input==2304 && output==5120))) ||
        (bits==5 && splits==16 && input==5120 && output==2304) ||
        (bits==5 && splits==32 && input==4096 && output==1024) ||
        (bits==6 && splits==4 && ((input==8192 && output==5120) ||
         (input==5120 && output==129280))) ||
        (bits==6 && splits==16 && input==5120 && output==1280) ||
        (bits==6 && splits==32 && input==5120 && output==512);
    if (cb==2 && measured_rows && measured_shape && a4)
        return {rows<=2?GemvKernel::baseline:GemvKernel::lds,rows,2,rows==8,true};
    if (rows==8 && cb==2 && (bits==5 || bits==6) && a4)
        return {GemvKernel::lds,8,2,true,false};
    const int capacity=rows<=2?rows:(rows<=4?4:8);
    if (cb==2 && (bits==5 || bits==6) && rows>=3 && a4)
        return {GemvKernel::lds,capacity,2,false,false};
    return {GemvKernel::baseline,capacity,1,false,false};
}
} // namespace strix::core
