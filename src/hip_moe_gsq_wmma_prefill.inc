// GSQ Q2_0 routed experts in the A8 profile: iu8 WMMA forms of the dot4 prefill kernels, bit-identical. Included
// by hip_moe_q4k_a8_known_rows.hip after hip_moe_gsq_f16_prefill.inc, inside its anonymous namespace.
//
// project_prefill_q20 / down_prefill_q20 (and the window and R1 forms) give each (row, slot, output) eight (gate/up)
// or four (down) chains: chain c adds, in ascending groups g = c mod 8 (or 4), delta_g * (d * float(I_g - S_g)), with
// I_g the int dot of the group's 32 activation codes with its codes q (0..3) and S_g the sum of the activation codes;
// the lanes' xor tree folds them as ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)) (down: (p0 + p1) + (p2 + p3)).
// Here two v_wmma_i32_16x16x16_iu8 form I_g exactly (magic-offset accumulator, |I_g| <= 12192), the stage holds
// 0x1.8p23 + float(S_g) (exact), so __int_as_float(r) minus it is float(I_g - S_g) exactly, and the FP32 steps and
// the fold are the dot4 ones. Layouts and batches as project_prefill_wmma / down_prefill_wmma (16 assignments per
// batch, rows 2e + h of D element e in half h); the gate/up epilogue is project_prefill_q20's.
template<unsigned Groups> struct GsqWmmaStage {
    wmma_v4i a[2U*Groups][wmma_batch]; // 16-value plane k of every row: codes [16k, 16k+16)
    float delta[Groups][wmma_batch];
    float msum[Groups][wmma_batch];    // 0x1.8p23 + float(int sum), exact
};
template<unsigned Groups,class Codes,class Deltas,class Sums>
__device__ __forceinline__ void gsq_wmma_stage_rows(GsqWmmaStage<Groups>& s,unsigned count,unsigned width,unsigned threads,
    const Codes& codes,const Deltas& deltas,const Sums& sums) {
    const unsigned planes=width/16U,groups=width/32U;
    for(unsigned item=threadIdx.x;item<wmma_batch*planes;item+=threads) {
        const unsigned a=item/planes,k=item%planes;
        wmma_v4i v{0,0,0,0};
        if(a<count)v=*reinterpret_cast<const wmma_v4i*>(codes(a)+16U*k);
        s.a[k][a]=v;
    }
    for(unsigned item=threadIdx.x;item<wmma_batch*groups;item+=threads) {
        const unsigned a=item/groups,g=item%groups,slot=wmma_row_slot(a);
        float e=0.0F,f=0x1.8p23F;
        if(a<count){e=deltas(a)[g];f=0x1.8p23F+static_cast<float>(sums(a)[g]);}
        s.delta[g][slot]=e;s.msum[g][slot]=f;
    }
}
// The 16 codes of one code dword as four int8 words in K order (byte j of word k = code 4k + j).
__device__ __forceinline__ void gsq_q20_codes(unsigned w,unsigned (&out)[4]) {
    const unsigned y0=w&0x03030303U,y1=(w>>2U)&0x03030303U,y2=(w>>4U)&0x03030303U,y3=(w>>6U)&0x03030303U;
    const unsigned p0=__builtin_amdgcn_perm(y1,y0,0x05010400U),p1=__builtin_amdgcn_perm(y1,y0,0x07030602U); // codes 0,1,4,5 / 8,9,12,13
    const unsigned q0=__builtin_amdgcn_perm(y3,y2,0x05010400U),q1=__builtin_amdgcn_perm(y3,y2,0x07030602U); // 2,3,6,7 / 10,11,14,15
    out[0]=__builtin_amdgcn_perm(q0,p0,0x05040100U);out[1]=__builtin_amdgcn_perm(q0,p0,0x07060302U);
    out[2]=__builtin_amdgcn_perm(q1,p1,0x05040100U);out[3]=__builtin_amdgcn_perm(q1,p1,0x07060302U);
}
__device__ __forceinline__ wmma_v4i gsq_operand(const unsigned (&v)[4]) {
    return wmma_v4i{static_cast<int>(v[0]),static_cast<int>(v[1]),static_cast<int>(v[2]),static_cast<int>(v[3])};
}
// Chains of one wave for one weight row (this lane's output): blocks ascending, group g = 2b + half adds to chain
// g % Chains, so a pass of Chains/2 blocks feeds every chain once. Elements e < V only (rows 2e, 2e + 1 of the
// batch); the WMMAs always cover all 16 rows.
template<unsigned Chains,unsigned V,unsigned Groups>
__device__ __forceinline__ void gsq_q20_chains(const unsigned char* row,const GsqWmmaStage<Groups>& s,unsigned blocks,unsigned i,unsigned h,
    float (&out)[8]) {
    constexpr unsigned Blocks=Chains/2U;
    float acc[Chains][8];
#pragma unroll
    for(unsigned c=0;c<Chains;++c)
#pragma unroll
        for(unsigned e=0;e<8U;++e)acc[c][e]=0.0F;
    const auto pass=[&](unsigned b0,const GsqQ20Raw (&x)[Blocks]) {
#pragma unroll
        for(unsigned j=0;j<Blocks;++j) {
            const unsigned b=b0+j,odd=b&1U,select=odd?0x07060504U:0x05040302U;
            const float d=scale_f32(static_cast<unsigned short>(odd?x[j].w[0]>>16U:x[j].w[0]&65535U));
#pragma unroll
            for(unsigned half=0;half<2U;++half) {
                const unsigned g=2U*b+half,c=2U*j+half;
                unsigned lo[4],hi[4];
                gsq_q20_codes(__builtin_amdgcn_perm(x[j].w[2U*half+1U],x[j].w[2U*half],select),lo);
                gsq_q20_codes(__builtin_amdgcn_perm(x[j].w[2U*half+2U],x[j].w[2U*half+1U],select),hi);
                const wmma_v8i r=group_dot(s.a[2U*g][i],gsq_operand(lo),s.a[2U*g+1U][i],gsq_operand(hi));
                float deltas[8],msums[8];
                row_values<V>(&s.delta[g][8U*h],deltas);row_values<V>(&s.msum[g][8U*h],msums);
#pragma unroll
                for(unsigned e=0;e<V;++e) {
                    const float integer=__int_as_float(r[e])-msums[e];
                    const float scaled=d*integer;
                    const float value=deltas[e]*scaled;
                    acc[c][e]=acc[c][e]+value;
                }
            }
        }
    };
    // Two pass buffers in turn: pass p + 1 loads while pass p computes (blocks is a multiple of Blocks).
    GsqQ20Raw x[2][Blocks];
#pragma unroll
    for(unsigned j=0;j<Blocks;++j)x[0][j]=gsq_q20_fetch(row,j);
#pragma unroll 1
    for(unsigned b=0;b<blocks;b+=2U*Blocks) {
        if(b+Blocks<blocks)
#pragma unroll
            for(unsigned j=0;j<Blocks;++j)x[1][j]=gsq_q20_fetch(row,b+Blocks+j);
        pass(b,x[0]);
        if(b+Blocks<blocks) {
            if(b+2U*Blocks<blocks)
#pragma unroll
                for(unsigned j=0;j<Blocks;++j)x[0][j]=gsq_q20_fetch(row,b+2U*Blocks+j);
            pass(b+Blocks,x[1]);
        }
    }
#pragma unroll
    for(unsigned e=0;e<V;++e) {
        if constexpr(Chains==8U) {
            const float s01=acc[0][e]+acc[1][e],s23=acc[2][e]+acc[3][e],s45=acc[4][e]+acc[5][e],s67=acc[6][e]+acc[7][e];
            const float s0123=s01+s23,s4567=s45+s67;
            out[e]=s0123+s4567;
        } else {
            const float s01=acc[0][e]+acc[1][e],s23=acc[2][e]+acc[3][e];
            out[e]=s01+s23;
        }
    }
}
// Gate/up of one CSR batch (<= 16 assignments) for Waves/2 16-output tiles: wave w runs matrix w%2 of tile w/2.
// Grid x = batch, y = output tile.
template<unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void project_prefill_wmma_q20(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    constexpr unsigned Outs=8U*Waves,Groups=max_hidden/32U;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ union { GsqWmmaStage<Groups> s; float tile[2][wmma_batch][Outs]; } lds;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    gsq_wmma_stage_rows<Groups>(lds.s,count,hidden,32U*Waves,[&](unsigned a){return at(view.codes,row_of[a]);},
        [&](unsigned a){return at(view.deltas,row_of[a]);},[&](unsigned a){return at(view.sums,row_of[a]);});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,m=wave&1U,t=wave>>1U;
    const unsigned tile_base=blockIdx.y*Outs,out=tile_base+16U*t+i,blocks=hidden/64U,valid=(count+1U)/2U;
    const auto* w=(m==0U?gate:up)+(static_cast<unsigned long long>(expert)*outputs+out)*gsq_a8::row_bytes(gsq_a8::type_q2_0,hidden);
    float result[8]={};
    if(valid<=2U)gsq_q20_chains<8U,2U>(w,lds.s,blocks,i,h,result);
    else if(valid<=4U)gsq_q20_chains<8U,4U>(w,lds.s,blocks,i,h,result);
    else gsq_q20_chains<8U,8U>(w,lds.s,blocks,i,h,result);
    __syncthreads(); // every wave is done with the staged rows: the tile reuses them
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count)lds.tile[m][2U*e+h][16U*t+i]=result[e];
    __syncthreads();
    // project_prefill_q20's epilogue for each staged (row, slot) and 32-output group of the CTA, one wave per pair.
    constexpr unsigned Quads=Outs/32U;
    for(unsigned job=wave;job<count*Quads;job+=Waves) {
        const unsigned a=job/Quads,quad=job%Quads,tile_index=tile_base/32U+quad;
        const auto r=row_of[a],slot=slot_of[a];
        float* output=at(view.output,r);
        const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
        const float g=lds.tile[0][a][32U*quad+lane],u=lds.tile[1][a][32U*quad+lane];
        const float hv=(g/(1.0F+expf(-g)))*u;
        output[at_slot+outputs]=u;
        const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
        if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}continue;}
        float x=fabsf(hv);
        for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,__shfl_xor(x,stride,32));
        const bool tiny=x>0.0F && x<0x1p-119F;
        const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
        const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
        const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
        int sum=code;
        for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
        output[at_slot]=hv;
        at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
        if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
    }
}
// Q2_0 decoder self-test (gsq_q20_decode_selftest): block b (64-block rows) has scale bits (b + seed) & 65535 (1024
// rows: every FP16 pattern; seeds 0 and 1: each pattern in an even and an odd block) and code byte j = (7b + 37j + (b >> 8) + seed) & 255 (every byte in
// every position). Each thread decodes one block: the FP16 forms of both TPC layouts (both parts of TPC 2) against
// round_f16(d * float(q - 1)) formed directly, and the iu8 operands against the codes. counts: [0] FP16 words tested,
// [1] FP16 words differing (non-NaN reference, any bit; NaN reference, a non-NaN result), [2] NaN references,
// [3] iu8 code bytes tested, [4] iu8 code bytes differing.
__global__ void gsq_q20_fill_kernel(unsigned char* blocks,unsigned count,unsigned seed) {
    const unsigned b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=count)return;
    unsigned char* p=blocks+18ULL*b;
    const unsigned d=(b+seed)&65535U;
    p[0]=static_cast<unsigned char>(d);p[1]=static_cast<unsigned char>(d>>8U);
    for(unsigned j=0;j<16U;++j)p[2U+j]=static_cast<unsigned char>(7U*b+37U*j+(b>>8U)+seed);
}
__global__ void gsq_q20_decode_kernel(const unsigned char* blocks,unsigned count,unsigned long long* counts) {
    const unsigned b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=count)return;
    const unsigned char* row=blocks+18ULL*(b&~63U),*p=blocks+18ULL*b;
    const unsigned block=b&63U;
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(p[0]|p[1]<<8U)));
    unsigned short ref[64];
    unsigned nan=0;
    for(unsigned k=0;k<64U;++k) {
        const int q=static_cast<int>((p[2U+k/4U]>>(2U*(k%4U)))&3U);
        const _Float16 v=static_cast<_Float16>(d*static_cast<float>(q-1));
        ref[k]=__builtin_bit_cast(unsigned short,v);nan+=(ref[k]&0x7fffU)>0x7c00U;
    }
    const auto differ=[&](unsigned k,unsigned short got) {
        const bool rn=(ref[k]&0x7fffU)>0x7c00U,gn=(got&0x7fffU)>0x7c00U;
        return rn?!gn:got!=ref[k];
    };
    unsigned long long tested=0,bad=0;
    const GsqQ20Raw raw=gsq_q20_fetch(row,block);
    {
        uint4 out[8];
        gsq_q20_halves<1U>(raw,block,0U,out);
        for(unsigned q=0;q<8U;++q){const unsigned w[4]{out[q].x,out[q].y,out[q].z,out[q].w};
            for(unsigned i=0;i<4U;++i){bad+=differ(8U*q+2U*i,static_cast<unsigned short>(w[i]))+differ(8U*q+2U*i+1U,static_cast<unsigned short>(w[i]>>16U));tested+=2U;}}
    }
    for(unsigned part=0;part<2U;++part) {
        uint4 out[4];
        gsq_q20_halves<2U>(raw,block,part,out);
        for(unsigned q=0;q<4U;++q){const unsigned w[4]{out[q].x,out[q].y,out[q].z,out[q].w};
            for(unsigned i=0;i<4U;++i){const unsigned k=32U*part+8U*q+2U*i;
                bad+=differ(k,static_cast<unsigned short>(w[i]))+differ(k+1U,static_cast<unsigned short>(w[i]>>16U));tested+=2U;}}
    }
    unsigned long long codes_tested=0,codes_bad=0;
    const unsigned odd=block&1U,select=odd?0x07060504U:0x05040302U;
    for(unsigned half=0;half<2U;++half)for(unsigned w=0;w<2U;++w) {
        unsigned c[4];
        gsq_q20_codes(__builtin_amdgcn_perm(raw.w[2U*half+w+1U],raw.w[2U*half+w],select),c);
        for(unsigned k=0;k<16U;++k) {
            const unsigned e=32U*half+16U*w+k,q=(p[2U+e/4U]>>(2U*(e%4U)))&3U;
            codes_bad+=((c[k/4U]>>(8U*(k%4U)))&255U)!=q;++codes_tested;
        }
    }
    atomicAdd(counts,tested);atomicAdd(counts+1,bad);atomicAdd(counts+2,static_cast<unsigned long long>(2U*nan));
    atomicAdd(counts+3,codes_tested);atomicAdd(counts+4,codes_bad);
}
// Down of one CSR batch (<= 16 assignments) for Waves 16-row tiles of the hidden output, width = max_rows.
// Grid x = batch, y = row tile.
template<unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void down_prefill_wmma_q20(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    constexpr unsigned Outs=16U*Waves,Groups=max_rows/32U;
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ GsqWmmaStage<Groups> s;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    const unsigned groups=width/32U;
    gsq_wmma_stage_rows<Groups>(s,count,width,32U*Waves,[&](unsigned a){return at(view.down_codes,row_of[a])+slot_of[a]*width;},
        [&](unsigned a){return at(view.down_deltas,row_of[a])+slot_of[a]*groups;},
        [&](unsigned a){return at(view.down_sums,row_of[a])+slot_of[a]*groups;});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,valid=(count+1U)/2U;
    const unsigned row=blockIdx.y*Outs+16U*wave+i;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*gsq_a8::row_bytes(gsq_a8::type_q2_0,width);
    float result[8]={};
    if(valid<=2U)gsq_q20_chains<4U,2U>(row_weights,s,width/64U,i,h,result);
    else if(valid<=4U)gsq_q20_chains<4U,4U>(row_weights,s,width/64U,i,h,result);
    else gsq_q20_chains<4U,8U>(row_weights,s,width/64U,i,h,result);
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count) {
        const unsigned a=2U*e+h;
        at(view.rawdots,row_of[a])[slot_of[a]*hidden+row]=result[e];
    }
}
