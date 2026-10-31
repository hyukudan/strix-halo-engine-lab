// Included inside native.cu's implementation namespace. The legacy WMMA and
// chronological SIMT paths above remain independent comparison arms.
using Half = cutlass::half_t;
using Row = cutlass::layout::RowMajor;
using Coord = cutlass::gemm::GemmCoord;
template<int M,int Stages>
using C1Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Half,Row,cutlass::ComplexTransform::kNone,8,
    Half,Row,cutlass::ComplexTransform::kNone,8,
    float,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<M,128,32>,
    cutlass::gemm::GemmShape<(M==128?64:16),64,32>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    Stages,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
using HotGemm = cutlass::gemm::device::GemmGrouped<C1Kernel<128,4>>;
using ColdGemm = cutlass::gemm::device::GemmGrouped<C1Kernel<32,3>>;
struct Problems {
    Coord size[2][32];
    Half *a[2][32], *b[2][32];
    float *d[2][32];
    int64_t lda[2][32],ldb[2][32],ldd[2][32];
};
static_assert(sizeof(Problems)<=32768);
static_assert(sizeof(Half)==sizeof(U16));

// One warp owns a tile. Cooperative 16-byte loads stage packed words once,
// followed by coalesced FP16 stores in the exact RowMajor CUTLASS B operand.
// A residency-sized grid strides over many tiles per CTA; bitrate dispatch
// happens once per matrix/CTA, outside the tile loop. No scales belong in B:
// suh/svh and their rounding boundaries remain in transforms/epilogues.
template<unsigned Bits>
__device__ __forceinline__ void decode_tiles(Matrix w,U16* q,U32* words) {
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    unsigned nt=w.output/16,total=w.input/16*nt,step=gridDim.x*8;
    unsigned tile=blockIdx.x*8+warp,kt=tile/nt,no=tile%nt;
    unsigned dk=step/nt,dn=step%nt; // amortize runtime division across all tiles
    for(;tile<total;tile+=step) {
        if(lane<2*Bits) {
            auto v=__ldg(reinterpret_cast<const uint4*>(w.packed+std::size_t(tile)*16*Bits)+lane);
            reinterpret_cast<uint4*>(words)[lane]=v;
        }
        __syncwarp();
        unsigned kr=kt*16,nc=no*16;
        #pragma unroll
        for(unsigned j=0;j<8;++j) {
            unsigned k=lane/16+2*j,n=lane%16;
            q[std::size_t(kr+k)*w.output+nc+n]=book(c1_state<Bits>(words,c1_position(k,n)));
        }
        __syncwarp(); // all readers finish before this warp reuses staging
        kt+=dk;no+=dn;
        if(no>=nt){no-=nt;++kt;}
    }
}
__global__ __launch_bounds__(256) void decode_window(const Expert* ex,const U32* counts,
    U16* qg,U16* qu,U16* qd,unsigned first,unsigned h,unsigned i) {
    unsigned local=blockIdx.y,which=blockIdx.z;
    if(!counts[first+local])return;
    Expert e=ex[first+local]; Matrix w=which==0?e.gate:(which==1?e.up:e.down);
    U16* q=(which==0?qg:which==1?qu:qd)+std::size_t(local)*h*i;
    __shared__ __align__(16) U32 packed[8][48];
    U32* words=packed[threadIdx.x/32];
    switch(w.bits) {
        case 3:decode_tiles<3>(w,q,words);break;
        case 4:decode_tiles<4>(w,q,words);break;
        case 5:decode_tiles<5>(w,q,words);break;
        case 6:decode_tiles<6>(w,q,words);break;
    }
}

// No histogram readback. Both lists contain fixed descriptor slots with zero-M
// problems for the other class (and empty experts); device-only scheduling
// skips those. M<=32 uses a 32-row tile; hot experts use 128-row tiles. Exact M
// is retained, so CUTLASS predicates tails rather than reading padded rows.
__global__ void problem_list(Problems* p,const U32* offsets,const U16* ag,
    const U16* au,U16* qg,U16* qu,float* sg,float* su,
    unsigned first,unsigned count,unsigned k,unsigned n,bool gu) {
    unsigned j=threadIdx.x,local=j/(gu?2:1),which=gu?j%2:0;
    if(local>=count) return;
    unsigned begin=offsets[first+local],m=offsets[first+local+1]-begin;
    for(unsigned cls=0;cls<2;++cls) {
        p->size[cls][j]=Coord(((m<=32)==bool(cls))?m:0,n,k);
        p->a[cls][j]=(Half*)(which?au:ag)+std::size_t(begin)*k;
        p->b[cls][j]=(Half*)(which?qu:qg)+std::size_t(local)*k*n;
        p->d[cls][j]=(which?su:sg)+std::size_t(begin)*n;
        p->lda[cls][j]=k; p->ldb[cls][j]=n; p->ldd[cls][j]=n;
    }
}
template<class Gemm>
cudaError_t run_grouped(Problems* p,unsigned cls,unsigned count,int blocks,cudaStream_t s) {
    typename Gemm::Arguments a(p->size[cls],count,blocks,{1.f,0.f},
        p->a[cls],p->b[cls],p->d[cls],p->d[cls],
        p->lda[cls],p->ldb[cls],p->ldd[cls],p->ldd[cls]);
    // Device-only problem visitor needs no host workspace or precomputation.
    // prepare_cutlass sets dynamic shared memory attributes once, outside the
    // timed region. This is the launch used by GemmGrouped::run.
    using Kernel=typename Gemm::GemmKernel;
    typename Kernel::Params params(a);
    cutlass::Kernel<Kernel><<<blocks,Kernel::kThreadCount,sizeof(typename Kernel::SharedStorage),s>>>(params);
    return cudaGetLastError();
}
cudaError_t gemms(Problems* p,unsigned count,const CutlassExecution& e,cudaStream_t s) {
    auto err=run_grouped<HotGemm>(p,0,count,e.hot_blocks,s);
    if(err!=cudaSuccess) return err;
    return run_grouped<ColdGemm>(p,1,count,e.cold_blocks,s);
}

// Warp/register output butterflies: complete 128-column blocks, no shared
// memory or CTA barriers. SwiGLU's down-input scale and first FP16 rounding
// occur here; the following down transform completes H128 and final rounding.
template<bool GU>
__global__ void c1_epilogue(const Expert* ex,const U32* offsets,const U32* map,
    const float* sg,const float* su,U16* ad,float* out,
    unsigned first,unsigned count,unsigned width) {
    unsigned e=first+blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(blockIdx.y>=count) return;
    unsigned blocks=width/128,begin=offsets[e],rows=offsets[e+1]-begin;
    Expert w=ex[e];
    for(unsigned b=blockIdx.x*8+warp;b<rows*blocks;b+=gridDim.x*8) {
        unsigned entry=begin+b/blocks,col=b%blocks*128;
        float g[4],u[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) {
            auto p=std::size_t(entry)*width+col+lane+32*j;
            g[j]=sg[p]; if constexpr(GU)u[j]=su[p];
        }
        h128(g,lane); if constexpr(GU)h128(u,lane);
        #pragma unroll
        for(unsigned j=0;j<4;++j) {
            unsigned c=col+lane+32*j;
            g[j]=__fmul_rn(g[j],hf((GU?w.gate.svh:w.down.svh)[c]));
            if constexpr(GU) {
                u[j]=__fmul_rn(u[j],hf(w.up.svh[c]));
                ad[std::size_t(entry)*width+c]=fh(__fmul_rn(swiglu(g[j],u[j]),hf(w.down.suh[c])));
            } else out[std::size_t(map[entry])*width+c]=g[j];
        }
    }
}
__global__ void c1_down_transform(U16* ad,const U32* offsets,unsigned first,unsigned count,unsigned width) {
    unsigned e=first+blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(blockIdx.y>=count)return;
    unsigned blocks=width/128,begin=offsets[e],rows=offsets[e+1]-begin;
    for(unsigned b=blockIdx.x*8+warp;b<rows*blocks;b+=gridDim.x*8) {
        std::size_t p=std::size_t(begin+b/blocks)*width+b%blocks*128+lane;
        float v[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j)v[j]=hf(ad[p+32*j]);
        h128(v,lane);
        #pragma unroll
        for(unsigned j=0;j<4;++j)ad[p+32*j]=fh(v[j]);
    }
}

cudaError_t launch_cutlass(const Call& v,void* ws,std::size_t bytes,cudaStream_t s,
    const CutlassExecution& exec,Profile* prof) {
    unsigned routes=v.experts==1?1:8;
    if(!shape(v.rows,v.experts,v.hidden,v.inter)||!ws||!v.x||!v.y||!v.matrices||
        (routes==8&&(v.experts<8||!v.ids||!v.weights))||!exec.decode_stream||
        exec.decode_stream==s||exec.hot_blocks<=0||exec.cold_blocks<=0||exec.decode_blocks<=0) return cudaErrorInvalidValue;
    Layout l(ws,v.rows,v.experts,v.hidden,v.inter,routes,true);
    if(bytes<l.size||reinterpret_cast<std::uintptr_t>(ws)%256)return cudaErrorInvalidValue;
    // Every base, row stride and expert offset is at least 16-byte aligned:
    // H/I multiples of 128, 256-byte arena allocations. No M padding required.
    auto* p=static_cast<Problems*>(l.problems);
    cudaError_t err;
    #define C1_CHECK(call) do { err=(call); if(err!=cudaSuccess)return err; } while(0)
    C1_CHECK(cudaMemsetAsync(l.counts,0,v.experts*4,s));
    unsigned entries=v.rows*routes;
    if(routes==1) dense_dispatch<<<(entries+255)/256,256,0,s>>>(l.counts,l.offsets,l.map,entries);
    else {
        histogram<<<(entries+255)/256,256,0,s>>>(v.ids,l.counts,entries,v.experts);
        prefix<<<1,1,0,s>>>(l.counts,l.offsets,l.cursor,v.experts);
        scatter<<<(entries+255)/256,256,0,s>>>(v.ids,l.cursor,l.map,entries,v.experts);
    }
    C1_CHECK(cudaEventRecord(exec.dispatch,s));
    C1_CHECK(cudaStreamWaitEvent(exec.decode_stream,exec.dispatch,0));
    for(unsigned first=0;first<v.experts;first+=16) {
        unsigned n=std::min(16u,v.experts-first),g=first/16;
        if(prof) C1_CHECK(cudaEventRecord(prof->begin[g],s));
        U16 *qg=g%2?l.qg2:l.qg,*qu=g%2?l.qu2:l.qu,*qd=g%2?l.qd2:l.qd;
        if(g>=2) C1_CHECK(cudaStreamWaitEvent(exec.decode_stream,exec.retired[g-2],0));
        if(prof) C1_CHECK(cudaEventRecord(prof->decode_start[g],exec.decode_stream));
        unsigned per_matrix=std::max(1u,(unsigned(exec.decode_blocks)+3*n-1)/(3*n));
        decode_window<<<dim3(per_matrix,n,3),256,0,exec.decode_stream>>>(v.matrices,l.counts,qg,qu,qd,first,v.hidden,v.inter);
        C1_CHECK(cudaGetLastError());
        if(prof) C1_CHECK(cudaEventRecord(prof->decoded[g],exec.decode_stream));
        C1_CHECK(cudaEventRecord(exec.ready[g],exec.decode_stream));
        transform<<<dim3(256,n),256,0,s>>>(v.matrices,v.x,l.activation,l.offsets,l.map,l.ag,l.au,l.ad,first,n,v.hidden,routes,false);
        if(prof) C1_CHECK(cudaEventRecord(prof->input[g],s));
        C1_CHECK(cudaStreamWaitEvent(s,exec.ready[g],0));
        if(prof) C1_CHECK(cudaEventRecord(prof->ready[g],s));
        problem_list<<<1,32,0,s>>>(p,l.offsets,l.ag,l.au,qg,qu,l.scratch_g,l.scratch_u,first,n,v.hidden,v.inter,true);
        if(prof) C1_CHECK(cudaEventRecord(prof->gu_start[g],s));
        C1_CHECK(gemms(p,n*2,exec,s));
        if(prof) C1_CHECK(cudaEventRecord(prof->gu_gemm[g],s));
        c1_epilogue<true><<<dim3(256,n),256,0,s>>>(v.matrices,l.offsets,l.map,l.scratch_g,l.scratch_u,l.ad,l.contribution,first,n,v.inter);
        if(prof) C1_CHECK(cudaEventRecord(prof->swiglu[g],s));
        c1_down_transform<<<dim3(256,n),256,0,s>>>(l.ad,l.offsets,first,n,v.inter);
        if(prof) C1_CHECK(cudaEventRecord(prof->down_input[g],s));
        problem_list<<<1,32,0,s>>>(p,l.offsets,l.ad,nullptr,qd,nullptr,l.scratch_g,nullptr,first,n,v.inter,v.hidden,false);
        if(prof) C1_CHECK(cudaEventRecord(prof->down_start[g],s));
        C1_CHECK(gemms(p,n,exec,s));
        if(prof) C1_CHECK(cudaEventRecord(prof->down_gemm[g],s));
        C1_CHECK(cudaEventRecord(exec.retired[g],s));
        c1_epilogue<false><<<dim3(256,n),256,0,s>>>(v.matrices,l.offsets,l.map,l.scratch_g,nullptr,l.ad,l.contribution,first,n,v.hidden);
        if(prof) C1_CHECK(cudaEventRecord(prof->down[g],s));
    }
    if(v.combine){ // a non-last expert batch leaves the merge to the last one
    if(prof) C1_CHECK(cudaEventRecord(prof->merge_begin,s));
    merge<<<unsigned((std::size_t(v.rows)*v.hidden+255)/256),256,0,s>>>(l.contribution,routes==1?nullptr:v.weights,v.y,v.rows,v.hidden,routes);
    if(prof) C1_CHECK(cudaEventRecord(prof->merge_end,s));
    }
    #undef C1_CHECK
    return cudaGetLastError();
}
