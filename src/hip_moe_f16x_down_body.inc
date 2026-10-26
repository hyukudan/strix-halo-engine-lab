// qpf2k/moe: the body of down_prefill_f16x, included verbatim by the kernel (hip_moe_q4k_a8_known_rows.hip, identical tokens: the
// macros below expand to the original lines) and by the P1 fused launch (hip_moe_prefill_p1.inc). STRIX_F16X_BATCH_TILE
// declares batch / tile, STRIX_F16X_LDS(Stage, Rows) the stage `sb` (indexable) and the rows table `rows`,
// STRIX_F16X_DOWN_GUARD the early return (P1 decides it once per CTA from an atomic read before the body: empty there).
    using T=F16Tile<Tune>;
    constexpr unsigned Batch=T::Batch,Columns=T::Columns,TPC=T::TPC,AParts=Batch*4U/T::Threads;
    STRIX_F16X_BATCH_TILE
    const auto& csr=view.csr;
    STRIX_F16X_DOWN_GUARD
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    using Stage=F16Stage<Batch,Columns,T::LTune>;
    STRIX_F16X_LDS(Stage,F16Rows<T::Table?Batch:1U>)
    const unsigned t=threadIdx.x;
    if constexpr(T::Table) if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=slot;
        rows.exponent[t]=live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0;
    }
    const auto row_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.row[a];else return a<count?csr.rows[begin+a]:0U;};
    const auto slot_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.slot[a];else return a<count?csr.slots[begin+a]:0U;};
    const auto exp_of=[&](unsigned a)->int{if constexpr(T::Table)return rows.exponent[a];else return a<count?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,row_of(a))[slot_of(a)]))):0;};
    __syncthreads();
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned groups=width/32U,steps=width/64U;
    // A: part c = t + Threads*j is row c/4, K values 16*(c%4).. of the step (FP32 h, scaled and rounded here).
    const float* a_src[AParts];float a_scale[AParts];bool a_live[AParts];unsigned a_row[AParts],a_part[AParts];
#pragma unroll
    for(unsigned j=0;j<AParts;++j) {
        const unsigned c=t+T::Threads*j;
        a_row[j]=c/4U;a_part[j]=c%4U;a_live[j]=a_row[j]<count;
        a_scale[j]=ldexpf(1.0F,exp_of(a_row[j]));
        // Probe 8192 (hot A, attribution only): every row reads the batch's first (row, slot).
        const unsigned ar=(Tune&8192U)?0U:a_row[j];
        a_src[j]=at(view.output,row_of(ar))+2ULL*slot_of(ar)*width+16U*a_part[j];
    }
    // B: column t % Columns decodes K blocks 2*step + part (TPC 2) or both blocks of the step (TPC 1).
    const unsigned b_col=t%Columns,b_part=t/Columns;
    constexpr unsigned BBlocks=2U/TPC;
    const unsigned b_expert=(Tune&16384U)?0U:expert; // probe 16384 (hot B): expert 0's weights
    const auto* b_src=weights+(static_cast<unsigned long long>(b_expert)*hidden+tile*Columns+b_col)*groups*block_bytes;
    struct Regs { uint4 a[AParts][4]{}; DownBlock<Q51> b[BBlocks]; };
    struct Cvt { uint4 a[AParts][2],b[BBlocks][4]; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if constexpr((Tune&1024U)!=0U)return;
        if constexpr((Tune&2048U)==0U)
#pragma unroll
        for(unsigned j=0;j<AParts;++j) {
            if constexpr(Core!=0U) {
#pragma unroll
                for(unsigned q=0;q<4U;++q)r.a[j][q]=f16_load(a_src[j]+64U*step+4U*q);
            } else if(a_live[j])
#pragma unroll
            for(unsigned q=0;q<4U;++q)r.a[j][q]=f16_load(a_src[j]+64U*step+4U*q);
        }
        if constexpr((Tune&4096U)==0U)
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k)r.b[k]=down_block<Q51>(b_src,2U*step+BBlocks*b_part+k);
    };
    const auto convert=[&](unsigned step,const Regs& r,Cvt& cv) {
#pragma unroll
        for(unsigned j=0;j<AParts;++j) {
            cv.a[j][0]=f16_from_f32(r.a[j][0],r.a[j][1],a_scale[j]);
            cv.a[j][1]=f16_from_f32(r.a[j][2],r.a[j][3],a_scale[j]);
            if constexpr(Core!=0U)if(!a_live[j]){cv.a[j][0]=uint4{0U,0U,0U,0U};cv.a[j][1]=uint4{0U,0U,0U,0U};}
        }
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k) {
            const unsigned half=BBlocks*b_part+k;
            if constexpr(Q51) {
                wmma_v4i b0,b1;float d,m;
                down_operands<true>(r.b[k],2U*step+half,b0,b1,d,m);
                const unsigned c[8]{static_cast<unsigned>(b0.x),static_cast<unsigned>(b0.y),static_cast<unsigned>(b0.z),static_cast<unsigned>(b0.w),
                    static_cast<unsigned>(b1.x),static_cast<unsigned>(b1.y),static_cast<unsigned>(b1.z),static_cast<unsigned>(b1.w)};
                // Q5_1: d*q + m, q <= 31.
                if((Tune&64U)!=0U&&__all(f16_q51_exact(r.b[k].head.x))) {
#pragma unroll
                    for(unsigned q=0;q<8U;q+=2U) {
                        const uint2 x=f16_affine_u8_fast(c[q],d*16777216.0F,m),y=f16_affine_u8_fast(c[q+1U],d*16777216.0F,m);
                        cv.b[k][q/2U]={x.x,x.y,y.x,y.y};
                    }
                } else {
#pragma unroll
                    for(unsigned q=0;q<8U;q+=2U) {
                        const uint2 x=f16_affine_u8(c[q],d,m),y=f16_affine_u8(c[q+1U],d,m);
                        cv.b[k][q/2U]={x.x,x.y,y.x,y.y};
                    }
                }
            } else f16_q80_block(r.b[k],2U*step+half,cv.b[k]);
        }
    };
    const auto store=[&](const Cvt& cv,Stage& st) {
#pragma unroll
        for(unsigned j=0;j<AParts;++j){*st.a16(a_row[j],16U*a_part[j])=cv.a[j][0];*st.a16(a_row[j],16U*a_part[j]+8U)=cv.a[j][1];}
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k)
#pragma unroll
            for(unsigned q=0;q<4U;++q)*st.b16(b_col,8U*(4U*(BBlocks*b_part+k)+q))=cv.b[k][q];
    };
    const auto stage=[&](unsigned step,const Regs& r,unsigned buf) {
        Stage& st=sb[buf];
        if constexpr((Tune&512U)!=0U) {
#pragma unroll
            for(unsigned j=0;j<AParts;++j){*st.a16(a_row[j],16U*a_part[j])=r.a[j][0];*st.a16(a_row[j],16U*a_part[j]+8U)=r.a[j][2];}
#pragma unroll
            for(unsigned k=0;k<BBlocks;++k)for(unsigned q=0;q<4U;++q) {
                uint4 w;__builtin_memcpy(&w,&r.b[k],sizeof(w)<sizeof(r.b[k])?sizeof(w):sizeof(r.b[k]));
                *st.b16(b_col,8U*(4U*(BBlocks*b_part+k)+q))=w;
            }
            return;
        }
        Cvt cv;
        convert(step,r,cv);
        store(cv,st);
    };
    const unsigned lane=t&31U,wave=Core?static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(t/32U))):t/32U,i=lane&15U,h=lane>>4U,rh=wave/T::Groups,cg=wave%T::Groups;
    const unsigned rbase=64U*rh,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned cols[2]{32U*cg,32U*cg+16U};
    f32x8 acc[4][2]={};
    if constexpr((Core&128U)!=0U) {
        const unsigned smb=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(mb)));
        f16_pipeline_d<prefill_core2_depth(Core),Regs>(steps,fetch,stage,[&](unsigned){f16_step_rows_s<Core>(sb[0],rbase,smb,cols,i,acc);});
    } else if constexpr(Core!=0U)
        f16_by_mb(mb,[&]<unsigned MB>(){f16_pipeline_d<prefill_core2_depth(Core),Regs>(steps,fetch,stage,[&](unsigned){f16_step_rows_u<MB,Core>(sb[0],rbase,cols,i,acc);});});
    else f16_pipeline<Tune,Regs>(steps,fetch,stage,[&](unsigned){if constexpr((Tune&256U)==0U)f16_step_rows(sb[0],rbase,mb,cols,i,acc);});
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-exp_of(a));
        float* out=at(view.rawdots,row_of(a))+slot_of(a)*hidden+tile*Columns+32U*cg+i;
        if constexpr((Tune&32768U)!=0U) { // probe: raw-dot stores skipped (data-dependent, never taken in practice)
            if(__float_as_uint(acc[mi][0][e])==0x7fc12345U)out[0]=acc[mi][0][e]*unscale;
            if(__float_as_uint(acc[mi][1][e])==0x7fc12345U)out[16]=acc[mi][1][e]*unscale;
        } else if constexpr((Core&4U)!=0U){f16_store_global(out,acc[mi][0][e]*unscale);f16_store_global(out+16,acc[mi][1][e]*unscale);}
        else {out[0]=acc[mi][0][e]*unscale;out[16]=acc[mi][1][e]*unscale;}
    }
    if constexpr((Core&512U)!=0U)fuse_merge_tail<Batch,Columns,T::Threads>(fuse,view,count,tile,hidden,row_of);
    else (void)fuse;
