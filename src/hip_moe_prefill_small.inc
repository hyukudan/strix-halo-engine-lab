// qpf2k/moe-mi SMALL (STRIX_QPF2K_MOE_SMALL=<rows>, 1..128; default 0 = the base kernels): count-class prefetch depth.
// Included by hip_moe_q4k_a8_known_rows.hip inside its anonymous namespace after hip_moe_prefill_p1.inc. Prefill only, tunes
// 80/82, without STRIX_QPF2K_MOE core2/P1 and without DA16.
//
// Cause (measured attribution): a CTA whose batch holds few assignments has little WMMA per K64
// step (MB <= 3 row tiles in half of its waves, none in the other half), so the base pipeline's one-step global prefetch no
// longer covers the load latency: the K loop of a small batch is latency-bound and costs nearly the slot time of a full one.
// Full batches hide it already (core2 depth 3 everywhere was +4.5 %, t1b), so the deeper pipeline is used ONLY for batches
// with at most `small` assignments, chosen per CTA (CTA-uniform: the batch's CSR count).
// Same bits: a small batch runs the core2 depth-3 body (Core 17: branch-free loads zeroed at staging, three register stages,
// the same plane steps, K order, operands and epilogue; bit-identical to the base in the core2 gates 1,309/0 and 1,683/0), a
// larger one the base body (Core 0). Each output element is produced by exactly one CTA, so results do not depend on the
// split. Gate/up only (one launch, as the base): VGPR = max of the two bodies (core 17 at tune 82: 178/192 with launch bounds
// 8 waves/SIMD, the residency of the base; scratch 0); same LDS image (static_assert). The down keeps the base kernel: its
// depth-3 form spills 36 B for Q8_0 (emit e3) and the down is memory-bound, not latency-bound by the attribution.
inline int& prefill_small_override(){static int v=-1;return v;}
inline unsigned prefill_small() {
    static const unsigned env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_SMALL");const int n=v?std::atoi(v):0;
        return n>=1&&n<=128?static_cast<unsigned>(n):0U;}();
    const int o=prefill_small_override();
    return o<0?env:(o>=1&&o<=128?static_cast<unsigned>(o):0U);
}
inline bool& prefill_small_flag(){static thread_local bool v=false;return v;} // test hook: last call took SMALL
constexpr unsigned small_core=17U; // core2 bit 0 + bit 4: prefetch depth 3
template<bool Q5K,unsigned Tune>
__global__ __launch_bounds__((F16TileX<Tune,0U>::Threads),(Tune==82U?8U:1U)) void project_prefill_f16x_small(const unsigned char* gate,
    const unsigned char* up,PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned small) {
    using L=F16LdsOf<F16TileX<Tune,0U>>;
    static_assert(std::is_same_v<typename L::Stage,typename F16LdsOf<F16TileX<Tune,small_core>>::Stage>&&
        std::is_same_v<typename L::Rows,typename F16LdsOf<F16TileX<Tune,small_core>>::Rows>,"SMALL: both bodies share one LDS image");
    __shared__ __align__(16) typename L::Stage sb[1];
    __shared__ typename L::Rows rows;
    const unsigned batch=blockIdx.y;
    const auto& csr=view.csr;
    // CTA-uniform class; a batch past the count (or a failed CSR, summary[2] = 0) takes the base body, whose guard returns.
    const bool is_small=batch<csr.summary[2]&&csr.begins[batch+1U]-csr.begins[batch]<=small;
    if(is_small)project_f16x_body<Q5K,Tune,small_core>(gate,up,view,hidden,outputs,a16,batch,blockIdx.x,sb,rows);
    else project_f16x_body<Q5K,Tune,0U>(gate,up,view,hidden,outputs,a16,batch,blockIdx.x,sb,rows);
}
