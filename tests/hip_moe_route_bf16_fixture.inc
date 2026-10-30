// Native candidate gate: production DA16 down -> compact route dots -> ordered MC tail.
// The CPU reference deliberately does not call the implementation's conversion helper.
std::uint16_t route_bf16_reference(float x) {
    const auto u=std::bit_cast<std::uint32_t>(x);
    if ((u&0x7fffffffU)>0x7f800000U) return static_cast<std::uint16_t>((u>>16U)|0x40U);
    const unsigned hi=u>>16U,lo=u&65535U;
    return static_cast<std::uint16_t>(hi+(lo>32768U||(lo==32768U&&(hi&1U))?1U:0U));
}
float route_bf16_widen(std::uint16_t x) { return std::bit_cast<float>(static_cast<unsigned>(x)<<16U); }
template<class T> T route_fixture_at(const std::vector<unsigned char>& bytes,std::uint64_t at) {
    T value;std::memcpy(&value,bytes.data()+at,sizeof(T));return value;
}
void route_fixture_require(bool ok,const char* what) { if(!ok) throw std::runtime_error(what); }
// The same storage epilogue as DA16, on exact synthetic FP32 dots. Single-threaded
// merge avoids cross-CTA header races in nonfinite cases; the production status logic is shared.
template<bool BF16>
__global__ void route_bf16_edge_kernel(const float* input,float* raw,const unsigned* routes,const float* weights,
    float* acc,std::uint16_t* output,unsigned* header,unsigned* status,const unsigned* unsafe) {
    for(unsigned s=threadIdx.x;s<10U;s+=blockDim.x) moe_route_store<BF16>(raw,s,input[s]);
    __syncthreads();
    if(threadIdx.x==0U) {
        const FixtureTailView view{{routes,0},{weights,0},{status,0},{raw,0},{acc,0},{output,0}};
        moe_prefill_merge_z_body<true,true,BF16>(view,header,1U,16U,10U,
            [](auto v,unsigned r){return fixture_row(v,r);},unsafe);
    }
}
void route_bf16_tests() {
    unsigned cases=0;
    kr::set_prefill_core2(0);kr::set_prefill_route(14);kr::set_prefill_da16(1);kr::set_prefill_small(0);
    // 48/208 rows select tunes 80/82; both have partial CSR batches with top_k=10.
    for(unsigned rows:{48U,208U}) for(unsigned gate:{12U,13U}) for(unsigned down:{7U,8U}) {
        Fixture f({"route BF16",16U,rows,gate,down,4U,true,true,false});
        const auto incumbent=f.run(3); // unchanged non-certificate DA16 writer
        f.view.unsafe=static_cast<unsigned*>(f.admission.base);
        const auto original=f.run(3);
        route_fixture_require(original==incumbent,"OFF certificate changed incumbent DA16 bytes");
        route_fixture_require(kr::prefill_certificate_produced(),"route BF16 original certificate missing");
        const auto raw=original[9];
        const auto weights=f.weights.read();
        f.view.route_bf16=true;
        const auto compact=f.run(3);
        route_fixture_require(kr::prefill_certificate_produced()&&kr::prefill_da16_ran(),"route BF16 writer not covered");
        route_fixture_require(kr::prefill_effective_tune(f.request,rows,down,3)==(rows==48U?80U:82U),"route BF16 tune coverage");
        route_fixture_require(route_fixture_at<unsigned>(compact.back(),0U)==0U,"route BF16 upstream status");
        for(unsigned r=0;r<rows;++r) {
            for(unsigned i=0;i<Fixture::K*Fixture::H;++i) {
                const float dot=route_fixture_at<float>(raw,r*f.rawdots.stride+4ULL*i);
                const auto stored=route_fixture_at<std::uint16_t>(compact[9],r*f.rawdots.stride+2ULL*i);
                route_fixture_require(stored==route_bf16_reference(dot),"DA16 compact route bit mismatch");
            }
            for(std::uint64_t i=2ULL*Fixture::K*Fixture::H;i<f.rawdots.stride;++i)
                route_fixture_require(compact[9][r*f.rawdots.stride+i]==0xa5U,"route BF16 wrote padded tail");
        }
        Buffer out(rows,2U*Fixture::H),record(1,sizeof(strix::DeferredMoeFirstError));
        for(bool bf16:{false,true}) for(unsigned tail:{0U,1U,2U}) {
            // tail 0 = MC mode 1; 1 = MC with accumulation for oracle inspection;
            // 2 = force unsafe: unchanged merge_z chain followed by finish/commit.
            f.view.route_bf16=bf16;(void)f.run(3);
            const unsigned unsafe=tail==2U?1U:0U;
            check(hipMemcpyAsync(f.view.unsafe,&unsafe,4U,hipMemcpyHostToDevice,f.stream),"route BF16 tail choice");
            out.poison(f.stream);record.poison(f.stream);
            if(bf16) {
                if(tail==0U)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<false,true>),dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,16U,Fixture::K,out.rows<std::uint16_t>());
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate_merge_commit_kernel<true,true>),dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,16U,Fixture::K,out.rows<std::uint16_t>());
            } else {
                if(tail==0U)
                    hipLaunchKernelGGL(candidate_merge_commit_kernel<false>,dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,16U,Fixture::K,out.rows<std::uint16_t>());
                else
                    hipLaunchKernelGGL(candidate_merge_commit_kernel<true>,dim3(10U,rows),dim3(256),0,f.stream,
                        f.view,f.view.csr.summary,Fixture::H,16U,Fixture::K,out.rows<std::uint16_t>());
            }
            hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,f.stream,f.view,f.view.csr.summary,rows,Fixture::H,
                out.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
            check(hipGetLastError(),"route BF16 merge/finish launch");check(hipStreamSynchronize(f.stream),"route BF16 merge/finish");
            const auto actual=out.read(),acc=f.clear.read(),statuses=f.status.read();
            unsigned header=0;check(hipMemcpy(&header,f.view.csr.summary,4U,hipMemcpyDeviceToHost),"route BF16 merge header");
            route_fixture_require(header==0U,"route BF16 finite merge header");
            for(unsigned r=0;r<rows;++r) {
                route_fixture_require(route_fixture_at<unsigned>(statuses,r*f.status.stride)==0U,"route BF16 row status");
                for(unsigned h=0;h<Fixture::H;++h) {
                    float sum=0.0F;
                    for(unsigned slot=0;slot<10U;++slot) {
                        float dot=route_fixture_at<float>(raw,r*f.rawdots.stride+4ULL*(slot*Fixture::H+h));
                        if(bf16)dot=route_bf16_widen(route_bf16_reference(dot));
                        const float w=route_fixture_at<float>(weights,r*f.weights.stride+4ULL*slot);
                        sum=std::fma(w,dot,sum);
                    }
                    route_fixture_require(route_fixture_at<std::uint16_t>(actual,r*out.stride+2ULL*h)==route_bf16_reference(sum),
                        "route BF16 CPU slot-order fmaf output mismatch");
                    if(tail!=0U) route_fixture_require(route_fixture_at<unsigned>(acc,r*f.clear.stride+4ULL*h)==std::bit_cast<unsigned>(sum),
                        "route BF16 CPU slot-order fmaf accumulation mismatch");
                }
            }
            ++cases;
        }
        f.view.route_bf16=false;
        route_fixture_require(f.run(3)==original,"route BF16 OFF changed original written bytes");
    }
    // RNE ties (both signs/parities), signed zeros, FP32/BF16 subnormals,
    // finite overflow, infinities and low-payload NaNs. Ten routes plus a poisoned tail.
    Buffer inputs(1,40U),raw(1,64U),routes(1,40U),weights(1,40U),acc(1,4U),output(1,2U),header(1,4U),status(1,4U),admission(1,4U),record(1,sizeof(strix::DeferredMoeFirstError));
    const unsigned rt[10]{0,1,2,3,4,5,6,7,8,9};float w[10];std::fill_n(w,10,1.0F);
    check(hipMemcpy(routes.base,rt,sizeof(rt),hipMemcpyHostToDevice),"edge routes");
    check(hipMemcpy(weights.base,w,sizeof(w),hipMemcpyHostToDevice),"edge weights");
    for(unsigned bits:{0U,0x80000000U,0x00000001U,0x00008000U,0x00018000U,0x007fffffU,0x00800000U,
        0x3f808000U,0x3f818000U,0xbf808000U,0xbf818000U,0x7f7fffffU,0xff7fffffU,0x7f800000U,0xff800000U,
        0x7f800001U,0xff800001U,0x7fc00000U,0x7fffffffU}) for(bool bf16:{false,true}) for(unsigned fallback:{0U,1U}) {
        float dots[10]{};dots[0]=std::bit_cast<float>(bits);
        check(hipMemcpy(inputs.base,dots,sizeof(dots),hipMemcpyHostToDevice),"edge input");
        check(hipMemset(header.base,0,4U),"edge header");check(hipMemset(status.base,0,4U),"edge status");
        raw.poison(nullptr);output.poison(nullptr);record.poison(nullptr);
        check(hipMemcpy(admission.base,&fallback,4U,hipMemcpyHostToDevice),"edge admission");
        if(bf16)
            hipLaunchKernelGGL(route_bf16_edge_kernel<true>,dim3(1),dim3(32),0,nullptr,static_cast<float*>(inputs.base),static_cast<float*>(raw.base),
                static_cast<unsigned*>(routes.base),static_cast<float*>(weights.base),static_cast<float*>(acc.base),
                static_cast<std::uint16_t*>(output.base),static_cast<unsigned*>(header.base),static_cast<unsigned*>(status.base),static_cast<unsigned*>(admission.base));
        else
            hipLaunchKernelGGL(route_bf16_edge_kernel<false>,dim3(1),dim3(32),0,nullptr,static_cast<float*>(inputs.base),static_cast<float*>(raw.base),
                static_cast<unsigned*>(routes.base),static_cast<float*>(weights.base),static_cast<float*>(acc.base),
                static_cast<std::uint16_t*>(output.base),static_cast<unsigned*>(header.base),static_cast<unsigned*>(status.base),static_cast<unsigned*>(admission.base));
        check(hipGetLastError(),"edge kernel");
        kr::PrefillRows v{};v.routes=routes.rows<const unsigned>();v.route_weights=weights.rows<const float>();
        v.status=status.rows<unsigned>();v.rawdots=raw.rows<float>();v.clear=acc.rows<float>();v.unsafe=static_cast<unsigned*>(admission.base);
        hipLaunchKernelGGL(candidate_finish_kernel,dim3(32),dim3(256),0,nullptr,v,static_cast<unsigned*>(header.base),1U,1U,
            output.rows<std::uint16_t>(),static_cast<unsigned*>(record.base));
        check(hipGetLastError(),"edge finish");check(hipStreamSynchronize(nullptr),"edge completion");
        float sum=0.0F;
        for(float dot:dots)sum=std::fma(1.0F,bf16?route_bf16_widen(route_bf16_reference(dot)):dot,sum);
        const auto sum_bits=std::bit_cast<unsigned>(sum);
        const unsigned expected_status=(sum_bits&0x7fffffffU)>=0x7f7f8000U?8U:0U;
        route_fixture_require(route_fixture_at<unsigned>(header.read(),0)==expected_status&&
            route_fixture_at<unsigned>(status.read(),0)==expected_status,"edge nonfinite status mismatch");
        // Nonfinite sums are refused before output publication, so the public output
        // remains the sentinel even when host/device NaN propagation payloads differ.
        const auto expected_output=expected_status?std::uint16_t{0xa5a5U}:route_bf16_reference(sum);
        route_fixture_require(route_fixture_at<std::uint16_t>(output.read(),0)==expected_output,"edge output bits mismatch");
        if(!std::isnan(sum)) route_fixture_require(route_fixture_at<unsigned>(acc.read(),0)==sum_bits,"edge accumulation bits mismatch");
        const auto stored=raw.read();
        if(bf16)route_fixture_require(route_fixture_at<std::uint16_t>(stored,0)==route_bf16_reference(dots[0]),"edge route rounding bits");
        else route_fixture_require(route_fixture_at<unsigned>(stored,0)==bits,"edge OFF route bits");
        for(unsigned i=bf16?20U:40U;i<64U;++i)route_fixture_require(stored[i]==0xa5U,"edge padded tail overwritten");
        ++cases;
    }
    kr::set_prefill_core2(-1);kr::set_prefill_route(-1);kr::set_prefill_da16(-1);kr::set_prefill_small(-1);
    std::printf("route BF16: %u down/merge and edge cases passed; native model qualification still required\n",cases);
}
