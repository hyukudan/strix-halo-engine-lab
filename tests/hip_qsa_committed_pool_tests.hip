#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/qsa_committed_pool.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <string_view>
namespace {
void check(hipError_t e) {if(e!=hipSuccess){std::fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));std::exit(2);}}
void require(bool ok,const char* msg) {if(!ok){std::fprintf(stderr,"FAIL: %s\n",msg);std::exit(1);}}
struct Buffer {
    void* p=nullptr;
    explicit Buffer(std::size_t bytes){check(hipMalloc(&p,bytes));}
    ~Buffer(){if(p)check(hipFree(p));}
    template<class T>T* as(){return static_cast<T*>(p);}
};
// Exercise retained storage after real source changes, rather than resetting the
// service before every comparison. Target and predictor have independent bytes.
std::uint64_t lifecycle_fixture(const strix::ModelContract& model,const strix::RuntimeTensorBinding& gain,
    float* queries,float* scores_a,float* scores_b,std::uint32_t* ids_a,std::uint32_t* ids_b,
    hipStream_t stream_a,hipStream_t stream_b) {
    using namespace strix;
    constexpr std::uint32_t capacity=4096,stride=capacity/4,selected_stride=2051;
    Buffer target_raw(capacity*256),predictor_raw(capacity*256);
    auto owner=HipQsaCommittedPool::create(mtp_qsa_pool_plan(capacity));
    auto slot=HipQsaCommittedPool::create(mtp_qsa_pool_plan(capacity));
    require(owner && slot,"lifecycle service allocations");
    std::uint64_t cases=0;
    const auto rewrite=[&](const auto& service,bool predictor,Buffer& raw,std::uint32_t first,std::uint32_t rows,unsigned version) {
        service->invalidate(predictor,first); // Before the external restore/replay copies.
        std::vector<std::uint16_t> words(std::size_t{rows}*128);
        for(std::size_t i=0;i<words.size();++i)
            words[i]=static_cast<std::uint16_t>(0x3d00U+(i*7U+version*29U)%0x400U);
        check(hipMemcpy(raw.as<std::uint16_t>()+std::size_t{first}*128,words.data(),words.size()*2,hipMemcpyHostToDevice));
    };
    const auto compare=[&](const auto& service,bool predictor,Buffer& raw,std::uint32_t first,std::uint32_t rows,unsigned accepted) {
        std::array<HipQsaPoolRowDescriptor,8> a{},b{};
        for(unsigned r=0;r<rows;++r) {
            const auto visible=first+r+1;
            a[r]={queries+r*512,scores_a+r*stride,ids_a+r*selected_stride,visible,visible/4,512,0,nullptr};
            b[r]=a[r];b[r].scores=scores_b+r*stride;b[r].selected=ids_b+r*selected_stride;
        }
        const auto raw_address=reinterpret_cast<std::uintptr_t>(raw.p);
        launch_hip_qsa_committed_pool_probe(false,predictor,service,model,raw_address,capacity,gain,
            first,rows,std::span<const HipQsaPoolRowDescriptor>(a.data(),rows),stream_a);
        launch_hip_qsa_committed_pool_probe(true,predictor,service,model,raw_address,capacity,gain,
            first,rows,std::span<const HipQsaPoolRowDescriptor>(b.data(),rows),stream_b);
        check(hipStreamSynchronize(stream_a));check(hipStreamSynchronize(stream_b));
        for(unsigned r=0;r<rows;++r) {
            std::vector<std::uint32_t> x(a[r].complete_blocks),y(x.size());
            check(hipMemcpy(x.data(),a[r].scores,x.size()*4,hipMemcpyDeviceToHost));
            check(hipMemcpy(y.data(),b[r].scores,y.size()*4,hipMemcpyDeviceToHost));
            require(x==y,"retained keys after window switch/rewrite: mutated score bytes");
            x.resize(std::min(a[r].complete_blocks,512U)*4+a[r].visible%4);y.resize(x.size());
            check(hipMemcpy(x.data(),a[r].selected,x.size()*4,hipMemcpyDeviceToHost));
            check(hipMemcpy(y.data(),b[r].selected,y.size()*4,hipMemcpyDeviceToHost));
            require(x==y,"retained keys after window switch/rewrite: selection bytes");
        }
        service->commit(predictor,first+accepted);++cases;
    };
    for(unsigned residue=0;residue<4;++residue) {
        owner->begin_epoch();slot->begin_epoch();
        rewrite(owner,false,target_raw,0,capacity,residue);
        rewrite(owner,true,predictor_raw,0,capacity,41+residue);
        const auto first=2052+residue;
        compare(owner,false,target_raw,first,8,2);
        // R8 pooled a rejected suffix. Serial fallback changes it, then R2/R4
        // start beyond the stale block that a window-local clamp would retain.
        rewrite(owner,false,target_raw,first+2,12,77+residue);
        for(unsigned row=2;row<10;++row)compare(owner,false,target_raw,first+row,1,1);
        compare(owner,false,target_raw,first+10,2,2);
        compare(owner,false,target_raw,first+12,4,3);
        compare(owner,true,predictor_raw,2047+residue,1,1);
        rewrite(owner,true,predictor_raw,2047+residue,8,99+residue);
        compare(owner,true,predictor_raw,2047+residue,1,1);
        // A second slot uses different source words, even with the same role
        // and stream. Returning to the first service must retain its own bytes.
        compare(slot,false,predictor_raw,first,4,4);
        compare(owner,false,target_raw,first+15,2,2);
        rewrite(owner,false,target_raw,0,capacity,111+residue);
        rewrite(owner,true,predictor_raw,0,capacity,133+residue);
        owner->commit(false,2100);owner->commit(true,2099);
        compare(owner,false,target_raw,2100,4,4);
        compare(owner,true,predictor_raw,2099,1,1);
        owner->cancel();owner->begin_epoch();
        rewrite(owner,false,target_raw,0,capacity,155+residue);
        compare(owner,false,target_raw,2049,8,3);
    }
    return cases;
}
}
int main(int argc,char** argv) {
    int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess || !devices) {std::puts("NO-REFERENCE: no HIP device");return 77;}
    using namespace strix;
    constexpr std::uint32_t maximum=262144,capacity=4*maximum+16,selected_capacity=2051;
    auto model=ModelContract::qwen38_flash_next();
    if(argc==3 && std::string_view(argv[1])=="--yarn-factor")
        model.qwen_rotary_table=std::make_shared<HipQwenRotaryTable>(RotaryDescriptor::yarn(static_cast<unsigned>(std::stoul(argv[2]))));
    else require(argc==1,"usage: fixture [--yarn-factor 2|4]");
    const auto plan=mtp_qsa_pool_plan(capacity);auto owner=HipQsaCommittedPool::create(plan);
    require(static_cast<bool>(owner),"admitted service allocation");
    require(owner->reserved_bytes()==plan.reserved_bytes,"one exact reservation");
    Buffer raw(std::size_t{capacity}*256),norm(128*4),queries(8*512*4),scores_a(std::size_t{8}*(maximum+4)*4),
        scores_b(std::size_t{8}*(maximum+4)*4),ids_a(8*selected_capacity*4),ids_b(8*selected_capacity*4);
    std::vector<std::uint16_t> raw_words(std::size_t{capacity}*128);
    for(std::size_t i=0;i<raw_words.size();++i)raw_words[i]=static_cast<std::uint16_t>(0x3e00U+(i*13U)%0x300U);
    check(hipMemcpy(raw.p,raw_words.data(),raw_words.size()*2,hipMemcpyHostToDevice));
    std::array<float,128> gains{};gains.fill(1);check(hipMemcpy(norm.p,gains.data(),sizeof(gains),hipMemcpyHostToDevice));
    RuntimeTensorBinding gain{};gain.address=reinterpret_cast<std::uintptr_t>(norm.p);gain.bytes=sizeof(gains);gain.allocation_generation=1;
    hipStream_t stream_a{},stream_b{};check(hipStreamCreateWithFlags(&stream_a,hipStreamNonBlocking));check(hipStreamCreateWithFlags(&stream_b,hipStreamNonBlocking));
    std::uint64_t cases=0;
    for(const auto blocks:{0U,1U,31U,32U,33U,511U,512U,513U,8192U,32768U,65536U,98304U,262144U})
    for(const auto rows:{1U,2U,4U,8U})for(std::uint32_t residue=0;residue<4;++residue) {
        const auto first=4U*(blocks?blocks-1U:0U)+residue;
        if(model.qwen_rotary_table && first+rows>model.effective_max_positions()) continue;
        std::array<float,8*512> q{};
        for(std::size_t i=0;i<q.size();++i)q[i]=static_cast<float>(static_cast<int>(i%17)-8)*0.03125F;
        if(blocks==31)q.fill(-0.0F);
        if(blocks==32)q.fill(std::bit_cast<float>(0x7fc12345U));
        if(blocks==33)q.fill(std::bit_cast<float>(0x7f800000U));
        if(blocks==511)q.fill(std::bit_cast<float>(0x7f7fffffU));
        check(hipMemcpy(queries.p,q.data(),sizeof(q),hipMemcpyHostToDevice));
        for(const bool selection:{false,true}) {
            std::array<HipQsaPoolRowDescriptor,8> a{},b{};
            for(unsigned r=0;r<rows;++r) {
                const auto visible=first+r+1;
                a[r]={queries.as<float>()+r*512,scores_a.as<float>()+std::size_t{r}*(maximum+4),ids_a.as<std::uint32_t>()+r*selected_capacity,
                    visible,visible/4,selection?512U:UINT32_MAX,0,nullptr};
                b[r]=a[r];b[r].scores=scores_b.as<float>()+std::size_t{r}*(maximum+4);b[r].selected=ids_b.as<std::uint32_t>()+r*selected_capacity;
            }
            owner->begin_epoch();
            launch_hip_qsa_committed_pool_probe(false,rows==1,owner,model,reinterpret_cast<std::uintptr_t>(raw.p),capacity,gain,
                first,rows,std::span<const HipQsaPoolRowDescriptor>(a.data(),rows),stream_a);
            launch_hip_qsa_committed_pool_probe(true,rows==1,owner,model,reinterpret_cast<std::uintptr_t>(raw.p),capacity,gain,
                first,rows,std::span<const HipQsaPoolRowDescriptor>(b.data(),rows),stream_b);
            check(hipStreamSynchronize(stream_a));check(hipStreamSynchronize(stream_b));
            for(unsigned r=0;r<rows;++r) {
                const auto count=a[r].complete_blocks;
                std::vector<std::uint32_t> x(count),y(count);
                if(count){check(hipMemcpy(x.data(),a[r].scores,count*4,hipMemcpyDeviceToHost));check(hipMemcpy(y.data(),b[r].scores,count*4,hipMemcpyDeviceToHost));}
                require(x==y,"raw/packed preselection or mutated score bytes");
                if(selection) {
                    const auto selected=std::min(count,512U)*4+a[r].visible%4;
                    x.resize(selected);y.resize(selected);
                    check(hipMemcpy(x.data(),a[r].selected,selected*4,hipMemcpyDeviceToHost));check(hipMemcpy(y.data(),b[r].selected,selected*4,hipMemcpyDeviceToHost));
                    require(x==y,"raw/packed selected IDs and chronological tails");
                }
            }
            owner->commit(rows==1,first+rows/2); // rejected suffix intersects all block residues
            owner->invalidate(rows==1,first); // replay/restore cannot retain proposal bytes
            ++cases;
        }
    }
    // Recorded completion handoff: writer on B must wait for the lease's A reads/writes.
    owner->begin_epoch();
    {
        auto lease=owner->before_rewrite(false,reinterpret_cast<std::uintptr_t>(raw.p),std::uint64_t{capacity}*256,1,gain,0,model,0,1,stream_a,true);
        check(hipMemsetAsync(const_cast<std::uint16_t*>(lease->key().packed),0x5a,256,stream_a));
    }
    {
        auto lease=owner->before_rewrite(false,reinterpret_cast<std::uintptr_t>(raw.p),std::uint64_t{capacity}*256,1,gain,0,model,0,1,stream_b,true);
        std::array<std::uint16_t,128> words{};
        check(hipMemcpyAsync(words.data(),lease->key().packed,256,hipMemcpyDeviceToHost,stream_b));check(hipStreamSynchronize(stream_b));
        require(std::all_of(words.begin(),words.end(),[](auto v){return v==0x5a5a;}),"cross-stream lease completion");
    }
    owner->cancel();bool refused=false;
    try{(void)owner->selector_lease(stream_a);}catch(const std::exception&){refused=true;}require(refused,"cancel stops new leases");
    owner->begin_epoch();{auto scratch=owner->selector_lease(stream_b);require(scratch->scratch()!=nullptr,"exclusive K1 scratch seam");}
    owner.reset();
    cases+=lifecycle_fixture(model,gain,queries.as<float>(),scores_a.as<float>(),scores_b.as<float>(),
        ids_a.as<std::uint32_t>(),ids_b.as<std::uint32_t>(),stream_a,stream_b);
    check(hipStreamDestroy(stream_a));check(hipStreamDestroy(stream_b));
    std::printf("PASS production MTP shared pool raw/packed rows, selections, frontiers and stream handoffs cases=%llu\n",static_cast<unsigned long long>(cases));
}
