// Component-only same-TU access. No production API accepts noncanonical shapes.
#include "../src/hip_hyper_mix_launcher.hip"
#include "strix/hip_setup_quiescence.hpp"
#include <bit>
#include <iostream>
#include <memory>

namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void checked(hipError_t error) {if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error));}
struct Buffer {
    strix::HipSetupQuiescenceGuard guard;void* allocation=nullptr;std::size_t bytes;
    explicit Buffer(std::size_t count):bytes(count) {guard.run([&]{checked(hipMalloc(&allocation,count+512U));checked(hipMemset(allocation,0xa5,count+512U));});}
    ~Buffer(){guard.recover_or_fail_stop();if(allocation)static_cast<void>(hipFree(allocation));}
    void* data()const{return static_cast<std::byte*>(allocation)+256U;}
    void put(const void* source){guard.run([&]{checked(hipMemcpy(data(),source,bytes,hipMemcpyHostToDevice));});}
    std::vector<std::byte> read()const{std::vector<std::byte> result(bytes);guard.run([&]{checked(hipMemcpy(result.data(),data(),bytes,hipMemcpyDeviceToHost));});return result;}
    void guards()const{std::array<std::byte,256> front{},back{};guard.run([&]{checked(hipMemcpy(front.data(),allocation,256,hipMemcpyDeviceToHost));checked(hipMemcpy(back.data(),static_cast<std::byte*>(data())+bytes,256,hipMemcpyDeviceToHost));});require(std::all_of(front.begin(),front.end(),[](auto b){return b==std::byte{0xa5};}) && std::all_of(back.begin(),back.end(),[](auto b){return b==std::byte{0xa5};}),"next-tile outer guard changed");}
};
struct Stream {
    strix::HipSetupQuiescenceGuard guard;hipStream_t value=nullptr;
    Stream(){guard.run([&]{checked(hipStreamCreateWithFlags(&value,hipStreamNonBlocking));});}
    ~Stream(){guard.recover_or_fail_stop();if(value)static_cast<void>(hipStreamDestroy(value));}
};
struct Graph {
    hipGraph_t value=nullptr;hipGraphExec_t executable=nullptr;
    ~Graph(){if(executable)static_cast<void>(hipGraphExecDestroy(executable));if(value)static_cast<void>(hipGraphDestroy(value));}
};
void set_scale(strix::Q8Block& block,std::uint16_t bits){std::memcpy(&block.scale,&bits,2);}
void run(unsigned width,unsigned rank,bool injection,unsigned kind) {
    constexpr unsigned streams=4;
    std::vector<float> input(width),inject(static_cast<std::size_t>(width)*streams);
    std::vector<strix::Q8Block> down(static_cast<std::size_t>(width/32U)*rank);
    for(unsigned i=0;i<width;++i)input[i]=std::ldexp(static_cast<float>(static_cast<int>((i*73U)%1023U)-511)/512.0F,static_cast<int>(i%11U)-5);
    for(std::size_t i=0;i<inject.size();++i)inject[i]=std::nextafter(static_cast<float>(static_cast<int>((i*17U)%63U)-31)/128.0F,i%2U?INFINITY:-INFINITY);
    constexpr std::array<std::uint16_t,8> scales{0U,0x8000U,1U,0x03ffU,0x3401U,0xb801U,0x3c01U,0x7bffU};
    for(std::size_t b=0;b<down.size();++b){set_scale(down[b],scales[(b*3U)%scales.size()]);for(unsigned j=0;j<32;++j)down[b].values[j]=static_cast<std::int8_t>(static_cast<int>((b*29U+j*11U)%256U)-128);}
    if(kind==1) {
        // Two terms in one serial chain: unfused multiply/add produces zero.
        std::fill(input.begin(),input.end(),0.0F);const float x=std::nextafter(1.0F,2.0F),coefficient=1.0009765625F*127.0F;
        volatile float product=x*coefficient;input[0]=-product;input[256]=x;
        for(auto& b:down){set_scale(b,0x3c00U);std::fill(std::begin(b.values),std::end(b.values),0);}
        for(unsigned row=0;row<rank;++row){auto* b=down.data()+static_cast<std::size_t>(row)*(width/32U);b[0].values[0]=1;set_scale(b[8],0x3c01U);b[8].values[0]=127;}
        require(std::fma(x,coefficient,-product)!=0.0F,"FMA witness is not discriminating");
    }
    if(kind==2 || kind==3){std::fill(input.begin(),input.end(),kind==2?-0.0F:-std::numeric_limits<float>::denorm_min());for(auto& b:down){set_scale(b,0x3800U);std::fill(std::begin(b.values),std::end(b.values),1);}}
    if(kind==4 || kind==5)for(auto& b:down)set_scale(b,kind==4?0x7e01U:0x7c01U);
    if(kind==6)for(auto& b:down){set_scale(b,0x7c00U);std::fill(std::begin(b.values),std::end(b.values),0);}
    if(kind==7){input[0]=std::bit_cast<float>(0xffc12345U);input[128]=std::bit_cast<float>(0x7fc54321U);}
    if(kind==8){input[0]=std::bit_cast<float>(0xffc12345U);for(unsigned row=0;row<rank;++row)set_scale(down[static_cast<std::size_t>(row)*(width/32U)],0x7e02U);}
    if(kind==9){inject[0]=std::bit_cast<float>(0xffc12345U);inject[128]=std::bit_cast<float>(0x7fc54321U);}
    if(kind==10){std::fill(input.begin(),input.end(),std::numeric_limits<float>::max());for(auto& b:down){set_scale(b,0x7bffU);std::fill(std::begin(b.values),std::end(b.values),-128);}}
    if(kind==11){input[width-1U]=std::bit_cast<float>(0x7fa12345U);inject[width-1U]=std::bit_cast<float>(0xffc54321U);}
    if(kind==12){input[0]=std::bit_cast<float>(0x7fc54321U);input[128]=std::bit_cast<float>(0xffc12345U);}
    if(kind==13){input[0]=std::bit_cast<float>(0x7fc54321U);for(unsigned row=0;row<rank;++row)set_scale(down[static_cast<std::size_t>(row)*(width/32U)],0xfe02U);}
    if(kind==14){inject[0]=std::bit_cast<float>(0x7fc54321U);inject[128]=std::bit_cast<float>(0xffc12345U);}
    if(kind==15){input[0]=0.0F;inject[0]=std::numeric_limits<float>::infinity();}
    if(kind==16){input[0]=std::bit_cast<float>(0xffc12345U);inject[0]=std::bit_cast<float>(0x7fa54321U);}
    Buffer d_input(input.size()*4U),d_down(down.size()*sizeof(strix::Q8Block)),d_inject(inject.size()*4U),low(rank*4U),inj(streams*4U);Stream stream;
    d_input.put(input.data());d_down.put(down.data());d_inject.put(inject.data());
    const auto input_before=d_input.read(),down_before=d_down.read(),inject_before=d_inject.read();
    const auto blocks=(rank+7U)/8U;
    const auto enqueue=[&](bool next) {
        auto* low_output=static_cast<float*>(low.data());auto* injection_output=injection?static_cast<float*>(inj.data()):nullptr;
        if(next)hipLaunchKernelGGL(strix::next_tile_hyper_projection_kernel,dim3(blocks+(injection?1U:0U)),dim3(128),0,stream.value,
            static_cast<float*>(d_input.data()),d_down.data(),width,streams,rank,low_output,injection?d_inject.data():nullptr,injection_output,blocks);
        else hipLaunchKernelGGL(strix::batched_typed_hyper_projection_kernel,dim3(blocks+(injection?1U:0U)),dim3(strix::kStagedThreads),0,stream.value,
            static_cast<float*>(d_input.data()),d_down.data(),width,streams,rank,low_output,injection?d_inject.data():nullptr,injection_output,blocks);
        checked(hipGetLastError());
    };
    const auto poison=[&]{checked(hipMemsetAsync(low.data(),0xa5,low.bytes,stream.value));checked(hipMemsetAsync(inj.data(),0x5a,inj.bytes,stream.value));};
    const auto semantic=[&]{auto result=low.read();const auto extra=inj.read();result.insert(result.end(),extra.begin(),extra.end());return result;};
    stream.guard.run([&]{poison();enqueue(false);});const auto expected=semantic();
    if(kind==1){float first=0;std::memcpy(&first,expected.data(),sizeof(first));require(std::isfinite(first) && first!=0.0F,"actual FMA witness must survive FP32 SiLU epilogue");}
    for(unsigned replay=0;replay<3;++replay){stream.guard.run([&]{poison();enqueue(true);});const auto actual=semantic();
        if(actual!=expected){const auto mismatch=std::mismatch(actual.begin(),actual.end(),expected.begin());std::cerr<<"NEXT_TILE_MISMATCH width="<<width<<" rank="<<rank<<" inject="<<injection<<" case="<<kind<<" replay="<<replay<<" byte="<<(mismatch.first-actual.begin())<<'\n';throw std::runtime_error("next-tile eager full F32 bits differ");}}
    Graph graph;checked(hipStreamBeginCapture(stream.value,hipStreamCaptureModeThreadLocal));
    try{enqueue(true);}catch(...){static_cast<void>(hipStreamEndCapture(stream.value,&graph.value));throw;}
    checked(hipStreamEndCapture(stream.value,&graph.value));std::size_t nodes=0;checked(hipGraphGetNodes(graph.value,nullptr,&nodes));require(nodes==1U,"component capture must contain exactly one projection");
    checked(hipGraphInstantiate(&graph.executable,graph.value,nullptr,nullptr,0));
    for(unsigned replay=0;replay<3;++replay){stream.guard.run([&]{poison();checked(hipGraphLaunch(graph.executable,stream.value));});require(semantic()==expected,"next-tile component graph F32 bits differ");}
    require(d_input.read()==input_before && d_down.read()==down_before && d_inject.read()==inject_before,"next-tile borrowed weights/input mutated");
    for(const auto* owner:{&d_input,&d_down,&d_inject,&low,&inj})owner->guards();
    if(!injection){const auto output=inj.read();require(std::all_of(output.begin(),output.end(),[](auto b){return b==std::byte{0x5a};}),"null-injection arm wrote injection owner");}
    std::cout<<"NEXT_TILE_COMPONENT width="<<width<<" rank="<<rank<<" injection="<<injection<<" case="<<kind<<" full_F32_bits_exact=true production_shape_admission=false\n";
}
}
int main()try{
    for(unsigned width:{32U,96U,128U,256U,288U,640U,10240U})for(unsigned rank:{1U,7U,8U,9U})for(bool inject:{false,true})run(width,rank,inject,0);
    run(10240,320,true,0);run(10240,320,false,0);
    for(unsigned kind=1;kind<=16;++kind)run(640,9,true,kind);
    std::cout<<"HC next-tile component fixtures passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
