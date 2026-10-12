#include <hip/hip_runtime.h>
#include "strix/integer_projection.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ip=strix::integer_projection;
namespace lp=strix::low_precision;
namespace {
using int2=int __attribute__((ext_vector_type(2)));
using int4=int __attribute__((ext_vector_type(4)));
using int8=int __attribute__((ext_vector_type(8)));
__device__ __constant__ int book[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
void check(hipError_t e,const char* op){if(e!=hipSuccess)throw std::runtime_error(std::string(op)+": "+hipGetErrorString(e));}
void require(bool condition,const char* msg){if(!condition)throw std::runtime_error(msg);}
bool same_bits(const std::vector<double>& expected,const std::vector<double>& actual,std::size_t start) {
    if(actual.size()<start+expected.size())return false;
    for(std::size_t j=0;j<expected.size();++j)
        if(std::bit_cast<std::uint64_t>(expected[j])!=std::bit_cast<std::uint64_t>(actual[start+j]))return false;
    return true;
}
template<class T> class Buffer {
public:
    explicit Buffer(std::size_t n):count(n){check(hipMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)),"allocate");}
    // Never free a possibly in-flight allocation after an unproved completion.
    // This one-stream experiment intentionally retains it on query failure.
    ~Buffer(){if(data&&hipStreamQuery(hipStreamLegacy)==hipSuccess)(void)hipFree(data);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& x){require(x.size()==count,"upload size");check(hipMemcpy(data,x.data(),count*sizeof(T),hipMemcpyHostToDevice),"upload");}
    std::vector<T> get()const{std::vector<T>x(count);check(hipMemcpy(x.data(),data,count*sizeof(T),hipMemcpyDeviceToHost),"readback");return x;}
    T* data=nullptr;std::size_t count;
};
__global__ void wave_size(unsigned* size){if(!threadIdx.x)size[0]=static_cast<unsigned>(warpSize);}
// One thread per actual row/group. Canonical sequential absmax and exact
// binary64 division; no cross-row statistics, reciprocal, fast math or atomics.
__global__ void prepare_kernel(const double* input,std::uint8_t* codes,double* scales,
    std::int32_t* sums,unsigned* error,std::uint32_t rows,std::uint32_t k,bool a4) {
    const std::uint64_t group=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(group>=static_cast<std::uint64_t>(rows)*(k/32U))return;
    const auto start=group*32U;
    double maximum=0;
    for(unsigned j=0;j<32;++j){const double x=input[start+j];
        if(!isfinite(x)){atomicOr(error,1U);return;}maximum=fmax(maximum,fabs(x));}
    const int qmax=a4?7:127;
    const double scale=maximum==0?1.0:maximum/static_cast<double>(qmax);
    if(!isfinite(scale)||scale<=0){atomicOr(error,2U);return;}
    int sum=0;
    for(unsigned j=0;j<32;++j) {
        const double x=input[start+j]/scale;
        const double lower=floor(x),fraction=x-lower;
        int q=static_cast<int>(lower);
        if(fraction>0.5||(fraction==0.5&&q%2!=0))++q;
        q=q>qmax?qmax:(q< -qmax?-qmax:q);
        codes[start+j]=static_cast<std::uint8_t>(static_cast<unsigned>(q)&(a4?15U:255U));sum+=q;
    }
    scales[group]=scale;sums[group]=sum;
}
template<bool Four,bool IQ,bool A4>
__global__ void project_kernel(const std::uint8_t* activations,const double* activation_scales,
    const std::int32_t* sums,const std::uint8_t* weights,const double* weight_scales,
    double* output,std::int32_t* integer_groups,unsigned* error,
    std::uint32_t rows,std::uint32_t columns,std::uint32_t k) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error This experimental fragment ABI is only built for gfx1151.
#endif
    const unsigned lane=threadIdx.x; // exactly one Wave32 per workgroup
    const std::uint32_t input_row=blockIdx.y*16U+lane%16U;
    const std::uint32_t column=blockIdx.x*16U+lane%16U;
    const unsigned groups=k/32U;
    double result[8]={};
    for(unsigned g=0;g<groups;++g) {
        int8 accumulator{};
        for(unsigned panel=0;panel<2;++panel) {
            std::uint32_t aw[4]={},bw[4]={};
            constexpr unsigned bits=Four?4U:8U,per=32U/bits,mask=(1U<<bits)-1U;
            for(unsigned j=0;j<16;++j) {
                const auto kk=g*32U+panel*16U+j;
                int av=0,bv=0;
                if(input_row<rows) {
                    const unsigned code=activations[static_cast<std::uint64_t>(input_row)*k+kk];
                    if constexpr(A4)av=code<8U?static_cast<int>(code):static_cast<int>(code)-16;
                    else av=code<128U?static_cast<int>(code):static_cast<int>(code)-256;
                }
                if(column<columns) {
                    const auto packed=weights[(static_cast<std::uint64_t>(column)*groups+g)*16U+j];
                    const unsigned code=panel==0?packed&15U:packed>>4U;
                    if constexpr(IQ)bv=book[code];else bv=static_cast<int>(code);
                } // mask AFTER interpretation: IQ index zero is not padding.
                aw[j/per]|=(static_cast<unsigned>(av)&mask)<<((j%per)*bits);
                bw[j/per]|=(static_cast<unsigned>(bv)&mask)<<((j%per)*bits);
            }
            if constexpr(Four) {
                int2 a{},b{};for(unsigned v=0;v<2;++v){a[v]=__builtin_bit_cast(int,aw[v]);b[v]=__builtin_bit_cast(int,bw[v]);}
                accumulator=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,false,b,accumulator,false);
            } else {
                int4 a{},b{};for(unsigned v=0;v<4;++v){a[v]=__builtin_bit_cast(int,aw[v]);b[v]=__builtin_bit_cast(int,bw[v]);}
                accumulator=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,IQ,b,accumulator,false);
            }
        }
        // Accumulator VGPR v maps row 2*v+lane/16, column lane%16.
        // Every group is bounded by 32*127*127=516128. No saturation.
        for(unsigned v=0;v<8;++v) {
            const auto row=blockIdx.y*16U+2U*v+lane/16U;
            if(row>=rows||column>=columns)continue;
            std::int64_t corrected=accumulator[v];
            if constexpr(!IQ)corrected-=8LL*sums[static_cast<std::uint64_t>(row)*groups+g];
            if(corrected< -516128LL||corrected>516128LL){atomicOr(error,4U);continue;}
            const auto index=static_cast<std::uint64_t>(row)*columns+column;
            integer_groups[index*groups+g]=static_cast<std::int32_t>(corrected);
            const double scale=weight_scales[static_cast<std::uint64_t>(column)*groups+g]*
                activation_scales[static_cast<std::uint64_t>(row)*groups+g];
            result[v]+=scale*static_cast<double>(corrected);
            if(!isfinite(result[v]))atomicOr(error,8U);
        }
    }
    for(unsigned v=0;v<8;++v){const auto row=blockIdx.y*16U+2U*v+lane/16U;
        if(row<rows&&column<columns)output[static_cast<std::uint64_t>(row)*columns+column]=result[v];}
}
std::vector<std::byte> fixture(ip::WeightEncoding e,std::size_t n,std::size_t k) {
    const std::size_t block=e==ip::WeightEncoding::q4_0?32:256,bytes=e==ip::WeightEncoding::q4_0?18:136;
    std::vector<std::byte> out(n*(k/block)*bytes);
    for(std::size_t b=0;b<n*(k/block);++b) {
        auto* p=out.data()+b*bytes;p[1]=std::byte{static_cast<unsigned char>(b%2?0xb8:0x38)};
        if(e==ip::WeightEncoding::iq4_xs){unsigned high=0;
            for(unsigned g=0;g<8;++g){const unsigned scale=(g*9U+static_cast<unsigned>(b))%64;
                high|=(scale>>4U)<<(2*g);p[4+g/2]|=std::byte{static_cast<unsigned char>((scale&15U)<<(4*(g%2)))};}
            p[2]=std::byte{static_cast<unsigned char>(high&255U)};p[3]=std::byte{static_cast<unsigned char>(high>>8U)};}
        for(std::size_t j=0;j<block/2;++j)p[(e==ip::WeightEncoding::q4_0?2:8)+j]=
            std::byte{static_cast<unsigned char>(((j+b)%16)|(((3*j+7*b)%16)<<4U))};
    }
    return out;
}
void run_case(ip::Shape s,ip::WeightEncoding e,lp::ActivationEncoding ae,ip::Instruction instruction) {
    ip::validate_shape(s,e);
    std::vector<double> x(s.rows*s.inner);
    for(std::size_t r=0;r<s.rows;++r)for(std::size_t j=0;j<s.inner;++j)
        x[r*s.inner+j]=(r==0&&s.rows>1)?0.0:static_cast<double>(static_cast<int>((j+5*r)%29)-14)/8.0;
    const auto w=ip::stage_weights(e,s.columns,s.inner,fixture(e,s.columns,s.inner));
    const auto a=ip::prepare(s,ae,x);ip::validate(w,a,instruction);
    const auto ref=ip::reference(w,a,instruction);
    Buffer<double> input(x.size()),ws(w.scales.size()),as(a.scales.size()+2),out(ref.output.size()+2);
    Buffer<std::uint8_t> ac(a.codes.size()+2),wc(w.packed.size());
    Buffer<std::int32_t> sums(a.sums.size()+2),ig(ref.groups.size()+2);
    Buffer<unsigned> error(1);
    input.put(x);ws.put(w.scales);wc.put(w.packed);error.put({0});
    as.put(std::vector<double>(as.count,12345.0));out.put(std::vector<double>(out.count,12345.0));
    ac.put(std::vector<std::uint8_t>(ac.count,0xa5));sums.put(std::vector<std::int32_t>(sums.count,1234567));
    ig.put(std::vector<std::int32_t>(ig.count,1234567));
    const auto rows=static_cast<std::uint32_t>(s.rows),columns=static_cast<std::uint32_t>(s.columns),k=static_cast<std::uint32_t>(s.inner);
    hipLaunchKernelGGL(prepare_kernel,dim3(static_cast<unsigned>((a.scales.size()+127)/128)),dim3(128),0,hipStreamLegacy,
        input.data,ac.data+1,as.data+1,sums.data+1,error.data,rows,k,ae==lp::ActivationEncoding::symmetric_a4);
    check(hipGetLastError(),"prepare launch");check(hipDeviceSynchronize(),"prepare completion");
    require(error.get()[0]==0,"device activation preparation rejected input");
    const auto dc=ac.get();const auto ds=as.get();const auto dsum=sums.get();
    require(dc.front()==0xa5&&dc.back()==0xa5&&ds.front()==12345&&ds.back()==12345&&
        dsum.front()==1234567&&dsum.back()==1234567,"activation preparation guard overwritten");
    require(std::equal(a.codes.begin(),a.codes.end(),dc.begin()+1),"device/CPU prepared codes differ");
    require(same_bits(a.scales,ds,1),"device/CPU binary64 scale bits differ");
    require(std::equal(a.sums.begin(),a.sums.end(),dsum.begin()+1),"device/CPU group sums differ");
    const dim3 grid((columns+15U)/16U,(rows+15U)/16U),threads(32);
#define LAUNCH(Four,IQ,A4) hipLaunchKernelGGL((project_kernel<Four,IQ,A4>),grid,threads,0,hipStreamLegacy,ac.data+1,as.data+1,sums.data+1,wc.data,ws.data,out.data+1,ig.data+1,error.data,rows,columns,k)
    if(instruction==ip::Instruction::iu4){LAUNCH(true,false,true);}
    else if(e==ip::WeightEncoding::iq4_xs){if(ae==lp::ActivationEncoding::symmetric_a4){LAUNCH(false,true,true);}else{LAUNCH(false,true,false);}}
    else{if(ae==lp::ActivationEncoding::symmetric_a4){LAUNCH(false,false,true);}else{LAUNCH(false,false,false);}}
#undef LAUNCH
    check(hipGetLastError(),"projection launch");check(hipDeviceSynchronize(),"projection completion");
    require(error.get()[0]==0,"device projection bounds/nonfinite check failed");
    const auto got=out.get();const auto groups=ig.get();
    require(got.front()==12345&&got.back()==12345&&groups.front()==1234567&&groups.back()==1234567,"projection guard overwritten");
    require(std::equal(ref.groups.begin(),ref.groups.end(),groups.begin()+1),"WMMA group result differs from independent direct integer formula");
    require(same_bits(ref.output,got,1),"WMMA ordered binary64 output bits differ from reference");
    // Re-run actual device preparation/projection for each row in isolation.
    // Reuse capacity only; dimensions and content describe the isolated row.
    for(std::size_t r=0;r<s.rows;++r){
        const auto solo=ip::prepare({1,s.columns,s.inner},ae,std::span<const double>(x).subspan(r*s.inner,s.inner));
        const auto sr=ip::reference(w,solo,instruction);require(same_bits(sr.output,got,1+r*s.columns),"row-local oracle bit mismatch");
        std::vector<double>single_input(x.size(),0);std::copy_n(x.begin()+static_cast<std::ptrdiff_t>(r*s.inner),s.inner,single_input.begin());
        input.put(single_input);error.put({0});out.put(std::vector<double>(out.count,12345.0));ig.put(std::vector<std::int32_t>(ig.count,1234567));
        ac.put(std::vector<std::uint8_t>(ac.count,0xa5));as.put(std::vector<double>(as.count,12345.0));
        sums.put(std::vector<std::int32_t>(sums.count,1234567));
        hipLaunchKernelGGL(prepare_kernel,dim3(static_cast<unsigned>((solo.scales.size()+127)/128)),dim3(128),0,hipStreamLegacy,
            input.data,ac.data+1,as.data+1,sums.data+1,error.data,1U,k,ae==lp::ActivationEncoding::symmetric_a4);
        check(hipGetLastError(),"solo preparation launch");check(hipDeviceSynchronize(),"solo preparation completion");
        require(error.get()[0]==0,"solo preparation status failed");
        const auto solo_codes=ac.get();const auto solo_scales=as.get();const auto solo_sums=sums.get();
        require(std::equal(solo.codes.begin(),solo.codes.end(),solo_codes.begin()+1)&&
            same_bits(solo.scales,solo_scales,1)&&std::equal(solo.sums.begin(),solo.sums.end(),solo_sums.begin()+1),"native solo preparation mismatch");
        require(solo_codes[1+s.inner]==0xa5&&solo_scales[1+solo.scales.size()]==12345.0&&
            solo_sums[1+solo.sums.size()]==1234567,"solo preparation wrote past logical row");
        const dim3 solo_grid((columns+15U)/16U,1);
#define SOLO(Four,IQ,A4) hipLaunchKernelGGL((project_kernel<Four,IQ,A4>),solo_grid,threads,0,hipStreamLegacy,ac.data+1,as.data+1,sums.data+1,wc.data,ws.data,out.data+1,ig.data+1,error.data,1U,columns,k)
        if(instruction==ip::Instruction::iu4){SOLO(true,false,true);}
        else if(e==ip::WeightEncoding::iq4_xs){if(ae==lp::ActivationEncoding::symmetric_a4){SOLO(false,true,true);}else{SOLO(false,true,false);}}
        else{if(ae==lp::ActivationEncoding::symmetric_a4){SOLO(false,false,true);}else{SOLO(false,false,false);}}
#undef SOLO
        check(hipGetLastError(),"solo projection launch");check(hipDeviceSynchronize(),"solo projection completion");
        const auto single_out=out.get();const auto single_groups=ig.get();
        require(error.get()[0]==0&&same_bits(sr.output,single_out,1),"native solo-vs-batched output bits differ");
        require(std::equal(sr.groups.begin(),sr.groups.end(),single_groups.begin()+1),"native solo integer groups differ");
        require(single_out[1+s.columns]==12345.0&&single_groups[1+sr.groups.size()]==1234567,
            "solo projection wrote past logical output");
    }
    std::cout<<"PASS M="<<s.rows<<" N="<<s.columns<<" K="<<s.inner<<" "<<ip::numerical_identity(e,ae,instruction)<<'\n';
}
void nonfinite_control(){Buffer<double> in(32),scales(1);Buffer<std::uint8_t> codes(32);Buffer<std::int32_t>sums(1);Buffer<unsigned>error(1);
    for(double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::denorm_min()}){
        std::vector<double>x(32,0);x[0]=bad;in.put(x);error.put({0});codes.put(std::vector<std::uint8_t>(32,0xa5));
        hipLaunchKernelGGL(prepare_kernel,dim3(1),dim3(128),0,hipStreamLegacy,in.data,codes.data,scales.data,sums.data,error.data,1U,32U,false);
        check(hipGetLastError(),"negative control launch");check(hipDeviceSynchronize(),"negative control completion");
        require(error.get()[0]!=0,"device nonfinite/underflow negative control accepted");
        const auto got=codes.get();require(std::all_of(got.begin(),got.end(),[](auto q){return q==0xa5;}),"rejected group wrote codes");}
}
void preparation_controls(){
    for(auto encoding:{lp::ActivationEncoding::symmetric_a8,lp::ActivationEncoding::symmetric_a4}) {
        const int qmax=encoding==lp::ActivationEncoding::symmetric_a4?7:127;
        for(unsigned kind=0;kind<4;++kind) {
            std::vector<double>x(32,0.0);
            if(kind==0){x[0]=qmax;x[1]=0.5;x[2]=1.5;x[3]=-0.5;x[4]=-1.5;x[5]=-0.0;}
            if(kind==1){x[0]=std::numeric_limits<double>::max();x[1]=-x[0];x[2]=1;}
            if(kind==2){x[0]=std::numeric_limits<double>::denorm_min()*qmax;x[1]=-x[0];x[2]=std::numeric_limits<double>::denorm_min();}
            const auto ref=ip::prepare({1,1,32},encoding,x);
            Buffer<double>in(32),scales(1);Buffer<std::uint8_t>codes(32);Buffer<std::int32_t>sums(1);Buffer<unsigned>error(1);
            in.put(x);error.put({0});
            hipLaunchKernelGGL(prepare_kernel,dim3(1),dim3(128),0,hipStreamLegacy,in.data,codes.data,scales.data,sums.data,error.data,1U,32U,encoding==lp::ActivationEncoding::symmetric_a4);
            check(hipGetLastError(),"preparation edge launch");check(hipDeviceSynchronize(),"preparation edge completion");
            require(error.get()[0]==0,"finite preparation edge rejected");
            require(codes.get()==ref.codes&&sums.get()==ref.sums&&same_bits(ref.scales,scales.get(),0),"preparation ties/zero/extreme/subnormal bits differ");
        }
    }
}
void projection_nonfinite_control(){
    Buffer<std::uint8_t>ac(32),wc(16);Buffer<double>as(1),ws(1),out(1);Buffer<std::int32_t>sums(1),ig(1);Buffer<unsigned>error(1);
    ac.put(std::vector<std::uint8_t>(32,127));wc.put(std::vector<std::uint8_t>(16,0xff));
    as.put({std::numeric_limits<double>::max()});ws.put({std::numeric_limits<double>::max()});sums.put({4064});error.put({0});
    hipLaunchKernelGGL((project_kernel<false,false,false>),dim3(1),dim3(32),0,hipStreamLegacy,ac.data,as.data,sums.data,wc.data,ws.data,out.data,ig.data,error.data,1U,1U,32U);
    check(hipGetLastError(),"nonfinite projection launch");check(hipDeviceSynchronize(),"nonfinite projection completion");
    require((error.get()[0]&8U)!=0,"overflowing binary64 projection did not report failure");
    // A failed status invalidates all output. The kernel never claims that its
    // partial/nonfinite output is usable and the harness does not compare it.
}
}
int main(int argc,char** argv){
    if(argc!=2||std::string(argv[1])!="--run"){std::cout<<"Unqualified gfx1151 Wave32 IU8/IU4 experiment. Explicit --run required.\n";return 2;}
    try{
        const char* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");
        require(!override_arch||!*override_arch,"architecture override prevents gfx1151 target attribution");
        check(hipSetDevice(0),"select device");
        hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0),"device properties");
        const std::string arch=p.gcnArchName;
        require(arch.substr(0,arch.find(':'))=="gfx1151","native experiment requires exact gfx1151");
        Buffer<unsigned> size(1);hipLaunchKernelGGL(wave_size,dim3(1),dim3(32),0,hipStreamLegacy,size.data);
        check(hipGetLastError(),"wave report launch");check(hipDeviceSynchronize(),"wave report completion");require(size.get()[0]==32,"requires Wave32 execution");
        nonfinite_control();preparation_controls();projection_nonfinite_control();
        run_case({3,19,640},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8);
        run_case({17,19,640},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu4);
        run_case({1,640,2560},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu4);
        run_case({1,19,32},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8);
        run_case({3,19,2560},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8);
        run_case({17,19,256},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8);
        run_case({1,2560,2560},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8);
        std::cout<<"Synthetic native arithmetic checks passed; no model or performance qualification.\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
