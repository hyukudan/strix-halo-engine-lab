// Compact old/new migration fixture. All arithmetic is exercised through the incumbent cores.
#include "../../src/hip_q8_rows_gemv.hpp"
#include "../../src/core/f16_dense_q8a8.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#if __has_include("../../src/hip/weight/registry.hpp")
#include "../../src/hip/weight/registry.hpp"
#include "strix/weight/source.hpp"
#define WFMT_HAS_NATIVE_SOURCES 1
#endif
#ifdef STRIX_WFMT_CAPTURE
#include "../../scripts/bench/wfmt/hip_transfer_capture.hpp"
#endif

namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t n) { check(hipMalloc(&p,n)); check(hipMemset(p,0,n)); }
    ~Buffer() { (void)hipFree(p); }
    template<class T> T* as() const { return static_cast<T*>(p); }
    template<class T> void upload(const std::vector<T>& v) const { check(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
    template<class T> std::vector<T> read(std::size_t n) const {
        std::vector<T> v(n); check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v;
    }
};
__device__ float* partial_trace;
__device__ unsigned* partial_count;
template<unsigned R,bool Diagnostic=false> struct Site {
    using Input = float;
    static constexpr unsigned rows=R;
    static constexpr bool fused_fma=true;
    const void* input[R];
    strix::q8_rows::Segments segments;
    std::uint32_t width,outputs;
    float* output;
    __device__ static float coefficient(unsigned scale,std::int8_t code) {
        float result;
        asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(result) : "v"(scale), "v"(static_cast<float>(code)));
        return result;
    }
    __device__ static void fma(float& acc,float c,float x) {
        acc=__builtin_fmaf(c,x,acc);
        if constexpr(Diagnostic) {
            const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
            const unsigned i=partial_count[t]++;
            if(i<256U) partial_trace[t*256U+i]=acc;
        }
    }
    __device__ static float add(float lower,float upper) {
        float result; asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper)); return result;
    }
    __device__ void store(std::uint32_t o,unsigned r,float v) const { output[r*outputs+o]=v; }
};
struct RawCols {
    const _Float16* weight; float* out; unsigned in,count;
    __device__ strix::core::f16_dense_tall::ColumnTile tile(unsigned i) const {
        return {weight+16U*i*in,out+16U*i,count,16U*i<count};
    }
};
__global__ void raw_kernel(const float* x,unsigned k,unsigned rows,RawCols c) {
    __shared__ strix::core::f16_dense_tall::Storage<1> lds;
    strix::core::f16_dense_tall::core<1>(x,k,rows,c,lds,strix::core::f16_dense_tall::q8a8::Store{c.count,0U,0U},true);
}
__global__ void operands_kernel(std::uint16_t* out) {
    using namespace strix::core::f16_dense_tall::q8a8;
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    if(i>=256U*16U) return;
    constexpr std::uint16_t scales[16]={0,1,0x3ff,0x400,0x7bff,0x8000,0x8001,0x83ff,0x8400,0x3c00,0xbc00,0x3555,0xb555,0x7c00,0xfc00,0x7e00};
    const unsigned code=i%256U, bits=scales[i/256U];
    const std::uint32_t word=code*0x01010101U;
    const auto a=detail::w16_scalar(word,0U,static_cast<float>(__builtin_bit_cast(_Float16,static_cast<std::uint16_t>(bits))));
    const auto pair=detail::w16_pair(word^0x80808080U,0U,__builtin_bit_cast(detail::Half2,bits|(bits<<16U)));
    out[3U*i]=__builtin_bit_cast(std::uint16_t,a);
    out[3U*i+1U]=__builtin_bit_cast(std::uint16_t,pair[0]);
    out[3U*i+2U]=__builtin_bit_cast(std::uint16_t,pair[1]);
}
template<unsigned Form> __global__ void prepared_kernel(strix::core::f16_dense_tall::q8a8::View v,std::uint32_t* out) {
    using namespace strix::core::f16_dense_tall;
    q8a8::Cols<8U,Form> c{v};
    U4 data[8];c.stage(0,threadIdx.x%16U,0,data);
    for(unsigned i=0;i<8U;++i) reinterpret_cast<U4*>(out)[threadIdx.x*8U+i]=data[i];
}
std::vector<unsigned char> weights(unsigned k,unsigned n) {
    std::vector<unsigned char> w(n*(k/32U)*34U);
    for(unsigned b=0;b<n*(k/32U);++b) {
        const std::uint16_t scale=static_cast<std::uint16_t>(0x2800U+(b%8U)*0x100U);
        std::memcpy(w.data()+34U*b,&scale,2);
        for(unsigned j=0;j<32;++j) w[34U*b+2U+j]=static_cast<unsigned char>((b*17U+j*13U)%63U-32U);
    }
    return w;
}
template<unsigned R> void rows_case(unsigned k,unsigned n) {
    using namespace strix::q8_rows;
    auto w=weights(k,n); std::vector<float> x(R*k);
    for(unsigned i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%127U)-63)*0.015625F;
    Buffer dw(w.size()), dx(x.size()*4U),dy(R*n*4U);dw.upload(w);dx.upload(x);
    Site<R> s{};s.segments=segments({{dw.p,n}});s.width=k;s.outputs=n;s.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) s.input[r]=dx.as<float>()+r*k;
    const auto read=[&] { check(hipGetLastError()); auto y=dy.read<float>(R*n); for(float v:y) if(!std::isfinite(v)) throw std::runtime_error("nonfinite row output");return y; };
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,false,false>),dim3((n+15U)/16U),dim3(256),0,0,s);const auto a=read();
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,true,false>),dim3((n+15U)/16U),dim3(256),R*k*4U,0,s);const auto b=read();
        if(a!=b) throw std::runtime_error("staged rows differ");
        hipLaunchKernelGGL((project_rows_stepped_kernel<Site<R>,2U,1U>),dim3((n+15U)/16U),dim3(256),2U*R*256U*4U,0,s);const auto c=read();
        if(a!=c) throw std::runtime_error("stepped rows differ");
        // Q6P32 all-zero codes with nonzero FP16 scales exercises the exact planar addressing.
        std::vector<unsigned char> q6(n*(k/32U)*26U,0);
        for(unsigned o=0;o<n;++o) for(unsigned chunk=0;chunk<k/256U;++chunk) {
            auto off=strix::q6p::chunk_offsets(k,chunk);
            for(unsigned b0=0;b0<8U;++b0) {const std::uint16_t h=0x2800U;std::memcpy(q6.data()+o*(k/32U)*26U+off.d+2U*b0,&h,2);}
        }
        Buffer dq(q6.size());dq.upload(q6);s.segments=segments({{dq.p,n}});
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,false,true>),dim3((n+15U)/16U),dim3(256),0,0,s);read();
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,true,true>),dim3((n+15U)/16U),dim3(256),R*k*4U,0,s);read();
    } else {
        constexpr unsigned sub=4U;
        hipLaunchKernelGGL((project_wide_rows_kernel<Site<R>,2U,sub>),dim3(((n+15U)/16U)*(R/sub)),dim3(256),0,0,s);const auto a=read();
        hipLaunchKernelGGL((project_wide_rows_staged_kernel<Site<R>,2U,sub>),dim3(((n+15U)/16U)*(R/sub)),dim3(256),0,0,s);const auto b=read();
        if(a!=b) throw std::runtime_error("wide staged differs");
        constexpr unsigned waves=R/sub<8U?R/sub:8U;
        hipLaunchKernelGGL((project_wide_rows_coef_kernel<Site<R>,2U,sub,8U,false>),dim3(((n+1U)/2U)*(R/sub/waves)),dim3(32U*waves),0,0,s);const auto c=read();
        if(a!=c) throw std::runtime_error("coefficient source differs");
        if constexpr(R%32U==0U) {
            constexpr unsigned groups=(65U+63U)/64U;
            const unsigned g=(n+63U)/64U;
            Buffer codes(groups*16U*(k/256U)*1024U),scales(groups*(k/256U)*64U*16U),xs(R*k*4U);
            hipLaunchKernelGGL((lane_pack_weights_kernel<Site<R>>),dim3((g*(k/256U)*64U*16U+255U)/256U),dim3(256),0,0,s,codes.as<unsigned char>(),scales.as<std::uint16_t>(),g);
            codes.read<unsigned char>(g*16U*(k/256U)*1024U);scales.read<std::uint16_t>(g*(k/256U)*64U*8U);
            hipLaunchKernelGGL((lane_pack_rows_kernel<Site<R>>),dim3((R*k+255U)/256U),dim3(256),0,0,s,xs.as<float>());
            xs.read<float>(R*k);
            hipLaunchKernelGGL((project_wide_rows_lane_kernel<Site<R>,8U,8U,2U>),dim3(g*(R/32U)),dim3(256),0,0,s,codes.as<unsigned char>(),scales.as<std::uint16_t>(),xs.as<float>());const auto d=read();
            if(a!=d) throw std::runtime_error("lane source differs");
        }
    }
}
template<unsigned R> void signed_case(unsigned k,unsigned n) {
    using namespace strix::q8_rows;
    auto w=weights(k,n);std::vector<float> x(R*k);
    for(unsigned i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%31U)-15)*0.03125F;
    Buffer dw(w.size()),dx(x.size()*4U),dy(R*n*4U),codes(dense_a8_pitch<R>()*k),deltas(dense_a8_pitch<R>()*(k/32U)*4U);
    dw.upload(w);dx.upload(x);
    Site<R> site{};site.segments=segments({{dw.p,n}});site.width=k;site.outputs=n;site.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) site.input[r]=dx.as<float>()+r*k;
    hipLaunchKernelGGL((dense_a8_quantize_kernel<Site<R>>),dim3((R*(k/8U)+255U)/256U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());
    const auto ac=codes.read<std::int8_t>(R*k);const auto ad=deltas.read<float>(dense_a8_pitch<R>()*(k/32U));
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit("activation_codes",ac.data(),ac.size());
    strix::weight::capture::emit("activation_scales",ad.data(),ad.size()*sizeof(float));
#endif
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,false,false>),dim3((n+7U)/8U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());const auto direct=dy.read<float>(R*n);
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,true,false>),dim3((n+7U)/8U),dim3(256),dense_a8_stage_bytes<R>(k),0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());const auto staged=dy.read<float>(R*n);
        if(direct!=staged) throw std::runtime_error("signed-A8 source staging differs");
        std::vector<unsigned char> q6(n*(k/32U)*26U,0);
        Buffer dq(q6.size());dq.upload(q6);site.segments=segments({{dq.p,n}});
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,false,true>),dim3((n+7U)/8U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
    } else {
        hipLaunchKernelGGL((project_rows_dense_a8_wmma_kernel<Site<R>,1U,1U>),dim3(((n+63U)/64U)*(R/16U)),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
        if constexpr(R%64U==0U) {
            hipLaunchKernelGGL((project_rows_dense_a8_wmma_lds2_kernel<Site<R>,4U,true>),dim3(((n+127U)/128U)*(R/64U)),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
        }
    }
}
void partial_case(bool q6) {
    using namespace strix::q8_rows;
    auto w=weights(256U,1U);
    if(q6) w.assign(208U,0);
    std::vector<float> x(256U,0.125F);
    Buffer dw(w.size()),dx(x.size()*4U),dy(4U),trace(256U*256U*4U),counts(256U*4U);
    dw.upload(w);dx.upload(x);
    auto* tp=trace.as<float>();auto* cp=counts.as<unsigned>();
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_trace),&tp,sizeof(tp)));
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_count),&cp,sizeof(cp)));
    Site<1U,true> site{};site.input[0]=dx.p;site.segments=segments({{dw.p,1U}});site.width=256;site.outputs=1;site.output=dy.as<float>();
    if(q6) hipLaunchKernelGGL((project_rows_kernel<Site<1U,true>,2U,false,true>),dim3(1),dim3(256),0,0,site);
    else hipLaunchKernelGGL((project_rows_kernel<Site<1U,true>,2U,false,false>),dim3(1),dim3(256),0,0,site);
    check(hipGetLastError());dy.read<float>(1);
    auto values=trace.read<float>(256U*256U);
    auto used=counts.read<unsigned>(256U);
    for(unsigned count:used) if(count>256U) throw std::runtime_error("diagnostic partial capacity");
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit("partials",values.data(),values.size()*sizeof(float));
#endif
}
void dense_case(unsigned k,unsigned n,unsigned rows) {
    namespace d=strix::core::f16_dense_tall;
    auto w=weights(k,n);std::vector<std::int8_t> a(rows*k);std::vector<float> scales(rows*k/32U,0.03125F),x(rows*k);
    for(unsigned i=0;i<a.size();++i) {a[i]=static_cast<std::int8_t>(static_cast<int>(i%63U)-31);x[i]=static_cast<float>(a[i])*0.03125F;}
    Buffer dw(w.size()),da(a.size()),ds(scales.size()*4U),dx(x.size()*4U),dy(rows*n*4U);
    dw.upload(w);da.upload(a);ds.upload(scales);dx.upload(x);
    d::q8a8::View v{dw.as<std::uint8_t>(),da.as<std::int8_t>(),ds.as<float>(),dy.as<float>(),k/32U*34U,0,k,n,rows,k,n,0,0};
    if(rows==1U) {
        Buffer prepared(16U*8U*16U);
        hipLaunchKernelGGL((prepared_kernel<0U>),dim3(1),dim3(16),0,0,v,prepared.as<std::uint32_t>());check(hipGetLastError());const auto a0=prepared.read<std::uint32_t>(16U*8U*4U);
        hipLaunchKernelGGL((prepared_kernel<1U>),dim3(1),dim3(16),0,0,v,prepared.as<std::uint32_t>());check(hipGetLastError());const auto a1=prepared.read<std::uint32_t>(16U*8U*4U);
        if(a0!=a1) throw std::runtime_error("prepared scalar packed fragments differ");
#ifdef STRIX_WFMT_CAPTURE
        strix::weight::capture::emit("prepared",a0.data(),a0.size()*sizeof(std::uint32_t));
#endif
    }
    hipLaunchKernelGGL((d::q8a8::kernel<1U,8U,0U>),dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,v);check(hipGetLastError());const auto scalar=dy.read<float>(rows*n);
    hipLaunchKernelGGL((d::q8a8::kernel<1U,8U,1U>),dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,v);check(hipGetLastError());const auto packed=dy.read<float>(rows*n);
    if(scalar!=packed) throw std::runtime_error("scalar packed source differs");
    std::vector<_Float16> f16(((n+15U)/16U)*16U*k);
    for(unsigned o=0;o<((n+15U)/16U)*16U;++o) for(unsigned j=0;j<k;++j) {
        const unsigned ro=std::min(o,n-1U);std::uint16_t scale;std::memcpy(&scale,w.data()+(ro*k/32U+j/32U)*34U,2U);
        f16[o*k+j]=static_cast<_Float16>(static_cast<float>(std::bit_cast<_Float16>(scale))*static_cast<float>(static_cast<std::int8_t>(w[(ro*k/32U+j/32U)*34U+2U+j%32U])));
    }
    Buffer df(f16.size()*2U);df.upload(f16);
    hipLaunchKernelGGL(raw_kernel,dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,dx.as<float>(),k,rows,RawCols{df.as<_Float16>(),dy.as<float>(),k,n});check(hipGetLastError());const auto raw=dy.read<float>(rows*n);
    if(scalar!=raw) throw std::runtime_error("raw F16 source differs");
}
}
int main() try {
#ifdef WFMT_HAS_NATIVE_SOURCES
    namespace wh=strix::weight::hip;
    static_assert(std::is_trivially_copyable_v<wh::MatrixView>);
    auto fixture=weights(128U,17U);
    strix::weight::WeightView view{};
    view.storage={"gguf.q8_0",1};view.shape={{{128U,17U,0U,0U}},2};
    view.payload={strix::weight::memory_source(std::as_bytes(std::span(fixture))),{0U,fixture.size()}};
    view.has_byte_strides=true;view.byte_strides={1U,136U,0U,0U};
    wh::LaunchShape shape{1U,17U,128U,136U,128U,17U,0U};
    const auto* registered=wh::find_source(view.storage,"f16-tile");
    if(wh::sources().size()!=6U || !registered || !registered->validate(view,shape)) throw std::runtime_error("HIP source registration validation");
    shape.activation_stride=127U;
    if(registered->validate(view,shape)) throw std::runtime_error("HIP source admitted invalid activation stride");
    shape.activation_stride=128U;shape.k=64U;
    if(registered->validate(view,shape)) throw std::runtime_error("HIP F16 source admitted partial stage");
    shape.k=96U;registered=wh::find_source(view.storage,"signed-a8");
    if(!registered || registered->paired_k!=64U || registered->validate(view,shape)) throw std::runtime_error("HIP signed source admitted odd block pair");
    if(wh::find_source({"gguf.iq3_s",1},"f16-tile")) throw std::runtime_error("unregistered HIP source unexpectedly admitted");
#endif
    Buffer op(4096U*3U*2U);hipLaunchKernelGGL(operands_kernel,dim3(16),dim3(256),0,0,op.as<std::uint16_t>());check(hipGetLastError());op.read<std::uint16_t>(4096U*3U);
    partial_case(false);partial_case(true);
    signed_case<1>(256U,17U);signed_case<2>(256U,17U);signed_case<7>(256U,17U);signed_case<8>(256U,17U);
    signed_case<16>(256U,17U);signed_case<32>(256U,17U);signed_case<64>(256U,17U);signed_case<128>(256U,17U);
    for(unsigned n:{1U,17U,65U}) {rows_case<1>(256U,n);rows_case<2>(512U,n);rows_case<7>(256U,n);rows_case<8>(512U,n);rows_case<16>(256U,n);rows_case<32>(512U,n);rows_case<64>(256U,n);}
    for(unsigned rows:{1U,2U,7U,8U,9U,31U,32U,33U,63U,64U,65U,127U,128U,129U}) dense_case(128U,17U,rows);
    std::cout<<"HIP dense source migration: operands/raw/Q8 scalar-packed/Q6/direct/staged/stepped/wide/coef/lane PASS\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
