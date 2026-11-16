// No HIP runtime, device allocation or multi-GiB backing allocation is needed:
// these tests exercise the same launch envelope called by registry admission.
#include "../../src/hip/weight/registry.hpp"
#include "byte_equal.hpp"
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace {
using namespace strix::weight;
using namespace strix::weight::hip;
void expect(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
WeightView tensor(std::uint64_t n,std::uint64_t k,unsigned block,unsigned bytes) {
    WeightView w{};
    w.shape={{n,k,0,0},2};
    w.payload.range.bytes=n*(k/block)*bytes;
    return w;
}
void addressing() {
    auto w=tensor(2,128,32,34);
    LaunchShape s{1,2,128,136,128,2,0};
    expect(bool(validate_launch(w,s,32,34,68)),"contiguous Q8 rejected");
    s.row_stride_bytes=272;
    expect(!validate_launch(w,s,32,34,68),"Q8 stride escapes owned payload");
    w=tensor(2,256,32,34);s.k_offset_bytes=1;
    expect(!validate_launch(w,s,32,34,68),"Q8 byte prefix admitted");
    s.k_offset_bytes=34;
    expect(!validate_launch(w,s,32,34,68),"Q8 pair-misaligned prefix admitted");
    expect(bool(validate_launch(w,s,32,34,34)),"scalar Q8 whole block rejected");
    s.k_offset_bytes=136;
    expect(bool(validate_launch(w,s,32,34,68)),"Q8 end-aligned subrow rejected");
    s.k_offset_bytes=204;
    expect(!validate_launch(w,s,32,34,68),"Q8 subrow crosses row boundary");
    s.k_offset_bytes=136;--w.payload.range.bytes;
    expect(!validate_launch(w,s,32,34,68),"last byte outside payload admitted");
    s.k_offset_bytes=UINT64_MAX;
    expect(!validate_launch(w,s,32,34,68),"overflowing prefix admitted");
    s.k_offset_bytes=0;w.shape.dims[1]=UINT64_MAX-31U;
    expect(!validate_launch(w,s,32,34,68),"overflowing storage row admitted");
    w=tensor(2,256,1,2);s={1,2,128,512,128,2,256};
    expect(bool(validate_launch(w,s,1,2,2)),"F16 subrow rejected");
    s.k_offset_bytes=1;
    expect(!validate_launch(w,s,1,2,2),"unaligned F16 prefix admitted");
    s.k_offset_bytes=0;w.has_byte_strides=true;w.byte_strides[0]=256;
    expect(!validate_launch(w,s,1,2,2),"declared stride mismatch admitted");
}
void q6_span() {
    auto w=tensor(65536,131072,32,26);
    LaunchShape s{1,65536,131072,106496,131072,65536,0};
    expect(!validate_launch(w,s,32,26,26,true),"Q6 6.5-GiB span admitted");
    // Both native and signed-A8 protocols share this cap. Width 64 satisfies
    // the signed route's additional even-pair / whole-dword requirement.
    constexpr std::uint64_t rows=UINT32_MAX/52U;
    w=tensor(rows+1U,64,32,26);s={1,rows,64,52,64,rows,0};
    expect(bool(validate_launch(w,s,32,26,26,true)),"Q6 below-limit span rejected");
    ++s.n;++s.output_stride;
    expect(!validate_launch(w,s,32,26,26,true),"Q6 last group crosses 32-bit boundary");
    w=tensor(2,256,32,26);s={1,2,128,208,128,2,0};
    expect(!validate_launch(w,s,32,26,26,true),"Q6 shortened planar row admitted");
    s.k_offset_bytes=26;
    expect(!validate_launch(w,s,32,26,26,true),"Q6 planar prefix admitted");
}
void bytes() {
    using strix::weight::test::byte_equal;
    expect(!byte_equal({0.0F},{-0.0F}),"signed-zero change hidden");
    expect(byte_equal({-0.0F,1.0F},{-0.0F,1.0F}),"identical bytes rejected");
    expect(!byte_equal({1.0F},{}),"length mismatch hidden");
    expect(byte_equal({},{}),"empty vectors rejected");
}
}
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("expected addressing, q6-span or bytes");
    const std::string_view name=argv[1];
    if(name=="addressing") addressing();
    else if(name=="q6-span") q6_span();
    else if(name=="bytes") bytes();
    else throw std::runtime_error("unknown test");
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
