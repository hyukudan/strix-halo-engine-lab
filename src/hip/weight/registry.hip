#include "registry.hpp"
#include "ggml_sources.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/host_q8.hpp"
#include <limits>
namespace strix::weight::hip {
namespace {
Status shape(const WeightView& w,const LaunchShape& s,FormatId format,unsigned block,unsigned bytes) {
    if(!(w.storage==format)) return {Error::unknown_format,"HIP source storage identity differs"};
    if(!s.m || !s.n || !s.k || s.m>UINT32_MAX || s.n>UINT32_MAX || s.k>UINT32_MAX ||
       s.k%block || s.activation_stride<s.k || s.output_stride<s.n)
        return {Error::unsupported_shape,"HIP source dimensions/activation/output strides"};
    const auto row=s.k/block*bytes;
    if(s.k_offset_bytes>UINT64_MAX-row || s.row_stride_bytes<s.k_offset_bytes+row || s.row_stride_bytes%2U)
        return {Error::unsupported_shape,"HIP source row stride or K prefix"};
    if(w.shape.rank!=2 || w.shape.dims[0]<s.n || w.shape.dims[1]<s.k ||
       (w.has_byte_strides && w.byte_strides[0]!=s.row_stride_bytes))
        return {Error::unsupported_shape,"HIP source launch differs from owned tensor"};
    return validate_weight(w);
}
}
Status Q8F16Source::validate(const WeightView& w,const LaunchShape& s) {
    auto status=shape(w,s,{"gguf.q8_0",1},32,34);
    if(!status) return status;
    if(s.k%128U || s.activation_stride%32U) return {Error::unsupported_shape,"HIP Q8A8 F16 staging requires K128/A32"};
    return {Error::ok,{}};
}
Status RawF16Source::validate(const WeightView& w,const LaunchShape& s) {
    auto status=shape(w,s,{"plain.f16",1},1,2);if(!status) return status;
    if(s.k%128U) return {Error::unsupported_shape,"HIP raw F16 core requires K128"};
    return {Error::ok,{}};
}
Status validate_q8_rows(const WeightView& w,const LaunchShape& s) {return shape(w,s,{"gguf.q8_0",1},32,34);}
Status validate_q6_rows(const WeightView& w,const LaunchShape& s) {return shape(w,s,{"image.q6p32",1},32,26);}
Status Q8SignedSource::validate(const WeightView& w,const LaunchShape& s) {
    auto status=Q8RowsSource::validate(w,s);if(!status) return status;
    if(s.k%64U || s.row_stride_bytes%4U) return {Error::unsupported_shape,"HIP signed-A8 source requires even whole-dword block pairs"};
    return {Error::ok,{}};
}
Status Q6SignedSource::validate(const WeightView& w,const LaunchShape& s) {
    auto status=Q6RowsSource::validate(w,s);if(!status) return status;
    if(s.k%64U || s.row_stride_bytes%4U) return {Error::unsupported_shape,"HIP signed-A8 planar source requires even whole-dword block pairs"};
    return {Error::ok,{}};
}
std::span<const SourceRegistration> sources() noexcept {
    static const SourceRegistration table[]={
        {{"plain.f16",1},"f16-tile","existing-f16-core",1,2,64,0,false,false,RawF16Source::validate},
        {{"gguf.q8_0",1},"f16-tile","dense-q8a8-f16-v1",32,2,64,0,false,false,Q8F16Source::validate},
        {{"gguf.q8_0",1},"native-scalar","existing-row-site",32,2,0,0,false,false,Q8RowsSource::validate},
        {{"image.q6p32",1},"native-scalar","existing-row-site",32,2,0,0,false,false,Q6RowsSource::validate},
        {{"gguf.q8_0",1},"signed-a8","existing-block-chain",32,4,0,64,false,false,Q8SignedSource::validate},
        {{"image.q6p32",1},"signed-a8","existing-block-chain",32,4,0,64,false,false,Q6SignedSource::validate}
    };
    return table;
}
const SourceRegistration* find_source(FormatId storage,std::string_view protocol) noexcept {
    for(const auto& s:sources()) if(s.storage==storage && s.protocol==protocol) return &s;
    return nullptr;
}
}  // namespace strix::weight::hip
