#pragma once
#include "strix/weight/resolve.hpp"
namespace strix::weight::hip {
struct LaunchShape {
    std::uint64_t m,n,k,row_stride_bytes,activation_stride,output_stride,k_offset_bytes;
};
// Host-testable launch envelope; storage metadata/ownership validation is also
// required by the registry. Q6 planar rows cannot be rebased to a shorter width.
inline Status validate_launch(const WeightView& w,const LaunchShape& s,
                              unsigned block,unsigned bytes,unsigned offset_alignment,bool planar=false) {
    if(!s.m || !s.n || !s.k || s.m>UINT32_MAX || s.n>UINT32_MAX || s.k>UINT32_MAX ||
       s.k%block || s.activation_stride<s.k || s.output_stride<s.n)
        return {Error::unsupported_shape,"HIP source dimensions/activation/output strides"};
    if(w.shape.rank!=2 || w.shape.dims[0]<s.n || w.shape.dims[1]<s.k ||
       w.shape.dims[1]%block || w.shape.dims[1]/block>UINT64_MAX/bytes)
        return {Error::unsupported_shape,"HIP source launch differs from owned tensor"};
    const auto stride=w.shape.dims[1]/block*bytes;
    const auto row=s.k/block*bytes;
    if(s.row_stride_bytes!=stride || (w.has_byte_strides && w.byte_strides[0]!=stride) ||
       s.k_offset_bytes%offset_alignment || s.k_offset_bytes>stride || row>stride-s.k_offset_bytes)
        return {Error::unsupported_shape,"HIP source row stride or K prefix"};
    const auto end=s.k_offset_bytes+row; // bounded by stride above
    if(end>w.payload.range.bytes || (s.n-1U)>(w.payload.range.bytes-end)/stride)
        return {Error::unsupported_shape,"HIP source addressed extent exceeds payload"};
    if(planar && (s.k_offset_bytes || s.k!=w.shape.dims[1] ||
       end>UINT32_MAX || (s.n-1U)>(UINT32_MAX-end)/stride))
        return {Error::unsupported_shape,"HIP planar source requires full rows and a below-4-GiB span"};
    return {Error::ok,{}};
}
struct SourceRegistration {
    FormatId storage;
    std::string_view protocol,arithmetic;
    std::uint32_t group_k,alignment,staged_k,paired_k;
    bool input_transform,output_transform;
    Status (*validate)(const WeightView&,const LaunchShape&);
};
std::span<const SourceRegistration> sources() noexcept;
// Admission scaffolding tested by fixtures; production launchers do not use this
// table yet. Callers retain incumbent launch and numeric Site policy.
const SourceRegistration* find_source(FormatId,std::string_view protocol) noexcept;
}  // namespace strix::weight::hip
