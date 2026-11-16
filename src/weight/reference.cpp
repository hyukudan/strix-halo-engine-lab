#include "formats/catalog.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <exception>
namespace strix::weight::detail {
Status elements(const Shape& shape, std::uint64_t& count) {
 count = 1;
 if (!shape.rank || shape.rank > 4) return {Error::unsupported_shape, "rank outside 1..4"};
 for (unsigned i=0;i<shape.rank;++i) {
   if (!shape.dims[i] || count > UINT64_MAX/shape.dims[i]) return {Error::malformed,"empty or overflowing shape"};
   count *= shape.dims[i];
 }
 return {Error::ok,{}};
}
Status read(const Slice& slice, std::vector<Byte>& bytes) {
 if (!slice.source || slice.range.offset > slice.source->size() ||
     slice.range.bytes > slice.source->size()-slice.range.offset || slice.range.bytes > SIZE_MAX)
   return {Error::malformed,"source range outside owning source"};
 bytes.resize(static_cast<std::size_t>(slice.range.bytes));
 return slice.source->read(slice.range, bytes);
}
const NamedSlice* sidecar(const WeightView& view, std::string_view name) {
 const NamedSlice* result = nullptr;
 for (const auto& part : view.sidecars) if (part.name == name) {
   if (result) return nullptr;
   result = &part;
 }
 return result;
}
Status unavailable(const WeightView&,const DecodeRequest&,DecodeBuffer& out) {
 out.written=0; return {Error::unsupported_arithmetic,"geometry-only storage has no semantic decoder"};
}
Status no_group(const WeightView&,std::uint64_t,SignedGroup32&) {
 return {Error::unsupported_arithmetic,"storage has no signed group32 operand contract"};
}
Status validate(const WeightView& view) {
 std::uint64_t count=0; auto status=elements(view.shape,count); if(!status) return status;
 const auto* format=find_format(builtin_registry(),view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 Geometry g{}; status=format->geometry(view.parameters,g); if(!status) return status;
 if (!g.block_elements || !g.block_bytes || count%g.block_elements || view.shape.dims[view.shape.rank-1]%g.row_alignment)
   return {Error::unsupported_shape,"shape does not meet storage block/row alignment"};
 if (g.payload_alignment && view.payload.range.offset%g.payload_alignment)
   return {Error::malformed,"payload alignment"};
 if(count/g.block_elements > UINT64_MAX/g.block_bytes) return {Error::malformed,"payload size overflow"};
 auto bytes=count/g.block_elements*g.block_bytes;
 // Gemma transport has contiguous FP16 input/output sidecars before the rings.
 if(view.storage.name=="exl3.gemma.transport") {
   if(view.shape.rank!=2 || view.shape.dims[0]%128 || view.shape.dims[1]%128)
     return {Error::unsupported_shape,"Gemma transport requires I/O multiples of 128"};
   const auto scales=view.shape.dims[0]+view.shape.dims[1];
   if(scales> (UINT64_MAX-bytes)/2) return {Error::malformed,"transport size overflow"};
   bytes += scales*2;
 }
 if(view.storage.name=="block.e4m3-e8m0.32x32") {
   if(view.shape.rank!=2 || view.shape.dims[0]%32 || view.shape.dims[1]%32)
     return {Error::unsupported_shape,"FP8 matrix requires 32x32 blocks"};
 }
 if(view.payload.range.bytes!=bytes) return {Error::malformed,"payload byte length differs from exact geometry"};
 std::vector<Byte> raw; status=read(view.payload,raw); if(!status) return status;
 for(const auto& part:view.sidecars) {
   std::uint64_t unused=0; status=elements(part.shape,unused); if(!status) return status;
   std::vector<Byte> data; status=read(part.data,data); if(!status) return status;
 }
 if(format->semantic_decode_supported &&
    (view.storage.name.starts_with("image.") || view.storage.name.starts_with("linear.") ||
     view.storage.name=="nvidia.nvfp4" || view.storage.name.starts_with("block."))) {
   std::vector<float> decoded(static_cast<std::size_t>(count));
   if(view.storage.name.starts_with("image.")) return decode_packed(view,raw,decoded);
   if(view.storage.name.starts_with("linear.")) return decode_channel(view,raw,decoded);
   if(view.storage.name=="nvidia.nvfp4") return decode_nvfp4(view,raw,decoded);
   return decode_fp8(view,raw,decoded);
 }
 return {Error::ok,{}};
}
CostDescriptor cost(const WeightView& view,ImageId) {
 std::uint64_t count=0;
 if(!elements(view.shape,count)) return {};
 std::uint64_t sidecar_bytes=0; for(const auto& part:view.sidecars) sidecar_bytes+=part.data.range.bytes;
 if(sidecar_bytes>UINT64_MAX-view.payload.range.bytes) return {};
 const auto packed=view.payload.range.bytes+sidecar_bytes;
 auto divisor=std::gcd(packed,count);
 CostDescriptor result{};
 result.packed_bytes_per_weight={packed/divisor,count/divisor};
 result.image_bytes_per_weight={0,1}; // unknown image cost is never fabricated
 result.fixed_metadata_bytes=0;
 result.operand=Scalar::f32;
 return result;
}
Status decode(const WeightView& view,const DecodeRequest& request,DecodeBuffer& out) {
 out.written=0;
 auto status=validate(view); if(!status) return status;
 std::uint64_t count=0; status=elements(view.shape,count); if(!status) return status;
 if(request.first_element>count || request.count>count-request.first_element)
   return {Error::unsupported_shape,"decode slice outside tensor"};
 if(request.output!=Scalar::f32) return {Error::unsupported_arithmetic,"reference decoder only widens to F32"};
 if(request.count>SIZE_MAX/sizeof(float) || out.bytes.size()<request.count*sizeof(float))
   return {Error::malformed,"decode output buffer too small"};
 bool known=request.arithmetic.name.empty();
 for(const auto& arithmetic:legacy_arithmetic()) known=known || arithmetic.id==request.arithmetic;
 if(!known) return {Error::unsupported_arithmetic,"unknown arithmetic identity"};
 std::vector<Byte> raw; status=read(view.payload,raw); if(!status) return status;
 std::vector<float> decoded(static_cast<std::size_t>(count));
 const auto name=view.storage.name;
 if(name.starts_with("plain.")) status=decode_plain(view,raw,decoded);
 else if(name=="gguf.iq2_xxs" || name=="gguf.q2_k" || name=="gguf.mxfp4" || name.starts_with("row.")) status=decode_ds41(view,raw,decoded);
 else if(name.starts_with("gguf.")) status=decode_ggml(view,raw,decoded);
 else if(name.starts_with("image.")) status=decode_packed(view,raw,decoded);
 else if(name.starts_with("block.")) status=decode_fp8(view,raw,decoded);
 else if(name.starts_with("exl3.")) status=decode_exl3(view,raw,decoded);
 else if(name.starts_with("linear.")) status=decode_channel(view,raw,decoded);
 else if(name=="nvidia.nvfp4") status=decode_nvfp4(view,raw,decoded);
 else status={Error::unsupported_arithmetic,"typed storage is not a usable linear weight"};
 if(!status) return status;
 std::memcpy(out.bytes.data(),decoded.data()+request.first_element,request.count*sizeof(float));
 out.written=static_cast<std::size_t>(request.count*sizeof(float));
 return {Error::ok,{}};
}
}
namespace strix::weight {
Status decode_reference(const Registry& registry,const WeightView& view,const DecodeRequest& request,DecodeBuffer& out) {
 out.written=0;
 const auto* format=find_format(registry,view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 try { return format->decode_reference(view,request,out); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
Status signed_group_reference(const Registry& registry,const WeightView& view,std::uint64_t group,SignedGroup32& out) {
 const auto* format=find_format(registry,view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 try { return format->signed_group_reference(view,group,out); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
}
