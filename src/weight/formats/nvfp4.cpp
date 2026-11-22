#include "catalog.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include <cmath>
#include <cstring>
namespace strix::weight::detail {
std::span<const FormatDescriptor> nvfp4_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"nvidia.nvfp4",1},"strix.nvfp4.safetensors@1",quantized_linear,true,fixed<16,8,16>)
 };
 return formats;
}
Status decode_nvfp4(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 if(view.shape.rank!=2 || view.shape.dims[view.shape.rank-1]%64) return {Error::unsupported_shape,"NVFP4 K must be a multiple of 64"};
 const auto width=view.shape.dims[1],rows=view.shape.dims[0];
 const auto* blocks=sidecar(view,"block_scales"); const auto* tensor=sidecar(view,"tensor_scale");
 if(!blocks || blocks->scalar!=Scalar::e4m3fn || blocks->shape.rank!=2 || blocks->shape.dims[0]!=rows ||
    blocks->shape.dims[1]!=width/16 || blocks->data.range.bytes!=out.size()/16 ||
    !tensor || tensor->scalar!=Scalar::f32 || tensor->shape.rank!=1 || tensor->shape.dims[0]!=1 || tensor->data.range.bytes!=4)
  return {Error::malformed,"NVFP4 scale sidecar schema"};
 std::vector<Byte> block_bytes,tensor_bytes;
 auto status=read(blocks->data,block_bytes); if(!status) return status;
 status=read(tensor->data,tensor_bytes); if(!status) return status;
 float tensor_scale=0; std::memcpy(&tensor_scale,tensor_bytes.data(),4);
 if(!std::isfinite(tensor_scale) || tensor_scale<=0) return {Error::malformed,"NVFP4 tensor scale must be positive finite"};
 for(auto byte:block_bytes) {
  const auto code=std::to_integer<std::uint8_t>(byte);
  if(code>0x7eU) return {Error::malformed,"NVFP4 block scale must be nonnegative finite E4M3"};
 }
 for(std::size_t i=0;i<out.size();++i) {
  const auto code=(std::to_integer<std::uint8_t>(raw[i/2])>>(4*(i%2)))&15;
  const auto scale=scalar_float::e4m3(std::to_integer<std::uint8_t>(block_bytes[i/16]));
  out[i]=microscaled::nvfp4_value(static_cast<std::uint8_t>(code),scale,tensor_scale);
 }
 return {Error::ok,{}};
}
}
