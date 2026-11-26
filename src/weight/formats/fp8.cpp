#include "catalog.hpp"
#include "strix/weight/formats/fp8_block_admission.hpp"
#include "strix/weight/formats/fp8_block.hpp"
namespace strix::weight::detail {
std::span<const FormatDescriptor> fp8_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"block.e4m3-e8m0.32x32",1},"F8_E4M3/F8_E8M0:32x32",quantized_linear,true,fixed<1024,1024,32,32>)
 };
 return formats;
}
Status decode_fp8(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 const auto width=view.shape.dims[1],rows=view.shape.dims[0];
 const auto* scale=sidecar(view,"block_scales");
 if(!scale || scale->scalar!=Scalar::e8m0 || scale->shape.rank!=2 || scale->shape.dims[0]!=rows/32 ||
    scale->shape.dims[1]!=width/32 || scale->data.range.bytes!=rows/32*(width/32))
  return {Error::malformed,"FP8 requires exact E8M0 32x32 scale sidecar"};
 std::vector<Byte> scales; auto status=read(scale->data,scales); if(!status) return status;
 const auto codes=std::span(reinterpret_cast<const std::uint8_t*>(raw.data()),raw.size());
 const auto exponents=std::span(reinterpret_cast<const std::uint8_t*>(scales.data()),scales.size());
 const auto reason=fp8_block_admission::check(codes,exponents,rows,width);
 if(!reason.empty()) return {Error::malformed,reason};
 const BlockScaledE4M3<32, 32, std::uint8_t> matrix(codes,exponents,rows,width);
 for(std::size_t row=0;row<rows;++row)
  matrix.decode_row(row,std::span(out).subspan(row*width,width));
 return {Error::ok,{}};
}
}
