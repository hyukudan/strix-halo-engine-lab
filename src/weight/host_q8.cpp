#include "strix/weight/host_q8.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "core/iu4_onfly.hpp"
#include "h3/te/te_sha256.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
namespace strix::weight {
namespace {
class RowSource final : public Source {
 std::shared_ptr<const Source> source_;
 std::uint64_t rows_,logical_,stride_,prefix_;
 Digest256 digest_{};
public:
 RowSource(std::shared_ptr<const Source> source,std::uint64_t rows,std::uint64_t logical,
           std::uint64_t stride,std::uint64_t prefix):source_(std::move(source)),rows_(rows),logical_(logical),stride_(stride),prefix_(prefix) {
  h3::te::Sha256 hash;
  std::array<Byte,65536> bytes{};
  for(std::uint64_t offset=0;offset<size();) {
   const auto count=static_cast<std::size_t>(std::min<std::uint64_t>(bytes.size(),size()-offset));
   const auto status=read({offset,count},std::span(bytes).first(count));
   if(!status) throw std::runtime_error(status.detail);
   hash.update(bytes.data(),count); offset+=count;
  }
  digest_.bytes=hash.finish();
 }
 Status read(ByteRange range,std::span<Byte> out) const override {
  if(range.offset>size() || range.bytes>size()-range.offset || out.size()!=range.bytes)
   return {Error::malformed,"Q8 source range"};
  std::uint64_t copied=0;
  while(copied<range.bytes) {
   const auto position=range.offset+copied,row=position/logical_,column=position%logical_;
   const auto count=std::min(logical_-column,range.bytes-copied);
   auto status=source_->read({row*stride_+prefix_+column,count},out.subspan(static_cast<std::size_t>(copied),static_cast<std::size_t>(count)));
   if(!status) return status;
   copied+=count;
  }
  return {Error::ok,{}};
 }
 std::uint64_t size() const noexcept override { return rows_*logical_; }
 Digest256 identity() const override { return digest_; }
 std::string shard_name() const override { return source_->shard_name(); }
};
}

Status matrix_load_geometry(FormatId storage, const Shape& shape, MatrixLoadGeometry& out) {
 if (!(storage == kBf16WeightStorage) && !(storage == kQ8WeightStorage) && !(storage == kQ4G64WeightStorage) &&
     !(storage == kQ6G64WeightStorage))
  return {Error::unknown_format, "no compact BF16 matrix loader for requested storage"};
 if (shape.rank != 2 || !shape.dims[0] || !shape.dims[1])
  return {Error::unsupported_shape, "weight matrix requires nonempty [N,K]"};
 Geometry geometry{};
 if (const auto status = format_geometry(storage, {}, geometry); !status) return status;
 if (shape.dims[1] % geometry.block_elements)
  return {Error::unsupported_shape, "weight row does not contain whole format blocks"};
 const auto blocks = shape.dims[1] / geometry.block_elements;
 if (blocks > UINT64_MAX / geometry.block_bytes)
  return {Error::memory_budget, "weight row byte count overflow"};
 const auto row_bytes = blocks * geometry.block_bytes;
 if (shape.dims[0] > UINT64_MAX / row_bytes)
  return {Error::memory_budget, "weight matrix byte count overflow"};
 out = {row_bytes, shape.dims[0] * row_bytes};
 return {Error::ok, {}};
}

namespace {
// Low-bit superblocks from staged BF16 rows (host_lowbit.hpp contract); K % 128 == 0 is checked by the geometry.
template <class Storage>
Status load_bf16_lowbit(const Slice& source, std::uint64_t input_bytes, std::span<Byte> out) {
 constexpr std::size_t block_bytes = Storage::elements * sizeof(std::uint16_t);
 std::array<Byte, 64 * block_bytes> staging;
 std::size_t written = 0;
 for (std::uint64_t offset = 0; offset < input_bytes;) {
  const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(staging.size(), input_bytes - offset));
  Slice chunk{};
  if (const auto status = subslice(source, {offset, count}, chunk); !status) return status;
  if (const auto status = read_slice(chunk, std::span(staging).first(count)); !status) return status;
  for (std::size_t block = 0; block < count; block += block_bytes) {
   float values[Storage::elements];
   for (std::size_t lane = 0; lane < Storage::elements; ++lane) {
    const auto pos = block + lane * sizeof(std::uint16_t);
    values[lane] = core::onfly::bf16_bits_to_f32(static_cast<std::uint16_t>(
        std::to_integer<unsigned>(staging[pos]) | (std::to_integer<unsigned>(staging[pos + 1]) << 8U)));
   }
   const auto encoded = lowbit::encode_block<Storage>(values, reinterpret_cast<std::uint8_t*>(out.data() + written));
   if (encoded.code == lowbit::EncodeStatus::nonfinite) return {Error::malformed, "nonfinite BF16 matrix weight"};
   if (!encoded) return {Error::malformed, "low-bit weight group scale exceeds the FP16 range"};
   written += Storage::bytes;
  }
  offset += count;
 }
 return {Error::ok, {}};
}
}  // namespace

Status load_bf16_matrix(const Slice& source, const Shape& shape, FormatId destination, std::span<Byte> out) {
 MatrixLoadGeometry input{}, output{};
 if (const auto status = matrix_load_geometry(kBf16WeightStorage, shape, input); !status) return status;
 if (const auto status = matrix_load_geometry(destination, shape, output); !status) return status;
 if (source.range.bytes != input.bytes || out.size() != output.bytes)
  return {Error::malformed, "weight matrix payload size differs from planned geometry"};
 if (destination == kBf16WeightStorage) return read_slice(source, out);
 if (destination == kQ4G64WeightStorage) return load_bf16_lowbit<Q4G64Storage>(source, input.bytes, out);
 if (destination == kQ6G64WeightStorage) return load_bf16_lowbit<Q6G64Storage>(source, input.bytes, out);

 constexpr std::size_t block_bytes = Q8Storage::elements * sizeof(std::uint16_t);
 std::array<Byte, 65536> staging;
 std::size_t written = 0;
 for (std::uint64_t offset = 0; offset < input.bytes;) {
  const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(staging.size(), input.bytes - offset));
  Slice chunk{};
  if (const auto status = subslice(source, {offset, count}, chunk); !status) return status;
  if (const auto status = read_slice(chunk, std::span(staging).first(count)); !status) return status;
  for (std::size_t block = 0; block < count; block += block_bytes) {
   float values[Q8Storage::elements];
   float amax = 0.0F;
   for (std::size_t lane = 0; lane < Q8Storage::elements; ++lane) {
    const auto pos = block + lane * sizeof(std::uint16_t);
    const auto bits = static_cast<std::uint16_t>(std::to_integer<unsigned>(staging[pos]) |
                                               (std::to_integer<unsigned>(staging[pos + 1]) << 8U));
    values[lane] = core::onfly::bf16_bits_to_f32(bits);
    if (!std::isfinite(values[lane])) return {Error::malformed, "nonfinite BF16 matrix weight"};
    amax = std::max(amax, std::abs(values[lane]));
   }
   auto* packed = reinterpret_cast<std::uint8_t*>(out.data() + written);
   const float scale = amax / 127.0F;
   if (scale < std::numeric_limits<float>::min()) {
    // This FP16 scale is zero. Canonicalize the block before an FP32
    // reciprocal of a subnormal could overflow in the shared encoder.
    std::fill_n(packed, Q8Storage::bytes, std::uint8_t{0});
   } else {
    if ((core::onfly::f32_to_f16_bits(scale) & 0x7c00U) == 0x7c00U)
     return {Error::malformed, "BF16 block scale overflows Q8_0 FP16 scale"};
    core::onfly::quantise_q8_0_block(values, packed);
   }
   written += Q8Storage::bytes;
  }
  offset += count;
 }
 return {Error::ok, {}};
}

namespace {
class PackedMatrixSource final : public Source {
    Slice input_;
    std::uint64_t bytes_;
    FormatId storage_;
    std::uint64_t elements_, block_;  // input elements and stored bytes per block
    mutable std::once_flag identity_once_;
    mutable Digest256 identity_{};

    Status blocks(std::uint64_t first, std::span<Byte> out) const {
        const auto input_block = elements_ * sizeof(std::uint16_t);
        const auto count = out.size() / block_;
        Slice input{};
        if (auto status = subslice(input_, {first * input_block, count * input_block}, input); !status)
            return status;
        return load_bf16_matrix(input, {{count, elements_}, 2}, storage_, out);
    }

public:
    PackedMatrixSource(Slice input, std::uint64_t bytes, FormatId storage, std::uint64_t elements, std::uint64_t block)
        : input_(std::move(input)), bytes_(bytes), storage_(storage), elements_(elements), block_(block) {}

    std::uint64_t size() const noexcept override { return bytes_; }
    std::string shard_name() const override { return input_.source->shard_name(); }

    Status read(ByteRange range, std::span<Byte> out) const override {
        if (range.offset > bytes_ || range.bytes > bytes_ - range.offset || range.bytes != out.size())
            return {Error::malformed, "packed matrix read outside its range"};
        auto offset = range.offset;
        std::array<Byte, std::max(Q8Storage::bytes, Q6G64Storage::bytes)> storage{};
        const auto block = std::span(storage).first(static_cast<std::size_t>(block_));
        while (!out.empty()) {
            const auto within = static_cast<std::size_t>(offset % block_);
            const auto whole = static_cast<std::size_t>(out.size() / block_ * block_);
            std::size_t count;
            if (within == 0 && whole != 0) {
                count = whole;
                if (auto status = blocks(offset / block_, out.first(count)); !status) return status;
            } else {
                count = std::min(out.size(), block.size() - within);
                if (auto status = blocks(offset / block_, block); !status) return status;
                std::copy_n(block.begin() + static_cast<std::ptrdiff_t>(within), count, out.begin());
            }
            out = out.subspan(count);
            offset += count;
        }
        return {Error::ok, {}};
    }

    Digest256 identity() const override {
        std::call_once(identity_once_, [&] {
            // Borrow this source for the duration of the synchronous digest.
            const auto self = std::shared_ptr<const Source>(input_.source, this);
            if (const auto status = digest_slice({self, {0, size()}}, identity_); !status)
                throw std::runtime_error(status.detail);
        });
        return identity_;
    }
};
}  // namespace

Status prepare_bf16_matrix_slice(const Slice& source, const Shape& shape, FormatId destination, Slice& out) {
    MatrixLoadGeometry input{}, output{};
    if (auto status = matrix_load_geometry(kBf16WeightStorage, shape, input); !status) return status;
    if (auto status = matrix_load_geometry(destination, shape, output); !status) return status;
    if (source.range.bytes != input.bytes)
        return {Error::malformed, "BF16 source size differs from matrix geometry"};
    Slice checked{};
    if (auto status = subslice(source, {0, input.bytes}, checked); !status) return status;
    if (destination == kBf16WeightStorage) {
        out = std::move(checked);
    } else {
        Geometry geometry{};
        if (auto status = format_geometry(destination, {}, geometry); !status) return status;
        out = {std::make_shared<PackedMatrixSource>(std::move(checked), output.bytes, destination,
                                                    geometry.block_elements, geometry.block_bytes),
               {0, output.bytes}};
    }
    return {Error::ok, {}};
}

Status q8_weight_from_legacy(std::span<const Byte> raw,std::shared_ptr<const void> owner,
                            std::uint32_t outputs,std::uint32_t width,std::uint32_t stride,
                            std::uint32_t prefix,TensorId tensor,WeightView& out) {
 if(!owner || !raw.data()) return {Error::stale_source,"legacy Q8 view requires an explicit live owner"};
 if(!outputs || !width || width%Q8Storage::elements) return {Error::unsupported_shape,"Q8 matrix requires positive N and K multiple of 32"};
 const auto logical=Q8Storage::row_bytes(width);
 if(logical>UINT32_MAX || std::uint64_t(prefix)+logical>stride ||
    std::uint64_t(outputs-1)*stride+prefix+logical>raw.size()) return {Error::malformed,"legacy Q8 row stride/prefix/buffer size"};
 try {
  auto base=view_source(raw,std::move(owner),"host-q8");
  auto source=std::make_shared<RowSource>(std::move(base),outputs,logical,stride,prefix);
  WeightView result{};
  result.tensor=tensor; result.storage=kQ8WeightStorage; result.shape={{outputs,width,0,0},2};
  result.geometry={Q8Storage::elements,Q8Storage::bytes,Q8Storage::elements,1,1,Q8Storage::elements,1,true};
  result.payload={std::move(source),{0,std::uint64_t(outputs)*logical}};
  auto status=validate_weight(result); if(!status) return status;
  out=std::move(result); return {Error::ok,{}};
 } catch(const std::bad_alloc&) { return {Error::memory_budget,"Q8 source preparation allocation"}; }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
Status validate_q8_shape(const WeightView& weight,std::uint64_t outputs,std::uint64_t width) {
 if(!(weight.storage==kQ8WeightStorage)) return {Error::unsupported_arithmetic,"only existing Q8_0 operand contract is registered; IQ/trellis/NVFP4 require separate qualified arithmetic"};
 if(weight.shape.rank!=2 || weight.shape.dims[0]!=outputs || weight.shape.dims[1]!=width ||
    !outputs || !width || outputs>UINT32_MAX || width>UINT32_MAX || width%Q8Storage::elements ||
    Q8Storage::row_bytes(width)>UINT32_MAX) return {Error::unsupported_shape,"Q8 logical shape differs from launch"};
 if(!weight.sidecars.empty()) return {Error::malformed,"raw Q8_0 has no sidecar operands"};
 return validate_weight(weight);
}
Status prepare_host_q8(const WeightView& weight,HostQ8Binding& out) {
 if(weight.shape.rank!=2) return {Error::unsupported_shape,"Q8 host preparation requires rank two"};
 auto status=validate_q8_shape(weight,weight.shape.dims[0],weight.shape.dims[1]); if(!status) return status;
 if(weight.payload.range.bytes>SIZE_MAX) return {Error::memory_budget,"Q8 host image size overflow"};
 try {
  HostQ8Binding result;
  result.source_=weight;
  auto bytes=std::make_shared<std::vector<Byte>>(static_cast<std::size_t>(weight.payload.range.bytes));
  status=read_slice(weight.payload,*bytes); if(!status) return status;
  result.bytes_=std::move(bytes);
  out=std::move(result); return {Error::ok,{}};
 } catch(const std::bad_alloc&) { return {Error::memory_budget,"Q8 host image allocation"}; }
}
Status HostQ8Binding::validate(const WeightView& weight,const std::uint8_t* pointer,std::uint32_t stride,std::uint32_t prefix) const {
 if(!data()) return {Error::malformed,"Q8 launch requires an owning prepared binding"};
 auto status=validate_q8_shape(weight,source_.shape.dims[0],source_.shape.dims[1]); if(!status) return status;
 if(weight.tensor!=source_.tensor || weight.payload.source!=source_.payload.source ||
    weight.payload.range.offset!=source_.payload.range.offset || weight.payload.range.bytes!=source_.payload.range.bytes ||
    weight.byte_strides!=source_.byte_strides || weight.has_byte_strides!=source_.has_byte_strides)
  return {Error::stale_source,"Q8 binding belongs to a different source"};
 if(pointer!=data() || stride!=row_bytes() || prefix)
  return {Error::malformed,"Q8 raw pointer/stride/prefix differs from prepared binding"};
 return {Error::ok,{}};
}
Status HostQ8Binding::acquire(ExecutionContext& context,ImageLease& lease) const {
 if(!image_ || !cache_) return {Error::unsupported_backend,"native Q8 launch requires a prepared runtime image"};
 return acquire_image(*cache_,key_,plan_,source_,context,lease);
}
Status prepare_q8_image(const WeightView& weight,const Plan& plan,std::shared_ptr<ImageCache> cache,
                        ExecutionContext& context,HostQ8Binding& out) {
 auto status=validate_q8_shape(weight,weight.shape.dims[0],weight.shape.dims[1]); if(!status) return status;
 if(!cache || plan.selected.resources.image_bytes!=weight.payload.range.bytes)
  return {Error::malformed,"Q8 runtime image must contain the complete compact payload"};
 HostQ8Binding result; result.source_=weight; result.plan_=plan; result.cache_=std::move(cache);
 status=make_image_key(weight,plan,"q8-compact","q8-binding-v2",result.key_); if(!status) return status;
 result.image_=std::make_shared<ImageLease>();
 status=acquire_image(*result.cache_,result.key_,plan,weight,context,*result.image_); if(!status) return status;
 status=record_last_use(*result.image_,context); if(!status) return status;
 out=std::move(result); return {Error::ok,{}};
}
}
