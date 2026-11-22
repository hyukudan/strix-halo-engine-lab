#include "formats/catalog.hpp"
#include "strix/weight/decoder.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"
#include <array>
#include <algorithm>
#include <exception>
namespace strix::weight {
namespace {
struct Wire { std::uint32_t type; FormatId id; };
constexpr Wire wires[] = {
 {0,{"plain.f32",1}}, {1,{"plain.f16",1}}, {2,{"gguf.q4_0",1}}, {3,{"gguf.q4_1",1}},
 {6,{"gguf.q5_0",1}}, {7,{"gguf.q5_1",1}}, {8,{"gguf.q8_0",1}}, {9,{"gguf.q8_1",1}},
 {10,{"gguf.q2_k",1}}, {11,{"gguf.q3_k",1}}, {12,{"gguf.q4_k",1}}, {13,{"gguf.q5_k",1}},
 {14,{"gguf.q6_k",1}}, {15,{"gguf.q8_k",1}}, {16,{"gguf.iq2_xxs",1}}, {17,{"gguf.iq2_xs",1}},
 {18,{"gguf.iq3_xxs",1}}, {19,{"gguf.iq1_s",1}}, {20,{"gguf.iq4_nl",1}}, {21,{"gguf.iq3_s",1}},
 {22,{"gguf.iq2_s",1}}, {23,{"gguf.iq4_xs",1}}, {24,{"typed.i8",1}}, {25,{"typed.i16",1}},
 {26,{"typed.i32",1}}, {27,{"typed.i64",1}}, {28,{"typed.f64",1}}, {29,{"gguf.iq1_m",1}},
 {30,{"plain.bf16",1}}, {34,{"gguf.tq1_0",1}}, {35,{"gguf.tq2_0",1}}, {39,{"gguf.mxfp4",1}},
 {42,{"gguf.q2_0.strix",1}}, {43,{"gguf.nvfp4_tile.strix",1}}, {200,{"gguf.iu4_g64.strix",1}}
};

constexpr auto mxfp8 = decoder_format(DecoderFormat::mxfp8);
static_assert(mxfp8.block_elements == mxfp8.block_bytes);

Status validate_mxfp8_scales(const WeightView& view) {
 const auto* scales = detail::sidecar(view, "block_scales");
 auto expected = view.shape;
 expected.dims[expected.rank-1] /= mxfp8.group_elements;
 if (!scales || scales->scalar != Scalar::e8m0 || scales->shape.rank != expected.rank ||
     !std::equal(expected.dims.begin(), expected.dims.begin()+expected.rank, scales->shape.dims.begin()))
   return {Error::malformed, "MXFP8 requires one E8M0 scale per row group"};
 return {Error::ok, {}};
}

Status read_mxfp8(const WeightView& view, std::vector<float>* decoded) {
 if (view.storage != mxfp8.id) return {Error::unknown_format, "expected MXFP8 storage"};
 auto status = validate_metadata(view);
 if (!status) return status;
 const auto count = view.payload.range.bytes;
 if (count > SIZE_MAX / sizeof(float)) return {Error::memory_budget, "MXFP8 tensor exceeds addressable scratch storage"};
 const auto width = static_cast<std::size_t>(view.shape.dims[view.shape.rank-1]);
 const auto rows = static_cast<std::size_t>(count) / width;
 microscaled::StagedMatrix matrix;
 status = matrix.load({microscaled::MatrixFormat::mxfp8, rows, width, view.payload,
                       detail::sidecar(view, "block_scales")->data, {}, {}});
 if (!status) return status;
 std::vector<float> scratch(decoded ? 0 : width);
 if (decoded) decoded->resize(static_cast<std::size_t>(count));
 for (std::size_t row = 0; row < rows; ++row) {
   auto out = decoded ? std::span(*decoded).subspan(row*width, width) : std::span(scratch);
   status = matrix.decode_row(row, out);
   if (!status) return status;
 }
 return {Error::ok, {}};
}

Status validate_mxfp8(const WeightView& view) {
 return read_mxfp8(view, nullptr);
}

Status decode_mxfp8(const WeightView& view, const DecodeRequest& request, DecodeBuffer& out) {
 out.written = 0;
 if (request.output != Scalar::f32 || !request.arithmetic.name.empty() || request.arithmetic.revision != 0)
   return {Error::unsupported_arithmetic, "MXFP8 reference supports storage widening to F32 only"};
 const auto count = view.payload.range.bytes;
 if (request.first_element > count || request.count > count-request.first_element)
   return {Error::unsupported_shape, "decode slice outside tensor"};
 if (request.count > out.bytes.size()/sizeof(float))
   return {Error::malformed, "decode output buffer too small"};
 std::vector<float> decoded;
 const auto status = read_mxfp8(view, &decoded);
 if (!status) return status;
 const auto bytes = std::as_bytes(std::span(decoded).subspan(static_cast<std::size_t>(request.first_element),
                                                             static_cast<std::size_t>(request.count)));
 std::ranges::copy(bytes, out.bytes.begin());
 out.written = bytes.size();
 return {Error::ok, {}};
}
}
FormatId gguf_format(std::uint32_t type) noexcept {
 for (const auto& wire : wires) if (wire.type == type) return wire.id;
 return {{}, 0};
}
std::uint32_t gguf_type(FormatId id) noexcept {
 for (const auto& wire : wires) if (wire.id == id) return wire.type;
 return UINT32_MAX;
}
const Registry& builtin_registry() {
 // Explicit construction, no dynamic registration or cross-TU initialisers.
 static const std::vector<FormatDescriptor> formats = [] {
   std::vector<FormatDescriptor> result;
   for (auto part : {detail::plain_formats(), detail::ggml_formats(), detail::ds41_formats(),
        detail::packed_formats(), detail::fp8_formats(), detail::exl3_formats(),
        detail::channel_formats(), detail::nvfp4_formats()})
     result.insert(result.end(), part.begin(), part.end());
   auto format = detail::descriptor(mxfp8.id, "F8_E4M3/F8_E8M0:g32", quantized_linear, true,
       detail::fixed<mxfp8.block_elements, mxfp8.block_bytes, mxfp8.group_elements>);
   format.validate = validate_mxfp8;
   format.decode_reference = decode_mxfp8;
   result.push_back(format);
   return result;
 }();
 static const Registry registry{formats, legacy_arithmetic()};
 return registry;
}
const FormatDescriptor* find_format(const Registry& registry, FormatId id) noexcept {
 for (const auto& format : registry.formats) if (format.id == id) return &format;
 return nullptr;
}
Status format_geometry(FormatId id, const Parameters& parameters, Geometry& geometry) {
 const auto* format = find_format(builtin_registry(), id);
 if (!format) return {Error::unknown_format, "unregistered storage identity"};
 return format->geometry(parameters, geometry);
}
Status validate_metadata(const WeightView& weight) {
 try {
   auto status = detail::validate_metadata(weight);
   if (status && weight.storage == mxfp8.id) return validate_mxfp8_scales(weight);
   return status;
 } catch (const std::exception& error) { return {Error::malformed, error.what()}; }
}

Status validate_weight(const WeightView& weight) {
 const auto* format = find_format(builtin_registry(), weight.storage);
 if (!format) return {Error::unknown_format, "unregistered storage identity"};
 try { return format->validate(weight); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
}
