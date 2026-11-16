#pragma once
#include "source.hpp"
#include "registry.hpp"
namespace strix::weight {
// One vendor-free storage contract; operand rounding and reduction remain typed.
struct Q8Storage {
 static constexpr std::uint32_t elements=32, bytes=34, scale_bytes=2;
 static constexpr std::uint64_t row_bytes(std::uint64_t width) noexcept { return width/elements*bytes; }
 static const std::uint8_t* block(const std::uint8_t* row,std::size_t group) noexcept { return row+group*bytes; }
 static std::uint16_t scale_bits(const std::uint8_t* packed) noexcept {
  return static_cast<std::uint16_t>(packed[0] | (std::uint16_t(packed[1])<<8));
 }
 static std::int8_t code(const std::uint8_t* packed,std::size_t lane) noexcept {
  return static_cast<std::int8_t>(packed[scale_bytes+lane]);
 }
};
// The explicit owner outlives every view. Row padding/prefixes are source
// addressing, not a new format or a second interpretation of Q8_0 bytes.
Status q8_weight_from_legacy(std::span<const Byte>,std::shared_ptr<const void> owner,
                            std::uint32_t outputs,std::uint32_t width,std::uint32_t row_bytes,
                            std::uint32_t k_bytes,TensorId,WeightView&);
struct HostQ8Binding {
 WeightView source;
 std::vector<Byte> bytes;
 std::uint32_t outputs=0,width=0,row_bytes=0;
 const std::uint8_t* data() const noexcept { return reinterpret_cast<const std::uint8_t*>(bytes.data()); }
};
// Explicit preparation, never allocation or conversion inside a numerical loop.
Status prepare_host_q8(const WeightView&,HostQ8Binding&);
Status validate_q8_shape(const WeightView&,std::uint64_t outputs,std::uint64_t width);
}
