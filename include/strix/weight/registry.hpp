#pragma once
#include "types.hpp"
namespace strix::weight {
inline constexpr FormatId kBf16WeightStorage{"plain.bf16", 1};
inline constexpr FormatId kQ8WeightStorage{"gguf.q8_0", 1};
// Low-bit resident row storages produced at load from a BF16 matrix (strix/weight/host_lowbit.hpp layout).
inline constexpr FormatId kQ4G64WeightStorage{"strix.q4_g64", 1};
inline constexpr FormatId kQ6G64WeightStorage{"strix.q6_g64", 1};
// Geometry-only entries until the Strix pack loader and device consumers land.
inline constexpr FormatId kSP4WeightStorage{"strix.sp4", 1};
inline constexpr FormatId kSP3WeightStorage{"strix.sp3", 1};
inline constexpr FormatId kSP2WeightStorage{"strix.sp2", 1};
[[nodiscard]] inline bool is_strixpack_storage(FormatId id) noexcept {
    return id == kSP4WeightStorage || id == kSP3WeightStorage || id == kSP2WeightStorage;
}
// Strix pack SP4 with the fixed 4-bit table {1,3,5,8,10,14,18,24} (strix/weight/host_lowbit.hpp sp::, 68 B per 128
// weights, one scale byte per 32): produced only by strix-lowbit-calib-pack (calibrated pack, never at load). The
// per-tensor base travels in the pack entry and reaches the consumers as ResidentMatrixView::outer_exponent.
inline constexpr FormatId kSP4TWeightStorage{"strix.sp4t", 1};
inline constexpr std::uint64_t plain_linear = 1ull << 0;
inline constexpr std::uint64_t quantized_linear = 1ull << 1;
inline constexpr std::uint64_t transformed_linear = 1ull << 2;
inline constexpr std::uint64_t row_lookup = 1ull << 3;
inline constexpr std::uint64_t typed_auxiliary = 1ull << 4;
// Empty name means the reader does not recognise the wire type.
FormatId gguf_format(std::uint32_t) noexcept;
std::uint32_t gguf_type(FormatId) noexcept; // UINT32_MAX if not a GGUF wire alias
Status format_geometry(FormatId, const Parameters&, Geometry&);
Status validate_weight(const WeightView&);
Status validate_metadata(const WeightView&);
Status validate_binding(const Registry&, const WeightMapEntry&);
}
