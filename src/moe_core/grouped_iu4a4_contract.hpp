#pragma once
// Rotated integer A4W4 operand class of the grouped expert projection (Home: shared moe_core contract; the HIP body is the
// grouped trait of core/iu4a4_dense.hpp on v_wmma_i32_16x16x16_iu4; a backend without integer 4-bit WMMA declines by
// name). Sibling of grouped_w4a4.hpp (NVFP4 activations on block-scaled FP4 MMA): same plan (TileRows of routed rows of
// one expert), same outputs; both operands change:
//   weights: once at load, each expert matrix (NVFP4 or MXFP8 as stored) is requantised to the rotated IU4 image of
//     dense-iu4a4-rot-v1 (iu4a4_rot::requantise_values on the EXACT BF16 operand value of every weight); a gate/up matrix
//     is built in the paired column order (iu4a4_rot::paired_column) so the SwiGLU epilogue meets gate and up in a lane.
//   activations: the A4 image of dense-iu4a4-rot-v1 (row-major: the grouped launch gathers rows). Gate/up: the TOKEN rows quantised once (not once per assignment),
//     read through the plan's assignment -> token map. Down: the activated rows h (FP32 [assignment][ffn], the BF16 value
//     of the incumbent epilogue) quantised in assignment order.
// Arithmetic identity "grouped-iu4a4-rot-v1": every output is iu4a4_rot::dot of its row and rotated column (the dense
// contract per tile); h = activate_bf16(gate, up) as the incumbent gate/up epilogue. Bits differ from the BF16 tile:
// FAST / TURBO class only, behind gate v2.
#include "core/iu4a4_rot.hpp"
#include "moe_core/grouped_iu4a4_plan.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__HIPCC__)
#define STRIX_GI4_HD __host__ __device__
#else
#define STRIX_GI4_HD
#endif

namespace strix::moe_core::grouped_iu4a4 {

namespace packed = strix::weight::microscaled;
namespace rot = strix::core::iu4a4_rot;
using grouped_microscaled::TileRows;
inline constexpr unsigned kCtaRows = 128U;  // rows per CTA of the grouped GEMM; taller plan tiles run as row slices
// Widest projection input (hidden and intermediate) the pass quantises in its one-CTA-per-row form (the verdict-gated
// launches of the token image and of h), and the most tiles its one-CTA validation launch admits (interval table in LDS).
inline constexpr unsigned kMaxRowInputs = 24576U;
inline constexpr unsigned kMaxPassTiles = 2048U;

constexpr std::size_t image_code_bytes(unsigned in, unsigned n) { return std::size_t(n) * in / 2U; }
constexpr std::size_t image_step_floats(unsigned in, unsigned n) { return std::size_t(n) * (in / rot::kBlock); }
inline bool supported(unsigned in, unsigned n, bool paired) {
    return in > 0U && in % rot::kBlock == 0U && in <= rot::kMaxInputs && n > 0U && n % (paired ? 32U : 16U) == 0U;
}

// Weight source of one expert matrix: rotated column n reads stored row paired_column(n, pair_rows) (pair_rows = 0:
// row n), value = the BF16 operand the EXACT tile decodes (PackedOperand::decode_pair), as FP32.
template <packed::MatrixFormat Format>
struct ExpertSource {
    packed::PackedMatrix matrix;
    unsigned pair_rows{};
    rot::WeightRule rule = rot::WeightRule::amax;  // weight image rule (contract v1 / v2)
    STRIX_GI4_HD float value(std::size_t row, unsigned k) const {
        const packed::PackedOperand<Format> op{matrix};
        const unsigned k0 = k & ~15U, j = k - k0;
        const auto address = op.tile16_address(row, k0);
        const std::uint8_t* codes = address.codes + (packed::PackedOperand<Format>::kNvfp4 ? j / 2U : (j & ~1U));
        const std::uint32_t pair = op.decode_pair(codes, packed::PackedOperand<Format>::decode_scale(*address.scale_code));
        const std::uint32_t half = (j & 1U) != 0U ? pair >> 16U : pair & 0xffffU;
        return weight::scalar_float::u2f(half << 16U);
    }
    STRIX_GI4_HD float requantise(unsigned col, unsigned b, unsigned n, std::uint8_t* dst) const {
        const std::size_t row = pair_rows != 0U ? rot::paired_column(col, pair_rows) : col;
        const auto v = [&](unsigned i) { return value(row, b * rot::kBlock + i); };
        return rot::requantise_values(v, b, col, n, dst, rule);
    }
};

// Expert index of a tile from its weights' code address (host check before binding: every expert matrix of the layer
// starts at codes_from + e * codes_stride, so the address is an index; the image then holds expert e at e).
inline bool expert_codes_strided(std::span<const packed::PackedMatrix> experts, const std::uint8_t* codes_from, std::size_t codes_stride) {
    for (std::size_t e = 0; e < experts.size(); ++e)
        if (experts[e].codes != codes_from + e * codes_stride) return false;
    return codes_stride > 0U;
}

static_assert(rot::kBlock == kInputBlock, "act_image_bytes assumes the contract's 128-input block");

// Host admission of a pass: nullptr, or why nothing may be enqueued. Rows (none, fewer assignments than rows, past the
// quantiser grid), operands, shapes the class takes, the activation workspace, and both image views: present, one or more
// expert slots, a per-expert extent no smaller than the shape's image, and the code-address rule (base, stride) the
// tiles' experts are located by.
inline const char* refusal(const Pass& p) {
    const unsigned gate_width = 2U * p.intermediate;
    if (p.rows == 0U || p.assignments < p.rows || std::size_t(p.assignments) + 15U > 0x7fffffffU)
        return "rows: none, fewer assignments than rows, or past the quantiser grid";
    if (!p.tokens || !p.h || !p.y || !p.act || !p.assignment_rows || !p.gate_up || !p.down || !p.index_base || !p.fault)
        return "missing operand";
    if (p.tiles == 0U || p.tiles > kMaxPassTiles || p.max_tile_rows == 0U || p.max_tile_rows > p.assignments ||
        !supported(p.hidden, gate_width, true) || !supported(p.intermediate, p.hidden, false) ||
        p.hidden > kMaxRowInputs || p.intermediate > kMaxRowInputs)
        return "shape the class does not take";
    if (reinterpret_cast<std::uintptr_t>(p.act) % 256U != 0U ||
        p.act_size < act_bytes(p.rows, p.assignments, p.hidden, p.intermediate))
        return "activation workspace unaligned or below act_bytes";
    if (!p.images) return "missing images";
    // Every expert slot starts on the 16-byte boundary the 16-byte operand loads need (base and per-expert extents), and
    // the image's total extent (experts x extent) is addressable.
    const auto image = [](const ExpertImage& i, unsigned in, unsigned n) {
        constexpr std::size_t kMax = ~std::size_t{0};
        return i.codes && i.steps && i.experts > 0U && i.code_bytes >= image_code_bytes(in, n) &&
               i.step_floats >= image_step_floats(in, n) && reinterpret_cast<std::uintptr_t>(i.codes) % 16U == 0U &&
               reinterpret_cast<std::uintptr_t>(i.steps) % 16U == 0U && i.code_bytes % 16U == 0U &&
               i.step_floats % 4U == 0U && i.code_bytes <= kMax / i.experts &&
               i.step_floats <= kMax / sizeof(float) / i.experts;
    };
    if (!image(p.images->gate_up, p.hidden, gate_width) || !image(p.images->down, p.intermediate, p.hidden))
        return "image view missing, without experts, below the shape's extent, off the 16-byte slot rule or past the address space";
    if (!p.images->gate_up_codes || !p.images->down_codes || p.images->gate_up_stride == 0U || p.images->down_stride == 0U)
        return "missing code-address rule";
    return nullptr;
}

}  // namespace strix::moe_core::grouped_iu4a4

#undef STRIX_GI4_HD
