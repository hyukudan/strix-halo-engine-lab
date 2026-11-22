#pragma once

#include "strix/weight/types.hpp"

#include <concepts>

namespace strix::weight {

enum class TransformKind : std::uint8_t { identity, hadamard, permute };
enum class HadamardBasis : std::uint8_t { sylvester, regular };

// Metadata for the planner's activation transform. A Hadamard sign vector is
// optional; a permutation gathers output[i] from input[indices[i]]. Catalogue
// rows declare the transform shape; a bound view supplies any operand data.
struct Transform {
    TransformKind kind = TransformKind::identity;
    unsigned n = 0;
    const std::int8_t* signs = nullptr;
    const std::uint32_t* indices = nullptr;
    HadamardBasis basis = HadamardBasis::sylvester;
};

template <class D>
concept BlockDecoder = requires(const typename D::View& view, typename D::Raw& raw,
                                unsigned row, unsigned group) {
    { D::kGroup } -> std::convertible_to<unsigned>;
    { D::kAlign } -> std::convertible_to<unsigned>;
    { D::identity } -> std::convertible_to<std::string_view>;
    { D::in_tf } -> std::convertible_to<Transform>;
    { D::out_tf } -> std::convertible_to<Transform>;
    D::load(view, row, group, raw);
};

// A: widen to the storage reference values. dot_then_scale advertises an
// integer-code operand path whose product arithmetic has its own identity.
template <class D>
concept ScalarGroup = BlockDecoder<D> && requires(const typename D::Raw& raw, float* out) {
    D::expand_f32(raw, out);
    { D::dot_then_scale } -> std::convertible_to<bool>;
};

// B: the fragment (and any fragment scale) has the backend's operand class.
template <class D, class Fragment>
concept TileFragment = BlockDecoder<D> && requires(const typename D::Raw& raw, Fragment& fragment) {
    D::fill(raw, fragment);
};

// C: describe a converted image for the planner to build and the cache to lease.
enum class ImageLifetime : std::uint8_t { load, session, request };

struct ImagePlan {
    FormatId source, image;
    std::uint64_t bytes;
    std::string_view arithmetic;
    ImageLifetime lifetime = ImageLifetime::load;
    std::string_view when{}; // Optional selection predicate in the enclosing plan.
};

enum class DecoderProtocol : std::uint8_t { a = 1, b = 2, c = 4 };

constexpr DecoderProtocol operator|(DecoderProtocol a, DecoderProtocol b) noexcept {
    return static_cast<DecoderProtocol>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

enum class ScaleLayout : std::uint8_t {
    none,
    block_f16,                  // One inline F16 multiplier per stored block.
    block_f16_offset,           // Inline F16 multiplier and additive offset.
    superblock_f16,             // Inline F16 multiplier and packed subblock multipliers.
    superblock_f16_min,         // Inline F16 bases for subblock multipliers and minima.
    block_e8m0,                 // One inline exponent per 32 values.
    tile_e8m0_sidecar,          // One E8M0 exponent per 32x32 tile.
    group_e8m0_sidecar,         // One E8M0 exponent per row group of 32.
    group_e4m3_outer_f32,       // E4M3 per 16, with an additional F32 outer scale.
    trellis_f16_sidecars,       // F16 input and output diagonal factors.
    trellis_f16_prefix,         // Same factors stored before the tile rings.
    group_f16_plane,            // F16 per group in a separate plane.
    block_f16_or_row_plane,     // F16 per block, inline or in the row's scale plane.
    chunk_f16_plane,            // F16 per 32 in the Q6P chunk's scale plane.
    channel_f32,               // F32 per output channel.
    group_and_channel,         // F32 or E4M3FN per group and F32 per output channel.
    token_head_f32,             // F32 per token, KV head and K/V half.
};

enum class DecoderFormat : std::uint8_t {
    f32, f16, bf16,
    q8_0, q4_0, q5_0, q5_1, q3_k, q4_k, q5_k, q6_k,
    iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, q2_0, iq2_xxs, q2_k, mxfp4,
    fp8_block, nvfp4, mxfp8, exl3, exl3_transport,
    iu4_g64, q6p32, cb4, u5, u6, int8_channel, w4a8_grouped,
    kv_e4m3, kv_int8, count,
};

struct DecoderFormatInfo {
    FormatId id;
    unsigned block_elements;
    unsigned block_bytes;     // Bytes in the block, excluding prefix/sidecar data.
    unsigned group_elements;  // Zero when selected by the row, head or admitted parameters.
    ScaleLayout scales;
    DecoderProtocol protocols;
    Transform in_tf{};
    Transform out_tf{};
    // For variable-rate trellis rings block_bytes is the bytes per bit of rate.
    // Bit i of the mask admits rate i + 1.
    unsigned rate_mask = 0;

    constexpr unsigned bytes_at_rate(unsigned bits) const noexcept {
        if (rate_mask == 0) return block_bytes;
        if (bits == 0 || bits > 8 || ((rate_mask >> (bits - 1)) & 1U) == 0) return 0;
        return block_bytes * bits;
    }

    constexpr bool uses(DecoderProtocol protocol) const noexcept {
        const auto mask = static_cast<unsigned>(protocol);
        return (static_cast<unsigned>(protocols) & mask) == mask;
    }
};

namespace detail {
inline constexpr auto ab = DecoderProtocol::a | DecoderProtocol::b;
inline constexpr auto abc = ab | DecoderProtocol::c;
inline constexpr Transform hadamard128{TransformKind::hadamard, 128};
inline constexpr Transform convrot256{TransformKind::hadamard, 256, nullptr, nullptr, HadamardBasis::regular};
} // namespace detail

// Storage geometry and applicable protocols. Backend availability and
// numerical qualification live elsewhere. Unscaled F32/F16/BF16 activation
// and state views reuse the corresponding scalar rows.
inline constexpr DecoderFormatInfo kDecoderFormats[] = {
    {{"plain.f32", 1},                   1,   4, 16, ScaleLayout::none, detail::ab},
    {{"plain.f16", 1},                   1,   2, 16, ScaleLayout::none, detail::ab},
    {{"plain.bf16", 1},                  1,   2, 16, ScaleLayout::none, detail::ab},
    {{"gguf.q8_0", 1},                  32,  34, 32, ScaleLayout::block_f16, detail::ab},
    {{"gguf.q4_0", 1},                  32,  18, 32, ScaleLayout::block_f16, detail::ab},
    {{"gguf.q5_0", 1},                  32,  22, 32, ScaleLayout::block_f16, detail::ab},
    {{"gguf.q5_1", 1},                  32,  24, 32, ScaleLayout::block_f16_offset, detail::ab},
    {{"gguf.q3_k", 1},                 256, 110, 256, ScaleLayout::superblock_f16, detail::ab},
    {{"gguf.q4_k", 1},                 256, 144, 256, ScaleLayout::superblock_f16_min, detail::ab},
    {{"gguf.q5_k", 1},                 256, 176, 256, ScaleLayout::superblock_f16_min, detail::ab},
    {{"gguf.q6_k", 1},                 256, 210, 256, ScaleLayout::superblock_f16, detail::ab},
    {{"gguf.iq2_s", 1},                256,  82, 32, ScaleLayout::superblock_f16, detail::abc},
    {{"gguf.iq3_xxs", 1},              256,  98, 32, ScaleLayout::superblock_f16, detail::abc},
    {{"gguf.iq3_s", 1},                256, 110, 32, ScaleLayout::superblock_f16, detail::abc},
    {{"gguf.iq4_nl", 1},                32,  18, 32, ScaleLayout::block_f16, detail::abc},
    {{"gguf.iq4_xs", 1},               256, 136, 32, ScaleLayout::superblock_f16, detail::abc},
    {{"gguf.q2_0.strix", 1},            64,  18, 32, ScaleLayout::block_f16, detail::ab},
    {{"gguf.iq2_xxs", 1},              256,  66, 32, ScaleLayout::superblock_f16, detail::ab},
    {{"gguf.q2_k", 1},                 256,  84, 32, ScaleLayout::superblock_f16_min, detail::ab},
    {{"gguf.mxfp4", 1},                 32,  17, 32, ScaleLayout::block_e8m0, detail::ab},
    {{"block.e4m3-e8m0.32x32", 1},    1024,1024, 32, ScaleLayout::tile_e8m0_sidecar, detail::ab},
    {{"nvidia.nvfp4", 1},               16,   8, 16, ScaleLayout::group_e4m3_outer_f32, detail::ab},
    {{"block.mxfp8.e4m3-e8m0.g32", 1},  32,  32, 32, ScaleLayout::group_e8m0_sidecar, detail::ab},
    {{"exl3.mul1.tiles16", 1},         256,  32, 16, ScaleLayout::trellis_f16_sidecars, detail::abc,
        detail::hadamard128, detail::hadamard128, 0xff},
    {{"exl3.gemma.transport", 1},      256,  32, 16, ScaleLayout::trellis_f16_prefix, detail::abc,
        detail::hadamard128, detail::hadamard128, 0xb8}, // 4, 5, 6, 8 bits.
    {{"gguf.iu4_g64.strix", 1},         64,  34, 64, ScaleLayout::group_f16_plane, detail::abc},
    {{"image.q6p32", 1},                32,  26, 32, ScaleLayout::chunk_f16_plane, detail::abc},
    {{"image.cb4", 1},                  32,  18, 32, ScaleLayout::block_f16_or_row_plane, detail::abc},
    {{"image.u5", 1},                   32,  22, 32, ScaleLayout::block_f16_or_row_plane, detail::abc},
    {{"image.u6", 1},                   32,  26, 32, ScaleLayout::block_f16_or_row_plane, detail::abc},
    {{"linear.i8-channel-convrot256", 1}, 1,   1,  0, ScaleLayout::channel_f32, detail::ab,
        detail::convrot256},
    {{"linear.w4a8-grouped", 1},         2,   1,  0, ScaleLayout::group_and_channel, detail::ab,
        detail::convrot256},
    {{"kv-e4m3-tok-v1", 1},             1,   1,  0, ScaleLayout::token_head_f32, DecoderProtocol::a},
    {{"kv-int8-tok-v1", 1},             1,   1,  0, ScaleLayout::token_head_f32, DecoderProtocol::a},
};

static_assert(sizeof(kDecoderFormats) / sizeof(*kDecoderFormats) == static_cast<unsigned>(DecoderFormat::count));

constexpr const DecoderFormatInfo& decoder_format(DecoderFormat format) noexcept {
    return kDecoderFormats[static_cast<unsigned>(format)];
}

constexpr const DecoderFormatInfo* find_decoder_format(FormatId id) noexcept {
    for (const auto& format : kDecoderFormats) {
        if (format.id.name == id.name && format.id.revision == id.revision) return &format;
    }
    return nullptr;
}

} // namespace strix::weight
