#include "strix/weight/decoder.hpp"

#include "core/iu4.hpp"
#include "gemma4/exl3/format.hpp"
#include "h3/gemm/operand.hpp"
#include "strix/cuda/core/nvfp4.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/kv_storage.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/weight/formats/ggml.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/reference.hpp"
#include "strix/weight/source.hpp"
#include "xport/crc32c_fold.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
using namespace strix;
using namespace strix::weight;
using F = DecoderFormat;
using P = DecoderProtocol;
using Trellis = core::trellis::Format;

static_assert(decoder_format(F::q8_0).block_bytes == ds41::q8_0_block_bytes);
static_assert(decoder_format(F::q8_0).block_elements == ds41::q8_0_block_elements);
static_assert(decoder_format(F::iq2_xxs).block_bytes == ds41::iq2_xxs_block_bytes);
static_assert(decoder_format(F::iq2_xxs).block_elements == ds41::super_block_elements);
static_assert(decoder_format(F::q2_k).block_bytes == ds41::q2_k_block_bytes);
static_assert(decoder_format(F::q2_k).block_elements == ds41::super_block_elements);
static_assert(decoder_format(F::mxfp4).block_bytes == ds41::mxfp4_block_bytes);
static_assert(decoder_format(F::mxfp4).block_elements == ds41::mxfp4_block_elements);
static_assert(decoder_format(F::iq3_s).block_bytes == formats::Iq3S::bytes);
static_assert(decoder_format(F::iq3_s).block_elements == formats::Iq3S::elements);
static_assert(decoder_format(F::iq4_nl).block_bytes == iq4_nl::block_bytes);
static_assert(decoder_format(F::iq4_nl).block_elements == iq4_nl::block_elements);
static_assert(decoder_format(F::q6p32).block_bytes == q6p::q6p32_bytes(1, 32));
static_assert(decoder_format(F::cb4).block_bytes == ds41::packed_group_bytes(ds41::PackedKind::cb4));
static_assert(decoder_format(F::u5).block_bytes == ds41::packed_group_bytes(ds41::PackedKind::u5));
static_assert(decoder_format(F::u6).block_bytes == ds41::packed_group_bytes(ds41::PackedKind::u6));
static_assert(decoder_format(F::nvfp4).block_elements == microscaled::kNvfp4BlockColumns);
static_assert(decoder_format(F::mxfp8).block_elements == microscaled::kMxfp8BlockColumns);
static_assert(decoder_format(F::exl3).block_elements == Trellis::kStates);
static_assert(decoder_format(F::exl3).bytes_at_rate(8) == Trellis::tile_bytes(8));
static_assert(decoder_format(F::exl3).bytes_at_rate(0) == 0);
static_assert(decoder_format(F::exl3).bytes_at_rate(9) == 0);
static_assert(decoder_format(F::exl3_transport).bytes_at_rate(3) == 0);
static_assert(decoder_format(F::exl3_transport).in_tf.n == 128);
static_assert(decoder_format(F::exl3_transport).out_tf.kind == TransformKind::hadamard);
static_assert(decoder_format(F::int8_channel).in_tf.basis == HadamardBasis::regular);
static_assert(decoder_format(F::w4a8_grouped).out_tf.kind == TransformKind::identity);
static_assert(kv::bytes_per_token(kv::Form::bf16, {2, 32}) == 2 * 2 * 32 * decoder_format(F::bf16).block_bytes);
static_assert(find_decoder_format({"gguf.q8_0", 2}) == nullptr);
static_assert(std::is_trivially_copyable_v<Transform>);

constexpr std::array<std::uint32_t, 2> permutation{1, 0};
constexpr Transform swapped{TransformKind::permute, 2, nullptr, permutation.data()};
static_assert(swapped.indices[0] == 1);

std::array<bool, static_cast<unsigned>(F::count)> covered{};

void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

// The CPU fixture stages one storage unit (a full superblock for subblock
// formats). The caller supplies the existing numerical primitive and sidecars.
// A CPU fragment in this fixture contains the widened coefficients of a tile.
template <F Format, unsigned Elements, class Widen>
struct CpuDecoder {
    static constexpr auto spec = decoder_format(Format);
    using Fragment = std::array<float, Elements>;
    struct View {
        const std::uint8_t* base;
        std::size_t row_pitch, group_pitch;
        unsigned bytes;
        const Widen* widen;
    };
    struct Raw {
        static constexpr unsigned blocks = (Elements + spec.block_elements - 1) / spec.block_elements;
        std::array<std::uint8_t, spec.rate_mask ? spec.bytes_at_rate(8) : spec.block_bytes * blocks> codes;
        unsigned bytes;
        const Widen* widen;
    };
    static constexpr unsigned kGroup = Elements, kAlign = 1;
    static constexpr std::string_view identity = "fixture-storage-f32-v1";
    static constexpr auto in_tf = spec.in_tf, out_tf = spec.out_tf;
    static constexpr bool dot_then_scale = Format == F::q8_0;

    static void load(const View& view, unsigned row, unsigned group, Raw& raw) {
        check(view.bytes <= raw.codes.size(), spec.id.name);
        std::copy_n(view.base + row * view.row_pitch + group * view.group_pitch, view.bytes, raw.codes.begin());
        raw.bytes = view.bytes;
        raw.widen = view.widen;
    }
    static void expand_f32(const Raw& raw, float* out) {
        (*raw.widen)(std::span(raw.codes).first(raw.bytes), std::span<float, Elements>(out, Elements));
    }
    static void fill(const Raw& raw, Fragment& fragment) requires (spec.uses(P::b)) {
        expand_f32(raw, fragment.data());
    }
};

template <F Format, unsigned Elements, class Widen>
auto expand(const std::uint8_t* base, unsigned bytes, std::size_t row_pitch, std::size_t group_pitch,
            unsigned row, unsigned group, const Widen& widen) {
    using D = CpuDecoder<Format, Elements, Widen>;
    static_assert(ScalarGroup<D>);
    static_assert(TileFragment<D, typename D::Fragment> == D::spec.uses(P::b));
    static_assert(std::is_trivially_copyable_v<typename D::View> && std::is_trivially_copyable_v<typename D::Raw>);
    typename D::Raw raw{};
    typename D::Fragment fragment{};
    D::load({base, row_pitch, group_pitch, bytes, &widen}, row, group, raw);
    if constexpr (Format == F::exl3 || Format == F::exl3_transport) D::fill(raw, fragment);
    else D::expand_f32(raw, fragment.data());
    return fragment;
}

template <F Format, unsigned Elements, class Widen>
auto expand(std::span<const std::uint8_t> bytes, const Widen& widen) {
    return expand<Format, Elements>(bytes.data(), static_cast<unsigned>(bytes.size()), 0, 0, 0, 0, widen);
}

std::vector<std::uint8_t> seeded(std::size_t size) {
    std::vector<std::uint8_t> result(size);
    std::uint32_t state = 0x31415926U;
    for (auto& byte : result) {
        state = state * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(state >> 24);
    }
    return result;
}

template <class T, std::size_t N>
Slice slice(std::span<const T, N> bytes) {
    const auto wire = std::as_bytes(bytes);
    return {memory_source(wire), {0, wire.size()}};
}

constexpr Shape matrix(unsigned rows, unsigned width) { return {{rows, width, 0, 0}, 2}; }

template <F Format>
WeightView tensor(std::span<const std::uint8_t> bytes, unsigned rows, unsigned width) {
    WeightView view{};
    view.storage = decoder_format(Format).id;
    view.shape = matrix(rows, width);
    view.payload = slice(bytes);
    return view;
}

template <F Format>
void compare(std::span<const float> actual, std::span<const float> expected) {
    const auto candidate = std::as_bytes(actual);
    const auto reference = std::as_bytes(expected);
    check(candidate.size() == reference.size(), "decoded extent");
    const auto crc = xport::crc32c_bytes(candidate);
    const auto reference_crc = xport::crc32c_bytes(reference);
    const bool equal_bytes = std::memcmp(candidate.data(), reference.data(), reference.size()) == 0;
    check(crc == reference_crc && equal_bytes, decoder_format(Format).id.name);
    covered[static_cast<unsigned>(Format)] = true;
    std::cout << decoder_format(Format).id.name << " CRC32C=" << std::hex << crc << std::dec << " EXACT\n";
}

template <F Format>
Geometry check_geometry(const WeightView& view) {
    Geometry previous{};
    check(static_cast<bool>(format_geometry(view.storage, view.parameters, previous)), "legacy geometry lookup");
    constexpr auto spec = decoder_format(Format);
    check(previous.block_elements == spec.block_elements &&
          previous.block_bytes == spec.bytes_at_rate(view.parameters.bits), spec.id.name);
    return previous;
}

template <F Format>
void compare_with_reference(const WeightView& view, std::span<const float> actual) {
    check_geometry<Format>(view);
    std::vector<float> expected(actual.size());
    DecodeBuffer output{std::as_writable_bytes(std::span(expected)), 0};
    const auto status = decode_reference(builtin_registry(), view, {0, actual.size(), {}, Scalar::f32}, output);
    check(static_cast<bool>(status), status.detail);
    check(output.written == std::as_bytes(actual).size(), "legacy bytes written");
    compare<Format>(actual, expected);
}

void store_half(std::uint8_t* dest, std::uint16_t value) {
    dest[0] = static_cast<std::uint8_t>(value);
    dest[1] = static_cast<std::uint8_t>(value >> 8);
}

template <F Format>
void fixed_blocks() {
    constexpr auto spec = decoder_format(Format);
    constexpr unsigned group = spec.block_elements == 1 ? spec.group_elements : spec.block_elements;
    constexpr unsigned groups = 3, rows = 2, width = groups * group;
    constexpr unsigned block_bytes = spec.block_bytes * (group / spec.block_elements);
    auto bytes = seeded(rows * groups * block_bytes);
    for (unsigned i = 0; i < rows * width / spec.block_elements; ++i) {
        auto* block = bytes.data() + i * spec.block_bytes;
        const auto half = static_cast<std::uint16_t>(i % 2 == 0 ? 0xbc00U : 0x3a00U);
        if constexpr (Format == F::f32) {
            constexpr std::array<std::uint32_t, 6> bits{0, 0x80000000U, 0x3fa00000U, 1, 0x7f800000U, 0x7fc00013U};
            const auto value = bits[i % bits.size()];
            std::memcpy(block, &value, sizeof(value));
        } else if constexpr (Format == F::f16 || Format == F::bf16) {
            constexpr std::array<std::uint16_t, 7> bits{0, 0x8000, 1, 0x3c00, 0xba00, 0x7e10, 0xff80};
            store_half(block, bits[i % bits.size()]);
        } else if constexpr (Format == F::mxfp4) {
            block[0] = static_cast<std::uint8_t>(125 + i);
        } else if constexpr (Format == F::q2_k) {
            store_half(block + 80, half);
            store_half(block + 82, 0x3400U);
        } else if constexpr (Format == F::q3_k || Format == F::q6_k) {
            store_half(block + spec.block_bytes - 2, half);
        } else {
            store_half(block, half);
            if constexpr (Format == F::q5_1 || Format == F::q4_k || Format == F::q5_k)
                store_half(block + 2, 0x3400U);
        }
    }
    const auto view = tensor<Format>(bytes, rows, width);
    const auto type = gguf_type(view.storage);
    if (const auto g = ggml_dequant::geometry(type))
        check(g->block_bytes == spec.block_bytes && g->block_elements == spec.block_elements, spec.id.name);
    std::vector<float> actual(rows * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned g = 0; g < groups; ++g) {
            const auto values = expand<Format, group>(bytes.data(), block_bytes, groups * block_bytes,
                block_bytes, row, g, [type](std::span<const std::uint8_t> raw, std::span<float, group> out) {
                    if constexpr (Format == F::iq2_xxs) ds41::decode_iq2_xxs(raw.template first<block_bytes>(), out);
                    else if constexpr (Format == F::q2_k) ds41::decode_q2_k(raw.template first<block_bytes>(), out);
                    else if constexpr (Format == F::mxfp4) ds41::decode_mxfp4(raw.template first<block_bytes>(), out);
                    else ggml_dequant::dequantize_row(type, std::as_bytes(raw), out);
                });
            std::ranges::copy(values, actual.begin() + row * width + g * group);
        }
    }
    compare_with_reference<Format>(view, actual);
}

template <F Format>
void floating_blocks() {
    constexpr unsigned rows = Format == F::fp8_block ? 64 : 2, width = 64;
    constexpr unsigned group = decoder_format(Format).group_elements;
    constexpr unsigned group_bytes = Format == F::nvfp4 ? group / 2 : group;
    constexpr unsigned groups = width / group;
    auto bytes = seeded(rows * groups * group_bytes);
    for (auto& byte : bytes) {
        if ((byte & 0x7fU) == 0x7fU && Format != F::nvfp4) byte ^= 1U;
    }
    constexpr std::array<std::uint8_t, 8> fp4_scales{0, 1, 0x38, 0x7e, 2, 0x40, 0x30, 0x60};
    constexpr std::array<std::uint8_t, 4> fp8_scales{126, 127, 130, 128};
    constexpr std::array<float, 1> outer{0.75F};
    std::vector<float> actual(rows * width), expected(rows * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned g = 0; g < groups; ++g) {
            const auto scale = Format == F::nvfp4 ? fp4_scales[row * groups + g] :
                fp8_scales[(Format == F::fp8_block ? row / 32 : row) * groups + g];
            const auto values = expand<Format, group>(bytes.data(), group_bytes, groups * group_bytes,
                group_bytes, row, g, [scale](std::span<const std::uint8_t> raw, std::span<float, group> out) {
                    if constexpr (Format == F::nvfp4) {
                        constexpr float tensor_scale = 0.75F;
                        microscaled::decode_nvfp4_row(raw, std::span(&scale, 1), tensor_scale, 1.0F, out);
                    } else {
                        const float factor = scalar_float::e8m0(scale);
                        for (unsigned i = 0; i < group; ++i) out[i] = scalar_float::e4m3(raw[i]) * factor;
                    }
                });
            std::ranges::copy(values, actual.begin() + row * width + g * group);
        }
        if constexpr (Format == F::mxfp8) {
            microscaled::decode_mxfp8_row(std::span(bytes).subspan(row * width, width),
                std::span(fp8_scales).subspan(row * groups, groups), std::span(expected).subspan(row * width, width));
        }
    }
    auto view = tensor<Format>(bytes, rows, width);
    if constexpr (Format == F::mxfp8) {
        compare<Format>(actual, expected);
    } else {
        if constexpr (Format == F::nvfp4) {
            view.sidecars.push_back({"block_scales", Scalar::e4m3fn, matrix(rows, groups), slice(std::span(fp4_scales))});
            view.sidecars.push_back({"tensor_scale", Scalar::f32, {{1, 0, 0, 0}, 1}, slice(std::span(outer))});
            namespace previous = cuda::core::nvfp4;
            for (unsigned i = 0; i < actual.size(); ++i) {
                const auto code = static_cast<std::uint8_t>((bytes[i / 2] >> (4U * (i % 2))) & 15);
                expected[i] = (previous::e2m1(code) * previous::e4m3(fp4_scales[i / group])) * outer[0];
            }
            check(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0, "old NVFP4 scalar helper");
        } else {
            view.sidecars.push_back({"block_scales", Scalar::e8m0, matrix(rows / 32, groups), slice(std::span(fp8_scales))});
        }
        compare_with_reference<Format>(view, actual);
    }
}

template <F Format>
void trellis(unsigned bits) {
    constexpr unsigned rows = 128, width = 128, tile = Trellis::kTile;
    constexpr unsigned prefix = Format == F::exl3_transport ? 2 * (rows + width) : 0;
    const unsigned tile_bytes = decoder_format(Format).bytes_at_rate(bits);
    auto bytes = seeded(prefix + (rows / tile) * (width / tile) * tile_bytes);
    for (unsigned i = 0; i < prefix; i += 2) store_half(bytes.data() + i, 0x3c00);
    auto view = tensor<Format>(bytes, rows, width);
    view.parameters.bits = bits;
    view.parameters.transform = "hadamard128";
    if constexpr (Format == F::exl3) {
        constexpr std::array<std::uint16_t, width> diagonal{};
        constexpr std::array<std::uint32_t, 1> marker{core::trellis::Mul1Constants::kMultiplier};
        view.sidecars = {
            {"input_scale", Scalar::f16, {{width, 0, 0, 0}, 1}, slice(std::span(diagonal))},
            {"output_scale", Scalar::f16, {{rows, 0, 0, 0}, 1}, slice(std::span(diagonal))},
            {"codebook_marker", Scalar::i32, {{1, 0, 0, 0}, 1}, slice(std::span(marker))},
        };
    }
    std::vector<float> actual(rows * width);
    for (unsigned n = 0; n < rows / tile; ++n) {
        for (unsigned k = 0; k < width / tile; ++k) {
            const auto values = expand<Format, tile * tile>(bytes.data() + prefix, tile_bytes, tile_bytes,
                (rows / tile) * tile_bytes, n, k, [bits](std::span<const std::uint8_t> raw, std::span<float, tile * tile> out) {
                    for (unsigned row = 0; row < tile; ++row) {
                        for (unsigned col = 0; col < tile; ++col) {
                            const auto state = Trellis::state_bytes(raw.data(), static_cast<int>(bits), Trellis::position(col, row));
                            out[row * tile + col] = gemma4::exl3::codebook(state);
                        }
                    }
                });
            for (unsigned row = 0; row < tile; ++row)
                std::copy_n(values.begin() + row * tile, tile, actual.begin() + (n * tile + row) * width + k * tile);
        }
    }
    compare_with_reference<Format>(view, actual);
}

template <F Format>
void channels(bool f32_group_scales = false) {
    constexpr unsigned rows = 2, width = 256, group = 16;
    constexpr bool w4 = Format == F::w4a8_grouped;
    constexpr unsigned groups = width / group;
    constexpr unsigned group_bytes = w4 ? group / 2 : group;
    const auto bytes = seeded(rows * groups * group_bytes);
    constexpr std::array<float, rows> scales{0.5F, -0.375F};
    std::array<std::uint8_t, rows * groups> scale_codes{};
    std::array<float, rows * groups> group_scales{};
    constexpr std::array<float, 16> codebook{-160, -8, -7, -6, -5, -4, -3, -2, 0, 2, 3, 4, 5, 6, 7, 160};
    for (unsigned i = 0; i < scale_codes.size(); ++i) {
        scale_codes[i] = static_cast<std::uint8_t>(0x30 + i % 16);
        group_scales[i] = static_cast<float>(i % 4) * 0.5F;
    }
    auto view = tensor<Format>(bytes, rows, width);
    view.parameters.transform = "convrot256";
    view.parameters.group_size = group;
    view.sidecars.push_back({"channel_scales", Scalar::f32, {{rows, 0, 0, 0}, 1}, slice(std::span(scales))});
    if constexpr (w4) {
        const auto group_slice = f32_group_scales ? slice(std::span<const float>(group_scales)) :
            slice(std::span<const std::uint8_t>(scale_codes));
        view.sidecars.push_back({"group_scales", f32_group_scales ? Scalar::f32 : Scalar::e4m3fn,
                                 matrix(rows, groups), group_slice});
        if (f32_group_scales)
            view.sidecars.push_back({"codebook", Scalar::f32, {{16, 0, 0, 0}, 1}, slice(std::span(codebook))});
    }
    std::vector<float> actual(rows * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned g = 0; g < groups; ++g) {
            const float channel = scales[row];
            const auto index = row * groups + g;
            const float rel = f32_group_scales ? group_scales[index] : scalar_float::e4m3(scale_codes[index]);
            const float* table = f32_group_scales ? codebook.data() : nullptr;
            const auto values = expand<Format, group>(bytes.data(), group_bytes, groups * group_bytes, group_bytes,
                row, g, [channel, rel, table](std::span<const std::uint8_t> raw, std::span<float, group> out) {
                    for (unsigned i = 0; i < group; ++i) {
                        const auto code = w4 ? h3::gemm::w4a8_code(raw[i / 2], i % 2, rel, table) :
                            std::bit_cast<std::int8_t>(raw[i]);
                        out[i] = static_cast<float>(code) * channel;
                    }
                });
            std::ranges::copy(values, actual.begin() + row * width + g * group);
        }
    }
    compare_with_reference<Format>(view, actual);
}

template <F Format>
ImagePlan image(std::size_t bytes) {
    static_assert(decoder_format(Format).uses(P::c));
    const auto source = decoder_format(Format == F::iu4_g64 ? F::bf16 : F::q8_0).id;
    return {source, decoder_format(Format).id, bytes, "fixture-storage-f32-v1", ImageLifetime::load};
}

template <F Format>
void packed(ds41::PackedLayout layout) {
    constexpr unsigned rows = 2, width = 320, group = 32, groups = width / group;
    constexpr unsigned block_bytes = decoder_format(Format).block_bytes;
    constexpr bool q6p32 = Format == F::q6p32;
    ds41::PackedChoice choice{};
    choice.kind = Format == F::u5 ? ds41::PackedKind::u5 :
                  Format == F::u6 ? ds41::PackedKind::u6 : ds41::PackedKind::cb4;
    for (unsigned i = 0; i < choice.table.size(); ++i) choice.table[i] = static_cast<std::int8_t>(2 * static_cast<int>(i) - 16);
    std::vector<std::uint8_t> q8(rows * groups * ds41::q8_0_block_bytes);
    for (unsigned b = 0; b < rows * groups; ++b) {
        auto* dest = q8.data() + b * ds41::q8_0_block_bytes;
        store_half(dest, static_cast<std::uint16_t>(0x3800U + b * 17U));
        for (unsigned i = 0; i < group; ++i) {
            const auto code = choice.kind == ds41::PackedKind::cb4 && !q6p32 ? choice.table[(i + b) % 16] :
                static_cast<std::int8_t>(static_cast<int>((i * 3 + b) % (Format == F::u5 ? 32 : 64)) -
                                         (Format == F::u5 ? 16 : 32));
            dest[2 + i] = std::bit_cast<std::uint8_t>(code);
        }
    }
    std::vector<std::uint8_t> bytes;
    if constexpr (q6p32) {
        bytes.resize(q6p::q6p32_bytes(rows, width));
        check(q6p::to_q6p32(q8.data(), rows, width, bytes.data()), "Q6P32 fixture conversion");
    } else {
        bytes = ds41::pack_rows(q8, width, choice, layout);
    }
    const auto plan = image<Format>(bytes.size());
    auto view = tensor<Format>(bytes, rows, width);
    check(plan.image == view.storage && plan.bytes == view.payload.range.bytes, "image plan extent");
    if constexpr (Format == F::cb4)
        view.sidecars.push_back({"codebook", Scalar::s8, {{16, 0, 0, 0}, 1}, slice(std::span<const std::int8_t>(choice.table))});
    if constexpr (!q6p32)
        view.parameters.transform = layout == ds41::PackedLayout::planar ? "planar" : "interleaved";
    std::vector<float> actual(rows * width), expected(rows * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned g = 0; g < groups; ++g) {
            const auto* packed_row = bytes.data() + row * groups * block_bytes;
            std::array<std::uint8_t, block_bytes> block{};
            if constexpr (q6p32) {
                // Q6P32 uses three planes per chunk of up to eight groups.
                const unsigned first = (g / 8) * 8, count = std::min(8U, groups - first), within = g - first;
                const auto* chunk = packed_row + first * block_bytes;
                std::copy_n(chunk + within * 16, 16, block.begin());
                std::copy_n(chunk + count * 16 + within * 8, 8, block.begin() + 16);
                std::copy_n(chunk + count * 24 + within * 2, 2, block.begin() + 24);
            } else if (layout == ds41::PackedLayout::planar) {
                constexpr unsigned high = block_bytes - 18;
                std::copy_n(packed_row + groups * (16 + high) + g * 2, 2, block.begin());
                std::copy_n(packed_row + g * 16, 16, block.begin() + 2);
                if constexpr (high != 0)
                    std::copy_n(packed_row + groups * 16 + g * high, high, block.begin() + 18);
            } else {
                std::copy_n(packed_row + g * block_bytes, block_bytes, block.begin());
            }
            const auto values = expand<Format, group>(block,
                [&choice](std::span<const std::uint8_t> raw, std::span<float, group> out) {
                    std::array<std::uint8_t, ds41::q8_0_block_bytes> q8_block{};
                    if constexpr (q6p32) q6p::to_q8_0(raw.data(), 1, group, q8_block.data());
                    else ds41::unpack_group(raw.data(), group, 0, choice, ds41::PackedLayout::interleaved, q8_block.data());
                    ds41::decode_q8_0(q8_block, out);
                });
            const auto offset = row * width + g * group;
            std::ranges::copy(values, actual.begin() + offset);
            ds41::decode_q8_0(std::span(q8).subspan((row * groups + g) * ds41::q8_0_block_bytes).first<ds41::q8_0_block_bytes>(),
                std::span(expected).subspan(offset).first<group>());
        }
    }
    check(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0, "original Q8_0 image values");
    compare_with_reference<Format>(view, actual);
}

void iu4_image() {
    constexpr unsigned rows = 16, width = 128, group = 64;
    constexpr unsigned code_bytes = rows * width / 2;
    std::vector<std::uint8_t> bytes(code_bytes + rows * (width / group) * 2);
    std::vector<float> actual(rows * width), expected(rows * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned k = 0; k < width; ++k) {
            const int q = static_cast<int>((k * 3 + row) % 16) - 8;
            const auto offset = core::iu4_weight_byte(static_cast<int>(k), static_cast<int>(row), rows);
            core::iu4_put(bytes[offset], static_cast<int>(k % 2), q);
            expected[row * width + k] = static_cast<float>(q) * (k < group ? 0.5F : 0.25F);
        }
        store_half(bytes.data() + code_bytes + row * 4, 0x3800);
        store_half(bytes.data() + code_bytes + row * 4 + 2, 0x3400);
    }
    const auto plan = image<F::iu4_g64>(bytes.size());
    const auto view = tensor<F::iu4_g64>(bytes, rows, width);
    const auto geometry = check_geometry<F::iu4_g64>(view);
    check(plan.bytes == rows * width / geometry.block_elements * geometry.block_bytes, "IU4 image extent");
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned g = 0; g < width / group; ++g) {
            std::array<std::uint8_t, 34> block{};
            for (unsigned tile = 0; tile < group / 16; ++tile) {
                const unsigned tile_index = g * (group / 16) + tile;
                std::copy_n(bytes.data() + tile_index * rows * 8 + row * 8, 8, block.begin() + tile * 8);
            }
            std::copy_n(bytes.data() + code_bytes + row * 4 + g * 2, 2, block.begin() + 32);
            const auto values = expand<F::iu4_g64, group>(block,
                [](std::span<const std::uint8_t> raw, std::span<float, group> out) {
                    const auto half = static_cast<std::uint16_t>(raw[32] | (static_cast<unsigned>(raw[33]) << 8));
                    const float scale = ggml_dequant::fp16(half);
                    for (unsigned i = 0; i < group; ++i)
                        out[i] = static_cast<float>(core::iu4_unpack(raw[i / 2], static_cast<int>(i % 2))) * scale;
                });
            std::ranges::copy(values, actual.begin() + row * width + g * group);
        }
    }
    // The incumbent descriptor is geometry-only. The existing host packer and
    // the input signed codes supply the reference for the generated image.
    compare<F::iu4_g64>(actual, expected);
}

template <F Format>
void kv_token() {
    constexpr unsigned width = 32, heads = 2, halves = 2;
    constexpr bool fp8 = Format == F::kv_e4m3;
    constexpr kv::Form form = fp8 ? kv::Form::e4m3_token : kv::Form::int8_token;
    constexpr std::array<float, heads * halves> scales{0.25F, 0.5F, 0.375F, 1.5F};
    constexpr auto spec = decoder_format(Format);
    static_assert(spec.block_elements == 1 && spec.block_bytes == sizeof(std::uint8_t));
    static_assert(kv::name(form) == spec.id.name);
    static_assert(kv::bytes_per_token(form, {heads, width}) ==
                  halves * heads * (width * spec.block_bytes + sizeof(float)));
    auto bytes = seeded(halves * heads * width);
    if constexpr (fp8) {
        for (auto& code : bytes)
            if ((code & 0x7fU) == 0x7fU) code ^= 1U;
    }
    std::array<float, halves * heads * width> actual{}, expected{};
    for (unsigned row = 0; row < heads * halves; ++row) {
        const auto scale = scales[row];
        const auto values = expand<Format, width>(bytes.data(), width, width, width, row, 0,
            [scale](std::span<const std::uint8_t> raw, std::span<float, width> out) {
                for (unsigned i = 0; i < width; ++i)
                    out[i] = (fp8 ? scalar_float::e4m3(raw[i]) : static_cast<float>(std::bit_cast<std::int8_t>(raw[i]))) * scale;
            });
        std::ranges::copy(values, actual.begin() + row * width);
        for (unsigned i = 0; i < width; ++i) {
            const auto code = bytes[row * width + i];
            // The cache declaration has no registered host format. Widen using
            // the existing scalar reference and the declared per-head FP32 scale.
            const auto signed_code = static_cast<int>(code) - (code >= 128 ? 256 : 0);
            expected[row * width + i] = (fp8 ? ds41::e4m3_to_float(code) : static_cast<float>(signed_code)) * scale;
        }
    }
    compare<Format>(actual, expected);
}

void catalogue() {
    for (const auto& row : kDecoderFormats) {
        check(!row.id.name.empty() && row.id.revision != 0 && row.block_bytes != 0 && row.block_elements != 0,
              "empty catalogue row");
        check(find_decoder_format(row.id) == &row, "duplicate catalogue ID");
        check(row.uses(P::a), "scalar reference protocol missing");
    }
}

template <F... Formats>
void fixed() {
    (fixed_blocks<Formats>(), ...);
}
} // namespace

int main() {
    try {
        catalogue();
        fixed<F::f32, F::f16, F::bf16, F::q8_0, F::q4_0, F::q5_0, F::q5_1, F::q3_k, F::q4_k, F::q5_k, F::q6_k,
              F::iq2_s, F::iq3_xxs, F::iq3_s, F::iq4_nl, F::iq4_xs, F::q2_0, F::iq2_xxs, F::q2_k, F::mxfp4>();
        floating_blocks<F::fp8_block>();
        floating_blocks<F::nvfp4>();
        floating_blocks<F::mxfp8>();
        for (unsigned bits = 1; bits <= 8; ++bits) {
            trellis<F::exl3>(bits);
            if (decoder_format(F::exl3_transport).bytes_at_rate(bits)) trellis<F::exl3_transport>(bits);
        }
        for (auto layout : {ds41::PackedLayout::interleaved, ds41::PackedLayout::planar}) {
            packed<F::cb4>(layout);
            packed<F::u5>(layout);
            packed<F::u6>(layout);
        }
        packed<F::q6p32>(ds41::PackedLayout::interleaved);
        iu4_image();
        channels<F::int8_channel>();
        channels<F::w4a8_grouped>();
        channels<F::w4a8_grouped>(true);
        kv_token<F::kv_e4m3>();
        kv_token<F::kv_int8>();
        check(std::ranges::all_of(covered, [](bool seen) { return seen; }), "unexercised catalogue row");
        std::cout << "weight_decoder_conformance: " << covered.size() << " formats, 0 mismatches (CRC32C + memcmp)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "weight_decoder_conformance: " << error.what() << '\n';
        return 1;
    }
}
