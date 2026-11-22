#include "strix/weight/load.hpp"

#include "core/iu4a4_rot.hpp"
#include "core/q8_0_transpose.hpp"
#include "strix/fnv.hpp"
#include "strix/weight/host_q8.hpp"
#include "xport/crc32c_fold.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace strix::weight {
namespace {

constexpr auto f32 = decoder_format(DecoderFormat::f32).id;
constexpr auto iu4 = decoder_format(DecoderFormat::iu4_g64).id;
constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();

struct PreparedImage {
    Shape shape{};
    std::size_t input_bytes{};
    std::vector<std::size_t> folds;
};

Status check_site(const SiteNeed& site) {
    const auto& payload = site.payload;
    if (!payload.source) return {Error::stale_source, site.name + ": missing source"};
    const auto size = payload.source->size();
    if (site.name.empty() || !payload.range.bytes || payload.range.offset > size ||
        payload.range.bytes > size - payload.range.offset)
        return {Error::malformed, site.name + ": invalid tensor range"};
    if (site.shape.rank == 0 || site.shape.rank > site.shape.dims.size())
        return {Error::unsupported_shape, site.name + ": invalid rank"};

    unsigned block_elements, block_bytes;
    if (const auto* decoder = find_decoder_format(site.decoder)) {
        if (!decoder->uses(site.protocol) || static_cast<unsigned>(site.protocol) == 0)
            return {Error::unsupported_arithmetic, site.name + ": decoder protocol unavailable"};
        // Sidecar and variable-rate formats reach this step as resolved
        // slices. Their complete geometry is owned by format admission.
        block_elements = decoder->block_elements;
        block_bytes = decoder->block_bytes;
        if (decoder->rate_mask || decoder->scales == ScaleLayout::tile_e8m0_sidecar)
            block_bytes = 0;
    } else {
        Geometry geometry{};
        if (auto status = format_geometry(site.decoder, {}, geometry); !status) return status;
        block_elements = geometry.block_elements;
        block_bytes = geometry.block_bytes;
    }

    const auto width = site.shape.dims[0];
    if (width == 0 || (block_bytes && (!block_elements || width % block_elements)))
        return {Error::unsupported_shape, site.name + ": invalid decoder width"};
    if (block_bytes && width / block_elements > max_bytes / block_bytes)
        return {Error::memory_budget, site.name + ": row bytes overflow"};
    std::uint64_t bytes = block_bytes ? width / block_elements * block_bytes : 0;
    for (unsigned axis = 1; axis < site.shape.rank; ++axis) {
        const auto extent = site.shape.dims[axis];
        if (extent == 0) return {Error::unsupported_shape, site.name + ": empty axis"};
        if (bytes > max_bytes / extent) return {Error::memory_budget, site.name + ": tensor bytes overflow"};
        bytes *= extent;
    }
    if (block_bytes && bytes != payload.range.bytes)
        return {Error::malformed, site.name + ": tensor bytes differ from decoder geometry"};
    return {Error::ok, {}};
}

Status prepare_image(const LoadPlan& plan, const LoadImage& row, PreparedImage& prepared) {
    if (row.sites.empty() || row.image.arithmetic.empty())
        return {Error::malformed, row.name + ": missing input or arithmetic"};
    const auto& first = plan.sites[row.sites.front()];
    prepared.shape = first.shape;
    const auto slow = static_cast<unsigned>(first.shape.rank) - 1U;

    const bool concatenate = !prepared.folds.empty() &&
        plan.folds[prepared.folds.front()].kind == LoadFoldKind::concatenate;
    if (row.sites.size() != 1 && !concatenate)
        return {Error::malformed, row.name + ": multiple inputs require a concatenation fold"};
    for (const auto i : row.sites) {
        const auto& site = plan.sites[i];
        if (!(site.decoder == row.image.source))
            return {Error::unknown_format, row.name + ": decoder differs from image source"};
        if (concatenate && (site.shape.rank != first.shape.rank ||
            !std::equal(site.shape.dims.begin(), site.shape.dims.begin() + slow, first.shape.dims.begin())))
            return {Error::unsupported_shape, row.name + ": incompatible concatenation axes"};
        if (site.payload.range.bytes > max_bytes - prepared.input_bytes)
            return {Error::memory_budget, row.name + ": input bytes overflow"};
        prepared.input_bytes += static_cast<std::size_t>(site.payload.range.bytes);
    }
    if (concatenate) {
        if (!(first.decoder == f32 || first.decoder == kBf16WeightStorage || first.decoder == kQ8WeightStorage))
            return {Error::unsupported_arithmetic, row.name + ": concatenation requires compact rows"};
        prepared.shape.dims[slow] = 0;
        for (const auto i : row.sites) {
            const auto extent = plan.sites[i].shape.dims[slow];
            if (extent > UINT64_MAX - prepared.shape.dims[slow])
                return {Error::unsupported_shape, row.name + ": concatenation extent overflow"};
            prepared.shape.dims[slow] += extent;
        }
    }
    for (const auto i : prepared.folds) {
        const auto& fold = plan.folds[i];
        if (fold.kind == LoadFoldKind::concatenate) {
            if (i != prepared.folds.front() || row.sites.size() < 2)
                return {Error::malformed, row.name + ": concatenation must come first with multiple inputs"};
        } else if (!(first.decoder == f32) ||
                   (fold.kind != LoadFoldKind::one_plus &&
                    (fold.kind != LoadFoldKind::scale || !std::isfinite(fold.factor)))) {
            return {Error::unsupported_arithmetic, row.name + ": invalid F32 fold"};
        }
    }

    auto& shape = prepared.shape;
    const auto width = shape.dims[0], rows = shape.dims[1];
    std::uint64_t bytes = prepared.input_bytes;
    bool valid = false;
    switch (row.conversion) {
    case LoadConversion::stored:
        valid = row.image.source == row.image.image;
        break;
    case LoadConversion::q8_transpose:
        valid = row.image.source == kQ8WeightStorage && row.image.image == kQ8WeightStorage &&
            (shape.rank == 2 || shape.rank == 3) && rows % core::q8_0::kBlock == 0;
        std::swap(shape.dims[0], shape.dims[1]);
        break;
    case LoadConversion::bf16_to_q8:
        valid = row.sites.size() == 1 && row.image.source == kBf16WeightStorage &&
            row.image.image == kQ8WeightStorage && shape.rank == 2 && width % Q8Storage::elements == 0;
        bytes = bytes / (Q8Storage::elements * sizeof(std::uint16_t)) * Q8Storage::bytes;
        break;
    case LoadConversion::iu4_rotated:
        valid = row.sites.size() == 1 && row.image.source == iu4 && row.image.image == kIu4RotatedImageStorage &&
            shape.rank == 2 && width <= core::iu4a4_rot::kMaxInputs && width % core::iu4a4_rot::kBlock == 0 &&
            rows <= std::numeric_limits<unsigned>::max() && rows % 16U == 0;
        // For each K=128 the original two F16 scales become one F32 step.
        break;
    case LoadConversion::runtime:
        valid = row.image.lifetime != ImageLifetime::load && prepared.folds.empty() &&
            !row.image.image.name.empty();
        bytes = row.image.bytes;
        break;
    }
    if (!valid) return {Error::unsupported_shape, row.name + ": incompatible conversion"};
    if (bytes == 0 || bytes != row.image.bytes)
        return {Error::malformed, row.name + ": image bytes differ from planned geometry"};
    return {Error::ok, {}};
}

void store_f32(Byte* dst, float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned i = 0; i < 4; ++i) {
        dst[i] = static_cast<Byte>(bits & 255U);
        bits >>= 8U;
    }
}

std::uint16_t load_u16(const Byte* src) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(src[0]) |
        (std::to_integer<unsigned>(src[1]) << 8U));
}

Status transcode_bf16(const Slice& source, std::span<Byte> output) {
    constexpr auto values_per_block = Q8Storage::elements;
    constexpr auto block_bytes = values_per_block * sizeof(std::uint16_t);
    std::array<Byte, 256U * block_bytes> scratch;
    auto* dst = reinterpret_cast<std::uint8_t*>(output.data());
    for (std::uint64_t offset = 0; offset < source.range.bytes;) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(scratch.size(), source.range.bytes - offset));
        Slice chunk{};
        if (auto status = subslice(source, {offset, count}, chunk); !status) return status;
        if (auto status = read_slice(chunk, std::span(scratch).first(count)); !status) return status;
        for (std::size_t block = 0; block < count; block += block_bytes) {
            float values[values_per_block];
            float amax = 0.0F;
            for (std::size_t lane = 0; lane < values_per_block; ++lane) {
                const auto bits = load_u16(scratch.data() + block + lane * sizeof(std::uint16_t));
                values[lane] = core::onfly::bf16_bits_to_f32(bits);
                if (!std::isfinite(values[lane])) return {Error::malformed, "nonfinite BF16 weight"};
                amax = std::max(amax, std::fabs(values[lane]));
            }
            const float scale = amax / 127.0F;
            if (scale != 0.0F && !std::isfinite(1.0F / scale))
                return {Error::malformed, "BF16 weight makes the Q8_0 reciprocal nonfinite"};
            // The incumbent encodes codes using the F32 scale even when it
            // rounds to zero in the stored F16 field.
            core::onfly::quantise_q8_0_block(values, dst);
            dst += Q8Storage::bytes;
        }
        offset += count;
    }
    return {Error::ok, {}};
}

Status rotate_iu4(const Shape& shape, std::span<const Byte> input, std::span<Byte> output) {
    namespace rot = core::iu4a4_rot;
    const auto width = static_cast<unsigned>(shape.dims[0]);
    const auto rows = static_cast<unsigned>(shape.dims[1]);
    const auto packed_bytes = std::size_t{rows} * (width / 2U);
    const auto src_scales = input.subspan(packed_bytes);
    std::vector<std::uint16_t> scales(src_scales.size() / 2U);
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = load_u16(src_scales.data() + 2U * i);
        if ((scales[i] & 0x7c00U) == 0x7c00U) return {Error::malformed, "nonfinite IU4 scale"};
    }
    const auto* src = reinterpret_cast<const std::uint8_t*>(input.data());
    auto* dst = reinterpret_cast<std::uint8_t*>(output.data());
    auto* steps = output.data() + packed_bytes;
    // Resident steps are [K/128][N/16][2][8], as in the HIP image.
    for (unsigned block = 0; block < width / rot::kBlock; ++block) {
        for (unsigned tile = 0; tile < rows; tile += 16U) {
            for (unsigned parity = 0; parity < 2U; ++parity) {
                for (unsigned pair = 0; pair < 8U; ++pair) {
                    const auto col = tile + 2U * pair + parity;
                    store_f32(steps, rot::requantise_codes(src, scales.data(), width, rows, col, block, dst));
                    steps += sizeof(float);
                }
            }
        }
    }
    return {Error::ok, {}};
}

Status stage_image(const LoadPlan& plan, const LoadImage& row, const PreparedImage& prepared,
                   ImageSink& sink, LoadReceipt& receipt) {
    const auto& first = plan.sites[row.sites.front()];
    std::vector<Byte> bytes;
    if (row.conversion == LoadConversion::bf16_to_q8) {
        bytes.resize(static_cast<std::size_t>(row.image.bytes));
        if (auto status = transcode_bf16(first.payload, bytes); !status) return status;
    } else {
        bytes.resize(prepared.input_bytes);
        auto remaining = std::span(bytes);
        for (const auto i : row.sites) {
            const auto& payload = plan.sites[i].payload;
            const auto count = static_cast<std::size_t>(payload.range.bytes);
            if (auto status = read_slice(payload, remaining.first(count)); !status) return status;
            remaining = remaining.subspan(count);
        }
    }
    for (const auto i : prepared.folds) {
        const auto& fold = plan.folds[i];
        if (fold.kind != LoadFoldKind::concatenate) {
            for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
                auto* at = bytes.data() + offset;
                const float input = decode_f32_le(at);
                store_f32(at, fold.kind == LoadFoldKind::one_plus ? 1.0F + input : fold.factor * input);
            }
        }
        receipt.folds[i] = {i, xport::crc32c_bytes(bytes)};
    }

    std::vector<std::uint8_t> transposed;
    std::vector<Byte> rotated;
    std::span<const Byte> resident = bytes;
    if (row.conversion == LoadConversion::q8_transpose) {
        for (std::size_t offset = 0; offset < bytes.size(); offset += Q8Storage::bytes) {
            if ((std::to_integer<unsigned>(bytes[offset + 1U]) & 0x7cU) == 0x7cU)
                return {Error::malformed, row.name + ": nonfinite Q8_0 scale"};
        }
        const auto& shape = prepared.shape;
        const auto matrices = shape.rank == 3 ? static_cast<std::size_t>(shape.dims[2]) : 1U;
        transposed = core::q8_0::transpose(reinterpret_cast<const std::uint8_t*>(bytes.data()), matrices,
            static_cast<std::size_t>(shape.dims[0]), static_cast<std::size_t>(shape.dims[1]));
        resident = std::as_bytes(std::span(transposed));
    } else if (row.conversion == LoadConversion::iu4_rotated) {
        rotated.resize(bytes.size());
        if (auto status = rotate_iu4(first.shape, bytes, rotated); !status) return status;
        resident = rotated;
    }

    if (auto status = sink.stage(row, prepared.shape, resident); !status) return status;
    receipt.images.push_back({first.tensor, row.name, row.image.bytes, 0});
    return {Error::ok, {}};
}

void hash_number(std::uint64_t& hash, std::uint64_t n) noexcept {
    for (unsigned i = 0; i < 8; ++i) {
        hash = host_fnv::byte(hash, static_cast<unsigned char>(n & 255U));
        n >>= 8U;
    }
}

void hash_id(std::uint64_t& hash, Id id) noexcept {
    hash = host_fnv::field(hash, id.name);
    hash_number(hash, id.revision);
}

} // namespace

std::uint64_t load_plan_identity(const LoadPlan& plan) noexcept {
    auto hash = host_fnv::text("strix.weight.load-plan.v1");
    hash_number(hash, plan.sites.size());
    for (const auto& site : plan.sites) {
        hash_number(hash, site.tensor);
        hash = host_fnv::field(hash, site.name);
        hash_id(hash, site.decoder);
        hash_number(hash, site.shape.rank);
        for (std::size_t axis = 0; axis < std::min<std::size_t>(site.shape.rank, site.shape.dims.size()); ++axis)
            hash_number(hash, site.shape.dims[axis]);
        hash_number(hash, site.payload.range.offset);
        hash_number(hash, site.payload.range.bytes);
        hash_number(hash, static_cast<unsigned>(site.protocol));
    }
    hash_number(hash, plan.images.size());
    for (const auto& row : plan.images) {
        hash = host_fnv::field(hash, row.name);
        hash_id(hash, row.image.source);
        hash_id(hash, row.image.image);
        hash_number(hash, row.image.bytes);
        hash = host_fnv::field(hash, row.image.arithmetic);
        hash_number(hash, static_cast<unsigned>(row.image.lifetime));
        hash = host_fnv::field(hash, row.image.when);
        hash_number(hash, static_cast<unsigned>(row.conversion));
        hash_number(hash, row.sites.size());
        for (const auto i : row.sites) hash_number(hash, i);
    }
    hash_number(hash, plan.folds.size());
    for (const auto& fold : plan.folds) {
        hash_number(hash, fold.image);
        hash_number(hash, static_cast<unsigned>(fold.kind));
        if (fold.kind == LoadFoldKind::scale) hash_number(hash, std::bit_cast<std::uint32_t>(fold.factor));
    }
    return hash;
}

Status load(const LoadPlan& plan, ImageSink& sink, LoadReceipt& receipt) {
    for (const auto& site : plan.sites) {
        if (auto status = check_site(site); !status) return status;
    }
    std::vector<PreparedImage> prepared(plan.images.size());
    for (std::size_t i = 0; i < plan.folds.size(); ++i) {
        const auto image = plan.folds[i].image;
        if (image >= prepared.size() || plan.images[image].image.lifetime != ImageLifetime::load)
            return {Error::malformed, "fold does not reference a load-lifetime image"};
        prepared[image].folds.push_back(i);
    }
    std::unordered_set<std::string_view> names;
    for (std::size_t i = 0; i < plan.images.size(); ++i) {
        const auto& row = plan.images[i];
        if (row.name.empty() || !names.insert(row.name).second)
            return {Error::malformed, row.name + ": empty or duplicate image name"};
        if (row.image.lifetime != ImageLifetime::load && row.image.lifetime != ImageLifetime::session &&
            row.image.lifetime != ImageLifetime::request)
            return {Error::malformed, row.name + ": invalid image lifetime"};
        for (const auto site : row.sites) {
            if (site >= plan.sites.size()) return {Error::malformed, row.name + ": missing input site"};
        }
        if (auto status = prepare_image(plan, row, prepared[i]); !status) return status;
    }

    LoadReceipt next{load_plan_identity(plan), {}, {}};
    next.images.reserve(plan.images.size());
    next.folds.resize(plan.folds.size());
    struct Transaction {
        ImageSink& sink;
        bool committed = false;
        ~Transaction() { if (!committed) sink.abort(); }
    } transaction{sink};
    for (std::size_t i = 0; i < plan.images.size(); ++i) {
        const auto& row = plan.images[i];
        if (row.image.lifetime == ImageLifetime::load) {
            if (auto status = stage_image(plan, row, prepared[i], sink, next); !status) return status;
        }
    }
    std::vector<std::uint32_t> crcs(next.images.size());
    if (auto status = sink.commit(crcs); !status) return status;
    transaction.committed = true;
    for (std::size_t i = 0; i < crcs.size(); ++i) next.images[i].crc32c = crcs[i];
    receipt = std::move(next);
    return {Error::ok, {}};
}

} // namespace strix::weight
