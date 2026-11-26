#include "strix/weight/pack_source.hpp"

#include "core/sha256.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/confined_file.hpp"
#include "strix/weight/formats/fp8_block.hpp"
#include "strix/weight/formats/microscaled.hpp"

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <limits>
#include <set>

namespace strix::weight {
namespace {
const Status ok{Error::ok, {}};

void append(core::Sha256& hash, const Digest256& digest) {
    hash.update(digest.bytes.data(), digest.bytes.size());
}

Digest256 matrix_digest(PackMatrixSource::Encoding encoding, const Digest256& codes, const Digest256& scales,
                        std::span<const Byte> amax = {}, std::span<const Byte> denominator = {}) {
    if (encoding == PackMatrixSource::Encoding::bf16) return codes;
    constexpr std::string_view tag = "strix-pack-source/v1";
    core::Sha256 hash;
    hash.update(tag.data(), tag.size());
    constexpr char kinds[] = "-mfnx";
    const char kind = kinds[static_cast<std::size_t>(encoding)];
    hash.update(&kind, 1);
    append(hash, codes);
    append(hash, scales);
    if (!amax.empty()) hash.update(amax.data(), amax.size());
    if (!denominator.empty()) hash.update(denominator.data(), denominator.size());
    return {hash.finish()};
}
}  // namespace

Status PackCheckpoint::open(const std::string& path, PackCheckpoint& out) {
    namespace fs = std::filesystem;
    try {
        PackCheckpoint result;
        fs::path input(path);
        if (fs::is_directory(input)) input /= "model.safetensors.index.json";
        const auto add_shard = [&](std::shared_ptr<const Source> source) -> Status {
            TensorIndex index;
            if (const auto s = index_safetensors(source, index); !s)
                return {s.code, source->shard_name() + ": " + s.detail};
            result.tensors_.insert(result.tensors_.end(), std::make_move_iterator(index.tensors.begin()),
                                   std::make_move_iterator(index.tensors.end()));
            result.shards_.push_back(std::move(source));
            return ok;
        };
        if (input.extension() == ".json") {
            const ConfinedDirectory directory(input.has_parent_path() ? input.parent_path() : fs::path("."));
            result.manifest_ = directory.open(input.filename());
            if (result.manifest_->size() > kSafetensorsManifestMaxBytes)
                return {Error::malformed, "safetensors manifest exceeds 64 MiB"};
            std::string json(static_cast<std::size_t>(result.manifest_->size()), '\0');
            if (const auto s = result.manifest_->read({0, json.size()}, std::as_writable_bytes(std::span(json))); !s)
                return s;
            std::set<std::string> shards;
            {
                const auto manifest = serve::parse_json_unique(json);
                const auto* weights = manifest.find("weight_map");
                if (!weights || weights->as_object().empty()) return {Error::malformed, "empty weight_map"};
                for (const auto& pair : weights->as_object()) {
                    const auto& name = pair.second.as_string();
                    if (!safetensors_shard_name_ok(name))
                        return {Error::malformed, "invalid safetensors shard name: " + name};
                    shards.insert(name);
                }
            }
            for (const auto& shard : shards)
                if (const auto s = add_shard(directory.open(shard)); !s) return s;
            std::vector<std::string> owners;
            owners.reserve(result.tensors_.size());
            for (const auto& tensor : result.tensors_) owners.push_back(tensor.payload.source->shard_name());
            std::vector<SafetensorsManifestEntry> entries;
            entries.reserve(result.tensors_.size());
            for (std::size_t i = 0; i < owners.size(); ++i) {
                const auto& tensor = result.tensors_[i];
                entries.push_back({tensor.name, owners[i], tensor.payload.range.bytes});
            }
            if (const auto s = validate_safetensors_manifest(entries, json); !s) return s;
        } else {
            std::shared_ptr<const Source> source;
            if (const auto s = open_source(input.string(), source, input.filename().string()); !s) return s;
            if (const auto s = add_shard(std::move(source)); !s) return s;
        }
        std::ranges::sort(result.tensors_, {}, &RawTensor::name);
        out = std::move(result);
        return ok;
    } catch (const std::exception& error) {
        return {Error::malformed, error.what()};
    }
}

const RawTensor* PackCheckpoint::find(std::string_view name) const {
    const auto it = std::ranges::lower_bound(tensors_, name, {}, &RawTensor::name);
    return it != tensors_.end() && it->name == name ? &*it : nullptr;
}

Status PackCheckpoint::digest(Digest256& out) const {
    if (shards_.empty()) return {Error::stale_source, "checkpoint is not open"};
    const auto fingerprint = [](const auto& source, auto& value) {
        return digest_slice({source, {0, source->size()}}, value);
    };
    if (!manifest_) return fingerprint(shards_.front(), out);
    constexpr std::string_view tag = "strix-pack-sharded-checkpoint/v1";
    core::Sha256 hash;
    hash.update(tag.data(), tag.size());
    Digest256 digest{};
    if (const auto s = fingerprint(manifest_, digest); !s) return s;
    append(hash, digest);
    // The manifest names every shard. Digest order is lexicographic by name.
    for (const auto& shard : shards_) {
        if (const auto s = fingerprint(shard, digest); !s) return s;
        append(hash, digest);
    }
    out = {hash.finish()};
    return ok;
}

Status PackMatrixSource::bind(const PackCheckpoint& checkpoint, const RawTensor& tensor, PackMatrixSource& out) {
    auto logical_rows = tensor.shape.dims[0], logical_columns = tensor.shape.dims[1];
    const bool mxfp4 = tensor.wire_dtype == "I8";
    const bool fp8 = tensor.wire_dtype == "F8_E4M3" || tensor.wire_dtype == "F8_E4M3FN";
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (tensor.shape.rank != 2 || logical_rows == 0 || logical_columns == 0 || logical_columns > limit / (mxfp4 ? 2U : 1U))
        return {Error::unsupported_shape, tensor.name + ": expected nonempty rank-2 matrix"};
    if (mxfp4) logical_columns *= 2U;
    if (logical_rows > limit / logical_columns) return {Error::unsupported_shape, tensor.name + ": matrix size overflow"};
    PackMatrixSource result{Encoding::bf16, static_cast<std::size_t>(logical_rows), static_cast<std::size_t>(logical_columns),
                            tensor.payload, {}};
    if (mxfp4 || fp8) {
        using Block = BlockScaledE4M3<128, 128, float>;
        const auto* scale = checkpoint.find(tensor.name + "_scale_inv");
        if (!scale || scale->shape.rank != 2 || scale->wire_dtype != (mxfp4 ? "F8_E8M0" : "F32") ||
            scale->shape.dims[0] != (mxfp4 ? logical_rows : Block::scale_rows(result.rows)) ||
            scale->shape.dims[1] != (mxfp4 ? logical_columns / 32U : Block::scale_columns(result.columns)) ||
            (mxfp4 && logical_columns % 32U != 0U))
            return {Error::malformed, tensor.name + ": incompatible or missing _scale_inv tensor"};
        result.encoding = mxfp4 ? Encoding::mxfp4 : Encoding::fp8_block;
        result.scales = scale->payload;
    } else if (tensor.wire_dtype != "BF16") {
        return {Error::unknown_format, tensor.name + ": expected BF16, MXFP4 or block E4M3"};
    }
    out = std::move(result);
    return ok;
}

Status bind_pack_bank(const PackCheckpoint& checkpoint, const RawTensor& tensor, microscaled::RowOrder order,
                      microscaled::ExpertSlabView& out) {
    using namespace microscaled;
    const auto invalid = [&] { return Status{Error::unsupported_shape, tensor.name + ": invalid microscaled expert bank"}; };
    const bool nv = tensor.wire_dtype == "U8";
    if ((!nv && tensor.wire_dtype != "F8_E4M3") || tensor.shape.rank != 3U)
        return {Error::unknown_format, tensor.name + ": expected rank-3 NVFP4 or MXFP8 code plane"};
    const auto [experts, rows, encoded, unused] = tensor.shape.dims;
    (void)unused;
    constexpr auto limit = std::numeric_limits<std::size_t>::max();
    if (experts == 0 || rows == 0 || encoded == 0 || experts > limit || rows > limit || encoded > limit / 2U)
        return invalid();
    const auto columns = encoded * (nv ? 2U : 1U);
    const auto block = nv ? kNvfp4BlockColumns : kMxfp8BlockColumns;
    if (rows > limit / sizeof(float) / columns || columns % block != 0) return invalid();
    const auto* scale = checkpoint.find(tensor.name + "_scale");
    if (!scale || scale->wire_dtype != (nv ? "F8_E4M3" : "U8") || scale->shape.rank != 3U ||
        scale->shape.dims != std::array<std::uint64_t, 4>{experts, rows, columns / block, 0})
        return {Error::malformed, tensor.name + ": incompatible or missing scale plane"};
    ExpertSlabView bank{nv ? MatrixFormat::nvfp4 : MatrixFormat::mxfp8, order, static_cast<std::size_t>(experts),
                        static_cast<std::size_t>(rows), static_cast<std::size_t>(columns),
                        tensor.payload, scale->payload, {}, {}};
    if (nv) {
        const auto* amax = checkpoint.find(tensor.name + "_amax");
        const auto* denominator = checkpoint.find(tensor.name + "_outer_scale_denominator");
        if (!amax || !denominator || amax->wire_dtype != "F32" || denominator->wire_dtype != "F32" ||
            amax->extents != std::vector<std::uint64_t>{experts} || denominator->extents != std::vector<std::uint64_t>{1})
            return {Error::malformed, tensor.name + ": incompatible or missing NVFP4 outer-scale planes"};
        bank.amax = amax->payload;
        bank.denominator = denominator->payload;
    }
    MatrixView check;
    if (const auto status = bank.matrix(0, check); !status) return status;
    out = std::move(bank);
    return ok;
}

Status pack_expert(const microscaled::ExpertSlabView& bank, std::size_t expert, PackMatrixSource& out) {
    microscaled::MatrixView matrix;
    if (const auto status = bank.matrix(expert, matrix); !status) return status;
    const auto encoding = matrix.format == microscaled::MatrixFormat::nvfp4 ?
        PackMatrixSource::Encoding::nvfp4 : PackMatrixSource::Encoding::mxfp8;
    out = {encoding, matrix.rows, matrix.columns, matrix.codes, matrix.block_scales,
           matrix.amax, matrix.denominator, bank.row_order};
    return ok;
}

std::string packed_expert_name(std::string_view bank, std::size_t expert) {
    return std::string(bank) + "#" + std::to_string(expert);
}

Status PackMatrixSource::digest(Digest256& out) const {
    Digest256 raw{}, scale{};
    if (const auto s = digest_slice(codes, raw); !s) return s;
    if (encoding != Encoding::bf16)
        if (const auto s = digest_slice(scales, scale); !s) return s;
    if (encoding == Encoding::nvfp4) {
        std::array<Byte, 4> max{}, divisor{};
        if (const auto s = read_slice(amax, max); !s) return s;
        if (const auto s = read_slice(denominator, divisor); !s) return s;
        out = matrix_digest(encoding, raw, scale, max, divisor);
    } else out = matrix_digest(encoding, raw, scale);
    return ok;
}

Status PackMatrixSource::decode(std::vector<float>& out, Digest256* digest_out) const {
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!rows || !columns || columns > limit || rows > limit / columns)
        return {Error::unsupported_shape, "invalid pack source matrix geometry"};
    const auto count = rows * columns;
    using Block = BlockScaledE4M3<128, 128, float>;
    if ((encoding == Encoding::bf16 && (codes.range.bytes != 2U * count || scales.range.bytes)) ||
        (encoding == Encoding::mxfp4 && (columns % 32U || codes.range.bytes != count / 2U ||
                                         scales.range.bytes != count / 32U)) ||
        (encoding == Encoding::fp8_block && (codes.range.bytes != count ||
                 scales.range.bytes != 4U * Block::scale_rows(rows) * Block::scale_columns(columns))) ||
        (encoding == Encoding::nvfp4 && (columns % microscaled::kNvfp4BlockColumns || codes.range.bytes != count / 2U ||
                                         scales.range.bytes != count / microscaled::kNvfp4BlockColumns)) ||
        (encoding == Encoding::mxfp8 && (columns % microscaled::kMxfp8BlockColumns || codes.range.bytes != count ||
                                         scales.range.bytes != count / microscaled::kMxfp8BlockColumns)))
        return {Error::malformed, "pack source payload disagrees with matrix geometry"};
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(codes.range.bytes));
    std::vector<std::uint8_t> scale(static_cast<std::size_t>(scales.range.bytes));
    if (const auto s = read_slice(codes, std::as_writable_bytes(std::span(raw))); !s) return s;
    if (!scale.empty())
        if (const auto s = read_slice(scales, std::as_writable_bytes(std::span(scale))); !s) return s;
    std::vector<float> matrix(count);
    if (encoding == Encoding::bf16) {
        for (std::size_t i = 0; i < matrix.size(); ++i)
            matrix[i] = scalar_float::u2f((std::uint32_t{raw[2U * i]} << 16U) |
                                         (std::uint32_t{raw[2U * i + 1U]} << 24U));
    } else if (encoding == Encoding::mxfp4) {
        try {
            for (std::size_t row = 0; row < rows; ++row)
                microscaled::decode_mxfp4_row(std::span(raw).subspan(row * (columns / 2U), columns / 2U),
                                             std::span(scale).subspan(row * (columns / 32U), columns / 32U),
                                             std::span(matrix).subspan(row * columns, columns));
        } catch (const std::invalid_argument& e) {
            return {Error::malformed, e.what()};
        }
    } else if (encoding == Encoding::nvfp4 || encoding == Encoding::mxfp8) {
        const bool nv = encoding == Encoding::nvfp4;
        std::array<Byte, 4> max{}, divisor{};
        if (nv) {
            if (const auto status = read_slice(amax, max); !status) return status;
            if (const auto status = read_slice(denominator, divisor); !status) return status;
        }
        const auto code_stride = nv ? columns / 2U : columns;
        const auto scale_stride = columns / (nv ? microscaled::kNvfp4BlockColumns : microscaled::kMxfp8BlockColumns);
        try {
            for (std::size_t r = 0; r < rows; ++r) {
                const auto physical = microscaled::physical_row(row_order, rows, r);
                const auto rc = std::span(raw).subspan(physical * code_stride, code_stride);
                const auto rs = std::span(scale).subspan(physical * scale_stride, scale_stride);
                const auto dst = std::span(matrix).subspan(r * columns, columns);
                if (nv) microscaled::decode_nvfp4_row(rc, rs, decode_f32_le(max.data()), decode_f32_le(divisor.data()), dst);
                else microscaled::decode_mxfp8_row(rc, rs, dst);
            }
        } catch (const std::invalid_argument& error) { return {Error::malformed, error.what()}; }
        if (digest_out)
            *digest_out = matrix_digest(encoding, digest_bytes(std::as_bytes(std::span(raw))),
                                        digest_bytes(std::as_bytes(std::span(scale))),
                                        nv ? std::span<const Byte>(max) : std::span<const Byte>{},
                                        nv ? std::span<const Byte>(divisor) : std::span<const Byte>{});
        out = std::move(matrix);
        return ok;
    } else {
        std::vector<float> scales_f32(scale.size() / 4U);
        for (std::size_t i = 0; i < scales_f32.size(); ++i)
            scales_f32[i] = decode_f32_le(reinterpret_cast<const Byte*>(scale.data()) + i * 4U);
        const Block fp8(raw, scales_f32, rows, columns);
        for (std::size_t row = 0; row < rows; ++row)
            fp8.decode_row(row, std::span(matrix).subspan(row * columns, columns));
    }
    if (digest_out)
        *digest_out = matrix_digest(encoding, digest_bytes(std::as_bytes(std::span(raw))),
                                             digest_bytes(std::as_bytes(std::span(scale))));
    out = std::move(matrix);
    return ok;
}

}  // namespace strix::weight
