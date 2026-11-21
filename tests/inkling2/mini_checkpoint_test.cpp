#include "inkling2/inkling2_mini_checkpoint.hpp"
#include "inkling2/inkling2_mini_resident_weights.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <iostream>

namespace {

using strix::serve::Json;
namespace wei = strix::weight;
namespace mini = strix::inkling2;

class HeaderSource final : public wei::Source {
public:
    HeaderSource(const std::string& header, std::uint64_t payload_size) : size_(8 + header.size() + payload_size) {
        for (unsigned shift = 0; shift < 64; shift += 8)
            header_.push_back(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 0xFFU));
        header_ += header;
    }
    wei::Status read(wei::ByteRange range, std::span<wei::Byte> out) const override {
        if (range.offset >= header_.size()) {
            ++payload_reads_;
            const auto overlay = payload_.find(range.offset);
            if (overlay == payload_.end() || overlay->second.size() != out.size() || range.bytes != out.size())
                return {wei::Error::malformed, "attempted to read unstaged fixture payload"};
            std::ranges::copy(overlay->second, out.begin());
            return {wei::Error::ok, {}};
        }
        if (range.bytes > header_.size() - range.offset || range.bytes != out.size())
            return {wei::Error::malformed, "attempted to read beyond the fixture header"};
        const auto bytes = std::as_bytes(std::span(header_));
        std::ranges::copy(bytes.subspan(static_cast<std::size_t>(range.offset), out.size()), out.begin());
        return {wei::Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return size_; }
    wei::Digest256 identity() const override { throw std::logic_error("attempted to hash fixture payload"); }
    std::string shard_name() const override { return std::string(mini::kMiniCheckpointShard); }
    std::vector<wei::Byte>& stage(std::uint64_t offset, std::size_t length, wei::Byte value = {}) {
        auto& bytes = payload_[offset];
        bytes.assign(length, value);
        return bytes;
    }
    std::vector<wei::Byte>& stage(const wei::Slice& slice, wei::Byte value = {}) {
        return stage(slice.range.offset, static_cast<std::size_t>(slice.range.bytes), value);
    }
    std::size_t payload_reads() const { return payload_reads_; }
private:
    std::string header_;
    std::uint64_t size_;
    std::map<std::uint64_t, std::vector<wei::Byte>> payload_;
    mutable std::size_t payload_reads_{};
};

struct Fixture {
    Json header = Json::object();
    Json weight_map = Json::object();
    std::uint64_t payload_size{};

    void add(const mini::MiniTextTensorSpec& tensor) {
        const std::uint64_t width = tensor.wire_dtype == "F32" ? 4U : (tensor.wire_dtype == "BF16" ? 2U : 1U);
        auto count = width;
        Json shape = Json::array();
        for (const auto dim : tensor.extents) {
            count *= dim;
            shape.push(Json(dim));
        }
        auto offsets = Json::array();
        offsets.push(Json(payload_size));
        payload_size += count;
        offsets.push(Json(payload_size));
        header.set(tensor.name, Json(Json::Object{
            {"dtype", Json(tensor.wire_dtype)},
            {"shape", shape},
            {"data_offsets", offsets},
        }));
        weight_map.set(tensor.name, Json(mini::kMiniCheckpointShard));
    }
    std::string index_json() const {
        return Json(Json::Object{
            {"metadata", Json(Json::Object{{"total_size", Json(payload_size)}})},
            {"weight_map", weight_map},
        }).dump();
    }
};

bool expects(const wei::Status& status, wei::Error expected, std::string_view detail = {}) {
    if (status.code == expected && status.detail.find(detail) != std::string::npos) return true;
    std::cerr << "Unexpected checkpoint validation result " << static_cast<int>(status.code)
              << " (" << status.detail << ")\n";
    return false;
}

bool check_weight_reads(HeaderSource& source, const wei::TensorIndex& index) {
    mini::MiniTextWeights text;
    if (!expects(text.bind(index), wei::Error::ok) || source.payload_reads() != 0) return false;

    const auto resident = mini::MiniResidentWeightPlan::make(text);
    if (source.payload_reads() != 0 || resident.ranges().size() != 742 || resident.ranges().empty())
        return false;
    std::uint64_t previous_file_end = 0;
    std::size_t previous_device_end = 0;
    for (const auto& range : resident.ranges()) {
        if (range.source.source.get() != &source || range.source.range.offset < previous_file_end ||
            range.destination_offset < previous_device_end || range.bytes != range.source.range.bytes ||
            range.destination_offset % mini::MiniResidentWeightPlan::kAlignment != 0)
            return false;
        previous_file_end = range.source.range.offset + range.bytes;
        previous_device_end = range.destination_offset + range.bytes;
    }
    if (resident.bytes() != previous_device_end) return false;
    try {
        (void)mini::MiniResidentWeights::bind(resident, {});
        return false;
    } catch (const std::invalid_argument&) {
    }

    mini::MiniExpertMatrixView fp4_view;
    constexpr unsigned expert = 73;
    if (!expects(text.expert_matrix(19, expert, mini::MiniExpertProjection::gate_up, fp4_view), wei::Error::ok))
        return false;
    const auto& fp4 = fp4_view.matrix;
    const auto* fp4_tensor = text.find("model.llm.layers.19.mlp.experts.w13_weight");
    const auto* fp4_scales = text.find("model.llm.layers.19.mlp.experts.w13_weight_scale");
    const auto* fp4_amax = text.find("model.llm.layers.19.mlp.experts.w13_weight_amax");
    const auto* fp4_denominator = text.find("model.llm.layers.19.mlp.experts.w13_weight_outer_scale_denominator");
    mini::MiniExpertLayerView fp4_layer;
    if (!expects(text.expert_layer(19, mini::MiniExpertProjection::gate_up, fp4_layer), wei::Error::ok) ||
        fp4_layer.experts != 128 || fp4_layer.row_order != wei::microscaled::RowOrder::interleaved_halves ||
        fp4_layer.codes.range.bytes != 128ULL * 2304ULL * 1152ULL ||
        fp4_layer.block_scales.range.bytes != 128ULL * 2304ULL * 144ULL ||
        fp4_layer.amax.range.bytes != 128U * 4U ||
        fp4_layer.denominator.range.bytes != 4U) return false;
    if (!fp4_tensor || !fp4_scales || !fp4_amax || !fp4_denominator ||
        fp4.format != wei::microscaled::MatrixFormat::nvfp4 || fp4.columns != 2304 || fp4.rows != 2304 ||
        fp4.codes.range.offset != fp4_tensor->payload.range.offset + expert * 2304ULL * 1152ULL ||
        fp4.block_scales.range.offset != fp4_scales->payload.range.offset + expert * 2304ULL * 144ULL ||
        fp4.amax.range.offset != fp4_amax->payload.range.offset + expert * 4U ||
        fp4.denominator.range.offset != fp4_denominator->payload.range.offset) return false;

    auto& fp4_codes = source.stage(fp4.codes, wei::Byte{0x22});
    fp4_codes[1152] = wei::Byte{0x44};  // First up row.
    fp4_codes[2304] = wei::Byte{0xa2};  // Second gate row.
    auto& fp4_scale_codes = source.stage(fp4.block_scales, wei::Byte{0x38});
    fp4_scale_codes[144] = wei::Byte{0x40};  // Up row uses a different block scale.
    fp4_scale_codes[288] = wei::Byte{0x30};  // Second gate row uses a different block scale.
    source.stage(fp4.amax) = {wei::Byte{0}, wei::Byte{0}, wei::Byte{0}, wei::Byte{0x41}};  // 8
    source.stage(fp4.denominator) = {wei::Byte{0}, wei::Byte{0}, wei::Byte{0x80}, wei::Byte{0x40}};  // 4

    mini::MiniExpertMatrixView fp8_view;
    if (!expects(text.expert_matrix(20, expert, mini::MiniExpertProjection::down, fp8_view), wei::Error::ok))
        return false;
    const auto& fp8 = fp8_view.matrix;
    const auto* fp8_scales = text.find("model.llm.layers.20.mlp.experts.w2_weight_scale");
    if (!fp8_scales || fp8.format != wei::microscaled::MatrixFormat::mxfp8 ||
        fp8.columns != 1152 || fp8.rows != 2304 || fp8.amax.source || fp8.denominator.source ||
        fp8.block_scales.range.offset != fp8_scales->payload.range.offset + expert * 2304ULL * 36ULL)
        return false;
    auto& fp8_codes = source.stage(fp8.codes, wei::Byte{0x38});
    fp8_codes[1152] = wei::Byte{0xc0};
    auto& fp8_scale_codes = source.stage(fp8.block_scales, wei::Byte{127});
    fp8_scale_codes[36] = wei::Byte{128};
    if (source.payload_reads() != 0) return false;

    mini::MiniExpertMatrix packed;
    std::vector<float> row(fp4.columns);
    if (!expects(packed.load(fp4_view), wei::Error::ok) || source.payload_reads() != 4 ||
        !expects(packed.decode_row(0, row), wei::Error::ok) || row[0] != 2.0F ||
        !expects(packed.decode_row(1, row), wei::Error::ok) || row[0] != 1.0F || row[1] != -1.0F ||
        !expects(packed.decode_row(1152, row), wei::Error::ok) || row[0] != 8.0F ||
        !expects(packed.decode_row(2304, row), wei::Error::unsupported_shape) ||
        source.payload_reads() != 4) return false;

    row.resize(fp8.columns);
    if (!expects(packed.load(fp8_view), wei::Error::ok) || source.payload_reads() != 6 ||
        !expects(packed.decode_row(0, row), wei::Error::ok) || row[0] != 1.0F ||
        !expects(packed.decode_row(1, row), wei::Error::ok) || row[0] != -4.0F ||
        !expects(text.expert_matrix(24, expert, mini::MiniExpertProjection::down, fp8_view),
                 wei::Error::unsupported_shape) ||
        !expects(text.expert_matrix(19, 128, mini::MiniExpertProjection::gate_up, fp4_view),
                 wei::Error::unsupported_shape))
        return false;

    constexpr std::string_view bf16_name = "model.llm.embed.weight";
    constexpr std::string_view f32_name = "model.llm.layers.0.mlp.gate.bias";
    const auto* bf16 = text.find(bf16_name);
    const auto* f32 = text.find(f32_name);
    if (!bf16 || !f32 || text.find("model.visual.test.weight")) return false;
    source.stage(bf16->payload.range.offset + 2U * 2304U, 4) =
        {wei::Byte{0x80}, wei::Byte{0x3f}, wei::Byte{0x00}, wei::Byte{0xc0}};
    source.stage(f32->payload.range.offset + 4U, 4) =
        {wei::Byte{0}, wei::Byte{0}, wei::Byte{0x60}, wei::Byte{0x40}};
    std::array<float, 2> scalars{};
    return expects(text.read(bf16_name, scalars, 2304), wei::Error::ok) &&
           scalars[0] == 1.0F && scalars[1] == -2.0F &&
           expects(text.read(f32_name, std::span(scalars).first(1), 1), wei::Error::ok) && scalars[0] == 3.5F &&
           expects(text.read(f32_name, scalars, 127), wei::Error::unsupported_shape) &&
           source.payload_reads() == 8;
}

}  // namespace

int main() {
    Fixture fixture;
    for (const auto& tensor : mini::mini_text_tensor_schema()) fixture.add(tensor);
    fixture.add({"model.visual.test.weight", "BF16", {4, 4}});

    wei::TensorIndex indexed;
    const auto source = std::make_shared<HeaderSource>(fixture.header.dump(), fixture.payload_size);
    if (!expects(wei::index_safetensors(source, indexed), wei::Error::ok)) return 1;
    mini::MiniCheckpointSummary summary;
    if (!expects(mini::validate_mini_text_checkpoint(indexed, summary), wei::Error::ok)) return 2;
    if (summary.text_tensors != 822 || summary.vision_tensors != 1 || summary.text_bytes != 18889358496ULL) return 3;
    if (!expects(mini::validate_mini_checkpoint_index(indexed, fixture.index_json()), wei::Error::ok)) return 4;
    if (!check_weight_reads(*source, indexed)) return 12;

    fixture.weight_map.set("model.visual.test.weight", "other.safetensors");
    if (!expects(mini::validate_mini_checkpoint_index(indexed, fixture.index_json()),
                 wei::Error::malformed, "model.visual.test.weight")) return 5;
    fixture.weight_map.set("model.visual.test.weight", Json(mini::kMiniCheckpointShard));
    ++fixture.payload_size;
    if (!expects(mini::validate_mini_checkpoint_index(indexed, fixture.index_json()),
                 wei::Error::malformed, "total_size")) return 6;

    constexpr std::string_view mx_scale = "model.llm.layers.20.mlp.experts.w13_weight_scale";
    auto found = std::ranges::find(indexed.tensors, mx_scale, &wei::RawTensor::name);
    if (found == indexed.tensors.end() || found->wire_dtype != "U8" ||
        found->extents != std::vector<std::uint64_t>{128, 2304, 72}) return 7;
    found->wire_dtype = "F8_E4M3";
    if (!expects(mini::validate_mini_text_checkpoint(indexed, summary), wei::Error::unknown_format, mx_scale))
        return 8;
    found->wire_dtype = "U8";
    found->extents[1] = 72;
    if (!expects(mini::validate_mini_text_checkpoint(indexed, summary), wei::Error::unsupported_shape, mx_scale))
        return 9;
    found->extents[1] = 2304;

    constexpr std::string_view fp4_outer = "model.llm.layers.19.mlp.experts.w2_weight_outer_scale_denominator";
    found = std::ranges::find(indexed.tensors, fp4_outer, &wei::RawTensor::name);
    if (found == indexed.tensors.end() || found->wire_dtype != "F32" ||
        found->extents != std::vector<std::uint64_t>{1}) return 10;
    indexed.tensors.erase(found);
    if (!expects(mini::validate_mini_text_checkpoint(indexed, summary), wei::Error::malformed, fp4_outer))
        return 11;
    return 0;
}
