#include "inkling2/inkling2_mini_checkpoint.hpp"
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
        if (range.offset > header_.size() || range.bytes > header_.size() - range.offset || range.bytes != out.size())
            return {wei::Error::malformed, "attempted to read fixture payload"};
        const auto bytes = std::as_bytes(std::span(header_));
        std::ranges::copy(bytes.subspan(static_cast<std::size_t>(range.offset), out.size()), out.begin());
        return {wei::Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return size_; }
    wei::Digest256 identity() const override { throw std::logic_error("attempted to hash fixture payload"); }
    std::string shard_name() const override { return std::string(mini::kMiniCheckpointShard); }
private:
    std::string header_;
    std::uint64_t size_;
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
