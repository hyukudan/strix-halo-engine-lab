#include "strix/ernie/inventory.hpp"
#include "strix/model_description.hpp"
#include "strix/profiles/ernie.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace strix;
using namespace strix::ernie;
using namespace strix::weight;

void require(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
}

class HeaderSource final : public Source {
public:
    HeaderSource(std::string shard, std::string json, std::uint64_t payload_bytes)
        : shard_(std::move(shard)), header_(8U, '\0'), payload_bytes_(payload_bytes) {
        const auto len = static_cast<std::uint64_t>(json.size());
        for (unsigned i = 0; i < 8U; ++i) header_[i] = static_cast<char>((len >> (8U * i)) & 255U);
        header_ += json;
    }

    Status read(ByteRange range, std::span<Byte> out) const override {
        if (range.offset > header_.size() || range.bytes > header_.size() - range.offset ||
            range.bytes != out.size())
            return {Error::malformed, "fixture forbids payload reads"};
        const auto bytes = std::as_bytes(std::span(header_)).subspan(
            static_cast<std::size_t>(range.offset), static_cast<std::size_t>(range.bytes));
        std::ranges::copy(bytes, out.begin());
        return {Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return header_.size() + payload_bytes_; }
    Digest256 identity() const override { return {}; }
    std::string shard_name() const override { return shard_; }

private:
    std::string shard_, header_;
    std::uint64_t payload_bytes_;
};

TensorIndex index(std::string shard, std::string name, std::string dtype,
                  std::string shape, std::uint64_t bytes) {
    auto json = "{\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape +
                ",\"data_offsets\":[0," + std::to_string(bytes) + "]}}";
    TensorIndex out;
    const auto status = index_safetensors(std::make_shared<HeaderSource>(
        std::move(shard), std::move(json), bytes), out);
    if (!status) throw std::runtime_error(status.detail);
    return out;
}

void check_description(const char* file) {
    const auto desc = model_description::load(file);
    auto field = [&](std::string_view name) -> const model_description::Value::Data& {
        const auto* value = desc.find(name);
        if (!value) throw std::runtime_error("missing model description field " + std::string(name));
        return value->data;
    };
    const auto& ratios = std::get<model_description::Value::Array>(field("attention.compress_ratios"));
    require(ratios.size() == kLayerCount, "layer count");
    for (unsigned layer = 0; layer < kLayerCount; ++layer) {
        const bool mla = std::ranges::find(kMlaLayers, std::int64_t{layer}) != std::ranges::end(kMlaLayers);
        require(std::get<std::int64_t>(ratios[layer].data) == (mla ? -2 : 128), "wrong attention schedule");
    }
    require(std::get<std::int64_t>(field("moe.n_routed_experts")) == kExpertCount, "expert count");
    require(std::get<std::int64_t>(field("weights.checkpoint.shards")) == kMainShardCount, "shard count");
    require(std::get<std::string>(field("attention.indexer.sidecar")) == kFlashIndexerShard, "indexer path");
    require(std::get<std::string>(field("presets.default")) == "exact" &&
            !desc.find("presets.fast") && !desc.find("presets.turbo"), "unexpected preset");
    require(kDefaultProfile == profile::parse("exact") && kProfileEngine.table.empty() &&
            kProfileEngine.turbo_version.empty(), "unexpected profile lever");
}

void check_inventory() {
    Inventory inventory;
    const auto expected_count = count_missing_known_tensors(inventory);
    require(expected_count == 131104U, "unexpected known tensor count");
    constexpr auto name = "model.layers.1.block_sparse_moe.experts.511.w2.weight";
    const auto codes = index(main_shard_name(5), name, "I8", "[2048,1024]", 2097152U);
    require(static_cast<bool>(add_shard(codes, inventory)), "expert tensor failed");
    const auto& expert = inventory.tensors.at(name);
    require(expert.module == Module::routed_expert &&
            expert.payload.source->shard_name() == "model-00005-of-00128.safetensors" &&
            expert.payload.range.bytes == 2097152U, "wrong expert role or shard");
    require(!add_shard(codes, inventory), "duplicate name was accepted");
    const auto bad = index(main_shard_name(2), "model.layers.1.block_sparse_moe.experts.511.w2.weight_scale_inv",
                           "I8", "[2048,64]", 131072U);
    require(!add_shard(bad, inventory), "incorrect scale dtype was accepted");
    constexpr auto gate = "model.layers.42.self_attn.core_attention.indexer.linear_gate.weight";
    const auto wrong_location = index(main_shard_name(128), gate, "BF16", "[64,4096]", 524288U);
    require(!add_shard(wrong_location, inventory), "sidecar role in main shard was accepted");
    const auto sidecar = index(std::string(kFlashIndexerShard), gate, "BF16", "[64,4096]", 524288U);
    require(static_cast<bool>(add_shard(sidecar, inventory)), "sidecar tensor failed");
    require(inventory.tensors.at(gate).module == Module::flash_indexer &&
            count_missing_known_tensors(inventory) == expected_count - 2U, "wrong sidecar role or coverage");
    const auto unseen = index(main_shard_name(128), "candidate.weight", "BF16", "[1]", 2U);
    require(static_cast<bool>(add_shard(unseen, inventory)) &&
            inventory.tensors.at("candidate.weight").module == Module::unclassified,
            "unseen tensor was not surfaced");
}
}  // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "expected model description path");
    check_description(argv[1]);
    check_inventory();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
