#include "strix/ernie/inventory.hpp"
#include "moe_core/latent_experts_reference.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/weight/formats/fp8_block.hpp"
#include "strix/model_description.hpp"
#include "strix/profiles/ernie.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
    require(expected_count == 140469U, "unexpected known tensor count");
    constexpr auto name = "model.layers.1.block_sparse_moe.experts.511.w2.weight";
    const auto codes = index(main_shard_name(5), name, "I8", "[2048,1024]", 2097152U);
    require(static_cast<bool>(add_shard(codes, inventory)), "expert tensor failed");
    const auto& expert = inventory.tensors.at(name);
    require(expert.module == Module::routed_expert && !expert.drafter && expert.layer == 1 &&
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
void check_manifest_and_draft() {
    Inventory weights;
    constexpr auto name = "model.mtp.2.block_sparse_moe.experts.0.w1.weight";
    const auto expert = index(main_shard_name(127), name, "I8", "[2048,1024]", 2097152U);
    require(static_cast<bool>(add_shard(expert, weights)), "draft expert failed");
    const auto& placed = weights.tensors.at(name);
    require(placed.module == Module::routed_expert && placed.drafter && placed.layer == 2, "draft identity");
    constexpr SafetensorsManifestEntry rows[]{{"matrix", "s1", 8}, {"scale", "s2", 4}};
    constexpr std::string_view manifest = R"({"metadata":{"total_size":12},"weight_map":{"matrix":"s1","scale":"s2"}})";
    require(static_cast<bool>(validate_safetensors_manifest(rows, manifest)), "two-shard manifest");
    require(!validate_safetensors_manifest(std::span(rows).first(1), manifest), "manifest extra entry accepted");
    const std::string missing[] = {"s2"};
    require(static_cast<bool>(validate_safetensors_manifest(std::span(rows).first(1), manifest, missing)),
            "explicit partial manifest");
    require(!validate_safetensors_manifest(rows,
            R"({"metadata":{"total_size":13},"weight_map":{"matrix":"s1","scale":"s2"}})"), "bad size accepted");
    require(!validate_safetensors_manifest(rows,
            R"({"metadata":{"total_size":12},"weight_map":{"matrix":"s2","scale":"s1"}})"), "shard swap accepted");
}

void check_fp8() {
    // 3x5 codes with 2x3 tiles: both trailing tiles are partial.
    using Decoder = BlockScaledE4M3<2, 3, float>;
    std::array<std::uint8_t, 15> codes{};
    codes.fill(0x38); // exact 1.0
    codes.back() = 0xb8;
    constexpr std::array scales{0.5F, 2.0F, 3.0F, 4.0F};
    const Decoder decoder(codes, scales, 3, 5);
    std::array<float, 5> row{};
    decoder.decode_row(1, row);
    require(row == std::array{0.5F, 0.5F, 0.5F, 2.0F, 2.0F}, "first FP8 scale row");
    decoder.decode_row(2, row);
    require(row == std::array{3.0F, 3.0F, 3.0F, 4.0F, -4.0F}, "partial FP8 scale row");
    constexpr std::array<std::uint8_t, 4> exponents{126, 127, 128, 129};
    BlockScaledE4M3<2, 3, std::uint8_t>(codes, exponents, 3, 5).decode_row(2, row);
    require(row == std::array{2.0F, 2.0F, 2.0F, 4.0F, -4.0F}, "E8M0 scale policy");
    try {
        (void)Decoder(codes, std::span(scales).first(3), 3, 5);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("short FP8 scale plane accepted");
}

void check_operator_traits() {
    auto policy = moe::route_policy_named("sigmoid", true, "float_serial", "selection", 4, 2,
                                           true, false, 0.0F, 2.0F);
    constexpr std::array select{0.5F, 0.5F, 0.25F, 0.75F}, combination{0.25F, 0.125F, 0.5F, 0.75F},
                         bias{0.0F, 0.0F, 0.25F, -0.5F}, gains{3.0F, 1.5F, 1.0F, 1.0F};
    std::array<std::uint32_t, 2> ids{};
    std::array<float, 2> out{};
    moe::route_select_split_ref(policy, select, combination, bias, gains, ids, out);
    require(ids == std::array<std::uint32_t, 2>{0, 1} && out == std::array{4.0F, 1.0F},
            "split scores, tie order, normalization, and per-expert gain");

    constexpr std::array hidden{2.0F, -4.0F, 8.0F};
    constexpr std::array gain{2.0F, 4.0F};
    constexpr std::array gates{0.0F, 0.0F, 0.0F};
    std::array<float, 3> row{};
    core::sigmoid_product_reference(hidden, gates, row, core::SigmoidGatePrecision::f32);
    require(row == std::array{1.0F, -2.0F, 4.0F}, "sigmoid gate");
    constexpr std::array arbitrary{1.001F, -0.504F, 1.0F};
    core::sigmoid_product_reference(arbitrary, gates, row, core::SigmoidGatePrecision::bf16);
    require(row[0] == 0.5F && row[2] == 0.5F, "sigmoid gate stores BF16 operands");

    moe_core::latent_experts_reference(hidden, gain, 0.75F,
        [](auto, auto y) { y[0] = -0.5F; y[1] = 0.5F; },
        [](auto x, auto y) { y[0] = 2.0F * x[0]; y[1] = -3.0F * x[1]; },
        [](auto x, auto y) { y[0] = x[0]; y[1] = x[1]; y[2] = x[0] + x[1]; },
        [](auto x, auto y) { std::ranges::copy(x, y.begin()); }, row);
    require(row == std::array{0.0F, -10.0F, 0.0F}, "latent norm, routed reduction, reprojection, and shared branch");
    moe_core::latent_experts_reference(hidden, gain, 2.375F,
        [](auto, auto y) { y[0] = -0.5F; y[1] = 0.5F; },
        [](auto x, auto y) {
            require(x[0] == -0.5F && x[1] == 0.5F, "experts receive the unnormalized latent");
            y[0] = 2.0F * x[0]; y[1] = -3.0F * x[1];
        },
        [](auto x, auto y) { y[0] = x[0]; y[1] = x[1]; y[2] = x[0] + x[1]; },
        [](auto x, auto y) { std::ranges::copy(x, y.begin()); }, row, moe_core::LatentNormSite::routed_output);
    require(row == std::array{1.0F, -7.0F, 4.0F}, "routed latent norm precedes reprojection");
}
}  // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "expected model description path");
    check_description(argv[1]);
    check_inventory();
    check_manifest_and_draft();
    check_fp8();
    check_operator_traits();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
