#include "strix/ernie/checkpoint.hpp"
#include "strix/ernie/engine.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace strix::ernie;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(float actual, float expected, float tolerance, const char* what) {
    if (!(std::fabs(actual - expected) <= tolerance))
        throw std::runtime_error(std::string(what) + ": got " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
}

struct Fixture {
    Inventory inventory;

    void bytes(std::string name, std::string dtype, std::vector<std::uint64_t> shape,
               std::span<const std::byte> data, Module module = Module::hca) {
        auto source = strix::weight::memory_source(data, "fixture");
        expect(inventory.tensors.emplace(std::move(name), Tensor{module, false, {},
            std::move(dtype), std::move(shape), {std::move(source), {0, data.size()}}}).second, "duplicate fixture key");
    }

    void floats(std::string name, std::span<const float> values, Module module = Module::hca) {
        std::vector<std::byte> wire;
        for (auto value : values) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            for (unsigned i = 0; i < 4U; ++i) wire.push_back(static_cast<std::byte>((bits >> (8U * i)) & 255U));
        }
        bytes(std::move(name), "F32", {values.size()}, wire, module);
    }

    void bf16(std::string name, std::span<const std::uint16_t> codes, Module module) {
        std::vector<std::byte> wire;
        for (const auto bits : codes) {
            wire.push_back(static_cast<std::byte>(bits & 255U));
            wire.push_back(static_cast<std::byte>(bits >> 8U));
        }
        bytes(std::move(name), "BF16", {codes.size()}, wire, module);
    }

    void fp8(std::string name, unsigned rows, unsigned columns, std::span<const std::uint8_t> codes) {
        const float scale[] = {1};
        const auto inverse = inverse_scale_name(name);
        bytes(std::move(name), "F8_E4M3", {rows, columns}, std::as_bytes(codes));
        bytes(inverse, "F32", {1, 1}, std::as_bytes(std::span(scale)));
    }

    void mx(std::string name, unsigned rows, unsigned columns, std::span<const std::uint8_t> codes,
            std::span<const std::uint8_t> scales) {
        const auto inverse = inverse_scale_name(name);
        bytes(std::move(name), "I8", {rows, columns / 2U}, std::as_bytes(codes), Module::routed_expert);
        bytes(inverse, "F8_E8M0", {rows, columns / 32U}, std::as_bytes(scales), Module::routed_expert);
    }

    void norms_and_attention(unsigned layer, bool live) {
        const auto base = target_layer_prefix(layer);
        const float ones[] = {1, 1}, sinks[] = {0, 0}, one[] = {1};
        floats(base + "input_layernorm.weight", ones);
        floats(base + "post_attention_layernorm.weight", ones);
        for (const auto* site : {"self_attention_hyper_connection.", "mlp_hyper_connection."}) {
            const auto p = base + site;
            for (const auto* scale : {"alpha_pre", "alpha_post", "alpha_res"}) floats(p + scale, one);
            floats(p + "mapping_proj.weight", std::array<float, 8 * 24>{});
            floats(p + "bias", std::array<float, 24>{});
        }
        const auto p = base + "self_attn.";
        fp8(p + "linear_q_down_proj.weight", 1, 2, std::array<std::uint8_t, 2>{0x38, 0});
        floats(p + "q_layernorm.weight", one);
        fp8(p + "linear_q_up_proj.weight", 4, 1, std::array<std::uint8_t, 4>{});
        fp8(p + "linear_kv_proj.weight", 2, 2, std::array<std::uint8_t, 4>{0, 0, 0x38, 0});
        floats(p + "kv_layernorm.weight", ones);
        floats(p + "core_attention.attn_sink", sinks);
        floats(p + "vha_postmix_U", std::array<float, 2>{0, 1});
        floats(p + "vha_postmix_V", std::array<float, 2>{1, 0});
        fp8(p + "linear_o_group_proj", 2, 2, std::array<std::uint8_t, 4>{0, 0x38, 0, 0x38});
        fp8(p + "gate_proj.weight", 2, 1, std::array<std::uint8_t, 2>{});
        fp8(p + "o_proj.weight", 2, 2,
            std::array<std::uint8_t, 4>{0, 0, 0, static_cast<std::uint8_t>(live ? 0x38 : 0)});
    }
};

ReferenceShape tiny_shape() {
    auto s = checkpoint_reference_shape();
    expect(s.hidden == 4096U && s.route.experts == 512U && s.route.top_k == 10U, "released geometry");
    expect(!layer_has_moe(0) && layer_has_moe(1) && layer_is_mla(7) && layer_is_mla(42), "released layer schedule");
    s.hidden = 2;
    s.vocab = 3;
    s.dense_width = 1;
    s.latent_width = s.expert_width = 32;
    s.heads = s.head_dim = s.rope_dim = s.o_groups = 2;
    s.q_rank = s.o_rank = s.vha_rank = 1;
    s.route.experts = 12; // Keep top-10 and include two experts outside the route.
    return s;
}

FirstRowReference engine(const CheckpointWeights& weights) {
    return FirstRowReference(tiny_shape(),
        [&weights](auto name, auto out, auto offset) { weights.read(name, out, offset); },
        [](float gate, float up, float beta, float linear_beta) {
            expect(beta == 4.0F && linear_beta == 25.0F, "TOML activation parameters");
            return gate * up;
        });
}

void test_decode() {
    Fixture f;
    std::vector<std::uint8_t> codes(129 * 2, 0x38U);
    codes.back() = 0xbcU; // -1.5
    f.bytes("fp8.weight", "F8_E4M3", {129, 2}, std::as_bytes(std::span(codes)));
    const std::array<float, 2> scale{2.0F, 0.5F};
    f.bytes("fp8.weight_scale_inv", "F32", {2, 1}, std::as_bytes(std::span(scale)));
    std::array<std::uint8_t, 32> mx{};
    mx[0] = 0xf2U;
    mx[16] = 0x53U;
    const std::uint8_t exponents[]{128U, 126U};
    f.mx("mx.weight", 1, 64, mx, exponents);
    const CheckpointWeights weights(std::move(f.inventory));
    std::vector<float> fp8(codes.size()), mxfp4(64);
    weights.read("fp8.weight", fp8);
    weights.read("mx.weight", mxfp4);
    expect(fp8[0] == 2 && fp8[255] == 2 && fp8[256] == 0.5F && fp8[257] == -0.75F,
           "128x128 F32-scale FP8 partial tile");
    expect(mxfp4[0] == 2 && mxfp4[1] == -12 && mxfp4[2] == 0 && mxfp4[32] == 0.75F && mxfp4[33] == 1.5F,
           "MXFP4 low-first and independent block scales");
}

void test_first_block() {
    Fixture f;
    f.norms_and_attention(0, true);
    const auto p = target_layer_prefix(0) + "mlp.";
    const std::array<std::uint8_t, 2> second{0, 0x38};
    f.fp8(p + "gate_proj.weight", 1, 2, second);
    f.fp8(p + "up_proj.weight", 1, 2, second);
    f.fp8(p + "down_proj.weight", 2, 1, std::array<std::uint8_t, 2>{0, 0x40}); // 2 * product
    const std::uint16_t embedding[]{0, 0x3f80, 0x3f80, 0, 0x4000, 0x4000};
    const std::uint16_t head[]{0x3f80, 0, 0, 0x3f80, 0x3f80, 0xbf80};
    f.bf16("model.embed_tokens.weight", embedding, Module::embedding);
    f.bf16("lm_head.weight", head, Module::lm_head);
    const float norm[]{1, 1}, scale[]{1};
    f.floats("model.norm.weight", norm);
    f.floats("model.mhc_contract.hc_head_fn", std::array<float, 32>{});
    f.floats("model.mhc_contract.hc_head_base", std::array<float, 4>{});
    f.floats("model.mhc_contract.hc_head_scale", scale);
    const CheckpointWeights weights(std::move(f.inventory));
    const auto ref = engine(weights);
    const float input[]{1, 0, 1, 0, 1, 0, 1, 0};
    const auto block = ref.block(0, input);
    expect(block.experts.empty(), "dense block emitted routed experts");
    near(block.attention_streams[1], std::sqrt(2.0F) / 2.0F, 0.0001F, "sink, VHA and gate");
    near(block.ffn[1], 4.0F / 3.0F, 0.005F, "FP8 dense GLU");
    const auto logits = ref.embedding_first_block_logits(1);
    const auto h = std::sqrt(2.0F) / 2.0F + 4.0F / 3.0F;
    const auto rms = std::sqrt((1.0F + h * h) / 2.0F);
    near(logits.at(0), 1.0F / rms, 0.005F, "raw logit 0");
    near(logits.at(1), h / rms, 0.005F, "raw logit 1");
    near(logits.at(2), (1.0F - h) / rms, 0.005F, "raw logit 2");
}

void test_moe() {
    Fixture f;
    f.norms_and_attention(1, false);
    const auto p = target_layer_prefix(1) + "block_sparse_moe.";
    const auto row = 1.4140625F; // BF16-rounded sqrt(2) from the selected hidden feature.
    f.floats(p + "gate.weight", std::array<float, 24>{});
    std::array<float, 24> mix{};
    mix[22] = std::log(3.0F) / row;
    f.floats(p + "gate.weight_1", mix);
    std::array<float, 12> bias{}, gain{};
    bias[11] = 1.0F;
    gain.fill(1);
    gain[11] = 2;
    f.floats(p + "e_score_correction_bias", bias);
    f.floats(p + "gate.routed_scaling_factor_param", gain);
    std::array<float, 32> norm{};
    norm.fill(1);
    f.floats(p + "latent_norm.weight", norm);
    std::array<std::uint8_t, 64> down{}, up{};
    down[0] = up[0] = 0x38;
    f.fp8(p + "fc1_latent_proj.weight", 32, 2, down);
    f.fp8(p + "fc2_latent_proj.weight", 2, 32, up);
    f.fp8(p + "shared_experts.w1.weight", 32, 2, down);
    f.fp8(p + "shared_experts.w3.weight", 32, 2, down);
    up[0] = 0;
    up[32] = 0x38;
    f.fp8(p + "shared_experts.w2.weight", 2, 32, up);

    std::array<std::uint8_t, 32 * 16> expert{};
    std::array<std::uint8_t, 32> scales{};
    scales.fill(127);
    for (unsigned id : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 11U}) {
        const auto e = p + "experts." + std::to_string(id) + ".";
        expert[0] = 2U;
        f.mx(e + "w1.weight", 32, 32, expert, scales);
        f.mx(e + "w3.weight", 32, 32, expert, scales);
        expert[0] = static_cast<std::uint8_t>(id == 11 ? 4 : 2); // Expert 11 doubles its down projection.
        f.mx(e + "w2.weight", 32, 32, expert, scales);
    }
    const CheckpointWeights weights(std::move(f.inventory));
    const float residual[]{1, 0, 1, 0, 1, 0, 1, 0};
    const auto result = engine(weights).block(1, residual);
    expect(result.experts == std::vector<std::uint32_t>{11, 0, 1, 2, 3, 4, 5, 6, 7, 8}, "top-10 and tie order");
    near(result.expert_weights.at(0), 0.75F / 5.25F * 3.0F * 2.0F, 1e-6F, "post-normalization gain");
    near(result.expert_weights.at(1), 0.5F / 5.25F * 3.0F, 1e-6F, "mix plane ignores selection bias");
    const float latent = 5.65625F; // BF16-rounded sqrt(32).
    const float routed = 3.0F * (4.5F + 3.0F) / 5.25F * latent * latent;
    near(result.ffn[0], routed, 0.001F, "weighted MXFP4 latent sum");
    near(result.ffn[1], row * row, 1e-6F, "ungated shared expert");
    for (std::size_t i = 0; i < result.streams.size(); i += 2) {
        near(result.streams[i], routed + 1.0F, 0.001F, "mHC routed residual");
        near(result.streams[i + 1], row * row, 1e-6F, "mHC shared residual");
    }
}

void test_consumers() {
    using enum TensorConsumer;
    const Tensor norm{Module::layer_norm, false, 0, "BF16", {4096}, {{}, {0, 8192}}};
    auto deferred = norm;
    expect(tensor_consumer("norm", norm) == first_row_reference, "bound norm");
    deferred.layer = 7;
    expect(tensor_consumer("norm", deferred) == mla_layer_pending, "deferred MLA block");
    deferred.drafter = true;
    expect(tensor_consumer("norm", deferred) == draft_pending, "deferred DSpark block");
    deferred.drafter = false;
    deferred.layer = 0;
    deferred.module = Module::compressor;
    expect(tensor_consumer("ape", deferred) == hca_history_pending, "deferred HCA history");
    deferred.layer.reset();
    deferred.module = Module::lm_head;
    expect(tensor_consumer("multimax_ranges", deferred) == multimax_pending &&
           tensor_consumer("lm_head.weight", deferred) == first_row_reference, "raw head and deferred MultiMax");
    deferred.module = Module::unclassified;
    expect(tensor_consumer("unknown", deferred) == unknown, "unknown tensor must be reported");

    Inventory inventory;
    inventory.tensors.emplace("model.layers.0.input_layernorm.weight", norm);
    deferred.module = Module::lm_head;
    inventory.tensors.emplace("lm_head.weight", deferred);
    const auto placement = pair_payload(inventory, 22);
    expect(placement.rank0.bytes == 8192 && placement.rank1.bytes == 8192 && placement.draft.tensors == 0,
           "rank placement preserves output globals");
}
} // namespace

int main() try {
    test_decode();
    test_first_block();
    test_moe();
    test_consumers();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
