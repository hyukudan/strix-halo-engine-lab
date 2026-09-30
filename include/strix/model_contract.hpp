#pragma once

#include <cstdint>
#include <string>

namespace strix {

// Captured from the official Qwen config pinned in artifacts/. These values
// describe shapes only; they do not define operator semantics.
struct ModelContract {
    std::uint64_t vocab_size = 248320;
    std::uint64_t hidden_size = 2560;
    std::uint64_t max_position_embeddings = 262144;
    std::uint64_t hidden_layers = 48;
    std::uint64_t full_attention_interval = 4;
    std::uint64_t ple_layer_1based = 2;
    std::uint64_t ple_embed_dim = 2560;
    std::uint64_t ple_conv_kernel_size = 4;
    std::uint64_t ngram_size = 3;
    std::uint64_t heads_per_ngram = 8;
    std::uint64_t ngram_vocab_size_base = 20'000'000;
    std::uint64_t ngram_vocab_divisor = 128;
    std::uint64_t ngram_hash_seed = 1234;
    std::uint64_t ple_eos_token_id = 248044;
    std::uint64_t hc_count = 4;
    std::uint64_t hc_lowrank = 320;
    std::uint64_t full_attention_layers = 12;
    std::uint64_t gdn_layers = 36;
    std::uint64_t attention_heads = 24;
    std::uint64_t kv_heads = 2;
    std::uint64_t head_dim = 256;
    std::uint64_t mtp_full_attention_layers = 1;
    std::uint64_t gdn_key_heads = 16;
    std::uint64_t gdn_key_head_dim = 128;
    std::uint64_t gdn_value_heads = 48;
    std::uint64_t gdn_value_head_dim = 128;
    std::uint64_t gdn_conv_kernel = 4;
    std::uint64_t indexer_n_heads = 4;
    std::uint64_t indexer_kv_heads = 1;
    std::uint64_t indexer_head_dim = 128;
    std::uint64_t indexer_budget = 2048;
    std::uint64_t indexer_compress_ratio = 4;
    float partial_rotary_factor = 0.25F;
    float rope_theta = 10'000'000.0F;
    std::uint64_t moe_intermediate_size = 640;
    std::uint64_t shared_expert_intermediate_size = 640;
    std::uint64_t num_experts = 512;
    std::uint64_t num_experts_per_token = 10;
    float rms_norm_epsilon = 1e-6F;
    bool norm_topk_probability = true;
    bool gdn_output_gate_sigmoid = true;
    bool gdn_qk_l2norm = true;
    float gdn_qk_l2_epsilon = 1e-6F;
    bool gdn_conv_silu = true;
    std::uint64_t bytes_bf16 = 2;
    std::uint64_t bytes_fp32 = 4;

    [[nodiscard]] static ModelContract qwen38_flash_next();

    [[nodiscard]] constexpr std::uint64_t rotary_dim() const noexcept {
        return static_cast<std::uint64_t>(
            static_cast<double>(head_dim) *
            partial_rotary_factor);
    }

    [[nodiscard]] constexpr std::uint64_t ple_head_count() const noexcept {
        return (ngram_size - 1ULL) * heads_per_ngram;
    }

    [[nodiscard]] constexpr std::uint64_t ple_head_dim() const noexcept {
        const auto heads = ple_head_count();
        return heads == 0 ? 0 : ple_embed_dim / heads;
    }

    [[nodiscard]] constexpr std::uint64_t ple_row_bytes_bf16() const noexcept {
        return ple_head_dim() * bytes_bf16;
    }

    [[nodiscard]] std::string validate() const;
};

}  // namespace strix
