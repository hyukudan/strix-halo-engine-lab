#pragma once

#include <cstdint>
#include <string>

namespace strix {

// Captured from the official Qwen config pinned in artifacts/. These values
// describe shapes only; they do not define operator semantics.
struct ModelContract {
    std::uint64_t hidden_layers = 48;
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
    std::uint64_t indexer_kv_heads = 1;
    std::uint64_t indexer_head_dim = 128;
    std::uint64_t indexer_compress_ratio = 4;
    std::uint64_t bytes_bf16 = 2;
    std::uint64_t bytes_fp32 = 4;

    [[nodiscard]] static ModelContract qwen38_flash_next();
    [[nodiscard]] std::string validate() const;
};

}  // namespace strix
