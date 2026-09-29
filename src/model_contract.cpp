#include "strix/model_contract.hpp"

namespace strix {

ModelContract ModelContract::qwen38_flash_next() {
    return {};
}

std::string ModelContract::validate() const {
    if (hidden_layers == 0 || full_attention_layers == 0 || gdn_layers == 0) {
        return "layer counts must be non-zero";
    }
    if (full_attention_layers + gdn_layers != hidden_layers) {
        return "full-attention plus GDN layers must equal hidden layers";
    }
    if (kv_heads == 0 || head_dim == 0 || bytes_bf16 == 0 || bytes_fp32 == 0) {
        return "head and dtype dimensions must be non-zero";
    }
    if (gdn_key_heads == 0 || gdn_key_head_dim == 0 ||
        gdn_value_heads == 0 || gdn_value_head_dim == 0 ||
        gdn_conv_kernel == 0) {
        return "GDN head and convolution dimensions must be non-zero";
    }
    if (gdn_value_heads % gdn_key_heads != 0) {
        return "GDN value heads must be divisible by key heads";
    }
    if (indexer_compress_ratio == 0) {
        return "indexer compression ratio must be non-zero";
    }
    return {};
}

}  // namespace strix
