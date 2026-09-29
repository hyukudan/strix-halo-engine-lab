#include "strix/model_contract.hpp"

namespace strix {

ModelContract ModelContract::qwen38_flash_next() {
    return {};
}

std::string ModelContract::validate() const {
    if (hidden_layers == 0 || full_attention_layers == 0 || gdn_layers == 0) {
        return "layer counts must be non-zero";
    }
    if (full_attention_layers > hidden_layers ||
        gdn_layers != hidden_layers - full_attention_layers) {
        return "full-attention plus GDN layers must equal hidden layers";
    }
    if (full_attention_interval == 0 ||
        hidden_layers % full_attention_interval != 0 ||
        hidden_layers / full_attention_interval != full_attention_layers ||
        ple_layer_1based == 0 ||
        ple_layer_1based > hidden_layers ||
        hc_count <= 1) {
        return "layer pattern, PLE layer and hyper-connection geometry must be valid";
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
    if (hidden_size == 0 ||
        moe_intermediate_size == 0 ||
        shared_expert_intermediate_size == 0 ||
        num_experts == 0 ||
        num_experts_per_token == 0 ||
        num_experts_per_token > num_experts) {
        return "MoE and hidden dimensions must be valid and non-zero";
    }
    return {};
}

}  // namespace strix
