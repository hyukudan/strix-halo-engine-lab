#include "strix/mtp_attention_contract.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto contract =
        strix::build_mtp_attention_structural_contract(
            model);

    expect(contract.full_attention_layers == 1 &&
               contract.attention_heads == 24 &&
               contract.kv_heads == 2 &&
               contract.head_dim == 256,
           "MTP attention uses the pinned one-layer full-attention geometry");
    expect(contract.qsa_indexer_required &&
               contract.indexer_heads == 4 &&
               contract.indexer_kv_heads == 1 &&
               contract.indexer_head_dim == 128 &&
               contract.indexer_budget == 2048 &&
               contract.indexer_compress_ratio == 4 &&
               contract.compressed_block_topk == 512,
           "MTP attention structural contract includes its own QSA indexer geometry and compressed top-k width");
    expect(contract.indexer_uses_attention_rope &&
               contract.indexer_qk_rmsnorm_required &&
               contract.compressed_keys_use_average_pool,
           "MTP QSA planning pins shared RoPE, Q/K RMSNorm and average-pooled compressed keys");

    expect(contract.kv_row_bytes == 2048 &&
               contract.raw_index_row_bytes == 256,
           "MTP attention state accounts KV and raw index rows per token");
    expect(!contract.sparse_selection_semantics_resolved,
           "MTP attention structure does not overclaim unresolved sparse selection semantics");

    strix::validate_mtp_attention_structural_contract(
        model,
        contract);

    auto overclaimed = contract;
    overclaimed.sparse_selection_semantics_resolved =
        true;
    bool rejected = false;
    try {
        strix::validate_mtp_attention_structural_contract(
            model,
            overclaimed);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "MTP attention validator rejects unproven QSA semantic resolution");

    if (failures == 0) {
        std::cout << "MTP attention structural contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
