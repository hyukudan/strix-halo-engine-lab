#include "strix/mtp_attention_contract.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_mul(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            "MTP attention geometry overflow");
    }
    return left * right;
}

}  // namespace

MtpAttentionStructuralContract
build_mtp_attention_structural_contract(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }

    MtpAttentionStructuralContract contract;
    contract.full_attention_layers =
        model.mtp_full_attention_layers;
    contract.attention_heads =
        model.attention_heads;
    contract.kv_heads =
        model.kv_heads;
    contract.head_dim =
        model.head_dim;
    contract.rotary_dim =
        model.rotary_dim();

    contract.qsa_indexer_required = true;
    contract.indexer_heads =
        model.indexer_n_heads;
    contract.indexer_kv_heads =
        model.indexer_kv_heads;
    contract.indexer_head_dim =
        model.indexer_head_dim;
    contract.indexer_budget =
        model.indexer_budget;
    contract.indexer_compress_ratio =
        model.indexer_compress_ratio;
    if (contract.indexer_budget %
        contract.indexer_compress_ratio != 0) {
        throw std::invalid_argument(
            "MTP QSA budget must divide compression ratio exactly");
    }
    contract.compressed_block_topk =
        contract.indexer_budget /
        contract.indexer_compress_ratio;

    contract.indexer_uses_attention_rope = true;
    contract.indexer_qk_rmsnorm_required = true;
    contract.compressed_keys_use_average_pool = true;

    contract.kv_row_bytes =
        checked_mul(
            checked_mul(
                checked_mul(
                    model.kv_heads,
                    model.head_dim),
                2ULL),
            model.bytes_bf16);
    contract.raw_index_row_bytes =
        checked_mul(
            checked_mul(
                model.indexer_kv_heads,
                model.indexer_head_dim),
            model.bytes_bf16);

    contract.no_tie_selection_semantics_resolved =
        true;
    contract.cutoff_tie_semantics_resolved =
        true;
    contract.sparse_selection_semantics_resolved =
        true;
    return contract;
}

void validate_mtp_attention_structural_contract(
    const ModelContract& model,
    const MtpAttentionStructuralContract& contract) {
    const auto expected =
        build_mtp_attention_structural_contract(
            model);

    if (contract.full_attention_layers !=
            expected.full_attention_layers ||
        contract.attention_heads !=
            expected.attention_heads ||
        contract.kv_heads !=
            expected.kv_heads ||
        contract.head_dim !=
            expected.head_dim ||
        contract.rotary_dim !=
            expected.rotary_dim ||
        !contract.qsa_indexer_required ||
        contract.indexer_heads !=
            expected.indexer_heads ||
        contract.indexer_kv_heads !=
            expected.indexer_kv_heads ||
        contract.indexer_head_dim !=
            expected.indexer_head_dim ||
        contract.indexer_budget !=
            expected.indexer_budget ||
        contract.indexer_compress_ratio !=
            expected.indexer_compress_ratio ||
        contract.compressed_block_topk !=
            expected.compressed_block_topk ||
        !contract.indexer_uses_attention_rope ||
        !contract.indexer_qk_rmsnorm_required ||
        !contract.compressed_keys_use_average_pool ||
        contract.kv_row_bytes !=
            expected.kv_row_bytes ||
        contract.raw_index_row_bytes !=
            expected.raw_index_row_bytes ||
        !contract.no_tie_selection_semantics_resolved ||
        !contract.cutoff_tie_semantics_resolved ||
        !contract.sparse_selection_semantics_resolved) {
        throw std::invalid_argument(
            "MTP attention structural contract is stale, incompatible or overclaims QSA semantics");
    }
}

}  // namespace strix
