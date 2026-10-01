#include "strix/qsa_selection_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {

QsaCompressedSelectionPlan
build_qsa_compressed_selection_plan(
    std::uint64_t token_budget,
    std::uint64_t compress_ratio) {
    if (token_budget == 0 ||
        compress_ratio < 2 ||
        token_budget %
            compress_ratio != 0) {
        throw std::invalid_argument(
            "QSA token budget must be non-zero and divisible by compression ratio >= 2");
    }

    QsaCompressedSelectionPlan plan;
    plan.token_budget =
        token_budget;
    plan.compress_ratio =
        compress_ratio;
    plan.block_topk =
        token_budget /
        compress_ratio;
    return plan;
}

std::vector<std::uint64_t>
expand_qsa_block_selection_reference(
    const QsaCompressedSelectionPlan& plan,
    std::span<const std::uint64_t> block_indices,
    std::uint64_t sequence_length) {
    if (plan.token_budget == 0 ||
        plan.compress_ratio < 2 ||
        plan.block_topk == 0 ||
        plan.block_topk *
                plan.compress_ratio !=
            plan.token_budget) {
        throw std::invalid_argument(
            "QSA compressed selection plan is invalid");
    }
    if (block_indices.size() >
        plan.block_topk) {
        throw std::invalid_argument(
            "QSA block selection exceeds block top-k");
    }

    std::vector<std::uint64_t> tokens;
    tokens.reserve(
        static_cast<std::size_t>(
            plan.token_budget));

    for (const auto block :
         block_indices) {
        if (block >
            std::numeric_limits<std::uint64_t>::max() /
                plan.compress_ratio) {
            throw std::overflow_error(
                "QSA compressed block offset overflow");
        }
        const auto begin =
            block * plan.compress_ratio;
        for (std::uint64_t offset = 0;
             offset <
                 plan.compress_ratio;
             ++offset) {
            const auto token =
                begin + offset;
            if (token >=
                sequence_length) {
                break;
            }
            tokens.push_back(token);
        }
    }

    std::sort(
        tokens.begin(),
        tokens.end());
    tokens.erase(
        std::unique(
            tokens.begin(),
            tokens.end()),
        tokens.end());

    if (tokens.size() >
        plan.token_budget) {
        tokens.resize(
            static_cast<std::size_t>(
                plan.token_budget));
    }
    return tokens;
}

}  // namespace strix
