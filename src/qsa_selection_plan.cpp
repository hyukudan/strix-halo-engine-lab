#include "strix/qsa_selection_plan.hpp"

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
        (token_budget +
         compress_ratio - 1ULL) /
        compress_ratio;
    plan.final_width =
        token_budget +
        compress_ratio - 1ULL;
    return plan;
}

std::vector<std::int64_t>
expand_qsa_block_selection_reference(
    const QsaCompressedSelectionPlan& plan,
    std::span<const std::int64_t> block_indices,
    std::uint64_t query_position,
    std::uint64_t sequence_length) {
    if (plan.token_budget == 0 ||
        plan.compress_ratio < 2 ||
        plan.block_topk == 0 ||
        plan.final_width !=
            plan.token_budget +
                plan.compress_ratio - 1ULL ||
        block_indices.size() !=
            plan.block_topk) {
        throw std::invalid_argument(
            "QSA compressed selection buffers do not match plan");
    }

    std::vector<std::int64_t> result;
    result.reserve(
        static_cast<std::size_t>(
            plan.final_width));

    std::uint64_t expanded_count = 0;
    for (const auto block :
         block_indices) {
        if (expanded_count >=
            plan.token_budget) {
            break;
        }

        if (block < 0) {
            const auto remaining =
                plan.token_budget -
                expanded_count;
            for (std::uint64_t offset = 0;
                 offset <
                     plan.compress_ratio &&
                 offset < remaining;
                 ++offset) {
                result.push_back(-1);
                ++expanded_count;
            }
            continue;
        }

        const auto positive_block =
            static_cast<std::uint64_t>(
                block);
        if (positive_block >
            std::numeric_limits<std::uint64_t>::max() /
                plan.compress_ratio) {
            throw std::overflow_error(
                "QSA compressed block offset overflow");
        }
        const auto begin =
            positive_block *
            plan.compress_ratio;

        for (std::uint64_t offset = 0;
             offset <
                 plan.compress_ratio &&
             expanded_count <
                 plan.token_budget;
             ++offset) {
            const auto token =
                begin + offset;
            result.push_back(
                token < sequence_length
                    ? static_cast<std::int64_t>(
                          token)
                    : -1);
            ++expanded_count;
        }
    }

    while (expanded_count <
           plan.token_budget) {
        result.push_back(-1);
        ++expanded_count;
    }

    if (query_position ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "QSA query position overflow");
    }
    const auto visible_tokens =
        query_position + 1ULL;
    const auto tail_start =
        (visible_tokens /
         plan.compress_ratio) *
        plan.compress_ratio;
    const auto tail_count =
        visible_tokens -
        tail_start;

    for (std::uint64_t offset = 0;
         offset <
             plan.compress_ratio - 1ULL;
         ++offset) {
        const auto token =
            tail_start + offset;
        const bool valid =
            offset < tail_count &&
            token < sequence_length;
        result.push_back(
            valid
                ? static_cast<std::int64_t>(
                      token)
                : -1);
    }

    // Match the reference packing rule: valid entries retain their relative
    // order, padding moves to the end. No deduplication or numeric sort occurs.
    std::vector<std::int64_t> packed;
    packed.reserve(result.size());
    for (const auto token : result) {
        if (token >= 0) {
            packed.push_back(token);
        }
    }
    while (packed.size() <
           static_cast<std::size_t>(
               plan.final_width)) {
        packed.push_back(-1);
    }
    return packed;
}

std::vector<double>
average_pool_qsa_key_group_reference(
    std::span<const double> values,
    std::uint64_t compress_ratio,
    std::uint64_t row_width) {
    if (compress_ratio == 0 ||
        row_width == 0 ||
        values.size() !=
            compress_ratio *
                row_width) {
        throw std::invalid_argument(
            "QSA average-pool group geometry mismatch");
    }

    std::vector<double> output(
        static_cast<std::size_t>(
            row_width),
        0.0);
    for (std::uint64_t row = 0;
         row < compress_ratio;
         ++row) {
        for (std::uint64_t column = 0;
             column < row_width;
             ++column) {
            output[
                static_cast<std::size_t>(
                    column)] +=
                values[
                    static_cast<std::size_t>(
                        row * row_width +
                        column)];
        }
    }
    for (auto& value : output) {
        value /=
            static_cast<double>(
                compress_ratio);
    }
    return output;
}

}  // namespace strix
