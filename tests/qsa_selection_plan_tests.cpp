#include "strix/qsa_selection_plan.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

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
    expect(strix::build_qsa_hierarchical_selection_layout(2048).scratch_bytes == 0 &&
        strix::build_qsa_hierarchical_selection_layout(2051).scratch_bytes == 0 &&
        strix::build_qsa_hierarchical_selection_layout(2052).scratch_bytes == 4096 &&
        strix::build_qsa_hierarchical_selection_layout(131072).scratch_bytes == 131072,
        "hierarchical scratch follows actual complete-block buckets including tails");
    for (const auto invalid : {0ULL, 131073ULL, ~0ULL}) {
        bool rejected = false;
        try { (void)strix::build_qsa_hierarchical_selection_layout(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "hierarchical capacity bounds reject zero and overflow without allocating");
    }
    const auto plan =
        strix::build_qsa_compressed_selection_plan(
            2048,
            4);
    expect(plan.block_topk == 512 &&
               plan.final_width == 2051,
           "QSA selection allocates 512 compressed blocks plus partial-group tail capacity");

    std::vector<std::int64_t> blocks(
        static_cast<std::size_t>(
            plan.block_topk),
        -1);
    blocks[0] = 0;
    blocks[1] = 2;

    const auto expanded =
        strix::expand_qsa_block_selection_reference(
            plan,
            blocks,
            10,
            12);

    expect(expanded.size() == 2051,
           "QSA expansion preserves fixed token-budget-plus-tail width");
    expect(
        std::vector<std::int64_t>(
            expanded.begin(),
            expanded.begin() + 9) ==
            std::vector<std::int64_t>({
                0, 1, 2, 3,
                8, 9, 10, 11,
                8}),
        "QSA expansion preserves selected blocks then appends visible partial-group tail");

    const std::vector<double> group{
        1.0, 5.0,
        3.0, 7.0,
        5.0, 9.0,
        7.0, 11.0};
    expect(
        strix::average_pool_qsa_key_group_reference(
            group,
            4,
            2) ==
            std::vector<double>({4.0, 8.0}),
        "QSA compressed key reference averages complete groups in FP64 host oracle");

    bool invalid = false;
    try {
        static_cast<void>(
            strix::build_qsa_compressed_selection_plan(
                2048,
                3));
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid,
           "QSA selection rejects non-divisible token budget");

    if (failures == 0) {
        std::cout << "QSA selection plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
