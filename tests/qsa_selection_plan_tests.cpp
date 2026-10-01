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
    const auto plan =
        strix::build_qsa_compressed_selection_plan(
            2048,
            4);
    expect(plan.block_topk == 512,
           "QSA compressed selection uses 512 blocks for 2048-token budget at ratio four");

    const std::vector<std::uint64_t> blocks{
        2, 0, 2};
    const auto expanded =
        strix::expand_qsa_block_selection_reference(
            plan,
            blocks,
            10);
    expect(expanded ==
               std::vector<std::uint64_t>({
                   0, 1, 2, 3,
                   8, 9}),
           "QSA block expansion is bounded by sequence length and deduplicates blocks");

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
