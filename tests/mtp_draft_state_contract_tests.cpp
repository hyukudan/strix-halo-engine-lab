#include "strix/mtp_draft_state_contract.hpp"

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
        strix::build_mtp_draft_state_contract(
            model);

    expect(contract.hidden_size == 2560 &&
               contract.hc_count == 4 &&
               contract.state_width == 10240,
           "MTP draft state preserves the four-stream wide residual");
    expect(contract.source ==
               strix::MtpDraftStateSource::
                   post_block_wide_residual_pre_head_mix,
           "MTP next-step state is captured before final head collapse");
    expect(contract.one_state_row_per_request,
           "MTP draft state publishes one wide state row per request");

    strix::validate_mtp_draft_state_contract(
        model,
        contract);

    auto wrong = contract;
    wrong.state_width =
        model.hidden_size;
    bool rejected = false;
    try {
        strix::validate_mtp_draft_state_contract(
            model,
            wrong);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected,
           "collapsed hidden cannot masquerade as next-step draft state");

    if (failures == 0) {
        std::cout << "MTP draft state contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
