#include "strix/mtp_draft_state_contract.hpp"

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
            "MTP draft-state width overflow");
    }
    return left * right;
}

}  // namespace

MtpDraftStateContract
build_mtp_draft_state_contract(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (model.hc_count <= 1) {
        throw std::invalid_argument(
            "MTP draft-state contract requires hyper-connections");
    }

    MtpDraftStateContract contract;
    contract.hidden_size =
        model.hidden_size;
    contract.hc_count =
        model.hc_count;
    contract.state_width =
        checked_mul(
            model.hidden_size,
            model.hc_count);
    contract.source =
        MtpDraftStateSource::
            post_block_wide_residual_pre_head_mix;
    contract.one_state_row_per_request = true;
    return contract;
}

void validate_mtp_draft_state_contract(
    const ModelContract& model,
    const MtpDraftStateContract& contract) {
    const auto expected =
        build_mtp_draft_state_contract(
            model);
    if (contract.hidden_size !=
            expected.hidden_size ||
        contract.hc_count !=
            expected.hc_count ||
        contract.state_width !=
            expected.state_width ||
        contract.source !=
            expected.source ||
        !contract.one_state_row_per_request) {
        throw std::invalid_argument(
            "MTP draft-state contract is incompatible");
    }
}

}  // namespace strix
