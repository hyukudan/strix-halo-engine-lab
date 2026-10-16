#include "strix/mtp_target_window_policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
MtpTargetWindowAssessment assess_mtp_target_window(
    std::span<const std::uint32_t> inputs,
    std::span<const std::uint32_t> predictions,
    std::uint32_t actual_pending,std::uint32_t output_budget,
    std::span<const std::uint32_t> eos,std::uint64_t vocabulary) {
    if ((inputs.size()!=2 && inputs.size()!=4 && inputs.size()!=8) ||
        predictions.size()!=inputs.size() || !vocabulary)
        throw std::invalid_argument("MTP target window requires matching R2/R4/R8 selections and a vocabulary");
    const auto valid=[vocabulary](std::uint32_t token) { return token<vocabulary; };
    if (!valid(actual_pending) || !std::all_of(inputs.begin(),inputs.end(),valid) ||
        !std::all_of(predictions.begin(),predictions.end(),valid) ||
        !std::all_of(eos.begin(),eos.end(),valid) || inputs.front()!=actual_pending)
        throw std::invalid_argument("MTP target window token or pending-input binding is invalid");
    if (output_budget<inputs.size())
        return {MtpTargetWindowDecision::insufficient_output_budget};
    for (std::size_t row=1;row<inputs.size();++row) {
        if (inputs[row]!=predictions[row-1])
            return {MtpTargetWindowDecision::target_input_mismatch,static_cast<std::uint32_t>(row)};
    }
    for (const auto input:inputs) {
        if (std::find(eos.begin(),eos.end(),input)!=eos.end())
            return {MtpTargetWindowDecision::eos_would_be_consumed};
    }
    return {MtpTargetWindowDecision::accept_complete_chunk};
}
} // namespace strix
