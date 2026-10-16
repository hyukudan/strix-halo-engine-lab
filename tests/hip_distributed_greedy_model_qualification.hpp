#pragma once
#include "strix/hip_distributed_greedy.hpp"
#include "hip_distributed_sequence_model_qualification.hpp"

namespace strix::qualification {
struct DistributedGreedyQualificationResult {
    std::uint32_t local_node = 0;
    std::uint64_t consumed_inputs = 0, emitted_tokens = 0;
    bool real_native_generation_completed = false, finish_confirmed = false;
    bool selected_sequence_matches_serial = false, locally_owned_state_equal = false;
    // This case compares selected output and local owned state. Actual full
    // terminal-row equality is the separate streaming-input qualification.
    // No physical link commissioning, bootstrap or memory-sharding claim.
};
// Both real peers call with independent actual full-serial reference owners,
// same immutable prompt/request, and distinct commissioned kind6 channels.
// Caller supplies fresh participants per EOS/budget/capacity/maxnew0 case.
[[nodiscard]] DistributedGreedyQualificationResult qualify_hip_distributed_greedy_request(
    HipDistributedSequenceSetup, const DistributedSequenceReference&,
    DistributedGreedyRequestIdentity, std::span<const std::uint32_t> agreed_prompt,
    std::shared_ptr<DistributedControlByteStream> greedy_control,
    const SerialSessionContext& actual_ordinary_anchor);
} // namespace strix::qualification
