#pragma once
#include "strix/distributed_greedy_control.hpp"
#include "strix/hip_distributed_sequence.hpp"
#include <functional>
#include <memory>
#include <span>

namespace strix {
struct HipDistributedGreedyResult {
    SerialGreedyRequestResult generation;
    bool cancelled = false, finish_confirmed = false;
};
struct HipDistributedGreedyTerminalResult {
    std::uint64_t consumed_inputs = 0, emitted_tokens = 0;
    bool pending_present = false;
    std::uint32_t pending_token = 0;
    DistributedGreedyFinishReason stop_reason = DistributedGreedyFinishReason::none;
    bool finish_confirmed = false;
};
// A real two-peer streaming consumer, not a native transport/bootstrap factory.
// Caller supplies commissioned live native owners plus an independently bounded,
// protected kind6 stream (distinct from step, token and GPU progress streams).
// Both peers retain exactly the same actual prompt, SHA256 of u32 little-endian.
// All local weights remain resident. Ordinary serial anchors stay quarantined.
// Serialized single request. No reconnect, retry, rollback after decision, or
// independent firstmax verification by the scalar-receiving coordinator.
class HipDistributedGreedyRequest final {
public:
    HipDistributedGreedyRequest(HipDistributedSequenceSetup,
        DistributedGreedyRequestIdentity, std::span<const std::uint32_t> agreed_prompt,
        std::shared_ptr<DistributedControlByteStream> greedy_control);
    ~HipDistributedGreedyRequest();
    HipDistributedGreedyRequest(const HipDistributedGreedyRequest&) = delete;
    HipDistributedGreedyRequest& operator=(const HipDistributedGreedyRequest&) = delete;
    [[nodiscard]] HipDistributedGreedyResult execute_coordinator(
        const std::function<bool()>& cancelled = {});
    [[nodiscard]] HipDistributedGreedyTerminalResult serve_terminal();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
