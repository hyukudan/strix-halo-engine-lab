#pragma once
#include "strix/shared_expert_prefetch.hpp"
#include <memory>
#include <string>
#include <utility>

namespace strix::decode_detail {
// Bind the actual launcher stats owner before execution, but read its live
// enqueue/capture counter only when the caller emits the completed receipt.
// Graph replays do not synthesize host dispatches from a token count.
class SharedPrefetchDispatchReceipt {
public:
    explicit SharedPrefetchDispatchReceipt(
        std::shared_ptr<const shared_expert_prefetch::RuntimeDispatchStats> owner)
        : owner_(std::move(owner)) {}

    [[nodiscard]] std::string completed_counts() const {
        if (!owner_) return {};
        return "  \"prefetch_q8_shared_expert_dispatched_projections\": " +
            std::to_string(owner_->dispatched_projections.load(std::memory_order_relaxed)) + ",\n";
    }

private:
    std::shared_ptr<const shared_expert_prefetch::RuntimeDispatchStats> owner_;
};
} // namespace strix::decode_detail
