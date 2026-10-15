#pragma once
#include <string>

namespace strix::decode_detail {
// Read the actual completion owner when the completed result is published.
// A prefix snapshot cannot describe tokens that have not executed yet.
template<class CompletionOwner>
[[nodiscard]] std::string deferred_moe_completed_tokens_receipt(const CompletionOwner* owner) {
    return "  \"deferred_moe_status_completed_tokens\": " +
        std::to_string(owner ? owner->completed_tokens() : 0) + ",\n";
}
} // namespace strix::decode_detail
