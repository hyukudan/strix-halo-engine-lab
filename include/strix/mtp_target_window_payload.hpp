#pragma once
#include "strix/serial_session.hpp"
#include "strix/mtp_selection_transaction.hpp"
#include <memory>
#include <optional>
#include <vector>
namespace strix {
class HipMtpTargetWindowCapture;
struct MtpTargetWindowLease {
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    SerialSessionContext base,committed;
    std::vector<std::uint32_t> tokens,predictions;
    std::vector<unsigned char> recorded;
    std::vector<ModelContentIdentity> logit_digests;
    std::vector<std::optional<ModelContentIdentity>> row_digests;
    std::uint64_t epoch=0;
    std::optional<MtpSelectionPrefix> selection;
    bool completed=false,sealed=false;
    std::optional<ModelContentIdentity> terminal_digest;
    [[nodiscard]] std::optional<ModelContentIdentity> terminal_logits_digest(std::uint32_t row) const {
        if(!row_digests.empty()) return row<row_digests.size()?row_digests[row]:std::nullopt;
        return row+1U==tokens.size()?terminal_digest:std::nullopt;
    }
};
} // namespace strix
