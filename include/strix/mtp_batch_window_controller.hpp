#pragma once
#include "strix/mtp_window_controller.hpp"
#include <array>
#include <map>
#include <span>
#include <vector>

namespace strix {
// Joint scheduling only: all selections still come from each request's target.
// A non-drafting request normally uses R2, commits one row and discards padding.
class MtpBatchWindowController {
public:
    struct Request {
        std::uint32_t slot=0, limit=8;
        std::span<const float> probabilities;
    };
    explicit MtpBatchWindowController(std::size_t slots):history_(slots) {}
    void reset(std::uint32_t slot) {history_.at(slot)={};}
    // Before paying for drafts: rows 0 means no draft; otherwise draft rows-1.
    [[nodiscard]] std::vector<std::uint32_t> plan(std::span<const Request>,std::uint32_t budget);
    // After drafting: choose all owners together, with the draft cost already sunk.
    [[nodiscard]] std::vector<std::uint32_t> choose(std::span<const Request>,std::uint32_t budget) const;
    void observe(std::span<const Request>,std::span<const std::uint32_t> rows,
        std::span<const std::uint32_t> accepted,double draft_ms,double target_ms,std::uint32_t draft_steps);
private:
    struct History {std::array<double,7> seen{},hits{};};
    std::vector<History> history_;
    // Costs belong to a whole topology, never to an individual session owner.
    std::map<std::array<std::uint32_t,3>,MtpWindowCostEstimate> costs_;
    MtpWindowCostEstimate draft_;
    std::array<std::uint64_t,9> plans_{};
    std::vector<std::uint32_t> probe_;
    [[nodiscard]] double expected(const Request&,std::uint32_t rows,bool prior) const;
    [[nodiscard]] double cost(std::span<const std::uint32_t>) const;
    [[nodiscard]] std::vector<std::uint32_t> search(std::span<const Request>,std::uint32_t,bool prior) const;
};
} // namespace strix
