#include "session.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::gemma4::eagle {
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("eagle session: ") + message);
}
}

Session::Session(std::unique_ptr<Drafter> drafter, Release release)
    : drafter_(std::move(drafter)), release_(std::move(release)) {
    require(bool(drafter_), "missing drafter");
}
Session::~Session() {
    if (release_ && drafter_) release_(std::move(drafter_));
}
std::array<std::uint32_t, 7> Session::propose(std::uint32_t anchor, std::uint64_t position) {
    return propose_window(anchor, position, max_depth);
}
std::array<std::uint32_t, 7> Session::propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) {
    std::array<std::uint32_t, 7> out{};
    last_ = {};
    k = std::min(k, max_depth);
    require(drafter_->frontier() == position || drafter_->frontier() == 0, "target/draft frontier mismatch");
    // No committed row yet (one-token prompt): any proposal is safe under exact verification; propose nothing useful.
    if (!k || drafter_->frontier() == 0) return out;
    last_ = drafter_->draft(anchor, position, k);
    for (std::uint32_t i = 0; i < k; ++i) out[i] = last_.tokens[i];
    return out;
}
bool Session::wants_hidden() const { return drafter_->config().needs_final(); }
void Session::stage_rows(std::span<const float> taps, std::span<const float> hidden, std::span<const std::uint32_t> next_tokens,
                         std::span<const std::uint64_t> positions) {
    require(!staged_, "previous stage not published/discarded");
    if (positions.empty()) return;
    // The coordinator stages at most eight rows per step (row cap), matching the drafter's stage limit.
    require(positions.size() <= max_rows, "at most eight committed rows per stage");
    const auto H = drafter_->config().target_hidden;
    require(!wants_hidden() || hidden.size() >= positions.size() * H, "final hidden rows required (RowPlan::export_hidden)");
    drafter_->stage({taps, wants_hidden() ? hidden.first(positions.size() * H) : std::span<const float>{}, next_tokens, positions});
    staged_ = true;
}
void Session::append(std::span<const float>, std::span<const std::uint64_t>) {
    throw std::runtime_error("eagle session: append needs next tokens; use stage_rows (staged-rows draft API)");
}
void Session::stage(std::span<const float>, std::span<const std::uint64_t> positions) {
    if (positions.empty()) return;
    throw std::runtime_error("eagle session: legacy stage has no next tokens; requires the staged-rows draft API (stage_rows)");
}
void Session::publish() {
    if (!staged_) return;
    drafter_->publish();
    staged_ = false;
}
void Session::discard() noexcept {
    if (staged_) drafter_->discard();
    staged_ = false;
}

engine::DraftFactory host_drafts(const Weights& weights, RopeView rope, LinearView target_embedding) {
    weights.validate();
    rope.validate();
    return [&weights, rope, target_embedding] {
        return std::make_unique<Session>(std::make_unique<HostDrafter>(weights, rope, target_embedding));
    };
}

} // namespace strix::gemma4::eagle
