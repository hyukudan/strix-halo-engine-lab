#pragma once

// engine::DraftSession adapter for e4b-eagle (alternative drafter behind the DFlash interface).
//
// An autoregressive drafter needs three things the current DraftSession does not carry:
//  (1) the window k before proposing (cost grows with depth; the policy already measures cycle cost per k),
//  (2) the token that follows each committed row (prompt[j+1] in prefill, verdict.emitted in decode),
//  (3) optionally the post-final-norm hidden row (tap set C = {11, 23, final}).
// The staged-rows draft API adds optional virtuals with DFlash-neutral defaults and defines
// STRIX_G4_DRAFT_ROWS_API. Before A1 the hooks below are plain methods (direct callers/tests only) and the legacy
// stage()/append() refuse, because priming without next tokens would violate CONTRACT.md section 3.
#include "eagle.hpp"
#include "gemma4/engine/coordinator.hpp"

#include <functional>
#include <memory>

#if defined(STRIX_G4_DRAFT_ROWS_API)
#define STRIX_G4_EAGLE_HOOK override
#else
#define STRIX_G4_EAGLE_HOOK
#endif

namespace strix::gemma4::eagle {

class Session final : public engine::DraftSession {
public:
    using Release = std::function<void(std::unique_ptr<Drafter>)>;
    explicit Session(std::unique_ptr<Drafter> drafter, Release release = {});
    ~Session() override;
    // Legacy interface: propose() drafts the full depth 7; stage()/append() refuse (no next tokens).
    std::array<std::uint32_t, 7> propose(std::uint32_t anchor, std::uint64_t position) override;
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override;
    void stage(std::span<const float> taps, std::span<const std::uint64_t> positions) override;
    void publish() override;
    void discard() noexcept override;
    // A1 hooks.
    std::array<std::uint32_t, 7> propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) STRIX_G4_EAGLE_HOOK;
    [[nodiscard]] bool wants_hidden() const STRIX_G4_EAGLE_HOOK;
    void stage_rows(std::span<const float> taps, std::span<const float> hidden, std::span<const std::uint32_t> next_tokens,
                    std::span<const std::uint64_t> positions) STRIX_G4_EAGLE_HOOK;
    // Diagnostics for policy/calibration: last proposals with per-depth drafter top-1 probability.
    [[nodiscard]] const Draft& last_draft() const { return last_; }
    [[nodiscard]] Drafter& drafter() { return *drafter_; }
private:
    std::unique_ptr<Drafter> drafter_;
    Release release_;
    Draft last_;
    bool staged_ = false;
};

// Host reference sessions (one HostDrafter per request). Weights/table/embedding outlive the factory.
[[nodiscard]] engine::DraftFactory host_drafts(const Weights& weights, RopeView rope, LinearView target_embedding);

// Device sessions (Linux/HIP build, strix_gemma4_eagle_device). Shares the immutable DeviceModel and reuses up to
// eight released drafters (reset clears only the ring tags/state). Uses the backend's resident target embedding when
// available, else uploads one immutable copy. prepare=true builds the model and one spare outside request timing.
[[nodiscard]] engine::DraftFactory device_drafts(const Weights& weights, RopeView rope, LinearView host_embedding,
                                                 engine::Backend* backend = nullptr, bool prepare = true);

} // namespace strix::gemma4::eagle
