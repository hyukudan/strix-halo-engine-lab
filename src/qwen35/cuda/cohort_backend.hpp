#pragma once

// qwen35 CUDA binding of the shared multi-request decoder (strix/serve/cohort_decode.hpp).
// Home: shared scheduling (serve::CohortDecoder); this file is only the family adapter onto the engine's slot API.
// CudaEngine::forward_slots computes every row bitwise as the slot alone (cuda_engine.hpp, "Multi-session"), which is
// the backend invariant the decoder needs: each request's ids equal its solo greedy run whatever shares the forward.

#include "strix/qwen35/cuda_engine.hpp"
#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::qwen35 {

class CudaCohortBackend final : public serve::CohortBackend {
public:
    static constexpr std::uint32_t kRowsPerForward = 32;  // forward_slots: total rows over all groups <= 32

    CudaCohortBackend(CudaEngine& engine, std::uint32_t lanes) : e_(engine), lanes_(lanes) {
        if (lanes == 0) throw std::invalid_argument("cohort backend: lanes >= 1");
        e_.configure_slots(lanes);
    }
    [[nodiscard]] std::uint32_t lanes() const override { return lanes_; }
    [[nodiscard]] std::uint32_t rows_per_forward() const override { return kRowsPerForward; }
    [[nodiscard]] std::uint32_t max_positions() const override { return e_.options().max_positions; }

    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) e_.reset_slot(lane);
        for (std::size_t off = 0; off < tokens.size(); off += kRowsPerForward) {
            const SlotRows g{lane, static_cast<std::uint32_t>(std::min<std::size_t>(kRowsPerForward, tokens.size() - off))};
            (void)e_.forward_slots(&g, 1, tokens.data() + off, LogitRows::none, true);
        }
    }

    void verify(std::span<const serve::CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        slots_.resize(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) slots_[i] = {groups[i].lane, groups[i].rows};
        pending_ = e_.forward_slots(slots_.data(), static_cast<std::uint32_t>(slots_.size()), tokens.data(),
                                    LogitRows::all, false);
        e_.read_top(pending_, static_cast<std::uint32_t>(top.size()), top.data());
    }

    [[nodiscard]] bool supports_final_logits() const override { return true; }

    std::span<const float> read_final_logits(std::uint32_t lane) override {
        std::uint32_t row_end = 0;
        for (const auto& slot : slots_) {
            row_end += slot.rows;
            if (slot.slot != lane) continue;
            logits_.resize(e_.config().vocab);
            e_.read_logits(pending_, row_end - 1U, logits_.data());
            return logits_;
        }
        throw std::logic_error("cohort backend: final logits requested for a lane outside the window");
    }

    void commit(std::span<const std::uint32_t> accept) override { e_.commit_slots(accept.data()); }

    void release(std::uint32_t) override {}  // the slot is reset by the next request's first prefill

private:
    CudaEngine& e_;
    std::uint32_t lanes_;
    std::vector<SlotRows> slots_;
    RowsOut pending_{};
    std::vector<float> logits_;
};

}  // namespace strix::qwen35
