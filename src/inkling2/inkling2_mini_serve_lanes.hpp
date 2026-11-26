#pragma once

#include "inkling2_mini_resident_decoder.hpp"
#include "inkling2_mini_resident_dspark.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"

#include <cstdint>
#include <limits>
#include <string>

namespace strix::inkling2 {

// The served lone-lane speculation of a CUDA Inkling 2 Mini cohort (serve::FixedKvLaneBinding): the command line's
// DSpark drafter (the model's MTP sidecar) over the served decoder's slot 0 DSpark taps. Slot 0's taps are its only
// tap plane: a lane on another slot, or slot 0 after one of its rows ran inside a cohort program (no tap store), drafts
// lookup-only (the shared LaneSpeculator asks ready() first). The drafter lives with the server; each new slot-0
// sequence restarts its projected context.
class MiniServeLaneBinding final : public serve::FixedKvLaneBinding {
public:
    MiniServeLaneBinding(MiniResidentDecoder& decoder, unsigned positions, const std::string& sidecar)
        : decoder_(decoder), drafter_(decoder, positions, sidecar) {}

    [[nodiscard]] spec::ChainDrafter* chain_drafter(std::uint32_t lane, std::uint32_t anchor,
                                                    std::uint64_t position) override {
        if (!ready(lane) || drafter_.failed() || position > std::numeric_limits<std::uint32_t>::max()) return nullptr;
        if (decoder_.slot0_sequence() != sequence_) {
            drafter_.restart_sequence();
            sequence_ = decoder_.slot0_sequence();
        }
        drafter_.start(anchor, static_cast<std::uint32_t>(position));
        return &drafter_;
    }
    [[nodiscard]] bool ready(std::uint32_t lane) const override {
        return lane == 0U && decoder_.slot0_taps_complete();
    }

private:
    MiniResidentDecoder& decoder_;
    MiniResidentDSpark drafter_;
    std::uint64_t sequence_ = 0;  // the slot-0 sequence the drafter's context was projected for (0: none yet)
};

} // namespace strix::inkling2
