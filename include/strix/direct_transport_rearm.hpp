#pragma once

#include "strix/transport.hpp"

#include <cstdint>

namespace strix::transport {

class DirectRingRearmQualification {
public:
    explicit DirectRingRearmQualification(
        std::uint32_t required_cycles = 3);

    void begin_cycle(
        std::uint64_t connection_epoch,
        std::uint64_t expected_first_sequence);

    void observe_first_completion(
        const TransferDescriptor& descriptor,
        const Completion& completion);

    void end_cycle();

    void abort_cycle() noexcept;

    [[nodiscard]] bool qualified() const noexcept {
        return completed_cycles_ >=
            required_cycles_;
    }

    [[nodiscard]] std::uint32_t completed_cycles() const noexcept {
        return completed_cycles_;
    }

    [[nodiscard]] bool cycle_active() const noexcept {
        return active_;
    }

private:
    std::uint32_t required_cycles_ = 0;
    std::uint32_t completed_cycles_ = 0;
    std::uint64_t epoch_ = 0;
    std::uint64_t expected_sequence_ = 0;
    bool active_ = false;
    bool first_completion_seen_ = false;
};

void publish_direct_rearm_qualification(
    DirectLinkLifecycle& link,
    const DirectRingRearmQualification& qualification);

}  // namespace strix::transport
