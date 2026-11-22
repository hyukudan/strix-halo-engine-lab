#pragma once

#include "strix/exec/descriptor.hpp"
#include "strix/exec/program.hpp"

#include <memory>

namespace strix::exec {

struct CpuStepIn {
    std::span<const StepRow> rows;
    std::uint32_t feed = 0;
    std::uint32_t step = 0;
    std::uint32_t sampling_policy = 0;
    std::span<const std::uint32_t> stops = {};
};

struct CpuTicket {
    std::uint64_t serial = 0;
};

struct CpuStepProfile {
    std::vector<float> launch_us;
    float total_us = 0;
};

// One serial owner. Workspace and bindings borrow host memory for each call.
// CPU references run synchronously; a completed ticket owns its readback until consumed.
// A reference that throws may already have changed its caller-owned state.
class CpuExecutor {
public:
    explicit CpuExecutor(std::span<std::byte> workspace = {});
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor&) = delete;
    CpuExecutor& operator=(const CpuExecutor&) = delete;

    // Stable addresses with the latest descriptor and final-step readback.
    [[nodiscard]] const StepBlock* descriptor() const noexcept;
    [[nodiscard]] const RowOut* output() const noexcept;
    [[nodiscard]] std::uint32_t commit_mask() const noexcept;

    [[nodiscard]] CpuTicket submit(const Program& program, const CpuStepIn& input);
    // Independent decode only. Returns up to eight steps in step-major order.
    [[nodiscard]] CpuTicket submit_chain(const Program& program, const CpuStepIn& first, std::uint32_t steps);
    [[nodiscard]] bool poll(CpuTicket ticket, std::span<RowOut> out);
    void wait(CpuTicket ticket, std::span<RowOut> out);
    // Layout changes need no cache invalidation. The next call uses its Program's offsets.
    void invalidate(std::uint64_t layout_epoch) noexcept;
    [[nodiscard]] CpuStepProfile profile(const Program& program, const CpuStepIn& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::exec
