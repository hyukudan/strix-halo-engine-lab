#pragma once

#include "executor_cpu.hpp"
#include "strix/exec/compiler.hpp"

namespace strix::exec {

struct CpuLaunchContext {
    const StepBlock& step;
    std::span<RowOut> picks;

    // Material row addresses use X1's allocation and row stride, including padding.
    [[nodiscard]] std::span<std::byte> arena_row(BufferId buffer, std::uint32_t row) const;

private:
    CpuLaunchContext(const StepBlock& input, std::span<RowOut> output, const Program& program,
                     std::span<std::byte> workspace)
        : step(input), picks(output), program_(program), workspace_(workspace) {}

    const Program& program_;
    std::span<std::byte> workspace_;
    friend class CpuExecutor;
};

using CpuReferenceFn = void (*)(std::span<const std::byte>, const Grid&, const CpuLaunchContext&);

// Use from the X1 resolver after choosing the CPU reference for the entry.
// `binding` supplies the entry identity, optional EXACT identity, and POD arguments.
[[nodiscard]] KernelBinding bind_cpu_reference(KernelBinding binding, CpuReferenceFn reference);

}  // namespace strix::exec
