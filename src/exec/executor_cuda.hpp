#pragma once

#include "strix/exec/program.hpp"
#include "strix/session/step_replay.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::exec {

struct CudaStepOut {
    std::uint32_t id;
    float top1, top2;
    std::uint32_t status;
};
static_assert(sizeof(CudaStepOut) == 16);

struct CudaStepIn {
    std::span<const session::StepReplayRow> rows;
    std::uint32_t feed = 0;
    std::uint32_t step = 0;
    std::uint32_t sampling_policy = 0;
};

struct CudaTicket {
    std::uint64_t serial = 0;
};

struct CudaGraphStats {
    std::uint64_t captures = 0;  // Attempts, including candidates for direct argument updates.
    std::uint64_t instantiations = 0;
    std::uint64_t rebinds = 0;
    std::size_t nodes = 0;
    std::size_t pdl_edges = 0;
};

struct CudaStepProfile {
    std::vector<float> launch_us;  // Program::launches order.
    float total_us;
};

// One device and one serial host owner. Construct on the selected device and
// keep it selected for all calls. Use stable, immutable compiler-owned
// Programs. Captured allocations must outlive every ticket that uses them.
class CudaExecutor {
public:
    // pdl is the device capability AND the binding's PDL contract. Each
    // successor with an 'after' edge must wait in at least one block, and
    // each block must wait before its dependent IO.
    explicit CudaExecutor(bool pdl, bool replay = true);
    ~CudaExecutor();
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    // Bind these stable device addresses into Program::args. Each thunk
    // enqueues exactly one kernel on the supplied stream to be capturable.
    [[nodiscard]] session::StepReplayBlock* descriptor() const noexcept;
    [[nodiscard]] CudaStepOut* output() const noexcept;

    [[nodiscard]] CudaTicket submit(const Program& program, const CudaStepIn& input);
    // The bound pick/accept launch writes the next input in place, advances
    // positions and masks stopped rows. Up to eight steps, returned in
    // step-major order.
    [[nodiscard]] CudaTicket submit_chain(const Program& program, const CudaStepIn& first, std::uint32_t steps);
    // Both consume a completed ticket. There are eight staging slots. The
    // result span must be rows * steps elements.
    [[nodiscard]] bool poll(CudaTicket ticket, std::span<CudaStepOut> out);
    void wait(CudaTicket ticket, std::span<CudaStepOut> out);

    // Call with a new epoch after any change to captured layout bindings.
    // Supply a Program with refreshed args if it captures a moved address.
    void invalidate(std::uint64_t layout_epoch) noexcept;
    [[nodiscard]] CudaGraphStats graph_stats(const ProgramKey& key) const noexcept;
    // Executes a step eagerly and waits for its per-launch timing events.
    [[nodiscard]] CudaStepProfile profile(const Program& program, const CudaStepIn& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::exec
