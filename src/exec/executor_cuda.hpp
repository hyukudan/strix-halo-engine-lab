#pragma once

#include "strix/exec/descriptor.hpp"
#include "strix/exec/program.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
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

enum class CudaDescriptorLayout { legacy, x3, resident };
enum class CudaNodeKind { kernel, copy_d2d, copy_d2h_pinned, memset };
enum class CudaEdgeKind { completion, programmatic, launch_completion };
enum class CudaReadback { executor, bound, resident };

struct CudaDependency {
    std::size_t predecessor;
    CudaEdgeKind kind = CudaEdgeKind::completion;
    bool operator==(const CudaDependency&) const = default;
};

struct CudaNodeBinding {
    CudaNodeKind kind = CudaNodeKind::kernel;
    std::vector<CudaDependency> after;
    bool operator==(const CudaNodeBinding&) const = default;
};

// Non-kernel commands have no thunk. Bind these POD payloads in Program::args.
struct CudaCopyArgs {
    void* destination;
    const void* source;
    std::size_t bytes;
};

struct CudaMemsetArgs {
    void* destination;
    std::size_t bytes;
    int value;
};

struct CudaProgramBinding {
    // Empty uses the legacy serial kernel plan with scalar BoundLaunch::after.
    // Otherwise one entry per launch; an empty after list makes that node a root.
    std::vector<CudaNodeBinding> nodes;
    // Bound mode uses the graph's pinned D2H destination and an empty ticket
    // result span. It is single-step; consume a reply before reusing its address.
    // Resident mode is kernel-only; the stream owner handles input and output.
    CudaReadback readback = CudaReadback::executor;
    bool operator==(const CudaProgramBinding&) const = default;
};

struct CudaSubmitOptions {
    const CudaProgramBinding* binding = nullptr;
    // Optional non-default CUDA stream owned by the resident session. Submit
    // waits for its prior work and queues a completion dependency back onto it.
    // The after-submit hook may then enqueue logits/flags on the resident stream.
    // Finish that stream before host validation, or wait the ticket before
    // reading the pinned reply. Retire the ticket with wait/poll in either case.
    void* resident_stream = nullptr;
};

struct CudaGraphStats {
    std::uint64_t captures = 0;  // Attempts, including candidates for direct argument updates.
    std::uint64_t instantiations = 0;
    std::uint64_t rebinds = 0;
    std::size_t nodes = 0;
    std::size_t pdl_edges = 0;
    std::size_t kernels = 0;
    std::size_t copies = 0;
    std::size_t memsets = 0;
    std::size_t edges = 0;
};

struct CudaStepProfile {
    std::vector<float> launch_us;  // Program::launches order.
    float total_us;
};

// One device and one serial host owner. Construct on the selected device and
// keep it selected for all calls. Use stable, immutable compiler-owned
// Programs and their immutable bindings. Captured allocations must outlive
// every ticket that uses them.
class CudaExecutor {
public:
    // pdl is the device capability AND the binding's PDL contract. Each
    // successor with an 'after' edge must wait in at least one block, and
    // each block must wait before its dependent IO.
    explicit CudaExecutor(bool pdl, bool replay = true,
                          CudaDescriptorLayout layout = CudaDescriptorLayout::legacy);
    ~CudaExecutor();
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    // Bind the selected descriptor's stable device address into Program::args.
    // The other accessor returns null. All three return null in resident mode.
    // Each kernel thunk enqueues one kernel.
    [[nodiscard]] session::StepReplayBlock* descriptor() const noexcept;
    [[nodiscard]] StepBlock* step_descriptor() const noexcept;
    [[nodiscard]] CudaStepOut* output() const noexcept;

    [[nodiscard]] CudaTicket submit(const Program& program, const CudaStepIn& input, CudaSubmitOptions options = {});
    // The bound pick/accept launch writes the next input in place, advances
    // positions and masks stopped rows. Up to eight steps, returned in
    // step-major order.
    [[nodiscard]] CudaTicket submit_chain(const Program& program, const CudaStepIn& first, std::uint32_t steps,
                                         CudaSubmitOptions options = {});
    // The native ABI uploads the caller's complete X3 block, including the
    // feed and sampling/stop header and padded rows.
    [[nodiscard]] CudaTicket submit_x3(const Program& program, const StepBlock& input, CudaSubmitOptions options = {});
    [[nodiscard]] CudaTicket submit_x3_chain(const Program& program, const StepBlock& first, std::uint32_t steps,
                                            CudaSubmitOptions options = {});
    // Optionally verify and install the graph before submitting other work.
    void prepare_resident(const Program& program, const CudaProgramBinding& binding);
    // In resident layout, install the kernel-only graph before invoking prepare.
    // The callback queues input work on the supplied non-default resident stream.
    // Graph replay, subsequent commit and readback are ordered on that stream.
    // The stream owner drains it before destroying the executor or its bindings.
    void submit_resident(const Program& program, const CudaProgramBinding& binding, void* resident_stream,
                         const std::function<void()>& prepare);
    // Both consume a completed ticket. There are eight staging slots. The
    // result span must be rows * steps elements.
    [[nodiscard]] bool poll(CudaTicket ticket, std::span<CudaStepOut> out);
    void wait(CudaTicket ticket, std::span<CudaStepOut> out);

    // Call with a new epoch after any change to captured layout bindings.
    // Supply a Program with refreshed args if it captures a moved address.
    void invalidate(std::uint64_t layout_epoch) noexcept;
    [[nodiscard]] CudaGraphStats graph_stats(const ProgramKey& key) const noexcept;
    // Executes a step eagerly and waits for its per-launch timing events.
    [[nodiscard]] CudaStepProfile profile(const Program& program, const CudaStepIn& input,
                                          const CudaProgramBinding* binding = nullptr);
    [[nodiscard]] CudaStepProfile profile_x3(const Program& program, const StepBlock& input,
                                             const CudaProgramBinding* binding = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::exec
