#pragma once
#include "strix/hip_launcher_catalog.hpp"
#include "strix/serial_session.hpp"
#include <memory>

namespace strix {
// Trusted scheduler owner: retains model, scratch and session allocations and
// excludes conflicting writes/rebinding until this graph is drained/destroyed.
// The returned resource metadata is rechecked before and after every replay.
class HipSerialGraphResources {
public:
    virtual ~HipSerialGraphResources() = default;
    [[nodiscard]] virtual const ExecutionRuntimeResources& current() const = 0;
};

// Explicit experimental C1 complete-DAG capture/replay. No model qualification
// or admission is inferred from constructing this object. Resource budgets are
// supplied by the owner; internal HIP graph memory still needs native profiling.
class HipSerialGraph {
public:
    HipSerialGraph(CompiledSerialExecution, std::shared_ptr<HipLauncherCatalog>,
        std::shared_ptr<const HipSerialGraphResources>, GraphMemoryRequest);
    ~HipSerialGraph();
    HipSerialGraph(const HipSerialGraph&) = delete;
    HipSerialGraph& operator=(const HipSerialGraph&) = delete;
    [[nodiscard]] EagerExecutionResult execute(SerialSessionContext&, std::uint32_t token);
    [[nodiscard]] bool drain() noexcept;
    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] const GraphCapturePlan& capture_plan() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
