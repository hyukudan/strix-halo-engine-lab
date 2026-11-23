#pragma once

#include "strix/exec/catalog.hpp"
#include "strix/exec/program.hpp"
#include "strix/graph/launch_plan.hpp"

#include <functional>
#include <map>

namespace strix::exec {

struct LaunchBindingRequest {
    const ProgramKey& key;
    const graph::LaunchPlan& plan;
    std::uint32_t launch_index;
    const BoundLaunch& launch;  // Ordered stage/port bindings; kernel fields are filled by the resolver.
    std::span<const BufferBinding> buffers;
};

struct BoundEntryCost {
    const KernelEntry* entry = nullptr;
    EntryCost cost{};
};

// The native binding declares the wait contract of the body it selects.
enum class PdlWait : std::uint8_t { undeclared, before_step_io, absent };
enum class PdlOptOut : std::uint8_t { none, kernel_without_wait, legacy_schedule };

struct PdlLeaf {
    PdlWait wait = PdlWait::undeclared;
    PdlOptOut opt_out = PdlOptOut::none;
};

struct KernelBinding {
    KernelId kernel{};
    LaunchFn fn = nullptr;
    std::string identity;        // Stable entry identity, including any compile-time numerical arguments.
    std::string chain_identity;
    // Stable arithmetic identity for the complete binding, including the chain
    // and numerical arguments. Leave empty when bitwise equivalence is undeclared.
    std::string exact_identity;
    Grid grid;
    // Thunk argument bytes, aligned to max_align_t in the program. Use stable
    // table addresses for state and movable weights. Workspace offsets come from the request.
    std::vector<std::byte> args;
    Cost predicted;
    bool capturable = true;
    bool grid_by_context = false;
    BoundEntryCost entry_cost{};
    PdlLeaf pdl{};
};

using KernelResolver = std::function<KernelBinding(const LaunchBindingRequest&)>;
// Called for material stage outputs. Unproduced inputs are external live-ins
// and fused prologue outputs are local to each launch that computes them.
using BufferResolver = std::function<BufferLayout(const ProgramKey&, const graph::Instance&)>;
using PlanProvider = std::function<graph::LaunchPlan(const graph::StageGraph&, const ProgramKey&)>;

struct CompilerOptions {
    std::uint64_t arena_alignment = 256;
    std::vector<StateBinding> states;
    // Model-specific contracts beyond the graph, and identities of loaded formats.
    std::string declaration_identity;
    std::uint64_t weight_format_identity = 0;
    std::uint64_t state_format_identity = 0;
};

// Host load-time owner. The declaration and its backing storage must stay
// unchanged for its lifetime. The provider fixes one launch plan per key; cache
// access is serialized by the owner. Supply it for get/warm, and return a plan
// for the key's position regime and row topology. References survive inserts.
class Compiler {
public:
    Compiler(const graph::StageGraph& declaration, BufferResolver buffers, KernelResolver kernels,
             CompilerOptions options = {}, PlanProvider provider = {});

    [[nodiscard]] Program compile(const ProgramKey& key, const graph::LaunchPlan& plan) const;
    [[nodiscard]] const Program& get(const ProgramKey& key);
    void warm(std::span<const ProgramKey> keys);

private:
    const graph::StageGraph& declaration_;
    BufferResolver buffers_;
    KernelResolver kernels_;
    CompilerOptions options_;
    PlanProvider provider_;
    std::uint64_t declaration_identity_;
    std::map<ProgramKey, Program> programs_;
};

}  // namespace strix::exec
