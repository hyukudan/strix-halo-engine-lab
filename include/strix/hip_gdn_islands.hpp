#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_resources.hpp"

#include <cstdint>
#include <memory>

namespace strix {
struct HipGdnLauncherConfig;
struct HipLaunchContext;
struct ExecutionDispatchContext;
class HipModelResidency;
class HipSetupQuiescenceGuard;
class HipGdnLauncherBundle;

struct HipGdnIslandBudget {
    std::uint32_t max_islands = 36;
    std::uint32_t max_kernel_nodes = 360;
    // Zero reports the observation without imposing a byte threshold.
    std::uint64_t observed_setup_delta_reject_bytes = 0;
};

struct HipGdnIslandReceipt {
    std::uint64_t organization_identity = 0;
    std::uint32_t islands = 0;
    std::uint32_t kernel_nodes = 0;
    bool memory_observed = false;
    std::int64_t observed_setup_free_delta_bytes = 0;
    std::uint64_t replays = 0;
};

// Fixed R1 GDN islands, constructed before registering the launch plan. Setup
// requires no active HipLaunchScope and exclusive ownership through quiescence.
// resources_owner must retain the actual D1 object and its borrowed resource
// owners; weights must be the residency whose binding is in those resources.
// Metadata must remain stable during each call. Replay is available only via
// the registered GDN launcher, inside the same D1 scope/resources binding.
class HipGdnIslands final {
public:
    [[nodiscard]] static std::shared_ptr<HipGdnIslands> create(
        const CompiledSerialExecution&, const HipGdnLauncherConfig&,
        const ExecutionRuntimeResources& d1_resources, const HipLaunchContext& d1_scope,
        std::shared_ptr<const HipModelResidency> weights,
        std::shared_ptr<const void> resources_owner,
        std::shared_ptr<const HipSetupQuiescenceGuard>, HipGdnIslandBudget = {});
    ~HipGdnIslands();
    HipGdnIslands(const HipGdnIslands&) = delete;
    HipGdnIslands& operator=(const HipGdnIslands&) = delete;
    HipGdnIslands(HipGdnIslands&&) = delete;
    HipGdnIslands& operator=(HipGdnIslands&&) = delete;
    [[nodiscard]] HipGdnIslandReceipt receipt() const noexcept;

private:
    struct Impl;
    explicit HipGdnIslands(std::unique_ptr<Impl>) noexcept;
    static void launch_gdn_island(const ExecutionDispatchContext&, void*);
    static void preflight_gdn_island(const ExecutionDispatchContext&, void*);
    std::unique_ptr<Impl> impl_;
    friend class HipGdnLauncherBundle;
};
} // namespace strix
