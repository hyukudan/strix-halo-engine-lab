#pragma once

#include "strix/device/launch_geometry.hpp"
#include "strix/exec/arena.hpp"
#include "strix/graph/stage_graph.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix::profile {
enum class Name : unsigned;
}

namespace strix::exec {

enum class Phase : std::uint8_t {
    prefill, decode, verify, draft, commit, snapshot, restore, state_only, image, encode, loop_step
};

struct ProgramKey {
    Phase phase = Phase::decode;
    std::uint32_t rows = 1;  // Decode/verify use 1, 2, 4, 8, 16 or 32. Prefill can be tall.
    graph::Pos pos = graph::Pos::any;
    bool tree = false;
    bool sampled = false;
    bool tap = false;
    std::uint8_t graph_id = 0;
    profile::Name tier{};
    std::uint8_t ctx_bucket = 0;  // 0 for fixed grids, otherwise a context class in [1, 8].

    auto operator<=>(const ProgramKey&) const = default;
};

enum class BufferStorage : std::uint8_t { arena, external, launch_local };

struct BufferLayout {
    BufferStorage storage = BufferStorage::arena;
    std::uint64_t row_bytes = 0;
    std::uint64_t row_stride = 0;  // 0 requests a packed stride.
};

struct BufferBinding {
    std::string symbol;
    std::uint32_t generation;  // 0 is a live-in, positive values are declared writes.
    BufferLayout layout;
    std::uint64_t arena_offset = 0;  // Used when layout.storage is arena.
};

struct StageBinding {
    std::uint32_t instance;  // Index into Program::stages, in LaunchPlan::instances order.
    std::array<BufferId, 4> inputs;
    BufferId output;
};

struct ProgramStage {
    std::string id;
    std::int32_t layer;
    graph::Format format;
};

// Opaque binding handle. The stable identity string is independent of this handle.
enum class KernelId : std::uint64_t {};

using Grid = device::LaunchGeometry;

using LaunchFn = void (*)(std::span<const std::byte> args, const Grid& grid, void* stream);
inline constexpr std::uint32_t no_launch = std::numeric_limits<std::uint32_t>::max();

struct BoundLaunch {
    KernelId kernel{};
    LaunchFn fn = nullptr;
    std::uint32_t args_offset = 0;
    std::uint32_t args_bytes = 0;
    Grid grid;
    std::uint8_t stream = 0;
    std::uint32_t after = no_launch;
    std::string entry;
    std::string identity;
    std::string chain_identity;
    std::int32_t layer = -1;
    std::uint32_t rows = 0;
    std::uint32_t kernels = 0;
    bool capturable = true;
    std::vector<StageBinding> prologue, stages, epilogue;
};

struct StateBinding {
    std::uint32_t declaration;
    std::uint32_t table_index;
    std::uint64_t mask;
};

// X3 owns the device step descriptor ABI. The program binds its row width.
struct StepBlock;
struct DescriptorLayout {
    std::uint32_t rows = 0;
};

struct Cost {
    double us = 0;
    std::uint64_t weight_bytes = 0, state_bytes = 0, macs = 0, launches = 0;
};

struct Program {
    ProgramKey key;
    std::vector<BoundLaunch> launches;
    std::vector<std::byte> args;
    std::vector<ProgramStage> stages;
    std::vector<BufferBinding> buffers;  // Indexed by BufferId.
    ArenaPlan arena;
    std::vector<StateBinding> states;
    DescriptorLayout descriptor;
    Cost predicted;
    std::uint64_t identity = 0;
    std::optional<std::uint64_t> numerical_identity;
    bool capturable = true;
};

}  // namespace strix::exec
