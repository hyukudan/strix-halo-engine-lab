#pragma once

#include "strix/exec/catalog.hpp"
#include "strix/exec/program.hpp"
#include "strix/receipt/receipt.hpp"

#include <optional>

namespace strix::exec {

struct CudaProgramBinding;
struct CudaStepIn;
class CudaExecutor;

// Per-node facts from the bound plan, in Program::launches order. Byte counts
// describe logical reads, and ops use the operand class's peak convention.
struct LaunchWork {
    std::uint64_t weight_bytes = 0, state_bytes = 0, ops = 0;
    const KernelEntry* entry = nullptr;
    double peak_tops = 0;
    bool exchange = false;
};

struct LaunchSample {
    std::uint32_t launch = 0;
    std::string entry, family, key;
    std::vector<std::string> stages;
    std::optional<double> ms, byte_roof_pct, compute_roof_pct;
    std::uint64_t weight_bytes = 0, state_bytes = 0, ops = 0;
    bool exchange = false;
};

struct StepProfile {
    ProgramKey key;
    std::string program;
    std::uint32_t repeats = 0;
    std::vector<LaunchSample> launches;
    std::optional<double> predicted_ms, cold_wall_ms, wall_ms, busy_ms, host_ms, exchange_ms, byte_floor_ms;
    std::uint64_t bytes_per_step = 0, exchanges = 0;
    std::uint32_t device_status = 0;
};

struct ProfileOptions {
    std::span<const LaunchWork> work;
    double byte_gbps = 0;  // Zero leaves the byte roof unreported.
    std::uint32_t repeats = 1;
};

struct ProfileTimings {
    // Means per replay. Launch intervals are from an eager, instrumented arm.
    std::span<const double> launch_ms;
    double wall_ms = 0, host_ms = 0;  // Production submit-to-wait and submit-only wall time.
    std::uint32_t device_status = 0;
    std::optional<double> cold_wall_ms;
};

// The SDK-free CPU stub uses the empty timing span to report plan facts only.
[[nodiscard]] StepProfile make_step_profile(const Program&, const ProfileOptions&, ProfileTimings = {});

// The caller owns a quiescent executor with replay enabled. Both arms execute
// the step repeatedly. Bind state that can be replayed without changing the
// work or its inputs (or use externally restored state via a separate run).
[[nodiscard]] StepProfile profile(CudaExecutor&, const Program&, const CudaStepIn&, const ProfileOptions&,
                                  const CudaProgramBinding* = nullptr);
[[nodiscard]] StepProfile profile(CudaExecutor&, const Program&, const StepBlock&, const ProfileOptions&,
                                  const CudaProgramBinding* = nullptr);

// Appends the R1 program row, then one row per program/entry/key with
// per-step bytes, node count and summed instrumented time for that key.
[[nodiscard]] receipt::Receipt profile_receipt(const StepProfile&, receipt::Receipt);
[[nodiscard]] std::string write_step_profile(const StepProfile&, receipt::Receipt);
[[nodiscard]] std::string write_step_table(const StepProfile&);

namespace bench {

enum class Kind { launch, byte_rate, exchange, entry };

struct Options {
    Kind kind = Kind::launch;
    std::uint32_t repeats = 20, nodes = 32, rows = 1, width = 4096;
    std::vector<std::uint64_t> bytes{4096, 1048576, 67108864};
    std::string entry, devices, device_id, device_profile, build, power_state, command;
    double byte_gbps = 0, peak_tops = 0;
};

// Defined only by the CUDA package. The app handles help, parsing and exchange
// without calling into the runtime.
struct Output {
    std::string json, table;
};

[[nodiscard]] Output run_cuda(const Options&);

}  // namespace bench
}  // namespace strix::exec
