#include "strix/launch_dispatch.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void validate_launcher(
    const ExecutionLauncher& launcher) {
    if (launcher.key.empty() ||
        launcher.implementation_identity == 0 ||
        launcher.function == nullptr) {
        throw std::invalid_argument(
            "execution launcher requires key, identity and function");
    }
}

}  // namespace

void ExecutionLauncherRegistry::add_kernel_launcher(
    ExecutionLauncher launcher) {
    validate_launcher(launcher);
    if (find_kernel_launcher(
            launcher.key) != nullptr) {
        throw std::invalid_argument(
            "duplicate kernel launcher key");
    }
    kernel_launchers_.push_back(
        std::move(launcher));
}

void ExecutionLauncherRegistry::add_runtime_launcher(
    SerialOp op,
    ExecutionLauncher launcher) {
    validate_launcher(launcher);
    if (find_runtime_launcher(op) != nullptr) {
        throw std::invalid_argument(
            "duplicate runtime launcher operator");
    }
    runtime_launchers_.push_back(
        RuntimeLauncherEntry{
            op,
            std::move(launcher),
        });
}

const ExecutionLauncher*
ExecutionLauncherRegistry::find_kernel_launcher(
    std::string_view kernel_id) const noexcept {
    const auto found =
        std::find_if(
            kernel_launchers_.begin(),
            kernel_launchers_.end(),
            [&](const ExecutionLauncher& launcher) {
                return launcher.key ==
                       kernel_id;
            });
    return found ==
            kernel_launchers_.end()
        ? nullptr
        : &*found;
}

const ExecutionLauncher*
ExecutionLauncherRegistry::find_runtime_launcher(
    SerialOp op) const noexcept {
    const auto found =
        std::find_if(
            runtime_launchers_.begin(),
            runtime_launchers_.end(),
            [&](const RuntimeLauncherEntry& entry) {
                return entry.op == op;
            });
    return found ==
            runtime_launchers_.end()
        ? nullptr
        : &found->launcher;
}

CompiledLaunchPlan compile_launch_plan(
    const CompiledSerialExecution& compiled,
    const ExecutionLauncherRegistry& launchers) {
    if (compiled.execution_identity == 0 ||
        compiled.bindings.bindings.size() !=
            compiled.execution.steps.size()) {
        throw std::invalid_argument(
            "launch plan requires a valid compiled execution");
    }

    CompiledLaunchPlan plan;
    plan.execution_identity =
        compiled.execution_identity;
    plan.steps.reserve(
        compiled.execution.steps.size());

    std::uint64_t hash = fnv_offset;
    hash_u64(
        hash,
        compiled.execution_identity);

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        const auto& step =
            compiled.execution.steps[index];
        const auto& binding =
            compiled.bindings.bindings[index];

        if (step.id != index ||
            binding.step_id != step.id ||
            binding.op != step.op) {
            throw std::invalid_argument(
                "compiled execution step/binding identity mismatch");
        }

        const ExecutionLauncher* launcher = nullptr;
        if (binding.host_or_runtime_step) {
            launcher =
                launchers.find_runtime_launcher(
                    step.op);
        } else {
            if (!binding.kernel_id.has_value()) {
                throw std::invalid_argument(
                    "kernel step lacks bound kernel id");
            }
            launcher =
                launchers.find_kernel_launcher(
                    *binding.kernel_id);
        }

        if (launcher == nullptr) {
            throw std::invalid_argument(
                "compiled step has no registered launcher");
        }

        plan.steps.push_back(
            CompiledStepLauncher{
                step.id,
                step.op,
                binding.host_or_runtime_step,
                launcher->implementation_identity,
                launcher->function,
                launcher->user_data,
            });

        hash_u64(hash, step.id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                step.op));
        hash_u64(
            hash,
            binding.host_or_runtime_step
                ? 1ULL
                : 0ULL);
        hash_u64(
            hash,
            launcher->implementation_identity);
    }

    plan.plan_identity = hash;
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "compiled launch plan identity must be non-zero");
    }
    return plan;
}

const CompiledStepLauncher&
DirectLaunchDispatcher::launcher_for(
    const ExecutionDispatchContext& context,
    bool runtime_step) const {
    if (context.step_id >=
        plan_.steps.size()) {
        throw std::out_of_range(
            "dispatch step id is outside compiled launch plan");
    }

    const auto& launcher =
        plan_.steps[context.step_id];
    if (launcher.step_id !=
            context.step_id ||
        launcher.op !=
            context.op ||
        launcher.runtime_step !=
            runtime_step ||
        launcher.function == nullptr) {
        throw std::logic_error(
            "dispatch context does not match compiled launcher");
    }
    return launcher;
}

void DirectLaunchDispatcher::dispatch_kernel(
    const ExecutionDispatchContext& context) {
    const auto& launcher =
        launcher_for(
            context,
            false);
    launcher.function(
        context,
        launcher.user_data);
}

void DirectLaunchDispatcher::dispatch_runtime_step(
    const ExecutionDispatchContext& context) {
    const auto& launcher =
        launcher_for(
            context,
            true);
    launcher.function(
        context,
        launcher.user_data);
}

}  // namespace strix
