#include "strix/execution_tensor_access.hpp"

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

std::uint64_t plan_identity(
    const ExecutionTensorAccessPlan& plan) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(
        hash,
        plan.manifest_layout_identity);

    for (const auto& access : plan.accesses) {
        hash_u64(hash, access.step_id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                access.op));
        hash_u64(hash, access.binding_slot);
        hash_u64(
            hash,
            access.tensor_indices.size());
        for (const auto tensor :
             access.tensor_indices) {
            hash_u64(hash, tensor);
        }
        hash_u64(
            hash,
            0xffffffffffffffffULL);
    }
    return hash;
}

}  // namespace

ExecutionTensorAccessPlan
compile_execution_tensor_access_plan(
    const SerialExecutionPlan& execution_plan,
    const ModelManifest& manifest,
    const ModelTensorIndex& tensor_index,
    std::span<const ExecutionTensorRequirement> requirements) {
    if (manifest.layout_identity == 0 ||
        tensor_index.manifest_layout_identity !=
            manifest.layout_identity) {
        throw std::invalid_argument(
            "tensor access compilation requires matching manifest/index identity");
    }

    ExecutionTensorAccessPlan plan;
    plan.manifest_layout_identity =
        manifest.layout_identity;
    plan.accesses.reserve(
        requirements.size());

    for (const auto& requirement :
         requirements) {
        if (requirement.step_id >=
            execution_plan.steps.size()) {
            throw std::invalid_argument(
                "tensor requirement step id is out of range");
        }
        const auto& step =
            execution_plan.steps[
                requirement.step_id];
        if (step.id !=
                requirement.step_id ||
            step.op != requirement.op) {
            throw std::invalid_argument(
                "tensor requirement does not match execution step identity");
        }

        auto selected =
            select_manifest_tensors(
                manifest,
                tensor_index,
                requirement.selector);

        if (requirement.cardinality ==
                TensorRequirementCardinality::exactly_one &&
            selected.size() != 1) {
            throw std::invalid_argument(
                "exactly-one tensor requirement is missing or ambiguous");
        }
        if (requirement.cardinality ==
                TensorRequirementCardinality::one_or_more &&
            selected.empty()) {
            throw std::invalid_argument(
                "one-or-more tensor requirement matched no tensors");
        }

        ExecutionTensorAccess access;
        access.step_id =
            requirement.step_id;
        access.op =
            requirement.op;
        access.binding_slot =
            requirement.binding_slot;
        access.tensor_indices =
            std::move(selected);
        plan.accesses.push_back(
            std::move(access));
    }

    std::sort(
        plan.accesses.begin(),
        plan.accesses.end(),
        [](const auto& left,
           const auto& right) {
            if (left.step_id != right.step_id) {
                return left.step_id <
                       right.step_id;
            }
            return left.binding_slot <
                   right.binding_slot;
        });

    for (std::size_t index = 1;
         index < plan.accesses.size();
         ++index) {
        const auto& previous =
            plan.accesses[index - 1];
        const auto& current =
            plan.accesses[index];
        if (previous.step_id ==
                current.step_id &&
            previous.binding_slot ==
                current.binding_slot) {
            throw std::invalid_argument(
                "duplicate tensor binding slot for execution step");
        }
    }

    plan.plan_identity =
        plan_identity(plan);
    if (plan.plan_identity == 0) {
        throw std::logic_error(
            "execution tensor access identity must be non-zero");
    }
    return plan;
}

const ExecutionTensorAccess*
find_execution_tensor_access(
    const ExecutionTensorAccessPlan& plan,
    std::uint32_t step_id,
    std::uint32_t binding_slot) noexcept {
    const auto found =
        std::lower_bound(
            plan.accesses.begin(),
            plan.accesses.end(),
            std::pair{
                step_id,
                binding_slot},
            [](const ExecutionTensorAccess& access,
               const auto& needle) {
                if (access.step_id !=
                    needle.first) {
                    return access.step_id <
                           needle.first;
                }
                return access.binding_slot <
                       needle.second;
            });

    if (found == plan.accesses.end() ||
        found->step_id != step_id ||
        found->binding_slot !=
            binding_slot) {
        return nullptr;
    }
    return &*found;
}

}  // namespace strix
