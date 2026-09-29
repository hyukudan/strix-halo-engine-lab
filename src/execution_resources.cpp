#include "strix/execution_resources.hpp"

#include <stdexcept>

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

void hash_ptr(
    std::uint64_t& hash,
    std::uintptr_t value) noexcept {
    hash_u64(
        hash,
        static_cast<std::uint64_t>(value));
}

}  // namespace

void validate_execution_runtime_resources(
    std::uint64_t compiled_execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources,
    std::uint64_t position) {
    if (compiled_execution_identity == 0) {
        throw std::invalid_argument(
            "compiled execution identity must be non-zero");
    }
    if (resources.scratch == nullptr ||
        resources.model_manifest == nullptr ||
        resources.model_weights == nullptr ||
        resources.tensor_access == nullptr ||
        resources.session_state_layout == nullptr ||
        resources.session_state_memory == nullptr) {
        throw std::invalid_argument(
            "compiled execution requires complete runtime resources");
    }

    validate_execution_memory_binding(
        compiled_execution_identity,
        arena,
        *resources.scratch);
    validate_model_runtime_binding(
        *resources.model_manifest,
        *resources.model_weights);
    if (resources.tensor_access->manifest_layout_identity !=
            resources.model_manifest->layout_identity ||
        resources.tensor_access->plan_identity == 0) {
        throw std::invalid_argument(
            "tensor access plan manifest identity mismatch");
    }
    validate_session_state_memory_binding(
        *resources.session_state_layout,
        *resources.session_state_memory);

    if (position >=
        resources.session_state_layout->context_tokens) {
        throw std::out_of_range(
            "execution position exceeds allocated session-state capacity");
    }
}

RuntimeExecutionIdentities
build_runtime_execution_identities(
    std::uint64_t compiled_execution_identity,
    const ExecutionArenaPlan& arena,
    const ExecutionRuntimeResources& resources) {
    validate_execution_runtime_resources(
        compiled_execution_identity,
        arena,
        resources,
        0);

    RuntimeExecutionIdentities identities;

    std::uint64_t compatibility = fnv_offset;
    hash_u64(
        compatibility,
        compiled_execution_identity);
    hash_u64(
        compatibility,
        resources.model_manifest->layout_identity);
    hash_u64(
        compatibility,
        resources.model_weights->model_content_identity);
    hash_u64(
        compatibility,
        resources.tensor_access->plan_identity);
    hash_u64(
        compatibility,
        resources.session_state_layout->layout_identity);

    if (compatibility == 0) {
        throw std::logic_error(
            "runtime compatibility identity must be non-zero");
    }
    identities.compatibility_identity =
        compatibility;

    std::uint64_t binding = compatibility;
    hash_ptr(
        binding,
        resources.scratch->arena_base);
    hash_u64(
        binding,
        resources.scratch->arena_bytes);
    hash_u64(
        binding,
        resources.scratch->allocation_generation);

    hash_ptr(
        binding,
        resources.session_state_memory->device_base);
    hash_u64(
        binding,
        resources.session_state_memory->device_bytes);
    hash_u64(
        binding,
        resources.session_state_memory->device_allocation_generation);
    hash_ptr(
        binding,
        resources.session_state_memory->host_base);
    hash_u64(
        binding,
        resources.session_state_memory->host_bytes);
    hash_u64(
        binding,
        resources.session_state_memory->host_allocation_generation);

    for (const auto& tensor :
         resources.model_weights->tensors) {
        hash_u64(
            binding,
            tensor.tensor_index);
        hash_u64(
            binding,
            static_cast<std::uint64_t>(
                tensor.storage));
        hash_ptr(
            binding,
            tensor.address);
        hash_u64(
            binding,
            tensor.bytes);
        hash_u64(
            binding,
            tensor.allocation_generation);
        hash_u64(
            binding,
            tensor.owner_node);
    }

    if (binding == 0) {
        throw std::logic_error(
            "runtime binding identity must be non-zero");
    }
    identities.binding_identity = binding;
    return identities;
}

}  // namespace strix
