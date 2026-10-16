#pragma once

#include "strix/execution_tensor_access.hpp"
#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace strix {

// Immutable, deeply owned lookup metadata. Construction is not permission to
// execute: PreparedEagerExecution must validate the actual dynamic resources and
// validate_source_metadata() before publishing this pointer in a run context.
// The caller excludes metadata mutation and retains the prepared owner through
// device quiescence. No request state or mutable "validated" flag is cached.
class PreparedTensorViews final {
public:
    PreparedTensorViews(std::uint64_t execution_identity,
        const SerialExecutionPlan& execution,
        const ModelManifest& manifest,
        const ModelRuntimeBinding& weights,
        const ExecutionTensorAccessPlan& access,
        std::shared_ptr<const void> actual_owner);
    ~PreparedTensorViews();
    PreparedTensorViews(const PreparedTensorViews&) = delete;
    PreparedTensorViews& operator=(const PreparedTensorViews&) = delete;
    PreparedTensorViews(PreparedTensorViews&&) = delete;
    PreparedTensorViews& operator=(PreparedTensorViews&&) = delete;

    // Full structural equality, including fields omitted by runtime identity
    // hashes. O(tensors + metadata + accesses), no allocation, once per entry.
    void validate_source_metadata(const ModelManifest&, const ModelRuntimeBinding&,
                                  const ExecutionTensorAccessPlan&) const;
    // Constant-time cross-binding check for each dispatch lookup. Replacing an
    // otherwise identical descriptor object requires a new prepared owner.
    void validate_source_handles(const ModelManifest*, const ModelRuntimeBinding*,
                                 const ExecutionTensorAccessPlan*) const;

    // Valid lookups allocate nothing and return references into this deep
    // snapshot. Tensor index and dense step/slot lookup are O(1); name lookup
    // and a sparse-slot fallback are O(log N).
    [[nodiscard]] const RuntimeTensorBinding& tensor(
        std::uint64_t execution_identity, std::uint32_t index) const;
    [[nodiscard]] const RuntimeTensorBinding& tensor(
        std::uint64_t execution_identity, std::string_view name) const;
    [[nodiscard]] const ExecutionTensorAccess& access(
        std::uint64_t execution_identity, std::uint32_t step_id,
        std::uint32_t binding_slot) const;
private:
    struct Impl;
    std::unique_ptr<const Impl> impl_;
};
} // namespace strix
