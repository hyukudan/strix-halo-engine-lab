#include "strix/prepared_tensor_views.hpp"

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
bool same_manifest(const ModelManifest& a, const ModelManifest& b) noexcept {
    if (a.architecture != b.architecture || a.layout_identity != b.layout_identity ||
        a.total_tensor_bytes != b.total_tensor_bytes ||
        a.file_backed_candidate_bytes != b.file_backed_candidate_bytes ||
        a.bounded_cache_candidate_bytes != b.bounded_cache_candidate_bytes ||
        a.unknown_tensor_count != b.unknown_tensor_count || a.tensors.size() != b.tensors.size()) return false;
    for (std::size_t i = 0; i < a.tensors.size(); ++i) {
        const auto& x = a.tensors[i]; const auto& y = b.tensors[i];
        if (x.name != y.name || x.dimensions != y.dimensions || x.ggml_type != y.ggml_type ||
            x.byte_size != y.byte_size || x.absolute_offset != y.absolute_offset || x.role != y.role ||
            x.residency != y.residency || x.placement.kind != y.placement.kind ||
            x.placement.node != y.placement.node || x.layer != y.layer) return false;
    }
    return true;
}
bool same_weights(const ModelRuntimeBinding& a, const ModelRuntimeBinding& b) noexcept {
    if (a.manifest_layout_identity != b.manifest_layout_identity ||
        a.model_content_identity != b.model_content_identity || a.tensors.size() != b.tensors.size()) return false;
    for (std::size_t i = 0; i < a.tensors.size(); ++i) {
        const auto& x = a.tensors[i]; const auto& y = b.tensors[i];
        if (x.tensor_index != y.tensor_index || x.storage != y.storage || x.address != y.address ||
            x.bytes != y.bytes || x.allocation_generation != y.allocation_generation ||
            x.owner_node != y.owner_node) return false;
    }
    return true;
}
bool same_access(const ExecutionTensorAccessPlan& a, const ExecutionTensorAccessPlan& b) noexcept {
    if (a.manifest_layout_identity != b.manifest_layout_identity || a.plan_identity != b.plan_identity ||
        a.accesses.size() != b.accesses.size()) return false;
    for (std::size_t i = 0; i < a.accesses.size(); ++i) {
        const auto& x = a.accesses[i]; const auto& y = b.accesses[i];
        if (x.step_id != y.step_id || x.op != y.op || x.binding_slot != y.binding_slot ||
            x.tensor_indices != y.tensor_indices) return false;
    }
    return true;
}
void validate_storage(RuntimeTensorStorage storage) {
    switch (storage) {
    case RuntimeTensorStorage::resident:
    case RuntimeTensorStorage::file_mapped:
    case RuntimeTensorStorage::registered_file_mapped:
    case RuntimeTensorStorage::repacked_cache:
    case RuntimeTensorStorage::remote: return;
    }
    throw std::invalid_argument("prepared tensor storage is invalid");
}
} // namespace

struct PreparedTensorViews::Impl {
    const std::uint64_t execution_identity;
    const std::shared_ptr<const void> actual_owner;
    const ModelManifest manifest;
    const ModelRuntimeBinding weights;
    const ExecutionTensorAccessPlan accesses;
    const ModelManifest* const manifest_handle;
    const ModelRuntimeBinding* const weights_handle;
    const ExecutionTensorAccessPlan* const access_handle;
    std::map<std::string, std::uint32_t, std::less<>> names;

    Impl(std::uint64_t identity, const SerialExecutionPlan& execution,
         const ModelManifest& source_manifest, const ModelRuntimeBinding& source_weights,
         const ExecutionTensorAccessPlan& source_access, std::shared_ptr<const void> owner)
        : execution_identity(identity), actual_owner(std::move(owner)), manifest(source_manifest),
          weights(source_weights), accesses(source_access), manifest_handle(&source_manifest),
          weights_handle(&source_weights), access_handle(&source_access) {
        if (!execution_identity || !actual_owner || execution.steps.empty() ||
            execution.steps.size() > std::numeric_limits<std::uint32_t>::max() ||
            manifest.tensors.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("prepared tensor views require an execution and retained actual owners");
        for (std::size_t i = 0; i < execution.steps.size(); ++i) {
            const auto& step = execution.steps[i];
            if (step.id != i || to_string(step.op) == "unknown")
                throw std::invalid_argument("prepared tensor execution steps must be dense and valid");
            for (const auto dependency : step.dependencies) if (dependency >= step.id)
                throw std::invalid_argument("prepared tensor execution dependency is invalid");
        }
        validate_model_runtime_binding(manifest, weights);
        for (const auto& tensor : weights.tensors) validate_storage(tensor.storage);
        const auto index = build_model_tensor_index(manifest); // verifies actual layout and unique names
        const auto requirements = build_serial_execution_tensor_requirements(execution);
        const auto expected = compile_execution_tensor_access_plan(execution, manifest, index, requirements);
        if (!same_access(expected, accesses))
            throw std::invalid_argument("prepared tensor access differs from canonical step/slot requirements");
        for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
            if (manifest.tensors[i].name.empty() ||
                !names.emplace(manifest.tensors[i].name, static_cast<std::uint32_t>(i)).second)
                throw std::invalid_argument("prepared tensor names must be nonempty and unique");
        }
    }
    void require_execution(std::uint64_t identity) const {
        if (!identity || identity != execution_identity)
            throw std::invalid_argument("prepared tensor execution identity differs");
    }
};

PreparedTensorViews::PreparedTensorViews(std::uint64_t execution_identity,
    const SerialExecutionPlan& execution, const ModelManifest& manifest,
    const ModelRuntimeBinding& weights, const ExecutionTensorAccessPlan& access,
    std::shared_ptr<const void> actual_owner)
    : impl_(std::make_unique<Impl>(execution_identity, execution, manifest, weights, access, std::move(actual_owner))) {}
PreparedTensorViews::~PreparedTensorViews() = default;

void PreparedTensorViews::validate_source_metadata(const ModelManifest& manifest,
    const ModelRuntimeBinding& weights, const ExecutionTensorAccessPlan& access) const {
    validate_source_handles(&manifest, &weights, &access);
    if (!same_manifest(impl_->manifest, manifest) || !same_weights(impl_->weights, weights) ||
        !same_access(impl_->accesses, access))
        throw std::invalid_argument("prepared tensor source metadata or bindings changed");
}
void PreparedTensorViews::validate_source_handles(const ModelManifest* manifest,
    const ModelRuntimeBinding* weights, const ExecutionTensorAccessPlan* access) const {
    if (manifest != impl_->manifest_handle || weights != impl_->weights_handle || access != impl_->access_handle)
        throw std::invalid_argument("prepared tensor source descriptor handles changed");
}
const RuntimeTensorBinding& PreparedTensorViews::tensor(std::uint64_t identity, std::uint32_t index) const {
    impl_->require_execution(identity);
    if (index >= impl_->weights.tensors.size()) throw std::out_of_range("dispatch tensor index is out of range");
    return impl_->weights.tensors[index];
}
const RuntimeTensorBinding& PreparedTensorViews::tensor(std::uint64_t identity, std::string_view name) const {
    impl_->require_execution(identity);
    const auto found = impl_->names.find(name);
    if (found == impl_->names.end()) throw std::invalid_argument("dispatch tensor name is not present in manifest");
    return impl_->weights.tensors[found->second];
}
const ExecutionTensorAccess& PreparedTensorViews::access(std::uint64_t identity,
    std::uint32_t step_id, std::uint32_t binding_slot) const {
    impl_->require_execution(identity);
    const auto* found = find_execution_tensor_access(impl_->accesses, step_id, binding_slot);
    if (!found) throw std::invalid_argument("dispatch tensor binding slot is not present for this step");
    return *found;
}
} // namespace strix
