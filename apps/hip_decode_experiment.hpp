#pragma once

#include "strix/hip_serial_graph.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"

// This diagnostic scheduler owns all allocations referenced by a graph or an
// integer projection pin. Metadata is copied and rebound to these strong owners;
// leaking a failed native graph cannot leave references to the caller's stack.
struct HipDecodeExperimentResources final : strix::HipSerialGraphResources {
    HipDecodeExperimentResources(const HipDecodeExperimentResources&) = delete;
    HipDecodeExperimentResources& operator=(const HipDecodeExperimentResources&) = delete;
    HipDecodeExperimentResources(HipDecodeExperimentResources&&) = delete;
    HipDecodeExperimentResources& operator=(HipDecodeExperimentResources&&) = delete;
    std::shared_ptr<strix::MappedModelFile> mapped;
    std::shared_ptr<strix::HipModelResidency> model;
    std::shared_ptr<strix::HipRuntimeAllocator> allocator;
    std::shared_ptr<strix::RuntimeExecutionMemoryOwner> memory;
    strix::ModelManifest manifest;
    strix::ExecutionTensorAccessPlan tensor_access;
    strix::SerialSessionStateLayout layout;
    strix::ExecutionRuntimeResources resources;
    HipDecodeExperimentResources(std::shared_ptr<strix::MappedModelFile> file,
        std::shared_ptr<strix::HipModelResidency> weights,
        std::shared_ptr<strix::HipRuntimeAllocator> allocation_provider,
        std::shared_ptr<strix::RuntimeExecutionMemoryOwner> allocations,
        const strix::ModelManifest& manifest_value,
        const strix::ExecutionTensorAccessPlan& access_value,
        const strix::SerialSessionStateLayout& layout_value)
        : mapped(std::move(file)), model(std::move(weights)), allocator(std::move(allocation_provider)),
          memory(std::move(allocations)), manifest(manifest_value), tensor_access(access_value), layout(layout_value) {
        if (!mapped || !model || !allocator || !memory || !memory->valid())
            throw std::invalid_argument("native decode experiment requires strong allocation owners");
        resources.scratch = &memory->scratch();
        resources.model_manifest = &manifest;
        resources.model_weights = &model->binding();
        resources.tensor_access = &tensor_access;
        resources.session_state_layout = &layout;
        resources.session_state_memory = &memory->state();
    }
    const strix::ExecutionRuntimeResources& current() const override {
        if (!memory->valid()) throw std::invalid_argument("native decode allocations retired");
        return resources;
    }
};
