#include "strix/prepared_tensor_views.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
bool forbid_allocations = false;
}
void* operator new(std::size_t bytes) {
    if (forbid_allocations) throw std::bad_alloc();
    if (auto* memory = std::malloc(bytes ? bytes : 1U)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace strix;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class Function> void reject(Function&& action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid prepared tensor source or lookup accepted");
}
struct AllocationOwner { std::array<float,8> payload{}; };
struct Fixture {
    std::shared_ptr<AllocationOwner> owner = std::make_shared<AllocationOwner>();
    ModelManifest manifest;
    ModelRuntimeBinding weights;
    SerialExecutionPlan execution;
    ExecutionTensorAccessPlan access;
    Fixture() {
        manifest.architecture = "host-only-prepared-tensor-fixture";
        for (unsigned i = 0; i < 2; ++i) {
            ManifestTensor tensor;
            tensor.name = i ? "output.weight" : "token_embd.weight";
            tensor.dimensions = {4,1}; tensor.ggml_type = 0; tensor.byte_size = 16;
            tensor.absolute_offset = i * 16U; tensor.role = i ? TensorRole::output_head : TensorRole::embedding;
            manifest.tensors.push_back(std::move(tensor));
        }
        manifest.total_tensor_bytes = 32;
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        weights.manifest_layout_identity = manifest.layout_identity;
        weights.model_content_identity.bytes[0] = 42;
        for (unsigned i = 0; i < 2; ++i)
            weights.tensors.push_back({i,RuntimeTensorStorage::resident,
                reinterpret_cast<std::uintptr_t>(owner->payload.data() + i * 4U),16,7U+i,0});
        ExecutionStep embedding; embedding.id = 0; embedding.op = SerialOp::token_embedding;
        ExecutionStep head; head.id = 1; head.op = SerialOp::lm_head; head.dependencies = {0};
        execution.steps = {embedding,head};
        access = compile_execution_tensor_access_plan(execution, manifest, build_model_tensor_index(manifest),
            build_serial_execution_tensor_requirements(execution));
    }
    std::unique_ptr<PreparedTensorViews> prepare() const {
        return std::make_unique<PreparedTensorViews>(17,execution,manifest,weights,access,owner);
    }
};
struct NoAllocations {
    NoAllocations() { forbid_allocations = true; }
    ~NoAllocations() { forbid_allocations = false; }
};
}
int main() try {
    Fixture fixture;
    auto cached = fixture.prepare();
    cached->validate_source_metadata(fixture.manifest,fixture.weights,fixture.access);
    cached->validate_source_handles(&fixture.manifest,&fixture.weights,&fixture.access);
    auto equivalent_manifest = fixture.manifest;
    auto equivalent_weights = fixture.weights;
    auto equivalent_access = fixture.access;
    reject([&]{cached->validate_source_handles(&equivalent_manifest,&fixture.weights,&fixture.access);});
    reject([&]{cached->validate_source_handles(&fixture.manifest,&equivalent_weights,&fixture.access);});
    reject([&]{cached->validate_source_handles(&fixture.manifest,&fixture.weights,&equivalent_access);});
    reject([&]{cached->validate_source_handles(nullptr,&fixture.weights,&fixture.access);});
    reject([&]{cached->validate_source_metadata(equivalent_manifest,equivalent_weights,equivalent_access);});
    const auto address = fixture.weights.tensors[0].address;
    require(cached->tensor(17,0U).address == address && &cached->tensor(17,0U) == &cached->tensor(17,"token_embd.weight"),
            "index/name lookup must identify the same deeply owned binding");
    require(cached->access(17,0,0).tensor_indices == std::vector<std::uint32_t>{0}, "embedding access differs");
    require(cached->access(17,1,0).tensor_indices == std::vector<std::uint32_t>{1}, "head access differs");
    reject([&]{ (void)cached->tensor(18,0U); });
    reject([&]{ (void)cached->tensor(0,"output.weight"); });
    reject([&]{ (void)cached->access(18,0,0); });
    reject([&]{ (void)cached->tensor(17,"missing.weight"); });
    reject([&]{ (void)cached->access(17,0,1); });
    reject([&]{ (void)cached->access(17,2,0); });
    bool bad_index = false;
    try { (void)cached->tensor(17,2U); } catch (const std::out_of_range&) { bad_index = true; }
    require(bad_index,"out-of-range tensor index accepted");

    // These mutations deliberately keep the advertised layout/access IDs fixed.
    // The full metadata comparison, not a hash-only shortcut, must reject them.
    for (unsigned mode = 0; mode < 24; ++mode) {
        auto saved_manifest = fixture.manifest; auto saved_weights = fixture.weights; auto saved_access = fixture.access;
        auto& manifest = fixture.manifest; auto& weights = fixture.weights; auto& access = fixture.access;
        if (mode == 0) manifest.tensors[0].name += ".changed";
        if (mode == 1) ++manifest.tensors[0].dimensions[0];
        if (mode == 2) ++manifest.tensors[0].ggml_type;
        if (mode == 3) ++manifest.tensors[0].byte_size;
        if (mode == 4) ++manifest.tensors[0].absolute_offset;
        if (mode == 5) manifest.tensors[0].role = TensorRole::unknown;
        if (mode == 6) manifest.tensors[0].residency = ResidencyPolicy::file_backed_candidate;
        if (mode == 7) manifest.tensors[0].placement.kind = PlacementKind::remote;
        if (mode == 8) ++manifest.tensors[0].placement.node;
        if (mode == 9) manifest.tensors[0].layer = 3;
        if (mode == 10) ++manifest.total_tensor_bytes;
        if (mode == 11) manifest.architecture += ".changed";
        if (mode == 12) ++weights.tensors[0].address;
        if (mode == 13) ++weights.tensors[0].allocation_generation;
        if (mode == 14) ++weights.tensors[0].bytes;
        if (mode == 15) weights.tensors[0].storage = RuntimeTensorStorage::registered_file_mapped;
        if (mode == 16) ++weights.tensors[0].owner_node;
        if (mode == 17) ++weights.model_content_identity.bytes[0];
        if (mode == 18) access.accesses[0].tensor_indices[0] = 1;
        if (mode == 19) ++access.accesses[0].binding_slot;
        if (mode == 20) access.accesses[0].op = SerialOp::lm_head;
        if (mode == 21) ++access.accesses[0].step_id;
        if (mode == 22) ++access.plan_identity;
        if (mode == 23) ++manifest.layout_identity;
        reject([&]{cached->validate_source_metadata(manifest,weights,access);});
        fixture.manifest = std::move(saved_manifest);
        fixture.weights = std::move(saved_weights);
        fixture.access = std::move(saved_access);
    }
    // Snapshot references are independent of the mutable metadata containers.
    fixture.manifest.tensors[0].name = "mutated.after.prepare";
    fixture.weights.tensors[0].address = 0;
    fixture.access.accesses[0].tensor_indices[0] = 1;
    require(cached->tensor(17,"token_embd.weight").address == address &&
            cached->access(17,0,0).tensor_indices[0] == 0,"snapshot aliases source descriptors");
    {
        NoAllocations no_allocations;
        for (unsigned i = 0; i < 1000; ++i) {
            cached->validate_source_handles(&fixture.manifest,&fixture.weights,&fixture.access);
            (void)cached->tensor(17,0U); (void)cached->tensor(17,"output.weight");
            (void)cached->access(17,1,0);
        }
    }

    for (unsigned mode = 0; mode < 10; ++mode) {
        Fixture bad;
        if (mode == 0) bad.manifest.layout_identity = 0;
        if (mode == 1) bad.weights.tensors[0].tensor_index = 1;
        if (mode == 2) bad.weights.tensors[0].address = 0;
        if (mode == 3) bad.weights.tensors[0].allocation_generation = 0;
        if (mode == 4) bad.access.accesses.push_back(bad.access.accesses[0]);
        if (mode == 5) bad.access.accesses[0].tensor_indices.clear();
        if (mode == 6) bad.access.accesses[0].tensor_indices[0] = 99;
        if (mode == 7) bad.access.accesses[0].op = SerialOp::lm_head;
        if (mode == 8) bad.execution.steps[0].id = 4;
        if (mode == 9) {bad.manifest.tensors[1].name = bad.manifest.tensors[0].name;
            bad.manifest.layout_identity = model_manifest_layout_identity(bad.manifest);
            bad.weights.manifest_layout_identity = bad.manifest.layout_identity;}
        reject([&]{(void)bad.prepare();});
    }
    Fixture lifetime;
    std::weak_ptr<AllocationOwner> weak = lifetime.owner;
    auto retained = lifetime.prepare(); lifetime.owner.reset();
    require(!weak.expired(),"cache did not retain actual allocation owner");
    (void)retained->tensor(17,0U);
    retained.reset();require(weak.expired(),"cache leaked retained allocation owner");
    Fixture missing_owner;
    reject([&]{PreparedTensorViews invalid(17,missing_owner.execution,missing_owner.manifest,
        missing_owner.weights,missing_owner.access,{});});
    reject([&]{PreparedTensorViews invalid(0,missing_owner.execution,missing_owner.manifest,
        missing_owner.weights,missing_owner.access,missing_owner.owner);});
    std::cout << "prepared tensor views source fixtures completed\n";
    return 0;
} catch (const std::exception& error) {
    forbid_allocations = false;
    std::cerr << error.what() << '\n'; return 1;
}
