#include "strix/prepared_tensor_views.hpp"

#include <array>
#include <cstddef>
#include <algorithm>
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
struct RoleFixture {
    std::shared_ptr<std::array<std::byte,1048576>> owner=std::make_shared<std::array<std::byte,1048576>>();
    ModelManifest manifest;
    ModelRuntimeBinding weights;
    SerialExecutionPlan execution;
    ExecutionTensorAccessPlan access;
    RoleFixture() {
        manifest.architecture="host-role-descriptor-fixture";
        const auto add=[&](std::string name,TensorRole role,std::uint32_t layer,std::vector<std::uint64_t> dims) {
            ManifestTensor t;t.name=std::move(name);t.role=role;t.layer=layer;t.ggml_type=0;t.dimensions=std::move(dims);
            t.byte_size=4;for(auto n:t.dimensions) t.byte_size*=n;
            t.absolute_offset=manifest.total_tensor_bytes;manifest.total_tensor_bytes+=t.byte_size;
            manifest.tensors.push_back(std::move(t));
        };
        add("blk.0.attn_qkv.weight",TensorRole::attention_projection,0,{64,96});
        add("blk.0.attn_gate.weight",TensorRole::attention_projection,0,{64,32});
        add("blk.0.ssm_beta.weight",TensorRole::recurrent_state,0,{64,4});
        add("blk.0.ssm_alpha.weight",TensorRole::recurrent_state,0,{64,4});
        add("blk.0.ssm_conv1d.weight",TensorRole::recurrent_state,0,{4,96});
        add("blk.0.ssm_dt.bias",TensorRole::recurrent_state,0,{4});
        add("blk.0.ssm_a.weight",TensorRole::recurrent_state,0,{4});
        add("blk.0.ssm_norm.weight",TensorRole::recurrent_state,0,{8});
        add("blk.0.ssm_out.weight",TensorRole::recurrent_state,0,{32,64});
        add("blk.1.shared_expert.gate.weight",TensorRole::dense_projection,1,{64,32});
        add("blk.1.shared_expert.up.weight",TensorRole::dense_projection,1,{64,32});
        add("blk.1.shared_expert.down.weight",TensorRole::dense_projection,1,{32,64});
        add("blk.1.shared_expert_gate.weight",TensorRole::dense_projection,1,{64});
        add("blk.2.attn_q.weight",TensorRole::attention_projection,2,{64,32});
        add("blk.2.attn_k.weight",TensorRole::attention_projection,2,{64,32});
        add("blk.2.attn_v.weight",TensorRole::attention_projection,2,{64,32});
        add("blk.2.attn_output.weight",TensorRole::attention_projection,2,{32,64});
        add("blk.2.attn_q_norm.weight",TensorRole::normalization,2,{8});
        add("blk.2.attn_k_norm.weight",TensorRole::normalization,2,{8});
        add("blk.2.indexer.q_proj.weight",TensorRole::sparse_index,2,{64,32});
        add("blk.2.indexer.k_proj.weight",TensorRole::sparse_index,2,{64,32});
        add("blk.2.indexer.q_norm.weight",TensorRole::sparse_index,2,{8});
        add("blk.2.indexer.k_norm.weight",TensorRole::sparse_index,2,{8});
        add("per_layer_token_embd.weight",TensorRole::ngram_table,3,{4,16});
        add("blk.3.ple_key.weight",TensorRole::ple_parameter,3,{4,16});
        add("blk.3.ple_value.weight",TensorRole::ple_parameter,3,{4,8});
        add("blk.3.ple_norm_key.weight",TensorRole::ple_parameter,3,{4});
        add("blk.3.ple_norm_query.weight",TensorRole::ple_parameter,3,{4});
        add("blk.3.ple_norm_conv.weight",TensorRole::ple_parameter,3,{4});
        add("blk.3.ple_conv1d.weight",TensorRole::ple_parameter,3,{4,4});
        add("blk.4.experts.gate.weight",TensorRole::moe_expert,4,{64,32,4});
        add("blk.4.experts.up.weight",TensorRole::moe_expert,4,{64,32,4});
        add("blk.4.experts.down.weight",TensorRole::moe_expert,4,{32,64,4});
        add("blk.5.experts.gate_up.weight",TensorRole::moe_expert,5,{64,64,4});
        add("blk.5.experts.down.weight",TensorRole::moe_expert,5,{32,64,4});
        require(manifest.total_tensor_bytes<=owner->size(),"role fixture actual backing too small");
        ExecutionStep gdn;gdn.id=0;gdn.op=SerialOp::gated_delta_net;gdn.layer=0;
        ExecutionStep shared;shared.id=1;shared.op=SerialOp::shared_expert;shared.layer=1;shared.dependencies={0};
        execution.steps={gdn,shared};
        for(const auto op:{SerialOp::indexed_attention,SerialOp::ple,SerialOp::moe_experts,SerialOp::moe_experts}) {
            ExecutionStep next;next.id=static_cast<std::uint32_t>(execution.steps.size());next.op=op;
            next.layer=next.id;next.dependencies={next.id-1U};execution.steps.push_back(std::move(next));
        }
        rebind();
    }
    void rebind() {
        manifest.layout_identity=model_manifest_layout_identity(manifest);
        weights={};weights.manifest_layout_identity=manifest.layout_identity;weights.model_content_identity.bytes[0]=7;
        for(std::size_t i=0;i<manifest.tensors.size();++i) {
            const auto& t=manifest.tensors[i];
            weights.tensors.push_back({static_cast<std::uint32_t>(i),RuntimeTensorStorage::resident,
                reinterpret_cast<std::uintptr_t>(owner->data()+t.absolute_offset),t.byte_size,9U,0U});
        }
        access=compile_execution_tensor_access_plan(execution,manifest,build_model_tensor_index(manifest),
            build_serial_execution_tensor_requirements(execution));
    }
    std::unique_ptr<PreparedTensorViews> prepare() const {
        return std::make_unique<PreparedTensorViews>(29,execution,manifest,weights,access,owner);
    }
};
struct NoAllocations {
    NoAllocations() { forbid_allocations = true; }
    ~NoAllocations() { forbid_allocations = false; }
};
}
int main() try {
    RoleFixture roles;
    auto role_cache=roles.prepare();
    const auto& gdn=role_cache->gdn_roles(29,0);
    const auto& shared=role_cache->shared_roles(29,1,64,32);
    require(gdn.qkv==0 && gdn.z==1 && gdn.beta==2 && gdn.alpha==3 && gdn.conv==4 &&
        gdn.dt==5 && gdn.a==6 && gdn.norm==7 && gdn.out==8,"prepared GDN typed roles differ from independent expected indices");
    require(shared.gate==9 && shared.up==10 && shared.down==11 && shared.shared_gate==12,
        "prepared shared typed roles confuse projection gate with scalar gate");
    const auto& indexed=role_cache->indexed_roles(29,2);
    require(indexed.q==13 && indexed.k==14 && indexed.v==15 && indexed.out==16 && indexed.q_norm==17 &&
        indexed.k_norm==18 && indexed.index_q==19 && indexed.index_k==20 && indexed.index_q_norm==21 && indexed.index_k_norm==22,
        "prepared QSA roles confuse attention and indexer/norm families");
    const auto& ple=role_cache->ple_roles(29,3);
    require(ple.table==23 && ple.key==24 && ple.value==25 && ple.norm_key==26 && ple.norm_query==27 && ple.norm_conv==28 && ple.conv==29,
        "prepared PLE roles confuse key projections with their norms");
    const auto& routed=role_cache->routed_roles(29,4,64,32,4);
    const auto& fused=role_cache->routed_roles(29,5,64,32,4);
    require(!routed.fused_gate_up && routed.gate==30 && routed.up==31 && routed.down==32 &&
        fused.fused_gate_up && fused.gate_up==33 && fused.down==34,"prepared MoE roles lost separate/fused geometry or experts");
    reject([&]{(void)role_cache->indexed_roles(29,0);});
    reject([&]{(void)role_cache->ple_roles(29,2);});
    reject([&]{(void)role_cache->routed_roles(29,4,64,32,5);});
    reject([&]{(void)role_cache->routed_roles(29,5,64,64,4);});
    reject([&]{(void)role_cache->gdn_roles(30,0);});
    reject([&]{(void)role_cache->gdn_roles(29,1);});
    reject([&]{(void)role_cache->shared_roles(29,0,64,32);});
    reject([&]{(void)role_cache->shared_roles(29,1,32,64);});
    reject([&]{(void)role_cache->shared_roles(29,1,64,33);});
    {
        NoAllocations no_allocations;
        for(unsigned i=0;i<1000U;++i) {
            (void)role_cache->gdn_roles(29,0);(void)role_cache->shared_roles(29,1,64,32);
            (void)role_cache->tensor(29,gdn.out);(void)role_cache->tensor(29,shared.shared_gate);
            (void)role_cache->indexed_roles(29,2);(void)role_cache->ple_roles(29,3);
            (void)role_cache->routed_roles(29,4,64,32,4);(void)role_cache->routed_roles(29,5,64,32,4);
        }
    }
    for(unsigned mode=0;mode<5U;++mode) {
        auto saved=roles.manifest;auto old_weights=roles.weights;auto old_access=roles.access;
        if(mode==0) roles.manifest.tensors[6].name="blk.0.ssm_alpha.other";
        if(mode==1) ++roles.manifest.tensors[9].dimensions[1];
        if(mode==2) ++roles.weights.tensors[8].allocation_generation;
        if(mode==3) roles.access.accesses[0].tensor_indices.pop_back();
        if(mode==4) roles.manifest.tensors[12].role=TensorRole::normalization;
        reject([&]{role_cache->validate_source_metadata(roles.manifest,roles.weights,roles.access);});
        roles.manifest=std::move(saved);roles.weights=std::move(old_weights);roles.access=std::move(old_access);
    }
    // Rebuild in a different manifest order: descriptors are derived from
    // semantics/geometry, not source positions or a hard-coded canonical list.
    std::reverse(roles.manifest.tensors.begin(),roles.manifest.tensors.end());roles.rebind();
    auto permuted=roles.prepare();
    require(roles.manifest.tensors[permuted->gdn_roles(29,0).a].name=="blk.0.ssm_a.weight" &&
        roles.manifest.tensors[permuted->shared_roles(29,1,64,32).shared_gate].name=="blk.1.shared_expert_gate.weight",
        "prepared role cache assumed original tensor order");
    require(roles.manifest.tensors[permuted->indexed_roles(29,2).index_k_norm].name=="blk.2.indexer.k_norm.weight" &&
        roles.manifest.tensors[permuted->ple_roles(29,3).value].name=="blk.3.ple_value.weight" &&
        roles.manifest.tensors[permuted->routed_roles(29,5,64,32,4).gate_up].name=="blk.5.experts.gate_up.weight",
        "remaining prepared roles assumed original tensor positions");
    RoleFixture ambiguous;
    ambiguous.manifest.tensors[6].name="blk.0.ssm_alpha.duplicate";ambiguous.rebind();
    auto ambiguous_cache=ambiguous.prepare();reject([&]{(void)ambiguous_cache->gdn_roles(29,0);});
    RoleFixture shared_ambiguous;
    shared_ambiguous.manifest.tensors[9].name="blk.1.shared_expert.up.duplicate";shared_ambiguous.rebind();
    auto shared_ambiguous_cache=shared_ambiguous.prepare();
    reject([&]{(void)shared_ambiguous_cache->shared_roles(29,1,64,32);});
    // Preserve the incumbent up-before-gate name priority rather than silently
    // choosing whichever role makes an ambiguous family look complete.
    RoleFixture shared_overlap;
    shared_overlap.manifest.tensors[9].name="blk.1.shared_expert.gate_up.weight";shared_overlap.rebind();
    auto shared_overlap_cache=shared_overlap.prepare();
    reject([&]{(void)shared_overlap_cache->shared_roles(29,1,64,32);});
    RoleFixture bad_indexed;bad_indexed.manifest.tensors[15].name="blk.2.attn_q.duplicate";bad_indexed.rebind();
    auto indexed_bad_cache=bad_indexed.prepare();reject([&]{(void)indexed_bad_cache->indexed_roles(29,2);});
    RoleFixture bad_ple;bad_ple.manifest.tensors[25].name="blk.3.ple_norm_key.duplicate";bad_ple.rebind();
    auto ple_bad_cache=bad_ple.prepare();reject([&]{(void)ple_bad_cache->ple_roles(29,3);});
    RoleFixture bad_routed;bad_routed.manifest.tensors[30].name="blk.4.experts.gate_up.weight";bad_routed.rebind();
    auto routed_bad_cache=bad_routed.prepare();reject([&]{(void)routed_bad_cache->routed_roles(29,4,64,32,4);});
    RoleFixture legacy;legacy.manifest.tensors[6].name="blk.0.ssm_in.weight";legacy.rebind();
    auto legacy_cache=legacy.prepare();reject([&]{(void)legacy_cache->gdn_roles(29,0);});
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
