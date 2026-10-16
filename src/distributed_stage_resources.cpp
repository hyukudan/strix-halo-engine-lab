#include "strix/distributed_stage_resources.hpp"
#include "strix/model_tensor_index.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
void hash(std::uint64_t& h,std::uint64_t n) noexcept {
    for(unsigned i=0;i<64U;i+=8U) {h^=(n>>i)&255U;h*=1099511628211ULL;}
}
bool same_access(const ExecutionTensorAccessPlan& a,const ExecutionTensorAccessPlan& b) {
    if(a.manifest_layout_identity!=b.manifest_layout_identity || a.plan_identity!=b.plan_identity || a.accesses.size()!=b.accesses.size()) return false;
    for(std::size_t i=0;i<a.accesses.size();++i) {
        const auto& x=a.accesses[i];const auto& y=b.accesses[i];
        if(x.step_id!=y.step_id || x.op!=y.op || x.binding_slot!=y.binding_slot || x.tensor_indices!=y.tensor_indices) return false;
    }
    return true;
}
bool same_placement(const DistributedPlacementPlan& a,const DistributedPlacementPlan& b) {
    if(a.plan_identity!=b.plan_identity || a.local_node!=b.local_node || a.compiled_execution_identity!=b.compiled_execution_identity ||
        a.manifest_layout_identity!=b.manifest_layout_identity || a.tensor_access_identity!=b.tensor_access_identity ||
        a.steps.size()!=b.steps.size() || a.boundaries.size()!=b.boundaries.size() || a.remote_nodes!=b.remote_nodes ||
        a.boundary_payload_bytes_per_token!=b.boundary_payload_bytes_per_token || a.executable!=b.executable || a.has_remote_work!=b.has_remote_work) return false;
    for(std::size_t i=0;i<a.steps.size();++i) {
        const auto& x=a.steps[i];const auto& y=b.steps[i];
        if(x.step_id!=y.step_id || x.op!=y.op || x.node!=y.node || x.remote!=y.remote || x.placement_ambiguous!=y.placement_ambiguous) return false;
    }
    for(std::size_t i=0;i<a.boundaries.size();++i) {
        const auto& x=a.boundaries[i];const auto& y=b.boundaries[i];
        if(x.boundary_id!=y.boundary_id || x.source_step!=y.source_step || x.destination_step!=y.destination_step ||
            x.source_node!=y.source_node || x.destination_node!=y.destination_node || x.payloads!=y.payloads || x.payload_bytes!=y.payload_bytes) return false;
    }
    return true;
}
}
DistributedStageResourcePlan DistributedStageResourcePlan::derive(const CompiledSerialExecution& serial,
    const ModelManifest& manifest,const ExecutionTensorAccessPlan& access,const DistributedPlacementPlan& canonical,std::uint32_t node) {
    if(node>1U || canonical.local_node!=0U || !serial.execution_identity || !serial.kernel_plan_identity ||
        serial.logical_rows!=1U || serial.request_concurrency!=1U || serial.device_graph_inputs ||
        !serial.arena.total_bytes || !serial.arena.alignment || (serial.arena.alignment&(serial.arena.alignment-1U)) ||
        serial.execution.steps.empty() || serial.execution.steps.size()>std::numeric_limits<std::uint32_t>::max() ||
        serial.execution.steps.size()!=serial.io.steps.size() || serial.execution.steps.size()!=serial.bindings.bindings.size() ||
        manifest.layout_identity!=model_manifest_layout_identity(manifest))
        throw std::invalid_argument("stage resource plan requires a canonical eager C1 source");
    validate_serial_execution_io_plan(serial.execution,serial.io);
    const auto expected_access=compile_execution_tensor_access_plan(serial.execution,manifest,build_model_tensor_index(manifest),
        build_serial_execution_tensor_requirements(serial.execution));
    if(!same_access(access,expected_access)) throw std::invalid_argument("stage access is not the complete canonical tensor-access plan");
    std::vector<std::uint32_t> nodes;for(const auto& s:canonical.steps) nodes.push_back(s.node);
    const auto expected=build_distributed_placement_plan(serial,manifest,access,nodes,0U);
    if(!same_placement(canonical,expected) || !canonical.executable || !canonical.has_remote_work)
        throw std::invalid_argument("stage placement differs from its complete canonical source");
    std::size_t cut=0;
    while(cut<nodes.size() && nodes[cut]==0U) ++cut;
    if(!cut || cut==nodes.size() || std::any_of(nodes.begin()+static_cast<std::ptrdiff_t>(cut),nodes.end(),[](auto n){return n!=1U;}))
        throw std::invalid_argument("capacity stage requires exactly one contiguous zero-to-one cut");
    const auto& left=serial.execution.steps[cut-1U];const auto& right=serial.execution.steps[cut];
    // Do not split the same layer's persistent operators or expose partial HC
    // temporaries as an initial capacity contract. Preserve global step IDs.
    if(!left.layer || !right.layer || *right.layer!=*left.layer+1U ||
        (right.op!=SerialOp::attn_hyper_mix && right.op!=SerialOp::ple))
        throw std::invalid_argument("capacity stage cut must be between complete adjacent layers");
    DistributedStageResourcePlan p;p.placement_=canonical;p.node_=node;
    p.first_=node?static_cast<std::uint32_t>(cut):0U;
    p.last_=node?static_cast<std::uint32_t>(nodes.size()-1U):static_cast<std::uint32_t>(cut-1U);
    p.execution_=serial.execution_identity;p.manifest_=manifest.layout_identity;p.access_=access.plan_identity;
    p.required_.resize(manifest.tensors.size());
    for(const auto& a:access.accesses) if(p.contains_step(a.step_id))
        for(const auto index:a.tensor_indices) {
            if(index>=p.required_.size()) throw std::invalid_argument("stage tensor index exceeds the source manifest");
            p.required_[index]=1U;
        }
    std::uint64_t h=14695981039346656037ULL;hash(h,0x5354475745494731ULL);hash(h,canonical.plan_identity);hash(h,node);
    for(std::size_t i=0;i<p.required_.size();++i) if(p.required_[i]) {
        const auto& t=manifest.tensors[i];
        if(t.placement.kind!=PlacementKind::local || !t.byte_size || t.byte_size>std::numeric_limits<std::uint64_t>::max()-p.weight_bytes_)
            throw std::invalid_argument("stage local tensor source is remote, empty or overflows");
        p.indices_.push_back(static_cast<std::uint32_t>(i));p.weight_bytes_+=t.byte_size;
        hash(h,i);hash(h,t.absolute_offset);hash(h,t.byte_size);hash(h,t.ggml_type);
    }
    if(p.indices_.empty() || !h) throw std::invalid_argument("stage has no materialized local weights");
    p.identity_=h;return p;
}
void DistributedStageResourcePlan::validate_source(const CompiledSerialExecution& serial,const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access) const {
    const auto next=derive(serial,manifest,access,placement_,node_);
    if(next.identity_!=identity_ || next.indices_!=indices_ || next.required_!=required_ || next.first_!=first_ || next.last_!=last_)
        throw std::invalid_argument("retained stage source or materialization scope changed");
}
void validate_stage_model_runtime_binding(const DistributedStageResourcePlan& plan,const ModelManifest& manifest,
    const ModelRuntimeBinding& binding) {
    if(manifest.layout_identity!=plan.manifest_identity() || manifest.layout_identity!=model_manifest_layout_identity(manifest) ||
        binding.manifest_layout_identity!=manifest.layout_identity || !binding.model_content_identity.valid() ||
        binding.tensors.size()!=manifest.tensors.size())
        throw std::invalid_argument("stage weight binding differs from canonical content/layout");
    for(std::size_t i=0;i<binding.tensors.size();++i) {
        const auto& r=binding.tensors[i];const auto& t=manifest.tensors[i];
        if(r.tensor_index!=i || r.bytes<t.byte_size) throw std::invalid_argument("stage tensor index or extent differs");
        if(plan.requires_tensor(static_cast<std::uint32_t>(i))) {
            if((!runtime_tensor_gpu_visible(r.storage) && !(r.storage==RuntimeTensorStorage::file_mapped && t.role==TensorRole::ngram_table)) || !r.address || !r.allocation_generation ||
                r.owner_node!=t.placement.node || t.placement.kind!=PlacementKind::local ||
                r.bytes>std::numeric_limits<std::uintptr_t>::max()-r.address)
                throw std::invalid_argument("required stage tensor has no complete actual local owner");
        } else if(r.storage!=RuntimeTensorStorage::remote || r.address || r.allocation_generation || r.owner_node!=1U-plan.local_node())
            throw std::invalid_argument("nonlocal stage tensor must be explicitly unmaterialized");
    }
}
} // namespace strix
