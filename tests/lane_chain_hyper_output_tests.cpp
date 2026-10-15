#include "strix/lane_chain_hyper_output.hpp"
#include "strix/execution_binding.hpp"
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <source_location>
namespace {
void expect(bool ok,std::source_location where=std::source_location::current()){if(!ok)throw std::runtime_error("lane-chain HC host contract failed at line "+std::to_string(where.line()));}
strix::ManifestTensor up(std::uint64_t h,std::uint64_t c,std::uint64_t r){strix::ManifestTensor t;t.ggml_type=8;t.dimensions={r,h*c};t.byte_size=r/32U*34U*h*c;return t;}
}
int main()try{
    expect(strix::lane_chain_hyper_output_lds(4,320)==44032);
    expect(strix::lane_chain_hyper_output_lds(8,224)==61952);
    expect(strix::lane_chain_hyper_output_lds(8,256)==0);
    for(auto h:{32U,64U,2560U})for(auto c:{2U,3U,4U,8U})for(auto r:{32U,96U,224U,320U,512U}){
        const auto t=up(h,c,r);expect(strix::lane_chain_hyper_output_eligible(t,h,c,r,t.absolute_offset)==(strix::lane_chain_hyper_output_lds(c,r)!=0));
        expect((34U*r)%16U==0); // Aligned stream tile is exactly32 Q8 rows.
        expect((32U*(r/32U*34U))==34U*r);
    }
    auto t=up(2560,4,320);
    for(unsigned bad=0;bad<7;++bad){auto x=t;if(bad==0)x.ggml_type=0;if(bad==1)--x.byte_size;if(bad==2)++x.dimensions[0];if(bad==3)x.absolute_offset=2;if(bad==4)x.placement.kind=strix::PlacementKind::remote;if(bad==5)x.placement.node=1;if(bad==6)x.dimensions.push_back(1);expect(!strix::lane_chain_hyper_output_eligible(x,2560,4,320,x.absolute_offset));}
    t.absolute_offset=2;expect(strix::lane_chain_hyper_output_eligible(t,2560,4,320,0));
    expect(!strix::lane_chain_hyper_output_eligible(up(36,4,320),36,4,320,0));
    // R320 row stride340 bytes =85dwords is a bank permutation for scalar
    // scale/code reads at fixed b,k; do not generalize to all rank%32 shapes.
    for(unsigned byte=0;byte<34;++byte){std::set<unsigned> banks;for(unsigned lane=0;lane<32;++lane)banks.insert(((lane*340U+byte)/4U)%32U);expect(banks.size()==32);}
    std::set<unsigned> short_banks;for(unsigned lane=0;lane<32;++lane)short_banks.insert(((lane*34U)/4U)%32U);expect(short_banks.size()<32);
    const strix::ModelContract model;strix::ModelManifest manifest;
    for(const auto& step:strix::build_serial_decode_execution_plan(model).steps){if(step.op!=strix::SerialOp::attn_hyper_mix && step.op!=strix::SerialOp::mlp_hyper_mix && step.op!=strix::SerialOp::final_hyper_mix)continue;auto x=up(model.hidden_size,model.hc_count,model.hc_lowrank);x.name=step.op==strix::SerialOp::final_hyper_mix?"output_hc_up.weight":"blk."+std::to_string(*step.layer)+(step.op==strix::SerialOp::attn_hyper_mix?".hc_attn_up.weight":".hc_ffn_up.weight");manifest.tensors.push_back(x);}
    strix::HipModelResidencyPolicy registered,copy;copy.mode=strix::HipModelResidencyMode::resident_copy;
    manifest.layout_identity=strix::model_manifest_layout_identity(manifest);expect(strix::lane_chain_hyper_output_steps(model,manifest,registered).size()==97);
    manifest.tensors.back().absolute_offset=2;manifest.layout_identity=strix::model_manifest_layout_identity(manifest);expect(strix::lane_chain_hyper_output_steps(model,manifest,registered).size()==96);expect(strix::lane_chain_hyper_output_steps(model,manifest,copy).size()==96);
    for(auto& x:manifest.tensors){x.absolute_offset=2;}
    manifest.layout_identity=strix::model_manifest_layout_identity(manifest);bool rejected=false;try{(void)strix::lane_chain_hyper_output_steps(model,manifest,registered);}catch(const std::invalid_argument&){rejected=true;}expect(rejected);
    std::uint32_t total_mixers=0;expect(strix::lane_chain_hyper_output_steps(model,manifest,copy,&total_mixers).size()==97 && total_mixers==97);
    // Copy alignment is relative to the shared coalesced allocation base.
    strix::ModelManifest layout;auto a=up(32,2,32),b=a;a.absolute_offset=2;a.byte_size=2;b.absolute_offset=16;b.byte_size=34;layout.tensors={a,b};
    copy.coalesce_gap_bytes=12;auto ranges=strix::build_copy_residency_ranges(layout,copy);expect(ranges.size()==1 && strix::copy_residency_tensor_offset(b,ranges)==14);
    expect(!strix::lane_chain_hyper_output_eligible(t,2560,4,320,strix::copy_residency_tensor_offset(b,ranges)));
    copy.coalesce_gap_bytes=11;ranges=strix::build_copy_residency_ranges(layout,copy);expect(ranges.size()==2 && strix::copy_residency_tensor_offset(b,ranges)==0);
    layout.tensors.front().placement.kind=strix::PlacementKind::remote;ranges=strix::build_copy_residency_ranges(layout,copy);expect(ranges.size()==1 && ranges.front().begin==16);
    strix::kernels::KernelRegistry registry;
    strix::ExecutionBindingRequest request;request.minimum_qualification=strix::kernels::QualificationStatus::experimental;
    strix::kernels::KernelVariant v;v.id="fixture.hc";v.family=strix::kernels::OperatorFamily::hyper_mix;v.phase=strix::kernels::ExecutionPhase::decode;v.shape=strix::execution_shape_for(model,strix::SerialOp::attn_hyper_mix,request);v.wave=strix::kernels::WaveSize::wave32;v.resources.workgroup_threads=256;v.numerics.input_precision="bf16";v.numerics.output_precision="bf16";v.numerics.accumulator_precision="fp32";v.numerics.row_invariant=true;v.numerics.deterministic=true;v.qualification=strix::kernels::QualificationStatus::experimental;v.graph_safe=true;v.input_layout="fixture";v.output_layout="fixture";v.compiler_identity="fixture";registry.add(v);v.id+=strix::lane_chain_hyper_output_suffix;registry.add(v);
    const auto plan=strix::build_serial_decode_execution_plan(model);request.lane_chain_hyper_output=true;
    for(const auto& step:plan.steps)if(step.op==strix::SerialOp::attn_hyper_mix || step.op==strix::SerialOp::mlp_hyper_mix)request.lane_chain_hyper_output_eligible_steps.push_back(step.id);
    const auto bound=strix::bind_serial_execution_plan(model,plan,registry,request);unsigned candidate_count=0,base_count=0;
    for(const auto& binding:bound.bindings)if(binding.family==strix::kernels::OperatorFamily::hyper_mix){expect(binding.kernel_id.has_value());if(binding.kernel_id->ends_with(strix::lane_chain_hyper_output_suffix))++candidate_count;else ++base_count;}
    expect(candidate_count==96 && base_count==1);
    auto bad_request=request;bad_request.lane_chain_hyper_output_eligible_steps.push_back(request.lane_chain_hyper_output_eligible_steps.front());rejected=false;try{(void)strix::bind_serial_execution_plan(model,plan,registry,bad_request);}catch(const std::invalid_argument&){rejected=true;}expect(rejected);
    const auto rejects=[&](const strix::ExecutionBindingRequest& bad){bool did=false;try{(void)strix::bind_serial_execution_plan(model,plan,registry,bad);}catch(const std::invalid_argument&){did=true;}expect(did);};
    bad_request=request;bad_request.lane_chain_hyper_output_eligible_steps.clear();rejects(bad_request);
    bad_request=request;bad_request.lane_chain_hyper_output=false;rejects(bad_request);
    bad_request=request;bad_request.lane_chain_hyper_output_eligible_steps.push_back(plan.steps.front().id);rejects(bad_request);
    std::cout<<"lane-chain HC output metadata/geometry/alignment/bank contracts PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
