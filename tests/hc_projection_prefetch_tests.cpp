#include "strix/hc_projection_prefetch.hpp"

#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace hc = strix::hc_projection_prefetch;
namespace {
int failures = 0;
void expect(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; ++failures; }
}
template<class F> void rejects(F&& action, const char* message, std::string_view gate = {}) {
    bool rejected = false;
    try { action(); }
    catch (const std::invalid_argument& error) {
        rejected = gate.empty() || gate == error.what();
        if (!rejected) std::cerr << "Unexpected gate: " << error.what() << '\n';
    }
    expect(rejected, message);
}

strix::CompiledSerialExecution execution(const strix::ModelContract& model) {
    strix::ExecutionBindingRequest request;
    request.prefetch_hc_projection_next_tile = true;
    strix::kernels::KernelRegistry registry;
    std::set<std::string> added;
    for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
        const auto family = strix::operator_family_for(step.op);
        if (!family) continue;
        const auto shape = strix::execution_shape_for(model, step.op, request);
        const auto key = std::to_string(static_cast<unsigned>(*family)) + "." + std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) continue;
        strix::kernels::KernelVariant variant;
        variant.id = "host.hc-preflight." + key;
        if (hc::mixer_op(step.op)) variant.id += hc::marker;
        variant.family = *family; variant.phase = strix::kernels::ExecutionPhase::decode;
        variant.shape = shape; variant.wave = strix::kernels::WaveSize::wave32;
        variant.resources.workgroup_threads = 128;
        variant.numerics.input_precision = "bf16";
        variant.numerics.accumulator_precision = "fp32";
        variant.numerics.output_precision = step.op == strix::SerialOp::lm_head ? "fp32" : "bf16";
        variant.numerics.row_invariant = true; variant.numerics.deterministic = true;
        variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
        variant.graph_safe = false; variant.input_layout = "row-major"; variant.output_layout = "row-major";
        variant.compiler_identity = "synthetic-host-hc-admission";
        registry.add(std::move(variant));
    }
    return strix::compile_serial_execution(model, registry, request, {});
}

struct Fixture {
    strix::ModelContract model = strix::ModelContract::qwen38_flash_next();
    strix::CompiledSerialExecution compiled = execution(model);
    strix::ModelManifest manifest;
    strix::ExecutionTensorAccessPlan access;
    strix::ModelRuntimeBinding resident;
    strix::ExecutionMemoryBinding memory{compiled.execution_identity, 0x100000000ULL, compiled.arena.total_bytes, 7};
    Fixture() {
        for (const auto& step : compiled.execution.steps) {
            if (!hc::mixer_op(step.op)) continue;
            const bool final = step.op == strix::SerialOp::final_hyper_mix;
            const std::string prefix = final ? "output_hc_" :
                "blk." + std::to_string(*step.layer) + (step.op == strix::SerialOp::attn_hyper_mix ? ".hc_attn_" : ".hc_ffn_");
            constexpr std::array<const char*, 4> suffixes{"norm", "down", "up", "inject"};
            // Independent fixture dimensions and byte sizes match the captured
            // canonical model, without asking the admission helper to derive them.
            const std::array<std::vector<std::uint64_t>, 4> shapes{{{10240}, {10240,320}, {320,10240}, {10240,4}}};
            constexpr std::array<std::uint64_t, 4> sizes{40960,3481600,3481600,163840};
            for (std::uint32_t slot = 0; slot < (final ? 3U : 4U); ++slot) {
                const auto index = static_cast<std::uint32_t>(manifest.tensors.size());
                strix::ManifestTensor tensor;
                tensor.name = prefix + suffixes[slot] + ".weight"; tensor.layer = step.layer;
                tensor.role = slot ? strix::TensorRole::dense_projection : strix::TensorRole::normalization;
                tensor.ggml_type = slot == 1U || slot == 2U ? 8U : 0U;
                tensor.dimensions = shapes[slot]; tensor.byte_size = sizes[slot];
                tensor.absolute_offset = 4096ULL + static_cast<std::uint64_t>(index) * 0x1000000ULL;
                manifest.tensors.push_back(std::move(tensor));
                access.accesses.push_back({step.id,step.op,slot,{index}});
                resident.tensors.push_back({index,strix::RuntimeTensorStorage::registered_file_mapped,
                    static_cast<std::uintptr_t>(0x1000000000ULL + static_cast<std::uint64_t>(index) * 0x1000000ULL),sizes[slot],3,0});
            }
        }
        manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
        access.manifest_layout_identity = resident.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 31;
    }
    hc::Plan plan() const { return hc::build_plan(model,manifest,compiled,access,true); }
    void validate(const hc::Plan& plan) const {
        hc::validate_runtime(plan,model,manifest,compiled,access,memory,resident);
    }
};

void views() {
    const hc::View normalized{0x10000,40960}, low{0x20000,1280}, down{0x1000000,3481600};
    const hc::View inject{0x2000000,163840}, output{0x30000,16};
    hc::validate_projection_views(normalized,low,down,inject,output);
    hc::validate_projection_views(normalized,low,down,std::nullopt,std::nullopt);
    hc::validate_projection_views(normalized,low,down,hc::View{down.address,163840},output);
    rejects([&]{hc::validate_projection_views(normalized,low,down,inject,std::nullopt);},"unpaired injection rejects");
    for (const auto bad : {hc::View{down.address+2,down.bytes},hc::View{down.address,3481599},
                          hc::View{std::numeric_limits<std::uintptr_t>::max()-15,3481600}})
        rejects([&]{hc::validate_projection_views(normalized,low,bad,inject,output);},"bad weight view rejects");
    rejects([&]{hc::validate_projection_views(normalized,{normalized.address,1280},down,inject,output);},
        "low cannot alias normalized", "HC next-tile mutable views overlap");
    rejects([&]{hc::validate_projection_views(normalized,low,down,inject,hc::View{low.address,16});},
        "injection cannot alias low", "HC next-tile mutable views overlap");
    rejects([&]{(void)hc::linear_extent(0,std::numeric_limits<std::uint64_t>::max(),320);},
        "geometry rejects before product can overflow");
}

void model_admission() {
    Fixture f;
    expect(f.manifest.tensors.size()==387,"canonical HC fixture has97norm/down/up and96inject tensors");
    const auto plan=f.plan(); expect(plan.mixers.size()==97,"all97 mixers admitted"); f.validate(plan);
    expect(plan.mixers.back().final,"last mixer is the final no-injection mixer");
    expect(hc::build_plan({}, {}, {}, {}, false).mixers.empty(),"all-off bypasses new model restrictions");
    hc::validate_runtime({}, {}, {}, {}, {}, {}, {});
    const auto down=plan.mixers.back().weights[1].index;
    const auto inject=plan.mixers[95].weights[3].index;
    for (const auto index : {down,inject}) {
        for (unsigned fault=0;fault<6;++fault) {
            auto bad=f; auto& tensor=bad.manifest.tensors[index];
            if(fault==0)tensor.ggml_type=1;
            if(fault==1)++tensor.dimensions[0];
            if(fault==2)tensor.absolute_offset+=8;
            if(fault==3)tensor.absolute_offset=std::numeric_limits<std::uint64_t>::max()-15;
            if(fault==4)--tensor.byte_size;
            if(fault==5)tensor.byte_size+=16;
            rejects([&]{(void)bad.plan();},"late producer file defect rejects before residency");
        }
        for(unsigned fault=0;fault<5;++fault) {
            auto bad=f;auto& owner=bad.resident.tensors[index];
            if(fault==0)owner.address+=8;
            if(fault==1)--owner.bytes;
            if(fault==2)owner.allocation_generation=0;
            if(fault==3)owner.storage=strix::RuntimeTensorStorage::file_mapped;
            if(fault==4)owner.address=std::numeric_limits<std::uintptr_t>::max()-15;
            rejects([&]{bad.validate(plan);},"late retained-owner defect rejects before state creation");
        }
    }
    auto missing=f;missing.access.accesses.pop_back();
    rejects([&]{(void)missing.plan();},"missing final up access rejects");
    auto duplicate=f;duplicate.access.accesses.push_back(duplicate.access.accesses.back());
    rejects([&]{(void)duplicate.plan();},"duplicate selected access rejects");
    auto stale=f;++stale.access.plan_identity;
    rejects([&]{stale.validate(plan);},"changed access snapshot rejects","HC next-tile prepared bindings changed");
    auto incomplete=plan;incomplete.mixers.pop_back();
    rejects([&]{f.validate(incomplete);},"incomplete planned mixer coverage rejects","HC next-tile prepared bindings changed");
    auto graph=f;graph.compiled.device_graph_inputs=true;
    rejects([&]{(void)graph.plan();},"graph setup rejects");
    auto no_generation=f;no_generation.memory.allocation_generation=0;
    rejects([&]{no_generation.validate(plan);},"missing arena generation rejects");
    const auto io=f.compiled.io.steps[plan.mixers.back().step_id];
    auto short_workspace=f;
    for(auto& allocation:short_workspace.compiled.arena.allocations)
        if(allocation.use==io.temporaries.front())allocation.bytes=42239;
    rejects([&]{short_workspace.validate(plan);},"actual final workspace must fit both slices",
        "HC next-tile view alignment or extent invalid");
    const auto temporary=strix::resolve_execution_buffer(f.compiled.execution_identity,f.compiled.arena,f.memory,io.temporaries.front());
    auto alias=f;alias.resident.tensors[plan.mixers.back().weights[0].index].address=temporary.address;
    rejects([&]{alias.validate(plan);},"normalization workspace cannot alias a retained norm",
        "HC next-tile workspace aliases retained weights");
}
} // namespace
int main() {
    try { views(); model_admission(); }
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    return failures ? 1 : 0;
}
