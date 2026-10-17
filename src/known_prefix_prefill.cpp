#include "strix/known_prefix_prefill.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix {
namespace {
void require(bool condition,const char* message) {if(!condition) throw std::invalid_argument(message);}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("known prefill storage overflow");
    return a+b;
}
std::uint64_t multiply(std::uint64_t a,std::uint64_t b) {
    if(b&&a>UINT64_MAX/b) throw std::overflow_error("known prefill storage overflow");
    return a*b;
}
void hash(std::uint64_t& h,std::uint64_t value) {
    for(unsigned shift=0;shift<64;shift+=8) {h^=(value>>shift)&255U;h*=1099511628211ULL;}
}
void hash_arena(std::uint64_t& h,const ExecutionArenaPlan& arena) {
    hash(h,arena.alignment);hash(h,arena.total_bytes);hash(h,arena.naive_nonaliased_bytes);hash(h,arena.allocations.size());
    for(const auto& a:arena.allocations) {
        hash(h,static_cast<std::uint64_t>(a.use.buffer.buffer_class));hash(h,a.use.buffer.slot);hash(h,a.use.generation);
        hash(h,a.first_step);hash(h,a.last_step);hash(h,a.offset);hash(h,a.bytes);
    }
}
std::uint64_t metadata_payload(std::uint32_t rows,std::size_t allocations) {
    // Declared arena-vector payload, not allocator overhead/RSS. The closed
    // selection plan remains bounded at eight rows independently of T.
    return add(multiply(rows,sizeof(ExecutionArenaPlan)),add(multiply(allocations,sizeof(std::uint64_t)),
        multiply(multiply(static_cast<std::uint64_t>(rows)+1U,allocations),sizeof(ExecutionArenaAllocation))));
}
std::pair<std::uint64_t,std::uint64_t> identities(const CompiledKnownPrefixPrefill& plan) {
    std::uint64_t numerical=14695981039346656037ULL;
    for(char c:std::string_view("known-prefix-prefill-step-major-tile8-v1")) {numerical^=static_cast<unsigned char>(c);numerical*=1099511628211ULL;}
    hash(numerical,plan.tile8.window_kernel_plan_identity);hash(numerical,plan.logical_rows);hash(numerical,8U);
    hash(numerical,static_cast<std::uint64_t>(plan.output_policy));
    for(auto row:plan.logit_row_indices) hash(numerical,row);
    auto execution=numerical;hash(execution,plan.tile8.window_execution_identity);hash_arena(execution,plan.arena.physical);
    for(auto stride:plan.arena.row_stride) hash(execution,stride);
    for(const auto& row:plan.arena.row_view) hash_arena(execution,row);
    require(execution&&numerical&&execution!=plan.tile8.window_execution_identity&&numerical!=plan.tile8.window_kernel_plan_identity,
        "known prefill identity is zero or collides with the closed window");return {execution,numerical};
}
}
CompiledKnownPrefixPrefill compile_known_prefix_prefill(const ModelContract& model,const CompiledSerialExecution& serial,
    const ModelManifest& manifest,const ExecutionTensorAccessPlan& access,const KnownPrefillRequest& request) {
    require(request.logical_rows>=16U&&request.logical_rows<=2048U,"known prefill rows must be within 16 and 2048");
    require(request.workspace_budget_bytes&&request.host_metadata_payload_budget_bytes,"known prefill requires explicit workspace and host metadata budgets");
    validate_known_window_output_policy(request.output_policy);
    const auto payload=metadata_payload(request.logical_rows,serial.arena.allocations.size());
    require(payload<=request.host_metadata_payload_budget_bytes,"known prefill host metadata payload exceeds its budget");
    CompiledKnownPrefixPrefill plan;plan.logical_rows=request.logical_rows;plan.output_policy=request.output_policy;
    plan.workspace_budget_bytes=request.workspace_budget_bytes;plan.host_metadata_payload_budget_bytes=request.host_metadata_payload_budget_bytes;
    plan.host_metadata_payload_bytes=payload;
    KnownWindowRequest tile;tile.logical_rows=8;tile.r2_shared_expert=request.r2_shared_expert;
    tile.r2_gdn=request.r2_gdn;tile.r2_gdn_f32_gates=request.r2_gdn_f32_gates;tile.routed_expert_union=request.routed_expert_union;
    tile.router_rows=request.router_rows;tile.combine_inject_rows=request.combine_inject_rows;
    // Preserve complete tile admission even when this T request omits heads.
    plan.tile8=compile_known_prefix_window(model,serial,manifest,access,tile);
    require(known_window_has_promoted_steps(plan.tile8),"known prefill requires an actual admitted multi-row projection entry");
    // Every stripe/interval scales uniformly from the closed eight-row arena.
    // Its offsets are multiples of eight alignments, including odd T shapes.
    require(plan.tile8.arena.physical.total_bytes%8U==0U,"known prefill tile arena cannot be uniformly scaled");
    require(multiply(plan.tile8.arena.physical.total_bytes/8U,request.logical_rows)<=request.workspace_budget_bytes,
        "known prefill workspace exceeds its explicit arena budget");
    plan.arena=build_known_prefill_arena(serial.arena,serial.arena.alignment,request.logical_rows);
    require(plan.arena.physical.total_bytes<=request.workspace_budget_bytes,"known prefill workspace exceeds its explicit arena budget");
    for(std::uint32_t row=0;row<request.logical_rows;++row)
        if(known_window_produces_logits(request.output_policy,row,request.logical_rows)) plan.logit_row_indices.push_back(row);
    const auto [execution,numerical]=identities(plan);plan.prefill_execution_identity=execution;plan.prefill_kernel_plan_identity=numerical;
    return plan;
}
void validate_compiled_known_prefix_prefill(const CompiledKnownPrefixPrefill& plan,const CompiledSerialExecution& serial) {
    require(plan.logical_rows>=16U&&plan.logical_rows<=2048U,"known prefill rows must be within 16 and 2048");
    validate_compiled_known_prefix_window(plan.tile8,serial);
    require(plan.tile8.logical_rows==8U&&plan.tile8.output_policy==KnownWindowOutputPolicy::all_rows&&
        known_window_has_promoted_steps(plan.tile8),"known prefill selection must retain its complete closed tile-eight contract");
    validate_known_window_output_policy(plan.output_policy);
    require(plan.workspace_budget_bytes&&plan.arena.physical.total_bytes<=plan.workspace_budget_bytes,
        "known prefill workspace exceeds its explicit arena budget");
    require(plan.host_metadata_payload_bytes==metadata_payload(plan.logical_rows,serial.arena.allocations.size())&&
        plan.host_metadata_payload_budget_bytes&&plan.host_metadata_payload_bytes<=plan.host_metadata_payload_budget_bytes,
        "known prefill host metadata payload budget differs");
    const auto& tile=plan.tile8.arena;
    require(tile.physical.total_bytes%8U==0U&&tile.physical.naive_nonaliased_bytes%8U==0U&&
        plan.arena.physical.alignment==tile.physical.alignment&&
        plan.arena.physical.total_bytes==multiply(tile.physical.total_bytes/8U,plan.logical_rows)&&
        plan.arena.physical.naive_nonaliased_bytes==multiply(tile.physical.naive_nonaliased_bytes/8U,plan.logical_rows)&&
        plan.arena.physical.allocations.size()==tile.physical.allocations.size()&&plan.arena.row_stride==tile.row_stride&&
        plan.arena.row_view.size()==plan.logical_rows,"known prefill physical arena or stripe differs");
    for(std::size_t i=0;i<tile.physical.allocations.size();++i) {
        const auto& old=tile.physical.allocations[i];const auto& actual=plan.arena.physical.allocations[i];
        require(old.offset%8U==0U&&actual.use==old.use&&actual.first_step==old.first_step&&actual.last_step==old.last_step&&
            actual.offset==multiply(old.offset/8U,plan.logical_rows)&&actual.bytes==multiply(tile.row_stride[i],plan.logical_rows),
            "known prefill physical allocation differs");
        for(std::uint32_t row=0;row<plan.logical_rows;++row) {
            const auto& view=plan.arena.row_view[row];
            require(view.alignment==plan.arena.physical.alignment&&view.total_bytes==plan.arena.physical.total_bytes&&
                view.naive_nonaliased_bytes==plan.arena.physical.naive_nonaliased_bytes&&view.allocations.size()==tile.physical.allocations.size(),
                "known prefill row arena header differs");
            const auto& a=view.allocations[i];
            require(a.use==actual.use&&a.first_step==actual.first_step&&a.last_step==actual.last_step&&
                a.offset==add(actual.offset,multiply(row,tile.row_stride[i]))&&a.bytes==tile.row_view[0].allocations[i].bytes,
                "known prefill row arena differs");
        }
    }
    std::vector<std::uint32_t> logits;
    for(std::uint32_t row=0;row<plan.logical_rows;++row)
        if(known_window_produces_logits(plan.output_policy,row,plan.logical_rows)) logits.push_back(row);
    require(logits==plan.logit_row_indices,"known prefill output row mapping differs");
    const auto [execution,numerical]=identities(plan);
    require(execution==plan.prefill_execution_identity&&numerical==plan.prefill_kernel_plan_identity&&
        execution!=serial.execution_identity,"known prefill execution or numerical identity differs");
}
} // namespace strix
