#pragma once

#include "catalog_cuda_decode.hpp"
#include "cuda_program_recipe.hpp"

#include <cmath>

namespace strix::exec::cuda_decode {

// Project only catalogue entries that the caller can lower for this pair.
// An unpriced entry requires the caller's explicit opt-in.
inline void project_exact_epilogue(graph::PlanStage& tail, const Catalog& catalog, std::string_view producer,
                                   graph::Form form, bool enabled) {
    tail.exact_epilogue_head = tail.exact_epilogue_entry = {};
    tail.require_exact_epilogue = true;
    tail.form = {form, 0};
    if (!enabled) return;
    constexpr auto fma = OperandClass::f32_fma;
    constexpr auto mma = OperandClass::bf16_mma;
    constexpr auto linear = graph::Op::linear;
    constexpr weight::FormatId bf16{"plain.bf16", 1};
    KernelKey fused{linear, {}, bf16, fma, RowsClass::r1};
    if (producer == "cuda-sm120-bf16-r1-fp32-group-v1") {
        if (form == graph::Form::store_bf16) fused.form = "r1.bf16_image.fp32_round";
        if (form == graph::Form::softcap) fused.form = "r1.bf16_image.fp32_softcap";
    } else if (producer == "cuda-sm120-bf16-r1-mma3-group-v1" && form == graph::Form::store_bf16) {
        fused.form = "r1.bf16_image.group_round";
        fused.act = mma;
    } else if (producer == "cuda-sm120-causal-convolution-r1-v1" && form == graph::Form::store_bf16) {
        fused.op = graph::Op::elementwise;
        fused.form = "causal_convolution.decode_round";
        fused.weight = {};
    }
    if (fused.form.empty()) return;
    const auto found = catalog.find(fused, device::ArchMask::sm120);
    if (found.size() != 1 || found.front()->exact != lever::Exactness::bitwise || !found.front()->capturable) return;
    tail.exact_epilogue_head = producer;
    tail.exact_epilogue_entry = found.front()->identity;
}

// The adapter calls this at the producer site and omits the tail only after
// successful emission. Its declaration must account for all scratch uses.
template <class Bind, class Args>
bool emit_exact_epilogue(CudaProgramRecipe& recipe, const graph::LaunchPlan& plan, const graph::Launch& launch,
                         graph::Form form, const Args& args, bool pdl, const Bind& bind_leaf) {
    if (launch.rows != 1 || launch.jobs != 1 || launch.epilogue.size() != 1) return false;
    const auto& chain = plan.chains.at(launch.epilogue_chain[0]);
    if (chain.count != 1 || chain.links[0] != graph::Link{form, 0}) return false;
    auto binding = bind_leaf(args);
    if (binding.identity != launch.entry) return false;
    binding.chain_identity = graph::form_name(form);
    recipe.kernel(std::move(binding), pdl);
    return true;
}

template <class Bind>
bool lower_exact_epilogue(CudaProgramRecipe& recipe, const graph::LaunchPlan& plan, const graph::Launch& launch,
                           const Bf16DenseGroup& head, std::span<const Bf16Round> tails, bool pdl,
                           const Bind& bind_leaf) {
    // One D3 value can name a bundle produced by a native homogeneous group.
    // Its cast tail covers every output of that bundle in projection order.
    if (head.mode != Bf16DenseMode::mma3 || head.count == 0 || head.count > head.projections.size() ||
        head.count != tails.size()) return false;
    auto rounded = head;
    for (std::size_t i = 0; i < tails.size(); ++i) {
        auto& projection = rounded.projections[i];
        if (projection.round_bf16 || projection.output != tails[i].values ||
            projection.outputs != tails[i].elements) return false;
        projection.round_bf16 = true;
    }
    rounded.mode = Bf16DenseMode::mma3_round;
    return emit_exact_epilogue(recipe, plan, launch, graph::Form::store_bf16, rounded, pdl, bind_leaf);
}

template <class Bind>
bool lower_exact_epilogue(CudaProgramRecipe& recipe, const graph::LaunchPlan& plan, const graph::Launch& launch,
                           const Bf16DenseGroup& head, const Bf16Round& tail, bool pdl, const Bind& bind_leaf) {
    if (head.mode == Bf16DenseMode::mma3)
        return lower_exact_epilogue(recipe, plan, launch, head, std::span(&tail, 1), pdl, bind_leaf);
    if (head.count != 1 || head.projections[0].round_bf16 ||
        head.projections[0].output != tail.values || head.projections[0].outputs != tail.elements) return false;
    if (head.mode == Bf16DenseMode::fp32)
        return emit_exact_epilogue(recipe, plan, launch, graph::Form::store_bf16,
                                   Bf16Fp32Epilogue{head, graph::Form::store_bf16}, pdl, bind_leaf);
    return false;
}

template <class Bind>
bool lower_exact_epilogue(CudaProgramRecipe& recipe, const graph::LaunchPlan& plan, const graph::Launch& launch,
                           const Bf16DenseGroup& head, const Bf16Softcap& tail, bool pdl, const Bind& bind_leaf) {
    if (head.count != 1 || head.mode != Bf16DenseMode::fp32 || head.projections[0].round_bf16 ||
        head.projections[0].output != tail.values || head.projections[0].outputs != tail.elements ||
        !(tail.cap > 0.0F) || !std::isfinite(tail.cap)) return false;
    return emit_exact_epilogue(recipe, plan, launch, graph::Form::softcap,
                               Bf16Fp32Epilogue{head, graph::Form::softcap, tail.cap}, pdl, bind_leaf);
}

template <class Bind>
bool lower_exact_epilogue(CudaProgramRecipe& recipe, const graph::LaunchPlan& plan, const graph::Launch& launch,
                           const CausalConvolution& head, const Bf16Round& tail, bool pdl, const Bind& bind_leaf) {
    if (head.round_bf16 || head.output != tail.values || head.plan.channels != tail.elements) return false;
    auto rounded = head;
    rounded.round_bf16 = true;
    return emit_exact_epilogue(recipe, plan, launch, graph::Form::store_bf16,
                               RoundedCausalConvolution{rounded}, pdl, bind_leaf);
}

}  // namespace strix::exec::cuda_decode
