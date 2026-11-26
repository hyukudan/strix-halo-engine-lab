#pragma once

#include "cuda_program_recipe.hpp"
#include "row_epilogue_plan.hpp"

namespace strix::exec {

// A failed match leaves the recipe untouched, so the caller can use the
// incumbent bindings. The selected body still declares its own PDL wait.
inline bool emit_row_epilogue(CudaProgramRecipe& recipe, const RowEpiloguePlan& plan, std::size_t index,
                              KernelBinding binding) {
    const auto* entry = plan.selections.at(index).entry;
    if (!entry || binding.identity != entry->identity) return false;
    if (entry->exact == lever::Exactness::bitwise) {
        if (binding.exact_identity.empty()) return false;
    } else {
        binding.exact_identity.clear();
    }
    const auto chain = plan.graph.launches.at(index).epilogue_chain[0];
    binding.chain_identity = std::string(entry->identity) + "/" + std::to_string(plan.graph.chains.identity(chain));
    binding.capturable = binding.capturable && entry->capturable;
    recipe.kernel(std::move(binding));
    return true;
}

}  // namespace strix::exec
