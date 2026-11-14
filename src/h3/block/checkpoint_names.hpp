#pragma once
#include <string>

namespace strix::h3::block {
enum class CheckpointNames { unchanged, image_dit, image_vae };
// Both raw and pruned H3 use blocks.*.attn.{qkv_proj,out_proj},
// blocks.*.mlp.{fc1,fc2}, condition_proj, token_refiner, final_layer.
// Pruning substitutes adaln_t_table for time_embedder; that is a policy change,
// never a fabricated alias for missing time_embedder tensors. Wrappers are
// removed once at the reader boundary, without rewriting/copying checkpoint data.
inline std::string checkpoint_name(std::string name, CheckpointNames policy) {
    if (policy==CheckpointNames::unchanged) return name;
    if (policy==CheckpointNames::image_dit) {
        for (const auto* p:{"model.diffusion_model.","diffusion_model.","transformer."}) {
            const std::string prefix=p;
            if (name.starts_with(prefix)) return name.substr(prefix.size());
        }
    } else {
        for (const auto* p:{"first_stage_model.","vae."}) {
            const std::string prefix=p;
            if (name.starts_with(prefix)) return name.substr(prefix.size());
        }
    }
    return name;
}
} // namespace strix::h3::block
