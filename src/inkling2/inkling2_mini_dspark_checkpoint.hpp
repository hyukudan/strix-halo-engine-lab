#pragma once

#include "inkling2_mini_dspark_taps.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/weight/tensor_access.hpp"

#include <array>
#include <initializer_list>
#include <utility>

namespace strix::inkling2 {

// Shape and tensor names of mtp.safetensors for the Mini checkpoint's
// rnn_hidden drafter. Use the shared reader for validation and payload slices.
struct MiniDSparkCheckpoint {
    static constexpr unsigned layers = 3U;
    static constexpr unsigned block = 8U;
    static constexpr unsigned intermediate = 9216U;
    static constexpr unsigned markov_rank = 256U;
    static constexpr unsigned context_window = 1024U;
    static constexpr float epsilon = 1e-6F;
    static constexpr float rope_theta = 1e6F;
    static constexpr std::array<unsigned, 2> local_layers{0U, 2U};
    static constexpr auto target_layers = MiniDSparkTapPlan::visits;

    static constexpr unsigned window(unsigned layer) {
        return layer == local_layers[0] || layer == local_layers[1] ? context_window : 0U;
    }

    static constexpr spec::BlockPolicy policy() {
        spec::BlockPolicy result;
        result.block = block;
        result.ring = context_window;
        result.draft_layers = layers;
        result.target_layer_count = MiniDSparkTapPlan::count;
        for (unsigned i = 0; i < result.target_layer_count; ++i)
            result.target_layers[i] = static_cast<std::uint16_t>(target_layers[i]);
        return result;
    }

    static weight::TensorSchema schema() {
        static const auto tensors = [] {
            std::vector<weight::TensorSpec> specs;
            const auto bf16 = [&](std::string name, std::initializer_list<std::uint64_t> shape) {
                specs.push_back({std::move(name), "BF16", shape});
            };
            bf16("fuse_weight", {target_layers.size(), target_layers.size()});
            for (unsigned i = 0; i < layers; ++i) {
                const auto prefix = "layers." + std::to_string(i) + ".";
                bf16(prefix + "dflare_norm.weight", {kHiddenWidth});
                bf16(prefix + "input_layernorm.weight", {kHiddenWidth});
                bf16(prefix + "post_attention_layernorm.weight", {kHiddenWidth});
                bf16(prefix + "mlp.down_proj.weight", {kHiddenWidth, intermediate});
                bf16(prefix + "mlp.gate_proj.weight", {intermediate, kHiddenWidth});
                bf16(prefix + "mlp.up_proj.weight", {intermediate, kHiddenWidth});
                const auto attn = prefix + "self_attn.";
                bf16(attn + "q_proj.weight", {kQueryWidth, kHiddenWidth});
                bf16(attn + "o_proj.weight", {kHiddenWidth, kQueryWidth});
                bf16(attn + "q_norm.weight", {kHeadWidth});
                bf16(attn + "k_norm.weight", {kHeadWidth});
                bf16(attn + "k_norm_ctx.weight", {kHeadWidth});
                bf16(attn + "k_proj.weight", {kKvWidth, kHiddenWidth});
                bf16(attn + "v_proj.weight", {kKvWidth, kHiddenWidth});
                bf16(attn + "k_proj_ctx.weight", {kKvWidth, kHiddenWidth});
                bf16(attn + "v_proj_ctx.weight", {kKvWidth, kHiddenWidth});
            }
            constexpr auto rnn_width = 3U * markov_rank;
            constexpr auto rnn_inputs = 5120U;
            bf16("markov_head.dflare_norm.weight", {kHiddenWidth});
            bf16("markov_head.hidden_out.bias", {kHiddenWidth});
            bf16("markov_head.hidden_out.weight", {kHiddenWidth, markov_rank});
            bf16("markov_head.markov_w1.weight", {kVocabulary, markov_rank});
            bf16("markov_head.rnn_proj.bias", {rnn_width});
            bf16("markov_head.rnn_proj.weight", {rnn_width, rnn_inputs});
            bf16("mask_embed", {kHiddenWidth});
            bf16("norm.weight", {kHiddenWidth});
            return specs;
        }();
        return {weight::Container::safetensors, "safetensors@1", tensors, {}};
    }
};

}  // namespace strix::inkling2
