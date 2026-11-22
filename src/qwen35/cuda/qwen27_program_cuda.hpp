#pragma once

#include "exec/catalog_cuda_step_rows.hpp"
#include "exec/cuda_program_recipe.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <variant>

namespace strix::qwen35 {
namespace cuda_program {

namespace leaf = exec::cuda_step_rows;

struct GdnVisit {
    std::array<leaf::Matrix, 4> inputs;  // QKV, gate, beta, alpha.
    std::array<float*, 4> projected;
    leaf::Gdn window;
    leaf::Matrix output;
};

struct AttentionVisit {
    std::array<leaf::Matrix, 3> inputs;  // Q/gate, K, V.
    const float* query_gain;
    const float* key_gain;
    void* key_cache;
    void* value_cache;
    leaf::Matrix output;
};

struct Visit {
    const float* input_gain;
    const float* post_gain;
    std::variant<GdnVisit, AttentionVisit> mixer;
    leaf::Matrix gate, up, down;
};

struct DecodeStorage {
    std::uint32_t hidden, vocabulary, rotary_dim;
    float epsilon;
    leaf::AttentionShape attention;
    leaf::Matrix embedding, head;
    const float* output_gain;
    const std::uint32_t* ids;
    const std::uint32_t* position;
    const float* cosines;
    const float* sines;
    float *x, *xn, *q_gate, *q, *k, *v, *attention_output, *scratch, *ffn_hidden, *logits;
    std::uint32_t* top;
    void* argmax_workspace;
};

// The capture covers trunk only. The resident owner orders descriptor/scatter,
// commit and reads on the surrounding stream.
inline exec::CudaProgramRecipe lower(const DecodeStorage& s, std::span<const Visit> visits,
                                     const leaf::Bindings& bind, bool programmatic_edges) {
    exec::CudaProgramRecipe recipe;
    const leaf::View hidden{1, s.hidden, s.hidden, s.hidden};
    const auto& attn = s.attention;
    const leaf::View query{attn.heads, attn.width, attn.heads * attn.width, attn.width};
    const leaf::View gate{attn.heads, attn.width, 2 * query.row_stride, 2 * query.group_stride};
    const leaf::View kv{attn.kv_heads, attn.width, attn.kv_heads * attn.width, attn.width};
    const auto add = [&](auto&&... args) {
        recipe.kernel(bind.bind(args...), programmatic_edges);
    };
    const auto rms = [&](const float* input, const float* gain, float* output) {
        add(leaf::Rms{input, hidden, gain, output, hidden, s.epsilon});
    };
    const auto project = [&](std::span<const leaf::Matrix> weights, std::span<float* const> outputs,
                             const float* input, const float* residual = nullptr, const leaf::Matrix* up = nullptr) {
        leaf::project(recipe, bind, programmatic_edges, weights, outputs, input, residual, up);
    };
    const auto single = [&](const leaf::Matrix& weight, const float* input, float* output, const float* residual = nullptr) {
        project(std::span(&weight, 1), std::span(&output, 1), input, residual);
    };

    recipe.stage("embedding");
    add(leaf::Embedding{s.embedding, s.ids, s.x});
    for (std::size_t index = 0; index < visits.size(); ++index) {
        const auto& visit = visits[index];
        recipe.visit(static_cast<std::int32_t>(index));
        recipe.stage("input_norm");
        rms(s.x, visit.input_gain, s.xn);
        if (const auto* gdn = std::get_if<GdnVisit>(&visit.mixer)) {
            recipe.stage("gdn");
            project(gdn->inputs, gdn->projected, s.xn);
            add(gdn->window, leaf::GdnPhase::convolution);
            add(gdn->window, leaf::GdnPhase::recurrence);
            single(gdn->output, gdn->window.output, s.x, s.x);
        } else {
            recipe.stage("attention");
            const auto& a = std::get<AttentionVisit>(visit.mixer);
            const std::array outputs{s.q_gate, s.k, s.v};
            project(a.inputs, outputs, s.xn);
            add(leaf::Rms{s.q_gate, gate, a.query_gain, s.q, query, s.epsilon});
            add(leaf::Rms{s.k, kv, a.key_gain, s.k, kv, s.epsilon});
            add(leaf::Rope{s.q, query, s.rotary_dim, s.cosines, s.sines, s.position});
            add(leaf::Rope{s.k, kv, s.rotary_dim, s.cosines, s.sines, s.position});
            add(leaf::KvStore{attn, s.k, s.v, a.key_cache, a.value_cache, s.position});
            const leaf::Attention args{attn, s.q, query, s.q_gate + attn.width, gate, a.key_cache, a.value_cache,
                                       s.position, s.attention_output, s.scratch};
            add(args, leaf::AttentionPhase::scores);
            add(args, leaf::AttentionPhase::values);
            add(args, leaf::AttentionPhase::combine);
            single(a.output, s.attention_output, s.x, s.x);
        }
        recipe.stage("ffn");
        rms(s.x, visit.post_gain, s.xn);
        project(std::span(&visit.gate, 1), std::span(&s.ffn_hidden, 1), s.xn, nullptr, &visit.up);
        single(visit.down, s.ffn_hidden, s.x, s.x);
    }
    recipe.visit(-1);
    recipe.stage("head");
    rms(s.x, s.output_gain, s.xn);
    single(s.head, s.xn, s.logits);
    const leaf::Pick pick{s.logits, s.argmax_workspace, s.vocabulary, s.top};
    add(pick, leaf::Leaf::pick_partial);
    add(pick, leaf::Leaf::pick_final);
    return recipe;
}

}  // namespace cuda_program
}  // namespace strix::qwen35
