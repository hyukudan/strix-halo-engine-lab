#pragma once
// F4 forms "eager selected decode": the first and last passes of a three-pass (scores, softmax, values) decode over a
// selected key list; the middle pass is eager_softmax.hpp. A query row has `attention_heads` heads of `head_dim`
// features; heads share KV heads in groups of attention_heads / kv_heads; `selected[slot]`, slot < count, are the token
// indices of the keys; scores / probabilities live at scores[head * maximum_selected + slot].
//
// Contract (bits depend on exactly this; host rule eager_selected_rule.hpp):
//   score_slot   one block of Threads lanes per (head, slot): flat block x = head * D + slot with D = maximum_selected
//                when control.padded() (graph-padded grid), else count. Lane f takes the single product
//                query(head, f) * K(f) (+0 for f >= head_dim); the products meet in the LDS halving
//                tree over Threads lanes (level h = Threads/2 .. 1, lane f < h: s[f] += s[f + h], one barrier per
//                level); lane 0 calls fail() if checks() and the root is not finite, then stores root / sqrtf(head_dim).
//   value_chain  one block per head, lane f < head_dim one feature: sum = fmaf(p_slot, V(token_slot), sum)
//                in ascending slot order from +0 (loads requested 8 slots ahead of the chain: latency only), then
//                epilogue.store(head, f, sum).
//   GatedRoundedOutput: attn = round(sum), gate = round(1 / (1 + expf(-gate(head, f)))), out = round(attn * gate)
//                stored to image[head * head_dim + f]; fail() if checks() and out is not finite.
// Policies (the only things an engine supplies; every member is called uniformly by the block):
//   Ops (operand layout of the engine, data only):
//     float query(const Geometry&, std::uint32_t head, std::uint32_t f) const;
//     K key_reader(const Geometry&, std::uint32_t kv_width, std::uint32_t token, std::uint32_t kv_head) const;
//                                                                       K(f) -> float (key feature f of that token)
//     V value_reader(const Geometry&, std::uint32_t kv_width, std::uint32_t kv_head, std::uint32_t f) const;
//                                                                       V(token) -> float (value feature f)
//     float gate(const Geometry&, std::uint32_t head, std::uint32_t f) const;           (GatedRoundedOutput only)
//   Control: the eager_softmax.hpp Control (count, checks, fail, round) plus bool padded() const.
// LDS: the caller owns `scratch` (Threads floats), so each kernel keeps its LDS layout.
// Home: shared (attn_core): no model names, no layout constants; Threads is a template parameter.
#include <cstdint>

#include <hip/hip_runtime.h>

namespace strix::attn_core::eager_selected {

struct Geometry {
    std::uint32_t attention_heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    __device__ __forceinline__ std::uint32_t kv_width() const { return kv_heads * head_dim; }
};

template <unsigned Threads, class Ops, class Control>
__device__ __forceinline__ void score_slot(float* scratch, const std::uint32_t* selected, std::uint32_t selected_count,
                                           std::uint32_t maximum_selected, const Geometry& g, float* scores,
                                           const Ops& ops, const Control& control) {
    static_assert(Threads >= 1U && (Threads & (Threads - 1U)) == 0U, "eager_selected: Threads must be a power of two");
    if (!control.count(selected_count)) return;
    const auto flat = static_cast<std::uint32_t>(blockIdx.x);
    const auto head = flat / (control.padded() ? maximum_selected : selected_count);
    const auto selected_slot = flat % (control.padded() ? maximum_selected : selected_count);
    if (head >= g.attention_heads || selected_slot >= selected_count) {
        return;
    }

    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const auto token = selected[selected_slot];
    const auto kv_head = head / (g.attention_heads / g.kv_heads);
    const auto kv_width = g.kv_width();
    const auto key_at = ops.key_reader(g, kv_width, token, kv_head);

    float product = 0.0F;
    if (feature < g.head_dim) product = ops.query(g, head, feature) * key_at(feature);
    scratch[feature] = product;
    __syncthreads();

    for (std::uint32_t stride = Threads / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] += scratch[feature + stride];
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (control.checks() && !isfinite(scratch[0])) control.fail();
        scores[static_cast<std::uint64_t>(head) * maximum_selected + selected_slot] =
            scratch[0] / sqrtf(static_cast<float>(g.head_dim));
    }
}

template <class Ops, class Epilogue, class Control>
__device__ __forceinline__ void value_chain(const std::uint32_t* selected, const float* scores,
                                            std::uint32_t selected_count, std::uint32_t maximum_selected,
                                            const Geometry& g, const Ops& ops, const Epilogue& epilogue,
                                            const Control& control) {
    // The chain is contracted (one fmaf per slot) whatever the translation unit's default contraction.
#pragma clang fp contract(fast)
    if (!control.count(selected_count)) return;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (head >= g.attention_heads || feature >= g.head_dim) {
        return;
    }
    const auto kv_head = head / (g.attention_heads / g.kv_heads);
    const auto kv_width = g.kv_width();

    const auto* head_scores = scores + static_cast<std::uint64_t>(head) * maximum_selected;
    const auto value_at = ops.value_reader(g, kv_width, kv_head, feature);
    float sum = 0.0F;
    std::uint32_t slot = 0;
    for (; slot + 8U <= selected_count; slot += 8U) {
        std::uint32_t token[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) token[i] = selected[slot + i];
        float value[8], score[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) { value[i] = value_at(token[i]); score[i] = head_scores[slot + i]; }
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) sum += score[i] * value[i];
    }
    for (; slot < selected_count; ++slot) {
        const float value = value_at(selected[slot]);
        sum += head_scores[slot] * value;
    }
    epilogue.store(g, head, feature, sum, control);
}

// Sigmoid-gated rounded output (gated attention): Image is any indexable store of floats (image[i] = value).
template <class Image, class Ops>
struct GatedRoundedOutput {
    const Ops& ops;
    Image image;
    template <class Control>
    __device__ __forceinline__ void store(const Geometry& g, std::uint32_t head, std::uint32_t feature, float sum,
                                          const Control& control) const {
        const float attn = control.round(sum);
        const float gate = control.round(1.0F / (1.0F + expf(-ops.gate(g, head, feature))));
        image[static_cast<std::uint64_t>(head) * g.head_dim + feature] = control.round(attn * gate);
        if (control.checks() && !isfinite(control.round(attn * gate))) control.fail();
    }
};

}  // namespace strix::attn_core::eager_selected
