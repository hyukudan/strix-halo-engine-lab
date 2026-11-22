#include "catalog_cuda_step_rows.hpp"

#include "cuda_program_recipe.hpp"
#include "strix/core/rows.hpp"
#include "strix/exec/catalog.hpp"
#include "strix/head_argmax.hpp"
#include "strix/weight/registry.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::exec::cuda_step_rows {
namespace {

using graph::Op;
using enum OperandClass;

weight::FormatId weight_id(Format format) {
    switch (format) {
    case Format::bf16: return weight::kBf16WeightStorage;
    case Format::q8: return weight::kQ8WeightStorage;
    case Format::f32: return {"plain.f32", 1};
    default: throw std::invalid_argument("unsupported CUDA row weight format");
    }
}

Grid grid(std::uint32_t blocks, std::uint32_t threads) {
    return {{blocks, 1, 1}, {threads, 1, 1}, 0};
}

template <class Args>
KernelBinding bind_leaf(const Thunks& thunks, Leaf leaf, Op op, FormId form, weight::FormatId format,
                        OperandClass act, Grid geometry, const Args& args) {
    const KernelKey key{op, form, format, act, RowsClass::r1};
    const auto entries = Catalog::builtin(Backend::cuda).find(key, device::ArchMask::sm120);
    if (entries.size() != 1 || !entries.front()->capturable)
        throw std::invalid_argument("missing capturable CUDA row catalogue entry");
    KernelBinding binding;
    binding.fn = thunks.at(static_cast<std::size_t>(leaf));
    binding.identity = entries.front()->identity;
    binding.grid = geometry;
    binding.predicted.launches = 1;
    const auto bytes = std::as_bytes(std::span(&args, 1));
    binding.args.assign(bytes.begin(), bytes.end());
    return binding;
}

bool can_group(std::span<const Matrix> weights) {
    return !weights.empty() && std::ranges::all_of(weights, [&](const Matrix& w) {
        return w.format == weights.front().format && w.in == weights.front().in;
    });
}

bool mma3_supported(const Matrix& w) {
    return (w.format == Format::q8 || w.format == Format::bf16) && w.in % 32 == 0 &&
        w.out % core::rows::kTileOut == 0;
}

}  // namespace

KernelBinding Bindings::bind(const Embedding& a) const {
    if (!a.weight.data || !a.ids || !a.output || a.weight.in == 0 || a.weight.out == 0)
        throw std::invalid_argument("invalid CUDA row embedding");
    const auto form = a.weight.format == Format::bf16 ? "embedding.bf16_to_f32" : "embedding.q8_to_f32";
    return bind_leaf(thunks_, Leaf::embedding, Op::gather, form, weight_id(a.weight.format), f32_fma, grid(1, 256), a);
}

KernelBinding Bindings::bind(const Rms& a) const {
    if (!a.input || !a.gain || !a.output || a.input_view.groups == 0 || a.input_view.width == 0 ||
        a.input_view.width > 8192 || a.input_view.groups != a.output_view.groups ||
        a.input_view.width != a.output_view.width)
        throw std::invalid_argument("invalid CUDA row RMS");
    return bind_leaf(thunks_, Leaf::rms, Op::norm, "norm.rms", {"plain.f32", 1}, f32_fma,
                     grid(a.input_view.groups, 256), a);
}

KernelBinding Bindings::bind(const Dense& a) const {
    if (a.count == 0 || a.count > a.weights.size() || (a.count != 1 && (a.residual || a.up)) || !a.input)
        throw std::invalid_argument("invalid CUDA row GEMV group");
    const auto weights = std::span(a.weights).first(a.count);
    if (!can_group(weights) || (a.up && (!a.mma || a.residual)))
        throw std::invalid_argument("unsupported CUDA row GEMV group");
    std::uint64_t outputs = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto& w = weights[i];
        const auto divisor = w.format == Format::q8 ? 32U : 8U;
        if (!w.data || !a.outputs[i] || w.in == 0 || w.in % divisor != 0 || w.out == 0 ||
            (a.mma && !mma3_supported(w)))
            throw std::invalid_argument("unsupported CUDA row GEMV dimensions");
        outputs += w.out;
    }
    constexpr std::uint32_t warps_per_fma_block = 8;
    if (outputs > std::numeric_limits<std::uint32_t>::max() - warps_per_fma_block)
        throw std::invalid_argument("CUDA row GEMV grid overflow");
    const auto blocks = static_cast<std::uint32_t>(a.mma ? outputs / core::rows::kTileOut :
        (outputs + warps_per_fma_block - 1) / warps_per_fma_block);
    return bind_leaf(thunks_, Leaf::dense, Op::linear, a.up ? "rows.gate_up_silu" : "rows.multi",
                     weight_id(weights.front().format), a.mma ? bf16_mma : f32_fma, grid(blocks, 256), a);
}

KernelBinding Bindings::bind(const Rope& a) const {
    if (!a.image || !a.cosines || !a.sines || !a.position || a.view.groups == 0 ||
        a.rotary_width == 0 || a.rotary_width % 2 != 0 || a.rotary_width > a.view.width)
        throw std::invalid_argument("invalid CUDA row RoPE");
    return bind_leaf(thunks_, Leaf::rope, Op::elementwise, "rope.device_position", {}, f32_fma,
                     grid(a.view.groups, 32), a);
}

KernelBinding Bindings::bind(const KvStore& a) const {
    if (!a.key || !a.value || !a.key_cache || !a.value_cache || !a.position || a.shape.kv_heads == 0)
        throw std::invalid_argument("invalid CUDA row KV append");
    return bind_leaf(thunks_, Leaf::kv_store, Op::store, "kv.bf16.linear", {}, f32_fma,
                     grid(a.shape.kv_heads, 256), a);
}

KernelBinding Bindings::bind(const Pick& a, Leaf leaf) const {
    if (!a.logits || !a.workspace || !a.ids || a.vocabulary == 0 ||
        (leaf != Leaf::pick_partial && leaf != Leaf::pick_final))
        throw std::invalid_argument("invalid CUDA row greedy pick");
    const bool partial = leaf == Leaf::pick_partial;
    return bind_leaf(thunks_, leaf, Op::argmax, partial ? "greedy.partial.pdl" : "greedy.final.pdl_index",
                     {}, f32_fma, grid(partial ? head_argmax::default_chunks(a.vocabulary) : 1U, 256), a);
}

KernelBinding Bindings::bind(const Gdn& a, GdnPhase phase) const {
    if (a.key_heads == 0 || a.value_heads == 0 || a.value_heads % a.key_heads != 0 ||
        a.key_width != 128 || a.value_width != 128 || a.conv_taps != 4 || !a.prepared)
        throw std::invalid_argument("unsupported CUDA row GDN dimensions");
    switch (phase) {
    case GdnPhase::convolution: {
        if (!a.convolution_weights || !a.convolution_state || !a.raw)
            throw std::invalid_argument("invalid CUDA row GDN convolution");
        const auto blocks = 2ULL * a.key_heads + a.value_heads;
        if (blocks > std::numeric_limits<std::uint32_t>::max() / 128)
            throw std::invalid_argument("CUDA row GDN convolution grid overflow");
        return bind_leaf(thunks_, Leaf::gdn_convolution, Op::elementwise, "gdn.convolution.parity", {}, f32_fma,
                         grid(static_cast<std::uint32_t>(blocks), 128), a);
    }
    case GdnPhase::recurrence:
        if (!a.ssm_a || !a.time_bias || !a.gain || !a.recurrent_state || !a.alpha || !a.beta || !a.z ||
            !a.output || !a.parity)
            throw std::invalid_argument("invalid CUDA row GDN recurrence");
        return bind_leaf(thunks_, Leaf::gdn_recurrence, Op::elementwise, "gdn.recurrence.parity", {}, f32_fma,
                         grid(a.value_heads, 128), a);
    }
    throw std::invalid_argument("unknown CUDA row GDN phase");
}

KernelBinding Bindings::bind(const Attention& a, AttentionPhase phase) const {
    const auto& s = a.shape;
    constexpr std::uint32_t chunk = 256;
    if (s.heads == 0 || s.kv_heads == 0 || s.heads % s.kv_heads != 0 || s.heads / s.kv_heads > 8 ||
        s.width != 256 || s.capacity == 0 || s.capacity > std::numeric_limits<std::uint32_t>::max() - (chunk - 1) ||
        !a.position || !a.scratch)
        throw std::invalid_argument("unsupported CUDA row split attention dimensions");
    const Grid chunks{{(s.capacity + chunk - 1) / chunk, s.kv_heads, 1}, {256, 1, 1}, 0};
    switch (phase) {
    case AttentionPhase::scores:
        if (!a.query || !a.key_cache || a.query_view.groups != s.heads || a.query_view.width != s.width)
            throw std::invalid_argument("invalid CUDA row attention scores");
        return bind_leaf(thunks_, Leaf::attention_scores, Op::attend, "attention.decode.bf16.scores", {}, f32_fma,
                         chunks, a);
    case AttentionPhase::values:
        if (!a.value_cache)
            throw std::invalid_argument("invalid CUDA row attention values");
        return bind_leaf(thunks_, Leaf::attention_values, Op::attend, "attention.decode.bf16.values", {}, f32_fma,
                         chunks, a);
    case AttentionPhase::combine:
        if (!a.output || (a.gate && (a.gate_view.groups != s.heads || a.gate_view.width != s.width)))
            throw std::invalid_argument("invalid CUDA row attention combine");
        return bind_leaf(thunks_, Leaf::attention_combine, Op::combine, "attention.decode.bf16.combine", {}, f32_fma,
                         grid(s.heads, 256), a);
    }
    throw std::invalid_argument("unknown CUDA row split attention phase");
}

void project(CudaProgramRecipe& recipe, const Bindings& bind, bool programmatic_edges,
             std::span<const Matrix> weights, std::span<float* const> outputs, const float* input,
             const float* residual, const Matrix* up) {
    if (weights.empty() || weights.size() > Dense{}.weights.size() || outputs.size() != weights.size())
        throw std::invalid_argument("invalid CUDA row projection count");
    if (up && (!up->data || weights.size() != 1 || weights.front().in != up->in ||
               weights.front().out != up->out || weights.front().format != up->format))
        throw std::invalid_argument("incompatible CUDA row gate and up weights");
    const auto emit = [&](std::span<const Matrix> group, std::span<float* const> destinations) {
        Dense args{};
        args.input = input;
        args.residual = residual;
        args.up = up ? up->data : nullptr;
        args.count = static_cast<std::uint32_t>(group.size());
        args.mma = mma3_supported(group.front());
        for (std::size_t i = 0; i < group.size(); ++i) {
            if (mma3_supported(group[i]) != args.mma)
                throw std::invalid_argument("step program requires one arithmetic form per projection group");
            args.weights[i] = group[i];
            args.outputs[i] = destinations[i];
        }
        recipe.kernel(bind.bind(args), programmatic_edges);
    };
    // This follows dense_rows_multi's incumbent caller. Mixed weight formats
    // are visited as ordered singletons.
    if (can_group(weights)) {
        emit(weights, outputs);
    } else {
        for (std::size_t i = 0; i < weights.size(); ++i)
            emit(weights.subspan(i, 1), outputs.subspan(i, 1));
    }
}

}  // namespace strix::exec::cuda_step_rows
