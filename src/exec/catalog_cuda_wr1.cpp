#include "catalog_cuda_wr1.hpp"
#include "strix/core/rows.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::exec::cuda_decode::wr1 {
namespace {

namespace mix = core::residual_mix;
constexpr std::array<std::string_view, static_cast<unsigned>(Leaf::count)> forms{
    "dense-fp32", "dense-mma3", "dense-mma3-round", "residual-mix", "bank-norm",
    "pair-norm", "controller", "rms-bf16", "relative-d16", "head-clip", "greedy-partial",
    "greedy-final", "softcap", "round", "sigmoid-product", "add"
};
constexpr unsigned grid_x_limit = 0x7fffffffU;
constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);

void require(bool valid) {
    if (!valid) throw std::invalid_argument("invalid WR1 CUDA leaf binding");
}

bool aligned(const void* pointer) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
}

bool positive(float value) { return value > 0.0F && std::isfinite(value); }

Grid pointwise(std::size_t elements, unsigned rows) {
    require(elements != 0 && elements <= max_floats / rows);
    return {{static_cast<unsigned>(std::min<std::size_t>((elements - 1U) / 256U + 1U, 65535U)), rows, 1},
            {256, 1, 1}, 0};
}

mix::NormalizeArgs normalize(mix::NormalizeArgs args, unsigned rows) {
    require(args.rows == rows && mix::fits(args) &&
            args.width <= std::numeric_limits<unsigned>::max() - mix::detail::kThreads + 1U);
    if (!args.input_stride) args.input_stride = args.width;
    if (!args.output_stride) args.output_stride = args.width;
    require(args.input != args.output || args.input_stride == args.output_stride);
    return args;
}

Greedy greedy(Greedy args, unsigned rows) {
    require(args.logits && args.partial && args.count != 0U && args.count <= max_floats / rows);
    if (!args.chunks) args.chunks = head_argmax::default_chunks(args.count);
    require(args.chunks <= head_argmax::kMaxChunks &&
            args.count <= std::numeric_limits<unsigned>::max() - 255U);
    return args;
}

}  // namespace

Bindings::Bindings(const StepBlock* descriptor, unsigned rows, const Thunks& thunks)
    : descriptor_(descriptor),
      index_(static_cast<std::size_t>(std::ranges::find(widths, rows) - widths.begin())), thunks_(thunks) {
    require(descriptor && index_ < widths.size());
}

template <class Args>
KernelBinding Bindings::leaf(Leaf which, Grid grid, const Args& args) const {
    const auto id = static_cast<unsigned>(which);
    KernelBinding result;
    result.fn = thunks_[index_][id];
    require(result.fn != nullptr);
    result.identity = "cuda-wr1-" + std::string(forms[id]) + "-r" + std::to_string(widths[index_]) + "-v1";
    result.exact_identity = result.identity;
    result.grid = grid;
    result.predicted.launches = 1;
    const LaunchArgs<Args> launch{descriptor_, args};
    const auto bytes = std::as_bytes(std::span(&launch, 1));
    result.args.assign(bytes.begin(), bytes.end());
    return result;
}

KernelBinding Bindings::bind(const Bf16DenseGroup& a) const {
    const auto rows = widths[index_];
    const bool mma = a.mode != Bf16DenseMode::fp32;
    const bool round_bf16 = a.mode == Bf16DenseMode::mma3_round;
    require((!mma || a.mode == Bf16DenseMode::mma3 || round_bf16) && a.count > 0 &&
            a.count <= a.projections.size() && aligned(a.input) && a.input_width != 0 &&
            a.input_width <= max_floats / rows &&
            a.input_width % (mma ? core::rows::kBlockIn : 8U) == 0U);
    std::uint64_t outputs = 0;
    for (const auto& p : std::span(a.projections).first(a.count)) {
        require(aligned(p.weights) && aligned(p.output) && p.output != a.input && p.outputs != 0 &&
                p.outputs <= max_floats / rows && p.round_bf16 == round_bf16 &&
                p.outputs <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / a.input_width &&
                (!mma || p.outputs % core::rows::kTileOut == 0U));
        outputs += p.outputs;
    }
    require(outputs <= std::numeric_limits<unsigned>::max() - 7U);
    const auto blocks = mma ? outputs / core::rows::kTileOut : (outputs - 1U) / 8U + 1U;
    const auto which = !mma ? Leaf::dense_fp32 : round_bf16 ? Leaf::dense_mma3_round : Leaf::dense_mma3;
    return leaf(which, {{static_cast<unsigned>(blocks), rows, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding Bindings::bind(const mix::MixArgs& a) const {
    const auto rows = widths[index_];
    require(a.rows == rows && mix::fits(a) &&
            a.width <= std::numeric_limits<unsigned>::max() - 128U * mix::kMaxBanks + 1U);
    const auto threads = a.bank_count > 1U ? std::array{128U, a.bank_count, 1U} : std::array{256U, 1U, 1U};
    return leaf(Leaf::mix, {{rows, 1, 1}, threads, 0}, a);
}

KernelBinding Bindings::bind(const mix::NormalizeArgs& a) const {
    const auto rows = widths[index_];
    return leaf(Leaf::bank, {{rows, 1, 1}, {mix::detail::kThreads, 1, 1}, 0}, normalize(a, rows));
}

KernelBinding Bindings::bind(const mix::NormalizePairArgs& a) const {
    const auto rows = widths[index_];
    return leaf(Leaf::pair, {{rows, 2, 1}, {mix::detail::kThreads, 1, 1}, 0},
                mix::NormalizePairArgs{normalize(a.first, rows), normalize(a.second, rows)});
}

KernelBinding Bindings::bind(const mix::ControllerArgs& a) const {
    require(a.current && a.message && a.forget && a.gain && a.output);
    return leaf(Leaf::controller, pointwise(a.elements, widths[index_]), a);
}

KernelBinding Bindings::bind(const RmsBf16& a) const {
    const auto rows = widths[index_];
    require(a.input && a.output && a.groups && a.groups <= grid_x_limit && a.width && a.width <= 8192U &&
            a.groups <= std::numeric_limits<unsigned>::max() / a.width &&
            a.groups <= max_floats / rows / a.width && positive(a.epsilon));
    return leaf(Leaf::rms, {{a.groups, rows, 1}, {256, 1, 1}, 0}, a);
}

KernelBinding Bindings::bind(const Bf16RelativeProjection16& a) const {
    const auto rows = widths[index_];
    require(a.input && a.weights && a.output && a.heads && a.relative_extent &&
            a.heads <= max_floats / rows / 16U &&
            a.relative_extent <= max_floats / std::max<std::size_t>(rows, 8U) / a.heads);
    return leaf(Leaf::relative, pointwise(std::size_t{a.heads} * a.relative_extent, rows), a);
}

KernelBinding Bindings::bind(const HeadClipBf16& a) const {
    const auto rows = widths[index_];
    require(a.input && a.output && a.heads && a.heads <= grid_x_limit && a.width && a.width <= 256U &&
            a.heads <= max_floats / rows / a.width && positive(a.epsilon) && positive(a.threshold) &&
            (!a.scale_floor || std::isfinite(a.scale_alpha)));
    const auto threads = a.width <= 32U ? 32U : a.width <= 128U ? 128U : 256U;
    return leaf(Leaf::clip, {{a.heads, rows, 1}, {threads, 1, 1}, 0}, a);
}

KernelBinding Bindings::bind_partial(const Greedy& a) const {
    const auto rows = widths[index_];
    const auto reduction = greedy(a, rows);
    return leaf(Leaf::greedy_partial, {{reduction.chunks, rows, 1}, {256, 1, 1}, 0}, reduction);
}

KernelBinding Bindings::bind(const Greedy& a) const {
    const auto rows = widths[index_];
    require(a.output != nullptr);
    return leaf(Leaf::greedy_final, {{rows, 1, 1}, {256, 1, 1}, 0}, greedy(a, rows));
}

KernelBinding Bindings::bind(const Bf16Softcap& a) const {
    require(a.values && positive(a.cap));
    return leaf(Leaf::softcap, pointwise(a.elements, widths[index_]), a);
}

KernelBinding Bindings::bind(const Bf16Round& a) const {
    require(a.values != nullptr);
    return leaf(Leaf::round, pointwise(a.elements, widths[index_]), a);
}

KernelBinding Bindings::bind(const Bf16SigmoidProduct& a) const {
    require(a.input && a.logits && a.output);
    return leaf(Leaf::sigmoid, pointwise(a.elements, widths[index_]), a);
}

KernelBinding Bindings::bind(const mix::AddArgs& a) const {
    require(a.lhs && a.rhs && a.output);
    return leaf(Leaf::add, pointwise(a.elements, widths[index_]), a);
}

}  // namespace strix::exec::cuda_decode::wr1
