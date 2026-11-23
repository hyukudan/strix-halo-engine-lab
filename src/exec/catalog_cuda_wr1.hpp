#pragma once

#include "catalog_cuda_decode.hpp"
#include "cuda_program_recipe.hpp"
#include "strix/exec/descriptor.hpp"

namespace strix::exec::cuda_decode::wr1 {

// Transient operands are packed by descriptor row unless the scalar
// argument already specifies a stride. Weights and learned queries are shared.
// These leaves have no persistent state operands.
inline constexpr std::array<unsigned, 5> widths{1, 2, 4, 8, 16};

struct Greedy {
    // Contiguous logits per row and Pick2 scratch sized by workspace2_bytes.
    // Padding writes only RowStatus::masked. Active rows emit both scores
    // and keep the scalar kNone ID contract when a logit is nonfinite.
    const float* logits;
    head_argmax::Pick2* partial;
    RowOut* output;
    unsigned count, chunks;
};

enum class Leaf : unsigned {
    dense_fp32, dense_mma3, dense_mma3_round, mix, bank, pair, controller,
    rms, relative, clip, greedy_partial, greedy_final, softcap, round, sigmoid, add, count
};

using Thunks = std::array<std::array<LaunchFn, static_cast<unsigned>(Leaf::count)>, widths.size()>;

// Separate width-indexed recipe table. Every thunk enqueues exactly one
// matching kernel. Use completion edges when adding these bindings.
class Bindings {
public:
    Bindings(const StepBlock* descriptor, unsigned rows, const Thunks& thunks);

    [[nodiscard]] KernelBinding bind(const Bf16DenseGroup&) const;
    [[nodiscard]] KernelBinding bind(const core::residual_mix::MixArgs&) const;
    [[nodiscard]] KernelBinding bind(const core::residual_mix::NormalizeArgs&) const;
    [[nodiscard]] KernelBinding bind(const core::residual_mix::NormalizePairArgs&) const;
    [[nodiscard]] KernelBinding bind(const core::residual_mix::ControllerArgs&) const;
    [[nodiscard]] KernelBinding bind(const RmsBf16&) const;
    [[nodiscard]] KernelBinding bind(const Bf16RelativeProjection16&) const;
    [[nodiscard]] KernelBinding bind(const HeadClipBf16&) const;
    [[nodiscard]] KernelBinding bind_partial(const Greedy&) const;
    [[nodiscard]] KernelBinding bind(const Greedy&) const;
    [[nodiscard]] KernelBinding bind(const Bf16Softcap&) const;
    [[nodiscard]] KernelBinding bind(const Bf16Round&) const;
    [[nodiscard]] KernelBinding bind(const Bf16SigmoidProduct&) const;
    [[nodiscard]] KernelBinding bind(const core::residual_mix::AddArgs&) const;

    template <class Args>
    void append(CudaProgramRecipe& recipe, const Args& args) const {
        recipe.kernel(bind(args), false, widths[index_]);
    }

    void append(CudaProgramRecipe& recipe, const Greedy& args) const {
        recipe.kernel(bind_partial(args), false, widths[index_]);
        recipe.kernel(bind(args), false, widths[index_]);
    }

private:
    template <class Args>
    KernelBinding leaf(Leaf which, Grid grid, const Args& args) const;

    const StepBlock* descriptor_;
    std::size_t index_;
    const Thunks& thunks_;
};

template <class Args>
struct LaunchArgs {
    const StepBlock* descriptor;
    Args scalar;
};

[[nodiscard]] const Thunks& native_thunks();

}  // namespace strix::exec::cuda_decode::wr1
