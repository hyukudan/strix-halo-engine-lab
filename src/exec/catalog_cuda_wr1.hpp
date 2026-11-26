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

// The early-launch contract of each leaf body, the same contract the one-row decode step declares for
// its dense and RMS leaves. The dense rows bodies (rows_body.inc, rows_mma_body.inc) and the RMS step
// body (After<> operands, norm_body.inc) issue the grid dependency wait before they read any operand a
// predecessor writes and before any global store; before it they read only kernel parameters, the step
// descriptor uploaded before the graph and constant weights. Such a leaf takes a programmatic edge from
// a preceding kernel. Every other leaf keeps a completion edge. A thunk table must honour this contract.
[[nodiscard]] constexpr PdlLeaf pdl_contract(Leaf which) noexcept {
    switch (which) {
    case Leaf::dense_fp32:
    case Leaf::dense_mma3:
    case Leaf::dense_mma3_round:
    case Leaf::rms: return {PdlWait::before_step_io, PdlOptOut::none};
    default: return {PdlWait::undeclared, PdlOptOut::legacy_schedule};
    }
}

// Separate width-indexed recipe table. Every thunk enqueues exactly one
// matching kernel. Callers can opt into programmatic edges under pdl_contract.
class Bindings {
public:
    Bindings(const StepBlock* descriptor, unsigned rows, const Thunks& thunks, bool early_launch = false);

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
        recipe.kernel(bind(args), widths[index_]);
    }

    void append(CudaProgramRecipe& recipe, const Greedy& args) const {
        recipe.kernel(bind_partial(args), widths[index_]);
        recipe.kernel(bind(args), widths[index_]);
    }

private:
    template <class Args>
    KernelBinding leaf(Leaf which, Grid grid, const Args& args) const;

    const StepBlock* descriptor_;
    std::size_t index_;
    const Thunks& thunks_;
    bool early_launch_;
};

template <class Args>
struct LaunchArgs {
    const StepBlock* descriptor;
    Args scalar;
};

[[nodiscard]] const Thunks& native_thunks();

}  // namespace strix::exec::cuda_decode::wr1
