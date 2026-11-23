#pragma once

#include "cuda_decode_lowering.hpp"

#include <array>
#include <optional>

namespace strix::exec::cuda_decode {

// Program-order fuser of the residual transition chain (B3 row owner). A decode program announces its residual adds,
// one-row bank readouts and input norms in program order, and every other leaf it emits is preceded by flush(). The
// fuser holds at most one add and one readout:
//   - a readout whose current (last) bank is the held add's output takes that add as its prologue;
//   - a norm (rms.bf16_gain.f32_image, one group) whose input is a raw row of the held readout attaches as that row's
//     epilogue;
//   - anything else first emits what is held, as the incumbent leaves, in program order.
// A held readout that gained a prologue or a norm is emitted as ONE residual.transition leaf, else as the incumbent
// readout (lower_bank_readout). The fused leaf stores every image the replaced leaves store, with their bits, and
// nothing is emitted between the replaced leaves in program order, so the program's numerical identity is unchanged
// (TRANSITION fixture; engine identity gates). Disabled, every call emits the incumbent leaf at once.
class TransitionChain {
public:
    using Mix = core::residual_mix::MixArgs;
    using Add = core::residual_mix::AddArgs;
    using Transition = core::residual_mix::TransitionArgs;

    TransitionChain() = default;
    explicit TransitionChain(R1Forms forms) : forms_(forms) {}

    [[nodiscard]] R1Forms forms() const noexcept { return forms_; }
    [[nodiscard]] bool idle() const noexcept { return !add_ && !read_; }

    template <class Emit>
    void add(const Add& args, const Emit& emit) {
        flush(emit);
        if (forms_.transition && args.lhs && args.rhs && args.output) add_ = args;
        else emit(args, false);
    }

    template <class Emit>
    void readout(const Mix& args, const Emit& emit) {
        if (forms_.transition && args.rows == 1U && args.bank_count != 0U) {
            Transition fused{};
            fused.mix = args;
            if (add_ && args.banks[args.bank_count - 1U] == add_->output) {
                fused.add = *add_;
                if (core::residual_mix::fits(fused)) {
                    add_.reset();
                    flush(emit);
                    read_ = fused;
                    return;
                }
                fused.add = {};
            }
            flush(emit);
            if (core::residual_mix::fits(fused)) {
                read_ = fused;
                return;
            }
        }
        flush(emit);
        lower_bank_readout(args, forms_, emit);
    }

    template <class Emit>
    void norm(const RmsBf16& args, const Emit& emit) {
        if (read_ && args.groups == 1U && args.width == read_->mix.width && attach(args)) return;
        flush(emit);
        emit(args, true);
    }

    // Emits what is held, in program order. Call before any other leaf of the program and at its end.
    template <class Emit>
    void flush(const Emit& emit) {
        if (add_) {
            emit(*add_, false);
            add_.reset();
        }
        if (read_) {
            const auto held = *read_;
            read_.reset();
            if (core::residual_mix::has_prologue(held) || core::residual_mix::norm_count(held) != 0U) emit(held, true);
            else lower_bank_readout(held.mix, forms_, emit);
        }
    }

private:
    bool attach(const RmsBf16& args) {
        auto candidate = *read_;
        const bool first = core::residual_mix::norm_count(candidate) == 0U;
        if (!first && candidate.norm_epsilon != args.epsilon) return false;
        for (unsigned q = 0; q < candidate.mix.query_count; ++q) {
            if (candidate.mix.output[q] != args.input || candidate.norm[q].output) continue;
            candidate.norm[q] = {args.gain, args.output};
            candidate.norm_epsilon = args.epsilon;
            if (!core::residual_mix::fits(candidate)) return false;
            read_ = candidate;
            return true;
        }
        return false;
    }

    R1Forms forms_{};
    std::optional<Add> add_;
    std::optional<Transition> read_;
};

}  // namespace strix::exec::cuda_decode
