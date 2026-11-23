#pragma once

#include "key.hpp"
#include "strix/numeric/math_gfx1151.hpp"

namespace strix::cuda::core::chain {

// Aux slots point to device data. Plain vector operands are contiguous
// [row][column], except broadcast bias, normalization gain/bias and the
// per-row device weight in route.weight_mul.
// Descriptors are read-only; explicitly mutable output planes hold side effects.
struct ScalarStore { void* data; };
struct Gate { const float* operand; float limit; bool bf16 = false; };
struct Gfx1151Silu {
    const float* up;  // [row][column]
    numeric::gfx1151::Tables tables;
};
struct Gfx1151Gate {
    Gate gate;
    numeric::gfx1151::Tables tables;
};
struct Gfx1151FoldedGate {
    const std::uint16_t* input;  // BF16 [row][input_stride]
    const float* weights;       // [features], shared across rows
    std::size_t input_stride;   // in BF16 elements
    unsigned features;         // independent of the width being gated
    numeric::gfx1151::Tables tables;
    float* sigmoid_out = nullptr;  // optional [row] copy for other chain consumers
};
struct Softcaps { const float* up; float gate_cap, up_cap; bool bf16 = false; };
struct Softcap { float cap; bool bf16_input = false; };
struct Affine { const float* scale; const float* shift; };
struct ClassBias { const unsigned* classes; const float* bias; };

enum class RmsOrder : std::uint8_t { cuda_warp, bank_halving };
struct Norm {
    float epsilon;
    const float* gain = nullptr;
    const float* bias = nullptr;
    const std::uint16_t* bf16_gain = nullptr;  // [group][column]
    RmsOrder order = RmsOrder::cuda_warp;
    bool bf16_output = false;  // cuda_warp uses the incumbent CUDA conversion
    float* inverse_out = nullptr;  // [row], for opened norms with a single CTA per row
};
struct Clip {
    float threshold, epsilon;
    unsigned position_floor = 0;
    float position_alpha = 0;
    const std::uint16_t* gain = nullptr;  // [group][column]
};
struct Rope {
    const float* cosine;
    const float* sine;
    unsigned half;  // NeoX layout; leave the suffix after 2 * half untouched.
};
struct Quant {
    void* codes;  // I4 uses one signed byte per code until packed by the consumer.
    float* scales;       // group-major, or one per row for E4M3
    int* sums = nullptr;
};
struct Emit {
    std::uint32_t* ids;
    float* values;
    unsigned count = 1;  // only used by topn
};
struct Mix {
    const float* residual;  // [row][4][width]
    const float* coefficients;  // [row][24]
    float* output;  // [row][4][width]
    float* pre_carry = nullptr;  // [row][4]
};
// One pointer per state and logical row, resolved to that row's pending
// image by the executor. State slot resolution and lifetime are executor
// responsibilities.
struct PendingRow {
    void* data;
    unsigned* nonfinite = nullptr;
};

// Per row/group. row is the dense row ordinal for operands and side outputs;
// group selects a gain vector. Producers set scratch to shared memory for
// row collectives (256 floats, or max(256, width) when a RoPE link is present).
// Recompute callers must leave Norm::inverse_out null when more than one CTA
// owns the logical row. A chain's side effects must target disjoint rows or
// the caller's own tiles; links do not establish inter-CTA ordering.
struct TileCtx {
    unsigned width;
    unsigned row;
    unsigned group = 0;
    unsigned position = 0;
    unsigned first_index = 0;
    float* scratch = nullptr;
    const PendingRow* pending = nullptr;
};

}  // namespace strix::cuda::core::chain
