#pragma once
// Head rows: H independent GEMVs of one weight layout run as ONE launch (backend-neutral plan, host + device).
//
//   y[h * head_outputs + o] = W[h * head_outputs + o] . x[h * input_stride + 0 .. width)
//
// W is one matrix of heads * head_outputs rows (the per-head matrices stored back to back, e.g. the MLA K_b / V_b
// absorb tensors); output `out` reads the input row of its head. A kernel that works on groups of `outs` consecutive
// outputs may use one input row per group only when no group straddles two heads (head_rows_refusal checks it), so
// every output keeps the products and the reduction tree of the same kernel launched per head.
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_HEAD_ROWS_HD __host__ __device__
#else
#define STRIX_HEAD_ROWS_HD
#endif

namespace strix::decode {

struct HeadRows {
    std::uint32_t heads = 0, head_outputs = 0;
    std::uint32_t input_stride = 0;  // elements between the input rows of two heads
};

// Element offset of the input row read by output `out`.
STRIX_HEAD_ROWS_HD constexpr std::uint64_t head_input_offset(std::uint32_t out, std::uint32_t head_outputs,
                                                             std::uint32_t input_stride) {
    return static_cast<std::uint64_t>(out / head_outputs) * input_stride;
}

// nullptr when a kernel with `outs` outputs per group and rows of `width` elements admits the plan.
constexpr const char* head_rows_refusal(const HeadRows& h, std::uint32_t width, unsigned outs) {
    if (h.heads == 0U || h.head_outputs == 0U || width == 0U || outs == 0U) return "head rows: empty geometry";
    if (h.head_outputs % outs != 0U) return "head rows: a group of outputs would straddle two heads";
    if (h.heads > 1U && h.input_stride < width) return "head rows: input rows of two heads overlap";
    if (static_cast<std::uint64_t>(h.heads) * h.head_outputs > 0xffffffffULL) return "head rows: too many outputs";
    return nullptr;
}

}  // namespace strix::decode
