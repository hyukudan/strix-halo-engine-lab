#pragma once
// The gated RMSNorm of one GDN value head (swish gate, width kGatedHead): ONE arithmetic shared by the chunked GDN's own norm kernel
// (gdn_chunk.cu chunk_norm_kernel) and the FAST4 folded quantiser (arch/sm120/nvfp4_tile.cuh, GatedHeadNorm), so the two produce the
// same bits. s0..s3 are warp_sum_xor(acc * acc) of the head's four 32-lane quarters, in position order.
#include "../common.cuh"

namespace strix::cuda_core {
inline constexpr unsigned kGatedHead = 128;

__device__ __forceinline__ float gated_head_inv(float s0, float s1, float s2, float s3, float eps) {
    const float ss = __fadd_rn(__fadd_rn(__fadd_rn(s0, s1), s2), s3);
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(kGatedHead)), eps)));
}

__device__ __forceinline__ float gated_head_value(float acc, float inv, float gain, float z) {
    return __fmul_rn(__fmul_rn(__fmul_rn(acc, inv), gain), cc_silu(z));
}
}  // namespace strix::cuda_core
