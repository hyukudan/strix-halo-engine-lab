#pragma once
// GLM-5.3-Flash Tier A device kernels, batch 1 (design notes sections 1-2): each launcher reproduces the
// host emulation in src/glm/host/glm_host.cpp bit for bit given the same inputs (gate K). All pointers are device
// pointers; rows are contiguous; `flag` is the device non-finite flag (checkpoints C1/C2/C5), ORed, never cleared here.
#include "glm/common/glm_math.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::glm::dev {

enum class WType : unsigned { F16 = 0, BF16 = 1, F32 = 2 };

// R1: y[r][o] = chain_i fmaf(x[r][i], W[o0 + o][i]) for o < n.
void dense(const float* x, const void* w, WType wt, unsigned in, unsigned o0, unsigned n, float* y, unsigned rows,
           hipStream_t s);
// R2/R3: y = (x * r) * w (w == nullptr: unweighted), r = 1 / sqrt(sumsq / n + eps); lanes per norm_lanes(n).
void rmsnorm(const float* x, const void* w, WType wt, unsigned n, float eps, float* y, unsigned rows, hipStream_t s);

// mHC. mix layout per row: pre[4] post[4] comb[16] (comb[j * 4 + i]). F_scratch: [rows][16384].
void hc_pre(const float* X, const float* fn, const float* base, const float* scale, float* F_scratch, float* mix,
            float* x_collapsed, unsigned rows, hipStream_t s);
// flag_stride 0: one flag for the call; 1: flag[r] per row (speculative verify, per-row C1).
void hc_expand(const float* X, const float* mix, const float* y, float* Xout, unsigned* flag, unsigned rows, hipStream_t s,
               unsigned flag_stride = 0);
void hc_final(const float* X, float* h, unsigned rows, hipStream_t s);

// KDA for `heads` rank heads; per row one sequence slot (state and conv at slot = row).
//   qkv [rows][3 * heads * 128], f / gate [rows][heads * 128], b [rows][heads]
//   S [rows][heads][128][128], conv_state [rows][3 * heads * 128][3]; conv_w [3 * heads * 128][4] BF16
//   scratch [rows][heads][4][128] (q, k, v, g) + beta [rows][heads]; y [rows][heads * 128]
struct KdaDev {
    unsigned heads;
    const std::uint16_t* conv_w;
    const float* A_log;
    const float* dt_bias;
    const std::uint16_t* o_norm;  // BF16 [128]
};
void kda_step(const KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
              float* conv_state, float* scratch, float* beta, float* o, float* y, unsigned* flag, unsigned rows,
              hipStream_t s);

// Router: logits [rows][288] (from dense) -> ids [rows][8] ascending, w [rows][8].
void route(const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned rows, hipStream_t s);
// y[r][d] = (chain_k fmaf(w[r][k], eo[r][k][d])) + shared[r][d]; eo [rows][8][4096].
void moe_combine(const float* w, const float* eo, const float* shared, float* y, unsigned rows, hipStream_t s);
// a = swiglu(g, u) elementwise (R15).
void swiglu(const float* g, const float* u, float* a, std::size_t n, hipStream_t s);
// Local argmax per row over n logits (ties -> lower id), id0 added; C5 flag on non-finite logits (flag_stride as hc_expand).
void argmax(const float* logits, unsigned n, std::uint32_t id0, float* value, std::uint32_t* id, unsigned* flag,
            unsigned rows, hipStream_t s, unsigned flag_stride = 0);

}  // namespace strix::glm::dev
