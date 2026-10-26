// GLM-5.3-Flash chunked prefill, Tier A kernels (see glm_prefill.hpp). Each kernel repeats the per-row expressions,
// chains and trees of the decode kernels in glm_tier_a.hip / glm_dsa.hip; only the work placement differs. Gate:
// tests/hip_glm_prefill_tests.hip (prefill == decode row by row, bitwise). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero.
#include "glm/prefill/glm_prefill.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::pf {

namespace {

using dev::DsaSlot;
using dev::IndexerDev;
using dev::KdaDev;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- KDA
// 4a for C rows of one slot: grid (head, row). The conv of row r reads the pre-conv inputs of positions t - 3 .. t - 1
// from the chunk (row r - j >= 0) or from the slot's conv state (oldest first); the state itself is written afterwards
// by kda_conv_commit_kernel. Expressions = kda_prep_kernel.
__global__ __launch_bounds__(128) void kda_prep_rows_kernel(KdaDev k, const float* qkv, const float* f, const float* b,
                                                            const float* conv_state, float* scratch, float* beta) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads, C = H * kKdaDim;
    float out[3];
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned ch = part * C + h * kKdaDim + i;
        const float* st = conv_state + static_cast<std::size_t>(ch) * 3;
        const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;
        float x[3];
        for (unsigned j = 0; j < 3; ++j) {  // x[j] = input of position t - 3 + j
            const int m = static_cast<int>(r) - 3 + static_cast<int>(j);
            x[j] = m >= 0 ? qkv[static_cast<std::size_t>(m) * 3 * C + ch] : st[3 + m];
        }
        const float nw = qkv[static_cast<std::size_t>(r) * 3 * C + ch];
        float acc = __builtin_fmaf(bf16_to_float(cw[0]), x[0], 0.0F);
        acc = __builtin_fmaf(bf16_to_float(cw[1]), x[1], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[2]), x[2], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
        out[part] = silu(acc);
    }
    const float sq = lds_tree<128>(s, __builtin_fmaf(out[0], out[0], 0.0F));
    const float sk = lds_tree<128>(s, __builtin_fmaf(out[1], out[1], 0.0F));
    const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
    float* sc = scratch + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim;
    sc[i] = (out[0] / nq) * kInvSqrt128;
    sc[kKdaDim + i] = out[1] / nk;
    sc[2 * kKdaDim + i] = out[2];
    const float expA = declared_exp(k.A_log[h]);
    sc[3 * kKdaDim + i] = kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(r) * C + h * kKdaDim + i] + k.dt_bias[h * kKdaDim + i]));
    if (i == 0) beta[static_cast<std::size_t>(r) * H + h] = sigmoid(b[static_cast<std::size_t>(r) * H + h]);
}

// Conv state after row C - 1: the inputs of the last three positions, oldest first.
__global__ void kda_conv_commit_kernel(const float* qkv, float* conv_state, unsigned channels, unsigned C) {
    const unsigned ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= channels) return;
    float* st = conv_state + static_cast<std::size_t>(ch) * 3;
    const float o0 = st[0], o1 = st[1], o2 = st[2];
#pragma unroll
    for (unsigned j = 0; j < 3; ++j) {
        const int m = static_cast<int>(C) - 3 + static_cast<int>(j);
        st[j] = m >= 0 ? qkv[static_cast<std::size_t>(m) * channels + ch] : (m == -3 ? o0 : (m == -2 ? o1 : o2));
    }
}

// 4b over C tokens, token-sequential: grid (head, value tile of kScanV columns); a quad of lanes owns column v and
// holds S[:, v] (128 fp32) in registers for the whole chunk: lane q of the quad owns keys g * 32 + q * 8 + j
// (g = 0..3, j = 0..7). Per token, the decode kda_update_kernel expressions for column v:
//   sp_k = S_kv * dec_k; u = lds_tree<128> pairing of (+0 + sp_k * k_k); d = (v - u) * beta; S_kv = fmaf(k_k, d, sp_k);
//   o_v = the same tree of (+0 + S_kv * q_k).
// Tree pairing as lds_tree<128>: levels 64 and 32 pair keys inside a lane; level 16 pairs lanes q and q ^ 2, level 8
// lanes q and q ^ 1 (shuffles; IEEE addition is commutative, so both lanes of a pair get the same bits); levels 4 .. 1
// inside each lane.
constexpr unsigned kScanV = 64, kScanTB = 16, kScanLanes = 4, kOwn = kKdaDim / kScanLanes;  // 32 keys per lane
__device__ __forceinline__ unsigned scan_key(unsigned q, unsigned i) {  // own key i in [0, 32)
    return (i / 8) * 32 + q * 8 + (i % 8);
}
__device__ __forceinline__ float scan_tree(const float* st, const float* w, unsigned q) {
    float tr[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)  // level 64: (l, l + 64) for l = g * 32 + q * 8 + j, g = 0, 1
        tr[i] = (0.0F + st[i] * w[scan_key(q, i)]) + (0.0F + st[i + 16] * w[scan_key(q, i + 16)]);
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + tr[j + 8];  // level 32: (l, l + 32), l = q * 8 + j
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + __shfl_xor(tr[j], 2);  // level 16
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) tr[j] = tr[j] + __shfl_xor(tr[j], 1);  // level 8
#pragma unroll
    for (unsigned hh = 4; hh > 0; hh >>= 1U)
#pragma unroll
        for (unsigned l = 0; l < hh; ++l) tr[l] = tr[l] + tr[l + hh];
    return tr[0];
}
__global__ __launch_bounds__(kScanLanes * kScanV) void kda_scan_kernel(unsigned H, float* S, const float* scratch,
                                                                       const float* beta, float* o, unsigned* flag,
                                                                       unsigned C) {
    __shared__ float sq[kScanTB][kKdaDim], sk[kScanTB][kKdaDim], sd[kScanTB][kKdaDim];
    const unsigned h = blockIdx.x, tid = threadIdx.x, q = tid % kScanLanes, v = blockIdx.y * kScanV + tid / kScanLanes;
    float* Sh = S + static_cast<std::size_t>(h) * kKdaDim * kKdaDim;
    float st[kOwn];
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) st[i] = Sh[scan_key(q, i) * kKdaDim + v];
    bool bad = false;
    for (unsigned b0 = 0; b0 < C; b0 += kScanTB) {
        const unsigned nb = C - b0 < kScanTB ? C - b0 : kScanTB;
        __syncthreads();
        for (unsigned idx = tid; idx < nb * kKdaDim; idx += kScanLanes * kScanV) {
            const unsigned j = idx / kKdaDim, kk = idx % kKdaDim;
            const float* sc = scratch + (static_cast<std::size_t>(b0 + j) * H + h) * 4 * kKdaDim;
            sq[j][kk] = sc[kk];
            sk[j][kk] = sc[kKdaDim + kk];
            sd[j][kk] = declared_exp(sc[3 * kKdaDim + kk]);
        }
        __syncthreads();
        for (unsigned j = 0; j < nb; ++j) {
            const std::size_t rh = static_cast<std::size_t>(b0 + j) * H + h;
            const float vv = scratch[rh * 4 * kKdaDim + 2 * kKdaDim + v];
            const float bt = beta[rh];
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) st[i] = st[i] * sd[j][scan_key(q, i)];
            const float dlt = (vv - scan_tree(st, sk[j], q)) * bt;
#pragma unroll
            for (unsigned i = 0; i < kOwn; ++i) {
                st[i] = __builtin_fmaf(sk[j][scan_key(q, i)], dlt, st[i]);
                bad = bad || !is_finite(st[i]);
            }
            const float ov = scan_tree(st, sq[j], q);
            if (q == 0) o[rh * kKdaDim + v] = ov;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < kOwn; ++i) Sh[scan_key(q, i) * kKdaDim + v] = st[i];
    if (bad && flag) atomicOr(flag, 1U);
}

// 4c: = kda_out_kernel (row-parallel already).
__global__ __launch_bounds__(128) void kda_out_rows_kernel(KdaDev k, const float* o, const float* gate, float* y) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads;
    const std::size_t at = (static_cast<std::size_t>(r) * H + h) * kKdaDim + i;
    const float v = o[at];
    const float ms = lds_tree<128>(s, __builtin_fmaf(v, v, 0.0F)) / static_cast<float>(kKdaDim);
    const float rr = inv_sqrt(ms + kEps);
    y[at] = ((v * rr) * bf16_to_float(k.o_norm[i])) * sigmoid(gate[at]);
}

// ---------------------------------------------------------------- DSA append
// dsa_store_kernel without the pooling (row-parallel: the pool reads other rows), position t0 + r.
__global__ __launch_bounds__(128) void dsa_store_rows_kernel(DsaSlot sl, unsigned t0, const float* latent, IndexerDev w,
                                                             const float* key, const float* gs, unsigned* flag) {
    __shared__ float s[128];
    const unsigned r = blockIdx.x, i = threadIdx.x, t = t0 + r;
    const float k = key[static_cast<std::size_t>(r) * kIdxDim + i];
    const float mu = lds_tree<128>(s, 0.0F + k) / static_cast<float>(kIdxDim);
    const float d = k - mu;
    const float var = lds_tree<128>(s, __builtin_fmaf(d, d, 0.0F)) / static_cast<float>(kIdxDim);
    const float rr = inv_sqrt(var + kLnEps);
    const std::uint16_t kb = float_to_bf16((d * rr) * f16_to_float(w.k_norm_w[i]) + f16_to_float(w.k_norm_b[i]));
    const std::uint16_t gb = float_to_bf16(gs[static_cast<std::size_t>(r) * kIdxDim + i]);
    if (!bf16_is_finite(kb) || !bf16_is_finite(gb)) atomicOr(flag, 1U);
    sl.ikey[static_cast<std::size_t>(t) * kIdxDim + i] = kb;
    sl.igate[static_cast<std::size_t>(t) * kIdxDim + i] = gb;
    for (unsigned l = i; l < kKvLora; l += 128) {
        const std::uint16_t lb = float_to_bf16(latent[static_cast<std::size_t>(r) * kKvLora + l]);
        if (!bf16_is_finite(lb)) atomicOr(flag, 1U);
        sl.latent[static_cast<std::size_t>(t) * kKvLora + l] = lb;
    }
}
// R12 for pools p0 + blockIdx.x (channel = lane), = the pooling branch of dsa_store_kernel.
__global__ __launch_bounds__(128) void dsa_pool_kernel(DsaSlot sl, unsigned p0, IndexerDev w, unsigned* flag) {
    const unsigned p = p0 + blockIdx.x, i = threadIdx.x, t0 = p * kKpool;
    float l4[kKpool], kv[kKpool], e[kKpool], pr[kKpool];
    for (unsigned j = 0; j < kKpool; ++j) {
        l4[j] = bf16_to_float(sl.igate[static_cast<std::size_t>(t0 + j) * kIdxDim + i]) + w.ape[j * kIdxDim + i];
        kv[j] = bf16_to_float(sl.ikey[static_cast<std::size_t>(t0 + j) * kIdxDim + i]);
    }
    float m = l4[0];
    for (unsigned j = 1; j < kKpool; ++j) m = l4[j] > m ? l4[j] : m;
    for (unsigned j = 0; j < kKpool; ++j) e[j] = declared_exp(l4[j] - m);
    const float sum = ((e[0] + e[1]) + e[2]) + e[3];
    for (unsigned j = 0; j < kKpool; ++j) pr[j] = e[j] / sum;
    const float K = ((pr[0] * kv[0] + pr[1] * kv[1]) + pr[2] * kv[2]) + pr[3] * kv[3];
    if (!is_finite(K)) atomicOr(flag, 1U);
    sl.pooled[static_cast<std::size_t>(p) * kIdxDim + i] = K;
}

// ---------------------------------------------------------------- indexer (R11, R5)
__global__ void scale_kernel(float* v, unsigned n, float c) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) v[i] = v[i] * c;
}
// Scores of a tile of kScoreRows rows x 128 pools: thread = pool p, its pooled key in registers; the row's qI is staged
// in LDS. Per (row, pool) = idx_score_kernel's chain. Rows with P_t <= 512 select every pool and need no scores.
constexpr unsigned kScoreRows = 16;
__global__ __launch_bounds__(128) void idx_score_rows_kernel(const float* pooled, unsigned t0, unsigned C, const float* qI,
                                                             const float* wh, std::uint32_t* keys, unsigned cap) {
    __shared__ float q[kIdxHeads * kIdxDim];
    __shared__ float w[kIdxHeads];
    const unsigned p = blockIdx.x * 128 + threadIdx.x, r0 = blockIdx.y * kScoreRows;
    const unsigned Pmax = (t0 + (r0 + kScoreRows < C ? r0 + kScoreRows : C)) / kKpool;  // pools of the tile's last row
    if (blockIdx.x * 128 >= Pmax) return;  // uniform per block
    float K[kIdxDim];
    const bool have = p < Pmax;
#pragma unroll
    for (unsigned i = 0; i < kIdxDim; ++i) K[i] = have ? pooled[static_cast<std::size_t>(p) * kIdxDim + i] : 0.0F;
    for (unsigned r = r0; r < r0 + kScoreRows && r < C; ++r) {
        const unsigned P = (t0 + r + 1) / kKpool;
        if (P <= kIdxPools) continue;  // uniform per block
        __syncthreads();
        for (unsigned i = threadIdx.x; i < kIdxHeads * kIdxDim; i += 128) q[i] = qI[static_cast<std::size_t>(r) * kIdxHeads * kIdxDim + i];
        if (threadIdx.x < kIdxHeads) w[threadIdx.x] = wh[static_cast<std::size_t>(r) * kIdxHeads + threadIdx.x];
        __syncthreads();
        if (p >= P) continue;
        float acc = 0.0F;
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            float dot = 0.0F;
#pragma unroll
            for (unsigned i = 0; i < kIdxDim; ++i) dot = __builtin_fmaf(q[h * kIdxDim + i], K[i], dot);
            const float sc = dot * kInvSqrt128;
            acc = __builtin_fmaf(w[h], sc > 0.0F ? sc : 0.0F, acc);
        }
        keys[static_cast<std::size_t>(r) * cap + p] = ordered_key(acc);
    }
}

// Scores v2: CTA = 64 pools x 4 rows, all 32 heads. Phase 1: the per-(row, head, pool) dot chains (i = 0..127
// ascending from +0: idx_score_rows_kernel's fmaf(q, K, dot)) as a register-blocked tile, thread = 4 pools x 8 (row,
// head) pairs, the pooled keys [i][pool] and qI [i][(row, head)] staged in LDS per 32-dim slice. Phase 2: per (row,
// pool) the head chain acc = fmaf(w[h], relu(dot * kInvSqrt128), acc) for h = 0..31 ascending from +0, from the dots
// staged in LDS. Same bits as the v1 kernel; rows with P_t <= 512 and pools p >= P_t are skipped as there.
constexpr unsigned kIs2Pools = 64, kIs2Rows = 4, kIs2K = 32, kIs2RH = kIs2Rows * kIdxHeads;  // 128 (row, head) pairs
__global__ __launch_bounds__(256) void idx_score_rows_v2_kernel(const float* pooled, unsigned t0, unsigned C, const float* qI,
                                                                const float* wh, std::uint32_t* keys, unsigned cap) {
    __shared__ alignas(16) float ks[kIs2K][kIs2Pools];  // 8 KB
    __shared__ alignas(16) float qs[kIs2K][kIs2RH];     // 16 KB
    __shared__ float dots[kIs2RH][kIs2Pools + 1];        // 33 KB (+1 column against bank conflicts)
    __shared__ float ws[kIs2RH];
    const unsigned p0 = blockIdx.x * kIs2Pools, r0 = blockIdx.y * kIs2Rows;
    const unsigned rlast = (r0 + kIs2Rows < C ? r0 + kIs2Rows : C) - 1;
    const unsigned Pmax = (t0 + rlast + 1) / kKpool;  // pools of the tile's last row (earlier rows have fewer)
    if (p0 >= Pmax || Pmax <= kIdxPools) return;    // uniform per block
    const unsigned tid = threadIdx.x, pg = tid % 16, rg = tid / 16;  // pools 4 pg .. + 3, (row, head) pairs 8 rg .. + 7
    float acc[4][8];
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) acc[a][b] = 0.0F;
    for (unsigned k0 = 0; k0 < kIdxDim; k0 += kIs2K) {
        __syncthreads();  // previous slice consumed
        for (unsigned x = tid; x < (kIs2K / 4) * kIs2Pools; x += 256) {  // lanes walk pools: conflict-free stores
            const unsigned pp = x % kIs2Pools, i4 = x / kIs2Pools;
            float4 v{0.0F, 0.0F, 0.0F, 0.0F};
            if (p0 + pp < Pmax) v = *reinterpret_cast<const float4*>(pooled + static_cast<std::size_t>(p0 + pp) * kIdxDim + k0 + 4 * i4);
            ks[4 * i4 + 0][pp] = v.x;
            ks[4 * i4 + 1][pp] = v.y;
            ks[4 * i4 + 2][pp] = v.z;
            ks[4 * i4 + 3][pp] = v.w;
        }
        for (unsigned x = tid; x < (kIs2K / 4) * kIs2RH; x += 256) {  // lanes walk (row, head): conflict-free stores
            const unsigned rh = x % kIs2RH, i4 = x / kIs2RH, r = r0 + rh / kIdxHeads, h = rh % kIdxHeads;
            float4 v{0.0F, 0.0F, 0.0F, 0.0F};
            if (r < C) v = *reinterpret_cast<const float4*>(qI + (static_cast<std::size_t>(r) * kIdxHeads + h) * kIdxDim + k0 + 4 * i4);
            qs[4 * i4 + 0][rh] = v.x;
            qs[4 * i4 + 1][rh] = v.y;
            qs[4 * i4 + 2][rh] = v.z;
            qs[4 * i4 + 3][rh] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned i = 0; i < kIs2K; ++i) {
            const float4 kv = *reinterpret_cast<const float4*>(&ks[i][4 * pg]);
            const float4 qa = *reinterpret_cast<const float4*>(&qs[i][8 * rg]), qb = *reinterpret_cast<const float4*>(&qs[i][8 * rg + 4]);
            const float kk[4] = {kv.x, kv.y, kv.z, kv.w}, qq[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 8; ++b) acc[a][b] = __builtin_fmaf(qq[b], kk[a], acc[a][b]);
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned a = 0; a < 4; ++a)
#pragma unroll
        for (unsigned b = 0; b < 8; ++b) dots[8 * rg + b][4 * pg + a] = acc[a][b];
    if (tid < kIs2RH) {
        const unsigned r = r0 + tid / kIdxHeads;
        ws[tid] = r < C ? wh[static_cast<std::size_t>(r) * kIdxHeads + tid % kIdxHeads] : 0.0F;
    }
    __syncthreads();
    const unsigned rr = tid / kIs2Pools, pp = tid % kIs2Pools, r = r0 + rr, p = p0 + pp;  // phase 2: thread = (row, pool)
    if (r >= C) return;
    const unsigned P = (t0 + r + 1) / kKpool;
    if (P <= kIdxPools || p >= P) return;
    float a2 = 0.0F;
    for (unsigned h = 0; h < kIdxHeads; ++h) {
        const float sc = dots[rr * kIdxHeads + h][pp] * kInvSqrt128;
        a2 = __builtin_fmaf(ws[rr * kIdxHeads + h], sc > 0.0F ? sc : 0.0F, a2);
    }
    keys[static_cast<std::size_t>(r) * cap + p] = ordered_key(a2);
}

// idx_select_kernel per row (blockIdx.x = row, t = t0 + row). P <= 512: every pool is selected, the list is 0 .. t.
__global__ __launch_bounds__(1024) void idx_select_rows_kernel(const std::uint32_t* keys_all, unsigned cap, unsigned t0,
                                                               std::uint32_t* tokens_all, std::uint32_t* n_all) {
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_prefix, sh_mask, sh_remaining, sh_carry_eq, sh_carry_out;
    __shared__ unsigned scan_eq[1024], scan_sel[1024];
    const unsigned l = threadIdx.x, row = blockIdx.x, t = t0 + row;
    const unsigned P = (t + 1) / kKpool;
    std::uint32_t* tokens = tokens_all + static_cast<std::size_t>(row) * kMaxTokens;
    if (P <= kIdxPools) {
        for (unsigned pos = l; pos <= t; pos += 1024) tokens[pos] = pos;
        if (l == 0) n_all[row] = t + 1;
        return;
    }
    const std::uint32_t* keys = keys_all + static_cast<std::size_t>(row) * cap;
    const unsigned k = kIdxPools;
    if (l == 0) {
        sh_prefix = 0;
        sh_mask = 0;
        sh_remaining = k;
        sh_carry_eq = 0;
        sh_carry_out = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned d = l; d < 256; d += 1024) hist[d] = 0;
        __syncthreads();
        const unsigned prefix = sh_prefix, mask = sh_mask;
        for (unsigned p = l; p < P; p += 1024)
            if ((keys[p] & mask) == prefix) atomicAdd(&hist[(keys[p] >> shift) & 255U], 1U);
        __syncthreads();
        if (l == 0) {
            unsigned rem = sh_remaining;
            for (int d = 255; d >= 0; --d) {
                if (hist[d] >= rem) {
                    sh_prefix = prefix | (static_cast<unsigned>(d) << shift);
                    break;
                }
                rem -= hist[d];
            }
            sh_remaining = rem;
            sh_mask = mask | (255U << shift);
        }
        __syncthreads();
    }
    const unsigned kstar = sh_prefix, take_eq = sh_remaining;
    for (unsigned base = 0; base < P; base += 1024) {
        const unsigned p = base + l;
        const unsigned key = p < P ? keys[p] : 0U;
        const unsigned eq = (p < P && key == kstar) ? 1U : 0U;
        scan_eq[l] = eq;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_eq[l - off] : 0U;
            __syncthreads();
            scan_eq[l] += v;
            __syncthreads();
        }
        const unsigned eq_rank = sh_carry_eq + scan_eq[l] - eq;
        const unsigned sel = (p < P && (key > kstar || (eq && eq_rank < take_eq))) ? 1U : 0U;
        scan_sel[l] = sel;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_sel[l - off] : 0U;
            __syncthreads();
            scan_sel[l] += v;
            __syncthreads();
        }
        if (sel) {
            const unsigned at = sh_carry_out + scan_sel[l] - 1;
            for (unsigned j = 0; j < kKpool; ++j) tokens[at * kKpool + j] = p * kKpool + j;
        }
        __syncthreads();
        if (l == 1023) {
            sh_carry_eq += scan_eq[1023];
            sh_carry_out += scan_sel[1023];
        }
        __syncthreads();
    }
    if (l == 0) {
        unsigned n = sh_carry_out * kKpool;
        for (unsigned pos = P * kKpool; pos <= t; ++pos) tokens[n++] = pos;
        n_all[row] = n;
    }
}

// ---------------------------------------------------------------- MLA (R13), split in four row-parallel stages
// (a) q-tilde = mla_absorb_kernel per row.
__global__ void mla_absorb_rows_kernel(const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt) {
    const unsigned l = blockIdx.x * 256 + threadIdx.x, h = blockIdx.y, r = blockIdx.z;
    if (l >= kKvLora) return;
    const float* qr = q + (static_cast<std::size_t>(r) * heads + h) * kMlaDim;
    float acc = 0.0F;
    for (unsigned d = 0; d < kMlaDim; ++d)
        acc = __builtin_fmaf(qr[d], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + d) * kKvLora + l]), acc);
    qt[(static_cast<std::size_t>(r) * heads + h) * kKvLora + l] = acc;
}
// (b) scores: grid (token block of 256, row); thread = list index j, all heads at once (one latent read per token);
// per (head, token) the chain over l ascending = mla_attn_kernel's, times the scale.
constexpr unsigned kLatBlk = 64;
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_rows_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                              const float* qt, float* e_all) {
    __shared__ float qs[H][kLatBlk];
    const unsigned r = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x, n = n_all[r];
    if (blockIdx.x * 256 >= n) return;  // uniform per block
    const bool act = j < n;
    const std::uint16_t* lat = sl.latent + static_cast<std::size_t>(act ? tokens_all[static_cast<std::size_t>(r) * kMaxTokens + j] : 0) * kKvLora;
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[H];
#pragma unroll
    for (unsigned h = 0; h < H; ++h) acc[h] = 0.0F;
    for (unsigned i0 = 0; i0 < kKvLora; i0 += kLatBlk) {
        __syncthreads();
        for (unsigned x = threadIdx.x; x < H * kLatBlk; x += 256) qs[x / kLatBlk][x % kLatBlk] = qr[(x / kLatBlk) * kKvLora + i0 + x % kLatBlk];
        __syncthreads();
        if (act)
            for (unsigned i = 0; i < kLatBlk; ++i) {
                const float c = bf16_to_float(lat[i0 + i]);
#pragma unroll
                for (unsigned h = 0; h < H; ++h) acc[h] = __builtin_fmaf(qs[h][i], c, acc[h]);
            }
    }
    if (act) {
        float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) e[h * kEStride + j] = acc[h] * kMlaScale;
    }
}
// (c) softmax per (head, row): mla_attn_kernel's max / exp / lane-strided sum / tree / divide, in place.
// kLane (o-bar v5): the normalized p are written lane-major instead, slot 9 * bitrev8(t % 256) + t / 256 (lane l's
// rounds m = 0..8 contiguous, lanes in k = bitrev8(l) order; slots of t >= n are +0, so all 2304 slots are written,
// each by its lane's thread). Same values and arithmetic; every thread reads only its own t, and the permuted stores
// start after a barrier that follows the last read.
__device__ __forceinline__ unsigned bitrev8(unsigned x) { return __brev(x) >> 24; }
template <bool kLane>
__global__ __launch_bounds__(256) void mla_softmax_rows_kernel(const std::uint32_t* n_all, unsigned heads, float* e_all) {
    __shared__ float s[256];
    const unsigned h = blockIdx.x, r = blockIdx.y, l = threadIdx.x, n = n_all[r];
    float* e = e_all + (static_cast<std::size_t>(r) * heads + h) * kEStride;
    float mx = -__builtin_huge_valf();
    for (unsigned t = l; t < n; t += 256) mx = e[t] > mx ? e[t] : mx;
    s[l] = mx;
    __syncthreads();
    for (unsigned hh = 128; hh > 0; hh >>= 1U) {
        if (l < hh) s[l] = s[l + hh] > s[l] ? s[l + hh] : s[l];
        __syncthreads();
    }
    const float m = s[0];
    __syncthreads();
    float a = 0.0F;
    for (unsigned t = l; t < n; t += 256) {
        e[t] = declared_exp(e[t] - m);
        a = a + e[t];
    }
    const float sum = lds_tree<256>(s, a);
    if constexpr (!kLane) {
        for (unsigned t = l; t < n; t += 256) e[t] = e[t] / sum;
    } else {
        float q[kObRounds];
#pragma unroll
        for (unsigned i = 0; i < kObRounds; ++i) q[i] = l + 256 * i < n ? e[l + 256 * i] / sum : 0.0F;
        __syncthreads();  // every thread's reads of e are done
        float* lane = e + kObRounds * bitrev8(l);
#pragma unroll
        for (unsigned i = 0; i < kObRounds; ++i) lane[i] = q[i];
    }
}
// (d) o-bar: grid (latent block of kObLat dims, row); lane j accumulates p_t * c_t[l] over t = j, j + 256, ... (fmaf,
// ascending t) for two heads x kObLat dims at a time, then 2 * kObLat lds_tree<256> pairings evaluated together.
constexpr unsigned kObLat = 16, kObTrees = 2 * kObLat;
__global__ __launch_bounds__(256) void mla_obar_rows_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            unsigned heads, const float* e_all, float* ob) {
    __shared__ float s[kObTrees][256];
    const unsigned lb = blockIdx.x * kObLat, r = blockIdx.y, j = threadIdx.x, n = n_all[r];
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    for (unsigned h0 = 0; h0 < heads; h0 += 2) {
        const float* e0 = e_all + (static_cast<std::size_t>(r) * heads + h0) * kEStride;
        const float* e1 = e0 + kEStride;
        float acc[2][kObLat];
#pragma unroll
        for (unsigned c = 0; c < kObLat; ++c) acc[0][c] = acc[1][c] = 0.0F;
        for (unsigned t = j; t < n; t += 256) {
            const std::uint16_t* lat = sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + lb;
            const float p0 = e0[t], p1 = e1[t];
#pragma unroll
            for (unsigned c = 0; c < kObLat; ++c) {
                const float cv = bf16_to_float(lat[c]);
                acc[0][c] = __builtin_fmaf(p0, cv, acc[0][c]);
                acc[1][c] = __builtin_fmaf(p1, cv, acc[1][c]);
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < kObLat; ++c) {
            s[c][j] = acc[0][c];
            s[kObLat + c][j] = acc[1][c];
        }
        __syncthreads();
        for (unsigned hh = 128; hh > 0; hh >>= 1U) {
            for (unsigned x = j; x < kObTrees * hh; x += 256) {
                const unsigned q = x / hh, l = x % hh;
                s[q][l] = s[q][l] + s[q][l + hh];
            }
            __syncthreads();
        }
        if (j < kObTrees) ob[(static_cast<std::size_t>(r) * heads + h0 + j / kObLat) * kKvLora + lb + j % kObLat] = s[j][0];
    }
}
// (e) output chain per (head, row), = the tail of mla_attn_kernel.
__global__ __launch_bounds__(256) void mla_out_rows_kernel(const float* ob, const std::uint16_t* kv_b, unsigned heads, float* o) {
    const unsigned h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const float* obr = ob + (static_cast<std::size_t>(r) * heads + h) * kKvLora;
    float acc = 0.0F;
    for (unsigned i = 0; i < kKvLora; ++i)
        acc = __builtin_fmaf(obr[i], f16_to_float(kv_b[static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + d) * kKvLora + i]), acc);
    o[(static_cast<std::size_t>(r) * heads + h) * kMlaDim + d] = acc;
}

// ---------------------------------------------------------------- MLA v2 (default): same per-output expressions, other
// placement. The absorb and output chains become tiled VALU GEMMs over rows (kv_b read once per 64 rows instead of per
// row); scores read q-tilde transposed with vector LDS loads; o-bar stages the row's latent tile once in LDS and runs the
// lds_tree<256> pairing as 3 LDS levels (128, 64, 32) + 5 shuffle levels (16 .. 1) inside wave 0.
constexpr unsigned kGT = 64, kGK = 32;  // GEMM tile: 64 rows x 64 cols, K step 32; thread = 4 x 4 outputs
// Y[r][h][c] = chain over k ascending of fmaf(A[r][h][k], float(W(h, k, c)), acc), k < K.
//   absorb: A = q [C][H][256], W(h, d, l) = kv_b[(h*512 + d)*512 + l], c = l < 512, K = 256
//   out:    A = ob [C][H][512], W(h, l, d) = kv_b[(h*512 + 256 + d)*512 + l], c = d < 256, K = 512
template <bool kOut>
__global__ __launch_bounds__(256) void mla_gemm_rows_kernel(const float* A, const std::uint16_t* kv_b, unsigned heads, unsigned C, float* Y) {
    constexpr unsigned K = kOut ? kKvLora : kMlaDim, NC = kOut ? kMlaDim : kKvLora;
    __shared__ float as[kGT][kGK + 1];
    __shared__ float ws[kGK][kGT + 1];
    const unsigned r0 = blockIdx.x * kGT, c0 = blockIdx.y * kGT, h = blockIdx.z, tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < K; k0 += kGK) {
        __syncthreads();
        for (unsigned x = tid; x < kGT * kGK; x += 256) {  // A tile [64 rows][32 k]
            const unsigned rr = x / kGK, kk = x % kGK, r = r0 + rr;
            as[rr][kk] = r < C ? A[(static_cast<std::size_t>(r) * heads + h) * K + k0 + kk] : 0.0F;
        }
        for (unsigned x = tid; x < kGK * kGT; x += 256) {  // W tile [32 k][64 c]
            std::size_t at;
            unsigned kk, cc;
            if (kOut) {  // kv_b row (h*512 + 256 + d), column l: consecutive x -> consecutive l (coalesced)
                cc = x / kGK;
                kk = x % kGK;
                at = static_cast<std::size_t>(h * 2 * kMlaDim + kMlaDim + c0 + cc) * kKvLora + k0 + kk;
            } else {     // kv_b row (h*512 + d), column l
                kk = x / kGT;
                cc = x % kGT;
                at = static_cast<std::size_t>(h * 2 * kMlaDim + k0 + kk) * kKvLora + c0 + cc;
            }
            ws[kk][cc] = f16_to_float(kv_b[at]);
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned kk = 0; kk < kGK; ++kk) {
            float a[4], w[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[tr * 4 + i][kk];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) w[j] = ws[kk][tc * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], w[j], acc[i][j]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r >= C) continue;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) Y[(static_cast<std::size_t>(r) * heads + h) * NC + c0 + tc * 4 + j] = acc[i][j];
    }
}
// Scores v2: thread = list index j; q-tilde of the row staged transposed [i][H] so 4 heads come in one LDS load; the
// token's latent read 8 dims at a time. Per (head, token) the chain over l ascending is unchanged.
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_v2_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            const float* qt, float* e_all) {
    __shared__ float4 qs[kLatBlk][H / 4];
    const unsigned r = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x, n = n_all[r];
    if (blockIdx.x * 256 >= n) return;  // uniform per block
    const bool act = j < n;
    const uint4* lat = reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(act ? tokens_all[static_cast<std::size_t>(r) * kMaxTokens + j] : 0) * kKvLora);
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[H];
#pragma unroll
    for (unsigned h = 0; h < H; ++h) acc[h] = 0.0F;
    for (unsigned i0 = 0; i0 < kKvLora; i0 += kLatBlk) {
        __syncthreads();
        for (unsigned x = threadIdx.x; x < kLatBlk * H; x += 256) {  // qs[i][h] = qt[h][i0 + i]
            const unsigned i = x % kLatBlk, h = x / kLatBlk;
            reinterpret_cast<float*>(qs[i])[h] = qr[h * kKvLora + i0 + i];
        }
        __syncthreads();
        if (act)
            for (unsigned i = 0; i < kLatBlk; i += 8) {
                const uint4 v = lat[(i0 + i) / 8];
                const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (unsigned ii = 0; ii < 8; ++ii) {
                    const float c = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
#pragma unroll
                    for (unsigned hq = 0; hq < H / 4; ++hq) {
                        const float4 q4 = qs[i + ii][hq];
                        acc[4 * hq + 0] = __builtin_fmaf(q4.x, c, acc[4 * hq + 0]);
                        acc[4 * hq + 1] = __builtin_fmaf(q4.y, c, acc[4 * hq + 1]);
                        acc[4 * hq + 2] = __builtin_fmaf(q4.z, c, acc[4 * hq + 2]);
                        acc[4 * hq + 3] = __builtin_fmaf(q4.w, c, acc[4 * hq + 3]);
                    }
                }
            }
    }
    if (act) {
        float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
        for (unsigned h = 0; h < H; ++h) e[h * kEStride + j] = acc[h] * kMlaScale;
    }
}
// Scores v3: CTA = 64 list entries x all heads of one row; thread = 2 entries (tp, tp + 32) x H / 8 heads (one wave
// shares its heads, so q-tilde comes as LDS broadcasts). Per 64-dim block the latent tile [dim][entry] (f32, the exact
// widening of the bf16 cache) and q-tilde [dim][head] are staged in LDS. Per (head, entry) the chain over l ascending
// from +0 is v2's (fmaf(q, c, acc)), times the scale: same bits.
constexpr unsigned kS3Tok = 64, kS3K = 64;
template <unsigned H>
__global__ __launch_bounds__(256) void mla_scores_v3_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                            const float* qt, float* e_all) {
    constexpr unsigned HPT = H / 8;  // heads per thread: 4 (TP2) or 8
    __shared__ float lt[kS3K][kS3Tok];
    __shared__ alignas(16) float qs[kS3K][H];
    const unsigned r = blockIdx.y, t0 = blockIdx.x * kS3Tok, n = n_all[r];
    if (t0 >= n) return;  // uniform per block
    const unsigned tid = threadIdx.x, tp = tid % 32, hg = tid / 32;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const float* qr = qt + static_cast<std::size_t>(r) * H * kKvLora;
    float acc[2][HPT];
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned b = 0; b < HPT; ++b) acc[a][b] = 0.0F;
    for (unsigned k0 = 0; k0 < kKvLora; k0 += kS3K) {
        __syncthreads();  // previous block's tiles consumed
        for (unsigned x = tid; x < kS3Tok * (kS3K / 8); x += 256) {  // lanes walk entries: conflict-free LDS stores
            const unsigned tt = x % kS3Tok, c8 = x / kS3Tok, j = t0 + tt;
            uint4 v{0U, 0U, 0U, 0U};
            if (j < n) v = *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[j]) * kKvLora + k0 + c8 * 8);
            const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (unsigned ii = 0; ii < 8; ++ii) lt[c8 * 8 + ii][tt] = bf16_to_float(static_cast<std::uint16_t>(wv[ii / 2] >> (16 * (ii % 2))));
        }
        for (unsigned x = tid; x < (kS3K / 4) * H; x += 256) {  // qs[i][h] = qt[h][k0 + i]; lanes walk heads: conflict-free stores
            const unsigned h = x % H, i4 = x / H;
            const float4 v = *reinterpret_cast<const float4*>(qr + static_cast<std::size_t>(h) * kKvLora + k0 + 4 * i4);
            qs[4 * i4 + 0][h] = v.x;
            qs[4 * i4 + 1][h] = v.y;
            qs[4 * i4 + 2][h] = v.z;
            qs[4 * i4 + 3][h] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned i = 0; i < kS3K; ++i) {
            const float c0 = lt[i][tp], c1 = lt[i][tp + 32];
#pragma unroll
            for (unsigned a = 0; a < HPT; a += 4) {
                const float4 q4 = *reinterpret_cast<const float4*>(&qs[i][hg * HPT + a]);
                acc[0][a + 0] = __builtin_fmaf(q4.x, c0, acc[0][a + 0]);
                acc[0][a + 1] = __builtin_fmaf(q4.y, c0, acc[0][a + 1]);
                acc[0][a + 2] = __builtin_fmaf(q4.z, c0, acc[0][a + 2]);
                acc[0][a + 3] = __builtin_fmaf(q4.w, c0, acc[0][a + 3]);
                acc[1][a + 0] = __builtin_fmaf(q4.x, c1, acc[1][a + 0]);
                acc[1][a + 1] = __builtin_fmaf(q4.y, c1, acc[1][a + 1]);
                acc[1][a + 2] = __builtin_fmaf(q4.z, c1, acc[1][a + 2]);
                acc[1][a + 3] = __builtin_fmaf(q4.w, c1, acc[1][a + 3]);
            }
        }
    }
    float* e = e_all + static_cast<std::size_t>(r) * H * kEStride;
#pragma unroll
    for (unsigned sidx = 0; sidx < 2; ++sidx) {
        const unsigned j = t0 + tp + 32 * sidx;
        if (j < n)
#pragma unroll
            for (unsigned a = 0; a < HPT; ++a) e[static_cast<std::size_t>(hg * HPT + a) * kEStride + j] = acc[sidx][a] * kMlaScale;
    }
}

// o-bar v4: no cross-thread reduction tree. tree_fma<256>'s fold s[l] += s[l + h] (h = 128 .. 1) is the adjacent-pair
// balanced tree over k = bitrev8(l) (operand order included; design notes section 2, checked symbolically), so one
// thread evaluates it exactly by walking lanes in k order and merging their chains in a binary counter (left = the
// older, lower-k value). CTA = 8 heads x 64 latent dims of one row; thread = 2 heads x 4 dims (8 outputs) of one
// quarter of k: sub-group s owns k in [64 s, 64 s + 64), i.e. lanes l = bitrev8(k) (l mod 4 fixed), and builds that
// subtree (6 levels); the 4 roots then combine as (R0 + R1) + (R2 + R3), the tree's top 2 levels. Per lane the chain is
// the contract's: fmaf(p_t, c_t, a) over t = l, l + 256, ... ascending from +0 (lanes without tokens stay +0).
// Staging per batch of 8 k-steps: the 32 lanes' tokens (9 slots per lane: t = l + 256 m, m < 9 since n <= 2051), their
// 64 latent dims (bf16) and the 8 heads' p, in LDS.
constexpr unsigned kOb4Heads = 8, kOb4Dims = 64, kOb4Steps = 8, kOb4M = 9, kOb4Slots = 4 * kOb4Steps * kOb4M;  // 288
static_assert((kMaxTokens + 255) / 256 <= kOb4M, "o-bar v4: 9 rounds of 256 cover the list");
__global__ __launch_bounds__(256) void mla_obar_v4_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) std::uint16_t lt[kOb4Slots][kOb4Dims];  // 36 KB
    __shared__ float ps[kOb4Heads][kOb4Slots];                      // 9 KB
    __shared__ float roots[4][kOb4Heads][kOb4Dims];                 // 8 KB: subtree roots of sub-groups 0..3
    const unsigned hb = blockIdx.x * kOb4Heads, d0 = blockIdx.y * kOb4Dims, r = blockIdx.z, n = n_all[r];
    const unsigned tid = threadIdx.x, dq = tid % 16, hp = (tid / 16) % 4, sg = tid / 64;
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    float stk[2][4][7];  // per output: binary-counter stack of the sub-group's 64-lane subtree
    unsigned cnt = 0;
    for (unsigned b = 0; b < 64 / kOb4Steps; ++b) {
        __syncthreads();  // previous batch consumed
        // stage: slot = (s * 8 + j) * 9 + m  <->  token t = bitrev8(64 s + 8 b + j) + 256 m
        for (unsigned x = tid; x < kOb4Slots * (kOb4Dims / 8); x += 256) {
            const unsigned slot = x / (kOb4Dims / 8), c8 = x % (kOb4Dims / 8);
            const unsigned lane = slot / kOb4M, m = slot % kOb4M;
            const unsigned t = bitrev8(64 * (lane / kOb4Steps) + kOb4Steps * b + lane % kOb4Steps) + 256 * m;
            if (t < n)
                *reinterpret_cast<uint4*>(&lt[slot][c8 * 8]) =
                    *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + c8 * 8);
        }
        for (unsigned x = tid; x < kOb4Heads * kOb4Slots; x += 256) {
            const unsigned h = x / kOb4Slots, slot = x % kOb4Slots;
            const unsigned lane = slot / kOb4M, m = slot % kOb4M;
            const unsigned t = bitrev8(64 * (lane / kOb4Steps) + kOb4Steps * b + lane % kOb4Steps) + 256 * m;
            if (t < n) ps[h][slot] = e_all[(static_cast<std::size_t>(r) * heads + hb + h) * kEStride + t];
        }
        __syncthreads();
        for (unsigned j = 0; j < kOb4Steps; ++j) {
            const unsigned l = bitrev8(64 * sg + kOb4Steps * b + j), slot0 = (sg * kOb4Steps + j) * kOb4M;
            float c[2][4];
#pragma unroll
            for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
            for (unsigned m = 0; m < kOb4M && l + 256 * m < n; ++m) {
                const uint2 v = *reinterpret_cast<const uint2*>(&lt[slot0 + m][dq * 4]);
                const float cv[4] = {bf16_to_float(static_cast<std::uint16_t>(v.x)), bf16_to_float(static_cast<std::uint16_t>(v.x >> 16)),
                                     bf16_to_float(static_cast<std::uint16_t>(v.y)), bf16_to_float(static_cast<std::uint16_t>(v.y >> 16))};
                const float p0 = ps[2 * hp][slot0 + m], p1 = ps[2 * hp + 1][slot0 + m];
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    c[0][q] = __builtin_fmaf(p0, cv[q], c[0][q]);
                    c[1][q] = __builtin_fmaf(p1, cv[q], c[1][q]);
                }
            }
            // binary-counter merge of this lane's chains: carry through the counter's set low bits, place at the first
            // clear one. Levels are compile-time indices (registers, no scratch); cnt is uniform, so no divergence.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 7; ++L) {
                if (placed) continue;
                if ((cnt >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            ++cnt;
        }
    }
    // 64 lanes merged: the sub-group's subtree root sits at level 6
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
        for (unsigned q = 0; q < 4; ++q) roots[sg][2 * hp + a][dq * 4 + q] = stk[a][q][6];
    __syncthreads();
    for (unsigned x = tid; x < kOb4Heads * kOb4Dims; x += 256) {
        const unsigned h = x / kOb4Dims, d = x % kOb4Dims;
        const float v = (roots[0][h][d] + roots[1][h][d]) + (roots[2][h][d] + roots[3][h][d]);
        ob[(static_cast<std::size_t>(r) * heads + hb + h) * kKvLora + d0 + d] = v;
    }
}

// o-bar v5 (heads 32): CTA = all 32 heads x 64 latent dims of one row (latent gathered once per row); thread = 2 heads
// x 4 dims over the full 256-lane tree. Lanes in k order (l = bitrev8(k)): lane l's chain fmaf(p_t, c_t, a) over
// t = l + 256 m ascending from +0, only for t < n (no padding FMAs), then the binary-counter merge (older / lower-k
// subtree + newer one) of design notes section 2: the 256-lane adjacent-pair tree = tree_fma<256>'s fold.
// p comes lane-major from mla_softmax_rows_kernel<true> (a batch of 8 k is 72 contiguous slots per head). Staging is
// double-buffered: batch b + 1's p / latent are loaded into registers before batch b is computed and stored to the
// other buffer after it; one barrier per batch.
constexpr unsigned kOb5Dims = 64, kOb5Steps = 8, kOb5Slots = kOb5Steps * kObRounds;  // 72
constexpr unsigned kOb5PVec = 32 * kOb5Slots / 4, kOb5LVec = kOb5Slots * kOb5Dims / 8;  // 576 float4, 576 uint4
__global__ __launch_bounds__(256) void mla_obar_v5_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all,
                                                          const float* e_all, float* ob) {
    __shared__ alignas(16) std::uint16_t lt[2][kOb5Slots][kOb5Dims];  // 18 KB
    __shared__ alignas(16) float ps[2][kOb5Slots][32];                 // 18 KB
    const unsigned d0 = blockIdx.x * kOb5Dims, r = blockIdx.y, n = n_all[r];
    const unsigned tid = threadIdx.x, dq = tid % 16, hp = tid / 16;  // dims d0 + 4 dq .. + 3, heads 2 hp, 2 hp + 1
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    const float* ep = e_all + static_cast<std::size_t>(r) * 32 * kEStride;
    float4 pr[3];
    uint4 lr[3];
    auto load = [&](unsigned b) {  // batch b (k = 8 b .. 8 b + 7) into registers
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb5PVec) {
                const unsigned h = x / (kOb5Slots / 4), q = x % (kOb5Slots / 4);
                pr[i] = *reinterpret_cast<const float4*>(ep + static_cast<std::size_t>(h) * kEStride + kOb5Slots * b + 4 * q);
            }
            if (x < kOb5LVec) {
                const unsigned slot = x / (kOb5Dims / 8), c8 = x % (kOb5Dims / 8);
                const unsigned t = bitrev8(kOb5Steps * b + slot / kObRounds) + 256 * (slot % kObRounds);
                lr[i] = t < n ? *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + d0 + 8 * c8)
                              : uint4{0, 0, 0, 0};
            }
        }
    };
    auto store = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
            const unsigned x = tid + 256 * i;
            if (x < kOb5PVec) {
                const unsigned h = x / (kOb5Slots / 4), q = x % (kOb5Slots / 4);
                ps[buf][4 * q + 0][h] = pr[i].x;
                ps[buf][4 * q + 1][h] = pr[i].y;
                ps[buf][4 * q + 2][h] = pr[i].z;
                ps[buf][4 * q + 3][h] = pr[i].w;
            }
            if (x < kOb5LVec) {
                const unsigned slot = x / (kOb5Dims / 8), c8 = x % (kOb5Dims / 8);
                *reinterpret_cast<uint4*>(&lt[buf][slot][8 * c8]) = lr[i];
            }
        }
    };
    float stk[2][4][8];  // binary-counter levels 0..7; the root (level 8) is formed by the last merge (k = 255)
    float root[2][4];
    load(0);
    store(0);
    __syncthreads();
    for (unsigned b = 0; b < 256 / kOb5Steps; ++b) {
        const unsigned cur = b & 1U;
        if (b + 1 < 256 / kOb5Steps) load(b + 1);
        for (unsigned j = 0; j < kOb5Steps; ++j) {
            const unsigned k = kOb5Steps * b + j, l = bitrev8(k);
            float c[2][4];
#pragma unroll
            for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) c[a][q] = 0.0F;
            for (unsigned m = 0; m < kObRounds && l + 256 * m < n; ++m) {
                const unsigned slot = kObRounds * j + m;
                const uint2 v = *reinterpret_cast<const uint2*>(&lt[cur][slot][4 * dq]);
                const float cv[4] = {bf16_to_float(static_cast<std::uint16_t>(v.x)), bf16_to_float(static_cast<std::uint16_t>(v.x >> 16)),
                                     bf16_to_float(static_cast<std::uint16_t>(v.y)), bf16_to_float(static_cast<std::uint16_t>(v.y >> 16))};
                const float2 pp = *reinterpret_cast<const float2*>(&ps[cur][slot][2 * hp]);
#pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    c[0][q] = __builtin_fmaf(pp.x, cv[q], c[0][q]);
                    c[1][q] = __builtin_fmaf(pp.y, cv[q], c[1][q]);
                }
            }
            // merge: carry through the set low bits of k (left operand = the stored, lower-k subtree), place at the
            // first clear bit; k = 255 carries through all 8 levels into the root. Levels are compile-time indices.
            bool placed = false;
#pragma unroll
            for (unsigned L = 0; L < 8; ++L) {
                if (placed) continue;
                if ((k >> L) & 1U) {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) c[a][q] = stk[a][q][L] + c[a][q];
                } else {
#pragma unroll
                    for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) stk[a][q][L] = c[a][q];
                    placed = true;
                }
            }
            if (!placed)
#pragma unroll
                for (unsigned a = 0; a < 2; ++a)
#pragma unroll
                    for (unsigned q = 0; q < 4; ++q) root[a][q] = c[a][q];
        }
        if (b + 1 < 256 / kOb5Steps) store(cur ^ 1U);
        __syncthreads();
    }
#pragma unroll
    for (unsigned a = 0; a < 2; ++a)
        *reinterpret_cast<float4*>(ob + (static_cast<std::size_t>(r) * 32 + 2 * hp + a) * kKvLora + d0 + 4 * dq) =
            float4{root[a][0], root[a][1], root[a][2], root[a][3]};
}

// o-bar v2: grid (latent block of 8 dims, row). The row's latent tile [n][8] bf16 is staged in LDS once; per pass of 4
// heads, lane j accumulates 4 x 8 fmaf chains over t = j, j + 256, ... (ascending t), then lds_tree<256>'s pairing:
// levels 128 / 64 / 32 through LDS, levels 16 .. 1 by __shfl_down inside wave 0 (lane l < h receives lane l + h).
constexpr unsigned kOb2Lat = 8, kOb2Heads = 4, kOb2Vals = kOb2Lat * kOb2Heads;
__global__ __launch_bounds__(256) void mla_obar_v2_kernel(DsaSlot sl, const std::uint32_t* tokens_all, const std::uint32_t* n_all, unsigned heads,
                                                          const float* e_all, float* ob) {
    __shared__ uint4 lt[kMaxTokens];             // latent dims [lb, lb + 8) of each listed token (16 B)
    __shared__ float buf[kOb2Vals][128];
    const unsigned lb = blockIdx.x * kOb2Lat, r = blockIdx.y, j = threadIdx.x, n = n_all[r];
    const std::uint32_t* tok = tokens_all + static_cast<std::size_t>(r) * kMaxTokens;
    for (unsigned t = j; t < n; t += 256) lt[t] = *reinterpret_cast<const uint4*>(sl.latent + static_cast<std::size_t>(tok[t]) * kKvLora + lb);
    for (unsigned h0 = 0; h0 < heads; h0 += kOb2Heads) {
        float acc[kOb2Heads][kOb2Lat];
#pragma unroll
        for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
            for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = 0.0F;
        __syncthreads();  // lt staged (first pass) / buf free (later passes)
        for (unsigned t = j; t < n; t += 256) {
            const uint4 v = lt[t];
            const std::uint32_t wv[4] = {v.x, v.y, v.z, v.w};
            float p[kOb2Heads];
#pragma unroll
            for (unsigned a = 0; a < kOb2Heads; ++a) p[a] = e_all[(static_cast<std::size_t>(r) * heads + h0 + a) * kEStride + t];
#pragma unroll
            for (unsigned c = 0; c < kOb2Lat; ++c) {
                const float cv = bf16_to_float(static_cast<std::uint16_t>(wv[c / 2] >> (16 * (c % 2))));
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a) acc[a][c] = __builtin_fmaf(p[a], cv, acc[a][c]);
            }
        }
        for (unsigned hh = 128; hh >= 32; hh >>= 1U) {  // cross-wave levels: lanes [hh, 2hh) hand their values to l - hh
            if (j >= hh && j < 2 * hh)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) buf[a * kOb2Lat + c][j - hh] = acc[a][c];
            __syncthreads();
            if (j < hh)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = acc[a][c] + buf[a * kOb2Lat + c][j];
            __syncthreads();
        }
        if (j < 32) {
#pragma unroll
            for (unsigned hh = 16; hh > 0; hh >>= 1U)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) acc[a][c] = acc[a][c] + __shfl_down(acc[a][c], hh, 32);
            if (j == 0)
#pragma unroll
                for (unsigned a = 0; a < kOb2Heads; ++a)
#pragma unroll
                    for (unsigned c = 0; c < kOb2Lat; ++c) ob[(static_cast<std::size_t>(r) * heads + h0 + a) * kKvLora + lb + c] = acc[a][c];
        }
    }
}

// R1 chains over rows: tiled VALU GEMM, 64 rows x 64 outputs per CTA, K step 32, thread = 4 x 4 outputs; per output
// the chain of dense_kernel (fmaf(x[i], w[o][i], acc), i ascending, from +0).
__device__ __forceinline__ float wload_pf(const void* w, dev::WType t, std::size_t k) {
    switch (t) {
        case dev::WType::F16: return f16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        case dev::WType::BF16: return bf16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        default: return static_cast<const float*>(w)[k];
    }
}
__global__ __launch_bounds__(256) void dense_rows_kernel(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n,
                                                         float* y, unsigned rows) {
    __shared__ float as[kGT][kGK + 1];
    __shared__ float ws[kGK][kGT + 1];
    const unsigned r0 = blockIdx.x * kGT, c0 = blockIdx.y * kGT, tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < in; k0 += kGK) {
        __syncthreads();
        for (unsigned v = tid; v < kGT * kGK; v += 256) {
            const unsigned rr = v / kGK, kk = v % kGK, r = r0 + rr;
            as[rr][kk] = r < rows ? x[static_cast<std::size_t>(r) * in + k0 + kk] : 0.0F;
            const unsigned o = c0 + rr;  // the same index pair for the weight tile: row o, column k0 + kk
            ws[kk][rr] = o < n ? wload_pf(w, wt, static_cast<std::size_t>(o0 + o) * in + k0 + kk) : 0.0F;
        }
        __syncthreads();
#pragma unroll 4
        for (unsigned kk = 0; kk < kGK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[tr * 4 + i][kk];
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) b[j] = ws[kk][tc * 4 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r >= rows) continue;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned o = c0 + tc * 4 + j;
            if (o < n) y[static_cast<std::size_t>(r) * n + o] = acc[i][j];
        }
    }
}

// hc_pre split in three launches (expressions of hc_pre_kernel): (1) per-row unweighted norm -> F; (2) the 24 chains
// z = F . fn as dense_rows; (3) per-row R6 on lane 0 + R7 collapse.
__global__ __launch_bounds__(256) void hc_norm_rows_kernel(const float* X, float* F) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
}
// Serial R6, shared by both work placements. Preserve every expression and loop order.
__device__ __forceinline__ void hc_mix_r6(const float* z, const float* base, const float* scale, float* m) {
    for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
    for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
    float* c = m + 2 * kHc;
    for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
    for (unsigned j = 0; j < kHc; ++j) {
        float mx = c[j * kHc];
        for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
        float e[kHc];
        for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
    }
    for (unsigned it = 0; it < kHcIters; ++it) {
        if (it > 0) {
            float sr[kHc];
            for (unsigned j = 0; j < kHc; ++j)
                sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
        }
        float sc[kHc];
        for (unsigned k = 0; k < kHc; ++k)
            sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j)
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
    }
}
__global__ __launch_bounds__(256) void hc_mix_rows_kernel(const float* X, const float* zg, const float* base, const float* scale, float* mix,
                                                          float* xc) {
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    if (l == 0) {  // R6 on lane 0 (host hc_mix order) = hc_pre_kernel
        const float* z = zg + static_cast<std::size_t>(r) * kHcMix;
        hc_mix_r6(z, base, scale, m);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

// V2 R1: 64 rows x 24 outputs, K=32, 128 threads, 4 x 3 live chains per thread.
// Each stage is loaded once per CTA. Prefetch the next tile to registers before consuming
// the current LDS stage, then publish to the other stage. The tile-end barrier protects
// both reuse of the old stage and reads of the newly filled stage.
constexpr unsigned kHcK = 32, kHcMixRows = 16;
static_assert(kHc * kHidden == 16384 && kHcMix == 24);
// Rows-row tiles (64 / 32 / 16), 2 * Rows threads, 4 x 3 chains per thread: smaller tiles give more CTAs (the K chain
// cannot be split), e.g. 2048 rows -> 32 / 64 / 128 CTAs. Every tile size gives the same bits (one fmaf chain per
// (row, output), k ascending from +0).
template <unsigned Rows>
__device__ __forceinline__ void hc_chain_prefetch(const float* F, const float* fn, float* next_a, float* next_w,
                                                 unsigned r0, unsigned rows, unsigned k0) {
    constexpr unsigned n = kHc * kHidden, T = 2 * Rows;
    const unsigned tid = threadIdx.x;
#pragma unroll
    for (unsigned q = 0; q < Rows * kHcK / T; ++q) {
        const unsigned v = tid + q * T, r = v / kHcK, k = v % kHcK;
        next_a[q] = r0 + r < rows ? F[static_cast<std::size_t>(r0 + r) * n + k0 + k] : 0.0F;
    }
#pragma unroll
    for (unsigned q = 0; q < kHcMix * kHcK / T; ++q) {
        const unsigned v = tid + q * T, o = v / kHcK, k = v % kHcK;
        next_w[q] = fn[static_cast<std::size_t>(o) * n + k0 + k];
    }
}
template <unsigned Rows>
__device__ __forceinline__ void hc_chain_stage(const float* next_a, const float* next_w,
                                              float as[][kHcK + 1], float ws[][kHcMix + 1]) {
    constexpr unsigned T = 2 * Rows;
    const unsigned tid = threadIdx.x;
#pragma unroll
    for (unsigned q = 0; q < Rows * kHcK / T; ++q) {
        const unsigned v = tid + q * T;
        as[v / kHcK][v % kHcK] = next_a[q];
    }
#pragma unroll
    for (unsigned q = 0; q < kHcMix * kHcK / T; ++q) {
        const unsigned v = tid + q * T;
        ws[v % kHcK][v / kHcK] = next_w[q];
    }
}
template <unsigned Rows>
__global__ __launch_bounds__(2 * Rows) void hc_chain_rows_v2_kernel(const float* F, const float* fn, float* z, unsigned rows) {
    constexpr unsigned n = kHc * kHidden, T = 2 * Rows;
    static_assert(kHcMix * kHcK % T == 0 && Rows * kHcK % T == 0);
    __shared__ float as[2][Rows][kHcK + 1];
    __shared__ float ws[2][kHcK][kHcMix + 1];
    const unsigned r0 = blockIdx.x * Rows, tid = threadIdx.x, tr = tid / 8, tc = tid % 8;
    float acc[4][3];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 3; ++j) acc[i][j] = 0.0F;
    float next_a[Rows * kHcK / T], next_w[kHcMix * kHcK / T];
    hc_chain_prefetch<Rows>(F, fn, next_a, next_w, r0, rows, 0);
    hc_chain_stage<Rows>(next_a, next_w, as[0], ws[0]);
    __syncthreads();
    for (unsigned k0 = 0; k0 < n; k0 += kHcK) {
        const unsigned stage = (k0 / kHcK) % 2;
        if (k0 + kHcK < n)
            hc_chain_prefetch<Rows>(F, fn, next_a, next_w, r0, rows, k0 + kHcK);
#pragma unroll 4
        for (unsigned k = 0; k < kHcK; ++k) {
            float a[4], b[3];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[i] = as[stage][tr * 4 + i][k];
#pragma unroll
            for (unsigned j = 0; j < 3; ++j) b[j] = ws[stage][k][tc * 3 + j];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 3; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
        }
        if (k0 + kHcK < n) hc_chain_stage<Rows>(next_a, next_w, as[1 - stage], ws[1 - stage]);
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned r = r0 + tr * 4 + i;
        if (r < rows)
#pragma unroll
            for (unsigned j = 0; j < 3; ++j) z[static_cast<std::size_t>(r) * kHcMix + tc * 3 + j] = acc[i][j];
    }
}
unsigned hc_rows_tile() {  // STRIX_GLM_PF_HC_TILE = 64 | 32 | 16 (same bits), default 32
    static const unsigned t = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC_TILE");
        if (e && std::string(e) == "64") return 64U;
        if (e && std::string(e) == "16") return 16U;
        return 32U;
    }();
    return t;
}

// One lane per row executes the shared serial R6; all lanes then share the collapse.
__global__ __launch_bounds__(256) void hc_mix_rows_v2_kernel(const float* X, const float* zg, const float* base,
                                                            const float* scale, float* mix, float* xc, unsigned rows) {
    __shared__ float m[kHcMixRows][kHcMix];
    const unsigned r0 = blockIdx.x * kHcMixRows, tid = threadIdx.x;
    if (tid < kHcMixRows && r0 + tid < rows) {
        const unsigned r = r0 + tid;
        hc_mix_r6(zg + static_cast<std::size_t>(r) * kHcMix, base, scale, m[tid]);
    }
    __syncthreads();
    for (unsigned v = tid; v < kHcMixRows * kHcMix; v += 256) {
        const unsigned rr = v / kHcMix, k = v % kHcMix;
        if (r0 + rr < rows) mix[static_cast<std::size_t>(r0 + rr) * kHcMix + k] = m[rr][k];
    }
    for (unsigned v = tid; v < kHcMixRows * kHidden; v += 256) {
        const unsigned rr = v / kHidden, d = v % kHidden, r = r0 + rr;
        if (r < rows) {
            const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
            xc[static_cast<std::size_t>(r) * kHidden + d] =
                ((m[rr][0] * Xr[d] + m[rr][1] * Xr[kHidden + d]) + m[rr][2] * Xr[2 * kHidden + d]) + m[rr][3] * Xr[3 * kHidden + d];
        }
    }
}

unsigned hc_version() {  // default v2 (measurement: PF-K 30/0 + synthetic 6-layer engine gate PASS, 2.72 -> 1.75 us/row/site); STRIX_GLM_PF_HC=1: v1
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_HC");
        return e && std::string(e) == "1" ? 1U : 2U;
    }();
    return v;
}

bool mla_v1() {
    static const bool v1 = [] {
        const char* e = std::getenv("STRIX_GLM_PF_MLA");
        return e && std::string(e) == "1";
    }();
    return v1;
}
bool idx_v1() {  // STRIX_GLM_PF_IDX=1: the v1 indexer score kernel (same bits); default v2
    static const bool v1 = [] {
        const char* e = std::getenv("STRIX_GLM_PF_IDX");
        return e && std::string(e) == "1";
    }();
    return v1;
}
unsigned obar_version() {  // default 5 (heads 32; heads 64 runs v2); STRIX_GLM_PF_OBAR=2 / 4: v2 / v4 (all same bits;
    // measurement 17:23: v5 18.1 vs v2 53.9 us/row per DSA layer at the 32K chunk, 11.9 vs 32.9 at t0 0; measurement: v4 5 % slower than v2)
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_OBAR");
        if (e && std::string(e) == "4") return 4U;
        if (e && std::string(e) == "2") return 2U;
        return 5U;
    }();
    return v;
}
bool scores_v2() {  // STRIX_GLM_PF_SCORES=2: the v2 scores kernel (same bits); default v3
    static const bool v2 = [] {
        const char* e = std::getenv("STRIX_GLM_PF_SCORES");
        return e && std::string(e) == "2";
    }();
    return v2;
}

}  // namespace

void kda_prefill(const KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
                 float* conv_state, float* scratch, float* beta, float* o, float* y, unsigned* flag, unsigned C,
                 hipStream_t s) {
    if (C == 0) return;
    const unsigned channels = 3 * k.heads * kKdaDim;
    kda_prep_rows_kernel<<<dim3(k.heads, C), 128, 0, s>>>(k, qkv, f, b, conv_state, scratch, beta);
    ok(hipGetLastError(), "glm pf kda_prep_rows");
    kda_conv_commit_kernel<<<(channels + 255) / 256, 256, 0, s>>>(qkv, conv_state, channels, C);
    ok(hipGetLastError(), "glm pf kda_conv_commit");
    kda_scan_kernel<<<dim3(k.heads, kKdaDim / kScanV), kScanLanes * kScanV, 0, s>>>(k.heads, S, scratch, beta, o, flag, C);
    ok(hipGetLastError(), "glm pf kda_scan");
    kda_out_rows_kernel<<<dim3(k.heads, C), 128, 0, s>>>(k, o, gate, y);
    ok(hipGetLastError(), "glm pf kda_out_rows");
}

void dsa_append_rows(const DsaSlot& slot, unsigned t0, const float* latent, const float* key, const float* gate,
                     const IndexerDev& w, unsigned* flag, unsigned C, hipStream_t s) {
    if (C == 0) return;
    dsa_store_rows_kernel<<<C, 128, 0, s>>>(slot, t0, latent, w, key, gate, flag);
    ok(hipGetLastError(), "glm pf dsa_store_rows");
    const unsigned p_lo = t0 / kKpool, p_hi = (t0 + C) / kKpool;
    if (p_hi > p_lo) {
        dsa_pool_kernel<<<p_hi - p_lo, 128, 0, s>>>(slot, p_lo, w, flag);
        ok(hipGetLastError(), "glm pf dsa_pool");
    }
}

void indexer_select_rows(const DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                         unsigned kpool_cap, std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s) {
    if (C == 0) return;
    const unsigned Pmax = (t0 + C) / kKpool;
    if (Pmax > kpool_cap) throw std::runtime_error("glm pf indexer_select_rows: key workspace too small");
    if (Pmax > kIdxPools) {
        scale_kernel<<<(C * kIdxHeads + 255) / 256, 256, 0, s>>>(wh, C * kIdxHeads, kInvSqrt32);
        ok(hipGetLastError(), "glm pf scale");
        if (idx_v1())
            idx_score_rows_kernel<<<dim3((Pmax + 127) / 128, (C + kScoreRows - 1) / kScoreRows), 128, 0, s>>>(
                slot.pooled, t0, C, qI, wh, keys, kpool_cap);
        else
            idx_score_rows_v2_kernel<<<dim3((Pmax + kIs2Pools - 1) / kIs2Pools, (C + kIs2Rows - 1) / kIs2Rows), 256, 0, s>>>(
                slot.pooled, t0, C, qI, wh, keys, kpool_cap);
        ok(hipGetLastError(), "glm pf idx_score_rows");
    }
    idx_select_rows_kernel<<<C, 1024, 0, s>>>(keys, kpool_cap, t0, tokens, n);
    ok(hipGetLastError(), "glm pf idx_select_rows");
}

void dense_rows(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n, float* y, unsigned rows,
                hipStream_t s) {
    if (rows == 0 || n == 0) return;
    if (in == 0 || in % kGK) throw std::runtime_error("glm pf dense_rows: in must be a positive multiple of 32");
    dense_rows_kernel<<<dim3((rows + kGT - 1) / kGT, (n + kGT - 1) / kGT), 256, 0, s>>>(x, w, wt, in, o0, n, y, rows);
    ok(hipGetLastError(), "glm pf dense_rows");
}

void hc_pre_rows_v1(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                 unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    hc_norm_rows_kernel<<<rows, 256, 0, s>>>(X, F);
    ok(hipGetLastError(), "glm pf hc_norm_rows");
    dense_rows(F, fn, dev::WType::F32, kHc * kHidden, 0, kHcMix, z, rows, s);
    hc_mix_rows_kernel<<<rows, 256, 0, s>>>(X, z, base, scale, mix, xc);
    ok(hipGetLastError(), "glm pf hc_mix_rows");
}

void hc_pre_rows_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                    unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    hc_norm_rows_kernel<<<rows, 256, 0, s>>>(X, F);
    ok(hipGetLastError(), "glm pf hc_norm_rows v2");
    const unsigned t = hc_rows_tile();
    if (t == 64) hc_chain_rows_v2_kernel<64><<<(rows + 63) / 64, 128, 0, s>>>(F, fn, z, rows);
    else if (t == 16) hc_chain_rows_v2_kernel<16><<<(rows + 15) / 16, 32, 0, s>>>(F, fn, z, rows);
    else hc_chain_rows_v2_kernel<32><<<(rows + 31) / 32, 64, 0, s>>>(F, fn, z, rows);
    ok(hipGetLastError(), "glm pf hc_chain_rows v2");
    hc_mix_rows_v2_kernel<<<(rows + kHcMixRows - 1) / kHcMixRows, 256, 0, s>>>(X, z, base, scale, mix, xc, rows);
    ok(hipGetLastError(), "glm pf hc_mix_rows v2");
}

void hc_pre_rows(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                 unsigned rows, hipStream_t s) {
    if (hc_version() == 2) hc_pre_rows_v2(X, fn, base, scale, F, z, mix, xc, rows, s);
    else hc_pre_rows_v1(X, fn, base, scale, F, z, mix, xc, rows, s);
}

namespace {
hipEvent_t* g_mla_prof = nullptr;  // 6 events: before absorb, scores, softmax, o-bar, out, after
void mla_mark(unsigned i, hipStream_t s) {
    if (g_mla_prof) ok(hipEventRecord(g_mla_prof[i], s), "glm pf mla profile event");
}
}  // namespace
void set_mla_profile(hipEvent_t* ev6) { g_mla_prof = ev6; }
bool mla_e_lane_major(unsigned heads) { return !mla_v1() && obar_version() == 5 && heads == 32; }

void mla_attention_rows(const DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                        const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                        hipStream_t s) {
    if (C == 0) return;
    if (heads != 32 && heads != 64) throw std::runtime_error("glm pf mla_attention_rows: heads must be 32 or 64");
    const bool v1 = mla_v1();  // STRIX_GLM_PF_MLA=1: the v1 kernels (same bits)
    const dim3 sg((kMaxTokens + 255) / 256, C);
    mla_mark(0, s);
    if (v1) {
        mla_absorb_rows_kernel<<<dim3(kKvLora / 256, heads, C), 256, 0, s>>>(q, kv_b, heads, qt);
        mla_mark(1, s);
        if (heads == 32) mla_scores_rows_kernel<32><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
        else mla_scores_rows_kernel<64><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
    } else {
        mla_gemm_rows_kernel<false><<<dim3((C + kGT - 1) / kGT, kKvLora / kGT, heads), 256, 0, s>>>(q, kv_b, heads, C, qt);
        mla_mark(1, s);
        if (scores_v2()) {
            if (heads == 32) mla_scores_v2_kernel<32><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
            else mla_scores_v2_kernel<64><<<sg, 256, 0, s>>>(slot, tokens, n, qt, e);
        } else {
            const dim3 g3((kMaxTokens + kS3Tok - 1) / kS3Tok, C);
            if (heads == 32) mla_scores_v3_kernel<32><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
            else mla_scores_v3_kernel<64><<<g3, 256, 0, s>>>(slot, tokens, n, qt, e);
        }
    }
    ok(hipGetLastError(), "glm pf mla absorb/scores");
    mla_mark(2, s);
    const unsigned obv = v1 ? 1U : mla_e_lane_major(heads) ? 5U : obar_version() == 5 ? 2U : obar_version();  // v5: heads 32 only
    if (obv == 5) mla_softmax_rows_kernel<true><<<dim3(heads, C), 256, 0, s>>>(n, heads, e);
    else mla_softmax_rows_kernel<false><<<dim3(heads, C), 256, 0, s>>>(n, heads, e);
    ok(hipGetLastError(), "glm pf mla_softmax_rows");
    mla_mark(3, s);
    if (v1) {
        mla_obar_rows_kernel<<<dim3(kKvLora / kObLat, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        mla_mark(4, s);
        mla_out_rows_kernel<<<dim3(heads, C), 256, 0, s>>>(ob, kv_b, heads, o);
    } else {
        if (obv == 5) mla_obar_v5_kernel<<<dim3(kKvLora / kOb5Dims, C), 256, 0, s>>>(slot, tokens, n, e, ob);
        else if (obv == 4) mla_obar_v4_kernel<<<dim3(heads / kOb4Heads, kKvLora / kOb4Dims, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        else mla_obar_v2_kernel<<<dim3(kKvLora / kOb2Lat, C), 256, 0, s>>>(slot, tokens, n, heads, e, ob);
        mla_mark(4, s);
        mla_gemm_rows_kernel<true><<<dim3((C + kGT - 1) / kGT, kMlaDim / kGT, heads), 256, 0, s>>>(ob, kv_b, heads, C, o);
    }
    ok(hipGetLastError(), "glm pf mla obar/out");
    mla_mark(5, s);
}

}  // namespace strix::glm::pf
