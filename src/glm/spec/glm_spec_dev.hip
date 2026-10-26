// GLM speculative decoding device kernels (see glm_spec_dev.hpp). The per-row arithmetic is the Tier A kernels' (glm_tier_a.hip
// kda_prep / kda_update / kda_out, glm_dsa.hip dsa_store), unchanged; only the row loop and what is stored differ.
// Build with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
#include "glm/spec/glm_spec_dev.hpp"

#include <stdexcept>
#include <string>

// Negative controls for the S1 test only (never set in a product build): GLMSPEC_MUTANT = 1 drops the KDA C2 flags,
// 2 drops the C3 cache flags, 3 drops the C4 pooled-key flag, 4 reports new-row C2 in the committed flag.
#ifndef GLMSPEC_MUTANT
#define GLMSPEC_MUTANT 0
#endif

namespace strix::glm::dev {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
__device__ __forceinline__ void flag_if(unsigned* flag, float v) {
    if (GLMSPEC_MUTANT != 1 && flag && !is_finite(v)) atomicOr(flag, 1U);
}
// R2 tree in LDS over L lanes (the glm_tier_a.hip helper, verbatim).
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

constexpr unsigned kMaxRows = 8;

// 4a for the rows of one slot: one block per head, lane i owns channel i of q, k and v of the head. The committed conv
// state advances through the replay rows' raw inputs and is stored; the new rows convolve from a register copy.
__global__ __launch_bounds__(128) void kda_verify_prep_kernel(KdaDev k, const float* qkv_rep, unsigned n_rep, const float* qkv,
                                                              const float* f, const float* b, unsigned n_new,
                                                              float* conv_state, float* scr, float* beta) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, i = threadIdx.x, H = k.heads, C = H * kKdaDim;
    float st[3][3];
    for (unsigned part = 0; part < 3; ++part) {
        const float* cs = conv_state + static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3;
        st[part][0] = cs[0];
        st[part][1] = cs[1];
        st[part][2] = cs[2];
    }
    for (unsigned r = 0; r < n_rep; ++r)
        for (unsigned part = 0; part < 3; ++part) {
            st[part][0] = st[part][1];
            st[part][1] = st[part][2];
            st[part][2] = qkv_rep[static_cast<std::size_t>(r) * 3 * C + part * C + h * kKdaDim + i];
        }
    for (unsigned part = 0; part < 3; ++part) {
        float* cs = conv_state + static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3;
        cs[0] = st[part][0];
        cs[1] = st[part][1];
        cs[2] = st[part][2];
    }
    const float expA = declared_exp(k.A_log[h]);
    for (unsigned r = 0; r < n_new; ++r) {
        const float* in = qkv + static_cast<std::size_t>(r) * 3 * C;
        float out[3];
        for (unsigned part = 0; part < 3; ++part) {
            const unsigned ch = part * C + h * kKdaDim + i;
            const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;
            const float nw = in[ch];
            float acc = __builtin_fmaf(bf16_to_float(cw[0]), st[part][0], 0.0F);
            acc = __builtin_fmaf(bf16_to_float(cw[1]), st[part][1], acc);
            acc = __builtin_fmaf(bf16_to_float(cw[2]), st[part][2], acc);
            acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
            out[part] = silu(acc);
            st[part][0] = st[part][1];
            st[part][1] = st[part][2];
            st[part][2] = nw;
        }
        const float sq = lds_tree<128>(s, __builtin_fmaf(out[0], out[0], 0.0F));
        const float sk = lds_tree<128>(s, __builtin_fmaf(out[1], out[1], 0.0F));
        const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
        float* sc = scr + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim;
        sc[i] = (out[0] / nq) * kInvSqrt128;
        sc[kKdaDim + i] = out[1] / nk;
        sc[2 * kKdaDim + i] = out[2];
        sc[3 * kKdaDim + i] = kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(r) * C + h * kKdaDim + i] + k.dt_bias[h * kKdaDim + i]));
        if (i == 0) beta[static_cast<std::size_t>(r) * H + h] = sigmoid(b[static_cast<std::size_t>(r) * H + h]);
    }
}

// 4b: grid (head, value tile of 32); lane = key index. The tile S[:, tile] lives in registers for the whole call: the
// replay rows update it and it is stored once; the new rows update the register copy and emit o.
__device__ __forceinline__ void kda_row_update(float (&Sr)[32], float* s, const float* sc, float bt, unsigned tile, unsigned kk,
                                               unsigned* flag, float* o_row) {
    const float q = sc[kk], key = sc[kKdaDim + kk], dec = declared_exp(sc[3 * kKdaDim + kk]);
#pragma unroll
    for (unsigned c = 0; c < 32; ++c) {
        const unsigned vv = tile * 32 + c;
        const float sp = Sr[c] * dec;
        const float u = lds_tree<128>(s, 0.0F + sp * key);  // host tree_sum: lane = +0 + term
        const float dlt = (sc[2 * kKdaDim + vv] - u) * bt;
        const float sn = __builtin_fmaf(key, dlt, sp);
        flag_if(flag, sn);
        Sr[c] = sn;
        if (o_row) {
            const float ov = lds_tree<128>(s, 0.0F + sn * q);
            if (kk == 0) o_row[vv] = ov;
        }
    }
}
__global__ __launch_bounds__(128) void kda_verify_update_kernel(unsigned H, float* S, const float* scr_rep, const float* beta_rep,
                                                                unsigned n_rep, const float* scr, const float* beta, unsigned n_new,
                                                                float* o, unsigned* flag, unsigned* row_flag) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, tile = blockIdx.y, kk = threadIdx.x;
    float* Sh = S + static_cast<std::size_t>(h) * kKdaDim * kKdaDim + static_cast<std::size_t>(kk) * kKdaDim + tile * 32;
    float Sr[32];
#pragma unroll
    for (unsigned c = 0; c < 32; ++c) Sr[c] = Sh[c];
    for (unsigned r = 0; r < n_rep; ++r)
        kda_row_update(Sr, s, scr_rep + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim, beta_rep[static_cast<std::size_t>(r) * H + h],
                       tile, kk, flag, nullptr);
    if (n_rep > 0) {
#pragma unroll
        for (unsigned c = 0; c < 32; ++c) Sh[c] = Sr[c];
    }
    for (unsigned r = 0; r < n_new; ++r)
        kda_row_update(Sr, s, scr + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim, beta[static_cast<std::size_t>(r) * H + h], tile,
                       kk, GLMSPEC_MUTANT == 4 ? flag : (row_flag ? row_flag + r : nullptr), o + (static_cast<std::size_t>(r) * H + h) * kKdaDim);
}

// 4c: one block per (head, row): R10 (glm_tier_a.hip kda_out_kernel, verbatim).
__global__ __launch_bounds__(128) void kda_verify_out_kernel(KdaDev k, const float* o, const float* gate, float* y) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads;
    const std::size_t at = (static_cast<std::size_t>(r) * H + h) * kKdaDim + i;
    const float v = o[at];
    const float ms = lds_tree<128>(s, __builtin_fmaf(v, v, 0.0F)) / static_cast<float>(kKdaDim);
    const float rr = inv_sqrt(ms + kEps);
    y[at] = ((v * rr) * bf16_to_float(k.o_norm[i])) * sigmoid(gate[at]);
}

// DSA rows: store (dsa_store_kernel's per-row part, verbatim) ...
__global__ __launch_bounds__(128) void dsa_rows_store_kernel(DsaSlot sl, unsigned t0, const float* latent, IndexerDev w,
                                                             const float* key, const float* gs, unsigned* row_flag) {
    __shared__ float s[128];
    const unsigned r = blockIdx.x, i = threadIdx.x, t = t0 + r;
    const float k = key[static_cast<std::size_t>(r) * kIdxDim + i];
    const float mu = lds_tree<128>(s, 0.0F + k) / static_cast<float>(kIdxDim);
    const float d = k - mu;
    const float var = lds_tree<128>(s, __builtin_fmaf(d, d, 0.0F)) / static_cast<float>(kIdxDim);
    const float rr = inv_sqrt(var + kLnEps);
    const std::uint16_t kb = float_to_bf16((d * rr) * f16_to_float(w.k_norm_w[i]) + f16_to_float(w.k_norm_b[i]));
    const std::uint16_t gb = float_to_bf16(gs[static_cast<std::size_t>(r) * kIdxDim + i]);
    if (GLMSPEC_MUTANT != 2 && (!bf16_is_finite(kb) || !bf16_is_finite(gb))) atomicOr(row_flag + r, 1U);
    sl.ikey[static_cast<std::size_t>(t) * kIdxDim + i] = kb;
    sl.igate[static_cast<std::size_t>(t) * kIdxDim + i] = gb;
    for (unsigned l = i; l < kKvLora; l += 128) {
        const std::uint16_t lb = float_to_bf16(latent[static_cast<std::size_t>(r) * kKvLora + l]);
        if (GLMSPEC_MUTANT != 2 && !bf16_is_finite(lb)) atomicOr(row_flag + r, 1U);
        sl.latent[static_cast<std::size_t>(t) * kKvLora + l] = lb;
    }
}
// ... then the pools closed inside the block (separate launch: a pool may span rows stored by other blocks).
__global__ __launch_bounds__(128) void dsa_rows_pool_kernel(DsaSlot sl, unsigned t0, IndexerDev w, unsigned* row_flag) {
    const unsigned r = blockIdx.x, i = threadIdx.x, t = t0 + r;
    if ((t + 1) % kKpool != 0) return;
    const unsigned tp = t + 1 - kKpool;
    float l4[kKpool], kv[kKpool], e[kKpool], p[kKpool];
    for (unsigned j = 0; j < kKpool; ++j) {
        l4[j] = bf16_to_float(sl.igate[static_cast<std::size_t>(tp + j) * kIdxDim + i]) + w.ape[j * kIdxDim + i];
        kv[j] = bf16_to_float(sl.ikey[static_cast<std::size_t>(tp + j) * kIdxDim + i]);
    }
    float m = l4[0];
    for (unsigned j = 1; j < kKpool; ++j) m = l4[j] > m ? l4[j] : m;
    for (unsigned j = 0; j < kKpool; ++j) e[j] = declared_exp(l4[j] - m);
    const float sum = ((e[0] + e[1]) + e[2]) + e[3];
    for (unsigned j = 0; j < kKpool; ++j) p[j] = e[j] / sum;
    const float K = ((p[0] * kv[0] + p[1] * kv[1]) + p[2] * kv[2]) + p[3] * kv[3];
    if (GLMSPEC_MUTANT != 3 && !is_finite(K)) atomicOr(row_flag + r, 1U);
    sl.pooled[static_cast<std::size_t>(t / kKpool) * kIdxDim + i] = K;
}

}  // namespace

void kda_verify(const KdaDev& k, const float* qkv_rep, const float* scr_rep, const float* beta_rep, unsigned n_rep,
                const float* qkv, const float* f, const float* gate, const float* b, unsigned n_new, float* S,
                float* conv_state, float* scr_new, float* beta_new, float* o, float* y, unsigned* flag, unsigned* row_flag,
                hipStream_t s) {
    if (n_rep > kMaxRows || n_new > kMaxRows) throw std::invalid_argument("kda_verify: at most 8 replay and 8 new rows");
    if (n_rep == 0 && n_new == 0) return;
    kda_verify_prep_kernel<<<k.heads, 128, 0, s>>>(k, qkv_rep, n_rep, qkv, f, b, n_new, conv_state, scr_new, beta_new);
    ok(hipGetLastError(), "glm kda_verify_prep");
    kda_verify_update_kernel<<<dim3(k.heads, kKdaDim / 32), 128, 0, s>>>(k.heads, S, scr_rep, beta_rep, n_rep, scr_new, beta_new,
                                                                          n_new, o, flag, row_flag);
    ok(hipGetLastError(), "glm kda_verify_update");
    if (n_new > 0) {
        kda_verify_out_kernel<<<dim3(k.heads, n_new), 128, 0, s>>>(k, o, gate, y);
        ok(hipGetLastError(), "glm kda_verify_out");
    }
}

void dsa_append_rows(const DsaSlot& slot, unsigned t0, unsigned R, const float* latent, const float* x,
                     const IndexerDev& w, float* key_scratch, float* gate_scratch, unsigned* row_flag, hipStream_t s) {
    if (R == 0 || R > kMaxRows) throw std::invalid_argument("dsa_append_rows: 1..8 rows");
    if (t0 + R > slot.cap) throw std::invalid_argument("dsa_append_rows: slot capacity");
    dense(x, w.wk, WType::F16, kHidden, 0, kIdxDim, key_scratch, R, s);
    dense(x, w.gate, WType::F16, kHidden, 0, kIdxDim, gate_scratch, R, s);
    dsa_rows_store_kernel<<<R, 128, 0, s>>>(slot, t0, latent, w, key_scratch, gate_scratch, row_flag);
    ok(hipGetLastError(), "glm dsa_rows_store");
    dsa_rows_pool_kernel<<<R, 128, 0, s>>>(slot, t0, w, row_flag);
    ok(hipGetLastError(), "glm dsa_rows_pool");
}

}  // namespace strix::glm::dev
