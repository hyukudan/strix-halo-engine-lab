// MiniMax DSpark drafter: gfx1151 kernels of "mm-dspark-v1" (see minimax_dspark.hpp for the declared orders) and the
// DeviceDrafter. Build with -ffp-contract=off: every product and sum below is one FP32 operation, as in the host code.
#include "hip_minimax_dspark.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace strix::minimax::dspark {
namespace {

void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("dspark: ") + what + ": " + hipGetErrorString(e));
}

__device__ inline float wave_sum(float v) {
    for (int d = 16; d != 0; d >>= 1) v = v + __shfl_xor(v, d, 32);
    return v;
}

// out[r][o] = W[o] . x[r], chunk-16 wave order; one wave per output, 8 outputs per block of 256.
template <int R>
__global__ void mxfp8_rows_kernel(const std::uint8_t* codes, const std::uint8_t* scales, const float* lut_g, const float* x,
                                  float* out, std::uint32_t outputs, std::uint32_t width) {
    __shared__ float lut[256];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    __syncthreads();
    const std::uint32_t lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (o >= outputs) return;
    const std::uint8_t* wr = codes + static_cast<std::size_t>(o) * width;
    const std::uint8_t* sr = scales + static_cast<std::size_t>(o) * (width / 32U);
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    for (std::uint32_t c = lane; c < width / 16U; c += 32U) {
        const uint4 q = *reinterpret_cast<const uint4*>(wr + static_cast<std::size_t>(c) * 16U);
        const float s = e8m0_value(sr[c >> 1U]);
        const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
        for (std::uint32_t e = 0; e < 16U; ++e) {
            const float w = lut[(words[e >> 2U] >> (8U * (e & 3U))) & 255U] * s;
            for (int r = 0; r < R; ++r) acc[r] = acc[r] + w * x[static_cast<std::size_t>(r) * width + c * 16U + e];
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
    if (lane == 0)
        for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o] = acc[r];
}

// The same per-lane order with the activations staged in LDS tile by tile (1024 inputs = 64 chunks, two per lane): lane
// l still takes the chunks l, l + 32, ... in ascending order, so the bits equal mxfp8_rows_kernel; the 8 waves of a
// block share each tile instead of every wave reading all R rows from L2. width must be a multiple of 1024. The tile is
// stored transposed (element e of chunk c at e * 64 + c) so the 32 lanes of a read hit 32 consecutive banks.
constexpr std::uint32_t kTile = 1024;
template <int R>
__global__ void mxfp8_rows_tiled_kernel(const std::uint8_t* codes, const std::uint8_t* scales, const float* lut_g, const float* x,
                                        float* out, std::uint32_t outputs, std::uint32_t width) {
    __shared__ float lut[256];
    __shared__ float xs[R * kTile];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    const std::uint32_t lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U);
    const bool valid = o < outputs;
    const std::uint8_t* wr = codes + static_cast<std::size_t>(valid ? o : 0U) * width;
    const std::uint8_t* sr = scales + static_cast<std::size_t>(valid ? o : 0U) * (width / 32U);
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    for (std::uint32_t t0 = 0; t0 < width; t0 += kTile) {
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < static_cast<std::uint32_t>(R) * kTile; i += 256U)
            xs[(i / kTile) * kTile + (i % 16U) * (kTile / 16U) + (i % kTile) / 16U] = x[static_cast<std::size_t>(i / kTile) * width + t0 + i % kTile];
        __syncthreads();
        if (!valid) continue;
        for (std::uint32_t cl = lane; cl < kTile / 16U; cl += 32U) {
            const std::uint32_t c = t0 / 16U + cl;
            const uint4 q = *reinterpret_cast<const uint4*>(wr + static_cast<std::size_t>(c) * 16U);
            const float s = e8m0_value(sr[c >> 1U]);
            const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
            for (std::uint32_t e = 0; e < 16U; ++e) {
                const float w = lut[(words[e >> 2U] >> (8U * (e & 3U))) & 255U] * s;
                for (int r = 0; r < R; ++r) acc[r] = acc[r] + w * xs[static_cast<std::uint32_t>(r) * kTile + e * (kTile / 16U) + cl];
            }
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
    if (valid && lane == 0)
        for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o] = acc[r];
}
// Tiled with the lane's weight slice prefetched into registers before the tile loop (CPL = chunks per lane = width/512):
// the barrier per tile no longer serializes the weight stream. Same per-lane order and bits as mxfp8_rows_kernel.
template <int R, int CPL>
__global__ void mxfp8_rows_tiled_pf_kernel(const std::uint8_t* codes, const std::uint8_t* scales, const float* lut_g, const float* x,
                                           float* out, std::uint32_t outputs, std::uint32_t width) {
    __shared__ float lut[256];
    __shared__ float xs[R * kTile];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    const std::uint32_t lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U);
    const bool valid = o < outputs;
    const std::uint8_t* wr = codes + static_cast<std::size_t>(valid ? o : 0U) * width;
    const std::uint8_t* sr = scales + static_cast<std::size_t>(valid ? o : 0U) * (width / 32U);
    uint4 wq[CPL];
    std::uint8_t sc[CPL];
    for (int j = 0; j < CPL; ++j) {
        const std::uint32_t c = lane + 32U * static_cast<std::uint32_t>(j);
        wq[j] = *reinterpret_cast<const uint4*>(wr + static_cast<std::size_t>(c) * 16U);
        sc[j] = sr[c >> 1U];
    }
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    for (int t = 0; t < CPL / 2; ++t) {
        const std::uint32_t t0 = static_cast<std::uint32_t>(t) * kTile;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < static_cast<std::uint32_t>(R) * kTile; i += 256U)
            xs[(i / kTile) * kTile + (i % 16U) * (kTile / 16U) + (i % kTile) / 16U] = x[static_cast<std::size_t>(i / kTile) * width + t0 + i % kTile];
        __syncthreads();
        for (int h = 0; h < 2; ++h) {
            const int j = 2 * t + h;
            const std::uint32_t cl = lane + 32U * static_cast<std::uint32_t>(h);
            const float s = e8m0_value(sc[j]);
            const std::uint32_t words[4] = {wq[j].x, wq[j].y, wq[j].z, wq[j].w};
            for (std::uint32_t e = 0; e < 16U; ++e) {
                const float w = lut[(words[e >> 2U] >> (8U * (e & 3U))) & 255U] * s;
                for (int r = 0; r < R; ++r) acc[r] = acc[r] + w * xs[static_cast<std::uint32_t>(r) * kTile + e * (kTile / 16U) + cl];
            }
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
    if (valid && lane == 0)
        for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o] = acc[r];
}

template <int R>
__global__ void bf16_rows_tiled_kernel(const std::uint16_t* bits, const float* x, float* out, std::uint32_t outputs, std::uint32_t width) {
    __shared__ float xs[R * kTile];
    const std::uint32_t lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U);
    const bool valid = o < outputs;
    const std::uint16_t* wr = bits + static_cast<std::size_t>(valid ? o : 0U) * width;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    for (std::uint32_t t0 = 0; t0 < width; t0 += kTile) {
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < static_cast<std::uint32_t>(R) * kTile; i += 256U)
            xs[(i / kTile) * kTile + (i % 16U) * (kTile / 16U) + (i % kTile) / 16U] = x[static_cast<std::size_t>(i / kTile) * width + t0 + i % kTile];
        __syncthreads();
        if (!valid) continue;
        for (std::uint32_t cl = lane; cl < kTile / 16U; cl += 32U) {
            const std::uint32_t c = t0 / 16U + cl;
            const uint4* p = reinterpret_cast<const uint4*>(wr + static_cast<std::size_t>(c) * 16U);
            const uint4 a = p[0], b = p[1];
            const std::uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (std::uint32_t e = 0; e < 16U; ++e) {
                const float w = bf16_value(static_cast<std::uint16_t>(words[e >> 1U] >> (16U * (e & 1U))));
                for (int r = 0; r < R; ++r) acc[r] = acc[r] + w * xs[static_cast<std::uint32_t>(r) * kTile + e * (kTile / 16U) + cl];
            }
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
    if (valid && lane == 0)
        for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o] = acc[r];
}

// Four outputs per wave (32 per block of 256): every activation element a lane loads is used for four weight rows, so
// the f32 activation traffic per weight byte drops 4x (d4: the multi-row linears were bound by activation reads). Per
// (row, output) the order is mxfp8_rows_kernel's (lane l: chunks l, l + 32, ... ascending, then the butterfly): same bits.
constexpr std::uint32_t kOpw = 4;
template <int R>
__global__ void mxfp8_rows_multi_kernel(const std::uint8_t* codes, const std::uint8_t* scales, const float* lut_g, const float* x,
                                        float* out, std::uint32_t outputs, std::uint32_t width) {
    __shared__ float lut[256];
    lut[threadIdx.x] = lut_g[threadIdx.x];
    __syncthreads();
    const std::uint32_t lane = threadIdx.x & 31U, o0 = (blockIdx.x * 8U + (threadIdx.x >> 5U)) * kOpw;
    if (o0 >= outputs) return;
    const std::uint32_t n_out = outputs - o0 < kOpw ? outputs - o0 : kOpw;
    float acc[kOpw][R];
    for (std::uint32_t i = 0; i < kOpw; ++i)
        for (int r = 0; r < R; ++r) acc[i][r] = 0.0F;
    for (std::uint32_t c = lane; c < width / 16U; c += 32U) {
        std::uint32_t words[kOpw][4];
        float s[kOpw];
        for (std::uint32_t i = 0; i < kOpw; ++i) {
            const std::uint32_t o = o0 + (i < n_out ? i : 0U);
            const uint4 q = *reinterpret_cast<const uint4*>(codes + static_cast<std::size_t>(o) * width + static_cast<std::size_t>(c) * 16U);
            words[i][0] = q.x;
            words[i][1] = q.y;
            words[i][2] = q.z;
            words[i][3] = q.w;
            s[i] = e8m0_value(scales[static_cast<std::size_t>(o) * (width / 32U) + (c >> 1U)]);
        }
        for (std::uint32_t e = 0; e < 16U; ++e) {
            float xv[R];
            for (int r = 0; r < R; ++r) xv[r] = x[static_cast<std::size_t>(r) * width + c * 16U + e];
            for (std::uint32_t i = 0; i < kOpw; ++i) {
                const float w = lut[(words[i][e >> 2U] >> (8U * (e & 3U))) & 255U] * s[i];
                for (int r = 0; r < R; ++r) acc[i][r] = acc[i][r] + w * xv[r];
            }
        }
    }
    for (std::uint32_t i = 0; i < kOpw; ++i)
        for (int r = 0; r < R; ++r) acc[i][r] = wave_sum(acc[i][r]);
    if (lane == 0)
        for (std::uint32_t i = 0; i < n_out; ++i)
            for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o0 + i] = acc[i][r];
}

template <int R>
__global__ void bf16_rows_kernel(const std::uint16_t* bits, const float* x, float* out, std::uint32_t outputs, std::uint32_t width) {
    const std::uint32_t lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (o >= outputs) return;
    const std::uint16_t* wr = bits + static_cast<std::size_t>(o) * width;
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    for (std::uint32_t c = lane; c < width / 16U; c += 32U) {
        const uint4* p = reinterpret_cast<const uint4*>(wr + static_cast<std::size_t>(c) * 16U);
        const uint4 a = p[0], b = p[1];
        const std::uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        for (std::uint32_t e = 0; e < 16U; ++e) {
            const float w = bf16_value(static_cast<std::uint16_t>(words[e >> 1U] >> (16U * (e & 1U))));
            for (int r = 0; r < R; ++r) acc[r] = acc[r] + w * x[static_cast<std::size_t>(r) * width + c * 16U + e];
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
    if (lane == 0)
        for (int r = 0; r < R; ++r) out[static_cast<std::size_t>(r) * outputs + o] = acc[r];
}

// Gemma RMSNorm over 6144, one row per block of 256: thread t sums t, t + 256, ...; waves; wave sums ascending.
__global__ void rms_norm_kernel(const float* x, const std::uint16_t* w, float* y) {
    __shared__ float part[8];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * hidden;
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * hidden;
    float acc = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += 256U) acc = acc + xr[i] * xr[i];
    acc = wave_sum(acc);
    if ((threadIdx.x & 31U) == 0) part[threadIdx.x >> 5U] = acc;
    __syncthreads();
    float total = 0.0F;
    for (int k = 0; k < 8; ++k) total = total + part[k];
    const float rstd = 1.0F / __builtin_sqrtf(total / static_cast<float>(hidden) + norm_eps);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += 256U) yr[i] = (xr[i] * rstd) * (1.0F + bf16_value(w[i]));
}

// One wave per (row, head): Gemma norm over 128 (lane l holds l, l + 32, l + 64, l + 96) then NeoX RoPE on the pair
// (l, l + 32), in place. Optional BF16 copy of the result.
__global__ void head_norm_rope_kernel(float* x, std::uint32_t rows, std::uint32_t count, const std::uint16_t* w, const float* cos_t,
                                      const float* sin_t, std::uint64_t pos0, std::uint16_t* copy16) {
    const std::uint32_t lane = threadIdx.x & 31U, unit = blockIdx.x * 8U + (threadIdx.x >> 5U);
    const std::uint32_t row = unit / count;
    if (row >= rows) return;
    float* v = x + static_cast<std::size_t>(unit) * head_dim;
    float a[4];
    float acc = 0.0F;
    for (std::uint32_t j = 0; j < 4U; ++j) {
        a[j] = v[lane + 32U * j];
        acc = acc + a[j] * a[j];
    }
    acc = wave_sum(acc);
    const float rstd = 1.0F / __builtin_sqrtf(acc / static_cast<float>(head_dim) + norm_eps);
    for (std::uint32_t j = 0; j < 4U; ++j) a[j] = (a[j] * rstd) * (1.0F + bf16_value(w[lane + 32U * j]));
    const std::size_t pos = pos0 + row;
    const float c = cos_t[pos * 32U + lane], s = sin_t[pos * 32U + lane];
    const float n0 = a[0], n1 = a[1];
    a[0] = n0 * c - n1 * s;
    a[1] = n1 * c + n0 * s;
    for (std::uint32_t j = 0; j < 4U; ++j) {
        v[lane + 32U * j] = a[j];
        if (copy16 != nullptr) copy16[static_cast<std::size_t>(unit) * head_dim + lane + 32U * j] = bf16_bits(a[j]);
    }
}

__global__ void to_bf16_kernel(const float* x, std::uint16_t* y, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) y[i] = bf16_bits(x[i]);
}
// Rows k (BF16, [n][512]) and v (f32 -> BF16) into ring slots (pos0 + r) mod 4096 of one layer.
__global__ void ring_store_kernel(const std::uint16_t* k16, const float* v, std::uint16_t* ring_k, std::uint16_t* ring_v,
                                  std::uint64_t pos0) {
    const std::size_t slot = (pos0 + blockIdx.x) % window;
    for (std::uint32_t i = threadIdx.x; i < kv_width; i += 256U) {
        ring_k[slot * kv_width + i] = k16[static_cast<std::size_t>(blockIdx.x) * kv_width + i];
        ring_v[slot * kv_width + i] = bf16_bits(v[static_cast<std::size_t>(blockIdx.x) * kv_width + i]);
    }
}

__global__ void embed_block_kernel(const std::uint16_t* anchor, const std::uint16_t* noise, float* r) {
    const std::uint16_t* src = blockIdx.x == 0 ? anchor : noise;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += 256U) r[static_cast<std::size_t>(blockIdx.x) * hidden + i] = bf16_value(src[i]);
}
__global__ void add_kernel(float* r, const float* o, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) r[i] = r[i] + o[i];
}
__global__ void swiglu_kernel(float* g, const float* u, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) g[i] = swiglu_oai(g[i], u[i]);
}

// Attention of one (row, head) per block of 128: keys = context positions [first, first + ctx) from the ring, then the
// 7 block rows. Scores by waves (key j on wave j mod 4), max, exp in place, then thread d sums sequentially over keys.
__global__ void attention_kernel(const float* q, const std::uint16_t* ring_k, const std::uint16_t* ring_v,
                                 const std::uint16_t* kb16, const std::uint16_t* vb16, std::uint64_t first, std::uint32_t ctx,
                                 float* out) {
    extern __shared__ float att_scores[];
    float* s = att_scores;
    __shared__ float red[128];
    const std::uint32_t row = blockIdx.x / heads, h = blockIdx.x % heads, g = h / group;
    const std::uint32_t keys = ctx + block, lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const float* qv = q + static_cast<std::size_t>(row) * q_width + static_cast<std::size_t>(h) * head_dim;
    float qa[4];
    for (std::uint32_t j = 0; j < 4U; ++j) qa[j] = qv[lane + 32U * j];
    for (std::uint32_t j = wave; j < keys; j += 4U) {
        const std::uint16_t* kr = j < ctx ? ring_k + ((first + j) % window) * kv_width : kb16 + static_cast<std::size_t>(j - ctx) * kv_width;
        float acc = 0.0F;
        for (std::uint32_t e = 0; e < 4U; ++e) acc = acc + qa[e] * bf16_value(kr[g * head_dim + lane + 32U * e]);
        acc = wave_sum(acc);
        if (lane == 0) s[j] = acc * attention_scale;
    }
    __syncthreads();
    float m = -__builtin_huge_valf();
    for (std::uint32_t j = threadIdx.x; j < keys; j += 128U) m = s[j] > m ? s[j] : m;
    red[threadIdx.x] = m;
    __syncthreads();
    for (std::uint32_t k = 64; k != 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] = red[threadIdx.x + k] > red[threadIdx.x] ? red[threadIdx.x + k] : red[threadIdx.x];
        __syncthreads();
    }
    m = red[0];
    for (std::uint32_t j = threadIdx.x; j < keys; j += 128U) s[j] = declared_exp(s[j] - m);
    __syncthreads();
    const std::uint32_t d = threadIdx.x;
    float sum = 0.0F, acc = 0.0F;
    for (std::uint32_t j = 0; j < keys; ++j) {
        const std::uint16_t* vr = j < ctx ? ring_v + ((first + j) % window) * kv_width : vb16 + static_cast<std::size_t>(j - ctx) * kv_width;
        sum = sum + s[j];
        acc = acc + s[j] * bf16_value(vr[g * head_dim + d]);
    }
    out[static_cast<std::size_t>(row) * q_width + static_cast<std::size_t>(h) * head_dim + d] = acc / sum;
}

// Chunked attention (attn_chunk = 64 keys). Pairs = row * 64 + head (448). Keys: context [first, first + ctx) from the
// ring, then the 7 block rows. A: scores of one kv head's 112 pairs over one key chunk (K staged in LDS), dot128 lane
// order. B: per pair, max then exp in place. C: per (kv head, chunk) partial sum / o[d] over the chunk's keys in order
// (V and e staged in LDS). D: per pair, chunk partials added in ascending chunk order from 0, o / sum.
constexpr std::uint32_t kChunk = attention_chunk, kKeysMax = context_keys + block;
__device__ inline const std::uint16_t* key_row(std::uint32_t j, std::uint32_t ctx, std::uint64_t first, const std::uint16_t* ring,
                                               const std::uint16_t* blk) {
    return j < ctx ? ring + ((first + j) % window) * kv_width : blk + static_cast<std::size_t>(j - ctx) * kv_width;
}
__global__ void attn_scores_kernel(const float* q, const std::uint16_t* ring_k, const std::uint16_t* kb16, std::uint64_t first,
                                   std::uint32_t ctx, float* scores) {
    __shared__ std::uint16_t ks[kChunk * head_dim];
    const std::uint32_t g = blockIdx.x, c0 = blockIdx.y * kChunk, row = blockIdx.z, keys = ctx + block;
    const std::uint32_t nk = keys - c0 < kChunk ? keys - c0 : kChunk;
    for (std::uint32_t i = threadIdx.x; i < nk * head_dim; i += 256U)
        ks[i] = key_row(c0 + i / head_dim, ctx, first, ring_k, kb16)[g * head_dim + i % head_dim];
    __syncthreads();
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    for (std::uint32_t item = wave; item < group * nk; item += 8U) {
        const std::uint32_t jj = item % nk, h = g * group + item / nk;
        const float* qv = q + static_cast<std::size_t>(row) * q_width + static_cast<std::size_t>(h) * head_dim;
        float acc = 0.0F;
        for (std::uint32_t e = 0; e < 4U; ++e) acc = acc + qv[lane + 32U * e] * bf16_value(ks[jj * head_dim + lane + 32U * e]);
        acc = wave_sum(acc);
        if (lane == 0) scores[static_cast<std::size_t>(row * heads + h) * kKeysMax + c0 + jj] = acc * attention_scale;
    }
}
__global__ void attn_exp_kernel(float* scores, std::uint32_t keys) {
    __shared__ float red[256];
    float* s = scores + static_cast<std::size_t>(blockIdx.x) * kKeysMax;
    float m = -__builtin_huge_valf();
    for (std::uint32_t j = threadIdx.x; j < keys; j += 256U) m = s[j] > m ? s[j] : m;
    red[threadIdx.x] = m;
    __syncthreads();
    for (std::uint32_t k = 128; k != 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] = red[threadIdx.x + k] > red[threadIdx.x] ? red[threadIdx.x + k] : red[threadIdx.x];
        __syncthreads();
    }
    m = red[0];
    for (std::uint32_t j = threadIdx.x; j < keys; j += 256U) s[j] = declared_exp(s[j] - m);
}
__global__ void attn_pv_kernel(const float* scores, const std::uint16_t* ring_v, const std::uint16_t* vb16, std::uint64_t first,
                               std::uint32_t ctx, float* po, float* ps) {
    __shared__ std::uint16_t vs[kChunk * head_dim];
    __shared__ float es[group * kChunk];
    const std::uint32_t g = blockIdx.x, ci = blockIdx.y, row = blockIdx.z, c0 = ci * kChunk, keys = ctx + block;
    const std::uint32_t nk = keys - c0 < kChunk ? keys - c0 : kChunk;
    for (std::uint32_t i = threadIdx.x; i < nk * head_dim; i += 256U)
        vs[i] = key_row(c0 + i / head_dim, ctx, first, ring_v, vb16)[g * head_dim + i % head_dim];
    for (std::uint32_t i = threadIdx.x; i < group * nk; i += 256U) {
        const std::uint32_t pl = i / nk, h = g * group + pl;
        es[pl * kChunk + i % nk] = scores[static_cast<std::size_t>(row * heads + h) * kKeysMax + c0 + i % nk];
    }
    __syncthreads();
    const std::uint32_t d = threadIdx.x & 127U, half = threadIdx.x >> 7U;
    for (std::uint32_t pl = half; pl < group; pl += 2U) {
        const std::uint32_t h = g * group + pl, pair = row * heads + h;
        float acc = 0.0F, sum = 0.0F;
        for (std::uint32_t j = 0; j < nk; ++j) {
            const float e = es[pl * kChunk + j];
            sum = sum + e;
            acc = acc + e * bf16_value(vs[j * head_dim + d]);
        }
        po[(static_cast<std::size_t>(ci) * block * heads + pair) * head_dim + d] = acc;
        if (d == 0) ps[static_cast<std::size_t>(ci) * block * heads + pair] = sum;
    }
}
__global__ void attn_combine_kernel(const float* po, const float* ps, std::uint32_t chunks, float* out) {
    const std::uint32_t pair = blockIdx.x, d = threadIdx.x;
    float acc = 0.0F, sum = 0.0F;
    for (std::uint32_t c = 0; c < chunks; ++c) {
        sum = sum + ps[static_cast<std::size_t>(c) * block * heads + pair];
        acc = acc + po[(static_cast<std::size_t>(c) * block * heads + pair) * head_dim + d];
    }
    out[static_cast<std::size_t>(pair) * head_dim + d] = acc / sum;
}

// wave_sum's xor butterfly as a reduce-scatter over 32 keys (one per lane): at stride D lane l keeps the half of its keys
// selected by bit D of l and adds the partner's value of the same key, own + partner; after D = 1 lane l holds key l.
// Strides are template constants so v[] stays in registers (measurement: a runtime stride turned every access into a select chain).
template <std::uint32_t D> __device__ __forceinline__ void reduce_scatter_step(float (&v)[32], std::uint32_t lane) {
    const bool hi = (lane & D) != 0;
#pragma unroll
    for (std::uint32_t i = 0; i < D; ++i) {
        const float keep = hi ? v[i + D] : v[i], send = hi ? v[i] : v[i + D];
        v[i] = keep + __shfl_xor(send, static_cast<int>(D), 32);
    }
}
__device__ __forceinline__ void reduce_scatter32(float (&v)[32], std::uint32_t lane) {
    reduce_scatter_step<16>(v, lane);
    reduce_scatter_step<8>(v, lane);
    reduce_scatter_step<4>(v, lane);
    reduce_scatter_step<2>(v, lane);
    reduce_scatter_step<1>(v, lane);
}

// Shared-K/V chunked attention (Options::attn_variant 1): the same declared order as attn_scores_kernel /
// attn_pv_kernel, one block per (kv head, chunk) for all 7 rows x 16 heads, so each K / V chunk is staged once instead of
// once per row.
// scores2: lane l's chain over e = 0..3 as attn_scores_kernel; the xor butterfly of wave_sum is evaluated as a
// reduce-scatter over 32 keys: at stride d lane l keeps the half of its keys selected by bit d of l and adds the partner's
// value of the same key (own + partner, as v + shfl_xor(v, d)), so every key sees the same pairs in the same operand
// order; after d = 1 lane l holds key l. 31 shuffle-adds per 32 keys instead of 5 per key.
__global__ __launch_bounds__(256) void attn_scores2_kernel(const float* q, const std::uint16_t* ring_k, const std::uint16_t* kb16,
                                                           std::uint64_t first, std::uint32_t ctx, float* scores) {
    __shared__ float ks[kChunk * head_dim];
    const std::uint32_t g = blockIdx.x, c0 = blockIdx.y * kChunk, keys = ctx + block;
    const std::uint32_t nk = keys - c0 < kChunk ? keys - c0 : kChunk;
    for (std::uint32_t i = threadIdx.x; i < nk * head_dim; i += 256U)
        ks[i] = bf16_value(key_row(c0 + i / head_dim, ctx, first, ring_k, kb16)[g * head_dim + i % head_dim]);
    __syncthreads();
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    for (std::uint32_t p = wave; p < block * group; p += 8U) {
        const std::uint32_t row = p / group, h = g * group + p % group;
        const float* qv = q + static_cast<std::size_t>(row) * q_width + static_cast<std::size_t>(h) * head_dim;
        float qa[4];
        for (std::uint32_t e = 0; e < 4U; ++e) qa[e] = qv[lane + 32U * e];
        float* srow = scores + static_cast<std::size_t>(row * heads + h) * kKeysMax + c0;
        for (std::uint32_t b0 = 0; b0 < nk; b0 += 32U) {
            float v[32];
#pragma unroll
            for (std::uint32_t jj = 0; jj < 32U; ++jj) {
                const std::uint32_t j = b0 + jj < nk ? b0 + jj : nk - 1;  // keys past nk: finite, never stored
                float acc = 0.0F;
#pragma unroll
                for (std::uint32_t e = 0; e < 4U; ++e) acc = acc + qa[e] * ks[j * head_dim + lane + 32U * e];
                v[jj] = acc;
            }
            reduce_scatter32(v, lane);
            if (b0 + lane < nk) srow[b0 + lane] = v[0] * attention_scale;
        }
    }
}
// scores3 (attn_variant 2, 3): attn_scores2_kernel's arithmetic on a (kv head, 32-key batch, row) grid of 128-thread
// blocks (wave w: heads 4w..4w + 3 of the kv group), K of the batch staged per block: score values do not depend on the
// key partition, only on each key's chain and butterfly (unchanged).
// Waves = minimum waves per EU asked of the compiler (0 = none): measurements showed scores3 at 224 VGPRs, 2.2x the d4 kernel.
template <int Waves>
__global__ __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(Waves > 0 ? Waves : 1))) void attn_scores3_kernel(const float* q, const std::uint16_t* ring_k, const std::uint16_t* kb16,
                                                           std::uint64_t first, std::uint32_t ctx, float* scores) {
    __shared__ float ks[32 * head_dim];
    const std::uint32_t g = blockIdx.x, b0 = blockIdx.y * 32U, row = blockIdx.z, keys = ctx + block;
    const std::uint32_t nk = keys - b0 < 32U ? keys - b0 : 32U;
    for (std::uint32_t i = threadIdx.x; i < nk * head_dim; i += 128U)
        ks[i] = bf16_value(key_row(b0 + i / head_dim, ctx, first, ring_k, kb16)[g * head_dim + i % head_dim]);
    __syncthreads();
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    for (std::uint32_t hh = 0; hh < group / 4U; ++hh) {
        const std::uint32_t h = g * group + wave * (group / 4U) + hh;
        const float* qv = q + static_cast<std::size_t>(row) * q_width + static_cast<std::size_t>(h) * head_dim;
        float qa[4];
        for (std::uint32_t e = 0; e < 4U; ++e) qa[e] = qv[lane + 32U * e];
        float v[32];
#pragma unroll
        for (std::uint32_t jj = 0; jj < 32U; ++jj) {
            const std::uint32_t j = jj < nk ? jj : nk - 1;  // keys past nk: finite, never stored
            float acc = 0.0F;
#pragma unroll
            for (std::uint32_t e = 0; e < 4U; ++e) acc = acc + qa[e] * ks[j * head_dim + lane + 32U * e];
            v[jj] = acc;
        }
        reduce_scatter32(v, lane);
        if (lane < nk) scores[static_cast<std::size_t>(row * heads + h) * kKeysMax + b0 + lane] = v[0] * attention_scale;
    }
}
// pv3 (attn_variant 3): attn_pv_kernel's grid (kv head, chunk, row), thread d holds V[j][d] of the chunk in registers,
// the 16 pairs' e in LDS (broadcast reads); the same acc chain per pair, the e-sum chain by one thread per pair.
__global__ __launch_bounds__(128) void attn_pv3_kernel(const float* scores, const std::uint16_t* ring_v, const std::uint16_t* vb16,
                                                       std::uint64_t first, std::uint32_t ctx, float* po, float* ps) {
    __shared__ float es[group * kChunk];
    const std::uint32_t g = blockIdx.x, ci = blockIdx.y, row = blockIdx.z, c0 = ci * kChunk, keys = ctx + block;
    const std::uint32_t nk = keys - c0 < kChunk ? keys - c0 : kChunk;
    for (std::uint32_t i = threadIdx.x; i < group * kChunk; i += 128U) {
        const std::uint32_t pl = i / kChunk, j = i % kChunk, h = g * group + pl;
        es[i] = j < nk ? scores[static_cast<std::size_t>(row * heads + h) * kKeysMax + c0 + j] : 0.0F;
    }
    const std::uint32_t d = threadIdx.x;
    float vv[kChunk];
#pragma unroll
    for (std::uint32_t j = 0; j < kChunk; ++j) vv[j] = j < nk ? bf16_value(key_row(c0 + j, ctx, first, ring_v, vb16)[g * head_dim + d]) : 0.0F;
    __syncthreads();
    for (std::uint32_t pl = 0; pl < group; ++pl) {
        const std::uint32_t pair = row * heads + g * group + pl;
        const float* ep = es + pl * kChunk;
        float acc = 0.0F;
#pragma unroll
        for (std::uint32_t j = 0; j < kChunk; ++j)
            if (j < nk) acc = acc + ep[j] * vv[j];
        po[(static_cast<std::size_t>(ci) * block * heads + pair) * head_dim + d] = acc;
    }
    if (threadIdx.x < group) {
        const std::uint32_t pl = threadIdx.x;
        float sum = 0.0F;
        for (std::uint32_t j = 0; j < nk; ++j) sum = sum + es[pl * kChunk + j];
        ps[static_cast<std::size_t>(ci) * block * heads + row * heads + g * group + pl] = sum;
    }
}
// pv2: thread (d, half) holds V[j][d] of the chunk in registers; per (row, head) pair the chain acc = acc + e_j v_j over
// j ascending as attn_pv_kernel; the pair's e-sum (same chain) by one thread per pair.
__global__ __launch_bounds__(256) void attn_pv2_kernel(const float* scores, const std::uint16_t* ring_v, const std::uint16_t* vb16,
                                                       std::uint64_t first, std::uint32_t ctx, float* po, float* ps) {
    __shared__ float es[block * group * kChunk];
    const std::uint32_t g = blockIdx.x, ci = blockIdx.y, c0 = ci * kChunk, keys = ctx + block;
    const std::uint32_t nk = keys - c0 < kChunk ? keys - c0 : kChunk;
    for (std::uint32_t i = threadIdx.x; i < block * group * kChunk; i += 256U) {
        const std::uint32_t p = i / kChunk, j = i % kChunk, row = p / group, h = g * group + p % group;
        es[i] = j < nk ? scores[static_cast<std::size_t>(row * heads + h) * kKeysMax + c0 + j] : 0.0F;
    }
    const std::uint32_t d = threadIdx.x & 127U, half = threadIdx.x >> 7U;
    float vv[kChunk];
#pragma unroll
    for (std::uint32_t j = 0; j < kChunk; ++j) vv[j] = j < nk ? bf16_value(key_row(c0 + j, ctx, first, ring_v, vb16)[g * head_dim + d]) : 0.0F;
    __syncthreads();
    for (std::uint32_t p = half; p < block * group; p += 2U) {
        const std::uint32_t row = p / group, h = g * group + p % group, pair = row * heads + h;
        const float* ep = es + p * kChunk;
        float acc = 0.0F;
#pragma unroll
        for (std::uint32_t j = 0; j < kChunk; ++j)
            if (j < nk) acc = acc + ep[j] * vv[j];
        po[(static_cast<std::size_t>(ci) * block * heads + pair) * head_dim + d] = acc;
    }
    if (threadIdx.x < block * group) {
        const std::uint32_t p = threadIdx.x, row = p / group, h = g * group + p % group;
        float sum = 0.0F;
        for (std::uint32_t j = 0; j < nk; ++j) sum = sum + es[p * kChunk + j];
        ps[static_cast<std::size_t>(ci) * block * heads + row * heads + h] = sum;
    }
}

// One wave per vocabulary row v in [v0, v1): bias = W2[v] . W1[prev] (lane l: 8l..8l + 7), value = base + bias; the block's
// best (value desc, id asc) into cand[blockIdx.x].
__global__ void markov_kernel(const std::uint16_t* w1, const std::uint16_t* w2, const std::uint32_t* prev, const float* base,
                              std::uint32_t v0, std::uint32_t v1, Candidate* cand) {
    __shared__ float best_v[8];
    __shared__ std::uint32_t best_i[8];
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, v = v0 + blockIdx.x * 8U + wave;
    Candidate c{-__builtin_huge_valf(), 0xffffffffU};
    if (v < v1) {
        const uint4 mw = *reinterpret_cast<const uint4*>(w1 + static_cast<std::size_t>(*prev) * markov_rank + 8U * lane);
        const uint4 ww = *reinterpret_cast<const uint4*>(w2 + static_cast<std::size_t>(v) * markov_rank + 8U * lane);
        const std::uint32_t ma[4] = {mw.x, mw.y, mw.z, mw.w}, wa[4] = {ww.x, ww.y, ww.z, ww.w};
        float acc = 0.0F;
        for (std::uint32_t e = 0; e < 8U; ++e) {
            const float m = bf16_value(static_cast<std::uint16_t>(ma[e >> 1U] >> (16U * (e & 1U))));
            const float w = bf16_value(static_cast<std::uint16_t>(wa[e >> 1U] >> (16U * (e & 1U))));
            acc = acc + m * w;
        }
        acc = wave_sum(acc);
        c = {base[v - v0] + acc, v};
    }
    if (lane == 0) {
        best_v[wave] = c.value;
        best_i[wave] = c.id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        Candidate b{best_v[0], best_i[0]};
        for (int k = 1; k < 8; ++k)
            if (better(best_v[k], best_i[k], b.value, b.id)) b = {best_v[k], best_i[k]};
        cand[blockIdx.x] = b;
    }
}
__global__ void pick_kernel(const Candidate* cand, std::uint32_t n, Candidate* out, std::uint32_t* prev_out) {
    __shared__ float rv[256];
    __shared__ std::uint32_t ri[256];
    Candidate b{-__builtin_huge_valf(), 0xffffffffU};
    for (std::uint32_t i = threadIdx.x; i < n; i += 256U)
        if (better(cand[i].value, cand[i].id, b.value, b.id)) b = cand[i];
    rv[threadIdx.x] = b.value;
    ri[threadIdx.x] = b.id;
    __syncthreads();
    for (std::uint32_t k = 128; k != 0; k >>= 1U) {
        const std::uint32_t t = threadIdx.x;
        if (t < k && better(rv[t + k], ri[t + k], rv[t], ri[t])) {
            rv[t] = rv[t + k];
            ri[t] = ri[t + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = Candidate{rv[0], ri[0]};
        if (prev_out != nullptr) *prev_out = ri[0];
    }
}

// One wave per block row k: chunk-16 order over [y_k ; W1[t_k]] (6400) with the confidence weight, + b, sigmoid.
__global__ void confidence_kernel(const float* y, const std::uint16_t* w1, const std::uint32_t* prev, const std::uint16_t* cw,
                                  float cb, float* out) {
    const std::uint32_t lane = threadIdx.x & 31U, k = threadIdx.x >> 5U;
    const float* yk = y + static_cast<std::size_t>(k) * hidden;
    const std::uint16_t* mk = w1 + static_cast<std::size_t>(prev[k]) * markov_rank;
    float acc = 0.0F;
    for (std::uint32_t c = lane; c < confidence_width / 16U; c += 32U)
        for (std::uint32_t e = 0; e < 16; ++e) {
            const std::uint32_t i = c * 16U + e;
            const float x = i < hidden ? yk[i] : bf16_value(mk[i - hidden]);
            acc = acc + bf16_value(cw[i]) * x;
        }
    acc = wave_sum(acc);
    if (lane == 0) out[k] = sigmoid(acc + cb);
}

unsigned blocks_of(std::size_t n, std::size_t per) { return static_cast<unsigned>((n + per - 1) / per); }

} // namespace

template <class T>
T* DeviceDrafter::alloc(std::size_t count) {
    void* p = nullptr;
    hip_ok(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
    allocations_.push_back(p);
    return static_cast<T*>(p);
}

DeviceDrafter::DeviceDrafter(const Weights& host, Options options) : opt_(options) {
    int dev = 0, warp = 0;
    hip_ok(hipGetDevice(&dev), "device");
    hip_ok(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev), "warp size");
    if (warp != 32) throw std::runtime_error("dspark: kernels assume wave32");
    const auto up8 = [&](const std::uint8_t* src, std::size_t n) {
        auto* d = alloc<std::uint8_t>(n);
        hip_ok(hipMemcpy(d, src, n, hipMemcpyHostToDevice), "upload");
        weight_bytes_ += n;
        return d;
    };
    const auto up16 = [&](const std::uint16_t* src, std::size_t n) {
        auto* d = alloc<std::uint16_t>(n);
        hip_ok(hipMemcpy(d, src, n * 2, hipMemcpyHostToDevice), "upload");
        weight_bytes_ += n * 2;
        return d;
    };
    const auto upmx = [&](const Mxfp8Matrix& m) {
        if (m.codes == nullptr) return m;  // DraftFile::open(dir, false): the linear hook supplies it
        const std::size_t n = static_cast<std::size_t>(m.outputs) * m.width;
        return Mxfp8Matrix{up8(m.codes, n), up8(m.scales, n / 32U), m.outputs, m.width};
    };
    dev_.fc = {up16(host.fc.bits, static_cast<std::size_t>(host.fc.outputs) * host.fc.width), host.fc.outputs, host.fc.width};
    dev_.hidden_norm = up16(host.hidden_norm, hidden);
    dev_.final_norm = up16(host.final_norm, hidden);
    for (std::uint32_t L = 0; L < layers; ++L) {
        const LayerWeights& s = host.layer[L];
        LayerWeights& d = dev_.layer[L];
        d.input_norm = up16(s.input_norm, hidden);
        d.post_norm = up16(s.post_norm, hidden);
        d.q_norm = up16(s.q_norm, head_dim);
        d.k_norm = up16(s.k_norm, head_dim);
        d.q = upmx(s.q);
        d.k = upmx(s.k);
        d.v = upmx(s.v);
        d.o = upmx(s.o);
        d.gate = upmx(s.gate);
        d.up = upmx(s.up);
        d.down = upmx(s.down);
    }
    dev_.markov_w1 = up16(host.markov_w1, static_cast<std::size_t>(vocab) * markov_rank);
    dev_.markov_w2 = up16(host.markov_w2, static_cast<std::size_t>(vocab) * markov_rank);
    dev_.confidence_w = up16(host.confidence_w, confidence_width);
    dev_.confidence_b = host.confidence_b;

    std::vector<float> lut(256), cs(opt_.capacity * 32U), sn(opt_.capacity * 32U);
    for (std::uint32_t c = 0; c < 256U; ++c) lut[c] = e4m3_value(static_cast<std::uint8_t>(c));
    for (std::uint64_t p = 0; p < opt_.capacity; ++p) rope_table(p, cs.data() + p * 32U, sn.data() + p * 32U);
    e4m3_lut_ = alloc<float>(256);
    rope_cos_ = alloc<float>(cs.size());
    rope_sin_ = alloc<float>(sn.size());
    hip_ok(hipMemcpy(e4m3_lut_, lut.data(), 256 * 4, hipMemcpyHostToDevice), "lut");
    hip_ok(hipMemcpy(rope_cos_, cs.data(), cs.size() * 4, hipMemcpyHostToDevice), "rope");
    hip_ok(hipMemcpy(rope_sin_, sn.data(), sn.size() * 4, hipMemcpyHostToDevice), "rope");

    const std::size_t ring = static_cast<std::size_t>(layers) * window * kv_width;
    for (std::uint32_t s = 0; s < opt_.slots; ++s) {
        ring_k_.push_back(alloc<std::uint16_t>(ring));
        ring_v_.push_back(alloc<std::uint16_t>(ring));
        ring_pos_.emplace_back(window, -1);
        ring_bytes_ += 4 * ring;
    }
    constexpr std::size_t chunk = 32;  // context rows per launch
    r_ = alloc<float>(block * hidden);
    x_ = alloc<float>(block * hidden);
    q_ = alloc<float>(block * q_width);
    kb_ = alloc<float>(block * kv_width);
    vb_ = alloc<float>(block * kv_width);
    att_ = alloc<float>(block * q_width);
    o_ = alloc<float>(block * hidden);
    gate_ = alloc<float>(block * inter);
    up_ = alloc<float>(block * inter);
    y_ = alloc<float>(block * hidden);
    ctx_fc_ = alloc<float>(chunk * hidden);
    ctx_c_ = alloc<float>(chunk * hidden);
    ctx_k_ = alloc<float>(chunk * kv_width);
    ctx_v_ = alloc<float>(chunk * kv_width);
    kb16_ = alloc<std::uint16_t>(std::max<std::size_t>(block, chunk) * kv_width);
    vb16_ = alloc<std::uint16_t>(block * kv_width);
    if (opt_.attn_chunk != 0 && opt_.attn_chunk != kChunk) throw std::invalid_argument("dspark: attn_chunk must be 0 or 64");
    const std::size_t max_chunks = (kKeysMax + kChunk - 1) / kChunk;
    att_s_ = alloc<float>(static_cast<std::size_t>(block) * heads * kKeysMax);
    att_po_ = alloc<float>(max_chunks * block * heads * head_dim);
    att_ps_ = alloc<float>(max_chunks * block * heads);
    cand_ = alloc<Candidate>(blocks_of(vocab, 8));
    pick_ = alloc<Candidate>(1);
    conf_ = alloc<float>(block);
    prev_dev_ = alloc<std::uint32_t>(block + 1);
}

DeviceDrafter::~DeviceDrafter() {
    for (void* p : allocations_) (void)hipFree(p);
}

void DeviceDrafter::device_synchronize() { hip_ok(hipDeviceSynchronize(), "device synchronize"); }

void DeviceDrafter::mxfp8_linear(const Mxfp8Matrix& w, const float* x, std::size_t rows, float* out, hipStream_t st) const {
    const dim3 grid(blocks_of(w.outputs, 8)), threads(256);
    if (opt_.multi_linears && rows <= 8) {
        const dim3 mgrid(blocks_of(w.outputs, 8U * kOpw));
#define STRIX_MMDS_MU(R) hipLaunchKernelGGL(mxfp8_rows_multi_kernel<R>, mgrid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width)
        switch (rows) {
        case 1: STRIX_MMDS_MU(1); break;
        case 2: STRIX_MMDS_MU(2); break;
        case 3: STRIX_MMDS_MU(3); break;
        case 4: STRIX_MMDS_MU(4); break;
        case 5: STRIX_MMDS_MU(5); break;
        case 6: STRIX_MMDS_MU(6); break;
        case 7: STRIX_MMDS_MU(7); break;
        default: STRIX_MMDS_MU(8); break;
        }
#undef STRIX_MMDS_MU
        hip_ok(hipGetLastError(), "mxfp8_rows_multi_kernel");
        return;
    }
    const bool tiled = opt_.tiled_linears && rows <= 8 && w.width % kTile == 0;
    const std::uint32_t cpl = w.width / 512U;
    const bool prefetch = tiled && opt_.prefetch_linears && (cpl == 12U || cpl == 16U || cpl == 24U);
    if (prefetch) {
#define STRIX_MMDS_PF(R)                                                                                                         \
    if (cpl == 12U)                                                                                                              \
        hipLaunchKernelGGL((mxfp8_rows_tiled_pf_kernel<R, 12>), grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width); \
    else if (cpl == 16U)                                                                                                         \
        hipLaunchKernelGGL((mxfp8_rows_tiled_pf_kernel<R, 16>), grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width); \
    else                                                                                                                         \
        hipLaunchKernelGGL((mxfp8_rows_tiled_pf_kernel<R, 24>), grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width)
        switch (rows) {
        case 1: STRIX_MMDS_PF(1); break;
        case 2: STRIX_MMDS_PF(2); break;
        case 3: STRIX_MMDS_PF(3); break;
        case 4: STRIX_MMDS_PF(4); break;
        case 5: STRIX_MMDS_PF(5); break;
        case 6: STRIX_MMDS_PF(6); break;
        case 7: STRIX_MMDS_PF(7); break;
        default: STRIX_MMDS_PF(8); break;
        }
#undef STRIX_MMDS_PF
        hip_ok(hipGetLastError(), "mxfp8_rows_tiled_pf_kernel");
        return;
    }
#define STRIX_MMDS_MX(R)                                                                                                         \
    if (tiled)                                                                                                                   \
        hipLaunchKernelGGL(mxfp8_rows_tiled_kernel<R>, grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width); \
    else                                                                                                                         \
        hipLaunchKernelGGL(mxfp8_rows_kernel<R>, grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width)
    switch (rows) {
    case 1: STRIX_MMDS_MX(1); break;
    case 2: STRIX_MMDS_MX(2); break;
    case 3: STRIX_MMDS_MX(3); break;
    case 4: STRIX_MMDS_MX(4); break;
    case 5: STRIX_MMDS_MX(5); break;
    case 6: STRIX_MMDS_MX(6); break;
    case 7: STRIX_MMDS_MX(7); break;
    case 8: STRIX_MMDS_MX(8); break;
    case 32: hipLaunchKernelGGL(mxfp8_rows_kernel<32>, grid, threads, 0, st, w.codes, w.scales, e4m3_lut_, x, out, w.outputs, w.width); break;
    default: throw std::invalid_argument("dspark: mxfp8_linear rows must be 1..8 or 32");
    }
#undef STRIX_MMDS_MX
    hip_ok(hipGetLastError(), "mxfp8_rows_kernel");
}

void DeviceDrafter::linear(std::uint32_t layer, std::uint32_t kind, const float* x, std::size_t rows, float* out,
                           hipStream_t st) const {
    if (!profile_) {
        linear_impl(layer, kind, x, rows, out, st);
        return;
    }
    hip_ok(hipDeviceSynchronize(), "profile");
    const auto t0 = std::chrono::steady_clock::now();
    linear_impl(layer, kind, x, rows, out, st);
    hip_ok(hipDeviceSynchronize(), "profile");
    lin_s_[kind] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++lin_n_[kind];
}

void DeviceDrafter::linear_group(std::uint32_t layer, std::initializer_list<std::uint32_t> kinds, const float* x, std::size_t rows,
                                 std::initializer_list<float*> outs, hipStream_t st) const {
    if (!group_hook_) {
        for (std::size_t i = 0; i < kinds.size(); ++i) linear(layer, kinds.begin()[i], x, rows, outs.begin()[i], st);
        return;
    }
    const auto n = static_cast<std::uint32_t>(kinds.size());
    if (profile_) {
        hip_ok(hipDeviceSynchronize(), "profile");
        const auto t0 = std::chrono::steady_clock::now();
        const bool done = group_hook_(layer, kinds.begin(), n, x, rows, outs.begin(), st);
        hip_ok(hipDeviceSynchronize(), "profile");
        if (done) {
            lin_s_[kinds.begin()[0]] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (const std::uint32_t k : kinds) ++lin_n_[k];
            return;
        }
    } else if (group_hook_(layer, kinds.begin(), n, x, rows, outs.begin(), st)) {
        return;
    }
    for (std::size_t i = 0; i < kinds.size(); ++i) linear(layer, kinds.begin()[i], x, rows, outs.begin()[i], st);
}

void DeviceDrafter::linear_impl(std::uint32_t layer, std::uint32_t kind, const float* x, std::size_t rows, float* out,
                                hipStream_t st) const {
    if (hook_ && hook_(layer, kind, x, rows, out, st)) return;
    if (kind == lin_fc) {
        bf16_linear(dev_.fc, x, rows, out, st);
        return;
    }
    const LayerWeights& w = dev_.layer[layer];
    const Mxfp8Matrix* m[7] = {&w.q, &w.k, &w.v, &w.o, &w.gate, &w.up, &w.down};
    if (m[kind]->codes == nullptr) throw std::runtime_error("dspark: block linear not loaded and not supplied by the hook");
    mxfp8_linear(*m[kind], x, rows, out, st);
}

void DeviceDrafter::bf16_linear(const Bf16Matrix& w, const float* x, std::size_t rows, float* out, hipStream_t st) const {
    const dim3 grid(blocks_of(w.outputs, 8)), threads(256);
    const bool tiled = opt_.tiled_linears && rows <= 8 && w.width % kTile == 0;
#define STRIX_MMDS_BF(R)                                                                                                         \
    if (tiled)                                                                                                                   \
        hipLaunchKernelGGL(bf16_rows_tiled_kernel<R>, grid, threads, 0, st, w.bits, x, out, w.outputs, w.width);                \
    else                                                                                                                         \
        hipLaunchKernelGGL(bf16_rows_kernel<R>, grid, threads, 0, st, w.bits, x, out, w.outputs, w.width)
    switch (rows) {
    case 1: STRIX_MMDS_BF(1); break;
    case 2: STRIX_MMDS_BF(2); break;
    case 3: STRIX_MMDS_BF(3); break;
    case 4: STRIX_MMDS_BF(4); break;
    case 5: STRIX_MMDS_BF(5); break;
    case 6: STRIX_MMDS_BF(6); break;
    case 7: STRIX_MMDS_BF(7); break;
    case 8: STRIX_MMDS_BF(8); break;
    case 32: hipLaunchKernelGGL(bf16_rows_kernel<32>, grid, threads, 0, st, w.bits, x, out, w.outputs, w.width); break;
    default: throw std::invalid_argument("dspark: bf16_linear rows must be 1..8 or 32");
    }
#undef STRIX_MMDS_BF
    hip_ok(hipGetLastError(), "bf16_rows_kernel");
}

void DeviceDrafter::rms_norm(const float* x, const std::uint16_t* weight, std::size_t rows, float* out, hipStream_t st) const {
    hipLaunchKernelGGL(rms_norm_kernel, dim3(static_cast<unsigned>(rows)), dim3(256), 0, st, x, weight, out);
    hip_ok(hipGetLastError(), "rms_norm_kernel");
}

void DeviceDrafter::head_norm_rope(float* x, std::size_t rows, std::uint32_t count, const std::uint16_t* weight, std::uint64_t pos0,
                                   hipStream_t st) const {
    if (pos0 + rows > opt_.capacity) throw std::out_of_range("dspark: position beyond the RoPE table");
    hipLaunchKernelGGL(head_norm_rope_kernel, dim3(blocks_of(rows * count, 8)), dim3(256), 0, st, x, static_cast<std::uint32_t>(rows),
                       count, weight, rope_cos_, rope_sin_, pos0, nullptr);
    hip_ok(hipGetLastError(), "head_norm_rope_kernel");
}

void DeviceDrafter::append(std::uint32_t slot, const float* taps_rows, std::size_t n, std::uint64_t pos0, hipStream_t st) {
    if (slot >= opt_.slots) throw std::invalid_argument("dspark: append slot");
    if (pos0 + n > opt_.capacity) throw std::out_of_range("dspark: append beyond capacity");
    // Rows older than the window of the last one are never read: skip them.
    const std::size_t skip = n > window ? n - window : 0;
    for (std::size_t r0 = skip; r0 < n;) {
        const std::size_t m = n - r0 >= 32 ? 32 : std::min<std::size_t>(8, n - r0);
        const std::uint64_t p = pos0 + r0;
        linear(0, lin_fc, taps_rows + r0 * tap_width, m, ctx_fc_, st);
        rms_norm(ctx_fc_, dev_.hidden_norm, m, ctx_c_, st);
        for (std::uint32_t L = 0; L < layers; ++L) {
            const LayerWeights& y = dev_.layer[L];
            linear_group(L, {lin_k, lin_v}, ctx_c_, m, {ctx_k_, ctx_v_}, st);
            hipLaunchKernelGGL(head_norm_rope_kernel, dim3(blocks_of(m * kv_heads, 8)), dim3(256), 0, st, ctx_k_,
                               static_cast<std::uint32_t>(m), kv_heads, y.k_norm, rope_cos_, rope_sin_, p, kb16_);
            const std::size_t off = static_cast<std::size_t>(L) * window * kv_width;
            hipLaunchKernelGGL(ring_store_kernel, dim3(static_cast<unsigned>(m)), dim3(256), 0, st, kb16_, ctx_v_,
                               ring_k_[slot] + off, ring_v_[slot] + off, p);
        }
        hip_ok(hipGetLastError(), "append");
        for (std::size_t r = 0; r < m; ++r) ring_pos_[slot][(p + r) % window] = static_cast<std::int64_t>(p + r);
        r0 += m;
    }
}

void DeviceDrafter::forward_block(std::uint32_t slot, const std::uint16_t* anchor_row, const std::uint16_t* noise_row, std::uint64_t position,
                          hipStream_t st) {
    if (slot >= opt_.slots) throw std::invalid_argument("dspark: block slot");
    if (position + block > opt_.capacity) throw std::out_of_range("dspark: block beyond capacity");
    const std::uint64_t first = position > context_keys ? position - context_keys : 0;
    const auto ctx = static_cast<std::uint32_t>(position - first);
    for (std::uint64_t p = first; p < position; ++p)
        if (ring_pos_[slot][p % window] != static_cast<std::int64_t>(p))
            throw std::runtime_error("dspark block: context position " + std::to_string(p) + " missing from the ring");
    const std::size_t nr = static_cast<std::size_t>(block) * hidden;
    hipLaunchKernelGGL(embed_block_kernel, dim3(block), dim3(256), 0, st, anchor_row, noise_row, r_);
    for (std::uint32_t L = 0; L < layers; ++L) {
        const LayerWeights& w = dev_.layer[L];
        rms_norm(r_, w.input_norm, block, x_, st);
        linear_group(L, {lin_q, lin_k, lin_v}, x_, block, {q_, kb_, vb_}, st);
        head_norm_rope(q_, block, heads, w.q_norm, position, st);
        hipLaunchKernelGGL(head_norm_rope_kernel, dim3(blocks_of(block * kv_heads, 8)), dim3(256), 0, st, kb_, block, kv_heads,
                           w.k_norm, rope_cos_, rope_sin_, position, kb16_);
        hipLaunchKernelGGL(to_bf16_kernel, dim3(blocks_of(block * kv_width, 256)), dim3(256), 0, st, vb_, vb16_,
                           static_cast<std::size_t>(block) * kv_width);
        const std::size_t off = static_cast<std::size_t>(L) * window * kv_width;
        if (opt_.attn_chunk == 0 || ctx + block <= kChunk) {  // one chunk is the v1 order: the single-kernel path, same bits
            hipLaunchKernelGGL(attention_kernel, dim3(block * heads), dim3(128), (ctx + block) * sizeof(float), st, q_,
                               ring_k_[slot] + off, ring_v_[slot] + off, kb16_, vb16_, first, ctx, att_);
        } else {
            const std::uint32_t keys = ctx + block, chunks = (keys + kChunk - 1) / kChunk;
            if (opt_.attn_variant == 1)
                hipLaunchKernelGGL(attn_scores2_kernel, dim3(kv_heads, chunks), dim3(256), 0, st, q_, ring_k_[slot] + off, kb16_, first, ctx,
                                   att_s_);
            else if (opt_.attn_variant == 2 || opt_.attn_variant == 3)
                hipLaunchKernelGGL((attn_scores3_kernel<0>), dim3(kv_heads, (keys + 31) / 32, block), dim3(128), 0, st, q_, ring_k_[slot] + off,
                                   kb16_, first, ctx, att_s_);
            else if (opt_.attn_variant == 4)
                hipLaunchKernelGGL((attn_scores3_kernel<8>), dim3(kv_heads, (keys + 31) / 32, block), dim3(128), 0, st, q_, ring_k_[slot] + off,
                                   kb16_, first, ctx, att_s_);
            else if (opt_.attn_variant == 5)
                hipLaunchKernelGGL((attn_scores3_kernel<12>), dim3(kv_heads, (keys + 31) / 32, block), dim3(128), 0, st, q_, ring_k_[slot] + off,
                                   kb16_, first, ctx, att_s_);
            else
                hipLaunchKernelGGL(attn_scores_kernel, dim3(kv_heads, chunks, block), dim3(256), 0, st, q_, ring_k_[slot] + off, kb16_, first,
                                   ctx, att_s_);
            hipLaunchKernelGGL(attn_exp_kernel, dim3(block * heads), dim3(256), 0, st, att_s_, keys);
            if (opt_.attn_variant == 1)
                hipLaunchKernelGGL(attn_pv2_kernel, dim3(kv_heads, chunks), dim3(256), 0, st, att_s_, ring_v_[slot] + off, vb16_, first, ctx,
                                   att_po_, att_ps_);
            else if (opt_.attn_variant >= 3)
                hipLaunchKernelGGL(attn_pv3_kernel, dim3(kv_heads, chunks, block), dim3(128), 0, st, att_s_, ring_v_[slot] + off, vb16_, first,
                                   ctx, att_po_, att_ps_);
            else
                hipLaunchKernelGGL(attn_pv_kernel, dim3(kv_heads, chunks, block), dim3(256), 0, st, att_s_, ring_v_[slot] + off, vb16_, first,
                                   ctx, att_po_, att_ps_);
            hipLaunchKernelGGL(attn_combine_kernel, dim3(block * heads), dim3(head_dim), 0, st, att_po_, att_ps_, chunks, att_);
        }
        linear(L, lin_o, att_, block, o_, st);
        hipLaunchKernelGGL(add_kernel, dim3(blocks_of(nr, 256)), dim3(256), 0, st, r_, o_, nr);
        rms_norm(r_, w.post_norm, block, x_, st);
        linear_group(L, {lin_gate, lin_up}, x_, block, {gate_, up_}, st);
        hipLaunchKernelGGL(swiglu_kernel, dim3(blocks_of(static_cast<std::size_t>(block) * inter, 256)), dim3(256), 0, st, gate_, up_,
                           static_cast<std::size_t>(block) * inter);
        linear(L, lin_down, gate_, block, o_, st);
        hipLaunchKernelGGL(add_kernel, dim3(blocks_of(nr, 256)), dim3(256), 0, st, r_, o_, nr);
    }
    rms_norm(r_, dev_.final_norm, block, y_, st);
    hip_ok(hipGetLastError(), "block");
}

void DeviceDrafter::markov_step(std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1, Candidate* out,
                                hipStream_t st) {
    if (v1 <= v0 || v1 > vocab) throw std::invalid_argument("dspark: markov range");
    hip_ok(hipMemcpyAsync(prev_dev_, &prev, 4, hipMemcpyHostToDevice, st), "prev");
    const unsigned nb = blocks_of(v1 - v0, 8);
    hipLaunchKernelGGL(markov_kernel, dim3(nb), dim3(256), 0, st, dev_.markov_w1, dev_.markov_w2, prev_dev_, base, v0, v1, cand_);
    hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(256), 0, st, cand_, nb, out, nullptr);
    hip_ok(hipGetLastError(), "markov");
    hip_ok(hipStreamSynchronize(st), "markov sync");  // `prev` is a host value read by the copy above
}

void DeviceDrafter::confidence(const std::array<std::uint32_t, block>& prev, float* out, hipStream_t st) {
    hip_ok(hipMemcpyAsync(prev_dev_, prev.data(), block * 4, hipMemcpyHostToDevice, st), "prev");
    hipLaunchKernelGGL(confidence_kernel, dim3(1), dim3(32 * block), 0, st, y_, dev_.markov_w1, prev_dev_, dev_.confidence_w,
                       dev_.confidence_b, out);
    hip_ok(hipGetLastError(), "confidence");
    hip_ok(hipStreamSynchronize(st), "confidence sync");
}

Candidate DeviceDrafter::markov_step_host(std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1, hipStream_t st) {
    markov_step(prev, base, v0, v1, pick_, st);
    Candidate c;
    hip_ok(hipMemcpy(&c, pick_, sizeof c, hipMemcpyDeviceToHost), "pick");
    return c;
}

std::array<float, block> DeviceDrafter::confidence_host(const std::array<std::uint32_t, block>& prev, hipStream_t st) {
    confidence(prev, conf_, st);
    std::array<float, block> c{};
    hip_ok(hipMemcpy(c.data(), conf_, block * 4, hipMemcpyDeviceToHost), "confidence");
    return c;
}

Draft DeviceDrafter::sample(std::uint32_t anchor, const float* base, hipStream_t st) {
    Draft d;
    std::array<std::uint32_t, block> prev{};
    std::uint32_t t = anchor;
    for (std::uint32_t k = 0; k < block; ++k) {
        prev[k] = t;
        markov_step(t, base + static_cast<std::size_t>(k) * vocab, 0, vocab, pick_, st);
        Candidate c;
        hip_ok(hipMemcpy(&c, pick_, sizeof c, hipMemcpyDeviceToHost), "pick");
        t = c.id;
        d.tokens[k] = t;
    }
    confidence(prev, conf_, st);
    hip_ok(hipMemcpy(d.confidence.data(), conf_, block * 4, hipMemcpyDeviceToHost), "confidence");
    return d;
}

} // namespace strix::minimax::dspark
