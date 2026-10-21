// gfx1151 kernels of the deepseek41 decode attention. The arithmetic of each kernel is its host reference in
// src/ds41_attention.cpp (same chains, same xor butterflies, same scalar helpers), so results match bit for bit.
#include "strix/hip_ds41_attention.hpp"

#include <cstdlib>
#include <stdexcept>

namespace strix::ds41::attn::hip {
#if defined(STRIX_DS41_ATTN_TIMING)
// Timing build only: thread 0 of every block records wall clock and shader cycles at entry and exit.
__device__ unsigned long long* g_ds41_timing;
void set_timing_buffer(void* p) {
    if (hipMemcpyToSymbol(HIP_SYMBOL(g_ds41_timing), &p, sizeof(p)) != hipSuccess) throw std::runtime_error("timing buffer");
}
#endif
namespace {
#if defined(STRIX_DS41_ATTN_TIMING)
struct BlockTimer {
    unsigned slot; unsigned long long w0, c0;
    __device__ explicit BlockTimer(unsigned s) : slot(s), w0(static_cast<unsigned long long>(wall_clock64())), c0(__builtin_readcyclecounter()) {}
    __device__ ~BlockTimer() {
        if (threadIdx.x != 0U || g_ds41_timing == nullptr) return;
        const unsigned long long b = blockIdx.x + gridDim.x * (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);
        unsigned long long* t = g_ds41_timing + (static_cast<unsigned long long>(slot) * 65536ULL + b) * 4ULL;
        t[0] = w0; t[1] = static_cast<unsigned long long>(wall_clock64()); t[2] = c0; t[3] = __builtin_readcyclecounter();
    }
};
#define DS41_TIMED(s) const BlockTimer ds41_timer_(s)
// phase mark k of the block (thread 0, after its outstanding memory operations): slots 7.. of the timing buffer
#define DS41_MARK(k) do { __builtin_amdgcn_s_waitcnt(0); if (threadIdx.x == 0U && g_ds41_timing != nullptr) {     const unsigned long long b_ = blockIdx.x + gridDim.x * (blockIdx.y + static_cast<unsigned long long>(gridDim.y) * blockIdx.z);     g_ds41_timing[((7ULL + (k) / 4U) * 65536ULL + b_) * 4ULL + (k) % 4U] = static_cast<unsigned long long>(wall_clock64()); } } while (0)
#else
#define DS41_TIMED(s) (void)0
#define DS41_MARK(k) (void)0
#endif

// xor lane exchange without LDS: DPP row_xmask inside the 16-lane rows, permlanex16 across them
template <unsigned D> __device__ __forceinline__ float xor_lane(float v) {
    const unsigned x = __builtin_bit_cast(unsigned, v);
    if constexpr (D == 16U) return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false, false));
    else return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(static_cast<int>(x), static_cast<int>(x), static_cast<int>(0x160U + D), 0xF, 0xF, false));
}
__device__ __forceinline__ float wave_sum(float v) {
    v = v + xor_lane<16>(v);
    v = v + xor_lane<8>(v);
    v = v + xor_lane<4>(v);
    v = v + xor_lane<2>(v);
    v = v + xor_lane<1>(v);
    return v;
}
__device__ __forceinline__ float wave_max(float v, int from = 16) {
    for (int d = from; d != 0; d /= 2) v = fmaxf(v, __shfl_xor(v, d, 32));
    return v;
}
// RMS norm of the wave's 512 vector (lane: 16 dims), in place: (x * r) * w.
__device__ __forceinline__ void norm512(float (&x)[16], const float* w, unsigned lane) {
    float acc = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc = fmaf(x[i], x[i], acc);
    const float r = 1.0F / sqrtf(wave_sum(acc) / 512.0F + kNormEps);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = (x[i] * r) * w[16 * lane + i];
}
// lanes 28..31 own dims 448..511 = rope pairs 8(lane-28)..+7
__device__ __forceinline__ void rope512(float (&x)[16], const RopeRow& rope, unsigned lane, bool inverse) {
    if (lane < 28U) return;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned pair = 8U * (lane - 28U) + i;
        rope_pair(x[2 * i], x[2 * i + 1], rope.c[pair], rope.s[pair], inverse);
    }
}

__global__ __launch_bounds__(32) void window_write_kernel(WindowView v) {
    DS41_TIMED(0U);
    const unsigned r = blockIdx.x, lane = threadIdx.x;
    float x[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = v.kv[r * kHeadDim + 16 * lane + i];
    norm512(x, v.norm, lane);
    rope512(x, v.rope[r], lane, false);
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(fmaxf(amax, __shfl_xor(amax, 1, 32)), 1e-4F);
    const int e = log2_ceil(amax * (1.0F / 448.0F));
    const float s = pow2f(e);
    std::uint8_t* entry = v.ring + (v.pos[r] % v.capacity) * kWindowEntryBytes;
    std::uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        w[i / 4] |= static_cast<std::uint32_t>(e4m3_encode(clampf(x[i] / s, -448.0F, 448.0F))) << (8U * (i % 4));
    *reinterpret_cast<uint4*>(entry + 16 * lane) = make_uint4(w[0], w[1], w[2], w[3]);
    if ((lane & 1U) == 0U) entry[512 + lane / 2] = static_cast<std::uint8_t>(e + 127);
}

// Ahead (STRIX_DS41_COMPRESS_AHEAD=1, default off): thread tid's index key row of wk is read kAheadSteps steps of 64
// weights ahead (the first ones before the latent barrier) instead of one step at a time;
// the chain of each output (512 fmaf over d ascending) and everything else are unchanged.
constexpr unsigned kAheadSteps = 3;
template <bool Ahead>
__global__ __launch_bounds__(128) void compress_kernel(CompressView v) {
    DS41_TIMED(1U);
    __shared__ float latent_s[kHeadDim];
    __shared__ float k_s[kIdxDim];
    const unsigned r = blockIdx.x, tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const std::uint64_t p = v.pos[r];
    [[maybe_unused]] const uint4* wrow = reinterpret_cast<const uint4*>(v.wk + tid * kHeadDim);  // 8 F16 weights per uint4
    [[maybe_unused]] uint4 ahead[kAheadSteps][8];
    float x[16];
    if (v.ratio == 2U) {
        const float* kv = v.kv + r * kHeadDim + 16 * lane;
        const float* sc = v.score + r * kHeadDim + 16 * lane;
        if (wave == 0U && r + 1U == v.rows) {  // the pending group after the last row
            const bool pending = (p % 2U) == 0U;
            for (unsigned i = 0; i < 16; ++i) {
                v.state_out[16 * lane + i] = pending ? kv[i] : 0.0F;
                v.state_out[kHeadDim + 16 * lane + i] = pending ? sc[i] : 0.0F;
            }
            if (lane == 0U) *v.state_valid_out = pending ? 1U : 0U;
        }
        if ((p % 2U) == 0U) return;  // uniform over the block (no barrier passed yet)
        if (wave == 0U) {
            const float* kv0 = r > 0U ? v.kv + (r - 1U) * kHeadDim + 16 * lane : v.state_in + 16 * lane;
            const float* sc0 = r > 0U ? v.score + (r - 1U) * kHeadDim + 16 * lane : v.state_in + kHeadDim + 16 * lane;
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) {
                const float m = fmaxf(sc0[i], sc[i]);
                const float e0 = exp_f(sc0[i] - m), e1 = exp_f(sc[i] - m);
                const float sum = e0 + e1;
                x[i] = kv0[i] * (e0 / sum) + kv[i] * (e1 / sum);
            }
        }
    } else if (wave == 0U) {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = v.kv[r * kHeadDim + 16 * lane + i];
    }
    if (wave == 0U) {
        norm512(x, v.norm, lane);
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) latent_s[16 * lane + i] = x[i];
    }
    if constexpr (Ahead) {  // in flight across the barrier
#pragma unroll
        for (unsigned a = 0; a < kAheadSteps; ++a)
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) ahead[a][u] = wrow[8 * a + u];
    }
    __syncthreads();
    if constexpr (!Ahead) {   // index key output tid: one 512-term fmaf chain over d ascending; 64 weights loaded ahead per step
        const std::uint16_t* row = v.wk + tid * kHeadDim;
        float acc = 0.0F;
        for (unsigned d0 = 0; d0 < kHeadDim; d0 += 64) {
            uint4 h[8];
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) h[u] = *reinterpret_cast<const uint4*>(row + d0 + 8 * u);
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) {
                const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    acc = fmaf(f16_to_float(static_cast<std::uint16_t>(hw[j / 2] >> (16U * (j % 2)))), latent_s[d0 + 8 * u + j], acc);
            }
        }
        k_s[tid] = acc;
    } else {  // the same chain; step n's weights were requested kAheadSteps steps earlier
        float acc = 0.0F;
#pragma unroll 1
        for (unsigned n = 0; n < kHeadDim / 64U; ++n) {  // not unrolled (one step's live values, as above): the queue shifts
            uint4 h[8];
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) h[u] = ahead[0][u];
#pragma unroll
            for (unsigned a = 0; a + 1U < kAheadSteps; ++a)
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) ahead[a][u] = ahead[a + 1U][u];
            if (n + kAheadSteps < kHeadDim / 64U) {
#pragma unroll
                for (unsigned u = 0; u < 8; ++u) ahead[kAheadSteps - 1U][u] = wrow[8 * (n + kAheadSteps) + u];
            }
#pragma unroll
            for (unsigned u = 0; u < 8; ++u) {
                const std::uint32_t hw[4] = {h[u].x, h[u].y, h[u].z, h[u].w};
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    acc = fmaf(f16_to_float(static_cast<std::uint16_t>(hw[j / 2] >> (16U * (j % 2)))), latent_s[64 * n + 8 * u + j], acc);
            }
        }
        k_s[tid] = acc;
    }
    __syncthreads();
    if (wave != 0U) return;
    const RopeRow& rope = v.rope[r];
    const std::uint64_t slot = p / v.ratio;
    float k[4];
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) k[o] = k_s[4 * lane + o];
    float ss = 0.0F;
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) ss = fmaf(k[o], k[o], ss);
    const float rk = 1.0F / sqrtf(wave_sum(ss) / 128.0F + kNormEps);
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) k[o] = (k[o] * rk) * v.knorm[4 * lane + o];
    if (lane >= 16U) {
        rope_pair(k[0], k[1], rope.c[2 * (lane - 16U)], rope.s[2 * (lane - 16U)], false);
        rope_pair(k[2], k[3], rope.c[2 * (lane - 16U) + 1], rope.s[2 * (lane - 16U) + 1], false);
    }
    float amax = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])), fmaxf(fabsf(k[2]), fabsf(k[3])));
    amax = fmaxf(wave_max(amax, 4), 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
    std::uint8_t* ie = v.idx_cache + slot * kIdxEntryBytes;
    std::uint8_t c[4];
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) c[o] = e2m1_encode(clampf(k[o] / s, -6.0F, 6.0F));
    ie[2 * lane] = static_cast<std::uint8_t>(c[0] | (c[1] << 4U));
    ie[2 * lane + 1] = static_cast<std::uint8_t>(c[2] | (c[3] << 4U));
    if ((lane & 7U) == 0U) ie[64 + lane / 8] = static_cast<std::uint8_t>(e + 127);
    // compressed KV: RoPE the latent, FP4 with an E4M3 scale per lane (16 dims)
    rope512(x, rope, lane, false);
    float am = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) am = fmaxf(am, fabsf(x[i]));
    am = fmaxf(am, 0.01171875F);
    const std::uint8_t sc = e4m3_encode(am / 6.0F);
    const float sv = e4m3_value(sc);
    std::uint32_t w[2] = {0, 0};
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        w[i / 8] |= static_cast<std::uint32_t>(e2m1_encode(clampf(x[i] / sv, -6.0F, 6.0F))) << (4U * (i % 8));
    std::uint8_t* ke = v.kv_cache + slot * kKvEntryBytes;
    *reinterpret_cast<uint2*>(ke + 8 * lane) = make_uint2(w[0], w[1]);
    ke[256 + lane] = sc;
}

// 4 E2M1 codes (16 bits: dims 4i..4i+3, low nibble first) -> 4 int8 values (2 x value).
__device__ __forceinline__ std::uint32_t e2m1x4_to_i8(std::uint32_t x) {
    const std::uint32_t sel = (x & 7U) | ((x << 4U) & 0x700U) | ((x << 8U) & 0x70000U) | ((x << 12U) & 0x7000000U);
    const std::uint32_t mag = __builtin_amdgcn_perm(0x0C080604U, 0x03020100U, sel);
    const std::uint32_t sg = ((x >> 3U) & 1U) | ((x >> 7U) & 1U) << 8U | ((x >> 11U) & 1U) << 16U | ((x >> 15U) & 1U) << 24U;
    std::uint32_t nz = mag | (mag >> 2U);
    nz = (nz | (nz >> 1U)) & 0x01010101U;
    const std::uint32_t m = (sg & nz) * 0xFFU;
    return (mag ^ m) + (m & 0x01010101U);
}

// Index queries, once per row: head tid/8, dims 16 (tid%8) .. +15; RoPE on head dims 64..127; E2M1 per 32
// (threads tid, tid^1) as packed int8 (2 x value), the block scale 2^e as float, and the head weight / 64.
struct QPrep { int qv[kIdxHeads][32]; float qs[kIdxHeads][4]; float w[kIdxHeads]; };
static_assert(sizeof(QPrep) == kIdxPrepBytes);
__global__ __launch_bounds__(256) void index_qprep_kernel(IndexView v) {
    DS41_TIMED(2U);
    const unsigned r = blockIdx.x, tid = threadIdx.x, h = tid / 8U, part = tid % 8U;
    QPrep& out = reinterpret_cast<QPrep*>(v.qprep)[r];
    float q[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) q[i] = v.q[(r * kIdxHeads + h) * kIdxDim + 16 * part + i];
    if (part >= 4U) {
        const RopeRow& rope = v.rope[r];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) rope_pair(q[2 * i], q[2 * i + 1], rope.c[8 * (part - 4U) + i], rope.s[8 * (part - 4U) + i], false);
    }
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
    amax = fmaxf(fmaxf(amax, __shfl_xor(amax, 1, 32)), 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        std::uint32_t packed = 0;
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            packed |= (static_cast<std::uint32_t>(e2m1_int2(e2m1_encode(clampf(q[4 * i + j] / s, -6.0F, 6.0F)))) & 255U) << (8U * j);
        out.qv[h][4 * part + i] = static_cast<int>(packed);
    }
    if ((part & 1U) == 0U) out.qs[h][part / 2] = s;
    if (tid < kIdxHeads) out.w[tid] = v.w[r * kIdxHeads + tid] * kIdxWeightScale;
}
// Scores: thread = position; its key decoded once, then every row in turn (each row's arithmetic unchanged). The
// rows' quantised queries are wave-uniform and no global store precedes their reads, so they come in with scalar
// loads (the dot products take them as scalar operands); scores wait in LDS and are stored after every kMaxRows rows
// (calls of up to kMaxCallRows rows: the key is still decoded once).
__global__ __launch_bounds__(256) void index_scores_kernel(IndexView v) {
    DS41_TIMED(3U);
    __shared__ float out_s[kMaxRows][256];
    const unsigned tid = threadIdx.x;
    const std::uint64_t t = static_cast<std::uint64_t>(blockIdx.x) * 256U + tid;
    if (t >= v.max_count) return;
    const QPrep* prep = static_cast<const QPrep*>(v.qprep);
    const std::uint32_t* e = reinterpret_cast<const std::uint32_t*>(v.idx_cache + t * kIdxEntryBytes);
    int kv8[32];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        const std::uint32_t w = e[i];
        kv8[2 * i] = static_cast<int>(e2m1x4_to_i8(w & 0xFFFFU));
        kv8[2 * i + 1] = static_cast<int>(e2m1x4_to_i8(w >> 16U));
    }
    const std::uint32_t sc = e[16];
    float ks[4];
#pragma unroll
    for (unsigned b = 0; b < 4; ++b) ks[b] = e8m0_value(static_cast<std::uint8_t>(sc >> (8U * b)));
#pragma unroll 1
    for (unsigned r0 = 0; r0 < v.rows; r0 += kMaxRows) {
        const unsigned nr = v.rows - r0 < kMaxRows ? v.rows - r0 : kMaxRows;
#pragma unroll 1
        for (unsigned rr = 0; rr < nr; ++rr) {
            const QPrep& q = prep[r0 + rr];
            float score = 0.0F;
#pragma unroll 2
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                float term[4];
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) {
                    int dot = 0;
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i) dot = __builtin_amdgcn_sudot4(true, q.qv[h][8 * b + i], true, kv8[8 * b + i], dot, false);
                    term[b] = ((static_cast<float>(dot) * q.qs[h][b]) * ks[b]) * 0.25F;
                }
                const float d = ((term[0] + term[1]) + term[2]) + term[3];
                score = fmaf(fmaxf(d, 0.0F), q.w[h], score);
            }
            out_s[rr][tid] = score;
        }
        for (unsigned rr = 0; rr < nr; ++rr) {
            const unsigned r = r0 + rr;
            if (t < (v.pos[r] + 1U) / v.ratio) v.scores[r * v.max_count + t] = out_s[rr][tid];
        }
    }
}

// ---- deterministic top-k: radix threshold over order keys, then ordered compaction ----
constexpr unsigned kSelThreads = 1024;
__device__ unsigned block_exclusive_scan(unsigned x, unsigned* warp_tot, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    unsigned incl = x;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
        if (lane >= static_cast<unsigned>(d)) incl += y;
    }
    if (lane == 31U) warp_tot[w] = incl;
    __syncthreads();
    if (w == 0U) {
        unsigned t = warp_tot[lane], ti = t;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const unsigned y = __shfl_up(ti, static_cast<unsigned>(d), 32);
            if (lane >= static_cast<unsigned>(d)) ti += y;
        }
        warp_tot[lane] = ti - t;
        if (lane == 31U) warp_tot[32] = ti;
    }
    __syncthreads();
    const unsigned r = warp_tot[w] + incl - x;
    total = warp_tot[32];
    __syncthreads();
    return r;
}
// Threshold T (the k-th largest key) and how many keys equal to T are taken (need); requires n > k. Per-wave
// histograms (integer counts, so the result does not depend on the order of the adds); wave 0 scans the 256 bins
// from the top, 8 per lane, with a lane prefix scan.
constexpr unsigned kSelWaves = kSelThreads / 32;
__device__ void radix_threshold(const std::uint32_t* keys, std::uint32_t n, std::uint32_t k, unsigned (*hist)[256],
    std::uint32_t& thr, std::uint32_t& need) {
    __shared__ std::uint32_t prefix_s, rem_s;
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    std::uint32_t prefix = 0, mask = 0, rem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned b = lane; b < 256U; b += 32U) hist[w][b] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n; i += kSelThreads) {
            const std::uint32_t key = keys[i];
            if ((key & mask) == prefix) atomicAdd(&hist[w][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (w == 0U) {
            unsigned cnt[8], local = 0;   // lane owns bins 255 - 8 lane - i, i = 0..7 (descending)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned b = 255U - 8U * lane - i;
                unsigned c = 0;
                for (unsigned ww = 0; ww < kSelWaves; ++ww) c += hist[ww][b];
                cnt[i] = c; local += c;
            }
            unsigned incl = local;
#pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const unsigned y = __shfl_up(incl, static_cast<unsigned>(d), 32);
                if (lane >= static_cast<unsigned>(d)) incl += y;
            }
            const unsigned before = incl - local;
            if (before < rem && incl >= rem) {   // exactly one lane
                unsigned cum = before, i = 0;
                while (cum + cnt[i] < rem) { cum += cnt[i]; ++i; }
                prefix_s = prefix | ((255U - 8U * lane - i) << shift);
                rem_s = rem - cum;
            }
        }
        __syncthreads();
        prefix = prefix_s; rem = rem_s;
        mask |= 255U << shift;
        __syncthreads();
    }
    thr = prefix; need = rem;
}
// Visits keys in index order; f(i, selected, out_index) for every i < n.
template <class F>
__device__ void ordered_select(const std::uint32_t* keys, std::uint32_t n, std::uint32_t thr, std::uint32_t need,
    unsigned* warp_tot, F f) {
    std::uint32_t eq_carry = 0, out_carry = 0;
    for (std::uint32_t base = 0; base < n; base += kSelThreads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const unsigned eq = i < n && key == thr ? 1U : 0U;
        unsigned eq_total;
        const unsigned eq_rank = block_exclusive_scan(eq, warp_tot, eq_total) + eq_carry;
        const unsigned sel = i < n && (key > thr || (eq != 0U && eq_rank < need)) ? 1U : 0U;
        unsigned sel_total;
        const unsigned out = block_exclusive_scan(sel, warp_tot, sel_total) + out_carry;
        if (i < n) f(i, sel != 0U, out);
        eq_carry += eq_total; out_carry += sel_total;
    }
}

__global__ __launch_bounds__(kSelThreads) void select_kernel(SelectView v) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    const unsigned r = blockIdx.x;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    std::uint32_t* keys = v.keys + static_cast<std::size_t>(r) * v.max_count;
    std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const float ninf = -__builtin_inff();
    if (v.pool == Pool::candidate_source && n != 0U) {
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) {
            float m = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) m = fmaxf(m, scores[t]);
            keys[b] = order_key(b == last ? __builtin_inff() : m);
        }
        __syncthreads();
        const std::uint32_t ninf_key = order_key(ninf);
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) cand[b] = keys[b] > ninf_key ? 1U : 0U;
        } else {
            std::uint32_t thr, need;
            radix_threshold(keys, nb, v.cand_blocks, hist, thr, need);
            ordered_select(keys, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
                cand[b] = s && keys[b] > ninf_key ? 1U : 0U;
            });
        }
        __syncthreads();
    }
    for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads)
        keys[t] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    __syncthreads();
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    std::uint32_t thr, need;
    radix_threshold(keys, n, v.topk, hist, thr, need);
    ordered_select(keys, n, thr, need, warp_tot, [&](std::uint32_t t, bool s, unsigned out) { if (s) idx[out] = t; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}

// Two-level selection for long rows (max_count > kSelSlice). Level one, one block per slice of 4096 positions: the
// slice's keys in LDS, its own top-k (same rule: key, then the smaller position) written in position order and
// padded to k with key 0 (below every float's key); for the candidate source also the slice's block maxima. Every
// element of the row's top-k is in its slice's top-k (fewer than k elements of the slice beat it), so level two, the
// same selection over the slices' survivors in slice order, gives exactly the row's top-k, in ascending positions.
// Scratch (keys, per row): survivor keys [S x k], survivor positions [S x k], block-maximum keys [nb].
constexpr unsigned kSelSlice = 4096;
struct SelScratch { std::uint32_t *keys, *pos, *bmax; unsigned slices; };
__device__ __forceinline__ SelScratch sel_scratch(const SelectView& v, unsigned r) {
    const unsigned slices = (v.max_count + kSelSlice - 1) / kSelSlice;
    std::uint32_t* base = v.keys + static_cast<std::size_t>(r) * v.max_count;
    return {base, base + static_cast<std::size_t>(slices) * v.topk, base + 2 * static_cast<std::size_t>(slices) * v.topk, slices};
}
__global__ __launch_bounds__(kSelThreads) void select_slice_kernel(SelectView v) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    __shared__ std::uint32_t skeys[kSelSlice];
    const unsigned s = blockIdx.x, r = blockIdx.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio), t0 = s * kSelSlice;
    if (t0 >= n) return;
    const std::uint32_t m = n - t0 < kSelSlice ? n - t0 : kSelSlice;
    const float* scores = v.scores + static_cast<std::size_t>(r) * v.max_count;
    const std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
    const SelScratch sc = sel_scratch(v, r);
    const float ninf = -__builtin_inff();
    for (std::uint32_t i = threadIdx.x; i < m; i += kSelThreads) {
        const std::uint32_t t = t0 + i;
        skeys[i] = order_key(v.pool == Pool::candidate_user && cand[t / v.block_size] == 0U ? ninf : scores[t]);
    }
    if (v.pool == Pool::candidate_source) {
        const std::uint32_t b0 = t0 / v.block_size, b1 = (t0 + m + v.block_size - 1) / v.block_size, last = (n - 1) / v.block_size;
        for (std::uint32_t b = b0 + threadIdx.x; b < b1; b += kSelThreads) {
            float mx = ninf;
            for (std::uint32_t t = b * v.block_size; t < (b + 1) * v.block_size && t < n; ++t) mx = fmaxf(mx, scores[t]);
            sc.bmax[b] = order_key(b == last ? __builtin_inff() : mx);
        }
    }
    __syncthreads();
    if (n <= v.topk) return;   // level two takes every position
    std::uint32_t* ok = sc.keys + static_cast<std::size_t>(s) * v.topk;
    std::uint32_t* op = sc.pos + static_cast<std::size_t>(s) * v.topk;
    if (m <= v.topk) {
        for (std::uint32_t i = threadIdx.x; i < v.topk; i += kSelThreads) { ok[i] = i < m ? skeys[i] : 0U; op[i] = t0 + i; }
        return;
    }
    std::uint32_t thr, need;
    radix_threshold(skeys, m, v.topk, hist, thr, need);
    ordered_select(skeys, m, thr, need, warp_tot, [&](std::uint32_t i, bool sel, unsigned out) {
        if (sel) { ok[out] = skeys[i]; op[out] = t0 + i; }
    });
}
// Level two: block 0 of a row selects the top-k among the survivors; block 1 (candidate source only) the candidate
// blocks among the block maxima.
__global__ __launch_bounds__(kSelThreads) void select_merge_kernel(SelectView v) {
    DS41_TIMED(4U);
    __shared__ unsigned hist[kSelWaves][256];
    __shared__ unsigned warp_tot[33];
    const unsigned r = blockIdx.y;
    const std::uint32_t n = static_cast<std::uint32_t>((v.pos[r] + 1U) / v.ratio);
    const SelScratch sc = sel_scratch(v, r);
    if (blockIdx.x == 1U) {
        if (n == 0U) return;
        std::uint8_t* cand = v.candidates + static_cast<std::size_t>(r) * ((v.max_count + v.block_size - 1) / v.block_size);
        const std::uint32_t nb = (n + v.block_size - 1) / v.block_size, ninf_key = order_key(-__builtin_inff());
        if (nb <= v.cand_blocks) {
            for (std::uint32_t b = threadIdx.x; b < nb; b += kSelThreads) cand[b] = sc.bmax[b] > ninf_key ? 1U : 0U;
            return;
        }
        std::uint32_t thr, need;
        radix_threshold(sc.bmax, nb, v.cand_blocks, hist, thr, need);
        ordered_select(sc.bmax, nb, thr, need, warp_tot, [&](std::uint32_t b, bool s, unsigned) {
            cand[b] = s && sc.bmax[b] > ninf_key ? 1U : 0U;
        });
        return;
    }
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(r) * v.topk;
    if (n <= v.topk) {
        for (std::uint32_t t = threadIdx.x; t < n; t += kSelThreads) idx[t] = t;
        if (threadIdx.x == 0U) v.idx_count[r] = n;
        return;
    }
    const std::uint32_t total = ((n + kSelSlice - 1) / kSelSlice) * v.topk;
    std::uint32_t thr, need;
    radix_threshold(sc.keys, total, v.topk, hist, thr, need);
    ordered_select(sc.keys, total, thr, need, warp_tot, [&](std::uint32_t i, bool s, unsigned out) { if (s) idx[out] = sc.pos[i]; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}

// ---- attention: one block per (key chunk of 64, 8 Hpw heads, row); 8 waves of Hpw heads; lane = 16 dims of each head ----
// Tiles of 16 keys: the block dequantises the tile once into LDS (thread -> key tid/16, dims 32 (tid%16) .. +31) while
// the next tile's bytes are in flight. Scores of 8 keys at a time: the lane fmaf chains, then a transposed xor
// butterfly (DPP and permlanex16, no LDS) that leaves key lane/4 in every lane after exactly the additions of the
// per-key butterfly, so each softmax term is evaluated once. The last block of a (row, head group) combines.
// Heads per wave Hpw (1, 2 or 4; 8 waves per block, so 8 Hpw heads per block) is chosen per call so that the grid has
// enough blocks (enqueue_attention); every choice computes each head identically.
constexpr unsigned kAttnThreads = 256, kAttnMinBlockHeads = kAttnThreads / 32U, kAttnGroups = kHeads / kAttnMinBlockHeads;
static_assert(kAttnGroups == kAttnGroupCount);

struct KeyRaw { uint4 c0, c1; std::uint32_t sc; };
// One key's dims 32 part .. +31 into its LDS row (zeros for a key past the chunk's end). Row layout: float4 i of lane l
// (dims 16 l + 4 i .. +3) at i * 32 + l, so the wave's reads of one float4 index are conflict-free.
__device__ __forceinline__ unsigned key_slot(unsigned part, unsigned i) { return (i % 4U) * 32U + 2U * part + i / 4U; }
__device__ __forceinline__ void store_key(float4* row, unsigned part, const KeyRaw& k, bool win, bool valid) {
    const std::uint32_t cw[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
    if (!valid) {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) row[key_slot(part, i)] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    } else if (win) {
        const float s = e8m0_value(static_cast<std::uint8_t>(k.sc));
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i];
            row[key_slot(part, i)] = make_float4(e4m3_value(static_cast<std::uint8_t>(w)) * s, e4m3_value(static_cast<std::uint8_t>(w >> 8U)) * s,
                                 e4m3_value(static_cast<std::uint8_t>(w >> 16U)) * s, e4m3_value(static_cast<std::uint8_t>(w >> 24U)) * s);
        }
    } else {
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const std::uint32_t w = cw[i / 2] >> (16U * (i % 2));
            const float s = e4m3_value(static_cast<std::uint8_t>(k.sc >> (8U * (i / 4))));
            row[key_slot(part, i)] = make_float4(e2m1_value(static_cast<std::uint8_t>(w & 15U)) * s, e2m1_value(static_cast<std::uint8_t>((w >> 4U) & 15U)) * s,
                                 e2m1_value(static_cast<std::uint8_t>((w >> 8U) & 15U)) * s, e2m1_value(static_cast<std::uint8_t>((w >> 12U) & 15U)) * s);
        }
    }
}
__device__ __forceinline__ float lane_value(float x, unsigned src) {
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), static_cast<int>(src)));
}
// One tile's online-softmax step for the wave's Hpw heads over its first cnt keys (loops over keys kept rolled: the
// kernel is instruction-fetch bound when fully unrolled).
template <unsigned kAttnHpw>
__device__ __forceinline__ void attention_tile(const float4 (*ks)[kHeadDim / 4], unsigned cnt, unsigned lane,
    const float (&q)[kAttnHpw][16], float (&m)[kAttnHpw], float (&l)[kAttnHpw], float (&acc)[kAttnHpw][16]) {
    const float ninf = -__builtin_inff();
    const bool b4 = (lane & 16U) != 0U, b3 = (lane & 8U) != 0U, b2 = (lane & 4U) != 0U;
    float s[kAttnHpw][2];
#pragma unroll 1
    for (unsigned sb = 0; sb < 2; ++sb) {
        float pt[kAttnHpw][8];
#pragma unroll
        for (unsigned jj = 0; jj < 8; ++jj) {
            float k[16];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const float4 kk = ks[8 * sb + jj][32 * i + lane];
                k[4 * i] = kk.x; k[4 * i + 1] = kk.y; k[4 * i + 2] = kk.z; k[4 * i + 3] = kk.w;
            }
#pragma unroll
            for (unsigned h = 0; h < kAttnHpw; ++h) {
                float t = 0.0F;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) t = fmaf(q[h][i], k[i], t);
                pt[h][jj] = t;
            }
        }
        // transposed butterfly: at xor 16 / 8 / 4 the lane keeps the half of its keys selected by lane bit 4 / 3 / 2
        // and adds the partner's partial of the same key (v + v^d as in the per-key butterfly); then xor 2 and 1
#pragma unroll
        for (unsigned h = 0; h < kAttnHpw; ++h) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const float keep = b4 ? pt[h][i + 4] : pt[h][i], send = b4 ? pt[h][i] : pt[h][i + 4];
                pt[h][i] = keep + xor_lane<16>(send);
            }
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const float keep = b3 ? pt[h][i + 2] : pt[h][i], send = b3 ? pt[h][i] : pt[h][i + 2];
                pt[h][i] = keep + xor_lane<8>(send);
            }
            {
                const float keep = b2 ? pt[h][1] : pt[h][0], send = b2 ? pt[h][0] : pt[h][1];
                pt[h][0] = keep + xor_lane<4>(send);
            }
            float t = pt[h][0];
            t = t + xor_lane<2>(t);
            t = t + xor_lane<1>(t);
            const float sv = 8U * sb + lane / 4U < cnt ? t * kAttnScale : ninf;
            if (sb == 0U) s[h][0] = sv; else s[h][1] = sv;
        }
    }
    float pr[kAttnHpw][2];
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
        float tm = fmaxf(s[h][0], s[h][1]);   // max is exact: any order gives the tile max
        tm = fmaxf(tm, xor_lane<4>(tm));
        tm = fmaxf(tm, xor_lane<8>(tm));
        tm = fmaxf(tm, xor_lane<16>(tm));
        const float mn = fmaxf(m[h], tm), sc = exp_f(m[h] - mn);
        pr[h][0] = exp_f(s[h][0] - mn);
        pr[h][1] = exp_f(s[h][1] - mn);
        l[h] = l[h] * sc;
#pragma unroll 1
        for (unsigned j = 0; j < cnt; ++j) l[h] = l[h] + lane_value(j < 8U ? pr[h][0] : pr[h][1], 4U * (j % 8));
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) acc[h][i] = acc[h][i] * sc;
        m[h] = mn;
    }
#pragma unroll 1
    for (unsigned j = 0; j < cnt; ++j) {
        float k[16];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const float4 kk = ks[j][32 * i + lane];
            k[4 * i] = kk.x; k[4 * i + 1] = kk.y; k[4 * i + 2] = kk.z; k[4 * i + 3] = kk.w;
        }
#pragma unroll
        for (unsigned h = 0; h < kAttnHpw; ++h) {
            const float pj = lane_value(j < 8U ? pr[h][0] : pr[h][1], 4U * (j % 8));
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) acc[h][i] = fmaf(pj, k[i], acc[h][i]);
        }
    }
}
// Combine of one (row, head) in chunk order, sink, 1 / L, inverse RoPE.
__device__ __forceinline__ void attention_combine_head(const AttentionView& v, unsigned r, unsigned h, unsigned lane, unsigned chunks) {
    const float* part = v.partial + (static_cast<std::size_t>(r) * v.heads + h) * kMaxChunks * (2 + kHeadDim);
    // every chunk's (m, l) and the lane's 16 accumulators are loaded at once (one memory round trip), then combined
    // in chunk order
    float mc[kMaxChunks], lc[kMaxChunks], pa[kMaxChunks][16], mm = -__builtin_inff();
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        if (c < chunks) {
            const float* pc = part + c * (2 + kHeadDim);
            mc[c] = pc[0]; lc[c] = pc[1];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) pa[c][i] = pc[2 + 16 * lane + i];
        }
    }
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) if (c < chunks) mm = fmaxf(mm, mc[c]);
    float lsum = 0.0F, acc[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        if (c < chunks) {
            const float e = exp_f(mc[c] - mm);
            lsum = fmaf(lc[c], e, lsum);
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) acc[i] = fmaf(pa[c][i], e, acc[i]);
        }
    }
    lsum = lsum + exp_f(v.sink[h] - mm);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc[i] = acc[i] / lsum;
    rope512(acc, v.rope[r], lane, true);
    const std::size_t at = (static_cast<std::size_t>(r) * v.heads + h) * kHeadDim + 16 * lane;
    float* out = v.o + at;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) out[i] = acc[i];
    if (v.codes == nullptr) return;
    // A8 of the 32-value group held by lanes 2k, 2k + 1 (16 values each), quantize_a8's bits: the order-free maximum
    // of |y|, delta = amax / 127 (1 for a zero or tiny group), code = rint(y / delta) clamped to +-127, exact sum.
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(acc[i]));
    amax = fmaxf(amax, xor_lane<1>(amax));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    int sum = 0;
    std::int8_t* code = v.codes + at;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        const float rounded = tiny ? 0.0F : rintf(acc[i] / delta);
        const int c = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        code[i] = static_cast<std::int8_t>(c);
        sum += c;
    }
    sum += __shfl_xor(sum, 1, 32);
    if ((lane & 1U) == 0U) {
        const std::size_t g = at / 32U;
        v.deltas[g] = delta;
        v.sums[g] = sum;
    }
}
__device__ __forceinline__ unsigned visible_keys(const AttentionView& v, unsigned r, unsigned& nw) {
    const std::uint64_t p = v.pos[r];
    nw = p + 1U < kWindow ? static_cast<unsigned>(p + 1U) : kWindow;
    return nw + (v.idx_count != nullptr ? v.idx_count[r] : 0U);
}
template <unsigned kAttnHpw>
__global__ __launch_bounds__(kAttnThreads) void attention_chunk_kernel(AttentionView v) {
    DS41_TIMED(5U);
    __shared__ float4 ks[kTile][kHeadDim / 4];
    __shared__ std::uint32_t cidx[kChunk];
    __shared__ unsigned last_s;
    const unsigned c = blockIdx.x, r = blockIdx.z, tid = threadIdx.x, lane = tid & 31U;
    const unsigned hb = (blockIdx.y * (kAttnThreads / 32U) + (tid >> 5U)) * kAttnHpw;
    unsigned nw;
    const unsigned n = visible_keys(v, r, nw), chunks = (n + kChunk - 1) / kChunk;
    if (c >= chunks) return;
    const unsigned j0 = c * kChunk, end = n - j0 < kChunk ? n : j0 + kChunk;
    const std::uint64_t p = v.pos[r];
    const unsigned first_slot = static_cast<unsigned>((p + 1U - nw) % v.capacity);
    if (tid < kChunk && j0 + tid < end && j0 + tid >= nw) cidx[tid] = v.idx[static_cast<std::size_t>(r) * v.topk + j0 + tid - nw];
    __syncthreads();
    DS41_MARK(0U);
    const unsigned tk = tid / 16U, part = tid % 16U;
    const auto load = [&](unsigned t0, KeyRaw& k) {
        const unsigned j = t0 + tk;
        if (j >= end) return;
        if (j < nw) {
            unsigned slot = first_slot + j;
            if (slot >= v.capacity) slot -= v.capacity;
            const std::uint8_t* e = v.ring + static_cast<std::size_t>(slot) * kWindowEntryBytes;
            const uint4* cp = reinterpret_cast<const uint4*>(e + 32 * part);
            k.c0 = cp[0]; k.c1 = cp[1];
            k.sc = e[512 + part];
        } else {
            const std::uint8_t* e = v.kv_cache + static_cast<std::size_t>(cidx[j - j0]) * kKvEntryBytes;
            k.c0 = *reinterpret_cast<const uint4*>(e + 16 * part);
            k.sc = *reinterpret_cast<const std::uint16_t*>(e + 256 + 2 * part);
        }
    };
    KeyRaw raw{};
    load(j0, raw);
    float q[kAttnHpw][16], m[kAttnHpw], l[kAttnHpw], acc[kAttnHpw][16];
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[h][i] = v.q[(static_cast<std::size_t>(r) * v.heads + hb + h) * kHeadDim + 16 * lane + i];
        rope512(q[h], v.rope[r], lane, false);
        m[h] = -__builtin_inff(); l[h] = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) acc[h][i] = 0.0F;
    }
    DS41_MARK(1U);
    for (unsigned t0 = j0; t0 < end; t0 += kTile) {
        const unsigned cnt = end - t0 < kTile ? end - t0 : kTile;
        store_key(ks[tk], part, raw, t0 + tk < nw, tk < cnt);
        __syncthreads();
        DS41_MARK(2U + 2U * ((t0 - j0) / kTile));
        if (t0 + kTile < end) load(t0 + kTile, raw);
        attention_tile<kAttnHpw>(ks, cnt, lane, q, m, l, acc);
        __syncthreads();
        DS41_MARK(3U + 2U * ((t0 - j0) / kTile));
    }
#pragma unroll
    for (unsigned h = 0; h < kAttnHpw; ++h) {
        float* pp = v.partial + ((static_cast<std::size_t>(r) * v.heads + hb + h) * kMaxChunks + c) * (2 + kHeadDim);
        if (lane == 0U) { pp[0] = m[h]; pp[1] = l[h]; }
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) pp[2 + 16 * lane + i] = acc[h][i];
    }
    DS41_MARK(10U);
    if (v.tickets == nullptr) return;
    // the last chunk block of this (row, head group) to finish combines its heads
    __threadfence();
    __syncthreads();
    std::uint32_t* ticket = v.tickets + r * kAttnGroups + blockIdx.y;
    if (tid == 0U) last_s = atomicAdd(ticket, 1U) == chunks - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
#pragma unroll 1
    for (unsigned h = 0; h < kAttnHpw; ++h) attention_combine_head(v, r, hb + h, lane, chunks);
    if (tid == 0U) *ticket = 0U;
    DS41_MARK(11U);
}
__global__ __launch_bounds__(32) void attention_combine_kernel(AttentionView v) {
    DS41_TIMED(6U);
    unsigned nw;
    const unsigned n = visible_keys(v, blockIdx.y, nw);
    attention_combine_head(v, blockIdx.y, blockIdx.x, threadIdx.x, (n + kChunk - 1) / kChunk);
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void check_rows(std::uint32_t rows) {
    if (rows == 0U || rows > kMaxCallRows) throw std::invalid_argument("ds41 attention: 1..kMaxCallRows rows per call");
}

}  // namespace

void enqueue_window_write(const WindowView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.capacity < kWindow + v.rows) throw std::invalid_argument("ds41 window ring capacity");
    hipLaunchKernelGGL(window_write_kernel, dim3(v.rows), dim3(32), 0, stream, v);
    check(hipGetLastError(), "ds41 window write");
}
int compress_ahead_mode = -1;  // set_compress_ahead: 0 / 1; -1 = STRIX_DS41_COMPRESS_AHEAD
void set_compress_ahead(int mode) { compress_ahead_mode = mode; }
void enqueue_compress(const CompressView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.ratio != 1U && v.ratio != 2U) throw std::invalid_argument("ds41 compress ratio");
    static const bool ahead_env = [] { const char* e = std::getenv("STRIX_DS41_COMPRESS_AHEAD"); return e && e[0] == '1'; }();
    if (compress_ahead_mode < 0 ? ahead_env : compress_ahead_mode != 0) hipLaunchKernelGGL(compress_kernel<true>, dim3(v.rows), dim3(128), 0, stream, v);
    else hipLaunchKernelGGL(compress_kernel<false>, dim3(v.rows), dim3(128), 0, stream, v);
    check(hipGetLastError(), "ds41 compress");
}
void enqueue_index_scores(const IndexView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.max_count == 0U) return;
    if (v.qprep == nullptr) throw std::invalid_argument("ds41 index scores: IndexView::qprep scratch (rows x kIdxPrepBytes) is required");
    hipLaunchKernelGGL(index_qprep_kernel, dim3(v.rows), dim3(256), 0, stream, v);
    hipLaunchKernelGGL(index_scores_kernel, dim3((v.max_count + 255U) / 256U), dim3(256), 0, stream, v);
    check(hipGetLastError(), "ds41 index scores");
}
void enqueue_select(const SelectView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.max_count <= kSelSlice) {
        hipLaunchKernelGGL(select_kernel, dim3(v.rows), dim3(kSelThreads), 0, stream, v);
    } else {
        const std::size_t slices = (v.max_count + kSelSlice - 1) / kSelSlice;
        if (2 * slices * v.topk + (v.max_count + v.block_size - 1) / v.block_size > v.max_count)
            throw std::invalid_argument("ds41 select: top-k or candidate block size too large for the key scratch");
        hipLaunchKernelGGL(select_slice_kernel, dim3(static_cast<unsigned>(slices), v.rows), dim3(kSelThreads), 0, stream, v);
        hipLaunchKernelGGL(select_merge_kernel, dim3(v.pool == Pool::candidate_source ? 2U : 1U, v.rows), dim3(kSelThreads), 0, stream, v);
    }
    check(hipGetLastError(), "ds41 select");
}
void enqueue_attention(const AttentionView& v, hipStream_t stream) {
    check_rows(v.rows);
    if (v.capacity < kWindow + v.rows || (v.idx_count != nullptr && v.topk > kTopK))
        throw std::invalid_argument("ds41 attention capacity / topk");
    if (v.heads == 0U || v.heads > kHeads || v.heads % kAttnMinBlockHeads != 0U) throw std::invalid_argument("ds41 attention heads: multiple of 8, at most 64");
    // the most heads per wave that still gives >= 40 blocks (one per two CUs), else 1
    const auto blocks = [&](unsigned hpw) { return kMaxChunks * (v.heads / (kAttnMinBlockHeads * hpw)) * v.rows; };
    const auto fits = [&](unsigned hpw) { return v.heads % (kAttnMinBlockHeads * hpw) == 0U && blocks(hpw) >= 40U; };
    const unsigned hpw = fits(4U) ? 4U : fits(2U) ? 2U : 1U;
    const dim3 grid(kMaxChunks, v.heads / (kAttnMinBlockHeads * hpw), v.rows);
    if (hpw == 4U) hipLaunchKernelGGL(attention_chunk_kernel<4>, grid, dim3(kAttnThreads), 0, stream, v);
    else if (hpw == 2U) hipLaunchKernelGGL(attention_chunk_kernel<2>, grid, dim3(kAttnThreads), 0, stream, v);
    else hipLaunchKernelGGL(attention_chunk_kernel<1>, grid, dim3(kAttnThreads), 0, stream, v);
    if (v.tickets == nullptr) hipLaunchKernelGGL(attention_combine_kernel, dim3(v.heads, v.rows), dim3(32), 0, stream, v);
    check(hipGetLastError(), "ds41 attention");
}

}  // namespace strix::ds41::attn::hip
