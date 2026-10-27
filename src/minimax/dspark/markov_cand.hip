// DSpark Markov chain over a candidate set (contract: markov_cand.hpp). Kernels qualified bitwise against CPU replicas in
// the selection benchmark (selection fixtures, canonical top-C sets, pick ids and score bits).
#include "markov_cand.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::minimax::dspark {
namespace {

constexpr std::uint32_t kFallback = 0xffffffffU;
struct Cand {
    float value;
    std::uint32_t id;
};

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("markov cand ") + what + ": " + hipGetErrorString(e));
}

__host__ __device__ inline float bf16f(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
__host__ __device__ inline bool better(float v, std::uint32_t i, float b, std::uint32_t bi) { return v > b || (v == b && i < bi); }
__device__ inline float wave_sum(float v) {
    for (int d = 16; d != 0; d >>= 1) v = v + __shfl_xor(v, d, 32);
    return v;
}
// = markov_kernel's per-candidate arithmetic (hip_minimax_dspark.hip)
__device__ __forceinline__ float lane_dot(const std::uint16_t* w1row, const std::uint16_t* w2row, unsigned lane) {
    const uint4 mw = *reinterpret_cast<const uint4*>(w1row + 8U * lane);
    const uint4 ww = *reinterpret_cast<const uint4*>(w2row + 8U * lane);
    const std::uint32_t ma[4] = {mw.x, mw.y, mw.z, mw.w}, wa[4] = {ww.x, ww.y, ww.z, ww.w};
    float acc = 0.0F;
    for (std::uint32_t e = 0; e < 8U; ++e) {
        const float m = bf16f(static_cast<std::uint16_t>(ma[e >> 1U] >> (16U * (e & 1U))));
        const float w = bf16f(static_cast<std::uint16_t>(wa[e >> 1U] >> (16U * (e & 1U))));
        acc = acc + m * w;
    }
    return wave_sum(acc);
}

struct Payload {  // ids [7][C] u32 | counts [7] u32 | scores [7][C] f32 | hval [7][hmax] f32
    std::uint32_t* ids;
    std::uint32_t* counts;
    float* scores;
    float* hval;
};
__host__ __device__ inline Payload view(unsigned char* p, unsigned C) {
    Payload v;
    v.ids = reinterpret_cast<std::uint32_t*>(p);
    v.counts = v.ids + std::size_t{CandMarkov::kRows} * C;
    v.scores = reinterpret_cast<float*>(v.counts + CandMarkov::kRows);
    v.hval = v.scores + std::size_t{CandMarkov::kRows} * C;
    return v;
}

__device__ __forceinline__ std::uint32_t okey(float f) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, f);
    if (u == 0x80000000U) u = 0U;  // -0 == +0
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}
// Exactly C entries of row blockIdx.x of logits [rows][n] in canonical order; ids reported as id_off + local index.
__global__ __launch_bounds__(1024) void topc_kernel(const float* logits, std::uint32_t n, std::uint32_t C, std::uint32_t id_off,
                                                    unsigned char* payload) {
    __shared__ std::uint32_t hist[256];
    __shared__ std::uint32_t s_prefix, s_need, s_n, s_bad, s_wave[32], s_taken;
    const Payload pv = view(payload, C);
    const unsigned row = blockIdx.x;
    const float* b = logits + static_cast<std::size_t>(row) * n;
    std::uint32_t* out = pv.ids + static_cast<std::size_t>(row) * C;
    if (threadIdx.x == 0) s_bad = 0;
    __syncthreads();
    for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x)
        if (!__builtin_isfinite(b[v])) atomicOr(&s_bad, 1U);
    __syncthreads();
    if (s_bad || C == 0 || C > n) {
        if (threadIdx.x == 0) pv.counts[row] = kFallback;
        return;
    }
    std::uint32_t prefix = 0, mask = 0, need = C;
    for (int pass = 3; pass >= 0; --pass) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned shift = 8U * static_cast<unsigned>(pass);
        for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x) {
            const std::uint32_t k = okey(b[v]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t acc = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            s_prefix = prefix | (static_cast<std::uint32_t>(d) << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask |= 255U << shift;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        s_n = 0;
        s_taken = 0;
    }
    __syncthreads();
    for (std::uint32_t v = threadIdx.x; v < n; v += blockDim.x)
        if (okey(b[v]) > prefix) out[atomicAdd(&s_n, 1U)] = id_off + v;
    __syncthreads();
    const std::uint32_t above = s_n;
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    for (std::uint32_t t0 = 0; t0 < n; t0 += blockDim.x) {  // ties: the lowest ids, ascending tiles
        if (s_taken >= need) break;
        const std::uint32_t v = t0 + threadIdx.x;
        const bool eq = v < n && okey(b[v]) == prefix;
        const std::uint32_t m = static_cast<std::uint32_t>(__ballot(eq));
        const std::uint32_t before = static_cast<std::uint32_t>(__builtin_popcount(m & ((1U << lane) - 1U)));
        if (lane == 0) s_wave[wave] = static_cast<std::uint32_t>(__builtin_popcount(m));
        __syncthreads();
        std::uint32_t off = 0;
        for (unsigned w = 0; w < wave; ++w) off += s_wave[w];
        const std::uint32_t rank = s_taken + off + before;
        if (eq && rank < need) out[above + rank] = id_off + v;
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t tot = 0;
            for (unsigned w = 0; w < blockDim.x / 32U; ++w) tot += s_wave[w];
            s_taken += tot;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) pv.counts[row] = C;
}

// ---- topc v2: the same canonical set over many CTAs per row (STRIX_MM_MARKOV_TOPC=v2) ----
// Key k = okey(x) (order-preserving, -0 -> +0). Row threshold T (full key) and tie cut: the selected set is
// {v : k(v) > T} u {v : k(v) == T, v <= cut}, exactly C entries, i.e. the C largest keys, ties by lower id (= topc_kernel).
//   hist    (nb x 7 CTAs): nonfinite flag + 2048-bin histogram of k >> 21 (global atomics, nonzero bins only)
//   compact (nb x 7 CTAs): boundary bin d1 (C-th largest key's top 11 bits); per chunk the count of keys with bin > d1
//            (all selected) and the (low 21 bits, id) of the keys with bin == d1 appended to the row's candidate buffer
//   refine  (7 CTAs):      exact radix select over the candidates only: low 21 bits (11 + 10), then, if not all keys == T
//            are taken, the need-th lowest id among them (22-bit ids, 11 + 11); per-chunk output offsets; counts; resets
//            the hist / flag / candidate counter for the next draft
//   emit    (nb x 7 CTAs): ids + scores of the selected keys in ascending id order (deterministic) + the owned-H values
constexpr std::uint32_t kBins = 2048, kCh = 4096, kV2Threads = 256, kPer = kCh / kV2Threads, kIdBits = 22, kLowMask = 0x1fffffU;
constexpr std::uint32_t kMaxChunks = (1U << kIdBits) / kCh;

struct Ws {  // hist [7][2048] | bad [7] | ccount [7] | thr [7] | cut [7] | gt [7][nb] | off [7][nb] | (8-aligned) cbuf [7][n]
    std::uint32_t *hist, *bad, *ccount, *thr, *cut, *gt, *off;
    uint2* cbuf;
};
__host__ __device__ inline std::size_t ws_words(std::uint32_t nb) {
    const std::size_t w = std::size_t{CandMarkov::kRows} * (kBins + 4U + 2U * nb);
    return (w + 1U) & ~std::size_t{1};
}
__host__ __device__ inline std::size_t ws_bytes(std::uint32_t nb, std::uint32_t n) {
    return ws_words(nb) * 4U + std::size_t{CandMarkov::kRows} * n * sizeof(uint2);
}
__host__ __device__ inline Ws ws_view(unsigned char* p, std::uint32_t nb) {
    Ws w;
    w.hist = reinterpret_cast<std::uint32_t*>(p);
    w.bad = w.hist + std::size_t{CandMarkov::kRows} * kBins;
    w.ccount = w.bad + CandMarkov::kRows;
    w.thr = w.ccount + CandMarkov::kRows;
    w.cut = w.thr + CandMarkov::kRows;
    w.gt = w.cut + CandMarkov::kRows;
    w.off = w.gt + std::size_t{CandMarkov::kRows} * nb;
    w.cbuf = reinterpret_cast<uint2*>(p + ws_words(nb) * 4U);
    return w;
}
// Exclusive prefix (thread order) of x over the block; total -> tot. Block size a multiple of 32; s_wave >= 32 words.
__device__ std::uint32_t block_scan(std::uint32_t x, std::uint32_t* s_wave, std::uint32_t& tot) {
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, nw = blockDim.x >> 5U;
    std::uint32_t inc = x;
    for (std::uint32_t d = 1; d < 32U; d <<= 1U) {
        const std::uint32_t y = __shfl_up(inc, static_cast<unsigned>(d), 32);
        if (lane >= d) inc += y;
    }
    if (lane == 31U) s_wave[wave] = inc;
    __syncthreads();
    std::uint32_t off = 0, t = 0;
    for (std::uint32_t w = 0; w < nw; ++w) {
        if (w < wave) off += s_wave[w];
        t += s_wave[w];
    }
    __syncthreads();
    tot = t;
    return off + inc - x;
}
// Over hist[0, nbins) (all threads read it): the digit d with sum(hist[d+1..]) < need <= sum(hist[d..]) (need >= 1, total
// >= need) -> s_out[0] = d, s_out[1] = sum(hist[d+1..]). Ends with a barrier.
__device__ void block_find(const std::uint32_t* hist, std::uint32_t nbins, std::uint32_t need, std::uint32_t* s_wave,
                           std::uint32_t* s_out) {
    const std::uint32_t t = threadIdx.x, lane = t & 31U, wave = t >> 5U, nw = blockDim.x >> 5U;
    const std::uint32_t per = (nbins + blockDim.x - 1U) / blockDim.x;
    std::uint32_t s = 0;
    for (std::uint32_t j = 0; j < per; ++j)
        if (t * per + j < nbins) s += hist[t * per + j];
    std::uint32_t suf = s;  // inclusive suffix sum within the wave (lanes >= this one)
    for (std::uint32_t d = 1; d < 32U; d <<= 1U) {
        const std::uint32_t y = __shfl_down(suf, static_cast<unsigned>(d), 32);
        if (lane + d < 32U) suf += y;
    }
    if (lane == 0) s_wave[wave] = suf;
    __syncthreads();
    std::uint32_t above = suf - s;  // keys in the bins after this thread's
    for (std::uint32_t w = wave + 1U; w < nw; ++w) above += s_wave[w];
    if (above < need && above + s >= need) {
        std::uint32_t acc = above, i = t * per + per - 1U;
        for (;; --i) {
            if (i < nbins) {
                if (acc + hist[i] >= need) break;
                acc += hist[i];
            }
        }
        s_out[0] = i;
        s_out[1] = acc;
    }
    __syncthreads();
}
__global__ __launch_bounds__(256) void v2_hist_kernel(const float* logits, std::uint32_t n, unsigned char* ws, std::uint32_t nb) {
    __shared__ std::uint32_t h[kBins];
    __shared__ std::uint32_t s_bad;
    const Ws w = ws_view(ws, nb);
    const std::uint32_t row = blockIdx.y, c0 = blockIdx.x * kCh, c1 = c0 + kCh < n ? c0 + kCh : n;
    const float* b = logits + static_cast<std::size_t>(row) * n;
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x) h[i] = 0;
    if (threadIdx.x == 0) s_bad = 0;
    __syncthreads();
    bool bad = false;
    for (std::uint32_t v = c0 + threadIdx.x; v < c1; v += blockDim.x) {
        const float x = b[v];
        bad = bad || !__builtin_isfinite(x);
        atomicAdd(&h[okey(x) >> 21U], 1U);
    }
    if (bad) atomicOr(&s_bad, 1U);
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x)
        if (h[i] != 0) atomicAdd(&w.hist[std::size_t{row} * kBins + i], h[i]);
    if (threadIdx.x == 0 && s_bad != 0) atomicOr(&w.bad[row], 1U);
}
__global__ __launch_bounds__(256) void v2_compact_kernel(const float* logits, std::uint32_t n, std::uint32_t C, unsigned char* ws,
                                                         std::uint32_t nb) {
    __shared__ std::uint32_t s_wave[32], s_out[2], s_base;
    const Ws w = ws_view(ws, nb);
    const std::uint32_t row = blockIdx.y, chunk = blockIdx.x;
    if (w.bad[row] != 0 || C == 0 || C > n) return;  // refine reports the fallback
    block_find(w.hist + std::size_t{row} * kBins, kBins, C, s_wave, s_out);
    const std::uint32_t d1 = s_out[0];
    const float* b = logits + static_cast<std::size_t>(row) * n;
    const std::uint32_t v0 = chunk * kCh + threadIdx.x * kPer;
    std::uint32_t key[kPer];
    std::uint32_t eq = 0, gt = 0;
    for (std::uint32_t j = 0; j < kPer; ++j) {
        key[j] = v0 + j < n ? okey(b[v0 + j]) : 0U;
        const std::uint32_t bin = key[j] >> 21U;
        gt += (v0 + j < n && bin > d1) ? 1U : 0U;
        eq += (v0 + j < n && bin == d1) ? 1U : 0U;
    }
    std::uint32_t eq_tot = 0, gt_tot = 0;
    std::uint32_t pos = block_scan(eq, s_wave, eq_tot);
    (void)block_scan(gt, s_wave, gt_tot);
    if (threadIdx.x == 0) {
        s_base = eq_tot != 0 ? atomicAdd(&w.ccount[row], eq_tot) : 0U;
        w.gt[std::size_t{row} * nb + chunk] = gt_tot;
    }
    __syncthreads();
    pos += s_base;
    uint2* cb = w.cbuf + static_cast<std::size_t>(row) * n;
    for (std::uint32_t j = 0; j < kPer; ++j)
        if (v0 + j < n && (key[j] >> 21U) == d1) cb[pos++] = make_uint2(key[j] & kLowMask, v0 + j);
}
// Radix select among the candidates (ties == false: key = low 21 bits; ties == true: only low == tlow, key = the id
// reversed so larger = lower id): the key t with #(key > t) < need <= #(key >= t); need -> need - #(key > t); eq = #(key == t).
__device__ std::uint32_t cta_select(const uint2* cb, std::uint32_t m, bool ties, std::uint32_t tlow, std::uint32_t nbits,
                                    std::uint32_t& need, std::uint32_t& eq, std::uint32_t* h, std::uint32_t* s_wave,
                                    std::uint32_t* s_out) {
    std::uint32_t prefix = 0, mask = 0, shift = nbits;
    while (shift != 0) {
        const std::uint32_t wd = shift < 11U ? shift : 11U, digits = 1U << wd;
        shift -= wd;
        for (std::uint32_t i = threadIdx.x; i < digits; i += blockDim.x) h[i] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
            const uint2 c = cb[i];
            if (ties && c.x != tlow) continue;
            const std::uint32_t k = ties ? ((1U << kIdBits) - 1U) - c.y : c.x;
            if ((k & mask) == prefix) atomicAdd(&h[(k >> shift) & (digits - 1U)], 1U);
        }
        __syncthreads();
        block_find(h, digits, need, s_wave, s_out);
        const std::uint32_t d = s_out[0];
        eq = h[d];
        need -= s_out[1];
        prefix |= d << shift;
        mask |= (digits - 1U) << shift;
        __syncthreads();
    }
    return prefix;
}
__global__ __launch_bounds__(1024) void v2_refine_kernel(std::uint32_t n, std::uint32_t C, unsigned char* ws, std::uint32_t nb,
                                                         unsigned char* payload) {
    __shared__ std::uint32_t h[kBins], cnt[kMaxChunks], s_wave[32], s_out[2];
    const Ws w = ws_view(ws, nb);
    const Payload pv = view(payload, C);
    const std::uint32_t row = blockIdx.x;
    const bool fallback = w.bad[row] != 0 || C == 0 || C > n;
    if (!fallback) {
        block_find(w.hist + std::size_t{row} * kBins, kBins, C, s_wave, s_out);
        const std::uint32_t d1 = s_out[0];
        std::uint32_t need = C - s_out[1], eq = 0;
        const std::uint32_t m = w.ccount[row];
        const uint2* cb = w.cbuf + static_cast<std::size_t>(row) * n;
        const std::uint32_t tlow = cta_select(cb, m, false, 0, 21U, need, eq, h, s_wave, s_out);
        std::uint32_t cut = 0xffffffffU;  // all keys == T taken
        if (need < eq) {
            std::uint32_t e2 = 0;
            cut = ((1U << kIdBits) - 1U) - cta_select(cb, m, true, tlow, kIdBits, need, e2, h, s_wave, s_out);
        }
        for (std::uint32_t i = threadIdx.x; i < nb; i += blockDim.x) cnt[i] = 0;
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
            const uint2 c = cb[i];
            if (c.x > tlow || (c.x == tlow && c.y <= cut)) atomicAdd(&cnt[c.y / kCh], 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            std::uint32_t acc = 0;
            for (std::uint32_t i = 0; i < nb; ++i) {
                w.off[std::size_t{row} * nb + i] = acc;
                acc += w.gt[std::size_t{row} * nb + i] + cnt[i];
            }
            w.thr[row] = (d1 << 21U) | tlow;
            w.cut[row] = cut;
            pv.counts[row] = acc == C ? C : kFallback;  // acc == C by construction
        }
    } else if (threadIdx.x == 0) {
        pv.counts[row] = kFallback;
    }
    __syncthreads();  // every read of hist / bad / ccount is done: reset for the next draft
    for (std::uint32_t i = threadIdx.x; i < kBins; i += blockDim.x) w.hist[std::size_t{row} * kBins + i] = 0;
    if (threadIdx.x == 0) {
        w.bad[row] = 0;
        w.ccount[row] = 0;
    }
}
__global__ __launch_bounds__(256) void v2_emit_kernel(const float* logits, std::uint32_t n, std::uint32_t C, std::uint32_t id_off,
                                                      const std::uint32_t* hown, std::uint32_t nh_own, std::uint32_t hmax,
                                                      unsigned char* ws, std::uint32_t nb, unsigned char* payload) {
    __shared__ std::uint32_t s_wave[32];
    const Ws w = ws_view(ws, nb);
    const Payload pv = view(payload, C);
    const std::uint32_t row = blockIdx.y, chunk = blockIdx.x;
    const float* b = logits + static_cast<std::size_t>(row) * n;
    for (std::uint32_t i = chunk * blockDim.x + threadIdx.x; i < hmax; i += nb * blockDim.x)  // = pack_kernel's hval
        pv.hval[static_cast<std::size_t>(row) * hmax + i] = i < nh_own ? b[hown[i] - id_off] : 0.0F;
    if (pv.counts[row] != C) return;
    const std::uint32_t T = w.thr[row], cut = w.cut[row], v0 = chunk * kCh + threadIdx.x * kPer;
    float x[kPer];
    std::uint32_t sel = 0, cnt = 0;
    for (std::uint32_t j = 0; j < kPer; ++j) {
        const std::uint32_t v = v0 + j;
        x[j] = v < n ? b[v] : 0.0F;
        const std::uint32_t k = okey(x[j]);
        if (v < n && (k > T || (k == T && v <= cut))) {
            sel |= 1U << j;
            ++cnt;
        }
    }
    std::uint32_t tot = 0;
    std::uint32_t pos = w.off[std::size_t{row} * nb + chunk] + block_scan(cnt, s_wave, tot);
    for (std::uint32_t j = 0; j < kPer; ++j)
        if ((sel >> j) & 1U) {
            pv.ids[static_cast<std::size_t>(row) * C + pos] = id_off + v0 + j;
            pv.scores[static_cast<std::size_t>(row) * C + pos] = x[j];
            ++pos;
        }
}
// scores of the selected ids and the values of the owned H members.
__global__ void pack_kernel(const float* logits, std::uint32_t n, std::uint32_t C, std::uint32_t id_off, const std::uint32_t* hown,
                            std::uint32_t nh_own, std::uint32_t hmax, unsigned char* payload) {
    const Payload pv = view(payload, C);
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
    const float* row = logits + static_cast<std::size_t>(k) * n;
    if (i < C && pv.counts[k] == C) pv.scores[static_cast<std::size_t>(k) * C + i] = row[pv.ids[static_cast<std::size_t>(k) * C + i] - id_off];
    if (i < hmax) pv.hval[static_cast<std::size_t>(k) * hmax + i] = i < nh_own ? row[hown[i] - id_off] : 0.0F;
}
// One chain step of row k over [own top-C | peer top-C | own H | peer H]; one wave per candidate, 8 per block.
__global__ void cand_kernel(const std::uint16_t* w1, const std::uint16_t* w2, const std::uint32_t* prev, unsigned k,
                            const unsigned char* own, const unsigned char* peer, unsigned C, const std::uint32_t* hown, unsigned nh_own,
                            const std::uint32_t* hpeer, unsigned nh_peer, unsigned hmax, Cand* cand) {
    __shared__ float best_v[8];
    __shared__ std::uint32_t best_i[8];
    const Payload a = view(const_cast<unsigned char*>(own), C), b = view(const_cast<unsigned char*>(peer), C);
    const std::uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U, q = blockIdx.x * 8U + wave;
    Cand c{-__builtin_huge_valf(), 0xffffffffU};
    if (q < 2 * C + nh_own + nh_peer) {
        std::uint32_t v;
        float bv;
        if (q < C) {
            v = a.ids[std::size_t{k} * C + q];
            bv = a.scores[std::size_t{k} * C + q];
        } else if (q < 2 * C) {
            v = b.ids[std::size_t{k} * C + q - C];
            bv = b.scores[std::size_t{k} * C + q - C];
        } else if (q < 2 * C + nh_own) {
            v = hown[q - 2 * C];
            bv = a.hval[std::size_t{k} * hmax + q - 2 * C];
        } else {
            v = hpeer[q - 2 * C - nh_own];
            bv = b.hval[std::size_t{k} * hmax + q - 2 * C - nh_own];
        }
        const float acc = lane_dot(w1 + static_cast<std::size_t>(*prev) * CandMarkov::kRank, w2 + static_cast<std::size_t>(v) * CandMarkov::kRank, lane);
        c = {bv + acc, v};
    }
    if (lane == 0) {
        best_v[wave] = c.value;
        best_i[wave] = c.id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        Cand bst{best_v[0], best_i[0]};
        for (int j = 1; j < 8; ++j)
            if (better(best_v[j], best_i[j], bst.value, bst.id)) bst = {best_v[j], best_i[j]};
        cand[blockIdx.x] = bst;
    }
}
// = pick_kernel (hip_minimax_dspark.hip), writing the next prev on device.
__global__ void pick_kernel(const Cand* cand, std::uint32_t n, Cand* out, std::uint32_t* prev_out) {
    __shared__ float rv[256];
    __shared__ std::uint32_t ri[256];
    Cand b{-__builtin_huge_valf(), 0xffffffffU};
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
        *out = Cand{rv[0], ri[0]};
        *prev_out = ri[0];
    }
}
// Sum of squares of each W2 row (one wave per row, lane chains of 8 then the butterfly): ranking only.
__global__ void norm_kernel(const std::uint16_t* w2, std::uint32_t rows, float* out) {
    const std::uint32_t lane = threadIdx.x & 31U, v = blockIdx.x * 8U + (threadIdx.x >> 5U);
    if (v >= rows) return;
    float acc = 0.0F;
    for (unsigned e = 0; e < 8; ++e) {
        const float x = bf16f(w2[static_cast<std::size_t>(v) * CandMarkov::kRank + 8U * lane + e]);
        acc = acc + x * x;
    }
    acc = wave_sum(acc);
    if (lane == 0) out[v] = acc;
}

}  // namespace

CandMarkov::CandMarkov(const std::uint16_t* w1, const std::uint16_t* w2, std::uint32_t vocab_size, std::uint32_t v0, std::uint32_t vn,
                       unsigned C, unsigned nh)
    : w1_(w1), w2_(w2), v0_(v0), vn_(vn), C_(C), nh_(nh) {
    if (C == 0 || C > vn || C > vocab_size - vn || nh == 0 || nh > vocab_size) throw std::invalid_argument("markov cand: bad C / |H|");
    if (vn > (1U << kIdBits)) throw std::invalid_argument("markov cand: vn > 2^22 (topc v2 id width)");
    const char* e = std::getenv("STRIX_MM_MARKOV_TOPC");
    const char* b = std::getenv("STRIX_MM_BASELINE");
    const bool base = b != nullptr && *b != '\0' && std::strcmp(b, "0") != 0;  // as mm_defaults::baseline()
    const std::string tm = e && *e ? e : (base ? "v1" : "v2");  // default v2 (exact, t12); apps export it via mm_defaults
    if (tm != "v1" && tm != "v2") throw std::invalid_argument("STRIX_MM_MARKOV_TOPC must be v1 or v2");
    topc_ = tm == "v2" ? TopC::v2 : TopC::v1;
    // H: identical on both ranks (same kernel, same replicated W2, host sort by (sum desc, id asc)).
    float* dn = nullptr;
    ok(hipMalloc(&dn, std::size_t{vocab_size} * 4), "norm alloc");
    hipLaunchKernelGGL(norm_kernel, dim3((vocab_size + 7) / 8), dim3(256), 0, nullptr, w2, vocab_size, dn);
    std::vector<float> nrm(vocab_size);
    ok(hipMemcpy(nrm.data(), dn, std::size_t{vocab_size} * 4, hipMemcpyDeviceToHost), "norm d2h");
    ok(hipFree(dn), "norm free");
    std::vector<std::uint32_t> order(vocab_size);
    for (std::uint32_t v = 0; v < vocab_size; ++v) order[v] = v;
    std::partial_sort(order.begin(), order.begin() + nh, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return nrm[a] > nrm[b] || (nrm[a] == nrm[b] && a < b); });
    std::vector<std::uint32_t> own, peer;
    for (unsigned i = 0; i < nh; ++i) (order[i] >= v0 && order[i] < v0 + vn ? own : peer).push_back(order[i]);
    nh_own_ = static_cast<unsigned>(own.size());
    nh_peer_ = static_cast<unsigned>(peer.size());
    hmax_ = std::max(nh_own_, nh_peer_);
    payload_ = (std::size_t{kRows} * C * 2 + kRows + std::size_t{kRows} * hmax_) * 4;
    ok(hipMalloc(&hid_own_, std::max<std::size_t>(1, own.size()) * 4), "h alloc");
    ok(hipMalloc(&hid_peer_, std::max<std::size_t>(1, peer.size()) * 4), "h alloc");
    if (!own.empty()) ok(hipMemcpy(hid_own_, own.data(), own.size() * 4, hipMemcpyHostToDevice), "h own");
    if (!peer.empty()) ok(hipMemcpy(hid_peer_, peer.data(), peer.size() * 4, hipMemcpyHostToDevice), "h peer");
    ok(hipMalloc(&dpay_own_, payload_), "payload");
    ok(hipMalloc(&dpay_peer_, payload_), "payload");
    ok(hipHostMalloc(&hsend_, payload_), "pinned");
    ok(hipHostMalloc(&hrecv_, payload_), "pinned");
    ok(hipMalloc(&dprev_, 4 * (kRows + 1)), "prev");
    ok(hipMalloc(&dcand_, std::size_t{(2 * C + nh + 7) / 8} * sizeof(Cand)), "cand");
    ok(hipMalloc(&dpick_, kRows * sizeof(Cand)), "pick");
    ok(hipHostMalloc(&hpick_, kRows * sizeof(Cand)), "pick pinned");
    // topc v2 workspace (hist / flags / counters zeroed here, then reset by v2_refine_kernel at every draft)
    nb_ = (vn + kCh - 1) / kCh;
    ok(hipMalloc(&ws_, ws_bytes(nb_, vn)), "topc ws");
    ok(hipMemset(ws_, 0, ws_words(nb_) * 4), "topc ws zero");
    ok(hipDeviceSynchronize(), "topc ws zero sync");
}

void CandMarkov::launch_select(const float* logits, hipStream_t st) {
    if (topc_ == TopC::v1) {
        hipLaunchKernelGGL(topc_kernel, dim3(kRows), dim3(1024), 0, st, logits, vn_, C_, v0_, dpay_own_);
        hipLaunchKernelGGL(pack_kernel, dim3((std::max(C_, hmax_) + 255) / 256, kRows), dim3(256), 0, st, logits, vn_, C_, v0_, hid_own_,
                           nh_own_, hmax_, dpay_own_);
    } else {
        unsigned char* ws = static_cast<unsigned char*>(ws_);
        hipLaunchKernelGGL(v2_hist_kernel, dim3(nb_, kRows), dim3(kV2Threads), 0, st, logits, vn_, ws, nb_);
        hipLaunchKernelGGL(v2_compact_kernel, dim3(nb_, kRows), dim3(kV2Threads), 0, st, logits, vn_, C_, ws, nb_);
        hipLaunchKernelGGL(v2_refine_kernel, dim3(kRows), dim3(1024), 0, st, vn_, C_, ws, nb_, dpay_own_);
        hipLaunchKernelGGL(v2_emit_kernel, dim3(nb_, kRows), dim3(kV2Threads), 0, st, logits, vn_, C_, v0_, hid_own_, nh_own_, hmax_, ws,
                           nb_, dpay_own_);
    }
    ok(hipGetLastError(), "select");
}

void CandMarkov::select_payload(const float* logits, unsigned char* host_out, hipStream_t st) {
    launch_select(logits, st);
    ok(hipMemcpyAsync(host_out, dpay_own_, payload_, hipMemcpyDeviceToHost, st), "payload d2h");
    ok(hipStreamSynchronize(st), "payload sync");
}

CandMarkov::~CandMarkov() {
    (void)hipFree(hid_own_);
    (void)hipFree(hid_peer_);
    (void)hipFree(dpay_own_);
    (void)hipFree(dpay_peer_);
    (void)hipHostFree(hsend_);
    (void)hipHostFree(hrecv_);
    (void)hipFree(dprev_);
    (void)hipFree(dcand_);
    (void)hipFree(dpick_);
    (void)hipHostFree(hpick_);
    (void)hipFree(ws_);
}

bool CandMarkov::draft(const float* logits, std::uint32_t anchor, const Exchange& exchange, std::array<std::uint32_t, kRows>& out,
                       hipStream_t st) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    launch_select(logits, st);
    ok(hipMemcpyAsync(hsend_, dpay_own_, payload_, hipMemcpyDeviceToHost, st), "send d2h");
    ok(hipStreamSynchronize(st), "send sync");
    const auto t1 = clk::now();
    std::size_t nx = 0;
    for (std::size_t off = 0; off < payload_; off += kChunk, ++nx) exchange(hsend_ + off, hrecv_ + off, std::min(kChunk, payload_ - off));
    const auto t2 = clk::now();
    // fallback decision from both ranks' counts (identical on both)
    const Payload mine = view(hsend_, C_), theirs = view(hrecv_, C_);
    for (unsigned k = 0; k < kRows; ++k)
        if (mine.counts[k] != C_ || theirs.counts[k] != C_) return false;
    ok(hipMemcpyAsync(dpay_peer_, hrecv_, payload_, hipMemcpyHostToDevice, st), "recv h2d");
    ok(hipMemcpyAsync(dprev_, &anchor, 4, hipMemcpyHostToDevice, st), "anchor");
    const unsigned nb = (2 * C_ + nh_own_ + nh_peer_ + 7) / 8;
    for (unsigned k = 0; k < kRows; ++k) {
        hipLaunchKernelGGL(cand_kernel, dim3(nb), dim3(256), 0, st, w1_, w2_, dprev_ + k, k, dpay_own_, dpay_peer_, C_, hid_own_,
                           nh_own_, hid_peer_, nh_peer_, hmax_, static_cast<Cand*>(dcand_));
        hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(256), 0, st, static_cast<Cand*>(dcand_), nb, static_cast<Cand*>(dpick_) + k,
                           dprev_ + k + 1);
    }
    ok(hipGetLastError(), "chain");
    ok(hipMemcpyAsync(hpick_, dpick_, kRows * sizeof(Cand), hipMemcpyDeviceToHost, st), "pick d2h");
    ok(hipStreamSynchronize(st), "chain sync");
    const auto t3 = clk::now();
    for (unsigned k = 0; k < kRows; ++k) out[k] = static_cast<const Cand*>(hpick_)[k].id;
    ++stats_.drafts;
    stats_.exchanges += nx;
    stats_.select += std::chrono::duration<double>(t1 - t0).count();
    stats_.exchange += std::chrono::duration<double>(t2 - t1).count();
    stats_.chain += std::chrono::duration<double>(t3 - t2).count();
    return true;
}

bool CandMarkov::draft(const float* logits, std::uint32_t anchor, const DeviceExchange& exchange,
                       std::array<std::uint32_t, kRows>& out, hipStream_t st) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    launch_select(logits, st);
    ok(hipStreamSynchronize(st), "select sync");  // same attribution point as the host form
    const auto t1 = clk::now();
    exchange(dpay_own_, dpay_peer_, payload_, st);  // one device-to-device exchange; the copy into dpay_peer_ is on st
    // fallback decision from both ranks' counts (identical on both): 2 x 7 words to the host
    const std::size_t coff = std::size_t{kRows} * C_ * 4;
    ok(hipMemcpyAsync(hsend_, dpay_own_ + coff, kRows * 4, hipMemcpyDeviceToHost, st), "counts own");
    ok(hipMemcpyAsync(hrecv_, dpay_peer_ + coff, kRows * 4, hipMemcpyDeviceToHost, st), "counts peer");
    ok(hipStreamSynchronize(st), "counts sync");
    const auto t2 = clk::now();
    const auto* mine = reinterpret_cast<const std::uint32_t*>(hsend_);
    const auto* theirs = reinterpret_cast<const std::uint32_t*>(hrecv_);
    for (unsigned k = 0; k < kRows; ++k)
        if (mine[k] != C_ || theirs[k] != C_) return false;
    ok(hipMemcpyAsync(dprev_, &anchor, 4, hipMemcpyHostToDevice, st), "anchor");
    const unsigned nb = (2 * C_ + nh_own_ + nh_peer_ + 7) / 8;
    for (unsigned k = 0; k < kRows; ++k) {
        hipLaunchKernelGGL(cand_kernel, dim3(nb), dim3(256), 0, st, w1_, w2_, dprev_ + k, k, dpay_own_, dpay_peer_, C_, hid_own_,
                           nh_own_, hid_peer_, nh_peer_, hmax_, static_cast<Cand*>(dcand_));
        hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(256), 0, st, static_cast<Cand*>(dcand_), nb, static_cast<Cand*>(dpick_) + k,
                           dprev_ + k + 1);
    }
    ok(hipGetLastError(), "chain");
    ok(hipMemcpyAsync(hpick_, dpick_, kRows * sizeof(Cand), hipMemcpyDeviceToHost, st), "pick d2h");
    ok(hipStreamSynchronize(st), "chain sync");
    const auto t3 = clk::now();
    for (unsigned k = 0; k < kRows; ++k) out[k] = static_cast<const Cand*>(hpick_)[k].id;
    ++stats_.drafts;
    stats_.exchanges += 1;
    stats_.select += std::chrono::duration<double>(t1 - t0).count();
    stats_.exchange += std::chrono::duration<double>(t2 - t1).count();
    stats_.chain += std::chrono::duration<double>(t3 - t2).count();
    return true;
}

}  // namespace strix::minimax::dspark
