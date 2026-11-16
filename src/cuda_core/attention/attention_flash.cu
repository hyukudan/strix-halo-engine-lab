// Prefill attention (FAST class): causal flash attention, head_dim 256, GQA, BF16 KV cache, tensor cores (mma.sync m16n8k16 bf16, FP32
// accumulate). See attention.hpp attention_prefill_flash for the contract (NOT bitwise equal to attention_decode).
//
// CTA = 128 query rows x 1 query head, 8 warps of 16 query rows. Key tiles of 32 keys through a 2-stage cp.async pipeline. Q lives in
// registers as bf16 A fragments (the head-dimension k permutation: k step ks, quad lane q holds d = 16 ks + 4 q .. +3, the same
// permutation on the K side, so a thread reads its 4 K values with one LDS.64). S = Q K^T; online softmax in FP32 (exp2); P (bf16) feeds
// O += P V with V B-fragments from ldmatrix.trans. K rows are padded to 544 B (conflict-free LDS.64), V rows to 528 B (conflict-free ldmatrix).
#include "strix/cuda_core/attention.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

constexpr int QB = 128, KT = 32, D = 256;
constexpr int K_STRIDE = 544, V_STRIDE = 528;  // bytes
constexpr int K_TILE_BYTES = KT * K_STRIDE, V_TILE_BYTES = KT * V_STRIDE;
constexpr int STAGE_BYTES = K_TILE_BYTES + V_TILE_BYTES;
constexpr int SMEM_BYTES = 2 * STAGE_BYTES;

__device__ __forceinline__ void cp16(std::uint32_t smem_addr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(g));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
    std::uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void ldsm_x4_trans(std::uint32_t (&r)[4], std::uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void load_kv_tile(std::uint32_t smem_stage, const __nv_bfloat16* kbase, const __nv_bfloat16* vbase, std::uint32_t key0,
                                             std::uint32_t nkeys_total) {
    // 32 keys x 512 B for K and V: 1024 16-byte chunks each over 256 threads
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t idx = threadIdx.x + 256 * i, r = idx >> 5, ch = idx & 31;
        const std::uint32_t key = min(key0 + r, nkeys_total - 1);
        cp16(smem_stage + r * K_STRIDE + ch * 16, kbase + static_cast<std::size_t>(key) * D + ch * 8);
        cp16(smem_stage + K_TILE_BYTES + r * V_STRIDE + ch * 16, vbase + static_cast<std::size_t>(key) * D + ch * 8);
    }
}

__global__ void __launch_bounds__(256, 1)
    flash_kernel(AttentionDims d, const float* q, GroupedView qv, const float* gate, GroupedView gv, const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                 std::uint32_t pos0, std::uint32_t rows, float* out) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t qb = gridDim.x - 1 - blockIdx.x;  // heavy (late) blocks first
    const std::uint32_t head = blockIdx.y, G = d.heads / d.kv_heads, kvh = head / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const std::uint32_t row_base = qb * QB, wrow0 = row_base + warp * 16;
    const std::uint32_t nkeys = pos0 + min(rows, row_base + QB);  // keys any row of this block may see
    const std::uint32_t ntiles = (nkeys + KT - 1) / KT;
    const __nv_bfloat16* kbase = kc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const __nv_bfloat16* vbase = vc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const std::uint32_t smem_base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));

    // prefetch tile 0
    load_kv_tile(smem_base, kbase, vbase, 0, nkeys);
    asm volatile("cp.async.commit_group;");

    // Q fragments (rows g and g + 8 of the warp's 16)
    std::uint32_t qa[16][4];
    {
        const std::uint32_t r0 = wrow0 + g, r1 = wrow0 + g + 8;
        const float* p0 = q + static_cast<std::size_t>(min(r0, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
        const float* p1 = q + static_cast<std::size_t>(min(r1, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const float4 a = *reinterpret_cast<const float4*>(p0 + 16 * ks), b = *reinterpret_cast<const float4*>(p1 + 16 * ks);
            qa[ks][0] = pack2(a.x, a.y);
            qa[ks][1] = pack2(b.x, b.y);
            qa[ks][2] = pack2(a.z, a.w);
            qa[ks][3] = pack2(b.z, b.w);
        }
    }
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float sl2 = d.scale * 1.4426950408889634f;
    const std::uint32_t qpos0 = pos0 + wrow0 + g, qpos1 = qpos0 + 8;

    for (std::uint32_t t = 0; t < ntiles; ++t) {
        const std::uint32_t stage = (t & 1) * STAGE_BYTES;
        if (t + 1 < ntiles) {
            load_kv_tile(smem_base + ((t + 1) & 1) * STAGE_BYTES, kbase, vbase, (t + 1) * KT, nkeys);
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        const unsigned char* sK = smem + stage;
        const std::uint32_t sV = smem_base + stage + K_TILE_BYTES;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(sK + (nt * 8 + g) * K_STRIDE + (16 * ks + 4 * q4) * 2);
                mma_bf16(s[nt], qa[ks], b.x, b.y);
            }
        }
        const std::uint32_t key0 = t * KT;
        // scale (log2 domain) and causal mask
        const bool need_mask = key0 + KT - 1 > pos0 + wrow0;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = s[nt][j] * sl2;
                if (need_mask) {
                    const std::uint32_t kp = key0 + nt * 8 + 2 * q4 + (j & 1), qp = j < 2 ? qpos0 : qpos1;
                    if (kp > qp) v = -INFINITY;
                }
                s[nt][j] = v;
            }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float n0s = n0 == -INFINITY ? 0.0f : n0, n1s = n1 == -INFINITY ? 0.0f : n1;
        const float al0 = exp2f(m0 - n0s), al1 = exp2f(m1 - n1s);  // m = -inf -> 0
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= al0; o[i][1] *= al0; o[i][2] *= al1; o[i][3] *= al1;
        }
        std::uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p00 = exp2f(s[nt][0] - n0s), p01 = exp2f(s[nt][1] - n0s), p10 = exp2f(s[nt][2] - n1s), p11 = exp2f(s[nt][3] - n1s);
            l0 += p00 + p01;
            l1 += p10 + p11;
            const std::uint32_t lo = pack2(p00, p01), hi = pack2(p10, p11);
            pa[nt >> 1][(nt & 1) * 2 + 0] = lo;
            pa[nt >> 1][(nt & 1) * 2 + 1] = hi;
        }
        // O += P V : a0 = (row g, keys 2q..) of n-tile 2j, a1 = (row g+8, n-tile 2j), a2 = (row g, n-tile 2j+1), a3 = (g+8, 2j+1)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t a[4] = {pa[j][0], pa[j][1], pa[j][2], pa[j][3]};
#pragma unroll
            for (int np = 0; np < 16; ++np) {
                std::uint32_t b[4];
                const std::uint32_t key = 16 * j + ((lane >> 3) & 1) * 8 + (lane & 7), dcol = 8 * (2 * np + (lane >> 4));
                ldsm_x4_trans(b, sV + key * V_STRIDE + dcol * 2);
                mma_bf16(o[2 * np], a, b[0], b[1]);
                mma_bf16(o[2 * np + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    // finalize
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    const std::uint32_t r0 = wrow0 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const std::uint32_t dd = 8 * nt + 2 * q4;
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t r = h ? r1 : r0;
            if (r >= rows) continue;
            float v0 = o[nt][2 * h] * (h ? i1 : i0), v1 = o[nt][2 * h + 1] * (h ? i1 : i0);
            if (gate != nullptr) {
                const float2 gt = *reinterpret_cast<const float2*>(gate + static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(head) * gv.gs() + dd);
                v0 *= cc_sigmoid(gt.x);
                v1 *= cc_sigmoid(gt.y);
            }
            *reinterpret_cast<float2*>(out + (static_cast<std::size_t>(r) * d.heads + head) * D + dd) = make_float2(v0, v1);
        }
    }
}

__global__ void iota_kernel(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = base + step * i;
}

}  // namespace

cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream) {
    if (d.head_dim != D || rows == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8 || pos0 + rows > d.max_positions)
        return cudaErrorInvalidValue;
    // Attributes belong to the current device; CUDA serializes the update.
    if (auto e = cudaFuncSetAttribute(flash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); e != cudaSuccess) return e;
    flash_kernel<<<dim3((rows + QB - 1) / QB, d.heads), 256, SMEM_BYTES, stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out);
    return cudaGetLastError();
}

cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream) {
    iota_kernel<<<(n + 127) / 128, 128, 0, stream>>>(p, n, base, step);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
