#include "attn_decode.cuh"

#include "attn_core/masks.hpp"
#include "attn_core/relative_bias.hpp"
#include "core/row_transforms.hpp"
#include "cuda_core/common.cuh"
#include "ptx.cuh"
#include "strix/cuda_core/attention_score_reduce.hpp"

namespace strix::cuda::core::attn {
namespace {

namespace legacy = ::strix::cuda_core;
constexpr unsigned kWarps = kSplitKeys / 32U;

// Only the generic kernels' device operands are passed to the fallback grid.
struct FallbackInput {
    const float* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* relative_bias;
    const float* sinks;
    const float* gate;
    std::uint32_t first_position;
    const std::uint32_t* device_position;
};

struct Job {
    DecodePlan plan;
    FallbackInput in;
    float* out;
    float* scratch;

    __device__ unsigned first() const {
        return in.device_position ? *in.device_position : in.first_position;
    }
    __device__ std::size_t row_head(unsigned row, unsigned head) const {
        return std::size_t{row} * plan.spec.heads + head;
    }
    __device__ const float* query(unsigned row, unsigned head) const {
        const unsigned stride = plan.spec.head_dim *
            (plan.spec.gate == OutputGate::sigmoid_interleaved ? 2U : 1U);
        return in.q + row_head(row, head) * stride;
    }
    __device__ std::size_t kv_offset(unsigned head, unsigned key, unsigned dim = 0) const {
        const auto& spec = plan.spec;
        if (spec.layout == KvLayout::head_major)
            return (std::size_t{head} * spec.cache_slots + key) * spec.head_dim + dim;
        const unsigned slot = spec.layout == KvLayout::ring ? key % spec.cache_slots : key;
        return (std::size_t{slot} * spec.kv_heads + head) * spec.head_dim + dim;
    }
    __device__ void store(unsigned row, unsigned head, unsigned dim, float value) const {
        const auto offset = row_head(row, head) * plan.spec.head_dim + dim;
        if (plan.spec.gate == OutputGate::sigmoid_bf16) {
            ::strix::core::row_transforms::Bf16SigmoidProduct{&value, in.gate + offset, out + offset}(0);
            return;
        }
        if (plan.spec.gate == OutputGate::sigmoid_interleaved)
            value = __fmul_rn(value, legacy::cc_sigmoid(query(row, head)[plan.spec.head_dim + dim]));
        out[offset] = value;
    }
    __device__ float* scores(std::size_t rh) const { return scratch + rh * plan.spec.max_context; }
    __device__ float* maxima(std::size_t rh) const { return scratch + plan.score_floats + rh * plan.splits; }
    __device__ float* denoms(std::size_t rh) const { return maxima(rh) + plan.stat_floats; }
    __device__ float* partial(std::size_t rh, unsigned split) const {
        return scratch + plan.score_floats + 2 * plan.stat_floats +
            (rh * plan.splits + split) * plan.spec.head_dim;
    }
};

// FP32 split fallback for shapes outside the native D256 head-major ABI. A CTA
// shares a K/V chunk among the heads of one group. The transposed butterfly
// preserves the order of each dot, and no chunk normalizes independently.
template <int GT, bool Dim256>
__global__ void __launch_bounds__(256, 2) score_splits(Job job) {
    using namespace legacy::score_reduce;
    constexpr int P = score_positions(GT);
    using Map = ScoreMap<GT, P>;
    __shared__ float warp_max[kWarps][GT];
    const auto& spec = job.plan.spec;
    const unsigned row = blockIdx.z, kv_head = blockIdx.y, split = blockIdx.x;
    const unsigned first = job.first();
    if (first > spec.max_context - spec.rows) return;
    const unsigned pos = first + row;
    const unsigned lo = attn_core::mask::window_lo(pos, spec.window);
    const unsigned begin = split * kSplitKeys, end = min(pos + 1U, begin + kSplitKeys);
    if (begin >= end || end <= lo) return;
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    const unsigned head0 = kv_head * job.plan.group;
    const unsigned lane_dims = Dim256 ? 8U : (spec.head_dim + 31U) / 32U;
    const unsigned dim0 = lane * lane_dims;
    const auto rh0 = job.row_head(row, head0);

    float query[GT][Dim256 ? 8 : 1];
    if constexpr (Dim256) {
#pragma unroll
        for (int g = 0; g < GT; ++g) {
            const bool live = static_cast<unsigned>(g) < job.plan.group;
            const float* q = job.query(row, live ? head0 + g : head0) + dim0;
#pragma unroll
            for (int d = 0; d < 8; ++d) query[g][d] = live ? q[d] : 0.0F;
        }
    }
    float mx[Map::C];
#pragma unroll
    for (int i = 0; i < Map::C; ++i) mx[i] = attn_core::mask::excluded_score;
    auto shuffle = [](float x, unsigned off) { return __shfl_xor_sync(0xffffffffU, x, static_cast<int>(off)); };
    auto add = [](float a, float b) { return __fadd_rn(a, b); };

    for (unsigned p0 = begin + warp; p0 < end; p0 += P * kWarps) {
        float dots[Map::N]{};
#pragma unroll
        for (int j = 0; j < P; ++j) {
            const unsigned key = p0 + j * kWarps;
            if (key < lo || key >= end) continue;
            const auto* k = job.in.k + job.kv_offset(kv_head, key);
            if constexpr (Dim256) {
                const uint4 packed = __ldg(reinterpret_cast<const uint4*>(k + dim0));
                const float values[8] = {
                    legacy::bf16_lo(packed.x), legacy::bf16_hi(packed.x),
                    legacy::bf16_lo(packed.y), legacy::bf16_hi(packed.y),
                    legacy::bf16_lo(packed.z), legacy::bf16_hi(packed.z),
                    legacy::bf16_lo(packed.w), legacy::bf16_hi(packed.w)};
#pragma unroll
                for (int d = 0; d < 8; ++d)
#pragma unroll
                    for (int g = 0; g < GT; ++g)
                        dots[j * GT + g] = __fadd_rn(dots[j * GT + g], __fmul_rn(query[g][d], values[d]));
            } else {
                const unsigned dim_end = min(spec.head_dim, dim0 + lane_dims);
                for (unsigned d = dim0; d < dim_end; ++d) {
                    const float value = __bfloat162float(k[d]);
                    for (unsigned g = 0; g < job.plan.group; ++g)
                        dots[j * GT + g] = __fadd_rn(dots[j * GT + g],
                            __fmul_rn(job.query(row, head0 + g)[d], value));
                }
            }
        }
        reduce<Map::N>(dots, lane, shuffle, add);
        const unsigned key = p0 + Map::position(lane) * kWarps;
        if (key >= lo && key < end) {
#pragma unroll
            for (int i = 0; i < Map::C; ++i) {
                const unsigned g = Map::head(lane, i);
                if (g < job.plan.group) {
                    const float score = __fmul_rn(dots[i], spec.scale);
                    mx[i] = fmaxf(mx[i], score);
                    if ((lane & replica_mask(Map::N)) == 0) job.scores(rh0 + g)[key] = score;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Map::C; ++i) {
        if constexpr (P == 2) mx[i] = fmaxf(mx[i], shuffle(mx[i], 16));
        const unsigned g = Map::head(lane, i);
        if (Map::position(lane) == 0 && (lane & replica_mask(Map::N)) == 0)
            warp_max[warp][g] = mx[i];
    }
    __syncthreads();
    if (threadIdx.x < job.plan.group) {
        const unsigned g = threadIdx.x;
        float mx_all = warp_max[0][g];
#pragma unroll
        for (unsigned w = 1; w < kWarps; ++w) mx_all = fmaxf(mx_all, warp_max[w][g]);
        job.maxima(rh0 + g)[split] = mx_all;
    }
}

template <unsigned Dim>
__global__ void __launch_bounds__(256) value_splits(Job job) {
    __shared__ float maxima[kMaxGroup], warp_sum[kMaxGroup][kWarps];
    __shared__ float weights[kSplitKeys][kMaxGroup];
    const auto& spec = job.plan.spec;
    const unsigned row = blockIdx.z, kv_head = blockIdx.y, split = blockIdx.x;
    const unsigned first = job.first();
    if (first > spec.max_context - spec.rows) return;
    const unsigned pos = first + row;
    const unsigned lo = attn_core::mask::window_lo(pos, spec.window);
    const unsigned begin = split * kSplitKeys, end = min(pos + 1U, begin + kSplitKeys);
    if (begin >= end || end <= lo) return;
    const unsigned tid = threadIdx.x, lane = tid & 31U, warp = tid >> 5;
    const auto rh = job.row_head(row, kv_head * job.plan.group);
    const unsigned dim = Dim == 0 ? spec.head_dim : Dim;
    if (warp < job.plan.group) {
        float mx = attn_core::mask::excluded_score;
        for (unsigned c = lo / kSplitKeys + lane; c <= pos / kSplitKeys; c += 32U)
            mx = fmaxf(mx, job.maxima(rh + warp)[c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, off));
        if (lane == 0) maxima[warp] = mx;
    }
    __syncthreads();
    const unsigned key = begin + tid;
    for (unsigned g = 0; g < job.plan.group; ++g) {
        const float e = key >= lo && key < end ?
            legacy::cc_expf(__fsub_rn(job.scores(rh + g)[key], maxima[g])) : 0.0F;
        weights[tid][g] = e;
        const float sum = legacy::warp_sum_xor(e);
        if (lane == 0) warp_sum[g][warp] = sum;
    }
    __syncthreads();
    if (tid < job.plan.group) {
        float sum = warp_sum[tid][0];
#pragma unroll
        for (unsigned w = 1; w < kWarps; ++w) sum = __fadd_rn(sum, warp_sum[tid][w]);
        job.denoms(rh + tid)[split] = sum;
    }
    for (unsigned d = tid; d < dim; d += kSplitKeys) {
        float acc[kMaxGroup]{};
        for (unsigned p = max(lo, begin); p < end; ++p) {
            const float value = __bfloat162float(job.in.v[job.kv_offset(kv_head, p, d)]);
            for (unsigned g = 0; g < job.plan.group; ++g)
                acc[g] = __fadd_rn(acc[g], __fmul_rn(weights[p - begin][g], value));
        }
        for (unsigned g = 0; g < job.plan.group; ++g) job.partial(rh + g, split)[d] = acc[g];
    }
}

__global__ void __launch_bounds__(256) merge_splits(Job job) {
    const auto& spec = job.plan.spec;
    const unsigned first = job.first();
    if (first > spec.max_context - spec.rows) return;
    const unsigned head = blockIdx.x, row = blockIdx.y, pos = first + row;
    const unsigned begin = attn_core::mask::window_lo(pos, spec.window) / kSplitKeys;
    const auto rh = job.row_head(row, head);
    for (unsigned d = threadIdx.x; d < spec.head_dim; d += kSplitKeys) {
        float denom = 0.0F, acc = 0.0F;
        for (unsigned c = begin; c <= pos / kSplitKeys; ++c) {
            denom = __fadd_rn(denom, job.denoms(rh)[c]);
            acc = __fadd_rn(acc, job.partial(rh, c)[d]);
        }
        job.store(row, head, d, __fdiv_rn(acc, denom));
    }
}

template <bool Dim256>
void launch_scores(Job job, dim3 grid, cudaStream_t stream) {
    if (job.plan.group <= 4) score_splits<4, Dim256><<<grid, kSplitKeys, 0, stream>>>(job);
    else if (job.plan.group <= 6) score_splits<6, Dim256><<<grid, kSplitKeys, 0, stream>>>(job);
    else score_splits<8, Dim256><<<grid, kSplitKeys, 0, stream>>>(job);
}

// Prepared BF16 fallback for dimensions and views outside the native flash ABI.
// One warp owns a query/head; other lanes of the MMA row tile are masked.
// The 32-key absolute tile, paired denominator adds, MMA order and PV
// rounding match the incumbent Pad2 path. Splitting the walk would change P.
template <unsigned Dim>
struct FlashTile {
    static constexpr unsigned kStride = 2U * Dim + 32U;
    static constexpr unsigned vStride = 2U * Dim + 16U;
    static constexpr unsigned vOffset = kFlashKeys * kStride;
    static constexpr unsigned bytes = vOffset + kFlashKeys * vStride;
    static constexpr unsigned stages = Dim <= 256 ? 2U : 1U;

    __device__ static void load(unsigned char* dst, Job job, unsigned kv_head,
                                unsigned base, unsigned first_key, unsigned last_key) {
        constexpr unsigned pieces = Dim / 8U;
        for (unsigned i = threadIdx.x; i < kFlashKeys * pieces; i += blockDim.x) {
            const unsigned r = i / pieces, c = i % pieces;
            const unsigned key = min(max(base + r, first_key), last_key);
            const auto offset = job.kv_offset(kv_head, key, c * 8U);
            ptx_mem::cp_async16(dst + r * kStride + c * 16U, job.in.k + offset);
            ptx_mem::cp_async16(dst + vOffset + r * vStride + c * 16U, job.in.v + offset);
        }
        ptx_mem::cp_async_commit();
    }
};

__device__ __forceinline__ unsigned pack2(float lo, float hi) {
    unsigned bits;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(hi), "f"(lo));
    return bits;
}

template <unsigned Dim>
__global__ void __launch_bounds__(128) online_flash(Job job) {
    using Tile = FlashTile<Dim>;
    extern __shared__ __align__(16) unsigned char smem[];
    const auto& spec = job.plan.spec;
    const unsigned first = job.first();
    if (first > spec.max_context - spec.rows) return;
    const unsigned head = blockIdx.x, row = blockIdx.y, kv_head = head / job.plan.group;
    const unsigned pos = first + row, last_key = first + spec.rows - 1U;
    const unsigned first_key = attn_core::mask::window_lo(first, spec.window);
    const unsigned lo = attn_core::mask::window_lo(pos, spec.window);
    const unsigned tile_start = first_key / kFlashKeys * kFlashKeys;
    const unsigned tiles = (last_key - tile_start) / kFlashKeys + 1U;
    constexpr float log2e = 1.4426950408889634F;
    constexpr float excluded = attn_core::mask::excluded_score;
    const float scale = spec.scale * log2e;
    const attn_core::RelativeBiasRows bias{job.in.relative_bias, spec.heads, spec.relative_extent};
    const unsigned lane = threadIdx.x & 31U, g = lane >> 2, q4 = lane & 3U;
    const bool active = threadIdx.x < 32U;
    // Retain the fixed MMA row and lane indices from the reference. Only
    // low row zero is live, so the two high-row probabilities stay zero.
    float m = excluded, denom = 0.0F;
    unsigned q[Dim / 16U][2];
    float acc[Dim / 8U][2]{};
    if (active) {
        if (g == 0 && spec.sinks) {
            m = job.in.sinks[job.row_head(row, head)] * log2e;
            denom = q4 == 0 ? 1.0F : 0.0F;
        }
        const float* query = job.query(row, head) + 4U * q4;
#pragma unroll
        for (unsigned k = 0; k < Dim / 16U; ++k) {
            const float* qk = query + 16U * k;
            q[k][0] = pack2(qk[0], qk[1]);
            q[k][1] = pack2(qk[2], qk[3]);
        }
    }
    if constexpr (Tile::stages == 2) Tile::load(smem, job, kv_head, tile_start, first_key, last_key);

    for (unsigned tile = 0; tile < tiles; ++tile) {
        const unsigned base = tile_start + tile * kFlashKeys;
        auto* stage = smem + (tile % Tile::stages) * Tile::bytes;
        if constexpr (Tile::stages == 2) {
            if (tile + 1U < tiles) {
                Tile::load(smem + ((tile + 1U) % Tile::stages) * Tile::bytes, job, kv_head,
                           base + kFlashKeys, first_key, last_key);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                ptx_mem::cp_async_wait_all();
            }
        } else {
            Tile::load(stage, job, kv_head, base, first_key, last_key);
            ptx_mem::cp_async_wait_all();
        }
        __syncthreads();
        if (active) {
            float score[4][4]{};
#pragma unroll
            for (unsigned k = 0; k < Dim / 16U; ++k) {
#pragma unroll
                for (unsigned nt = 0; nt < 4U; ++nt) {
                    const uint2 b = *reinterpret_cast<const uint2*>(
                        stage + (8U * nt + g) * Tile::kStride + 32U * k + 8U * q4);
                    ptx::mma_bf16(score[nt], q[k][0], q[k][0], q[k][1], q[k][1], b.x, b.y);
                }
            }
            float mx = excluded;
#pragma unroll
            for (unsigned nt = 0; nt < 4U; ++nt) {
#pragma unroll
                for (unsigned j = 0; j < 2U; ++j) {
                    const unsigned key = base + 8U * nt + 2U * q4 + j;
                    if (g == 0 && attn_core::mask::causal_ok(key, pos, lo)) {
                        float v = score[nt][j] * scale;
                        v += bias.score_bias(row, head, pos, key) * log2e;
                        score[nt][j] = v;
                    } else {
                        score[nt][j] = excluded;
                    }
                }
                mx = fmaxf(mx, fmaxf(score[nt][0], score[nt][1]));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, 2));
            const float next = fmaxf(m, mx);
            const float safe_next = next == excluded ? 0.0F : next;
            const float alpha = exp2f(m - safe_next);
            m = next;
            denom *= alpha;
#pragma unroll
            for (unsigned i = 0; i < Dim / 8U; ++i) {
                acc[i][0] *= alpha;
                acc[i][1] *= alpha;
            }
            unsigned probability[4];
#pragma unroll
            for (unsigned nt = 0; nt < 4U; ++nt) {
                const float pa = exp2f(score[nt][0] - safe_next), pb = exp2f(score[nt][1] - safe_next);
                denom += pa + pb;
                probability[nt] = pack2(pa, pb);
            }
#pragma unroll
            for (unsigned j = 0; j < 2U; ++j) {
                const unsigned key = 16U * j + ((lane >> 3) & 1U) * 8U + (lane & 7U);
#pragma unroll
                for (unsigned d = 0; d < Dim / 16U; ++d) {
                    unsigned b[4];
                    const unsigned col = 8U * (2U * d + (lane >> 4));
                    ptx::ldsm_x4_trans(b, stage + Tile::vOffset + key * Tile::vStride + 2U * col);
#pragma unroll
                    for (unsigned half = 0; half < 2U; ++half) {
                        auto& low = acc[2U * d + half];
                        float tile_acc[4]{low[0], low[1], 0.0F, 0.0F};
                        ptx::mma_bf16(tile_acc, probability[2U * j], 0, probability[2U * j + 1U], 0,
                                      b[2U * half], b[2U * half + 1U]);
                        low[0] = tile_acc[0];
                        low[1] = tile_acc[1];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        denom += __shfl_xor_sync(0xffffffffU, denom, 1);
        denom += __shfl_xor_sync(0xffffffffU, denom, 2);
        if (g == 0) {
            const float inv = 1.0F / denom;
#pragma unroll
            for (unsigned i = 0; i < Dim / 8U; ++i) {
                const unsigned d = 8U * i + 2U * q4;
                job.store(row, head, d, acc[i][0] * inv);
                job.store(row, head, d + 1U, acc[i][1] * inv);
            }
        }
    }
}

template <unsigned Dim>
cudaError_t launch(Job job, cudaStream_t stream) {
    const auto& spec = job.plan.spec;
    if constexpr (Dim != 0) {
        if (spec.order == DecodeOrder::online_bf16) {
            constexpr unsigned bytes = FlashTile<Dim>::stages * FlashTile<Dim>::bytes;
            if constexpr (bytes > 48U * 1024U) {
                if (auto status = cudaFuncSetAttribute(online_flash<Dim>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); status != cudaSuccess) return status;
            }
            online_flash<Dim><<<dim3(spec.heads, spec.rows), 128, bytes, stream>>>(job);
            return cudaGetLastError();
        }
    }
    const unsigned splits = job.in.device_position ? job.plan.splits :
        (job.in.first_position + spec.rows + kSplitKeys - 1U) / kSplitKeys;
    const dim3 grid(splits, spec.kv_heads, spec.rows);
    launch_scores<Dim == 256>(job, grid, stream);
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    value_splits<Dim><<<grid, kSplitKeys, 0, stream>>>(job);
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    merge_splits<<<dim3(spec.heads, spec.rows), kSplitKeys, 0, stream>>>(job);
    return cudaGetLastError();
}

}  // namespace

cudaError_t decode(const DecodePlan& plan, const DecodeInput& in, float* out,
                   float* scratch, std::size_t scratch_floats, cudaStream_t stream) {
    const auto& spec = plan.spec;
    const auto aligned = [](const void* ptr) { return (reinterpret_cast<std::uintptr_t>(ptr) & 15U) == 0; };
    const bool mapped = in.row_map.pos || in.row_map.slot || in.row_map.self;
    const bool strided = in.q_view.n || in.gate_view.n;
    const bool separate_gate = spec.gate == OutputGate::sigmoid_bf16 ||
                               spec.gate == OutputGate::sigmoid_strided;
    if (!in.q || !in.k || !in.v || !out || in.q == out ||
        (separate_gate && (!in.gate || in.gate == out)) ||
        (spec.sinks && !in.sinks) || (spec.relative_extent != 0 && !in.relative_bias) ||
        (plan.scratch_floats != 0 && (!scratch || scratch_floats < plan.scratch_floats)) ||
        (!in.row_map.pos && !in.device_position && in.first_position > spec.max_context - spec.rows) ||
        ((spec.order == DecodeOrder::online_bf16 || spec.head_dim == 256) &&
            (!aligned(in.k) || !aligned(in.v)))) return cudaErrorInvalidValue;

    const cuda_core::AttentionDims dims{spec.heads, spec.kv_heads, spec.head_dim, spec.scale, spec.cache_slots};
    const detail::PlanPosition position{in.device_position, in.first_position, spec.max_context - spec.rows};
    if (uses_native_split(spec)) {
        const bool interleaved = spec.gate == OutputGate::sigmoid_interleaved;
        const auto stride = spec.head_dim * (interleaved ? 2U : 1U);
        const cuda_core::GroupedView packed{spec.rows, spec.heads, spec.head_dim};
        const cuda_core::GroupedView q_view = in.q_view.n ? in.q_view :
            cuda_core::GroupedView{spec.rows, spec.heads, spec.head_dim, spec.heads * stride, stride};
        const cuda_core::GroupedView gate_view = interleaved ? q_view : (in.gate_view.n ? in.gate_view : packed);
        const auto valid_view = [&](const cuda_core::GroupedView& view) {
            return view.n == spec.head_dim && view.rows >= spec.rows && view.groups == spec.heads;
        };
        if (!valid_view(q_view) || (separate_gate && !valid_view(gate_view))) return cudaErrorInvalidValue;
        const float* gate = interleaved ? in.q + spec.head_dim : (separate_gate ? in.gate : nullptr);
        if (spec.order == DecodeOrder::online_e4m3) {
            static_assert(kE4m3SplitKeys == cuda_core::kAttn8Chunk);
            if (!in.shadow || !in.shadow->k8 || !in.shadow->ks || !in.shadow->v8t || !in.shadow->vs ||
                (!in.device_position && !in.row_map.pos) || !aligned(in.q) ||
                q_view.rs() % 4U != 0 || q_view.gs() % 4U != 0) return cudaErrorInvalidValue;
            return cuda_core::attention_decode_kv(dims, in.q, q_view, gate, gate_view, in.k, in.v, in.shadow,
                                                   in.device_position, spec.rows, out, scratch, stream, in.row_map);
        }
        if (in.shadow) return cudaErrorInvalidValue;
        return detail::decode_split(dims, in.q, q_view, gate, gate_view, in.k, in.v, position, spec.rows,
                                    out, scratch, stream, in.row_map);
    }
    if (mapped || strided || in.shadow) return cudaErrorInvalidValue;
    if (spec.order == DecodeOrder::online_bf16 && spec.head_dim == 128) {
        const cuda_core::PreparedBf16Flash prepared{
            in.relative_bias, spec.sinks ? in.sinks : nullptr, spec.relative_extent, spec.window,
            spec.layout == KvLayout::ring};
        const auto first = in.device_position ? 0U : in.first_position;
        if (cuda_core::can_launch_prepared_bf16_flash(dims, prepared, in.q, in.k, in.v, first,
                                                       spec.rows, out, cuda_core::FlashPipe::pad2)) {
            return detail::decode_prepared(dims, prepared, in.q, in.k, in.v, position,
                                           spec.rows, separate_gate ? in.gate : nullptr, out, stream);
        }
    }
    const Job job{plan, {in.q, in.k, in.v, in.relative_bias, in.sinks, in.gate,
                         in.first_position, in.device_position}, out, scratch};
    switch (plan.dim_class) {
        case 64: return launch<64>(job, stream);
        case 96: return launch<96>(job, stream);
        case 128: return launch<128>(job, stream);
        case 192: return launch<192>(job, stream);
        case 256: return launch<256>(job, stream);
        case 512: return launch<512>(job, stream);
        default: return launch<0>(job, stream);
    }
}

}  // namespace strix::cuda::core::attn
