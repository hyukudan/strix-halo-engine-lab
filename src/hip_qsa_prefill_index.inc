// Indexer q/k input projections of a prefill group (included by hip_indexed_attention_launcher.hip; needs
// bf16_to_float, round_bf16, QsaRowsInputViews and qsa_rows_matrix).
//
// linear_to_float_rows_kernel computes every (row, output) of a BF16 matrix as 256 chains: chain t is the
// ascending contracted chain over features t, t + 256, t + 512, ... starting from 0, and the chains meet in the
// 256-leaf LDS tree (stride 128 down to 1). Its blocks run one output for 16 rows and every block reads the 16
// rows again (two-byte loads, ~200 MB of L2 traffic per 64 rows).
//
// rows_kernel computes the same chains and the same tree. A block stages RB rows once in LDS as FP32
// ([feature][row], BF16 widening is exact); each wave owns OW outputs and lane l owns chains l + 32j (j < 8) of
// every (output, row) pair, so the tree's strides 128, 64 and 32 are in-lane adds of the same leaves and 16..1 are
// lane l + d moves (permlanex16, DPP row_shl) into lane l: no LDS tree and one barrier per block.
namespace qsa_index {
template <unsigned D> __device__ __forceinline__ float lane_down(float value) {
    if constexpr (D == 16U) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    } else {
        const auto bits = __float_as_int(value);
        return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x100 + static_cast<int>(D), 0xf, 0xf, false));
    }
}
// grid (outputs / (W * OW), Rows / RB), block W * 32, dynamic LDS width * RB floats; width % 256 == 0 and
// 16-byte aligned rows (checked by the caller).
template <unsigned Rows, unsigned RB, unsigned OW, unsigned W>
__global__ __launch_bounds__(W * 32U) void rows_kernel(QsaRowsInputViews<Rows> views, std::uint32_t width) {
    static_assert(Rows % RB == 0U && (RB == 4U || RB == 8U));
    extern __shared__ __align__(16) float act[];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U, r0 = blockIdx.y * RB;
    const unsigned outputs = views.first[views.count];
    // Stage the block's rows: unit u = (row u % RB, features 8 (u / RB) .. + 7).
    const unsigned units = RB * width / 8U;
    for (unsigned u = tid; u < units; u += W * 32U) {
        const unsigned r = u % RB, f = u / RB * 8U;
        const uint4 x = *reinterpret_cast<const uint4*>(views.input[r0 + r] + f);
        const unsigned words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            act[(f + 2U * j) * RB + r] = __uint_as_float(words[j] << 16U);
            act[(f + 2U * j + 1U) * RB + r] = __uint_as_float(words[j] & 0xffff0000U);
        }
    }
    const std::uint16_t* weight[OW];
    unsigned g[OW];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o) {
        g[o] = blockIdx.x * (W * OW) + wave * OW + o;
        std::uint32_t row = 0;
        const auto m = qsa_rows_matrix(views, min(g[o], outputs - 1U), row);
        weight[o] = static_cast<const std::uint16_t*>(views.weights[m]) + static_cast<std::uint64_t>(row) * width + lane;
    }
    float sum[OW][RB][8];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RB; ++r)
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) sum[o][r][j] = 0.0F;
    __syncthreads();
    const unsigned steps = width / 256U;
    for (unsigned k = 0; k < steps; ++k) {
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned f = k * 256U + 32U * j + lane;
            float w[OW];
#pragma unroll
            for (unsigned o = 0; o < OW; ++o) w[o] = bf16_to_float(weight[o][k * 256U + 32U * j]);
            float a[RB];
#pragma unroll
            for (unsigned q = 0; q < RB / 4U; ++q) {
                const float4 x = *reinterpret_cast<const float4*>(&act[f * RB + 4U * q]);
                a[4U * q] = x.x; a[4U * q + 1U] = x.y; a[4U * q + 2U] = x.z; a[4U * q + 3U] = x.w;
            }
#pragma unroll
            for (unsigned o = 0; o < OW; ++o)
#pragma unroll
                for (unsigned r = 0; r < RB; ++r) sum[o][r][j] += a[r] * w[o];
        }
    }
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RB; ++r) {
            float (&v)[8] = sum[o][r];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) v[j] = v[j] + v[j + 4U];
            v[0] = v[0] + v[2]; v[1] = v[1] + v[3];
            float x = v[0] + v[1];
            x = x + lane_down<16U>(x);
            x = x + lane_down<8U>(x);
            x = x + lane_down<4U>(x);
            x = x + lane_down<2U>(x);
            x = x + lane_down<1U>(x);
            if (lane == 0U && g[o] < outputs) {
                std::uint32_t row = 0;
                const auto m = qsa_rows_matrix(views, g[o], row);
                views.output[m][r0 + r][row] = round_bf16(x);
            }
        }
}
// rows_body: a verbatim copy of rows_kernel's body (r0 = the workgroup's first row of views) for rows_batch_kernel;
// rows_kernel itself is left untouched (instruction-identical).
template <unsigned Rows, unsigned RB, unsigned OW, unsigned W>
__device__ __forceinline__ void rows_body(const QsaRowsInputViews<Rows>& views, std::uint32_t width, unsigned r0) {
    static_assert(Rows % RB == 0U && (RB == 4U || RB == 8U));
    extern __shared__ __align__(16) float act[];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    const unsigned outputs = views.first[views.count];
    // Stage the block's rows: unit u = (row u % RB, features 8 (u / RB) .. + 7).
    const unsigned units = RB * width / 8U;
    for (unsigned u = tid; u < units; u += W * 32U) {
        const unsigned r = u % RB, f = u / RB * 8U;
        const uint4 x = *reinterpret_cast<const uint4*>(views.input[r0 + r] + f);
        const unsigned words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            act[(f + 2U * j) * RB + r] = __uint_as_float(words[j] << 16U);
            act[(f + 2U * j + 1U) * RB + r] = __uint_as_float(words[j] & 0xffff0000U);
        }
    }
    const std::uint16_t* weight[OW];
    unsigned g[OW];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o) {
        g[o] = blockIdx.x * (W * OW) + wave * OW + o;
        std::uint32_t row = 0;
        const auto m = qsa_rows_matrix(views, min(g[o], outputs - 1U), row);
        weight[o] = static_cast<const std::uint16_t*>(views.weights[m]) + static_cast<std::uint64_t>(row) * width + lane;
    }
    float sum[OW][RB][8];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RB; ++r)
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) sum[o][r][j] = 0.0F;
    __syncthreads();
    const unsigned steps = width / 256U;
    for (unsigned k = 0; k < steps; ++k) {
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned f = k * 256U + 32U * j + lane;
            float w[OW];
#pragma unroll
            for (unsigned o = 0; o < OW; ++o) w[o] = bf16_to_float(weight[o][k * 256U + 32U * j]);
            float a[RB];
#pragma unroll
            for (unsigned q = 0; q < RB / 4U; ++q) {
                const float4 x = *reinterpret_cast<const float4*>(&act[f * RB + 4U * q]);
                a[4U * q] = x.x; a[4U * q + 1U] = x.y; a[4U * q + 2U] = x.z; a[4U * q + 3U] = x.w;
            }
#pragma unroll
            for (unsigned o = 0; o < OW; ++o)
#pragma unroll
                for (unsigned r = 0; r < RB; ++r) sum[o][r][j] += a[r] * w[o];
        }
    }
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RB; ++r) {
            float (&v)[8] = sum[o][r];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) v[j] = v[j] + v[j + 4U];
            v[0] = v[0] + v[2]; v[1] = v[1] + v[3];
            float x = v[0] + v[1];
            x = x + lane_down<16U>(x);
            x = x + lane_down<8U>(x);
            x = x + lane_down<4U>(x);
            x = x + lane_down<2U>(x);
            x = x + lane_down<1U>(x);
            if (lane == 0U && g[o] < outputs) {
                std::uint32_t row = 0;
                const auto m = qsa_rows_matrix(views, g[o], row);
                views.output[m][r0 + r][row] = round_bf16(x);
            }
        }
}
// STRIX_QPF2K_QSA bit 7 (0x80): every Rows-row group of one prefill call in one launch. Workgroup row block y runs
// rows_body over group y / (Rows / RB) (views read from the device table instead of the kernel arguments), rows
// (y % (Rows / RB)) * RB ..: each output is the one rows_kernel computes for that group (same body, operands,
// chains and tree); grid (ceil(outputs / (W * OW)), groups * Rows / RB), every group with the same matrices.
template <unsigned Rows, unsigned RB, unsigned OW, unsigned W>
__global__ __launch_bounds__(W * 32U) void rows_batch_kernel(const QsaRowsInputViews<Rows>* groups, std::uint32_t width) {
    constexpr unsigned per = Rows / RB;
    rows_body<Rows, RB, OW, W>(groups[blockIdx.y / per], width, (blockIdx.y % per) * RB);
}
} // namespace qsa_index

// q / k / index-query norm + partial RoPE of a prefill group in one launch (norm_rope_rows_kernel runs one launch
// per tensor). Every value is norm_rope_body's (no status path): the same LDS norm tree, inv_rms, weight and
// BF16 roundings and the same rotation expression; only the angle's cosine and sine, which depend on the row's
// position and the pair alone, come from rope_table_kernel (the same expressions, evaluated once per (row, pair)
// instead of once per (row, head, feature)).
namespace qsa_index {
template <unsigned Rows> struct NormGroupViews {
    float* values[3][Rows];
    const void* weights[3];
    std::uint32_t heads[3], dim[3], stride[3];
    std::uint64_t position[Rows];
    float2* table; // [Rows][rotary_dim / 2] (cosine, sine)
};
template <unsigned Rows, bool Batch = false>
__global__ __launch_bounds__(64) void rope_table_kernel(NormGroupViews<Rows> single, std::uint32_t rotary_dim, float theta, const NormGroupViews<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const NormGroupViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.y];
        else return single;
    }();
    const auto row = static_cast<std::uint32_t>(blockIdx.x), pair = static_cast<std::uint32_t>(threadIdx.x);
    const auto half = rotary_dim / 2U;
    if (pair >= half) return;
    const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
    const float angle = static_cast<float>(views.position[row]) * powf(theta, -exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    views.table[static_cast<std::uint64_t>(row) * half + pair] = make_float2(cosine, sine);
}
// grid (heads[0] + heads[1] + heads[2], Rows), block 256 (dim <= 256).
template <SmallWeightKind Kind, unsigned Rows, bool Batch = false>
__global__ __launch_bounds__(256) void norm_rope_group_kernel(NormGroupViews<Rows> single, float epsilon, std::uint32_t rotary_dim, const NormGroupViews<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const NormGroupViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.z];
        else return single;
    }();
    if constexpr (Batch) if (blockIdx.x >= views.heads[0] + views.heads[1] + views.heads[2]) return;
    __shared__ float source[256], reduction[256];
    const auto row = static_cast<std::uint32_t>(blockIdx.y), feature = static_cast<std::uint32_t>(threadIdx.x);
    auto head = static_cast<std::uint32_t>(blockIdx.x);
    unsigned s = 0;
    while (s < 2U && head >= views.heads[s]) head -= views.heads[s++];
    const auto head_dim = views.dim[s];
    const bool active = feature < head_dim;
    float* values = views.values[s][row];
    const auto index = static_cast<std::uint64_t>(head) * views.stride[s] + feature;
    const float input = active ? values[index] : 0.0F;
    if (active) { source[feature] = input; reduction[feature] = input * input; }
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) reduction[feature] += reduction[feature + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / static_cast<float>(head_dim) + epsilon;
    const float inv_rms = rsqrtf(mean);
    if (active) source[feature] = round_bf16(input * inv_rms * load_small_weight<Kind>(views.weights[s], feature));
    __syncthreads();
    if (!active) return;
    float result = source[feature];
    if (feature < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = feature % half;
        const float2 cs = views.table[static_cast<std::uint64_t>(row) * half + pair];
        const float cosine = cs.x;
        const float sine = cs.y;
        if (feature < half) result = source[feature] * cosine - source[feature + half] * sine;
        else result = source[feature] * cosine + source[feature - half] * sine;
    }
    values[index] = round_bf16(result);
}
} // namespace qsa_index

// Indexer block scores of a prefill group with the per-block epilogue moved out of the block loop. In
// block_score_pooled_rows_kernel every (row, block) step broadcasts the four head sums to all lanes, takes the
// relu sum and the division in every lane and keeps lane i's; here lanes 0/8/16/24 (which hold the head sums
// after the reduce-scatter) park them in LDS, and after the loop lane i forms its block's relu sum, in head order
// from 0, and the division once: the same operands and operations.
namespace qsa_index {
template <unsigned Rows, unsigned RowsPerWave>
__global__ __launch_bounds__(256) void block_score_parked_kernel(QsaRowsScoreViews<Rows> views, const float* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim) {
#pragma clang fp contract(off)
    constexpr unsigned Heads = 4U, Dim = 128U;
    __shared__ float park[8][RowsPerWave][32][Heads];
    const auto lane = threadIdx.x % 32U, w = threadIdx.x / 32U;
    const auto wave = blockIdx.x * 8U + w;
    const auto chunks = (scored_blocks + 31U) / 32U;
    const auto r0 = wave / chunks * RowsPerWave, base = wave % chunks * 32U;
    if (r0 >= Rows) return;
    float q[RowsPerWave][Heads][4];
    std::uint32_t blocks[RowsPerWave];
#pragma unroll
    for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
        blocks[rr] = views.blocks[r0 + rr];
#pragma unroll
        for (unsigned h = 0; h < Heads; ++h)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) q[rr][h][j] = views.index_query[r0 + rr][h * Dim + lane + 32U * j];
    }
    const auto last = min(32U, scored_blocks - base);
    const bool writer = (lane & 7U) == 0U;
    const unsigned head = lane / 8U;
    for (unsigned i = 0; i < last; ++i) {
        const auto* key = pooled_keys + static_cast<std::uint64_t>(base + i) * Dim + lane;
        const float k0 = key[0], k1 = key[32], k2 = key[64], k3 = key[96];
#pragma unroll
        for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
            float partial[Heads];
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) {
                const float p0 = q[rr][h][0] * k0, p1 = q[rr][h][1] * k1, p2 = q[rr][h][2] * k2, p3 = q[rr][h][3] * k3;
                partial[h] = (p0 + p2) + (p1 + p3);
            }
            qsa_scatter<Heads, 0U>(partial, lane);
            if (writer) park[w][rr][i][head] = partial[0];
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    if (lane < last) {
#pragma unroll
        for (unsigned rr = 0; rr < RowsPerWave; ++rr) {
            float total = 0.0F;
#pragma unroll
            for (unsigned h = 0; h < Heads; ++h) total += fmaxf(park[w][rr][lane][h], 0.0F);
            if (base + lane < blocks[rr]) views.block_scores[r0 + rr][base + lane] = total / sqrtf(static_cast<float>(head_dim));
        }
    }
}
} // namespace qsa_index
