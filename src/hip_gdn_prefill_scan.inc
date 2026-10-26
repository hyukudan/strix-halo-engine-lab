// GDN prefill, beta/alpha gates of all rows (included by hip_gdn_launcher.hip after gdn_prefill_gates_kernel).
//
// gdn_prefill_gates_kernel forms every (row, output) as 256 chains (chain t: the contracted chain over features
// t, t + 256, ... from 0) and the 256-leaf tree (strides 128 and 64 in LDS, 32 in the wave, 16..1 shuffles),
// then the sigmoid / softplus transform. Its blocks run 2 outputs for 8 rows and load every activation with a
// two-byte load, so each row is read 48 times per layer and the FP32 weights 1024 times per 8K rows.
//
// gdn_gates_rows_kernel computes the same chains, tree and transform: a set of 256 threads runs O outputs for the
// block's RB rows (thread t owns chain t of every pair), S sets share the rows, staged per 256-feature step in LDS
// as FP32 ([feature][row]); the tree takes the pairs through LDS, C at a time, one wave per pair, lane l holding
// leaves l + 32j (strides 128, 64, 32 in-lane, 16..1 lane moves into lane l), the operands of every addition the same.
namespace gdn_gates {
constexpr unsigned kChains = 256U;
template <unsigned D> __device__ __forceinline__ float lane_down(float value) {
    if constexpr (D == 16U) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    } else {
        const auto bits = __float_as_int(value);
        return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x100 + static_cast<int>(D), 0xf, 0xf, false));
    }
}
template <unsigned RB, unsigned S, unsigned C> constexpr unsigned lds_floats() {
    constexpr unsigned act = kChains * (RB + 4U), tree = S * C * kChains;
    return act > tree ? act : tree;
}
// grid ((2 * heads) / (O * S), count / RB), block S * 256; width % 256 == 0, rows 16-byte aligned (checked by the caller).
template <unsigned RB, unsigned O, unsigned S, unsigned C>
__global__ __launch_bounds__(S * 256U) void gdn_gates_rows_kernel(const std::uint16_t* input, std::uint64_t input_stride,
    const float* beta_weights, const float* alpha_weights, GdnPrefillRows rows, const void* dt, SmallWeightKind dt_kind, const void* a,
    SmallWeightKind a_kind, std::uint32_t width, std::uint32_t heads) {
    static_assert(RB % 4U == 0U && (O * RB) % C == 0U && C % 8U == 0U);
    constexpr unsigned Pitch = RB + 4U, Threads = S * 256U, Units = RB * kChains / 8U;
    constexpr unsigned UnitsPerThread = (Units + Threads - 1U) / Threads;
    __shared__ __align__(16) float lds[lds_floats<RB, S, C>()];
    const unsigned tid = threadIdx.x, set = tid / kChains, t = tid % kChains, lane = t % 32U, wave = t / 32U;
    const unsigned g0 = blockIdx.x * (O * S) + set * O, r0 = blockIdx.y * RB;
    const float* weight[O];
#pragma unroll
    for (unsigned o = 0; o < O; ++o) {
        const unsigned g = g0 + o, out = g >= heads ? g - heads : g;
        weight[o] = (g >= heads ? alpha_weights : beta_weights) + static_cast<std::uint64_t>(out) * width + t;
    }
    float sum[O][RB];
#pragma unroll
    for (unsigned o = 0; o < O; ++o)
#pragma unroll
        for (unsigned r = 0; r < RB; ++r) sum[o][r] = 0.0F;
    uint4 staged[UnitsPerThread];
    const auto fetch = [&](unsigned k) {
#pragma unroll
        for (unsigned i = 0; i < UnitsPerThread; ++i) {
            const unsigned u = tid + i * Threads;
            if (Units % Threads == 0U || u < Units)
                staged[i] = *reinterpret_cast<const uint4*>(input + (r0 + u / 32U) * input_stride + k * kChains + u % 32U * 8U);
        }
    };
    const unsigned steps = width / kChains;
    fetch(0U);
    for (unsigned k = 0; k < steps; ++k) {
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < UnitsPerThread; ++i) {
            const unsigned u = tid + i * Threads;
            if (Units % Threads == 0U || u < Units) {
                const unsigned r = u / 32U, f = u % 32U * 8U;
                const unsigned words[4] = {staged[i].x, staged[i].y, staged[i].z, staged[i].w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) {
                    lds[(f + 2U * j) * Pitch + r] = __uint_as_float(words[j] << 16U);
                    lds[(f + 2U * j + 1U) * Pitch + r] = __uint_as_float(words[j] & 0xffff0000U);
                }
            }
        }
        __syncthreads();
        if (k + 1U < steps) fetch(k + 1U);
        float w[O];
#pragma unroll
        for (unsigned o = 0; o < O; ++o) w[o] = weight[o][k * kChains];
#pragma unroll
        for (unsigned q = 0; q < RB / 4U; ++q) {
            const float4 x4 = *reinterpret_cast<const float4*>(&lds[t * Pitch + 4U * q]);
            const float x[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
            for (unsigned o = 0; o < O; ++o)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) sum[o][4U * q + j] += x[j] * w[o];
        }
    }
    float* tree = lds + set * C * kChains;
#pragma unroll
    for (unsigned c0 = 0; c0 < O * RB; c0 += C) {
        __syncthreads();
#pragma unroll
        for (unsigned c = 0; c < C; ++c) tree[c * kChains + t] = sum[(c0 + c) / RB][(c0 + c) % RB];
        __syncthreads();
        for (unsigned c = wave; c < C; c += 8U) {
            float v[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) v[j] = tree[c * kChains + lane + 32U * j];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) v[j] = gdn_window_add(v[j], v[j + 4U]);
            v[0] = gdn_window_add(v[0], v[2]); v[1] = gdn_window_add(v[1], v[3]);
            float total = gdn_window_add(v[0], v[1]);
            total = gdn_window_add(total, lane_down<16U>(total));
            total = gdn_window_add(total, lane_down<8U>(total));
            total = gdn_window_add(total, lane_down<4U>(total));
            total = gdn_window_add(total, lane_down<2U>(total));
            total = gdn_window_add(total, lane_down<1U>(total));
            const unsigned p = c0 + c, g = g0 + p / RB, row = r0 + p % RB;
            if (lane == 0U) {
                const bool alpha = g >= heads;
                const unsigned out = alpha ? g - heads : g;
                const float raw = round_bf16(total);
                const auto offset = row * rows.stride + out;
                if (alpha) {
                    const float biased = raw + load_small(dt, dt_kind, out);
                    rows.gate[offset] = load_small(a, a_kind, out) * softplus(biased);
                } else rows.beta[offset] = round_bf16(1.0F / (1.0F + expf(-raw)));
            }
        }
    }
}
} // namespace gdn_gates

// Recurrence + gated RMSNorm of a whole value head per block (STRIX_GDN_FUSED_NORM). gdn_prefill_lanes_kernel
// writes every row's core output to global memory and gdn_prefill_norm_rows_kernel reads it back with z; here a
// block owns all 128 columns of one value head (W waves, G lanes per column, the same per-column chains, trees and
// staging), each row's outputs go to the LDS value slots of the chunk being consumed (the slot's value was read
// first), and during the next chunk one wave per row of the previous chunk runs the norm kernel's body on them
// (same tree, rsqrtf, sigmoid and products, same A8 quantizer), the two wave halves taking alternate chunks.
namespace gdn_gates {
template <GdnRecurrentStorage Storage, unsigned G, unsigned W, unsigned Chunk, bool A8>
__global__ __launch_bounds__(W * 32U)
void gdn_prefill_lanes_norm_kernel(GdnPrefillRows rows, std::uint32_t count, float* state, std::uint32_t key_heads,
    std::uint32_t value_heads, const void* norm_weights, SmallWeightKind norm_kind, float epsilon, std::int8_t* a8_codes, float* a8_deltas) {
#pragma clang fp contract(off)
    using Shape = GdnLaneShape<G>;
    constexpr unsigned Threads = W * 32U, Columns = W * Shape::columns, K = Shape::keys;
    static_assert(Columns == 128U && 2U * Chunk <= W);
    constexpr unsigned KeyQuads = Chunk * 64U / Threads, ValueQuads = (Chunk * Columns / 4U + Threads - 1U) / Threads;
    static_assert(Chunk * 64U % Threads == 0U && KeyQuads > 0U && 2U * Chunk <= Threads);
    using Stage = GdnLaneStage<G, Columns, Chunk>;
    __shared__ Stage stage[2];
    const auto thread = static_cast<std::uint32_t>(threadIdx.x), lane = thread % 32U, wave = thread / 32U;
    const auto sub = lane % G, local = thread / 32U * Shape::columns + lane / G;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x);
    const auto column = local;
    const auto key_width = key_heads * 128U, key_offset = value_head % key_heads * 128U;
    const auto value_offset = key_width * 2U + value_head * 128U;
    float4 key_query[KeyQuads], values[ValueQuads];
    float scalar = 0.0F;
    const auto fetch = [&](std::uint32_t first) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            const float* source = rows.qkv + (first + row) * rows.stride + key_offset +
                (within < 32U ? key_width + within * 4U : (within - 32U) * 4U);
            key_query[i] = *reinterpret_cast<const float4*>(source);
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads, row = q / (Columns / 4U);
            if (q < Chunk * Columns / 4U)
                values[i] = *reinterpret_cast<const float4*>(rows.qkv + (first + row) * rows.stride + value_offset + q % (Columns / 4U) * 4U);
        }
        if (thread < 2U * Chunk)
            scalar = (thread < Chunk ? rows.beta : rows.gate)[(first + thread % Chunk) * rows.stride + value_head];
    };
    const auto commit = [&](Stage& s) {
#pragma unroll
        for (unsigned i = 0; i < KeyQuads; ++i) {
            const auto q = thread + i * Threads, row = q / 64U, within = q % 64U;
            const auto key = (within % 32U) * 4U;
            auto& target = within < 32U ? s.key[row] : s.query[row];
            const float parts[4] = {key_query[i].x, key_query[i].y, key_query[i].z, key_query[i].w};
#pragma unroll
            for (unsigned c = 0; c < 4U; ++c) target[(key + c) % G][(key + c) / G] = parts[c];
        }
#pragma unroll
        for (unsigned i = 0; i < ValueQuads; ++i) {
            const auto q = thread + i * Threads;
            if (q < Chunk * Columns / 4U)
                *reinterpret_cast<float4*>(&s.value[q / (Columns / 4U)][q % (Columns / 4U) * 4U]) = values[i];
        }
        if (thread < Chunk) s.beta[thread] = scalar;
        else if (thread < 2U * Chunk) s.decay[thread - Chunk] = expf(scalar);
    };
    // Gated RMSNorm of row first + r from the value slots of s (gdn_prefill_norm_rows_kernel's body).
    const auto norm = [&](const Stage& s, std::uint32_t first, unsigned r) {
        const std::uint64_t row = first + r, base = row * rows.stride + value_head * 128U;
        float value[4], square[4];
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            value[part] = s.value[r][part * 32U + lane];
            square[part] = value[part] * value[part];
        }
        const float inv = rsqrtf(gdn_window_key_sum(square) / static_cast<float>(128U) + epsilon);
#pragma unroll
        for (std::uint32_t part = 0; part < 4U; ++part) {
            const auto channel = part * 32U + lane;
            const float gate_value = 1.0F / (1.0F + expf(-rows.z[base + channel]));
            const float result = value[part] * inv * load_small(norm_weights, norm_kind, channel) * gate_value;
            rows.core[base + channel] = result;
            if constexpr (A8) {
                constexpr std::uint64_t pitch = q8_rows::kDenseA8WideRows;
                const std::uint64_t width = static_cast<std::uint64_t>(value_heads) * 128U, feature = value_head * 128U + channel;
                q8_rows::dense_a8_quantize_lane(result, a8_codes + row * width + feature,
                    a8_deltas + row / pitch * (width / 32U * pitch) + feature / 32U * pitch + row % pitch);
            }
        }
    };
    float retained[K];
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        retained[j] = state[gdn_window_state_index<Storage>(value_head, sub + j * G, column, 128U)];
    fetch(0U);
    commit(stage[0]);
    __syncthreads();
    std::uint32_t first = 0, buffer = 0, chunk = 0;
    for (; first < count; first += Chunk, buffer ^= 1U, ++chunk) {
        const bool more = first + Chunk < count;
        if (more) fetch(first + Chunk);
        // The previous chunk's norm: waves (chunk % 2) * Chunk + r take its row r.
        if (chunk) {
            const unsigned r = wave - (chunk % 2U) * Chunk;
            if (r < Chunk) norm(stage[buffer ^ 1U], first - Chunk, r);
        }
        const Stage& s = stage[buffer];
        const float* key = s.key[0][sub];
        const float* query = s.query[0][sub];
        float* value = &stage[buffer].value[0][local];
        const float* beta = s.beta;
        const float* decay = s.decay;
#pragma unroll 1
        for (unsigned r = 0; r < Chunk; ++r) {
            float products[K];
#pragma unroll
            for (unsigned j = 0; j < K; ++j) {
                retained[j] = retained[j] * *decay;
                products[j] = retained[j] * key[j];
            }
            const float delta = (*value - gdn_lane_group_sum<G>(gdn_lane_tree(products))) * *beta;
#pragma unroll
            for (unsigned j = 0; j < K; ++j) {
                retained[j] = __builtin_fmaf(key[j], delta, retained[j]);
                products[j] = retained[j] * query[j];
            }
            const float out = gdn_lane_group_sum<G>(gdn_lane_tree(products));
            if (sub == 0U) *value = out;
            key += G * Shape::pitch; query += G * Shape::pitch; value += Columns; ++beta; ++decay;
        }
        gdn_lds_barrier(); // the norm above has read stage[buffer ^ 1] before it is refilled
        if (more) commit(stage[buffer ^ 1U]);
        gdn_lds_barrier();
    }
    // Norm of the last chunk.
    {
        const unsigned r = wave - (chunk % 2U) * Chunk;
        if (r < Chunk && first - Chunk + r < count) norm(stage[buffer ^ 1U], first - Chunk, r);
    }
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        state[gdn_window_state_index<Storage>(value_head, sub + j * G, column, 128U)] = retained[j];
}
} // namespace gdn_gates
