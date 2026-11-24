// F9 `recur` gated delta rows body (see gated_delta_rows.hpp). Textual fragment: the whole body of a __global__ function
// whose scope has the template parameter `gdn_recurrent_lds::Form F`, a policy object `policy` (gated_delta_policy.hpp) and
// the parameters qkv, beta, gate, state, state_out, core, key_heads, value_heads, head_dim, rows, qkv_stride, core_stride.
// Former recurrent_rows_kernel body (src/hip_gdn_launcher.hip), lifted verbatim; the gate, lane products and update are the
// policy hooks, and the gate row stride is the policy's (value heads for a scalar gate, value heads * head_dim for a per-key
// channel gate). A fragment and not a device function so the kernel keeps its machine code.
// NOLINTBEGIN
    using ::strix::ops::recur::kStepThreads;
    using ::strix::ops::recur::kStepValueTile;
    const auto tile_count = head_dim / kStepValueTile;
    const auto value_head = static_cast<std::uint32_t>(blockIdx.x) / tile_count;
    const auto tile = static_cast<std::uint32_t>(blockIdx.x) % tile_count;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    if (value_head >= value_heads || lane >= head_dim) return;
    const auto key_head = value_head % key_heads;
    const auto value_base = tile * kStepValueTile;
    const auto q_base = static_cast<std::uint64_t>(key_head) * head_dim;
    const auto k_base = static_cast<std::uint64_t>(key_heads) * head_dim + q_base;
    const auto v_base = static_cast<std::uint64_t>(key_heads) * head_dim * 2ULL + static_cast<std::uint64_t>(value_head) * head_dim;
    __shared__ float reduction[gdn_recurrent_lds::extent(F)];
    __shared__ float deltas[kStepValueTile];
    float retained[kStepValueTile];
#pragma unroll
    for (std::uint32_t column = 0; column < kStepValueTile; ++column)
        retained[column] = state[(static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base + column];
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* q = qkv + r * qkv_stride;
        const float* b = beta + static_cast<std::uint64_t>(r) * value_heads;
        const float* g = gate + static_cast<std::uint64_t>(r) * policy.gate_row_stride(value_heads, head_dim);
        float* out = core + r * core_stride;
        const float decay = policy.decay(g, value_head, head_dim, lane);
#pragma unroll
        for (std::uint32_t column = 0; column < kStepValueTile; ++column) {
            retained[column] = retained[column] * decay;
            reduction[gdn_recurrent_lds::index(F, lane, column)] = policy.lane_term(retained[column], q[k_base + lane]);
        }
        __syncthreads();
        for (std::uint32_t stride = kStepThreads / 2U; stride > 0; stride >>= 1U) {
            if (lane < stride) {
#pragma unroll
                for (std::uint32_t column = 0; column < kStepValueTile; ++column)
                    reduction[gdn_recurrent_lds::index(F, lane, column)] +=
                        reduction[gdn_recurrent_lds::index(F, lane + stride, column)];
            }
            __syncthreads();
        }
        if (lane == 0) {
#pragma unroll
            for (std::uint32_t column = 0; column < kStepValueTile; ++column)
                deltas[column] = (q[v_base + value_base + column] - reduction[gdn_recurrent_lds::index(F, 0U, column)]) * b[value_head];
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t column = 0; column < kStepValueTile; ++column) {
            retained[column] = policy.update(q[k_base + lane], deltas[column], retained[column]);
            policy.observe(retained[column]);
            reduction[gdn_recurrent_lds::index(F, lane, column)] = policy.lane_term(retained[column], q[q_base + lane]);
        }
        __syncthreads();
        for (std::uint32_t stride = kStepThreads / 2U; stride > 0; stride >>= 1U) {
            if (lane < stride) {
#pragma unroll
                for (std::uint32_t column = 0; column < kStepValueTile; ++column)
                    reduction[gdn_recurrent_lds::index(F, lane, column)] +=
                        reduction[gdn_recurrent_lds::index(F, lane + stride, column)];
            }
            __syncthreads();
        }
        if (lane == 0) {
#pragma unroll
            for (std::uint32_t column = 0; column < kStepValueTile; ++column)
                out[static_cast<std::uint64_t>(value_head) * head_dim + value_base + column] = reduction[gdn_recurrent_lds::index(F, 0U, column)];
        }
        __syncthreads();  // the next row rewrites reduction[] and deltas[]
    }
#pragma unroll
    for (std::uint32_t column = 0; column < kStepValueTile; ++column)
        state_out[(static_cast<std::uint64_t>(value_head) * head_dim + lane) * head_dim + value_base + column] = retained[column];
// NOLINTEND
