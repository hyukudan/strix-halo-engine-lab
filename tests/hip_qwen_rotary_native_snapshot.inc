// Frozen project-owned native kernels from 54594de6. Test reference only.
namespace frozen_native {
template <SmallWeightKind Kind>
__device__ __forceinline__ void norm_rope_body(
    float* values,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t head_stride,
    const void* norm_weights,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    std::uint64_t position,
    GraphPositionKernelArguments dynamic) {
    if (dynamic.position != nullptr) position = *dynamic.position;
    // Another head can report arithmetic failure while this block starts.
    // Broadcast a single atomic observation before any block-wide barriers;
    // per-thread early exits could otherwise diverge around __syncthreads.
    __shared__ unsigned graph_reject;
    if (dynamic.status != nullptr) {
        if (threadIdx.x == 0) graph_reject =
            atomicAdd(dynamic.status, 0U) != 0 || position >= dynamic.context_tokens ? 1U : 0U;
        __syncthreads();
        if (graph_reject != 0) return;
    }
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (head >= head_count ||
        feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* source = shared;
    auto* reduction =
        shared + head_dim;

    const auto index =
        static_cast<std::uint64_t>(
            head) *
            head_stride +
        feature;
    const float input =
        values[index];
    source[feature] = input;
    reduction[feature] =
        input * input;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            reduction[feature] +=
                reduction[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float mean = reduction[0] / static_cast<float>(head_dim) + epsilon;
    const float inv_rms = rsqrtf(mean);
    if (dynamic.status != nullptr && (!isfinite(reduction[0]) || !isfinite(mean) ||
        mean <= 0 || !isfinite(inv_rms))) {
        // The shared reduction and scalar are identical for every thread in
        // this head, so this return is uniform before the remaining barrier.
        if (feature == 0) atomicOr(dynamic.status, 8U);
        return;
    }
    source[feature] =
        round_bf16(
            input *
            inv_rms *
            load_small_weight<Kind>(
                 norm_weights,
                 feature));
    if (dynamic.status != nullptr && !isfinite(source[feature])) atomicOr(dynamic.status, 8U);
    __syncthreads();

    float result =
        source[feature];
    if (feature < rotary_dim) {
        const auto half =
            rotary_dim / 2U;
        const auto pair =
            feature % half;
        const float exponent =
            2.0F *
            static_cast<float>(pair) /
            static_cast<float>(
                rotary_dim);
        const float angle =
            static_cast<float>(
                position) *
            powf(
                theta,
                -exponent);
        const float cosine =
            cosf(angle);
        const float sine =
            sinf(angle);

        if (feature < half) {
            result =
                source[feature] *
                    cosine -
                source[
                    feature +
                    half] *
                    sine;
        } else {
            result =
                source[feature] *
                    cosine +
                source[
                    feature -
                    half] *
                    sine;
        }
    }
    const float rounded = round_bf16(result);
    if (dynamic.status != nullptr && (!isfinite(result) || !isfinite(rounded))) {
        atomicOr(dynamic.status, 8U);
        return;
    }
    values[index] = rounded;
}
template <SmallWeightKind Kind>
__global__ void norm_rope_kernel(float* values, std::uint32_t head_count, std::uint32_t head_dim,
    std::uint32_t head_stride, const void* norm_weights, float epsilon, std::uint32_t rotary_dim,
    float theta, std::uint64_t position, GraphPositionKernelArguments dynamic) {
    norm_rope_body<Kind>(values, head_count, head_dim, head_stride, norm_weights, epsilon,
        rotary_dim, theta, position, dynamic);
}
template <SmallWeightKind Kind>
__global__ void block_score_kernel(
    const float* index_query,
    const std::uint16_t* raw_keys,
    std::uint32_t index_heads,
    std::uint32_t head_dim,
    std::uint32_t compress_ratio,
    const void* key_norm,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    float* block_scores, GraphSelectionArguments dynamic) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (dynamic.position.position != nullptr) {
        std::uint32_t visible = 0;
        if (!graph_visible(dynamic, visible) || visible / compress_ratio <= dynamic.block_topk ||
            block >= visible / compress_ratio) return;
    }
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch =
        shared + head_dim;

    float mean = 0.0F;
    const auto token_start =
        static_cast<std::uint64_t>(
            block) *
        compress_ratio;
    for (std::uint32_t offset = 0;
         offset < compress_ratio;
         ++offset) {
        mean +=
            bf16_to_float(
                raw_keys[
                    (token_start +
                     offset) *
                        head_dim +
                    feature]);
    }
    mean /=
        static_cast<float>(
            compress_ratio);

    // Upstream casts the FP32 mean back to raw-key dtype before k_norm.
    mean =
        round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] =
        mean * mean;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (dynamic.position.status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv_rms))) {
        if (feature == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            load_small_weight<Kind>(
                 key_norm,
                 feature));
    __syncthreads();

    float rotated =
        pooled[feature];
    if (feature < rotary_dim) {
        const auto half =
            rotary_dim / 2U;
        const auto pair =
            feature % half;
        const float exponent =
            2.0F *
            static_cast<float>(pair) /
            static_cast<float>(
                rotary_dim);
        const float angle =
            static_cast<float>(
                token_start) *
            powf(
                theta,
                -exponent);
        const float cosine =
            cosf(angle);
        const float sine =
            sinf(angle);
        if (feature < half) {
            rotated =
                pooled[feature] *
                    cosine -
                pooled[
                    feature +
                    half] *
                    sine;
        } else {
            rotated =
                pooled[feature] *
                    cosine +
                pooled[
                    feature -
                    half] *
                    sine;
        }
    }
    // Every lane must finish reading its RoPE partner before any lane writes
    // the rotated value back into the shared pooled vector.
    __syncthreads();
    pooled[feature] =
        round_bf16(rotated);
    if (dynamic.position.status != nullptr && !isfinite(pooled[feature])) atomicOr(dynamic.position.status, 8U);
    __syncthreads();

    __shared__ float total;
    if (feature == 0) {
        total = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t head = 0;
         head < index_heads;
         ++head) {
        scratch[feature] =
            index_query[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim +
                feature] *
            pooled[feature];
        __syncthreads();

        for (std::uint32_t stride =
                 head_dim / 2U;
             stride > 0;
             stride >>= 1U) {
            if (feature < stride) {
                scratch[feature] +=
                    scratch[
                        feature +
                        stride];
            }
            __syncthreads();
        }
        if (feature == 0) {
            if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(total)) atomicOr(dynamic.position.status, 8U);
        block_scores[block] =
            total /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}
template <SmallWeightKind Kind>
__device__ __forceinline__ void block_score_body(
    const float* index_query,
    const std::uint16_t* raw_keys,
    std::uint32_t index_heads,
    std::uint32_t head_dim,
    std::uint32_t compress_ratio,
    const void* key_norm,
    float epsilon,
    std::uint32_t rotary_dim,
    float theta,
    float* block_scores, GraphSelectionArguments dynamic) {
    const auto block =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (dynamic.position.position != nullptr) {
        std::uint32_t visible = 0;
        if (!graph_visible(dynamic, visible) || visible / compress_ratio <= dynamic.block_topk ||
            block >= visible / compress_ratio) return;
    }
    const auto feature =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (feature >= head_dim) {
        return;
    }

    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch =
        shared + head_dim;

    float mean = 0.0F;
    const auto token_start =
        static_cast<std::uint64_t>(
            block) *
        compress_ratio;
    for (std::uint32_t offset = 0;
         offset < compress_ratio;
         ++offset) {
        mean +=
            bf16_to_float(
                raw_keys[
                    (token_start +
                     offset) *
                        head_dim +
                    feature]);
    }
    mean /=
        static_cast<float>(
            compress_ratio);

    // Upstream casts the FP32 mean back to raw-key dtype before k_norm.
    mean =
        round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] =
        mean * mean;
    __syncthreads();

    for (std::uint32_t stride =
             head_dim / 2U;
         stride > 0;
         stride >>= 1U) {
        if (feature < stride) {
            scratch[feature] +=
                scratch[
                    feature +
                    stride];
        }
        __syncthreads();
    }

    const float inv_rms =
        rsqrtf(
            scratch[0] /
                static_cast<float>(
                    head_dim) +
            epsilon);
    if (dynamic.position.status != nullptr && (!isfinite(scratch[0]) || !isfinite(inv_rms))) {
        if (feature == 0) atomicOr(dynamic.position.status, 8U);
        return;
    }
    pooled[feature] =
        round_bf16(
            mean *
            inv_rms *
            load_small_weight<Kind>(
                 key_norm,
                 feature));
    __syncthreads();

    float rotated =
        pooled[feature];
    if (feature < rotary_dim) {
        const auto half =
            rotary_dim / 2U;
        const auto pair =
            feature % half;
        const float exponent =
            2.0F *
            static_cast<float>(pair) /
            static_cast<float>(
                rotary_dim);
        const float angle =
            static_cast<float>(
                token_start) *
            powf(
                theta,
                -exponent);
        const float cosine =
            cosf(angle);
        const float sine =
            sinf(angle);
        if (feature < half) {
            rotated =
                pooled[feature] *
                    cosine -
                pooled[
                    feature +
                    half] *
                    sine;
        } else {
            rotated =
                pooled[feature] *
                    cosine +
                pooled[
                    feature -
                    half] *
                    sine;
        }
    }
    // Every lane must finish reading its RoPE partner before any lane writes
    // the rotated value back into the shared pooled vector.
    __syncthreads();
    pooled[feature] =
        round_bf16(rotated);
    if (dynamic.position.status != nullptr && !isfinite(pooled[feature])) atomicOr(dynamic.position.status, 8U);
    __syncthreads();

    __shared__ float total;
    if (feature == 0) {
        total = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t head = 0;
         head < index_heads;
         ++head) {
        scratch[feature] =
            index_query[
                static_cast<std::uint64_t>(
                    head) *
                    head_dim +
                feature] *
            pooled[feature];
        __syncthreads();

        for (std::uint32_t stride =
                 head_dim / 2U;
             stride > 0;
             stride >>= 1U) {
            if (feature < stride) {
                scratch[feature] +=
                    scratch[
                        feature +
                        stride];
            }
            __syncthreads();
        }
        if (feature == 0) {
            if (dynamic.position.status != nullptr && !isfinite(scratch[0])) atomicOr(dynamic.position.status, 8U);
            total +=
                fmaxf(
                    scratch[0],
                    0.0F);
        }
        __syncthreads();
    }

    if (feature == 0) {
        if (dynamic.position.status != nullptr && !isfinite(total)) atomicOr(dynamic.position.status, 8U);
        block_scores[block] =
            total /
            sqrtf(
                static_cast<float>(
                    head_dim));
    }
}
template <SmallWeightKind Kind, unsigned Rows>
__global__ void norm_rope_rows_kernel(QsaRowsNormViews<Rows> views, std::uint32_t head_count,
    std::uint32_t head_dim, std::uint32_t head_stride, const void* norm_weights, float epsilon,
    std::uint32_t rotary_dim, float theta) {
    norm_rope_body<Kind>(views.values[blockIdx.y], head_count, head_dim, head_stride, norm_weights,
        epsilon, rotary_dim, theta, views.position[blockIdx.y], GraphPositionKernelArguments{});
}
template <SmallWeightKind Kind, unsigned Rows>
__global__ void block_score_rows_kernel(QsaRowsScoreViews<Rows> views, const std::uint16_t* raw_keys,
    std::uint32_t index_heads, std::uint32_t head_dim, std::uint32_t compress_ratio,
    const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta) {
    const auto block = static_cast<std::uint32_t>(blockIdx.x);
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch = shared + head_dim;
    float mean = 0.0F;
    const auto token_start = static_cast<std::uint64_t>(block) * compress_ratio;
    for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
        mean += bf16_to_float(raw_keys[(token_start + offset) * head_dim + feature]);
    mean /= static_cast<float>(compress_ratio);
    mean = round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] = mean * mean;
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) scratch[feature] += scratch[feature + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + epsilon);
    pooled[feature] = round_bf16(mean * inv_rms * load_small_weight<Kind>(key_norm, feature));
    __syncthreads();
    float rotated = pooled[feature];
    if (feature < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = feature % half;
        const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
        const float angle = static_cast<float>(token_start) * powf(theta, -exponent);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
        else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
    }
    __syncthreads();
    pooled[feature] = round_bf16(rotated);
    __syncthreads();
    __shared__ float total;
    constexpr unsigned Span = Rows <= 8U ? Rows : 8U;
    const unsigned base = Rows <= 8U ? 0U : static_cast<unsigned>(blockIdx.y) * 8U;
    for (unsigned rr = 0; rr < Span; ++rr) {
        const unsigned r = base + rr;
        if (block >= views.blocks[r]) continue; // uniform per block
        if (feature == 0) total = 0.0F;
        __syncthreads();
        for (std::uint32_t head = 0; head < index_heads; ++head) {
            scratch[feature] = views.index_query[r][static_cast<std::uint64_t>(head) * head_dim + feature] * pooled[feature];
            __syncthreads();
            for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
                if (feature < stride) scratch[feature] += scratch[feature + stride];
                __syncthreads();
            }
            if (feature == 0) total += fmaxf(scratch[0], 0.0F);
            __syncthreads();
        }
        if (feature == 0) views.block_scores[r][block] = total / sqrtf(static_cast<float>(head_dim));
    }
}
template <SmallWeightKind Kind, bool Batch = false>
__global__ void block_pool_kernel(const std::uint16_t* raw_keys, std::uint32_t first_block, std::uint32_t head_dim,
    std::uint32_t compress_ratio, const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta, float* pooled_keys,
    const uint2* bounds = nullptr, std::uint32_t groups = 0U) {
    auto local = static_cast<std::uint32_t>(blockIdx.x);
    if constexpr (Batch) {
        const auto absolute = first_block + local;
        const auto group = qsa_prefill_block_group(bounds, groups, absolute);
        if (group == groups) return;
        const auto range = bounds[group];
        if (absolute < range.x) return;
        local = absolute - range.x;
        first_block = range.x;
    }
    const auto block = first_block + local;
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    if (feature >= head_dim) return;
    extern __shared__ float shared[];
    auto* pooled = shared;
    auto* scratch = shared + head_dim;
    float mean = 0.0F;
    const auto token_start = static_cast<std::uint64_t>(block) * compress_ratio;
    for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
        mean += bf16_to_float(raw_keys[(token_start + offset) * head_dim + feature]);
    mean /= static_cast<float>(compress_ratio);
    mean = round_bf16(mean);
    pooled[feature] = mean;
    scratch[feature] = mean * mean;
    __syncthreads();
    for (std::uint32_t stride = head_dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) scratch[feature] += scratch[feature + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(head_dim) + epsilon);
    pooled[feature] = round_bf16(mean * inv_rms * load_small_weight<Kind>(key_norm, feature));
    __syncthreads();
    float rotated = pooled[feature];
    if (feature < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = feature % half;
        const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
        const float angle = static_cast<float>(token_start) * powf(theta, -exponent);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        if (feature < half) rotated = pooled[feature] * cosine - pooled[feature + half] * sine;
        else rotated = pooled[feature] * cosine + pooled[feature - half] * sine;
    }
    pooled_keys[static_cast<std::uint64_t>(block) * head_dim + feature] = round_bf16(rotated);
}
template <unsigned Rows, bool Batch = false>
__global__ __launch_bounds__(64) void rope_table_kernel(qsa_index::NormGroupViews<Rows> single, std::uint32_t rotary_dim, float theta, const qsa_index::NormGroupViews<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const qsa_index::NormGroupViews<Rows>& {
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
template <SmallWeightKind Kind>
__global__ void norm_rope_rows_kernel(MtpBatchNormRopeRows rows, std::uint32_t head_count, std::uint32_t head_dim,
    std::uint32_t head_stride, const void* norm_weights, float epsilon, std::uint32_t rotary_dim, float theta) {
    norm_rope_body<Kind>(rows.values[blockIdx.y], head_count, head_dim, head_stride, norm_weights, epsilon, rotary_dim,
        theta, rows.position[blockIdx.y], GraphPositionKernelArguments{});
}
template <SmallWeightKind Kind>
__global__ void block_score_rows_kernel(MtpBatchSelectRows rows, std::uint32_t index_heads, std::uint32_t head_dim,
    std::uint32_t compress_ratio, const void* key_norm, float epsilon, std::uint32_t rotary_dim, float theta,
    GraphSelectionArguments dynamic) {
    const auto r = blockIdx.y;
    if (blockIdx.x >= rows.blocks[r]) return;
    block_score_body<Kind>(rows.index_query[r], rows.raw_keys[r], index_heads, head_dim, compress_ratio, key_norm, epsilon,
        rotary_dim, theta, rows.block_scores[r], dynamic);
}
} // namespace frozen_native
