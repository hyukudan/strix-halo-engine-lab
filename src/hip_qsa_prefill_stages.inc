// Shared per-group and opt-in batched prefill store/pool kernels.
// Batch changes only descriptor selection; the workgroup arithmetic below is common.
// New-block intervals are ordered and disjoint, including empty intervals. A
// flattened grid avoids padding every group to the first group's context prefix.
__device__ __forceinline__ unsigned qsa_prefill_block_group(const uint2* bounds, unsigned groups, unsigned block) {
    unsigned lo = 0, hi = groups;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo) / 2U;
        const auto range = bounds[mid];
        if (block >= range.x + range.y) lo = mid + 1U;
        else hi = mid;
    }
    return lo;
}
// Appends every row's raw index key and K/V before any row attends.
template <unsigned Rows> struct QsaRowsStoreViews {
    const float* index_key[Rows];
    const float* k[Rows];
    const float* v[Rows];
    std::uint64_t position[Rows];
    std::uint16_t* journal; // optional [Rows][index_dim] raw keys for the tail commit
};
template <unsigned Rows, bool Batch = false>
__global__ void store_rows_kernel(QsaRowsStoreViews<Rows> single, std::uint16_t* raw_state,
    std::uint16_t* kv_state, std::uint32_t index_dim, std::uint32_t kv_width, const QsaRowsStoreViews<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const QsaRowsStoreViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.z];
        else return single;
    }();
    const auto r = blockIdx.y;
    const auto feature = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto position = views.position[r];
    if (feature < index_dim) {
        const auto bits = float_to_bf16(views.index_key[r][feature]);
        raw_state[position * static_cast<std::uint64_t>(index_dim) + feature] = bits;
        if constexpr (Rows <= 8U) {
            if (views.journal) views.journal[static_cast<std::uint64_t>(r) * index_dim + feature] = bits;
        } else {
            // Prefill groups: the journal holds 8 row keys; the group's last 8-row tile writes
            // them, as the last serial tile would (earlier tiles' keys are overwritten there).
            if (views.journal && r >= Rows - 8U) views.journal[static_cast<std::uint64_t>(r - (Rows - 8U)) * index_dim + feature] = bits;
        }
    }
    if (feature < kv_width) {
        const auto row = position * static_cast<std::uint64_t>(kv_width) * 2ULL;
        kv_state[row + feature] = float_to_bf16(views.k[r][feature]);
        kv_state[row + kv_width + feature] = float_to_bf16(views.v[r][feature]);
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
