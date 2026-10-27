// MC certificate: |w| <= 1 and every stored raw dot |d| < 2^120 imply all ten FP32 FMA
// intermediates < 2^124. Admitted calls cannot set merge bit 8. Unproven calls execute the
// original merge_z body and publish only after its final header is known. No public admission bits.
// A row status implies header[0] != 0 before merge (production CSR/projection invariant).
template <bool Acc>
__device__ __forceinline__ void moe_prefill_merge_commit_element(std::uint32_t* header, std::uint32_t* status,
    const std::uint32_t* routes, const float* rawdots, const float* route_weights, float* accumulation, std::uint16_t* output,
    std::uint32_t out, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k) {
    if (header[0] || *status) { if (Acc && out < hidden_size) accumulation[out] = 0.0F; return; }
    if (out >= hidden_size) return;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) {
        if (Acc) accumulation[out] = 0.0F;
        output[out] = 0U;
        return;
    }
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = rawdots[static_cast<std::uint64_t>(slot) * hidden_size + out];
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    if (Acc) accumulation[out] = current;
    const auto bits = __float_as_uint(current);
    if ((bits & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); return; }
    output[out] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
// The base commit's block (0, 0) without its element stores: one 256-thread block after the merge. RowStatus(r) returns
// row r's status word.
template <class RowStatus>
__device__ __forceinline__ void moe_prefill_commit_record(std::uint32_t* header, std::uint32_t row_count,
    const RowStatus& row_status, DeferredMoeFirstError* record) {
    const std::uint32_t failure = header[0];
    __shared__ std::uint32_t first_bad;
    if (threadIdx.x == 0U) first_bad = row_count;
    __syncthreads();
    for (std::uint32_t r = threadIdx.x; r < row_count; r += blockDim.x)
        if (*row_status(r)) { atomicMin(&first_bad, r); break; }
    __syncthreads();
    if (threadIdx.x == 0U) {
        if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) header[word] = 0U;
        if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) header[word] = 0U;
        if (first_bad < row_count)
            merge_deferred_moe_first_error(*record, *row_status(first_bad), DeferredMoeDomain::q4k_a8);
    }
}

template <bool Fused = false, bool Acc = false, class View, class RowAt>
__device__ __forceinline__ void moe_prefill_merge_z_body(View view, std::uint32_t* header,
    std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k, const RowAt& row_at, const unsigned* unsafe = nullptr) {
    const auto row = blockIdx.y;
    auto* status = row_at(view.status, row);
    const auto out = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (header[0] || *status) { if (out < hidden_size) row_at(view.accumulation, row)[out] = 0.0F; return; }
    // The private word is read after the original header/status guards, before any changed stores.
    if constexpr (Fused) if (*unsafe == 0U) {
        moe_prefill_merge_commit_element<Acc>(header, status, row_at(view.routes, row), row_at(view.rawdots, row),
            row_at(view.route_weights, row), row_at(view.accumulation, row), row_at(view.output, row),
            out, hidden_size, experts, top_k);
        return;
    }
    if (out >= hidden_size) return;
    const auto* routes = row_at(view.routes, row);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) { row_at(view.accumulation, row)[out] = 0.0F; return; }
    const auto* rawdots = row_at(view.rawdots, row);
    const auto* route_weights = row_at(view.route_weights, row);
    float dots[10];
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = rawdots[static_cast<std::uint64_t>(slot) * hidden_size + out];
    float current = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = route_weights[slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    row_at(view.accumulation, row)[out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}

template <class View, class RowAt, class Store>
__device__ __forceinline__ void moe_prefill_commit_body(View view, std::uint32_t* header,
    std::uint32_t row_count, std::uint32_t hidden_size, std::uint32_t* record, unsigned logical_x, unsigned logical_y, const RowAt& row_at, const Store& store) {
    const std::uint32_t failure = header[0];
    if (logical_x == 0U && logical_y == 0U) {
        // The serial merge over rows 0..T-1 records only the first nonzero row status (a merge of
        // a zero status changes nothing): the block finds that row in parallel, thread 0 merges it.
        __shared__ std::uint32_t first_bad;
        if (threadIdx.x == 0U) first_bad = row_count;
        __syncthreads();
        for (std::uint32_t r = threadIdx.x; r < row_count; r += blockDim.x)
            if (*row_at(view.status, r)) { atomicMin(&first_bad, r); break; }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) header[word] = 0U;
            if (failure & 7U) for (std::uint32_t word = 8U; word < 24U; ++word) header[word] = 0U;
            if (first_bad < row_count)
                merge_deferred_moe_first_error(*reinterpret_cast<DeferredMoeFirstError*>(record),
                    *row_at(view.status, first_bad), DeferredMoeDomain::q4k_a8);
        }
    }
    if (failure) return;
    const auto out = logical_x * blockDim.x + threadIdx.x;
    if (out < hidden_size) store(row_at(view.output, logical_y) + out,
        row_at(view.accumulation, logical_y)[out]);
}

template <unsigned CTAs, class View, class RowAt, class Store>
__device__ __forceinline__ void moe_prefill_finish_body(View view, unsigned* header, unsigned row_count,
    unsigned hidden, unsigned* record, const unsigned* unsafe, const RowAt& row_at, const Store& store) {
    static_assert(CTAs > 0U);
    if (header[0] || *unsafe == 0U) {
        if (blockIdx.x == 0U)
            moe_prefill_commit_record(header, row_count, [&](unsigned r) { return row_at(view.status, r); },
                reinterpret_cast<DeferredMoeFirstError*>(record));
        return;
    }
    const unsigned tiles=(hidden+blockDim.x-1U)/blockDim.x;
    for (unsigned logical=blockIdx.x; logical<tiles*row_count; logical+=CTAs)
        moe_prefill_commit_body(view, header, row_count, hidden, record, logical%tiles, logical/tiles, row_at, store);
}
