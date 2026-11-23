// Shared greedy partial reduction.
    const std::uint32_t row = blockIdx.y, chunk = blockIdx.x;
    if constexpr (Masked) {
        if (!binding.valid(row)) return;
    }
    std::uint32_t lo = 0, hi = 0;
    ha::chunk_range(v.count, v.chunks, chunk, lo, hi);
    const float* x = v.logits + row * v.stride;
    if constexpr (std::is_same_v<P, ha::Pick2>) {
        ha::Pick2 p;
#pragma unroll 4
        for (std::uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
            const float y = x[i];
            const std::uint32_t g = v.first_index + i;
            if (!isfinite(y)) ++p.first.nonfinite;
            else ha::push2(p, y, g);
        }
        p = block_merge(p);
        if (threadIdx.x == 0U) reinterpret_cast<ha::Pick2*>(v.partial)[std::size_t{row} * v.chunks + chunk] = p;
    } else {
        static_assert(std::is_same_v<P, ha::Pick>, "head pick: Pick or Pick2");
        ha::Pick p;
#pragma unroll 4
        for (std::uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
            const float y = x[i];
            const std::uint32_t g = v.first_index + i;
            if (!isfinite(y)) ++p.nonfinite;
            else if (ha::better(y, g, p.value, p.index)) { p.value = y; p.index = g; }
        }
        p = block_merge(p);
        if (threadIdx.x == 0U) v.partial[std::size_t{row} * v.chunks + chunk] = p;
    }
