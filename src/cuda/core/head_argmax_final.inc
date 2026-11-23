// Shared greedy final reduction and emit.
    const std::uint32_t row = blockIdx.x;
    if constexpr (Masked) {
        if (!binding.valid(row)) {
            if (threadIdx.x == 0U) emit.masked(row);
            return;
        }
    }
    if constexpr (std::is_same_v<P, ha::Pick2>) {
        const auto* part = reinterpret_cast<const ha::Pick2*>(v.partial);
        ha::Pick2 p;
        for (std::uint32_t c = threadIdx.x; c < v.chunks; c += kThreads) p = ha::merge2(p, part[std::size_t{row} * v.chunks + c]);
        p = block_merge(p);
        if (threadIdx.x == 0U) emit(row, p);
    } else {
        ha::Pick p;
        for (std::uint32_t c = threadIdx.x; c < v.chunks; c += kThreads) p = ha::merge(p, v.partial[std::size_t{row} * v.chunks + c]);
        p = block_merge(p);
        if (threadIdx.x == 0U) emit(row, p);
    }
