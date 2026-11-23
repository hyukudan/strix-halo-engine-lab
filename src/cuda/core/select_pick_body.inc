// Included in the two pick entry points. Keeping the body in kernel scope
// preserves the instruction schedule of the materialized-row instances.
    const float negative_infinity = __uint_as_float(0xff800000U);
    const auto emit = [&](unsigned i, unsigned slot) {
        const unsigned id = row.ids ? row.ids[i] : i;
        for (unsigned j = 0; j < spec.expansion; ++j)
            row.selected[slot * spec.expansion + j] = id * spec.expansion + j;
        if constexpr (Order == KeyOrder::ordered)
            if (row.selected_keys) row.selected_keys[slot] = row.keys[i];
    };
    if constexpr (Order == KeyOrder::raw_float)
        if (row.pool == Pool::source && row.groups != 0) make_pool(scratch, row);
    unsigned kept = min(row.groups, spec.top_k);
    if (row.groups <= spec.top_k) {
        for (unsigned i = threadIdx.x; i < row.groups; i += pick_threads) emit(i, i);
    } else {
        const unsigned reserved = static_cast<unsigned>(spec.force_newest);
        const auto actual = pick(scratch, row.groups - reserved, kept - reserved,
            Order == KeyOrder::eligible_float, [&](unsigned i) {
                if constexpr (Order == KeyOrder::ordered) return Key{row.keys[i], true};
                else {
                    const bool masked = Order == KeyOrder::raw_float && row.pool == Pool::user &&
                                        row.candidates[i / row.pool_width] == 0;
                    const float value = masked ? negative_infinity : row.scores[i];
                    return Key{float_key(value, Order != KeyOrder::raw_float),
                               Order != KeyOrder::eligible_float || value > negative_infinity};
                }
            }, [&](unsigned i, unsigned slot) {
                emit(i, slot);
                if constexpr (Order == KeyOrder::eligible_float)
                    if (row.mask_scores) row.mask_scores[i] = negative_infinity;
            });
        if (spec.force_newest && threadIdx.x == 0) emit(row.groups - 1U, kept - 1U);
        if constexpr (Order == KeyOrder::eligible_float) {
            for (unsigned i = threadIdx.x; i < kept - reserved - actual; i += pick_threads) emit(0, i);
            if (row.mask_scores && actual < kept - reserved && threadIdx.x == 0) row.mask_scores[0] = negative_infinity;
        }
    }
    if (threadIdx.x == 0) {
        kept *= spec.expansion;
        for (unsigned j = 0; j < row.tail; ++j) row.selected[kept + j] = row.groups * spec.expansion + j;
        if (row.count) *row.count = kept + row.tail;
    }
