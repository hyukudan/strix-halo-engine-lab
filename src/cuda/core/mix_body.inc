// Shared CUDA bank softmax schedule.
    constexpr unsigned bank_lanes = ParallelBanks ? kPairedLanes : kLanes;
    __shared__ float tree[ParallelBanks ? bank_lanes * residual::kMaxBanks : bank_lanes];
    __shared__ float inverse[residual::kMaxBanks];
    __shared__ float probabilities[residual::kMaxQueries][residual::kMaxBanks];
    const unsigned lane = threadIdx.x;
    const unsigned first_bank = ParallelBanks ? threadIdx.y : 0U;
    const unsigned banks_per_slice = ParallelBanks ? 1U : args.bank_count;
    const unsigned out_lane = first_bank * bank_lanes + lane;
    const unsigned out_stride = bank_lanes * blockDim.y;
    auto* scratch = tree + first_bank * bank_lanes;
    const float width = static_cast<float>(args.width);
    const float score_scale = 1.0F / sqrtf(width);

    for (std::size_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        if constexpr (Masked) {
            if (!binding.valid(static_cast<unsigned>(row))) continue;
        }
        const auto offset = row * args.width;
        for (unsigned local_bank = 0; local_bank < banks_per_slice; ++local_bank) {
            const unsigned bank = first_bank + local_bank;
            const float* source = args.banks[bank] + offset;
            float scale = 1.0F;
            if constexpr (!Normalized) {
                float squares = norm::square_partial<kLanes>(source, args.width, lane);
                if constexpr (ParallelBanks)
                    squares += norm::square_partial<kLanes>(source, args.width, lane + bank_lanes);
                const float sum = norm::lds_halving_sum<bank_lanes>(scratch, squares);
                scale = 1.0F / sqrtf(sum / width + args.epsilon);
                if (lane == 0U) inverse[bank] = scale;
            }
            for (unsigned q = 0; q < args.query_count; ++q) {
                const float* query = args.queries + std::size_t{q} * args.width;
                float dot = dot_partial<Normalized>(query, source, args.width, lane, scale);
                if constexpr (ParallelBanks)
                    dot += dot_partial<Normalized>(query, source, args.width, lane + bank_lanes, scale);
                const float score = norm::lds_halving_sum<bank_lanes>(scratch, dot) * score_scale;
                if (lane == 0U) probabilities[q][bank] = score;
            }
        }
        __syncthreads();
        if (out_lane < args.query_count) {
            float* probs = probabilities[out_lane];
            float maximum = probs[0];
            for (unsigned bank = 1; bank < args.bank_count; ++bank) maximum = fmaxf(maximum, probs[bank]);
            float denominator = 0.0F;
            for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                probs[bank] = expf(probs[bank] - maximum);
                denominator += probs[bank];
            }
            for (unsigned bank = 0; bank < args.bank_count; ++bank) probs[bank] /= denominator;
        }
        __syncthreads();
        for (unsigned q = 0; q < args.query_count; ++q) {
            for (unsigned i = out_lane; i < args.width; i += out_stride) {
                float value = 0.0F;
                for (unsigned bank = 0; bank < args.bank_count; ++bank) {
                    const float scale = Normalized ? 1.0F : inverse[bank];
                    const float source = bank_value<Normalized>(args.banks[bank][offset + i], scale);
                    value = fmaf(probabilities[q][bank], source, value);
                }
                args.output[q][offset + i] = value;
            }
        }
        __syncthreads();
    }
