// DS41 model glue for the library policy-v1 core. No GEMM kernels live here.
struct IntegerTile {
    int rows = 0, first_col = 0, split = 0, splits = 1;
    const unsigned* packed = nullptr;
    unsigned expert = 0, begin = 0, base_col = 0;
};
struct RouteAdapter {
    static constexpr bool kActive = true, kTiles = true;
    // next() yields one tile per CTA (iteration 0 only).
    static constexpr bool kSingleTile = true;
    using Tile = IntegerTile;
    PrefillRoutesView routes;
    const signed char* codes;
    const float* deltas;
    unsigned width;
    bool input_slots;
    unsigned band = 0;
    __device__ __forceinline__ bool next(Tile& t, unsigned iter, int bn) const {
        if (iter) return false;
        const unsigned halves = routes.batch_rows == 32U ? 2U : 1U;
        const auto block = strix::core::integer_block(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, band);
        const unsigned batch = block.batch / halves, offset = (block.batch % halves) * 16U;
        if (batch >= *routes.batch_total || offset >= routes.batch_count[batch]) return false;
        t.expert = routes.batch_expert[batch];
        t.begin = routes.batch_begin[batch] + offset;
        t.rows = int(min(16U, routes.batch_count[batch] - offset));
        t.first_col = int(block.column) * bn;
        t.base_col = block.column;
        return true;
    }
    __device__ __forceinline__ const signed char* activation_codes() const { return codes; }
    __device__ __forceinline__ const float* activation_deltas() const { return deltas; }
    __device__ __forceinline__ unsigned slot(const Tile& t, int r) const { return routes.entry_slots[t.begin + unsigned(r)]; }
    __device__ __forceinline__ unsigned source(const Tile& t, int r) const {
        return input_slots ? slot(t, r) : routes.entry_rows[t.begin + unsigned(r)];
    }
    // Rows by staged source index: a core that stages source() in LDS passes the value back.
    __device__ __forceinline__ const signed char* code_row_of(unsigned src) const { return codes + std::uint64_t(src) * width; }
    __device__ __forceinline__ float delta_of(unsigned src, unsigned g) const { return deltas[std::uint64_t(src) * (width / 32U) + g]; }
    __device__ __forceinline__ const signed char* code_row(const Tile& t, int r) const { return code_row_of(source(t, r)); }
    __device__ __forceinline__ float delta(const Tile& t, int r, int g) const { return delta_of(source(t, r), unsigned(g)); }
};
struct GateUpAdapter : RouteAdapter {
    static constexpr bool kFinish = true;
    PrefillGateUpView view;
    __device__ __forceinline__ unsigned row_bytes() const { return unsigned(view.row_bytes); }
    __device__ __forceinline__ const unsigned long long* grid() const { return prefill_iq2_grid; }
    __device__ __forceinline__ const unsigned char* weight_row(const Tile& t, int col) const {
        const unsigned local = unsigned(col - t.first_col);
        const void* matrix = local < 32U ? view.gate : view.up;
        return static_cast<const unsigned char*>(matrix) + std::uint64_t(t.expert) * view.expert_stride_bytes +
               std::uint64_t(t.base_col * 32U + local % 32U) * view.row_bytes;
    }
    __device__ __forceinline__ void store_raw_at(const Tile& t, unsigned dst, int col, float v) const {
        if (!view.gate_out) return;
        const unsigned local = unsigned(col - t.first_col);
        float* out = local < 32U ? view.gate_out : view.up_out;
        out[std::uint64_t(dst) * view.inter + t.base_col * 32U + local % 32U] = v;
    }
    __device__ __forceinline__ void store_raw(const Tile& t, int r, int col, float v) const { store_raw_at(t, slot(t, r), col, v); }
    template<bool PairMajor = false>
    __device__ __forceinline__ void finish_tile(const Tile& t, const float* values, int bm, int bn) const {
        finish_rows<PairMajor>(t, values, nullptr, bm, bn);
    }
    // finish_tile with the destination slots staged by the core (nullptr: read the routes).
    template<bool PairMajor = false>
    __device__ __forceinline__ void finish_rows(const Tile& t, const float* values, const unsigned* staged, int bm, int bn) const {
        const auto slot_of = [&](unsigned r) { return staged ? staged[r] : slot(t, int(r)); };
        const unsigned lane = threadIdx.x & 31U, waves = blockDim.x / 32U;
        for (unsigned r = threadIdx.x / 32U; r < unsigned(bm); r += waves) {
            if (r >= unsigned(t.rows)) continue; // uniform wave
            const float gate = values[PairMajor ? strix::core::integer_pair_value_offset(0, r, lane) : r * unsigned(bn) + lane];
            const float up = values[PairMajor ? strix::core::integer_pair_value_offset(1, r, lane) : r * unsigned(bn) + lane + 32U];
            if constexpr (PairMajor) {
                if (view.gate_out) {
                    const auto dst = std::uint64_t(slot_of(r)) * view.inter + t.base_col * 32U + lane;
                    view.gate_out[dst] = gate;
                    view.up_out[dst] = up;
                }
            }
            const float x = swiglu_clamped(gate, up, view.limit);
            const auto a = strix::core::quantize_a8_row32(x);
            const std::uint64_t s = slot_of(r), col = std::uint64_t(t.base_col) * 32U + lane;
            view.act_codes[s * view.inter + col] = a.code;
            if (!lane) {
                const auto g = s * (view.inter / 32U) + t.base_col;
                view.act_deltas[g] = a.delta; view.act_sums[g] = a.sum;
            }
        }
    }
};
struct DownAdapter : RouteAdapter {
    static constexpr bool kFinish = false;
    PrefillDownView view;
    __device__ __forceinline__ unsigned row_bytes() const { return unsigned(view.row_bytes); }
    __device__ __forceinline__ const unsigned long long* grid() const { return nullptr; }
    __device__ __forceinline__ const unsigned char* weight_row(const Tile& t, int col) const {
        return static_cast<const unsigned char*>(view.weights) + std::uint64_t(t.expert) * view.expert_stride_bytes +
               std::uint64_t(col) * view.row_bytes;
    }
    __device__ __forceinline__ void store_raw_at(const Tile&, unsigned dst, int col, float v) const {
        view.out[std::uint64_t(dst) * view.outputs + unsigned(col)] = v;
    }
    __device__ __forceinline__ void store_raw(const Tile& t, int r, int col, float v) const { store_raw_at(t, slot(t, r), col, v); }
    __device__ __forceinline__ void finish_tile(const Tile&, const float*, int, int) const {}
    template<bool = false> __device__ __forceinline__ void finish_rows(const Tile&, const float*, const unsigned*, int, int) const {}
};
bool fast_i4_down() {
    const char* scope = std::getenv("STRIX_DS41_FAST_I4_SCOPE");
    if (!scope || std::strcmp(scope, "all") == 0) return true;
    if (std::strcmp(scope, "gu") == 0) return false;
    throw std::invalid_argument("STRIX_DS41_FAST_I4_SCOPE must be all or gu");
}
unsigned integer_prefill_band() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_BAND");
    return v && *v ? unsigned(std::strtoul(v, nullptr, 10)) : 4U;
}
template<WeightFormat F>
void launch_gate_up_core(const PrefillGateUpView& view, hipStream_t queue) {
    GateUpAdapter ad{{view.routes, view.codes, view.deltas, view.hidden, false}, view};
    const unsigned blocks = view.routes.max_batches * (view.routes.batch_rows == 32U ? 2U : 1U);
    if constexpr (F == WeightFormat::iq2_xxs) {
        if (fast_i4()) {
            if (fast_moe()) throw std::invalid_argument("FAST_I4 and FAST_MOE are distinct policies");
            (void)fast_i4_down();
            integer_core::launch<integer_core::Contract<F, true>, 64>(ad, blocks, view.inter / 32U, view.hidden, 2U * view.inter, queue);
            return;
        }
    }
    if (fast_moe() && F == WeightFormat::iq2_xxs)
        integer_core::launch<integer_core::Contract<F, false, true>, 64>(ad, blocks, view.inter / 32U, view.hidden, 2U * view.inter, queue);
    else {
        if constexpr (F != WeightFormat::mxfp4) ad.band = integer_prefill_band();
        // The grouped gate/up plan loads whole chunks (no partial-chunk path).
        if (F == WeightFormat::iq2_xxs && view.hidden % 1024U != 0U)
            throw std::invalid_argument("grouped IQ2_XXS gate/up needs a hidden width that is a multiple of 1024");
        integer_core::launch<integer_core::Contract<F>, 64>(ad, blocks, view.inter / 32U, view.hidden, 2U * view.inter, queue);
    }
}
template<WeightFormat F>
void launch_down_core(const PrefillDownView& view, hipStream_t queue) {
    DownAdapter ad{{view.routes, view.codes, view.deltas, view.width, true}, view};
    const unsigned blocks = view.routes.max_batches * (view.routes.batch_rows == 32U ? 2U : 1U);
    if constexpr (F == WeightFormat::q2_k) {
        if (fast_i4() && fast_i4_down()) {
            if (fast_moe()) throw std::invalid_argument("FAST_I4 and FAST_MOE are distinct policies");
            integer_core::launch<integer_core::Contract<F, true>, 32>(ad, blocks, view.outputs / 32U, view.width, view.outputs, queue);
            return;
        }
    }
    if (fast_moe() && F == WeightFormat::q2_k)
        integer_core::launch<integer_core::Contract<F, false, true>, 32>(ad, blocks, view.outputs / 32U, view.width, view.outputs, queue);
    else {
        if constexpr (F != WeightFormat::mxfp4) {
            ad.band = integer_prefill_band();
            integer_core::launch<integer_core::Contract<F, false, false, true>, 64>(ad, blocks, view.outputs / 64U, view.width, view.outputs, queue);
        } else integer_core::launch<integer_core::Contract<F>, 32>(ad, blocks, view.outputs / 32U, view.width, view.outputs, queue);
    }
}
