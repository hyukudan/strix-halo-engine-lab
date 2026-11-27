// Hy4-flash verify rows (device side), included INSIDE struct Forward::Impl (hy4_hip_forward.hip): the rows stages reuse
// the step's private helpers (linear, norm, engram, index_keys, index_select) and caches. Contract and host replay:
// hy4_hip_rows.hpp. No kernel of its own: every launch is the step's shared entry, with Rows = R where the entry's
// per-row arithmetic does not depend on the row count:
//  - Q8_0 Linears: shared q8_rows with a Rows = R Site (q8_rows::LinearRowsF32, the stock site of the step's LinearF32): every (output,
//    row) accumulator gets the same products in the same column order and the same lane / leaf tree for any Rows
//    (hip_q8_rows_gemv.hpp reduce: lane bits 16..1, leaf bits 2..0, whatever the pair count);
//  - absorb heads K_b / V_b: q8_rows launch_heads with a Rows = 2 head Site (launch_heads admits 1..2 rows; Outs 4 both);
//  - other Linears (router): launch_hip_dense_rows with rows = R (rows tile, w5b: bits equal; row_loop = the reference);
//  - row-independent kernels in one launch over R rows: HC mixes / collapse / expand, rmsnorm (one block per row), rope
//    (positions[r]), cache row stores (positions iota + p0), query images, route wave, swiglu, epilogues, combine;
//  - per row, in row order: rmsnorm of the kv latent (strided rows), indexer keys / select, flash at the row's own
//    position, routed experts (the C1 A8 probe / down admit one logical row), Engram (n-gram history up to the row).
// Cache stores of all R rows precede the rows' reads: legal because every ring is sized for the verify rows
// (rows::cache_rows -> layer_caches max_rows; rows::check_rewind replays it).

struct PinnedRows {  // pinned host buffer owned by the verify rows (freed after the stream: ~Forward syncs first)
    void* p = nullptr;
    PinnedRows() = default;
    PinnedRows(const PinnedRows&) = delete;
    PinnedRows& operator=(const PinnedRows&) = delete;
    ~PinnedRows() {
        if (p) (void)hipHostFree(p);
    }
    void* get(std::size_t bytes, unsigned flags) {
        check(hipHostMalloc(&p, std::max<std::size_t>(bytes, 16U), flags), "verify rows pinned staging");
        return p;
    }
};
struct VerifyRows {
    std::uint32_t max = 0;               // ForwardOptions::verify_rows (0: no rows forward)
    std::uint32_t p0 = 0, rows = 0;      // the last pass (rows 0: committed)
    float *x0 = nullptr, *x1 = nullptr, *u = nullptr, *xn = nullptr, *f = nullptr, *partial = nullptr;
    hcc::MixRow<kSub>* mix = nullptr;
    hcc::MixRow<kHead>* mixh = nullptr;
    float *gate = nullptr, *qa = nullptr, *qv = nullptr, *ckv = nullptr, *qfull = nullptr, *olat = nullptr, *o = nullptr;
    std::uint16_t* qt = nullptr;
    float *g = nullptr, *up = nullptr, *act = nullptr, *rlog = nullptr, *rw = nullptr, *z = nullptr, *rawdots = nullptr,
          *acc = nullptr, *routed = nullptr, *shared = nullptr, *logits = nullptr;
    std::uint32_t *ids = nullptr, *dpos = nullptr;
    // routed experts of the window in the shared A8 union rows form (strix/moe/expert_union.hpp): per-row A8 scratch + plan
    std::uint8_t *ucodes = nullptr, *udeltas = nullptr, *usums = nullptr, *upout = nullptr, *udcodes = nullptr,
                 *uddeltas = nullptr, *udsums = nullptr, *uplan = nullptr;
    // the window's routed experts in the union rows form (each distinct expert read once; bitwise == the per-row loop).
    // Default from STRIX_HY4_ROWS_EXPERT_UNION (kExpertUnionDefault when unset); Forward::set_rows_expert_union overrides.
    bool expert_union = false;
    std::uint8_t* greedy_partials = nullptr;
    std::uint64_t greedy_bytes = 0;
    std::size_t engram_row_bytes = 0;
    PinnedRows pos_pin, reply_pin, engram_pin;
    std::uint32_t* pos_host = nullptr;          // [max][4] position words
    strix::HipGreedyRowsReply* replies = nullptr;  // host-mapped coherent, [max]
    std::uint8_t* engram_host = nullptr;        // [max] x the step's Engram staging (one slot per ordinal)
    std::vector<float> logits_host;
    std::vector<std::uint32_t> greedy;
};
VerifyRows vr;

void allocate_verify() {
    const std::size_t R = vr.max, h = c.hidden, hmax = 64U * c.v_head, row = c.kv_lora + c.qk_rope, K = c.top_k, L = c.latent;
    const std::size_t qmax = 64U * std::size_t{c.qk_head()};
    vr.expert_union = strix::envcfg::flag("STRIX_HY4_ROWS_EXPERT_UNION", rows::kExpertUnionDefault);
    vr.x0 = alloc<float>(R * 4U * h); vr.x1 = alloc<float>(R * 4U * h);
    vr.u = alloc<float>(R * h); vr.xn = alloc<float>(R * h); vr.f = alloc<float>(R * h);
    vr.partial = alloc<float>(std::max(hcc::mix_partial_floats<kSub>(vr.max, c.hidden), hcc::mix_partial_floats<kHead>(vr.max, c.hidden)));
    vr.mix = alloc<hcc::MixRow<kSub>>(R); vr.mixh = alloc<hcc::MixRow<kHead>>(R);
    vr.gate = alloc<float>(R * hmax); vr.qa = alloc<float>(R * c.q_lora); vr.qv = alloc<float>(R * qmax);
    vr.ckv = alloc<float>(R * row); vr.qfull = alloc<float>(R * 64U * row); vr.qt = alloc<std::uint16_t>(R * 64U * row);
    vr.olat = alloc<float>(R * 64U * c.kv_lora); vr.o = alloc<float>(R * hmax);
    const std::size_t ff = std::max(c.dense_ff, c.shared_ff);
    vr.g = alloc<float>(R * ff); vr.up = alloc<float>(R * ff); vr.act = alloc<float>(R * ff);
    vr.rlog = alloc<float>(R * c.experts); vr.rw = alloc<float>(R * K); vr.ids = alloc<std::uint32_t>(R * K);
    vr.z = alloc<float>(R * L); vr.rawdots = alloc<float>(R * K * L); vr.acc = alloc<float>(R * L);
    {
        const std::size_t I = c.expert_ff;
        vr.ucodes = alloc<std::uint8_t>(R * L); vr.udeltas = alloc<std::uint8_t>(R * (L / 32U) * 4U);
        vr.usums = alloc<std::uint8_t>(R * (L / 32U) * 4U); vr.upout = alloc<std::uint8_t>(R * K * 2U * I * 4U);
        vr.udcodes = alloc<std::uint8_t>(R * K * I); vr.uddeltas = alloc<std::uint8_t>(R * K * (I / 32U) * 4U);
        vr.udsums = alloc<std::uint8_t>(R * K * (I / 32U) * 4U);
        vr.uplan = alloc<std::uint8_t>(strix::moe::expert_union::plan_bytes);
    }
    vr.routed = alloc<float>(R * h); vr.shared = alloc<float>(R * h);
    vr.dpos = alloc<std::uint32_t>(4U * R);
    vr.pos_host = static_cast<std::uint32_t*>(vr.pos_pin.get(4U * R * sizeof(std::uint32_t), hipHostMallocDefault));
    vr.engram_row_bytes = std::max<std::size_t>(1U, c.engram_layers.size()) * c.engram_heads() * c.engram_dim;
    vr.engram_host = static_cast<std::uint8_t*>(vr.engram_pin.get(R * vr.engram_row_bytes, hipHostMallocDefault));
    if (last == c.layers) {
        vr.logits = alloc<float>(R * c.vocab);
        vr.greedy_bytes = strix::hip_greedy_rows_scratch_bytes(vr.max, c.vocab);
        vr.greedy_partials = alloc<std::uint8_t>(vr.greedy_bytes);
        vr.replies = static_cast<strix::HipGreedyRowsReply*>(
            vr.reply_pin.get(R * sizeof(strix::HipGreedyRowsReply), hipHostMallocMapped | hipHostMallocCoherent));
    }
}

// The single-row stages that read Impl state (position, position words, xn, normalised q_a, Engram staging) run for
// verify row r; the previous state comes back at scope exit.
struct RowBind {
    Impl& m;
    std::uint32_t saved_pos;
    std::uint32_t* saved_dpos;
    float *saved_xn, *saved_qa;
    std::uint8_t* saved_stage;
    RowBind(Impl& impl, std::uint32_t r)
        : m(impl), saved_pos(impl.pos), saved_dpos(impl.dpos), saved_xn(impl.xn), saved_qa(impl.qa), saved_stage(impl.engram_stage) {
        impl.pos = impl.vr.p0 + r;
        impl.dpos = impl.vr.dpos + 4U * r;
        impl.xn = impl.vr.xn + std::size_t{r} * impl.c.hidden;
        impl.qa = impl.vr.qa + std::size_t{r} * impl.c.q_lora;
        impl.engram_stage = impl.vr.engram_host + r * impl.vr.engram_row_bytes;
    }
    RowBind(const RowBind&) = delete;
    RowBind& operator=(const RowBind&) = delete;
    ~RowBind() {
        m.pos = saved_pos;
        m.dpos = saved_dpos;
        m.xn = saved_xn;
        m.qa = saved_qa;
        m.engram_stage = saved_stage;
    }
};

template <unsigned R>
void q8_vrows(const std::uint8_t* weights, const float* x, std::uint32_t width, std::uint32_t outputs, float* y, bool round,
              const float* bias) {
    strix::q8_rows::LinearRowsF32<R> site{};
    for (unsigned r = 0; r < R; ++r) site.input[r] = x + std::size_t{r} * width;
    site.segments = strix::q8_rows::segments({{weights, outputs}});
    site.width = width;
    site.outputs = outputs;
    site.out = y;
    site.bias = bias;
    site.round = round && store_epilogue;
    strix::q8_rows::launch(site, false, s);
}
// linear() over R rows x[R][width] -> y[R][outputs] (tight rows); R == 1 is linear() itself.
void linear_vrows(const std::string& name, const float* x, std::uint32_t width, std::uint32_t outputs, std::uint32_t R, float* y,
                  bool round = true, const float* bias = nullptr) {
    if (R == 1U) {
        linear(name, x, width, outputs, y, round, bias);
        return;
    }
    const auto& tw = t(name);
    // The FAST head copy (head_q8) is the step's decode head (linear("output.weight")): the verify rows read the same Q8_0
    // rows through the shared q8 rows form, so a row stays bitwise the step at that position with the copy on too.
    if (head_q8 != nullptr && name == "output.weight") {
        if (tw.dims.at(0) != width) throw std::runtime_error("hy4 hip forward: shape of " + name);
        linear_vrows_w(head_q8, 8U, x, width, outputs, R, y, round, bias);
        return;
    }
    if (tw.dims.at(0) != width || tw.bytes != row_bytes(tw.type, width) * outputs)
        throw std::runtime_error("hy4 hip forward: shape of " + name);
    linear_vrows_w(tw.p, tw.type, x, width, outputs, R, y, round, bias);
}
// The output rows [r.first, r.first + r.count) of a Linear over R rows -> y[R][r.count] (tight): a tensor-parallel share.
// Every output's arithmetic is independent of the range (the step's linear_rows over R rows).
void linear_vrows_range(const std::string& name, const float* x, std::uint32_t width, std::uint32_t outputs, htp::Range r,
                        std::uint32_t R, float* y) {
    const auto& tw = t(name);
    if (tw.dims.at(0) != width || tw.bytes != row_bytes(tw.type, width) * outputs || r.first + r.count > outputs)
        throw std::runtime_error("hy4 hip forward: shape of " + name);
    const std::uint8_t* const wptr = tw.p + row_bytes(tw.type, width) * r.first;
    const auto n = static_cast<std::uint32_t>(r.count);
    if (R == 1U) linear(wptr, tw.type, x, width, n, y, true);
    else linear_vrows_w(wptr, tw.type, x, width, n, R, y, true, nullptr);
}
// linear_vrows over a weight pointer (R >= 2): rows of `outputs` weight rows of GGUF `type`.
void linear_vrows_w(const std::uint8_t* weights, std::uint32_t type, const float* x, std::uint32_t width, std::uint32_t outputs,
                    std::uint32_t R, float* y, bool round, const float* bias) {
    const std::size_t total = std::size_t{R} * outputs;
    if (type == 8U && width % 32U == 0U) {
        switch (R) {
        case 2U: q8_vrows<2U>(weights, x, width, outputs, y, round, bias); break;
        case 3U: q8_vrows<3U>(weights, x, width, outputs, y, round, bias); break;
        case 4U: q8_vrows<4U>(weights, x, width, outputs, y, round, bias); break;
        case 5U: q8_vrows<5U>(weights, x, width, outputs, y, round, bias); break;
        case 6U: q8_vrows<6U>(weights, x, width, outputs, y, round, bias); break;
        case 7U: q8_vrows<7U>(weights, x, width, outputs, y, round, bias); break;
        case 8U: q8_vrows<8U>(weights, x, width, outputs, y, round, bias); break;
        default: throw std::invalid_argument("hy4 hip forward: verify rows outside 1..8");
        }
        if (round && !store_epilogue) glu::launch_epilogue<glu::Round::bf16>(y, bias, outputs, total, y, s);
        check(hipGetLastError(), "q8 verify rows launch");
        return;
    }
    HipDenseRowsView v;
    v.x = x;
    v.input = HipActivationStorage::fp32;
    v.weights = weights;
    v.ggml_type = type;
    v.width = width;
    v.outputs = outputs;
    v.rows = R;
    v.out = y;
    launch_hip_dense_rows(v, s);
    if (round) glu::launch_epilogue<glu::Round::bf16>(y, bias, outputs, total, y, s);
    check(hipGetLastError(), "linear verify rows launch");
}
// launch_heads_f32() over R rows: pairs of rows per launch (launch_heads admits 1..2 rows), the odd row alone.
void absorb_vrows(const std::uint8_t* wt, std::uint32_t heads, std::uint32_t width, std::uint32_t head_outputs, const float* x,
                  std::uint32_t x_stride, std::size_t x_row, float* y, std::uint32_t y_stride, std::size_t y_row, std::uint32_t R) {
    for (std::uint32_t r = 0; r < R; r += 2U) {
        if (R - r < 2U) {
            strix::q8_rows::launch_heads_f32(wt, heads, width, head_outputs, x + r * x_row, x_stride, y + r * y_row, y_stride, s);
            break;
        }
        strix::q8_rows::HeadsRowsF32<2U> site{};
        site.input[0] = x + r * x_row;
        site.input[1] = x + (r + 1U) * x_row;
        site.segments = strix::q8_rows::segments({{wt, heads * head_outputs}});
        site.width = width;
        site.outputs = heads * head_outputs;
        site.head_outputs = head_outputs;
        site.input_stride = x_stride;
        site.out_stride = y_stride;
        site.row_stride = y_row;
        site.out = y + r * y_row;
        strix::q8_rows::launch_heads(site, s);
    }
}

// The cache side of attention_vrows: rows vr.p0 .. vr.p0 + R - 1 of layer l's latent (kv_a -> kv norm -> rope -> store) and,
// indexed, its token / group keys, from vr.xn alone (never from the rows' attention or FFN output). attention_vrows calls
// it at this point of its launch order; nextn_rows calls it alone for the rows whose outputs the caller never reads.
void cache_vrows(std::uint32_t l, std::uint32_t R) {  // vr.xn -> latent[l] (+ tkeys[l] / gkeys[l])
    namespace rp = strix::ops::rope;
    namespace dcr = strix::attn_core::cache_rows;
    const std::uint32_t kl = c.kv_lora, rope = c.qk_rope, row = kl + rope;
    const bool idx = indexed_of(l);
    const std::uint32_t* positions = iota + vr.p0;
    linear_vrows(blk(l, "attn_kv_a_mqa.weight"), vr.xn, c.hidden, row, R, vr.ckv);
    const float* kvn = t(blk(l, "attn_kv_a_norm.weight")).f();
    for (std::uint32_t r = 0; r < R; ++r) norm(vr.ckv + std::size_t{r} * row, kvn, kl, c.rms_eps, vr.ckv + std::size_t{r} * row);
    rp::launch<rp::Round::bf16>({vr.ckv, R, 1U, row, row, kl, rope / 2U, idx ? cos_idx : cos_swa, idx ? sin_idx : sin_swa, positions, false}, s);
    check(dcr::store_rows({vr.ckv, row, positions, 0U, R, latent_rows[l], latent[l], flag}, s), "latent verify rows store");
    if (idx && !dfly_block_id(l))  // a DFly2 block has no indexer: it never writes the KV-share layer's keys
        for (std::uint32_t r = 0; r < R; ++r) {
            const RowBind bind(*this, r);
            index_keys(l, false);  // verify rows: no shared same-input projection ran, project here
        }
}

// Text layers and the next-token block (l = c.layers, the rows form of nextn_rows): heads_of / indexed_of read its form.
void attention_vrows(std::uint32_t l, std::uint32_t R) {  // vr.xn -> vr.f
    namespace rp = strix::ops::rope;
    namespace dcr = strix::attn_core::cache_rows;
    const std::uint32_t H = heads_of(l), qk = c.qk_head(), kl = c.kv_lora, rope = c.qk_rope, vd = c.v_head, h = c.hidden;
    const bool idx = indexed_of(l);
    if (idx ? H != 32U : H != 64U) throw std::runtime_error("hy4 hip forward: head count outside the flash presets");
    const std::uint32_t row = kl + rope;
    const std::size_t qrow = std::size_t{H} * qk, orow = std::size_t{H} * vd, frow = std::size_t{H} * row, lrow = std::size_t{H} * kl;
    const std::uint32_t* positions = iota + vr.p0;  // positions[r] = p0 + r
    linear_vrows(blk(l, "attn_gate.weight"), vr.xn, h, H * vd, R, vr.gate);
    linear_vrows(blk(l, "attn_q_a.weight"), vr.xn, h, c.q_lora, R, vr.qa);
    norm_rows(vr.qa, t(blk(l, "attn_q_a_norm.weight")).f(), c.q_lora, R, vr.qa);
    linear_vrows(blk(l, "attn_q_b.weight"), vr.qa, c.q_lora, H * qk, R, vr.qv);
    const float* cs = idx ? cos_idx : cos_swa;
    const float* sn = idx ? sin_idx : sin_swa;
    rp::launch<rp::Round::bf16>({vr.qv, R, H, H * qk, qk, c.qk_nope, rope / 2U, cs, sn, positions, false}, s);
    cache_vrows(l, R);
    const auto& kb = t(blk(l, "attn_k_b.weight"));
    if (kb.type == 8U) absorb_vrows(kb.p, H, c.qk_nope, kl, vr.qv, qk, qrow, vr.qfull, row, frow, R);
    else {  // one grouped launch per row over the H head slices (the step's form; bitwise == one linear() per head)
        const auto kb_head = row_bytes(kb.type, c.qk_nope) * kl;
        for (std::uint32_t r = 0; r < R; ++r)
            linear_groups(kb.p, kb.type, kb_head, H, nullptr, H, vr.qv + r * qrow, qk, c.qk_nope, kl, vr.qfull + r * frow, row);
    }
    check(hipMemcpy2DAsync(vr.qfull + kl, row * 4U, vr.qv + c.qk_nope, qk * 4U, rope * 4U, std::size_t{H} * R, hipMemcpyDeviceToDevice, s),
          "q rope copy rows");
    check(dcr::store_rows({vr.qfull, row, nullptr, 0U, H * R, {row, 0U}, vr.qt, flag}, s), "query operand images");
    const float* sinks = t(blk(l, "attn_sinks.weight")).f();
    // Flash: one launch over the rows that need no per-row selection (vrows_flash_split), as forward_rows does. The core
    // runs one work-group per (row, head group) and a work-group reads only its own row's query, list and count, so a
    // row's output bits do not depend on the grid; the batched launch lets the rows' 1-2 work-groups run side by side
    // instead of one launch after another (41 R flash launches per pass -> 41 + the selecting rows).
    const VrowsFlashSplit fs = vrows_flash_split(idx, vr.p0, R);
    if (fs.batched != 0U) {
        if (idx) {  // the causal prefix: list = iota (stride 0), length of row r = iota[p0 + 1 + r] = p0 + r + 1
            DsaPolicy::Args a{latent[l], iota, iota + vr.p0 + 1U, 0U, 0U, 0U, 0U, vr.qt, sinks, vr.olat};
            strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, fs.batched * DsaSpec::kGroups, s, vr.p0 + fs.batched);
        } else {  // window rows: row r at position p0 + r
            SwaPolicy::Args a{latent[l], nullptr, nullptr, 0U, vr.p0, c.window, latent_rows[l].ring - 1U, vr.qt, sinks, vr.olat};
            strix::attn_core::sparse::launch_flash_rows<SwaPolicy, 32U>(a, fs.batched * SwaSpec::kGroups, s,
                                                                        std::min(vr.p0 + fs.batched, c.window));
        }
        check(hipGetLastError(), "flash verify rows launch");
    }
    for (std::uint32_t r = fs.batched; r < R; ++r) {  // selecting rows: index_select fills the one shared list per row
        {
            const RowBind bind(*this, r);
            index_select(l);
        }
        DsaPolicy::Args a{latent[l], ilist, list_n, kIndexListStride, 0U, 0U, 0U, vr.qt + r * frow, sinks, vr.olat + r * lrow};
        strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, DsaSpec::kGroups, s, index_list_tokens(vr.p0 + r));
        check(hipGetLastError(), "flash verify row launch");
    }
    const auto& vb = t(blk(l, "attn_v_b.weight"));
    if (vb.type == 8U) absorb_vrows(vb.p, H, kl, vd, vr.olat, kl, lrow, vr.o, vd, orow, R);
    else {  // one grouped launch per row (the step's form)
        const auto vb_head = row_bytes(vb.type, kl) * vd;
        for (std::uint32_t r = 0; r < R; ++r)
            linear_groups(vb.p, vb.type, vb_head, H, nullptr, H, vr.olat + r * lrow, kl, kl, vd, vr.o + r * orow, vd);
    }
    glu::launch_epilogue<glu::Round::bf16>(vr.o, nullptr, H * vd, R * orow, vr.o, s);
    glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(vr.o, vr.gate, R * orow, vr.o, s);
    linear_vrows(blk(l, "attn_output.weight"), vr.o, H * vd, h, R, vr.f);
    norm_rows(vr.f, t(blk(l, "post_attention_norm.weight")).f(), h, R, vr.f);
    check(hipGetLastError(), "attention verify rows launch");
}

// The step's routed experts for one row: A8 probe (gate + up) and down raw dots of the row's top-k -> raw[K][latent].
void expert_row(std::uint32_t l, const float* z_r, const std::uint32_t* ids_r, float* raw_r, std::uint32_t kslots = 0U) {
    const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
    const auto& wu = t(blk(l, "ffn_up_exps.weight"));
    const auto& wd = t(blk(l, "ffn_down_exps.weight"));
    std::uint32_t* const ps = dstatus_all + 8U * std::size_t{l};
    std::uint32_t* const ds = ps + 4;
    const auto dev = [](const void* p, std::uint64_t bytes) { return a8::DeviceBytes{reinterpret_cast<std::uintptr_t>(p), bytes}; };
    const std::uint32_t nk = kslots != 0U ? kslots : c.top_k;  // a TP rank's share: its top_k / 2 slots
    const std::uint64_t K = nk, I = c.expert_ff, L = c.latent;
    a8::Request rq;
    rq.enabled = true;
    rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(wg.type);
    rq.input = a8::InputKind::fp32;
    rq.shape = {c.latent, c.expert_ff, c.experts, nk, 1U};
    rq.gate_type = wg.type;
    rq.up_type = wu.type;
    a8::Bindings b{dev(z_r, L * 4U), dev(wg.p, wg.bytes), dev(wu.p, wu.bytes), dev(ids_r, K * 4U), dev(pcodes, L),
                   dev(pdeltas, L / 32U * 4U), dev(psums, L / 32U * 4U), dev(ps, 4U), dev(pout, K * 2U * I * 4U)};
    a8::enqueue_probe(rq, b, s);
    a8::DownRequest dq;
    dq.enabled = true;
    dq.approved_numerical_identity = a8::down_numerical_identity(wd.type);
    dq.shape = {c.latent, c.expert_ff, c.experts, nk};
    dq.weight_type = wd.type;
    dq.swiglu_limit = c.swiglu_limit;
    a8::DownBindings db{dev(pout, K * 2U * I * 4U), dev(wd.p, wd.bytes), dev(ids_r, K * 4U), dev(dcodes, K * I),
                        dev(ddeltas, K * I / 32U * 4U), dev(dsums, K * I / 32U * 4U), dev(raw_r, K * L * 4U), dev(ds, 4U), {0U, 0U}};
    a8::enqueue_down_activation_quantize(dq, db, s);
    a8::enqueue_down_rawdots(dq, db, s);
}

// The window's routed experts in the shared union rows form: every distinct expert of the R rows is read once (Q4_K gate / up,
// K-quant down) and the rows that chose it are computed with their own A8 codes: raw[r][K][latent] bitwise == expert_row(r)
// for every r (tests/hip_moe_a8_union_tests.hip); 5 launches per layer instead of 5 x R.
void expert_rows_union(std::uint32_t l, std::uint32_t R, const std::uint32_t* row_ids = nullptr, std::uint32_t kslots = 0U,
                       float* raw_out = nullptr) {  // defaults: vr.ids [R][top_k] -> vr.rawdots; a TP share passes its slots
    const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
    const auto& wu = t(blk(l, "ffn_up_exps.weight"));
    const auto& wd = t(blk(l, "ffn_down_exps.weight"));
    std::uint32_t* const ps = dstatus_all + 8U * std::size_t{l};
    std::uint32_t* const ds = ps + 4;
    const auto dev = [](const void* p, std::uint64_t bytes) { return a8::DeviceBytes{reinterpret_cast<std::uintptr_t>(p), bytes}; };
    const std::uint32_t nk = kslots != 0U ? kslots : c.top_k;
    const std::uint32_t* const sid = row_ids != nullptr ? row_ids : vr.ids;
    float* const raw = raw_out != nullptr ? raw_out : vr.rawdots;
    const std::uint64_t K = nk, I = c.expert_ff, L = c.latent, n = R;
    const a8::DeviceBytes plan = dev(vr.uplan, strix::moe::expert_union::plan_bytes);
    a8::Request rq;
    rq.enabled = true;
    rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(wg.type);
    rq.input = a8::InputKind::fp32;
    rq.shape = {c.latent, c.expert_ff, c.experts, nk, R};
    rq.gate_type = wg.type;
    rq.up_type = wu.type;
    a8::Bindings b{dev(vr.z, n * L * 4U), dev(wg.p, wg.bytes), dev(wu.p, wu.bytes), dev(sid, n * K * 4U), dev(vr.ucodes, n * L),
                   dev(vr.udeltas, n * (L / 32U) * 4U), dev(vr.usums, n * (L / 32U) * 4U), dev(ps, 4U), dev(vr.upout, n * K * 2U * I * 4U)};
    a8::enqueue_probe_union(rq, b, plan, s);
    a8::DownRequest dq;
    dq.enabled = true;
    dq.approved_numerical_identity = a8::down_numerical_identity(wd.type);
    dq.shape = {c.latent, c.expert_ff, c.experts, nk};
    dq.weight_type = wd.type;
    dq.swiglu_limit = c.swiglu_limit;
    a8::DownBindings db{dev(vr.upout, n * K * 2U * I * 4U), dev(wd.p, wd.bytes), dev(sid, n * K * 4U), dev(vr.udcodes, n * K * I),
                        dev(vr.uddeltas, n * K * (I / 32U) * 4U), dev(vr.udsums, n * K * (I / 32U) * 4U),
                        dev(raw, n * K * L * 4U), dev(ds, 4U), {0U, 0U}};
    a8::enqueue_down_activation_quantize_union(dq, db, R, plan, s);
    a8::enqueue_down_rawdots_union(dq, db, R, plan, s);
}

void ffn_vrows(std::uint32_t l, std::uint32_t R) {  // vr.xn -> vr.f
    const std::uint32_t h = c.hidden;
    if (dense_of(l)) {
        linear_vrows(blk(l, "ffn_gate.weight"), vr.xn, h, c.dense_ff, R, vr.g);
        linear_vrows(blk(l, "ffn_up.weight"), vr.xn, h, c.dense_ff, R, vr.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(vr.g, vr.up, std::size_t{R} * c.dense_ff, glu::no_clamp, vr.act, s);
        linear_vrows(blk(l, "ffn_down.weight"), vr.act, c.dense_ff, h, R, vr.f);
    } else {
        const std::size_t K = c.top_k, L = c.latent;
        linear_vrows(blk(l, "ffn_gate_inp.weight"), vr.xn, h, c.experts, R, vr.rlog, false);
        strix::moe_core::enqueue_route_wave<preset::route, glu::DeclaredExp>(vr.rlog, t(blk(l, "exp_probs_b.bias")).f(), vr.ids, vr.rw, R, s);
        strix::moe_core::route_tap(l, R, c.top_k, c.experts, vr.ids, vr.rw, s);
        linear_vrows(blk(l, "ffn_routed_down.weight"), vr.xn, h, c.latent, R, vr.z);
        const bool kquant_down = t(blk(l, "ffn_down_exps.weight")).type == 12U || t(blk(l, "ffn_down_exps.weight")).type == 14U;
        const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
        if (wg.type == kTypeBf16)  // the next-token block: each row's BF16 slices by the grouped dense rows (the step's form)
            for (std::uint32_t r = 0; r < R; ++r)
                experts_dense(wg, t(blk(l, "ffn_up_exps.weight")), t(blk(l, "ffn_down_exps.weight")), vr.z + r * L, vr.ids + r * K,
                              vr.rawdots + r * K * L);
        else if (vr.expert_union && R > 1U && wg.type == 12U && kquant_down)
            expert_rows_union(l, R);
        else
            for (std::uint32_t r = 0; r < R; ++r) expert_row(l, vr.z + r * L, vr.ids + r * K, vr.rawdots + r * K * L);
        glu::launch_epilogue<glu::Round::bf16>(vr.rawdots, nullptr, L, R * K * L, vr.rawdots, s);  // each expert's down output
        strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 16U, 1536U>(vr.rw, vr.rawdots, nullptr, vr.acc, R, s);
        linear_vrows(blk(l, "ffn_routed_up.weight"), vr.acc, c.latent, h, R, vr.routed);
        linear_vrows(blk(l, "ffn_gate_shexp.weight"), vr.xn, h, c.shared_ff, R, vr.g);
        linear_vrows(blk(l, "ffn_up_shexp.weight"), vr.xn, h, c.shared_ff, R, vr.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(vr.g, vr.up, std::size_t{R} * c.shared_ff, glu::no_clamp, vr.act, s);
        linear_vrows(blk(l, "ffn_down_shexp.weight"), vr.act, c.shared_ff, h, R, vr.shared);
        strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 1U, 3072U>(nullptr, vr.routed, vr.shared, vr.f, R, s);
    }
    norm_rows(vr.f, t(blk(l, "post_ffw_norm.weight")).f(), h, R, vr.f);
    check(hipGetLastError(), "ffn verify rows launch");
}

// ---- verify rows under tensor parallelism (Forward::enable_tp): the step's shares (attention() / ffn()) over R rows ----
// Each split result is written tight into this rank's slot of a tp.rows swap buffer ([R][count]: slot q = rank q's share),
// the shared gather swaps the two slots, and two 2D copies place them into the one-node [R][n] rows (slot q -> columns
// [q count, (q + 1) count), the order of htp::half). Every split output is a whole per-output / per-head / per-slot result
// of the same entry as the one-node rows form (Q8_0 / dense rows over an output range, absorb / flash over a head range,
// the A8 probe / down over a slot range), the joins are concatenations and the combine reads the 16 slot rows in slot
// order: bitwise the one-node pass. Replicated as in the step: q_a, the latent / indexer cache rows, the router, routed
// down / up, shared down, HC, norms, Engram, and here also the head (its logits join would move R x vocab / 2 x 4 B over
// the link, about what splitting the BF16 head saves).
float* tp_rows_mine(std::size_t slot_floats) const {
    return tp.rows + std::size_t{tp.flip} * 2U * tp.rows_slot + std::size_t{tp.rank} * slot_floats;
}
// The swap of a join. Every verify join carries BF16-rounded shares (rounded Linear rows, the BF16 gate / GLU / expert
// epilogues), so with tp.bf16 this rank's slot crosses as high halves (tp_join's wire; a word that is not a bfloat16
// raises the flag and the pass refuses) and the place widens them back: the FP32 words of the FP32 swap.
void tp_rows_swap(std::size_t slot_floats) {
    if (slot_floats > tp.rows_slot) throw std::logic_error("hy4 hip forward: tp verify rows join larger than its slot");
    float* const base = tp.rows + std::size_t{tp.flip} * 2U * tp.rows_slot;
    if (!tp.bf16) return tp_gather(base, slot_floats * 4U);
    tp.rows_wire = tp_wire(slot_floats);
    const std::size_t mine = std::size_t{tp.rank} * slot_floats;
    check(xbw::enqueue_pack(base + mine, tp.rows_wire + mine, slot_floats, flag, s), "tp verify rows join pack");
    tp_gather(tp.rows_wire, bf16_wire::wire_bytes(slot_floats));
}
// slot q's rows [R][count] at element `at` (row pitch `pitch`) of the swapped buffer -> full[R][n] columns q count ..
void tp_rows_place(float* full, std::size_t cols, std::uint32_t R, std::size_t count, std::size_t slot_floats, std::size_t at,
                   std::size_t pitch) {
    if (tp.bf16) {
        check(xbw::enqueue_place({tp.rows_wire, slot_floats, at, pitch, count, R, full, cols}, s), "tp verify rows join widen");
        return;
    }
    const float* const base = tp.rows + std::size_t{tp.flip} * 2U * tp.rows_slot + at;
    for (std::size_t q = 0; q < 2U; ++q)
        check(hipMemcpy2DAsync(full + q * count, cols * 4U, base + q * slot_floats, pitch * 4U, count * 4U, R, hipMemcpyDeviceToDevice, s),
              "tp verify rows join");
}
void tp_rows_next() noexcept { tp.flip ^= 1U; }  // the other buffer for the next join

void attention_vrows_tp(std::uint32_t l, std::uint32_t R) {  // vr.xn -> vr.f, this rank's heads
    namespace rp = strix::ops::rope;
    namespace dcr = strix::attn_core::cache_rows;
    const std::uint32_t H = heads_of(l), qk = c.qk_head(), kl = c.kv_lora, rope = c.qk_rope, vd = c.v_head, h = c.hidden;
    const bool idx = indexed_of(l);
    if (idx ? H != 32U : H != 64U) throw std::runtime_error("hy4 hip forward: head count outside the flash presets");
    const std::uint32_t row = kl + rope;
    const auto hs = share(H);
    const auto h0 = static_cast<std::uint32_t>(hs.first), Hs = static_cast<std::uint32_t>(hs.count);
    // this rank's heads, tight per row: q [R][Hs qk], query image [R][Hs row], latent out [R][Hs kl], o [R][Hs vd]
    const std::size_t qs = std::size_t{Hs} * qk, fs = std::size_t{Hs} * row, ls = std::size_t{Hs} * kl, os = std::size_t{Hs} * vd;
    const std::uint32_t* positions = iota + vr.p0;
    linear_vrows_range(blk(l, "attn_gate.weight"), vr.xn, h, H * vd, {std::size_t{h0} * vd, os}, R, vr.gate);
    linear_vrows(blk(l, "attn_q_a.weight"), vr.xn, h, c.q_lora, R, vr.qa);
    norm_rows(vr.qa, t(blk(l, "attn_q_a_norm.weight")).f(), c.q_lora, R, vr.qa);
    linear_vrows_range(blk(l, "attn_q_b.weight"), vr.qa, c.q_lora, H * qk, {std::size_t{h0} * qk, qs}, R, vr.qv);
    const float* cs = idx ? cos_idx : cos_swa;
    const float* sn = idx ? sin_idx : sin_swa;
    rp::launch<rp::Round::bf16>({vr.qv, R, Hs, Hs * qk, qk, c.qk_nope, rope / 2U, cs, sn, positions, false}, s);
    cache_vrows(l, R);
    const auto& kb = t(blk(l, "attn_k_b.weight"));
    const auto kb_head = row_bytes(kb.type, c.qk_nope) * kl;
    if (kb.type == 8U) absorb_vrows(kb.p + h0 * kb_head, Hs, c.qk_nope, kl, vr.qv, qk, qs, vr.qfull, row, fs, R);
    else
        for (std::uint32_t r = 0; r < R; ++r)
            linear_groups(kb.p + h0 * kb_head, kb.type, kb_head, Hs, nullptr, Hs, vr.qv + r * qs, qk, c.qk_nope, kl, vr.qfull + r * fs, row);
    check(hipMemcpy2DAsync(vr.qfull + kl, row * 4U, vr.qv + c.qk_nope, qk * 4U, rope * 4U, std::size_t{Hs} * R, hipMemcpyDeviceToDevice, s),
          "q rope copy rows (tp)");
    check(dcr::store_rows({vr.qfull, row, nullptr, 0U, Hs * R, {row, 0U}, vr.qt, flag}, s), "query operand images (tp)");
    const float* sinks = t(blk(l, "attn_sinks.weight")).f() + h0;
    // this rank's head group (one 16-head tile indexed / one of the two 32-head groups sliding): ONE launch over the rows
    // that need no per-row selection (attention_vrows' split; row = token of the tight [R][Hs][row] image), then the
    // selecting rows one launch each after their index_select. A work-group reads only its own row's query, list and
    // count, so a row's bits do not depend on the grid (the per-row launches it replaces: R x layers -> layers).
    const VrowsFlashSplit split = vrows_flash_split(idx, vr.p0, R);
    if (split.batched != 0U) {
        if (idx) {  // the causal prefix: list = iota (stride 0), length of row r = iota[p0 + 1 + r] = p0 + r + 1
            DsaHalfPolicy::Args a{latent[l], iota, iota + vr.p0 + 1U, 0U, 0U, 0U, 0U, vr.qt, sinks, vr.olat};
            strix::attn_core::sparse::launch_flash_rows<DsaHalfPolicy, 32U>(a, split.batched, s, vr.p0 + split.batched);
        } else {  // window rows: row r at position p0 + r
            SwaHalfPolicy::Args a{latent[l], nullptr, nullptr, 0U, vr.p0, c.window, latent_rows[l].ring - 1U, vr.qt, sinks, vr.olat};
            strix::attn_core::sparse::launch_flash_rows<SwaHalfPolicy, 32U>(a, split.batched, s, std::min(vr.p0 + split.batched, c.window));
        }
        check(hipGetLastError(), "flash verify rows launch (tp)");
    }
    for (std::uint32_t r = split.batched; r < R; ++r) {  // indexed rows that select: the one shared list per row
        {
            const RowBind bind(*this, r);
            index_select(l);
        }
        DsaHalfPolicy::Args a{latent[l], ilist, list_n, kIndexListStride, 0U, 0U, 0U, vr.qt + r * fs, sinks, vr.olat + r * ls};
        strix::attn_core::sparse::launch_flash_rows<DsaHalfPolicy, 32U>(a, 1U, s, index_list_tokens(vr.p0 + r));
        check(hipGetLastError(), "flash verify row launch (tp)");
    }
    const auto& vb = t(blk(l, "attn_v_b.weight"));
    const auto vb_head = row_bytes(vb.type, kl) * vd;
    const std::size_t so = std::size_t{R} * os;
    float* const om = tp_rows_mine(so);
    if (vb.type == 8U) absorb_vrows(vb.p + h0 * vb_head, Hs, kl, vd, vr.olat, kl, ls, om, vd, os, R);
    else
        for (std::uint32_t r = 0; r < R; ++r)
            linear_groups(vb.p + h0 * vb_head, vb.type, vb_head, Hs, nullptr, Hs, vr.olat + r * ls, kl, kl, vd, om + r * os, vd);
    glu::launch_epilogue<glu::Round::bf16>(om, nullptr, Hs * vd, so, om, s);
    glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(om, vr.gate, so, om, s);
    tp_rows_swap(so);  // o = [rank 0 heads | rank 1 heads] per row: the one-node o rows
    tp_rows_place(vr.o, std::size_t{H} * vd, R, os, so, 0U, os);
    tp_rows_next();
    const auto fr = share(h);
    const std::size_t sf = std::size_t{R} * fr.count;
    float* const fm = tp_rows_mine(sf);
    linear_vrows_range(blk(l, "attn_output.weight"), vr.o, H * vd, h, fr, R, fm);
    tp_rows_swap(sf);
    tp_rows_place(vr.f, h, R, fr.count, sf, 0U, fr.count);
    tp_rows_next();
    norm_rows(vr.f, t(blk(l, "post_attention_norm.weight")).f(), h, R, vr.f);
    check(hipGetLastError(), "attention verify rows launch (tp)");
}

void ffn_vrows_tp(std::uint32_t l, std::uint32_t R) {  // vr.xn -> vr.f, this rank's rows / slots
    const std::uint32_t h = c.hidden;
    if (dense_of(l)) {
        const auto fr = share(c.dense_ff), hr = share(h);
        const std::size_t sa = std::size_t{R} * fr.count;
        float* const am = tp_rows_mine(sa);
        linear_vrows_range(blk(l, "ffn_gate.weight"), vr.xn, h, c.dense_ff, fr, R, vr.g);
        linear_vrows_range(blk(l, "ffn_up.weight"), vr.xn, h, c.dense_ff, fr, R, vr.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(vr.g, vr.up, sa, glu::no_clamp, am, s);
        tp_rows_swap(sa);
        tp_rows_place(vr.act, c.dense_ff, R, fr.count, sa, 0U, fr.count);
        tp_rows_next();
        const std::size_t sf = std::size_t{R} * hr.count;
        float* const fm = tp_rows_mine(sf);
        linear_vrows_range(blk(l, "ffn_down.weight"), vr.act, c.dense_ff, h, hr, R, fm);
        tp_rows_swap(sf);
        tp_rows_place(vr.f, h, R, hr.count, sf, 0U, hr.count);
        tp_rows_next();
    } else {
        const std::size_t K = c.top_k, L = c.latent;
        linear_vrows(blk(l, "ffn_gate_inp.weight"), vr.xn, h, c.experts, R, vr.rlog, false);
        strix::moe_core::enqueue_route_wave<preset::route, glu::DeclaredExp>(vr.rlog, t(blk(l, "exp_probs_b.bias")).f(), vr.ids, vr.rw, R, s);
        linear_vrows(blk(l, "ffn_routed_down.weight"), vr.xn, h, c.latent, R, vr.z);
        const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
        const auto& wdt = t(blk(l, "ffn_down_exps.weight"));
        const bool kquant_down = wdt.type == 12U || wdt.type == 14U;
        // this rank's slots (htp::moe: top_k / 2 from slot_first) and shared-expert activations, one join per layer:
        // slot = [R][Ks][L] routed slot rows | [R][shared_ff / 2] activations
        const htp::MoeLayout m = htp::moe(tp.rank, c.top_k, c.latent, c.shared_ff);
        const auto Ks = static_cast<std::uint32_t>(m.slots);
        const auto sr = share(c.shared_ff);
        const std::size_t raw_n = std::size_t{R} * Ks * L, sa = raw_n + std::size_t{R} * sr.count;
        float* const mine = tp_rows_mine(sa);
        check(hipMemcpy2DAsync(tp.tids, Ks * 4U, vr.ids + m.slot_first, K * 4U, Ks * 4U, R, hipMemcpyDeviceToDevice, s), "tp slot ids");
        if (wg.type == kTypeBf16)  // the next-token block (drafter under TP): each row's 8 BF16 slots, the step's form
            for (std::uint32_t r = 0; r < R; ++r)
                experts_dense(wg, t(blk(l, "ffn_up_exps.weight")), wdt, vr.z + r * L, tp.tids + r * Ks, mine + std::size_t{r} * Ks * L, Ks);
        else if (vr.expert_union && R > 1U && wg.type == 12U && kquant_down)
            expert_rows_union(l, R, tp.tids, Ks, mine);
        else
            for (std::uint32_t r = 0; r < R; ++r) expert_row(l, vr.z + r * L, tp.tids + r * Ks, mine + std::size_t{r} * Ks * L, Ks);
        glu::launch_epilogue<glu::Round::bf16>(mine, nullptr, L, raw_n, mine, s);  // each expert's down output
        linear_vrows_range(blk(l, "ffn_gate_shexp.weight"), vr.xn, h, c.shared_ff, sr, R, vr.g);
        linear_vrows_range(blk(l, "ffn_up_shexp.weight"), vr.xn, h, c.shared_ff, sr, R, vr.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(vr.g, vr.up, std::size_t{R} * sr.count, glu::no_clamp, mine + raw_n, s);
        tp_rows_swap(sa);
        tp_rows_place(vr.rawdots, K * L, R, std::size_t{Ks} * L, sa, 0U, std::size_t{Ks} * L);  // 16 slot rows per row, slot order
        tp_rows_place(vr.act, c.shared_ff, R, sr.count, sa, raw_n, sr.count);
        tp_rows_next();
        strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 16U, 1536U>(vr.rw, vr.rawdots, nullptr, vr.acc, R, s);
        linear_vrows(blk(l, "ffn_routed_up.weight"), vr.acc, c.latent, h, R, vr.routed);
        linear_vrows(blk(l, "ffn_down_shexp.weight"), vr.act, c.shared_ff, h, R, vr.shared);
        strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 1U, 3072U>(nullptr, vr.routed, vr.shared, vr.f, R, s);
    }
    norm_rows(vr.f, t(blk(l, "post_ffw_norm.weight")).f(), h, R, vr.f);
    check(hipGetLastError(), "ffn verify rows launch (tp)");
}

template <class Sub>
void site_vrows(std::uint32_t l, const char* prefix, const char* norm_name, std::uint32_t R, Sub&& sub) {
    const std::string p(prefix);
    check(hcc::enqueue_mixes<kSub>(vr.x0, t(blk(l, (p + "_fn.weight").c_str())).f(), t(blk(l, (p + "_base.weight").c_str())).f(),
                                   t(blk(l, (p + "_scale.weight").c_str())).f(), R, c.hidden, 1U, c.hc_eps, c.rms_eps, vr.mix, vr.partial, s),
          "hc mixes verify rows");
    check(hcc::enqueue_collapse<kSub>(vr.x0, vr.mix, R, c.hidden, vr.u, s), "hc collapse verify rows");
    norm_rows(vr.u, t(blk(l, norm_name)).f(), c.hidden, R, vr.xn);
    sub();
    check(hcc::enqueue_expand<kSub>(hcc::SubSum{vr.f, nullptr, nullptr}, vr.x0, vr.mix, R, c.hidden, vr.x1, s), "hc expand verify rows");
    std::swap(vr.x0, vr.x1);
}

// One verify pass: rows r = 0 .. R-1 = tokens[r] at positions pos + r; greedy id per row (vr.greedy), logits rows on the
// host when asked (vr.logits_host [R][vocab]). Synchronises. commit_rows(k) must follow before the next pass / step.
void forward_rows(std::span<const std::uint32_t> toks, bool want_logits) {
    const auto R = static_cast<std::uint32_t>(toks.size());
    if (vr.max == 0U) throw std::logic_error("hy4 hip forward: forward_rows needs ForwardOptions::verify_rows");
    if (vr.rows != 0U) throw std::logic_error("hy4 hip forward: forward_rows before commit_rows of the previous pass");
    if (R == 0U || R > vr.max) throw std::invalid_argument("hy4 hip forward: verify rows outside 1 .. verify_rows");
    if (first != 0U || last != c.layers) throw std::logic_error("hy4 hip forward: forward_rows needs the whole model on this rank");
    if (pos + R > context) throw std::runtime_error("hy4 hip forward: context full");
    for (const auto tk : toks)
        if (tk >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
    vr.p0 = pos;
    vr.rows = R;
    tokens.insert(tokens.end(), toks.begin(), toks.end());
    times = {};
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t r = 0; r < R; ++r) {
        const auto words = rows::position_words(vr.p0 + r, kIdxRatio);
        std::copy(words.begin(), words.end(), vr.pos_host + 4U * r);
    }
    check(hipMemcpyAsync(vr.dpos, vr.pos_host, 16U * R, hipMemcpyHostToDevice, s), "verify position words upload");
    check(hipMemsetAsync(dstatus_all, 0, 8U * c.layers * 4U, s), "expert status clear");
    const std::size_t n = 4U * std::size_t{c.hidden};
    {
        const auto& te = t("token_embd.weight");
        std::vector<std::uint32_t> four(4U * std::size_t{R});
        for (std::size_t i = 0; i < four.size(); ++i) four[i] = toks[i / 4U];
        launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, four, vr.x0, c.hidden, s);
        glu::launch_epilogue<glu::Round::bf16>(vr.x0, nullptr, n * R, n * R, vr.x0, s);  // the BF16 embedding module's output
    }
    for (std::uint32_t l = first; l < last; ++l) {
        const auto found = std::find(c.engram_layers.begin(), c.engram_layers.end(), l);
        if (found != c.engram_layers.end())
            for (std::uint32_t r = 0; r < R; ++r) {
                const RowBind bind(*this, r);
                engram(l, static_cast<std::uint32_t>(found - c.engram_layers.begin()), vr.x0 + r * n);
            }
        if (tp.world == 2U) {  // Forward::enable_tp: both ranks run this pass with their shares
            site_vrows(l, "hc_attn", "attn_norm.weight", R, [&] { attention_vrows_tp(l, R); });
            site_vrows(l, "hc_ffn", "ffn_norm.weight", R, [&] { ffn_vrows_tp(l, R); });
        } else {
            site_vrows(l, "hc_attn", "attn_norm.weight", R, [&] { attention_vrows(l, R); });
            site_vrows(l, "hc_ffn", "ffn_norm.weight", R, [&] { ffn_vrows(l, R); });
        }
    }
    pos = vr.p0 + R;
    check(hcc::enqueue_mixes<kHead>(vr.x0, t("output_hc_fn.weight").f(), t("output_hc_base.weight").f(), t("output_hc_scale.weight").f(), R,
                                    c.hidden, 1U, c.hc_eps, c.rms_eps, vr.mixh, vr.partial, s),
          "head mixes verify rows");
    check(hcc::enqueue_collapse<kHead>(vr.x0, vr.mixh, R, c.hidden, vr.u, s), "head collapse verify rows");
    norm_rows(vr.u, t("output_norm.weight").f(), c.hidden, R, vr.xn);
    linear_vrows("output.weight", vr.xn, c.hidden, c.vocab, R, vr.logits, false);
    std::array<strix::ExecutionBufferAddress, rows::kMaxVerifyRows> lg{};
    for (std::uint32_t r = 0; r < R; ++r)
        lg[r] = {{}, reinterpret_cast<std::uintptr_t>(vr.logits + std::size_t{r} * c.vocab), std::uint64_t{c.vocab} * 4U, 0U};
    const strix::ExecutionBufferAddress pa{{}, reinterpret_cast<std::uintptr_t>(vr.greedy_partials), vr.greedy_bytes, 0U};
    strix::enqueue_hip_greedy_rows(std::span<const strix::ExecutionBufferAddress>(lg.data(), R), c.vocab, pa, vr.replies, s);
    if (want_logits) {
        vr.logits_host.resize(std::size_t{R} * c.vocab);
        check(hipMemcpyAsync(vr.logits_host.data(), vr.logits, vr.logits_host.size() * 4U, hipMemcpyDeviceToHost, s), "verify logits read");
    } else vr.logits_host.clear();
    std::uint32_t bad = 0;
    check(hipMemcpyAsync(&bad, flag, 4, hipMemcpyDeviceToHost, s), "flag read");
    check(hipMemcpyAsync(stage->status, dstatus_all, 8U * c.layers * 4U, hipMemcpyDeviceToHost, s), "expert status read");
    check(hipStreamSynchronize(s), "verify rows sync");
    times.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    for (std::uint32_t l = first; l < last; ++l)
        if (stage->status[8U * l] != 0U || stage->status[8U * l + 4U] != 0U)
            throw std::runtime_error("hy4 hip forward: verify rows layer " + std::to_string(l) + " expert status probe " +
                                     std::to_string(stage->status[8U * l]) + " down " + std::to_string(stage->status[8U * l + 4U]));
    if (bad & bf16_wire::kInexactFlag) throw std::runtime_error("hy4 hip forward: a TP join value is not a bfloat16 (verify rows, exact wire refused)");
    if (bad) throw std::runtime_error("hy4 hip forward: non-finite value in a cache / operand row (verify rows)");
    vr.greedy.resize(R);
    for (std::uint32_t r = 0; r < R; ++r) {
        if (vr.replies[r].status != 0U) throw std::runtime_error("hy4 hip forward: non-finite logit (verify rows greedy status)");
        vr.greedy[r] = vr.replies[r].token;
    }
}
// Keep the first k rows of the last pass (0 <= k <= rows): position p0 + k, token history cut there. The rejected rows'
// cache rows are rewritten by the next pass / step at their positions before any read (rings sized for the verify rows).
void commit_rows(std::uint32_t k) {
    if (vr.rows == 0U) throw std::logic_error("hy4 hip forward: commit_rows without a verify pass");
    if (k > vr.rows) throw std::invalid_argument("hy4 hip forward: commit_rows beyond the pass");
    pos = vr.p0 + k;
    tokens.resize(pos);
    vr.rows = 0U;
}
