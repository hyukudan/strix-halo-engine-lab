// ---- DFly2 block drafter (Forward::dfly_block; host oracle hy4/ref/hy4_dfly_ref.cpp draft, hc=chain, cand=256) ---------
// Included inside Forward::Impl after the verify rows (hy4_hip_rows.inc): the draft layers ARE target DSA layers, so draft
// layer i runs as block df.first + i through the target's own rows forms (attention_vrows: gate, q_a, sinks, post norm;
// norm_rows; the q8 / dense rows), with its latent rows aliasing the target cache of the KV-share layer (spec::hy4_dfly2()
// target_layers): the block's own kv rows land at positions p .. p + k of that cache (the verify pass rewrites them before
// any read; the drafter never writes the target's indexer keys). Index Share below the indexer budget = the causal prefix
// (vrows_flash_split batched rows); a block whose rows would select refuses (the retained selection is not kept).
// Model numerics: BF16-rounded Linear / norm / residual rows as the target's rows forms; the drafter's bits change which
// tokens are proposed, never an emitted id (the target verifies greedily).
struct Dfly {
    bool on = false;
    std::uint32_t first = 0, layers = 0, ff = 0, fill = 0, cand = 256, budget = 0;
    std::array<std::uint32_t, spec::max_draft_layers> share{};
    float *x = nullptr, *y = nullptr, *cat = nullptr, *eh = nullptr;  // [rows][h] residual, [rows][h] final rows, [2h], [h]
    float *g = nullptr, *up = nullptr, *act = nullptr;                // [rows][ff] (ff >= hidden: the correction reuses them)
    float *hid = nullptr, *e = nullptr, *hc = nullptr;                // [h] target hidden of p - 1, E(prev), corrected row
    std::uint16_t* cand_rows = nullptr;                               // [cand][h] BF16 head rows of the candidates
    const std::uint8_t* rank_head = nullptr;                          // Q8_0 head the uncorrected ranking reads
    float* scores = nullptr;                                          // [cand]
    PinnedRows logits_pin, scores_pin;
    float *logits_host = nullptr, *scores_host = nullptr;             // [rows - 1][vocab], [cand]
    std::vector<std::uint32_t> cand_ids;
    std::uint64_t bytes = 0, blocks = 0;
    double ms = 0.0;
} df;

// Load (after allocate_verify): the shard packed as blocks df.first + i (hy4/ref/hy4_dfly_pack.hpp), uploaded once.
void allocate_dfly(const std::filesystem::path& shard, const std::string& format) {
    namespace dp = strix::hy4::dfly;
    if (first != 0U || last != c.layers) throw std::invalid_argument("hy4 hip forward: the DFly2 drafter runs on a whole-model rank");
    if (vr.max < 2U) throw std::invalid_argument("hy4 hip forward: the DFly2 drafter needs verify rows >= 2 (its block rows)");
    const auto policy = spec::hy4_dfly2();
    const auto geo = dp::Geometry::official();
    if (geo.hidden != c.hidden || geo.heads != 32U || geo.kv_lora != c.kv_lora || geo.nope != c.qk_nope || geo.rope != c.qk_rope ||
        geo.v_head != c.v_head || geo.q_lora != c.q_lora || policy.draft_layers != geo.layers)
        throw std::runtime_error("hy4 hip forward: the DFly2 geometry is not this pack's DSA layer");
    df.first = c.layers + 1U;
    df.layers = geo.layers;
    df.ff = geo.ff;
    df.fill = policy.fill_token;
    df.budget = policy.share_budget;
    df.share = {};
    for (std::uint32_t i = 0; i < geo.layers; ++i) {
        df.share[i] = policy.target_layers[i];
        if (df.share[i] >= c.layers || !indexed_of(df.share[i]) || heads_of(df.share[i]) != 32U)
            throw std::runtime_error("hy4 hip forward: DFly2 KV-share layer is not an indexed 32-head layer of this pack");
    }
    {
        auto weights = dp::shard_weights(shard);
        const auto pack = dp::build_pack(geo, *weights, df.first, dp::parse_pack_format(format));
        for (const auto& pt : pack.tensors) {
            auto* d = alloc<std::uint8_t>(pt.bytes.size());
            check(hipMemcpy(d, pt.bytes.data(), pt.bytes.size(), hipMemcpyHostToDevice), "dfly weights upload");
            w[pt.name] = Tensor{d, pt.ggml_type, pt.dims, pt.bytes.size()};
        }
        df.bytes = pack.total_bytes();
    }
    // the blocks' per-layer entries: latent rows alias the KV-share layer's cache (no own cache, no indexer keys)
    while (latent.size() < df.first) {  // no next-token block: an empty slot at c.layers
        latent_rows.push_back({});
        latent.push_back(nullptr);
        tk_rows.push_back({});
        tkeys.push_back(nullptr);
        gkeys.push_back(nullptr);
    }
    for (std::uint32_t i = 0; i < geo.layers; ++i) {
        latent_rows.push_back(latent_rows[df.share[i]]);
        latent.push_back(latent[df.share[i]]);
        tk_rows.push_back({});
        tkeys.push_back(nullptr);
        gkeys.push_back(nullptr);
    }
    const std::size_t R = vr.max, h = c.hidden, ffw = std::max<std::size_t>(df.ff, h);
    df.x = alloc<float>(R * h); df.y = alloc<float>(R * h); df.cat = alloc<float>(2U * h); df.eh = alloc<float>(h);
    df.g = alloc<float>(R * ffw); df.up = alloc<float>(R * ffw); df.act = alloc<float>(R * ffw);
    df.hid = alloc<float>(h); df.e = alloc<float>(h); df.hc = alloc<float>(h);
    df.cand_rows = alloc<std::uint16_t>(std::size_t{df.cand} * h);
    df.scores = alloc<float>(df.cand);
    df.logits_host = static_cast<float*>(df.logits_pin.get((R - 1U) * std::size_t{c.vocab} * sizeof(float), hipHostMallocDefault));
    df.scores_host = static_cast<float*>(df.scores_pin.get(std::size_t{df.cand} * sizeof(float), hipHostMallocDefault));
    // The uncorrected ranking only chooses the 256 candidates (the corrected rescoring reads the head's own rows): it reads
    // a Q8_0 head, the FAST decode copy when present, else a drafter-only copy (shared transcode, ggml Q8_0 rule). Drafter
    // numerics change acceptance only, so EXACT keeps its BF16 decode head and still drafts from the Q8_0 copy.
    if (head_q8 != nullptr) df.rank_head = head_q8;
    else {
        const auto& tw = t("output.weight");
        if (tw.type == 8U) df.rank_head = tw.p;
        else {
            if (tw.type != 30U || tw.dims.size() != 2U || tw.dims[0] % 32U) throw std::runtime_error("hy4 dfly: BF16 [K % 32] head");
            const auto K = static_cast<std::uint32_t>(tw.dims[0]), N = static_cast<std::uint32_t>(tw.dims[1]);
            auto* d = alloc<std::uint8_t>(row_bytes(8U, K) * N);
            check(strix::core::convert_bf16_q8_0(reinterpret_cast<const std::uint16_t*>(tw.p), static_cast<int>(K), static_cast<int>(N), d, s),
                  "dfly rank head q8 convert");
            check(hipStreamSynchronize(s), "dfly rank head q8 convert sync");
            df.rank_head = d;
            df.bytes += row_bytes(8U, K) * N;
        }
    }
    df.on = true;
    report.queue_line += "dfly2 drafter " + format + " blocks " + std::to_string(df.first) + ".." +
                         std::to_string(df.first + df.layers - 1U) + " GiB=" + std::to_string(static_cast<double>(df.bytes) / 1073741824.0) + "\n";
}
[[nodiscard]] bool dfly_block_id(std::uint32_t l) const noexcept { return df.on && l >= df.first && l < df.first + df.layers; }

// The candidates of one uncorrected row: the df.cand best ids (value desc., the lower id on ties), canonical order.
void dfly_candidates(const float* row) {
    const std::uint32_t n = c.vocab, N = df.cand;
    const auto better = [row](std::uint32_t a, std::uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); };
    auto& kept = df.cand_ids;
    kept.resize(N);
    for (std::uint32_t t = 0; t < N; ++t) kept[t] = t;
    std::make_heap(kept.begin(), kept.end(), better);  // heap top = the worst kept id
    for (std::uint32_t t = N; t < n; ++t)
        if (better(t, kept.front())) {
            std::pop_heap(kept.begin(), kept.end(), better);
            kept.back() = t;
            std::push_heap(kept.begin(), kept.end(), better);
        }
}

// One block: rows r = 0 .. k at positions p .. p + k (row 0 = the anchor, the others = the fill id); df.hid = the target
// hidden (output_norm) of position p - 1. Writes the k proposals (tokens p + 1 .. p + k). Synchronises.
// dfly_block's position guards as a query (no throw): rows >= 2 within the verify rows, inside the context and below the
// index-share budget (the causal prefix: every row dense).
bool dfly_room(std::uint64_t p, std::uint32_t R) const noexcept {
    if (!df.on || R < 2U || R > vr.max || p == 0U || p + R > context || p + R > df.budget) return false;
    return index_dense_rows(static_cast<std::uint32_t>(p), R) == R;
}
void dfly_block(std::uint32_t anchor, std::uint32_t p, std::uint32_t k, std::uint32_t* out) {
    if (!df.on) throw std::logic_error("hy4 hip forward: dfly_block needs ForwardOptions::dfly");
    const std::uint32_t R = k + 1U, h = c.hidden;
    if (k == 0U || R > vr.max) throw std::invalid_argument("hy4 hip forward: DFly2 block rows outside 2 .. verify rows");
    if (p == 0U || p + R > context) throw std::invalid_argument("hy4 hip forward: DFly2 block position");
    if (index_dense_rows(p, R) != R || p + R > df.budget)
        throw std::runtime_error("hy4 hip forward: DFly2 block beyond the index-share budget (the target's selection is not retained)");
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t nh = std::size_t{R} * h;
    vr.p0 = p;
    std::array<std::uint32_t, rows::kMaxVerifyRows> block{};
    block[0] = anchor;
    for (std::uint32_t r = 1; r < R; ++r) block[r] = df.fill;
    const auto& te = t("token_embd.weight");
    launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, std::span<const std::uint32_t>(block.data(), R),
                              df.x, c.hidden, s);
    glu::launch_epilogue<glu::Round::bf16>(df.x, nullptr, nh, nh, df.x, s);  // e: BF16 values
    // slot-0 fusion: x0 = e0 + gate * eh_proj(cat(enorm(e0), hnorm(h))) (the gate folded into eh_proj's rows at pack time)
    norm(df.x, t("dfly.enorm").f(), h, c.rms_eps, df.cat);
    norm(df.hid, t("dfly.hnorm").f(), h, c.rms_eps, df.cat + h);
    linear("dfly.eh_proj", df.cat, 2U * h, h, df.eh);
    glu::launch_epilogue<glu::Round::bf16>(df.x, df.eh, h, h, df.x, s);
    for (std::uint32_t i = 0; i < df.layers; ++i) {
        const std::uint32_t l = df.first + i;
        norm_rows(df.x, t(blk(l, "attn_norm.weight")).f(), h, R, vr.xn);
        attention_vrows(l, R);                                                    // vr.xn -> vr.f (post_proj_norm)
        glu::launch_epilogue<glu::Round::bf16>(df.x, vr.f, nh, nh, df.x, s);     // x = bf(x + attn)
        norm_rows(df.x, t(blk(l, "ffn_norm.weight")).f(), h, R, vr.xn);
        linear_vrows(blk(l, "ffn_gate.weight"), vr.xn, h, df.ff, R, df.g);
        linear_vrows(blk(l, "ffn_up.weight"), vr.xn, h, df.ff, R, df.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(df.g, df.up, std::size_t{R} * df.ff, glu::no_clamp, df.act, s);
        linear_vrows(blk(l, "ffn_down.weight"), df.act, df.ff, h, R, vr.f);     // dense SwiGLU, no post-FFN norm
        glu::launch_epilogue<glu::Round::bf16>(df.x, vr.f, nh, nh, df.x, s);     // x = bf(x + ffn)
    }
    norm_rows(df.x + h, t("dfly.final_norm").f(), h, k, df.y);                  // rows 1 .. k: y_j
    if (k >= 2U) linear_vrows_w(df.rank_head, 8U, df.y, h, c.vocab, k, vr.logits, false, nullptr);  // the uncorrected ranking, once
    else {  // one row: the dense rows entry (the q8 rows form starts at 2 rows)
        HipDenseRowsView v;
        v.x = df.y;
        v.input = HipActivationStorage::fp32;
        v.weights = df.rank_head;
        v.ggml_type = 8U;
        v.width = h;
        v.outputs = c.vocab;
        v.out = vr.logits;
        launch_hip_dense_rows(v, s);
    }
    check(hipMemcpyAsync(df.logits_host, vr.logits, std::size_t{k} * c.vocab * sizeof(float), hipMemcpyDeviceToHost, s), "dfly logits");
    check(hipStreamSynchronize(s), "dfly logits sync");
    // the hidden_correction chain: slot j reads y_j' = y_j + C(E(prev), y_j), prev = the anchor, then the slot j-1 pick
    const auto& head = t("output.weight");
    const HipEmbeddingTable head_rows = head_q8 != nullptr ? HipEmbeddingTable{head_q8, 8U, c.hidden, c.vocab}
                                                           : HipEmbeddingTable{head.p, head.type, c.hidden, c.vocab};
    std::uint32_t prev = anchor;
    for (std::uint32_t j = 0; j < k; ++j) {
        const float* yj = df.y + std::size_t{j} * h;
        launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, std::span<const std::uint32_t>(&prev, 1),
                                  df.e, c.hidden, s);
        glu::launch_epilogue<glu::Round::bf16>(df.e, nullptr, h, h, df.e, s);
        norm(yj, t("dfly.hc_hnorm").f(), h, c.rms_eps, df.cat);
        norm(df.e, t("dfly.hc_enorm").f(), h, c.rms_eps, df.cat + h);
        linear("dfly.hc_gate", df.cat, 2U * h, h, df.g);
        linear("dfly.hc_up", df.cat, 2U * h, h, df.up);
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(df.g, df.up, h, glu::no_clamp, df.act, s);
        linear("dfly.hc_down", df.act, h, h, df.hc);
        glu::launch_epilogue<glu::Round::bf16>(df.hc, yj, h, h, df.hc, s);       // y_j' = bf(y_j + C)
        dfly_candidates(df.logits_host + std::size_t{j} * c.vocab);               // host, overlaps the launches above
        launch_hip_embedding_rows(head_rows, HipActivationStorage::bf16, df.cand_ids, df.cand_rows, c.hidden, s);
        HipDenseRowsView v;
        v.x = df.hc;
        v.input = HipActivationStorage::fp32;
        v.weights = df.cand_rows;
        v.ggml_type = 30U;
        v.width = h;
        v.outputs = df.cand;
        v.out = df.scores;
        launch_hip_dense_rows(v, s);
        check(hipMemcpyAsync(df.scores_host, df.scores, std::size_t{df.cand} * sizeof(float), hipMemcpyDeviceToHost, s), "dfly scores");
        check(hipStreamSynchronize(s), "dfly scores sync");
        std::uint32_t best = 0;  // corrected argmax among the candidates, the lower id on ties
        for (std::uint32_t q = 1; q < df.cand; ++q) {
            const float a = df.scores_host[q], b = df.scores_host[best];
            if (a > b || (a == b && df.cand_ids[q] < df.cand_ids[best])) best = q;
        }
        prev = df.cand_ids[best];
        out[j] = prev;
    }
    ++df.blocks;
    df.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
