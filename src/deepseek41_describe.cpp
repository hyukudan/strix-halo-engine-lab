#include "strix/deepseek41_describe.hpp"

#include "strix/deepseek41_plan.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/ds41_pipeline_wire.hpp"
#include "strix/ds41_prefill.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace strix::deepseek41 {
namespace {

using serve::Json;

bool ends(const std::string& s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint64_t elements(const ManifestTensor& t) {
    std::uint64_t n = 1;
    for (const auto d : t.dimensions) n *= d;
    return n;
}

Json u64(std::uint64_t v) { return Json(v); }
Json strings(const std::vector<std::string>& v) {
    Json a = Json::array();
    for (const auto& s : v) a.push(Json(s));
    return a;
}

// Which part of its layer a tensor belongs to.
enum class Bucket { attn_split, attn_replicated, bridge, mix, engram_kv, engram_table, shared, experts };

Bucket bucket_of(const std::string& name, bool bridge_layer) {
    const auto base = name.substr(name.find('.', 4) + 1);  // after "blk.<l>."
    if (base == "ffn_gate_exps.weight" || base == "ffn_up_exps.weight" || base == "ffn_down_exps.weight") return Bucket::experts;
    if (ends(base, "_shexp.weight")) return Bucket::shared;
    if (base == "engram_embd.weight") return Bucket::engram_table;
    if (base.rfind("engram_", 0) == 0) return Bucket::engram_kv;
    if (base.rfind("hc_", 0) == 0 || base == "ffn_norm.weight" || base == "ffn_gate_inp.weight" || base.rfind("exp_probs_b", 0) == 0) return Bucket::mix;
    // The first decoder layer's KV-source path (compressor + index keys) runs on every prompt row: the bridge.
    if (bridge_layer && (base.rfind("attn_compressor_", 0) == 0 || base == "indexer.attn_k.weight" || base == "indexer.k_norm.weight"))
        return Bucket::bridge;
    // Tensor-parallel attention splits q_b (by heads), wo_a (by groups), wo_b (by K halves) and the per-head sinks.
    if (base == "attn_q_b.weight" || base == "attn_output_a.weight" || base == "attn_output_b.weight" || base == "attn_sinks.weight")
        return Bucket::attn_split;
    return Bucket::attn_replicated;
}

struct Sums {
    std::uint64_t bytes = 0, params = 0;
    void add(const ManifestTensor& t, bool matrix) {
        bytes += t.byte_size;
        if (matrix) params += elements(t);
    }
};

Json prefill_rows(std::uint64_t rows, std::uint64_t total) {
    if (rows >= total) return Json("all");
    if (rows <= 1U) return Json("last");
    Json o = Json::object();
    o.set("max", u64(rows));
    return o;
}

}  // namespace

std::vector<std::string> deepseek41_capabilities() {
    return {"ds41.attention", "ds41.core", "ds41.engram", "ds41.lookup", "ds41.moe-routed", "ds41.moe-shared"};
}

std::string model_revision(const std::string& file_name, const TargetModel& model) {
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(model.manifest.layout_identity));
    return file_name + "@" + hex;
}

Json model_parts(const TargetModel& model, const PartsOptions& options) {
    const auto& c = model.config;
    const std::uint32_t n = c.num_hidden_layers, H = c.hidden_size, K = c.num_experts_per_tok;
    if (n == 0U || model.layers.size() != n) throw std::invalid_argument("deepseek41 describe: model without layers");
    // the first decoder layer: the KV source every later layer reads (layer 20 in the released model)
    const auto first_decoder = static_cast<std::uint32_t>(std::max(model.layers[n - 1U].kv_source_layer, 0));
    const auto plan = build_decode_plan(model, 0, n);
    ds41::SuffixReplayGeometry geometry;
    geometry.layers = n;
    geometry.first_decoder = first_decoder;
    geometry.window = c.sliding_window;
    geometry.draft_rows = 0;  // the engine's prefill default (no drafter seed)
    constexpr std::uint64_t total = std::uint64_t{1} << 20U;
    const auto cone = ds41::plan_suffix_replay(geometry, 0, total);
    const std::uint64_t ring_bytes = std::uint64_t{c.sliding_window + 8U} * (c.head_dim + c.head_dim / 32U);  // AttentionHost::ring_capacity
    const std::uint64_t a8_in = H + H / 32U * 4U + std::uint64_t{K} * 8U;  // A8 codes + scales + expert ids/weights per row
    const std::uint64_t half_sum = std::uint64_t{H} * 4U;

    std::vector<std::map<Bucket, Sums>> per(n);
    std::uint64_t embed = 0, head = 0, head_params = 0;
    for (const auto& t : model.manifest.tensors) {
        if (t.name == "token_embd.weight") { embed += t.byte_size; continue; }
        if (t.name == "output.weight") { head += t.byte_size; head_params += elements(t); continue; }
        if (t.name == "output_norm.weight") { head += t.byte_size; continue; }
        if (!t.layer || *t.layer >= n) throw std::invalid_argument("deepseek41 describe: tensor outside the layers: " + t.name);
        const auto l = *t.layer;
        const bool matrix = t.dimensions.size() >= 2U;
        per[l][bucket_of(t.name, l == first_decoder)].add(t, matrix);
    }

    Json parts = Json::array();
    {
        Json p = Json::object();
        p.set("id", Json("embed"));
        p.set("kind", Json("embedding"));
        p.set("role", Json("lookup"));
        p.set("layer", Json(-1));
        p.set("weight_bytes", u64(embed));
        Json d = Json::object();
        d.set("read_bytes", u64(std::uint64_t{H} * 2U));
        p.set("decode", d);
        Json io = Json::object();
        io.set("out_bytes_per_row", u64(half_sum));
        p.set("io", io);
        p.set("requires", strings({"ds41.lookup"}));
        parts.push(p);
    }
    const auto attn_core_ops = [&](std::uint32_t l) {  // QK and PV over the window and the selected compressed positions
        const std::uint64_t positions = c.sliding_window + (model.layers[l].compress_ratio != 0U ? c.index_topk : 0U);
        return positions * c.num_attention_heads * c.head_dim * 4U;
    };
    for (std::uint32_t l = 0; l < n; ++l) {
        const auto& a = model.layers[l];
        auto& b = per[l];
        const auto rows = prefill_rows(total - cone.layers[l].out_begin, total);  // output rows (the K/V projection covers a window more)
        const std::string id = "L" + std::to_string(l);
        {
            const auto& sp = b[Bucket::attn_split];
            const auto& rp = b[Bucket::attn_replicated];
            const std::uint64_t bytes = sp.bytes + rp.bytes, params = sp.params + rp.params;
            Json p = Json::object();
            p.set("id", Json(id + ".attn"));
            p.set("kind", Json("attention"));
            p.set("role", Json("anchor"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(bytes));
            d.set("ops_per_row", u64(2U * params + attn_core_ops(l)));
            p.set("decode", d);
            Json f = Json::object();
            f.set("ops_per_row", u64(2U * params + attn_core_ops(l)));
            f.set("rows", rows);
            p.set("prefill", f);
            Json s = Json::object();
            const auto& st = plan.state[l];
            s.set("bytes_fixed", u64(ring_bytes + st.compress_carry_bytes));
            if (a.kv_source)
                s.set("bytes_per_token", u64((st.kv_bytes_per_position + st.index_key_bytes_per_position) / std::max(a.compress_ratio, 1U)));
            p.set("state", s);
            Json io = Json::object();
            io.set("out_bytes_per_row", u64(half_sum));
            p.set("io", io);
            p.set("shardable", Json(true));
            p.set("replicated_fraction", Json(bytes ? static_cast<double>(rp.bytes) / static_cast<double>(bytes) : 0.0));
            p.set("requires", strings({"ds41.attention"}));
            parts.push(p);
        }
        {
            const auto& m = b[Bucket::mix];
            Json p = Json::object();
            p.set("id", Json(id + ".mix"));
            p.set("kind", Json("router-mix"));
            p.set("role", Json("anchor"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(m.bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(m.bytes));
            d.set("ops_per_row", u64(2U * m.params));
            p.set("decode", d);
            Json f = Json::object();
            f.set("ops_per_row", u64(2U * m.params));
            f.set("rows", rows);
            p.set("prefill", f);
            p.set("requires", strings({"ds41.core"}));
            parts.push(p);
        }
        if (a.engram) {
            const auto& e = b[Bucket::engram_kv];
            Json p = Json::object();
            p.set("id", Json(id + ".engram_kv"));
            p.set("kind", Json("engram-projection"));
            p.set("role", Json("anchor"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(e.bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(e.bytes));
            d.set("ops_per_row", u64(2U * e.params));
            p.set("decode", d);
            Json f = Json::object();
            f.set("rows", rows);
            p.set("prefill", f);
            p.set("requires", strings({"ds41.engram"}));
            parts.push(p);
            const std::uint64_t row_bytes = std::uint64_t{model.engram.rows_per_token()} * model.engram.row_bytes;
            Json q = Json::object();
            q.set("id", Json(id + ".engram_table"));
            q.set("kind", Json("engram-table"));
            q.set("role", Json("lookup"));
            q.set("layer", u64(l));
            q.set("weight_bytes", u64(0));
            q.set("storage_bytes", u64(b[Bucket::engram_table].bytes));
            Json qd = Json::object();
            qd.set("read_bytes", u64(row_bytes));
            q.set("decode", qd);
            Json qf = Json::object();
            qf.set("rows", rows);
            q.set("prefill", qf);
            Json io = Json::object();
            io.set("out_bytes_per_row", u64(row_bytes));
            q.set("io", io);
            q.set("requires", strings({"ds41.lookup"}));
            parts.push(q);
        }
        if (l + 1U == first_decoder && first_decoder > 0U) {  // the next layer's KV-source path, every prompt row
            const auto& br = per[first_decoder][Bucket::bridge];
            const std::uint64_t mix_ops = 2U * std::uint64_t{H} * c.hc_mult * (2U + c.hc_mult) * c.hc_mult;  // hc_attn_fn of the next layer
            Json p = Json::object();
            p.set("id", Json(id + ".ced_bridge"));
            p.set("kind", Json("ced-bridge"));
            p.set("role", Json("anchor"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(br.bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(br.bytes));
            d.set("ops_per_row", u64(2U * br.params + mix_ops));
            p.set("decode", d);
            Json f = Json::object();
            f.set("ops_per_row", u64(2U * br.params + mix_ops));
            f.set("rows", Json("all"));
            p.set("prefill", f);
            p.set("requires", strings({"ds41.attention"}));
            parts.push(p);
        }
        {
            const auto& sh = b[Bucket::shared];
            Json p = Json::object();
            p.set("id", Json(id + ".shared"));
            p.set("kind", Json("shared-expert"));
            p.set("role", Json("branch"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(sh.bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(sh.bytes));
            d.set("ops_per_row", u64(2U * sh.params));
            p.set("decode", d);
            Json f = Json::object();
            f.set("rows", rows);
            p.set("prefill", f);
            Json io = Json::object();
            io.set("in_bytes_per_row", u64(a8_in));
            io.set("out_bytes_per_row", u64(half_sum));
            p.set("io", io);
            p.set("shardable", Json(true));
            p.set("requires", strings({"ds41.moe-shared"}));
            parts.push(p);
        }
        {
            const auto& ex = b[Bucket::experts];
            const std::uint64_t per_expert = ex.bytes / c.n_routed_experts, per_expert_params = ex.params / c.n_routed_experts;
            // an extra row of a step reads its experts not already read: K (1 - K / n_experts) on average (uniform routing)
            const auto extra = static_cast<std::uint64_t>(static_cast<double>(K * per_expert) * (1.0 - static_cast<double>(K) / c.n_routed_experts));
            Json p = Json::object();
            p.set("id", Json(id + ".experts"));
            p.set("kind", Json("routed-experts"));
            p.set("role", Json("branch"));
            p.set("layer", u64(l));
            p.set("weight_bytes", u64(ex.bytes));
            Json d = Json::object();
            d.set("read_bytes", u64(K * per_expert));
            d.set("read_bytes_extra_row", u64(extra));
            d.set("ops_per_row", u64(2U * K * per_expert_params));
            p.set("decode", d);
            Json f = Json::object();
            f.set("rows", rows);
            p.set("prefill", f);
            Json io = Json::object();
            io.set("in_bytes_per_row", u64(a8_in));
            io.set("out_bytes_per_row", u64(half_sum));
            io.set("combine", Json("per-shard"));
            p.set("io", io);
            p.set("shardable", Json(true));
            p.set("requires", strings({"ds41.moe-routed"}));
            parts.push(p);
        }
    }
    {
        Json p = Json::object();
        p.set("id", Json("head"));
        p.set("kind", Json("lm-head"));
        p.set("role", Json("head"));
        p.set("layer", u64(n));
        p.set("weight_bytes", u64(head));
        Json d = Json::object();
        d.set("read_bytes", u64(head));
        d.set("ops_per_row", u64(2U * head_params));
        p.set("decode", d);
        Json f = Json::object();
        f.set("rows", Json("last"));
        p.set("prefill", f);
        p.set("shardable", Json(true));
        p.set("requires", strings({"ds41.core"}));
        parts.push(p);
    }

    // Pipeline cuts: the boundary's side bytes per row (the stream is stream_bytes_per_row): the FFN mix of the layer
    // before the cut, the KV entries of the KV sources the later layers read (per token: entry bytes / ratio), the top-k
    // of the index sources they attend with (at most index_topk ids) and the candidate blocks at the nominal context.
    const auto g = ds41::pipe::boundary_geometry(c);
    Json cuts = Json::array();
    for (std::uint32_t cut = 1; cut < n; ++cut) {  // after_layer = cut - 1 <= n - 2
        const auto bp = ds41::pipe::plan_boundary(model.layers, cut);
        std::uint64_t side = std::uint64_t{g.mix_floats()} * 4U;
        for (const auto& s : bp.kv_sources) side += (g.kv_entry_bytes + g.idx_entry_bytes) / s.ratio;
        side += bp.topk_sources.size() * (1U + std::uint64_t{g.topk}) * 4U;
        if (bp.candidate_source >= 0) side += (options.nominal_context / bp.candidate_ratio + g.candidate_block - 1U) / g.candidate_block;
        Json cj = Json::object();
        cj.set("after_layer", u64(cut - 1U));
        cj.set("side_bytes_per_token", u64(side));
        cuts.push(cj);
    }

    Json out = Json::object();
    out.set("schema", Json("tensormeld/model-parts-v1"));
    Json m = Json::object();
    m.set("id", Json(options.model_id));
    m.set("revision", Json(options.revision));
    m.set("numerics", Json(options.numerics));
    m.set("family", Json("moe-ced-mhc"));
    out.set("model", m);
    out.set("provenance", Json("engine-describe"));
    out.set("layers", u64(n));
    out.set("stream_bytes_per_row", u64(g.stream_bytes_per_row()));
    out.set("token_bytes", u64(4));
    out.set("cuts", cuts);
    out.set("parts", parts);
    out.set("notes", Json("Generated by the engine from the model file header (no weight read): bytes are the GGUF tensors' bytes; ops are 2 x "
                          "the matrix parameters (+ QK/PV over the window and the selected positions for attention); prefill rows from the "
                          "exact suffix replay plan (cone, first decoder layer " + std::to_string(first_decoder) +
                          "); routed experts' extra-row bytes assume uniform routing; cut side bytes per row: FFN mix + KV entries / ratio "
                          "+ top-k (max) + candidate blocks at " + std::to_string(options.nominal_context) + " positions."));
    return out;
}

Json engine_describe(const TargetModel& model, const EngineDescribeOptions& o) {
    Json out = Json::object();
    out.set("schema", Json("tensormeld/engine-describe-v1"));
    Json e = Json::object();
    e.set("name", Json(o.engine_name));
    e.set("revision", Json(o.revision));
    e.set("executable_sha256", Json(o.executable_sha256));
    e.set("backend", Json(o.backend));
    out.set("engine", e);
    out.set("capabilities", strings(deepseek41_capabilities()));
    Json modes = Json::array();
    {
        Json g = Json::object();
        g.set("id", Json("ep2-tp"));
        g.set("world", Json(2));
        g.set("capabilities", strings(deepseek41_capabilities()));
        g.set("transports", strings({"usb4", "tcp"}));
        modes.push(g);
    }
    out.set("group_modes", modes);
    out.set("roles", strings(o.roles));
    if (o.state_export || o.state_import) {
        Json s = Json::object();
        s.set("export", Json(o.state_export));
        s.set("import", Json(o.state_import));
        s.set("format", Json("strix/ds41-attention-state-v1"));
        s.set("transports", strings(o.state_transports));
        s.set("exact", Json(true));
        out.set("state", s);
    }
    if (!o.argv_template.is_null()) out.set("argv_template", o.argv_template);
    Json models = Json::array();
    models.push(model_parts(model, o.parts));
    out.set("models", models);
    std::string notes = "pipeline wire " + std::string(ds41::pipe::kProtocol) + "; max_positions " + std::to_string(o.max_positions);
    if (!o.notes.empty()) notes += "; " + o.notes;
    out.set("notes", Json(notes));
    return out;
}

std::string source_revision() {
#ifdef STRIX_SOURCE_REVISION
    return STRIX_SOURCE_REVISION;
#else
    return "unknown";
#endif
}

std::string self_executable_sha256() { return model_content_identity_hex(sha256_file_content("/proc/self/exe")); }

std::string describe_engine_file(const std::filesystem::path& gguf, EngineDescribeOptions o) {
    const auto model = validate_target(read_file(gguf));
    if (o.revision.empty()) o.revision = source_revision();
    if (o.executable_sha256.empty()) o.executable_sha256 = self_executable_sha256();
    if (o.parts.revision.empty()) o.parts.revision = model_revision(gguf.filename().string(), model);
    return engine_describe(model, o).dump();
}

}  // namespace strix::deepseek41
