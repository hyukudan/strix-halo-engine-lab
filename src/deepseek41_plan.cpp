#include "strix/deepseek41_plan.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

namespace strix::deepseek41 {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("deepseek41 plan: " + message); }

// Window ring entry: 512 E4M3 bytes + one UE8M0 scale per 32; compressed KV entry: 256 E2M1 bytes + one E4M3
// scale per 16; index key: 64 E2M1 bytes + one E8M0 scale per 32.
std::uint64_t ring_entry_bytes(const Config& c) { return c.head_dim + c.head_dim / 32U; }
std::uint64_t kv_entry_bytes(const Config& c) { return c.head_dim / 2U + c.head_dim / 16U; }
std::uint64_t index_key_bytes(const Config& c) { return c.index_head_dim / 2U + c.index_head_dim / 32U; }

// The weight storage type an operation expects (checked against the manifest).
std::uint32_t expected_type(Op op, const std::string& name) {
    const auto ends = [&](std::string_view suffix) {
        return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    switch (op) {
    case Op::q_a: case Op::q_b: case Op::kv: case Op::o_a: case Op::o_b: case Op::shared_expert: return 8;  // Q8_0
    case Op::routed_experts: return ends("ffn_down_exps.weight") ? 10U : 16U;                               // Q2_K / IQ2_XXS
    case Op::embed: return 1;                                                                                  // F16
    case Op::head: return ends("output.weight") ? 8U : 0U;
    case Op::engram: return ends("engram_embd.weight") ? 24U : ends("engram_kv.weight") ? 1U : 0U;
    case Op::hc_mix: return ends("_fn.weight") ? 1U : 0U;
    case Op::compress: return ends("_norm.weight") ? 0U : 1U;
    case Op::index: return 1;
    default: return 0;  // F32 norms, sinks, router
    }
}

}  // namespace

std::string_view to_string(Op op) noexcept {
    switch (op) {
    case Op::embed: return "embed";
    case Op::engram: return "engram";
    case Op::hc_mix: return "hc_mix";
    case Op::hc_pre: return "hc_pre";
    case Op::rms_norm: return "rms_norm";
    case Op::q_a: return "q_a";
    case Op::q_b: return "q_b";
    case Op::kv: return "kv";
    case Op::window_write: return "window_write";
    case Op::compress: return "compress";
    case Op::index: return "index";
    case Op::attention: return "attention";
    case Op::o_a: return "o_a";
    case Op::o_b: return "o_b";
    case Op::hc_post: return "hc_post";
    case Op::router: return "router";
    case Op::shared_expert: return "shared_expert";
    case Op::routed_experts: return "routed_experts";
    case Op::combine: return "combine";
    case Op::head: return "head";
    case Op::argmax: return "argmax";
    }
    return "unknown";
}

std::string_view to_string(Owner owner) noexcept {
    switch (owner) {
    case Owner::formats_moe: return "formats_moe";
    case Owner::attention: return "attention";
    case Owner::layer: return "layer";
    case Owner::integration: return "integration";
    }
    return "unknown";
}

std::string_view to_string(Buffer buffer) noexcept {
    switch (buffer) {
    case Buffer::stream: return "stream";
    case Buffer::mix_attn: return "mix_attn";
    case Buffer::mix_ffn: return "mix_ffn";
    case Buffer::pre_carry: return "pre_carry";
    case Buffer::x: return "x";
    case Buffer::qr: return "qr";
    case Buffer::q: return "q";
    case Buffer::kvrow: return "kvrow";
    case Buffer::o: return "o";
    case Buffer::o_low: return "o_low";
    case Buffer::y: return "y";
    case Buffer::shared: return "shared";
    case Buffer::route_ids: return "route_ids";
    case Buffer::route_w: return "route_w";
    case Buffer::raw: return "raw";
    case Buffer::logits: return "logits";
    case Buffer::token: return "token";
    }
    return "unknown";
}

std::size_t Plan::buffer_elements(Buffer buffer) const noexcept {
    const std::size_t H = config.hidden_size, hc = config.hc_mult;
    switch (buffer) {
    case Buffer::stream: return hc * H;
    case Buffer::mix_attn: case Buffer::mix_ffn: return 2U * hc + hc * hc;
    case Buffer::pre_carry: return hc;
    case Buffer::x: case Buffer::y: case Buffer::shared: return H;
    case Buffer::qr: return config.q_lora_rank;
    case Buffer::q: case Buffer::o: return std::size_t{config.num_attention_heads} * config.head_dim;
    case Buffer::kvrow: return config.head_dim;
    case Buffer::o_low: return std::size_t{config.o_groups} * config.o_lora_rank;
    case Buffer::route_ids: case Buffer::route_w: return config.num_experts_per_tok;
    case Buffer::raw: return std::size_t{config.num_experts_per_tok} * H;
    case Buffer::logits: return config.vocab_size;
    case Buffer::token: return 1;
    }
    return 0;
}

Plan build_decode_plan(const TargetModel& model, std::uint32_t first, std::uint32_t end, bool embed, bool head) {
    const Config& c = model.config;
    if (first >= end || end > c.num_hidden_layers) fail("layer range [" + std::to_string(first) + ", " + std::to_string(end) + ") is empty or out of range");
    Plan plan;
    plan.config = c;
    plan.first_layer = first;
    plan.end_layer = end;
    plan.embed = embed;
    plan.head = head;
    plan.expert_split = c.n_routed_experts / plan.ranks;

    using B = Buffer;
    const auto add = [&](Op op, std::int32_t layer, std::string site, std::vector<std::string> weights, std::vector<B> in,
                         std::vector<B> out, Owner owner, std::string identity) {
        plan.steps.push_back({op, layer, std::move(site), std::move(weights), std::move(in), std::move(out), owner, std::move(identity)});
    };
    if (embed) add(Op::embed, -1, "", {"token_embd.weight"}, {B::token}, {B::stream, B::pre_carry}, Owner::layer, "f16-row-to-f32");

    for (std::uint32_t l = first; l < end; ++l) {
        const auto L = static_cast<std::int32_t>(l);
        const std::string p = "blk." + std::to_string(l) + ".";
        const auto& a = model.layers[l];
        if (a.engram)
            add(Op::engram, L, "", {p + "engram_q_norm.weight", p + "engram_k_norm.weight", p + "engram_kv.weight", p + "engram_embd.weight"},
                {B::token, B::stream}, {B::stream}, Owner::layer, "engram-24-rows-e4m3-e8m0-gated-v1");
        // attention sublayer
        add(Op::hc_mix, L, "attn", {p + "hc_attn_fn.weight", p + "hc_attn_base.weight", p + "hc_attn_scale.weight"}, {B::stream},
            {B::mix_attn}, Owner::layer, "hc-rms-f16-proj-sinkhorn20-v1");
        add(Op::hc_pre, L, "attn", {}, {B::stream, B::pre_carry}, {B::x}, Owner::layer, "hc-pre-ascending-copies-v1");
        add(Op::rms_norm, L, "attn", {p + "attn_norm.weight"}, {B::x}, {B::x}, Owner::layer, "rmsnorm-eps1e-20-v1");
        add(Op::q_a, L, "", {p + "attn_q_a.weight"}, {B::x}, {B::qr}, Owner::layer, "q8_0-a8-product-v1");
        add(Op::rms_norm, L, "q", {p + "attn_q_a_norm.weight"}, {B::qr}, {B::qr}, Owner::layer, "rmsnorm-eps1e-20-v1");
        add(Op::q_b, L, "", {p + "attn_q_b.weight"}, {B::qr}, {B::q}, Owner::layer, "q8_0-a8-product-v1");
        add(Op::kv, L, "", {p + "attn_kv.weight"}, {B::x}, {B::kvrow}, Owner::layer, "q8_0-a8-product-v1");
        add(Op::window_write, L, "", {p + "attn_kv_a_norm.weight"}, {B::kvrow}, {}, Owner::attention,
            a.compress_ratio ? "kvnorm512+rope64-yarn+ring-e4m3-ue8m0-32-v1" : "kvnorm512+rope64-theta10000+ring-e4m3-ue8m0-32-v1");
        if (a.kv_source) {
            std::vector<std::string> w{p + "attn_compressor_kv.weight", p + "attn_compressor_norm.weight", p + "indexer.attn_k.weight",
                                       p + "indexer.k_norm.weight"};
            if (a.compressor_gate) w.push_back(p + "attn_compressor_gate.weight");
            add(Op::compress, L, "", std::move(w), {B::x}, {}, Owner::attention,
                a.compress_ratio > 1 ? "compress-ratio2-softmax-gate-fp4e4m3-16-v1" : "compress-ratio1-fp4e4m3-16-v1");
        }
        if (a.index_source)
            add(Op::index, L, "", {p + "indexer.attn_q_b.weight", p + "indexer.proj.weight"}, {B::x, B::qr}, {}, Owner::attention,
                a.candidate_source ? "index-fp4-relu-top512-candidate-source-v1"
                                   : a.uses_candidates ? "index-fp4-relu-top512-candidate-user-v1" : "index-fp4-relu-top512-v1");
        add(Op::attention, L, "", {p + "attn_sinks.weight"}, {B::q}, {B::o}, Owner::attention,
            a.compress_ratio ? "rope-q+mqa-d512-window+compressed-sink-chunk64-inverse-rope-v1" : "rope-q+mqa-d512-window-sink-chunk64-inverse-rope-v1");
        add(Op::o_a, L, "", {p + "attn_output_a.weight"}, {B::o}, {B::o_low}, Owner::layer, "a8+q8_0-a8-product-grouped8-v1");
        add(Op::o_b, L, "", {p + "attn_output_b.weight"}, {B::o_low}, {B::y}, Owner::layer, "q8_0-a8-ksplit2-h0+h1-v1");
        add(Op::hc_post, L, "attn", {}, {B::y, B::stream, B::mix_attn}, {B::stream}, Owner::layer, "hc-post-ascending-copies-v1");
        // FFN sublayer
        add(Op::hc_mix, L, "ffn", {p + "hc_ffn_fn.weight", p + "hc_ffn_base.weight", p + "hc_ffn_scale.weight"}, {B::stream},
            {B::mix_ffn}, Owner::layer, "hc-rms-f16-proj-sinkhorn20-v1");
        add(Op::hc_pre, L, "ffn", {}, {B::stream, B::mix_attn}, {B::x}, Owner::layer, "hc-pre-ascending-copies-v1");
        add(Op::rms_norm, L, "ffn", {p + "ffn_norm.weight"}, {B::x}, {B::x}, Owner::layer, "rmsnorm-eps1e-20-v1");
        add(Op::router, L, "", {p + "ffn_gate_inp.weight", p + "exp_probs_b.bias", p + "exp_probs_b_vl.bias"}, {B::x},
            {B::route_ids, B::route_w}, Owner::formats_moe, "router-sqrtsoftplus-biased-top6-x1.5-v1");
        add(Op::shared_expert, L, "", {p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight", p + "ffn_down_shexp.weight"}, {B::x},
            {B::shared}, Owner::formats_moe, "q8_0-a8-swiglu-clamp10-v1");
        add(Op::routed_experts, L, "", {p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight"},
            {B::x, B::route_ids}, {B::raw}, Owner::formats_moe, "iq2xxs-q2k-a8-swiglu-clamp10-raw-slots-v1");
        add(Op::combine, L, "", {}, {B::shared, B::raw, B::route_w, B::route_ids}, {B::y}, Owner::formats_moe,
            "combine-shared+low-half+high-half-ascending-id-v1");
        add(Op::hc_post, L, "ffn", {}, {B::y, B::stream, B::mix_ffn}, {B::stream, B::pre_carry}, Owner::layer, "hc-post-ascending-copies-v1");

        LayerState s;
        s.layer = l;
        s.window_ring_bytes = std::uint64_t{c.sliding_window} * ring_entry_bytes(c);
        if (a.kv_source) {
            s.kv_bytes_per_position = kv_entry_bytes(c);
            s.index_key_bytes_per_position = index_key_bytes(c);
            if (a.compressor_gate) s.compress_carry_bytes = 2ULL * a.compress_ratio * c.head_dim * sizeof(float);
        }
        plan.fixed_state_bytes += s.window_ring_bytes + s.compress_carry_bytes;
        plan.state.push_back(s);
    }
    // Context bytes per token: compressed entries are written once per `ratio` tokens.
    std::uint64_t num = 0;  // in units of 1/2 byte to keep ratio-2 entries exact
    for (const auto& s : plan.state) {
        const auto ratio = model.layers[s.layer].compress_ratio;
        if (s.kv_bytes_per_position != 0) num += 2U * (s.kv_bytes_per_position + s.index_key_bytes_per_position) / ratio;
    }
    plan.state_bytes_per_token = num / 2U;
    if (head) {
        add(Op::head, -1, "", {"output_norm.weight", "output.weight"}, {B::stream, B::pre_carry}, {B::logits}, Owner::layer,
            "head-hc-pre+rmsnorm+q8_0-a8-product-v1");
        add(Op::argmax, -1, "", {}, {B::logits}, {B::token}, Owner::layer, "argmax-lowest-index-split-vocab-v1");
    }
    validate_plan(plan, model);
    return plan;
}

void validate_plan(const Plan& plan, const TargetModel& model) {
    std::map<std::string, const ManifestTensor*> by_name;
    for (const auto& t : model.manifest.tensors) by_name.emplace(t.name, &t);
    for (const auto& step : plan.steps) {
        for (const auto& name : step.weights) {
            const auto it = by_name.find(name);
            if (it == by_name.end()) fail(std::string(to_string(step.op)) + " reads " + name + ", which is not in the manifest");
            if (it->second->ggml_type != expected_type(step.op, name))
                fail(std::string(to_string(step.op)) + " expects another storage type for " + name);
        }
        for (const auto b : step.inputs)
            if (plan.buffer_elements(b) == 0) fail("empty buffer " + std::string(to_string(b)));
    }
}

}  // namespace strix::deepseek41
