// Hy4-flash DFly2 drafter naming map: data only (see include/strix/weight/hy4_dfly_map.hpp). Names, shapes and dtypes are
// the header of the official shard model-00148-of-00148.safetensors (87 tensors, BF16, 1,691,383,434 bytes).
#include "strix/weight/hy4_dfly_map.hpp"


namespace strix::weight::tensor_map {
namespace {

using L = Layers;
using S = Sym;
constexpr Extent X(S a, S b = S::one) { return Extent{a, b}; }
constexpr std::array<Extent, 4> D(Extent a, Extent b = {}, Extent c = {}, Extent d = {}) { return {a, b, c, d}; }
constexpr std::uint64_t B = type_bit(30);  // the official drafter is BF16

constexpr Entry kDfly2[] = {
    {"model.dfly_layers.mtp_fusion.enorm.weight", L::global, "dfly.fusion", D(X(S::hidden)), B},
    {"model.dfly_layers.mtp_fusion.hnorm.weight", L::global, "dfly.fusion", D(X(S::hidden)), B},
    {"model.dfly_layers.mtp_fusion.gate", L::global, "dfly.fusion", D(X(S::hidden)), B},
    {"model.dfly_layers.mtp_fusion.eh_proj.weight", L::global, "dfly.fusion", D(X(S::two, S::hidden), X(S::hidden)), B},
    {"model.dfly_layers.hidden_correction.embed_norm.weight", L::global, "dfly.correction", D(X(S::hidden)), B},
    {"model.dfly_layers.hidden_correction.hidden_norm.weight", L::global, "dfly.correction", D(X(S::hidden)), B},
    {"model.dfly_layers.hidden_correction.gate_proj.weight", L::global, "dfly.correction", D(X(S::two, S::hidden), X(S::hidden)), B},
    {"model.dfly_layers.hidden_correction.up_proj.weight", L::global, "dfly.correction", D(X(S::two, S::hidden), X(S::hidden)), B},
    {"model.dfly_layers.hidden_correction.down_proj.weight", L::global, "dfly.correction", D(X(S::hidden), X(S::hidden)), B},
    {"model.dfly_layers.confidence_head.proj.weight", L::global, "dfly.confidence", D(X(S::hidden), X(S::one)), B},
    {"model.dfly_layers.confidence_head.proj.bias", L::global, "dfly.confidence", D(X(S::one)), B},
    {"model.dfly_layers.final_norm.weight", L::global, "dfly.norm", D(X(S::hidden)), B},
    // Draft layers: pre-norm MLA with the target layer's sandwich norm, learned sink and output gate; dense SwiGLU.
    {"model.dfly_layers.{L}.input_layernorm.weight", L::all, "dfly.norm", D(X(S::hidden)), B},
    {"model.dfly_layers.{L}.post_attention_layernorm.weight", L::all, "dfly.norm", D(X(S::hidden)), B},
    {"model.dfly_layers.{L}.self_attn.q_a_layernorm.weight", L::all, "dfly.norm", D(X(S::q_lora)), B},
    {"model.dfly_layers.{L}.self_attn.kv_a_layernorm.weight", L::all, "dfly.norm", D(X(S::kv_lora)), B},
    {"model.dfly_layers.{L}.self_attn.post_proj_norm.weight", L::all, "dfly.norm", D(X(S::hidden)), B},
    {"model.dfly_layers.{L}.self_attn.learnable_sink_param", L::all, "dfly.attention.sinks", D(X(S::heads)), B},
    {"model.dfly_layers.{L}.self_attn.q_a_proj.weight", L::all, "dfly.attention.q_a", D(X(S::hidden), X(S::q_lora)), B},
    {"model.dfly_layers.{L}.self_attn.q_b_proj.weight", L::all, "dfly.attention.q_b", D(X(S::q_lora), X(S::heads, S::qk_head)), B},
    {"model.dfly_layers.{L}.self_attn.kv_a_proj_with_mqa.weight", L::all, "dfly.attention.kv_a", D(X(S::hidden), X(S::kv_a)), B},
    {"model.dfly_layers.{L}.self_attn.kv_b_proj.weight", L::all, "dfly.attention.kv_b", D(X(S::kv_lora), X(S::heads, S::kv_b_head)), B},
    {"model.dfly_layers.{L}.self_attn.linear_gate.weight", L::all, "dfly.attention.gate", D(X(S::hidden), X(S::heads, S::v_head)), B},
    {"model.dfly_layers.{L}.self_attn.o_proj.weight", L::all, "dfly.attention.o", D(X(S::heads, S::v_head), X(S::hidden)), B},
    {"model.dfly_layers.{L}.mlp.gate_proj.weight", L::all, "dfly.mlp.gate", D(X(S::hidden), X(S::dense_ff)), B},
    {"model.dfly_layers.{L}.mlp.up_proj.weight", L::all, "dfly.mlp.up", D(X(S::hidden), X(S::dense_ff)), B},
    {"model.dfly_layers.{L}.mlp.down_proj.weight", L::all, "dfly.mlp.down", D(X(S::dense_ff), X(S::hidden)), B},
};

}  // namespace

std::span<const Entry> hy4_dfly2_table() { return kDfly2; }

Model hy4_dfly2_model() {
    Model m;
    m.blocks = 5;
    const auto set = [&m](S s, std::uint64_t v) { m.value[static_cast<std::size_t>(s)] = v; };
    set(S::hidden, 3072);
    set(S::heads, 32);
    set(S::q_lora, 1024);
    set(S::kv_lora, 512);
    set(S::kv_a, 512 + 64);
    set(S::qk_nope, 192);
    set(S::qk_head, 192 + 64);
    set(S::v_head, 256);
    set(S::kv_b_head, 192 + 256);
    set(S::dense_ff, 9216);
    m.layer_class.assign(m.blocks, 1U << static_cast<std::uint32_t>(L::all));
    return m;
}

bool safetensors_infos(const TensorIndex& index, std::vector<gguf::TensorInfo>& out, std::string& error) {
    out.clear();
    out.reserve(index.tensors.size());
    for (const auto& t : index.tensors) {
        gguf::TensorInfo info;
        info.name = t.name;
        if (t.wire_dtype == "F32") info.ggml_type = 0;
        else if (t.wire_dtype == "F16") info.ggml_type = 1;
        else if (t.wire_dtype == "BF16") info.ggml_type = 30;
        else {
            error = "safetensors dtype " + t.wire_dtype + " has no checker type: " + t.name;
            return false;
        }
        if (t.opaque()) {
            error = "safetensors tensor of rank > 4: " + t.name;
            return false;
        }
        info.dimensions.assign(t.extents.rbegin(), t.extents.rend());
        out.push_back(std::move(info));
    }
    return true;
}

}  // namespace strix::weight::tensor_map
