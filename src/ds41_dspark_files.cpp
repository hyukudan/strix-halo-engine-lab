#include "strix/ds41_dspark_files.hpp"

#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <fstream>
#include <span>
#include <stdexcept>

namespace strix::ds41::dspark {
namespace {

std::vector<std::uint8_t> read_range(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(bytes);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read in " + path);
    return out;
}
void load(LoadedWeights& m, const std::string& path, const std::vector<std::string>& names) {
    const auto file = gguf::read_file(path);
    for (const auto& name : names) {
        const auto it = std::find_if(file.tensors.begin(), file.tensors.end(), [&](const auto& t) { return t.name == name; });
        if (it == file.tensors.end()) throw std::runtime_error("missing tensor " + name);
        m.bytes[name] = read_range(path, it->absolute_offset, it->byte_size);
    }
}
std::span<const float> f32(const LoadedWeights& m, const std::string& name) {
    const auto& b = m.bytes.at(name);
    return {reinterpret_cast<const float*>(b.data()), b.size() / 4U};
}
F16Matrix f16(const LoadedWeights& m, const std::string& name, std::uint32_t width) {
    const auto& b = m.bytes.at(name);
    return {{reinterpret_cast<const std::uint16_t*>(b.data()), b.size() / 2U}, width, static_cast<std::uint32_t>(b.size() / 2U / width)};
}
Q8Matrix q8(const LoadedWeights& m, const std::string& name, std::uint32_t width) {
    const auto& b = m.bytes.at(name);
    const std::uint64_t row = width / 32U * 34U;
    if (b.size() % row != 0U) throw std::runtime_error(name + " is not a Q8_0 matrix of width " + std::to_string(width));
    return {b, width, static_cast<std::uint32_t>(b.size() / row)};
}

} // namespace

std::unique_ptr<LoadedWeights> load_weights(const std::string& target, const std::string& draft) {
    auto m = std::make_unique<LoadedWeights>();
    const char* stage_names[] = {"hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight", "attn_norm.weight", "hc_ffn_fn.weight",
                                 "hc_ffn_base.weight", "hc_ffn_scale.weight", "ffn_norm.weight", "attn_sinks.weight", "attn_q_a.weight",
                                 "attn_q_b.weight", "attn_q_a_norm.weight", "attn_kv.weight", "attn_kv_a_norm.weight", "attn_output_a.weight",
                                 "attn_output_b.weight", "ffn_gate_inp.weight", "exp_probs_b.bias", "exp_probs_b_vl.bias",
                                 "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight",
                                 "ffn_up_exps.weight", "ffn_down_exps.weight"};
    std::vector<std::string> names{"mtp.0.main_proj.weight", "mtp.0.main_norm.weight", "mtp.2.norm.weight", "mtp.2.markov_head.embed.weight",
                                   "mtp.2.markov_head.head.weight", "mtp.2.confidence_head.proj.weight"};
    for (int s = 0; s < 3; ++s)
        for (const char* n : stage_names) names.push_back("mtp." + std::to_string(s) + "." + n);
    load(*m, draft, names);
    load(*m, target, {"token_embd.weight", "output.weight"});
    auto& w = m->weights;
    const auto& L = *m;
    w.main_proj = q8(L, "mtp.0.main_proj.weight", main_width);
    w.main_norm = f32(L, "mtp.0.main_norm.weight");
    w.final_norm = f32(L, "mtp.2.norm.weight");
    w.markov_embed = f16(L, "mtp.2.markov_head.embed.weight", markov_rank);
    w.markov_head = f16(L, "mtp.2.markov_head.head.weight", markov_rank);
    w.confidence = f32(L, "mtp.2.confidence_head.proj.weight");
    w.token_embd = f16(L, "token_embd.weight", hidden);
    w.output = q8(L, "output.weight", hidden);
    for (std::uint32_t s = 0; s < stages; ++s) {
        const std::string p = "mtp." + std::to_string(s) + ".";
        auto& st = w.stage[s];
        st.hc_attn_fn = f16(L, p + "hc_attn_fn.weight", stream_width); st.hc_ffn_fn = f16(L, p + "hc_ffn_fn.weight", stream_width);
        st.hc_attn_scale = f32(L, p + "hc_attn_scale.weight"); st.hc_attn_base = f32(L, p + "hc_attn_base.weight");
        st.hc_ffn_scale = f32(L, p + "hc_ffn_scale.weight"); st.hc_ffn_base = f32(L, p + "hc_ffn_base.weight");
        st.attn_norm = f32(L, p + "attn_norm.weight"); st.ffn_norm = f32(L, p + "ffn_norm.weight"); st.sinks = f32(L, p + "attn_sinks.weight");
        st.q_norm = f32(L, p + "attn_q_a_norm.weight"); st.kv_norm = f32(L, p + "attn_kv_a_norm.weight");
        st.q_a = q8(L, p + "attn_q_a.weight", hidden); st.q_b = q8(L, p + "attn_q_b.weight", q_rank); st.kv = q8(L, p + "attn_kv.weight", hidden);
        st.out_a = q8(L, p + "attn_output_a.weight", o_group_width); st.out_b = q8(L, p + "attn_output_b.weight", o_width);
        st.router = f32(L, p + "ffn_gate_inp.weight"); st.bias = f32(L, p + "exp_probs_b.bias"); st.bias_image = f32(L, p + "exp_probs_b_vl.bias");
        st.routed.gate_up_format = WeightFormat::mxfp4; st.routed.down_format = WeightFormat::mxfp4;
        st.routed.gate = L.bytes.at(p + "ffn_gate_exps.weight"); st.routed.up = L.bytes.at(p + "ffn_up_exps.weight");
        st.routed.down = L.bytes.at(p + "ffn_down_exps.weight"); st.routed.hidden = hidden; st.routed.inter = inter;
        st.shared.gate_up_format = WeightFormat::q8_0; st.shared.down_format = WeightFormat::q8_0;
        st.shared.gate = L.bytes.at(p + "ffn_gate_shexp.weight"); st.shared.up = L.bytes.at(p + "ffn_up_shexp.weight");
        st.shared.down = L.bytes.at(p + "ffn_down_shexp.weight"); st.shared.hidden = hidden; st.shared.inter = inter;
        if (st.q_a.outputs != q_rank || st.q_b.outputs != q_width || st.kv.outputs != head_dim || st.out_a.outputs != o_width ||
            st.out_b.outputs != hidden || st.router.size() != static_cast<std::size_t>(experts) * hidden)
            throw std::runtime_error("unexpected draft stage geometry");
    }
    if (w.main_proj.outputs != hidden || w.output.outputs != vocab || w.token_embd.outputs != vocab || w.markov_head.outputs != vocab ||
        w.confidence.size() != confidence_width)
        throw std::runtime_error("unexpected drafter geometry");
    return m;
}

} // namespace strix::ds41::dspark
