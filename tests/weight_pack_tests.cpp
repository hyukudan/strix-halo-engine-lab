// Shared weight-pack admission (strix/weight_pack.hpp) with the Qwen family declarations (strix/qwen_weight_packs.hpp):
// inventories shaped like the product UD-Q4_K_XL file, the GSQ-RCO IQ3_S file and a production/GSQ composite.
#include "strix/qwen_weight_packs.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
namespace t = strix::qwen_packs::t;
struct Inv {
    std::vector<std::string> names;
    std::vector<std::uint32_t> types;
    std::vector<std::uint64_t> dims{2560, 640};
    std::vector<strix::weight_pack::Tensor> view() const {
        std::vector<strix::weight_pack::Tensor> v;
        for (std::size_t i = 0; i < names.size(); ++i) v.push_back({names[i], types[i], dims});
        return v;
    }
};
Inv layer(std::uint32_t gate, std::uint32_t up, std::uint32_t down, std::uint32_t dense, std::uint32_t hc) {
    Inv i;
    i.names = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
               "blk.0.attn_qkv.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.hc_attn_up.weight",
               "blk.0.ffn_gate_inp.weight", "blk.0.ssm_a", "per_layer_token_embd.weight", "blk.0.indexer.proj.weight"};
    i.types = {gate, up, down, dense, dense, hc, t::f32, t::f32, t::iq4_nl, t::bf16};
    return i;
}
}  // namespace

int main() {
    using strix::qwen_packs::admit;
    {
        const auto inv = layer(t::q4_k, t::q5_k, t::q5_1, t::q8_0, t::q8_0);
        const auto a = admit(inv.view());
        check(a.pack && a.pack->id == "qwen38-ud-q4kxl-a8-v1", "product file -> product pack");
        check(a.pack && a.pack->arithmetic == "exact", "product pack is exact");
        check(a.tensors_by_type.at(t::f32) == 2, "census counts types");
    }
    {
        const auto inv = layer(t::iq3_xxs, t::iq2_s, t::q2_0, t::q6_k, t::bf16);
        const auto a = admit(inv.view());
        check(a.pack && a.pack->id == "qwen38-gsq-rco-iq3s-v1", "GSQ-RCO IQ file -> gsq pack");
        check(a.pack && a.pack->arithmetic.substr(0, 5) == "fast:", "gsq pack is a FAST class");
        check(a.reason.find("qwen38-ud-q4kxl-a8-v1: blk.0.ffn_gate_exps") == 0, "reason: why the product pack declined");
    }
    {  // production dense + GSQ Q2_0 gate/up/down: the composite is decodable but not a qualified pack
        const auto inv = layer(t::q2_0, t::q2_0, t::q2_0, t::q8_0, t::q8_0);
        const auto a = admit(inv.view());
        check(a.pack && a.pack->id == "qwen38-gguf-native-v1", "composite -> generic native label");
        check(a.reason.find("ffn_gate_exps") != std::string::npos, "the refusals name the tensor");
    }
    {
        const auto inv = layer(t::q4_k, t::q4_k, 99U, t::q8_0, t::q8_0);
        const auto a = admit(inv.view());
        check(a.pack == nullptr, "undecodable type -> no pack");
        check(a.reason.find("type 99") != std::string::npos, "refusal names the type");
    }
    {
        auto x = layer(t::iq3_xxs, t::iq2_s, t::q2_0, t::q6_k, t::bf16);
        const auto f0 = admit(x.view()).fingerprint;
        check(admit(x.view()).fingerprint == f0, "fingerprint is deterministic");
        x.types[0] = t::iq3_s;
        check(admit(x.view()).fingerprint != f0, "fingerprint follows tensor types");
    }
    check(admit({}).pack == nullptr, "empty inventory is refused");
    std::printf("weight_pack tests: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
