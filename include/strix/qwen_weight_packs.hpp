// The weight packs the Qwen3.8-Flash-Next family declares to the shared admission (strix/weight_pack.hpp).
// Declaration only: the routes are the shared CUDA cores (A8 dense/MoE kernels, load-time transcode, native-type
// MoE through the shared decoders of strix/cuda_qwen_quant.cuh). Order matters: the first pack admitting a file wins.
//   qwen38-ud-q4kxl-a8-v1      the product file: routed Q4_K/Q5_K gate/up, Q5_1/Q8_0 down, dense Q8_0/F32 -> exact A8.
//   qwen38-gsq-rco-iq3s-v1     ISTA-DASLab GSQ-RCO IQ3_S (and its IQ3_XXS / IQ2_XS siblings: same type ladders):
//                              routed IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS gate/up, IQ4_NL/Q2_0 down; dense K-/IQ-quants/BF16.
//                              FAST route: dense -> Q8_0 at load (quantize_row_q8_0 rule on the exactly decoded
//                              values), routed experts native (FP32 dots on the exact host-oracle decode).
//   qwen38-gguf-native-v1      any other mix the shared decoders read (e.g. a production/GSQ composite): same FAST
//                              route, labelled so it is never mistaken for a qualified pack.
#pragma once

#include "strix/gguf_reader.hpp"
#include "strix/weight_pack.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::qwen_packs {

inline constexpr std::string_view gsq_a8_contract="gsq-iq3s-a8-dot4-cuda-v1";
// Tensor-core GSQ experts with exact operands (BF16 rows, integer codes, FP32 group scales; src/cuda/core/native_mma.hpp).
// FAST until its gate passes; explicit selection only, never a default.
inline constexpr std::string_view gsq_bf16_contract="gsq-iq-bf16mma-cuda-v1";
// TURBO integer tensor-core GSQ experts (shared A8 int8 activations x integer codes, exact int32 group dots, one FP32 FMA per
// group; src/cuda/core/turbo_i8.hpp). TURBO class: selected by the TURBO profile or explicitly, never by EXACT/FAST.
inline constexpr std::string_view gsq_i8_turbo_contract="gsq-iq-i8mma-turbo-cuda-v1";
enum class ExpertArithmetic : std::uint8_t { incumbent, gsq_a8_dot4, gsq_bf16_mma, gsq_i8_turbo };
inline constexpr std::string_view expert_arithmetic_name(ExpertArithmetic arithmetic) {
    return arithmetic==ExpertArithmetic::incumbent?"incumbent":
           arithmetic==ExpertArithmetic::gsq_a8_dot4?gsq_a8_contract:
           arithmetic==ExpertArithmetic::gsq_bf16_mma?gsq_bf16_contract:gsq_i8_turbo_contract;
}

namespace t {
inline constexpr std::uint32_t f32 = 0, f16 = 1, q5_1 = 7, q8_0 = 8, q4_k = 12, q5_k = 13, q6_k = 14, iq3_xxs = 18,
                               iq4_nl = 20, iq3_s = 21, iq2_s = 22, iq4_xs = 23, bf16 = 30, q2_0 = 42;
}
inline constexpr std::array<std::uint32_t, 2> a8_gate{t::q4_k, t::q5_k}, a8_down{t::q5_1, t::q8_0},
    a8_dense{t::q8_0, t::f32}, ple{t::iq4_nl}, indexer{t::bf16, t::f32}, gsq_down{t::iq4_nl, t::q2_0};
inline constexpr std::array<std::uint32_t, 4> gsq_gate{t::iq2_s, t::iq3_xxs, t::iq3_s, t::iq4_xs};
// every type the load-time transcode and the native MoE decode (strix/cuda_qwen_quant.cuh geometry table)
inline constexpr std::array<std::uint32_t, 14> decodable{t::f32, t::f16, t::q5_1, t::q8_0, t::q4_k, t::q5_k, t::q6_k,
                                                         t::iq3_xxs, t::iq4_nl, t::iq3_s, t::iq2_s, t::iq4_xs, t::bf16,
                                                         t::q2_0};

inline constexpr std::array<weight_pack::Rule, 5> product_rules{{
    {"routed gate/up", "_gate_exps", a8_gate}, {"routed gate/up", "_up_exps", a8_gate},
    {"routed down", "_down_exps", a8_down}, {"PLE table", "per_layer_token_embd", ple},
    {"indexer", ".indexer.", indexer}}};
inline constexpr std::array<weight_pack::Rule, 6> product_all{{
    product_rules[0], product_rules[1], product_rules[2], product_rules[3], product_rules[4],
    {"dense", "", a8_dense}}};
inline constexpr std::array<weight_pack::Rule, 5> gsq_all{{
    {"routed gate/up", "_gate_exps", gsq_gate}, {"routed gate/up", "_up_exps", gsq_gate},
    {"routed down", "_down_exps", gsq_down}, {"PLE table", "per_layer_token_embd", ple},
    {"dense (transcoded)", "", decodable}}};
inline constexpr std::array<weight_pack::Rule, 1> native_all{{{"any (decoded)", "", decodable}}};
inline constexpr std::array<std::uint32_t,5> gsq_a8_gate{t::iq2_s,t::iq3_xxs,t::iq3_s,t::iq4_xs,t::q2_0};
inline constexpr std::array<weight_pack::Rule,5> gsq_a8_rules{{
    {"routed gate/up","_gate_exps",gsq_a8_gate},{"routed gate/up","_up_exps",gsq_a8_gate},
    {"routed down","_down_exps",gsq_down},{"PLE table","per_layer_token_embd",ple},
    {"dense (transcoded)","",decodable}}};
// Explicit-only: never insert this candidate ahead of a historical first-match
// declaration. Selecting a new arithmetic contract must not change old packs.
inline constexpr weight_pack::Decl gsq_a8_pack{
    "qwen38-gsq-rco-iq3s-a8-v1","fast:gsq-iq3s-a8-dot4-cuda-v1",
    "legacy dense images; routed GSQ signed-group A8 dp4a/IMMA, ordered four-chain products (unqualified FAST)",gsq_a8_rules};
// Same storage rules as the A8 candidate; only the expert arithmetic differs (BF16 MMA, no activation quantization).
inline constexpr weight_pack::Decl gsq_bf16_pack{
    "qwen38-gsq-rco-iq3s-bf16mma-v1","fast:gsq-iq-bf16mma-cuda-v1",
    "legacy dense images; routed GSQ integer codes x BF16 rows on BF16 MMA, FP32 group scales (unqualified FAST)",gsq_a8_rules};
// Same storage rules again; the expert arithmetic is the TURBO integer tensor-core contract (unqualified TURBO).
inline constexpr weight_pack::Decl gsq_i8_turbo_pack{
    "qwen38-gsq-rco-iq3s-i8turbo-v1","turbo:gsq-iq-i8mma-turbo-cuda-v1",
    "legacy dense images; routed GSQ integer codes x A8 int8 rows on int8 IMMA, one FP32 FMA per group (unqualified TURBO)",gsq_a8_rules};

inline constexpr std::array<weight_pack::Decl, 3> family{{
    {"qwen38-ud-q4kxl-a8-v1", "exact", "A8 dense + A8 MoE kernels (cuda-qwen-a8-*)", product_all},
    {"qwen38-gsq-rco-iq3s-v1", "fast:gsq-iq-q8transcode-v1",
     "dense/HC/PLE-proj/embd/head -> Q8_0 at load, small vectors/router -> F32; routed experts native IQ (FP32 dots)",
     gsq_all},
    {"qwen38-gguf-native-v1", "fast:gguf-native-q8transcode-v1",
     "as qwen38-gsq-rco-iq3s-v1 for any mix of decodable types (unqualified composite)", native_all}}};

inline weight_pack::Admission admit(std::span<const weight_pack::Tensor> tensors,std::string_view requested_arithmetic={}) {
    static constexpr std::array<weight_pack::TypePair, 1> paired{{{"_gate_exps", "_up_exps"}}};
    if(requested_arithmetic==gsq_a8_contract || requested_arithmetic==gsq_bf16_contract || requested_arithmetic==gsq_i8_turbo_contract) {
        const auto* decl=requested_arithmetic==gsq_a8_contract?&gsq_a8_pack:
                         requested_arithmetic==gsq_bf16_contract?&gsq_bf16_pack:&gsq_i8_turbo_pack;
        auto admission=weight_pack::admit(std::span<const weight_pack::Decl>(decl,1),tensors,paired);
        if(admission.pack) {
            bool gate=false,down=false;
            for(const auto& tensor:tensors) { gate|=tensor.name.find("_gate_exps")!=std::string_view::npos; down|=tensor.name.find("_down_exps")!=std::string_view::npos; }
            if(!gate || !down) { admission.pack=nullptr; admission.reason=std::string(requested_arithmetic)+" requires routed gate/up and down tensors"; }
        }
        return admission;
    }
    if(!requested_arithmetic.empty() && requested_arithmetic!="incumbent") {
        auto admission=weight_pack::admit({},tensors,paired);
        admission.reason="unsupported explicit Qwen expert arithmetic: "+std::string(requested_arithmetic);
        return admission;
    }
    return weight_pack::admit(family, tensors, paired);
}
// The one entry point of the engine (driver load) and of --print-effective-config: the merged tensor list of every part.
inline weight_pack::Admission admit_file(const gguf::FileInfo& info,std::string_view requested_arithmetic={}) {
    std::vector<weight_pack::Tensor> v;
    v.reserve(info.tensors.size());
    for (const auto& t : info.tensors) v.push_back({t.name, t.ggml_type, t.dimensions});
    return admit(v,requested_arithmetic);
}
// Header-only read of every split part of `path` (no tensor data): the admission of a model path.
inline weight_pack::Admission admit_model(const std::string& path,std::string_view requested_arithmetic={}) {
    gguf::FileInfo all;
    for (const auto& p : weight_pack::gguf_parts(path)) {
        auto part = gguf::read_file(p);
        for (auto& t : part.tensors) all.tensors.push_back(std::move(t));
    }
    return admit_file(all,requested_arithmetic);
}
using weight_pack::hex64;

}  // namespace strix::qwen_packs
