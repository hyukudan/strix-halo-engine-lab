#include "schema.hpp"
#include "strix/cuda/dense_kernel_ids.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/qwen/qsa_output_weight_plan.hpp"
#include <algorithm>
#include <iterator>
#include <utility>

namespace strix::weight {
namespace {
constexpr std::string_view qsa_output_suffix = ".attn_output.weight";
}

Status bind_qwen(const TensorIndex& index,WeightMap& out) {
    static constexpr FormatId linear[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1},{"gguf.q4_0",1},{"gguf.q5_0",1},{"gguf.q5_1",1},{"gguf.q8_0",1},{"gguf.q3_k",1},{"gguf.q4_k",1},{"gguf.q5_k",1},{"gguf.q6_k",1},{"gguf.iq4_nl",1},{"gguf.iq4_xs",1},{"gguf.iq2_s",1},{"gguf.iq3_xxs",1},{"gguf.iq3_s",1},{"gguf.q2_0.strix",1}};
    static constexpr FormatId plain[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1},{"gguf.q4_0",1},{"gguf.q5_0",1},{"gguf.q5_1",1},{"gguf.q8_0",1},{"gguf.q3_k",1},{"gguf.q4_k",1},{"gguf.q5_k",1},{"gguf.q6_k",1},{"gguf.iq4_nl",1},{"gguf.iq4_xs",1},{"gguf.iq2_s",1},{"gguf.iq3_xxs",1},{"gguf.iq3_s",1},{"gguf.q2_0.strix",1}};
    static constexpr FormatId auxiliary[] = {{"plain.f32",1}};
    static constexpr detail::RolePattern roles[] = {{".ssm_a",Role::auxiliary,true,true},{".ssm_d",Role::auxiliary,true,true},{"embed",Role::embedding},{"token_embd",Role::embedding},{"lm_head",Role::head},
    {qsa_output_suffix,Role::attention_o,false,true},{"output.weight",Role::head},
    {"ffn_gate_inp",Role::router},{"router",Role::router},{"norm",Role::norm},{".bias",Role::bias},
    {"q_proj",Role::attention_q},{"attn_q",Role::attention_q},{"k_proj",Role::attention_k},{"attn_k",Role::attention_k},
    {"v_proj",Role::attention_v},{"attn_v",Role::attention_v},{"o_proj",Role::attention_o},{"attn_output",Role::attention_o},
    {"down",Role::dense_down},{"gate",Role::dense_gate},{"up",Role::dense_up}};
    auto native_linear=std::vector<FormatId>(std::begin(linear),std::end(linear));
    native_linear.push_back({"nvidia.nvfp4",1});
    const detail::FamilySchema schema{"qwen",native_linear,plain,roles,false,false,false,false,auxiliary};
    return detail::bind_model(index,schema,out);
}

namespace {

constexpr FormatId q8_0{"gguf.q8_0", 1};

using HipOutputKernel = HipIndexedAttentionOutputKernel;

constexpr Candidate qsa_output_candidate(KernelId kernel, Backend backend, ArithmeticId arithmetic) {
    Candidate candidate{};
    candidate.kernel = kernel;
    candidate.storage = q8_0;
    candidate.arithmetic = arithmetic;
    candidate.image = {"image.raw", 1};
    candidate.backend = backend;
    candidate.strategy = Strategy::native;
    candidate.exactness = Exactness::legacy_bitwise;
    candidate.cost.packed_bytes_per_weight = candidate.cost.image_bytes_per_weight = {34, 32};
    return candidate;
}

constexpr Candidate qsa_output_candidates[] = {
    qsa_output_candidate(static_cast<KernelId>(HipOutputKernel::tree), Backend::hip, {"hip-qwen-qsa-output-tree", 1}),
    qsa_output_candidate(static_cast<KernelId>(HipOutputKernel::wave8), Backend::hip, {"hip-qwen-qsa-output-tree", 1}),
    qsa_output_candidate(static_cast<KernelId>(HipOutputKernel::dense_a8), Backend::hip, {"hip-qwen-dense-a8-q80", 1}),
    qsa_output_candidate(cuda::kDenseA8ImmaKernel, Backend::cuda, {"cuda-qwen-imma-four-chain", 1}),
};

bool is_qsa_output(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    if (!name.starts_with(prefix) || !name.ends_with(qsa_output_suffix) || name.size() <= prefix.size() + qsa_output_suffix.size())
        return false;
    name.remove_prefix(prefix.size());
    name.remove_suffix(qsa_output_suffix.size());
    return name.find_first_not_of("0123456789") == std::string_view::npos;
}

} // namespace

KernelId hip_qwen_qsa_output_kernel(bool dense_a8, bool wave8) {
    if (dense_a8) return static_cast<KernelId>(HipOutputKernel::dense_a8);
    return static_cast<KernelId>(wave8 ? HipOutputKernel::wave8 : HipOutputKernel::tree);
}

Status resolve_qwen_qsa_output(const WeightMapEntry& entry, Backend backend, KernelId incumbent,
                              TensorLaunchPlan& out) {
    if (entry.binding.role != Role::attention_o)
        return {Error::unsupported_role, "QSA output plan requires an attention output projection"};
    const auto& shape = entry.weight.shape;
    if (shape.rank != 2 || (backend == Backend::cuda && shape.dims[1] % 64U != 0))
        return {Error::unsupported_shape, "QSA output plan requires packed matrix rows (K multiple of 64 on CUDA)"};

    const auto* candidate = std::find_if(std::begin(qsa_output_candidates), std::end(qsa_output_candidates),
                                        [incumbent](const Candidate& option) { return option.kernel == incumbent; });
    if (candidate == std::end(qsa_output_candidates))
        return {Error::unsupported_backend, "QSA output incumbent has no backend registration"};
    ResolveRequest request{};
    request.tensor = &entry;
    request.hardware.backend = backend;
    request.workload = {1, shape.dims[0], shape.dims[1], 0, 0, {}, 1, false, false};
    request.required_arithmetic = candidate->arithmetic;
    request.maximum_permission = Exactness::legacy_bitwise;
    request.incumbent = incumbent;
    Plan launch{};
    auto status = resolve(builtin_registry(), request, qsa_output_candidates, {}, launch);
    if (status) out = {std::move(launch), entry.weight.payload.range, entry.weight.geometry};
    return status;
}

Status resolve_qwen_qsa_outputs(const TensorIndex& index, const WeightMap& admitted, Backend backend,
                               KernelId incumbent, TensorLaunchPlans& out) {
    TensorLaunchPlans plans;
    for (const auto& raw : index.tensors) {
        if (raw.wire_type != 8U || !is_qsa_output(raw.name)) continue;
        const auto entry = std::find_if(admitted.begin(), admitted.end(), [&](const WeightMapEntry& weight) {
            return weight.binding.tensor == raw.id;
        });
        if (entry == admitted.end()) return {Error::malformed, raw.name + ": missing admitted weight"};
        TensorLaunchPlan plan{};
        auto status = resolve_qwen_qsa_output(*entry, backend, incumbent, plan);
        if (!status) return {status.code, raw.name + ": " + status.detail};
        if (!plans.emplace(raw.name, std::move(plan)).second)
            return {Error::malformed, raw.name + ": duplicate QSA output weight"};
    }
    out = std::move(plans);
    return {Error::ok, {}};
}
}
