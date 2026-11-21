#pragma once

#include "resolve.hpp"

#include <map>

namespace strix::weight {

struct TensorIndex;

enum class QwenQsaOutputKernel : KernelId {
    hip_tree = 0x515301,
    hip_wave8 = 0x515302,
    hip_dense_a8 = 0x515303,
    cuda_dense_a8 = 0x515304,
};

constexpr QwenQsaOutputKernel hip_qwen_qsa_output_kernel(bool dense_a8, bool wave8) {
    if (dense_a8) return QwenQsaOutputKernel::hip_dense_a8;
    return wave8 ? QwenQsaOutputKernel::hip_wave8 : QwenQsaOutputKernel::hip_tree;
}

struct QwenQsaOutputPlan {
    Plan launch;
    ByteRange payload;
    Geometry layout;
};
using QwenQsaOutputPlans = std::map<std::string, QwenQsaOutputPlan, std::less<>>;

Status resolve_qwen_qsa_output(const WeightMapEntry&, Backend, QwenQsaOutputKernel, QwenQsaOutputPlan&);
// Bind the native Q8_0 trunk sites. Other source formats keep their existing routes.
Status resolve_qwen_qsa_outputs(const TensorIndex&, const WeightMap&, Backend, QwenQsaOutputKernel,
                              QwenQsaOutputPlans&);
std::string qwen_qsa_output_plans_json(const QwenQsaOutputPlans&);

} // namespace strix::weight
