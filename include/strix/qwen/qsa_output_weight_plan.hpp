#pragma once

#include "strix/weight/tensor_launch_plan.hpp"

namespace strix::weight {

struct TensorIndex;

KernelId hip_qwen_qsa_output_kernel(bool dense_a8, bool wave8);
Status resolve_qwen_qsa_output(const WeightMapEntry&, Backend, KernelId, TensorLaunchPlan&);
// Bind the native Q8_0 trunk sites. Other source formats keep their existing routes.
Status resolve_qwen_qsa_outputs(const TensorIndex&, const WeightMap&, Backend, KernelId,
                                TensorLaunchPlans&);

} // namespace strix::weight
