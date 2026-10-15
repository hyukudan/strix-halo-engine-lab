#pragma once
#include "strix/hip_hyper_mix_launcher.hpp"
#include <hip/hip_runtime_api.h>
namespace strix {
// Test-only same-input output seam. Caller owns exact live normalized/low/Q8
// and output extents, retains them through completion, and drains explicitly.
void enqueue_lane_chain_hyper_output_test(const HipHyperMixLauncherConfig&,const float*,const float*,const void*,void*,std::uint32_t rows,bool lane_chain,hipStream_t);
}
