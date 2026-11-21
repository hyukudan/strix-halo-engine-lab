#include "strix/cuda/token_sampling_device.hpp"
#include <cuda_runtime.h>

namespace {
constexpr unsigned kDefaultCases = 96U;
}

#define STRIX_TEST_TARGET CudaTokenSamplerDevice
#define STRIX_TEST_DRAFT_SAMPLER CudaTokenDraftSamplerDevice
#define STRIX_TEST_DECIDE cuda_decide_token_for_testing
#define STRIX_TEST_DRAFT cuda_draft_for_testing
#define STRIX_TEST_RT(name) cuda##name

#include "token_sampling_device_cases.inl"
