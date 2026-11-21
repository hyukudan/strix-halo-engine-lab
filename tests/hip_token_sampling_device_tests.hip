#include "strix/hip_token_sampling_device.hpp"
#include <hip/hip_runtime.h>

namespace {
constexpr unsigned kDefaultCases = 400U;
}

#define STRIX_TEST_TARGET HipTokenSamplerDevice
#define STRIX_TEST_DRAFT_SAMPLER HipTokenDraftSamplerDevice
#define STRIX_TEST_DECIDE hip_decide_token_for_testing
#define STRIX_TEST_DRAFT hip_draft_for_testing
#define STRIX_TEST_RT(name) hip##name

#include "token_sampling_device_cases.inl"
