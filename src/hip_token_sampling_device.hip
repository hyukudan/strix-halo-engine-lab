#include "strix/hip_check.hpp"
#include "strix/hip_token_sampling_device.hpp"
#include "strix/hip_token_sampling.hpp"

#include <hip/hip_runtime.h>

namespace strix {
namespace {

using u64 = unsigned long long;
using u128 = unsigned __int128;
using Stream = hipStream_t;
using Event = hipEvent_t;
constexpr auto host_alloc_flags = hipHostMallocDefault;

using strix::hip_check;

void check(hipError_t error, const char* message) { hip_check(error, message); }

template <typename T>
T* mapped_device_pointer(T* pointer) { return pointer; }

void read_logits_row(const ExecutionBufferAddress& row, std::uint32_t vocabulary, void* stream, std::vector<float>& out) {
    read_hip_logits_row(row, vocabulary, stream, out);
}

} // namespace
} // namespace strix

#define STRIX_SAMPLE_TARGET HipTokenSamplerDevice
#define STRIX_SAMPLE_DRAFT HipTokenDraftSamplerDevice
#define STRIX_SAMPLE_HOST_DRAFT host_token_draft
#define STRIX_SAMPLE_TEST_DECIDE hip_decide_token_for_testing
#define STRIX_SAMPLE_TEST_DRAFT hip_draft_for_testing
#define STRIX_SAMPLE_RT(name) hip##name
#define STRIX_SAMPLE_HOST_ALLOC hipHostMalloc
#define STRIX_SAMPLE_HOST_FREE hipHostFree
#define STRIX_SAMPLE_LAUNCH hipLaunchKernelGGL

#include "core/token_sampling_device_body.inl"
