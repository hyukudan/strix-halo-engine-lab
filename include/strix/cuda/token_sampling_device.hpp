#pragma once
#include "strix/token_sampling_device.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix {

class CudaTokenSamplerDevice {
public:
    CudaTokenSamplerDevice(std::uint32_t vocabulary,std::uint32_t max_rows);
    ~CudaTokenSamplerDevice();
    CudaTokenSamplerDevice(const CudaTokenSamplerDevice&) = delete;
    CudaTokenSamplerDevice& operator=(const CudaTokenSamplerDevice&) = delete;
    // Decisions of every row (the caller applies the stopping rule). Waits for `stream`. A row without a device
    // decision (not expected) is decided with the host transform after reading its logits.
    [[nodiscard]] std::vector<TokenDecision> decide(TokenSampler& sampler,std::span<const DeviceSampleRow> rows,void* stream);
    [[nodiscard]] std::uint64_t host_fallbacks() const noexcept;
    struct Impl; // also used by the testing functions
private:
    std::unique_ptr<Impl> impl_;
};

// Draft chain sampling on the device. Each step samples the draft head's logits (entry r is token ids[r]) with
// the host sampler's proposal uniform, writes the token to the step's device slot (the next step's input) and
// keeps the step's distribution on the device; nothing waits for the device until finish().
class CudaTokenDraftSamplerDevice {
public:
    CudaTokenDraftSamplerDevice(std::uint32_t capacity,std::uint32_t max_steps);
    ~CudaTokenDraftSamplerDevice();
    CudaTokenDraftSamplerDevice(const CudaTokenDraftSamplerDevice&) = delete;
    CudaTokenDraftSamplerDevice& operator=(const CudaTokenDraftSamplerDevice&) = delete;
    // The head's active entries: `rows` entries with ids `host_ids` (empty: entry r is token r). Re-uploads the
    // token order only when `rows` changed (the id list is append-only). Enqueued on `stream`.
    void set_vocabulary(std::span<const std::uint32_t> host_ids,std::uint32_t rows,void* stream);
    // Enqueues step `step`: the proposal at `position` from `logits` (device, `rows` entries); the token goes to
    // `slot` (device). Does not wait.
    void step(const TokenSampler& sampler,std::uint32_t step,const float* logits,std::uint64_t position,
        std::uint32_t* slot,void* stream);
    // After the stream has completed: the drafts of steps [0, steps) with device distributions (valid until the
    // next set_vocabulary or step), counted in `sampler`. Throws on a non-finite draft row.
    void finish(TokenSampler& sampler,std::uint32_t steps,std::vector<TokenDraft>& out);
    struct Impl; // also used by the testing functions
private:
    std::unique_ptr<Impl> impl_;
};

// A host copy of a draft whose distribution lives on the device (unchanged when it is already on the host).
[[nodiscard]] TokenDraft cuda_host_token_draft(const TokenDraft& draft,void* stream);

// Testing: the device decision for explicit uniforms, as decide_token(make_token_distribution(logits), draft, ...).
// device_q: the draft distribution is read from device memory (token ids ascending with zero entries).
[[nodiscard]] TokenDecision cuda_decide_token_for_testing(std::span<const float> logits,const TokenSamplingPolicy& policy,
    const TokenDraft* draft,double u_plain,double u_accept,double u_residual,bool* host_fallback = nullptr,
    bool device_q = false);
// Testing: one device draft step over `logits` with entry ids `ids` (empty: identity) and proposal uniform `u`,
// returned with its distribution copied to the host.
[[nodiscard]] TokenDraft cuda_draft_for_testing(std::span<const float> logits,std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy,double u);

} // namespace strix
