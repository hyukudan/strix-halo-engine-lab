#pragma once
#include "strix/qwen_prefill_handoff.hpp"
#include <hip/hip_runtime.h>
#include "xport/hip_pipe_staging.hpp"

namespace strix::qwen_pp {
inline void pair_hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity) {
    return pair::make_hip_pipe_staging(capacity, [] { return hip_execution_stream(); });
}

class HipStateStaging final : public StateStaging {
public:
    ~HipStateStaging() override {
        (void)hipSetDevice(device_);
        if (stream_) { (void)hipStreamSynchronize(stream_); (void)hipStreamDestroy(stream_); }
        for (auto* const& slot : slots_) if (slot) (void)hipHostFree(slot);
    }
    void allocate() {
        pair_hip_check(hipGetDevice(&device_), "pair handoff staging device");
        for (auto*& slot : slots_)
            pair_hip_check(hipHostMalloc(reinterpret_cast<void**>(&slot), seam_piece_bytes, hipHostMallocDefault), "pair handoff pinned slot");
        pair_hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "pair handoff copy stream");
    }
    std::span<std::byte> buffer(std::size_t slot) override { return {slots_.at(slot), seam_piece_bytes}; }
    void copy_device(bool save, void* device, std::span<std::byte> host) override {
        pair_hip_check(hipSetDevice(device_), "pair handoff copy device");
        pair_hip_check(hipMemcpyAsync(save ? host.data() : device, save ? device : host.data(), host.size(),
            save ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice, stream_), "pair handoff async state copy");
        pair_hip_check(hipStreamSynchronize(stream_), "pair handoff state slot ready");
    }
private:
    int device_ = 0;
    hipStream_t stream_ = nullptr;
    std::array<std::byte*, 2> slots_{};
};
std::unique_ptr<StateStaging> make_hip_state_staging() {
    auto staging = std::make_unique<HipStateStaging>();
    staging->allocate();
    return staging;
}
} // namespace strix::qwen_pp
