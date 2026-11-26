#pragma once

#include "strix/pair/state_staging.hpp"
#include "xport/hip_pipe_staging.hpp"

#include <array>
#include <memory>

namespace strix::pair {

class HipStateStaging final : public StateStaging {
public:
    ~HipStateStaging() override {
        (void)hipSetDevice(device_);
        if (stream_) { (void)hipStreamSynchronize(stream_); (void)hipStreamDestroy(stream_); }
        for (auto* slot : slots_) if (slot) (void)hipHostFree(slot);
    }
    void allocate(std::size_t capacity) {
        capacity_ = capacity;
        pipe_hip_check(hipGetDevice(&device_), "pair handoff staging device");
        for (auto*& slot : slots_)
            pipe_hip_check(hipHostMalloc(reinterpret_cast<void**>(&slot), capacity_, hipHostMallocDefault), "pair handoff pinned slot");
        pipe_hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "pair handoff copy stream");
    }
    std::span<std::byte> buffer(std::size_t slot) override { return {slots_.at(slot), capacity_}; }
    void copy_device(bool save, void* device, std::span<std::byte> host) override {
        pipe_hip_check(hipSetDevice(device_), "pair handoff copy device");
        pipe_hip_check(hipMemcpyAsync(save ? host.data() : device, save ? device : host.data(), host.size(),
            save ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice, stream_), "pair handoff async state copy");
        pipe_hip_check(hipStreamSynchronize(stream_), "pair handoff state slot ready");
    }
private:
    int device_ = 0;
    hipStream_t stream_ = nullptr;
    std::size_t capacity_ = 0;
    std::array<std::byte*, 2> slots_{};
};

inline std::unique_ptr<StateStaging> make_hip_state_staging(std::size_t capacity) {
    auto staging = std::make_unique<HipStateStaging>();
    staging->allocate(capacity);
    return staging;
}

} // namespace strix::pair
