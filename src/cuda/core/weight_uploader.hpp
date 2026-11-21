#pragma once

#include "core/resident_device.hpp"
#include "core/weight_load_queue.hpp"

#include <mutex>

namespace strix::cuda::core {

// The shared queue overlaps bounded shard reads with transfers on this
// stream. A completed upload ticket has no outstanding staging references.
class WeightUploader final : public strix::core::WeightUploader {
public:
    std::uint8_t* allocate(std::uint64_t bytes) override {
        void* result = nullptr;
        namespace api = strix::core::resident_device;
        api::check(cudaMallocHost(&result, static_cast<std::size_t>(bytes)), "allocate weight upload staging");
        return static_cast<std::uint8_t*>(result);
    }
    void release(std::uint8_t* staging, std::uint64_t) noexcept override { (void)cudaFreeHost(staging); }

    Token submit(const std::uint8_t* staging, const Copy* copies, std::size_t count) override {
        namespace api = strix::core::resident_device;
        std::lock_guard lock(mutex_);
        api::select(device_);
        try {
            for (std::size_t i = 0; i < count; ++i)
                api::copy(copies[i].device, staging + copies[i].staging_offset,
                          static_cast<std::size_t>(copies[i].bytes), api::host_to_device, stream_.get());
            api::synchronize(stream_.get());
        } catch (...) {
            (void)cudaStreamSynchronize(stream_.get());
            throw;
        }
        return ++latest_;
    }
    bool done(Token) override { return true; }
    void wait(Token) override {}
    void retire(Token) noexcept override {}
    void drain() override {}

private:
    int device_{strix::core::resident_device::current_device()};
    strix::core::resident_device::StreamOwner stream_;
    std::mutex mutex_;
    Token latest_{};
};

}  // namespace strix::cuda::core
