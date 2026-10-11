#pragma once

#include "strix/host_upload.hpp"

#include <cstdint>
#include <memory>

namespace strix {

// Pinned host inputs copied on the explicit legacy default stream. Cancellation cannot
// abort an enqueued HIP copy; event/stream completion is its retirement proof.
// Prepare/submit/poll require the same active HIP device; a change retains the
// source instead of treating a foreign device's default stream as quiescent.
class HipHostUploadBackend final : public HostUploadBackend {
public:
    explicit HipHostUploadBackend(std::uint32_t maximum_operations = 2,
                                  std::uint64_t teardown_polls = 4096);
    ~HipHostUploadBackend() override;
    HipHostUploadBackend(const HipHostUploadBackend&) = delete;
    HipHostUploadBackend& operator=(const HipHostUploadBackend&) = delete;

    [[nodiscard]] std::shared_ptr<HostUploadSource> allocate_source(
        std::uint64_t bytes, std::uint64_t allocation_generation) override;
    [[nodiscard]] HostUploadHandle prepare(
        const ExecutionBufferAddress& destination,
        std::shared_ptr<const HostUploadSource> source,
        std::uint64_t bytes) override;
    void submit(HostUploadHandle handle) override;
    [[nodiscard]] HostUploadStatus poll(HostUploadHandle handle) override;
    void cancel(HostUploadHandle handle) override;
    void retire(HostUploadHandle handle) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix
