#pragma once
#include "strix/gpu_visibility_pattern.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace strix::transport {
struct GpuVisibilityRange {
    std::uintptr_t address = 0;
    std::uint64_t bytes = 0;
};
struct HipGpuVisibilityProbePolicy {
    std::uint64_t maximum_bytes = 16ULL * 1024 * 1024;
    std::chrono::milliseconds timeout{10000};
};
// Serialized experimental helper. All buffers and the stream must belong to
// retained_owner, which is held until probe GPU work is actually complete.
// Failed/timed-out GPU work retains storage, event and owner for process life.
// This helper does not import/arm/repost DMA, reset hardware, or grant readiness.
class HipGpuVisibilityProbe final {
public:
    HipGpuVisibilityProbe(std::shared_ptr<void> retained_owner,
        std::function<hipStream_t()> stream, std::function<void()> require_current,
        HipGpuVisibilityProbePolicy = {});
    ~HipGpuVisibilityProbe();
    HipGpuVisibilityProbe(const HipGpuVisibilityProbe&) = delete;
    HipGpuVisibilityProbe& operator=(const HipGpuVisibilityProbe&) = delete;
    // Waits for producer completion before returning. System-scope event fences
    // are retained; this alone is not proof of cross-device cache visibility.
    void produce(std::span<const GpuVisibilityRange>, GpuVisibilityPattern);
    [[nodiscard]] GpuVisibilityResult verify(std::span<const GpuVisibilityRange>,
        GpuVisibilityPattern);
    // Sensitivity control on a separately owned execution destination only.
    // Flip a byte, verify rejection on GPU, restore it, verify the full payload.
    // Must never target an imported/posted RX/TX pool or a model-live buffer.
    [[nodiscard]] bool negative_control(std::span<const GpuVisibilityRange>,
        GpuVisibilityPattern, std::uint64_t byte_offset);
    [[nodiscard]] bool close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::transport
