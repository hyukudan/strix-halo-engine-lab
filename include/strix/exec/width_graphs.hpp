#pragma once

// One captured multi-request decode program per program WIDTH, replayed over one device step block.
//
// The program of width W reads every per-request operand (id, position, slot) from the device
// StepBlock returned by block() and every per-session state address through a slot directory
// (strix/exec/step_lanes.hpp, strix/state/lane_view.hpp), so it is captured once and replayed for any request set of width W:
// one 1,056 B host-to-device copy and one graph launch per step instead of thousands of launches.
// A width whose capture fails stays eager until reset(); the caller then runs its eager path.

#if defined(__HIPCC__) || defined(__CUDACC__)

#include "strix/device/runtime.hpp"
#include "strix/exec/descriptor.hpp"
#include "strix/exec/graph_runner.hpp"

#include <array>
#include <cstddef>
#include <new>

#if defined(__HIPCC__)
#define STRIX_WIDTH_RT(name) hip##name
#else
#define STRIX_WIDTH_RT(name) cuda##name
#endif

namespace strix::exec {

template <class Binding>
class BoundWidthStepGraphs {
public:
    using Policy = WidthReplayPolicy;
    using Stream = device::native::Stream;

    explicit BoundWidthStepGraphs(GraphRows layout = GraphRows::cohort) : runner_(layout) {}
    explicit BoundWidthStepGraphs(const StreamGraphPrograms& programs) : runner_(programs) {}
    ~BoundWidthStepGraphs() {
        invalidate();
        if (device_) (void)STRIX_WIDTH_RT(Free)(device_);
#if defined(__HIPCC__)
        if (staging_) (void)hipHostFree(staging_);
#else
        if (staging_) (void)cudaFreeHost(staging_);
#endif
    }
    BoundWidthStepGraphs(const BoundWidthStepGraphs&) = delete;
    BoundWidthStepGraphs& operator=(const BoundWidthStepGraphs&) = delete;

    // The device step block every captured kernel reads. Stable for the lifetime of this object.
    [[nodiscard]] StepBlock* block() {
        if (!device_) {
            void* address{};
            device::native::check(STRIX_WIDTH_RT(Malloc)(&address, sizeof(StepBlock)), "allocate device step block");
            device_ = static_cast<StepBlock*>(address);
        }
        if (!staging_) {
            void* address{};
#if defined(__HIPCC__)
            device::native::check(hipHostMalloc(&address, sizeof(StepBlock)), "allocate step block staging");
#else
            device::native::check(cudaMallocHost(&address, sizeof(StepBlock)), "allocate step block staging");
#endif
            staging_ = static_cast<StepBlock*>(address);
        }
        return device_;
    }

    // Stage the block for an eager lane-mode launch (no graph) in stream order.
    void upload(Stream stream, const StepBlock& input) {
        (void)block();
        *staging_ = input;
        device::native::check(STRIX_WIDTH_RT(MemcpyAsync)(device_, staging_, sizeof(input), device::native::host_to_device, stream),
              "upload step block");
    }

    // Replay (or capture once, then replay) the program of width input.h.rows. False: nothing ran
    // (graphs off, unstable layout, or this width declined) and the caller runs the step eagerly. A
    // graph launch error has uncertain submission state and propagates. The serial owner drains the
    // stream (which also retires the staging copy) before the next call.
    template <class Work>
    bool try_launch(Stream stream, bool enabled, bool stable_layout, const StepBlock& input,
                    Work&& work, const Binding& binding = {}) {
        return runner_.run(*this, stream, enabled, stable_layout, input, binding, work);
    }

    // Uses the immutable stream program table passed at construction. A nonrectangular
    // group layout or an unlisted shape declines before capture, upload or launch.
    template <class Work>
    bool try_launch_for_streams(Stream stream, bool enabled, bool stable_layout, const StepBlock& input,
                                std::uint32_t active_streams, Work&& work, const Binding& binding = {}) {
        return runner_.run_for_streams(*this, stream, enabled, stable_layout, input, binding, work, active_streams);
    }

    [[nodiscard]] bool capturing() const noexcept { return capturing_; }
    [[nodiscard]] unsigned instantiations() const noexcept { return instantiations_; }
    // Kernel/copy/memset nodes of the installed program of `width`; 0 when none is installed.
    [[nodiscard]] std::size_t nodes(std::uint32_t width) const noexcept {
        return runner_.uses_stream_programs() ? 0U : nodes_for_key(width);
    }
    [[nodiscard]] std::size_t nodes_for_streams(std::uint32_t active_streams) const noexcept {
        return runner_.rows_per_stream(active_streams) ? nodes_for_key(active_streams) : 0U;
    }
    [[nodiscard]] const Policy& policy() const noexcept { return runner_.policy(); }

    void invalidate() noexcept { runner_.invalidate(*this); }
    void reset() noexcept { runner_.reset(*this); }

private:
    friend class StepGraphRunner<Binding>;

    [[nodiscard]] std::size_t nodes_for_key(std::uint32_t width) const noexcept {
        const auto at = Policy::index(width);
        return at < Policy::kWidths && exec_[at] ? nodes_[at] : 0U;
    }

    void launch(Stream stream, std::uint32_t at) {
        device::native::check(STRIX_WIDTH_RT(GraphLaunch)(exec_[at], stream), "launch width step graph");
    }

    template <class Work>
    bool capture(Stream stream, std::uint32_t at, Work& work) {
        try {
            (void)block();
        } catch (...) {
            return false;
        }
        const bool installed = record(stream, at, work);
        if (!installed) (void)STRIX_WIDTH_RT(GetLastError)();
        return installed;
    }

    template <class Work>
    bool record(Stream stream, std::uint32_t at, Work& work) {
        discard(at);
        const auto graph = device::native::record_step_graph(stream, capturing_, work);
        if (!graph) return false;
        std::size_t count = 0;
        if (!device::native::inspect_raw_step_graph(graph, count)) {
            (void)STRIX_WIDTH_RT(GraphDestroy)(graph);
            return false;
        }
        const auto status = device::native::instantiate_step_graph(&exec_[at], graph);
        (void)STRIX_WIDTH_RT(GraphDestroy)(graph);
        if (status != STRIX_WIDTH_RT(Success)) {
            exec_[at] = nullptr;
            return false;
        }
        nodes_[at] = count;
        ++instantiations_;
        return true;
    }

    void discard(std::uint32_t at) noexcept {
        if (exec_[at]) (void)STRIX_WIDTH_RT(GraphExecDestroy)(exec_[at]);
        exec_[at] = nullptr;
        nodes_[at] = 0U;
    }
    StepGraphRunner<Binding> runner_;
    std::array<STRIX_WIDTH_RT(GraphExec_t), Policy::kWidths> exec_{};
    std::array<std::size_t, Policy::kWidths> nodes_{};
    StepBlock* device_{};
    StepBlock* staging_{};
    bool capturing_{};
    unsigned instantiations_{};
};

using WidthStepGraphs = BoundWidthStepGraphs<IndirectGraphBinding>;

}  // namespace strix::exec

#undef STRIX_WIDTH_RT
#endif
