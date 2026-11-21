#pragma once
// HIP device operations of xport::ChannelCommunicator (channel_communicator.hpp) for any Channel<Traits> whose Tag is
// {kind, layer, step}: stream / event ordering, the RX-flag stream wait, stream-ordered copies. No host
// synchronisation except in teardown (synchronize / synchronize_event).
// Home: shared (xport HIP binding). Bound by: the pair gather link (tp_gather_hip.hpp) and the DS41 prefill / decode
// exchange (src/hip_deepseek41_prefill_xport.hpp).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "strix/dist/communicator.hpp"
#include "strix/hip_check.hpp"
#include "channel_hip.hpp"

namespace strix::xport {

template<class Traits>
struct HipChannelRuntime {
    using Stream = hipStream_t;
    using Event = hipEvent_t;
    using InputTag = dist::Tag;
    Event bridge = nullptr;
    HipChannelRuntime() = default;
    HipChannelRuntime(const HipChannelRuntime&) = delete;
    HipChannelRuntime& operator=(const HipChannelRuntime&) = delete;
    ~HipChannelRuntime() { destroy(bridge); }
    static void check(hipError_t e) { ::strix::hip_check_message(e, "channel runtime: HIP fence/copy failed"); }
    static typename Traits::Tag tag(const InputTag& value) {
        return {static_cast<typename Traits::Kind>(value.kind), value.layer, value.request};
    }
    static void record(Event& event, Stream stream) {
        if (!event) check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        check(hipEventRecord(event, stream));
    }
    static void wait(Event event, Stream stream) { check(hipStreamWaitEvent(stream, event, 0)); }
    void order(Stream producer, Stream posting) {
        if (producer == posting) return;
        record(bridge, producer);
        wait(bridge, posting);
    }
    static void acquire(const RxView& view, Stream stream) {
        check(hipStreamWaitValue64(stream, const_cast<std::uint64_t*>(view.flag), view.sequence, hipStreamWaitValueGte,
                                   ~std::uint64_t{0}));
    }
    static void copy(void* dst, const void* src, std::size_t bytes, Stream stream) {
        check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream));
    }
    static void destroy(Event event) { if (event) (void)hipEventDestroy(event); }
    static void synchronize(Stream stream) { (void)hipStreamSynchronize(stream); }
    static void synchronize_event(Event event) { (void)hipEventSynchronize(event); }
};

}  // namespace strix::xport
