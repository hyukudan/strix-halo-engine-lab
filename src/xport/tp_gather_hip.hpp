#pragma once
// HIP binding of the shared pair gather link (strix/pair/gather_link.hpp): the pair Communicator opened by
// dist::open_pair, the route, and the start-up link measurement (dist/link_cost.hpp) over the SAME gather form the
// decode runs, agreed by both ranks so both price trips with identical bits.
// Channel route (STRIX_TP_CHANNEL=1): the in-token gathers ride the shared device-waited channel
// (xport::Channel<tp_gather::Traits> borrowing the Communicator's transport, xport::ChannelCommunicator, HIP runtime
// hip_channel_runtime.hpp): no host wait per gather; the same bytes land in the same slots in the same stream order.
// Home: shared (backend binding, HIP). Bound by: qwen35 dense HIP exact TP decode (apps/hip_qwen35_decode_main.hip).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/dist/communicator.hpp"
#include "strix/dist/link_cost.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/hip_check.hpp"
#include "strix/pair/gather_link.hpp"
#include "channel.hpp"
#include "channel_communicator.hpp"
#include "hip_channel_runtime.hpp"
#include "tp_gather_traits.hpp"

namespace strix::xport {
extern template struct ChannelOptions<tp_gather::Traits>;  // both instantiated once in tp_gather.hip (library strix_tp_xport)
extern template class Channel<tp_gather::Traits>;
}

namespace strix::xport::tp_gather {

struct HipOps {
    using Stream = hipStream_t;
    static Stream stream(void* s) noexcept { return static_cast<hipStream_t>(s); }
    static void sync(Stream s) { ::strix::hip_check_message(hipStreamSynchronize(s), "tp gather: stream error"); }
    static dist::Tag tag(std::uint64_t call) noexcept { return dist::Tag{std::uint32_t(dist::TagKind::raw_slots), 0U, call}; }
};
// The wire identity's Kind mirrors dist::TagKind by value (the Runtime converts a dist::Tag by value).
static_assert(std::uint32_t(Kind::control) == std::uint32_t(dist::TagKind::control) &&
              std::uint32_t(Kind::half_sum) == std::uint32_t(dist::TagKind::half_sum) &&
              std::uint32_t(Kind::raw_slots) == std::uint32_t(dist::TagKind::raw_slots) &&
              std::uint32_t(Kind::argmax) == std::uint32_t(dist::TagKind::argmax),
              "tp gather: Kind must follow dist::TagKind");

using Exchange = Channel<Traits>;
using Runtime = HipChannelRuntime<Traits>;
using ChannelComm = ChannelCommunicator<Exchange, dist::Communicator, Runtime>;

class HipPairGather {
public:
    using Link = pair::GatherLink<HipOps, dist::Communicator, ChannelComm>;

    // `capacity`: the largest gather message in bytes (every rank the same). `stream`: the engine stream every gather is
    // enqueued on (the channel posts and waits on it). `channel_requested`: the caller's resolved switch; the route
    // itself is the shared policy pair::select_route. Both ranks construct at the same point with the same settings.
    HipPairGather(dist::OpenedTransport opened, std::uint32_t rank, std::size_t capacity, hipStream_t stream, bool channel_requested)
        : capacity_(capacity), stream_(stream) {
        dist::Group grp;
        grp.rank = rank;
        const bool direct = opened.direct != nullptr;  // --tp-transport usb4gpu: device exchange when the rings allow it
        comm_ = direct ? std::make_unique<dist::Communicator>(grp, std::move(opened.host), std::move(opened.direct), capacity)
                       : std::make_unique<dist::Communicator>(grp, std::move(opened.host), capacity);
        const auto route = pair::select_route(direct, channel_requested);
        if (route == pair::GatherRoute::channel) {
            typename Exchange::Options o;
            o.rank = rank;
            o.capacity = capacity;
            o.stream = stream;
            o.from_env();  // STRIX_TP_XPORT_* (depth, CPUs, timeout)
            channel_ = std::make_unique<Exchange>(comm_->transport(), o);
            fast_ = std::make_unique<ChannelComm>(comm_.get(), channel_.get(), runtime_);
            link_ = Link(rank, pair::GatherRoute::channel, comm_.get(), fast_.get());
        } else {
            link_ = Link(rank, route, comm_.get(), nullptr);
        }
    }
    HipPairGather(const HipPairGather&) = delete;
    HipPairGather& operator=(const HipPairGather&) = delete;

    [[nodiscard]] Link& link() noexcept { return link_; }

    // Times the gather form on the engine stream at the start-up sizes that fit the capacity, fits latency + bytes / rate,
    // and agrees field-wise with the peer. Restarts the counters. One sample = kProbeChain gathers enqueued back to back
    // and ONE stream wait, divided by kProbeChain: the decode step never waits on the stream after a gather (the next
    // work is stream-ordered behind it), so a stream wait per probed gather would price a host wake + an empty-queue
    // relaunch that the step does not run (Hy4 TP2 staged: 237 us probed vs ~80-128 us effective per in-step trip).
    static constexpr unsigned kProbeChain = 16U;
    [[nodiscard]] dist::LinkCost measure_link() {
        hipStream_t stream = stream_;
        std::vector<std::uint64_t> sizes;
        for (const auto b : dist::default_link_sizes) if (b <= capacity_) sizes.push_back(b);
        if (sizes.empty()) throw std::invalid_argument("tp gather: capacity below the smallest link probe size");
        const std::size_t slot = std::size_t(sizes.back());
        void* buf = nullptr;
        ::strix::hip_check_message(hipMalloc(&buf, 2U * slot), "tp gather: probe buffer");
        dist::LinkCost mine;
        try {
            ::strix::hip_check_message(hipMemsetAsync(buf, 0, 2U * slot, stream), "tp gather: probe buffer");
            auto samples = dist::measure([&](std::uint64_t bytes) {
                for (unsigned k = 0; k < kProbeChain; ++k) link_.gather(stream, buf, slot, std::size_t(bytes));
                HipOps::sync(stream);
            }, sizes, 2U, 35U);
            for (auto& x : samples) {
                x.p50_us /= double(kProbeChain);
                x.p90_us /= double(kProbeChain);
            }
            mine = dist::fit(samples);
        } catch (...) {
            (void)hipStreamSynchronize(stream);
            (void)hipFree(buf);
            throw;
        }
        ::strix::hip_check_message(hipFree(buf), "tp gather: probe buffer");
        link_.reset_counters();
        if (fast_) fast_->drain();  // the channel's progress work is complete before the transport carries the agreement
        return dist::agree(comm_->transport(), mine);
    }
    // One measured scalar agreed by both ranks (dist::agree_max over the pair transport, the channel drained first):
    // e.g. the timed one-node pass a placement is calibrated with. Both ranks call it at the same point.
    [[nodiscard]] double agree_max(double mine) {
        if (fast_) fast_->drain();
        return dist::agree_max(comm_->transport(), mine);
    }
    // After the last gather of a run, before its results are printed (both ranks): surfaces a failed integrity check of
    // the channel's last messages (pair::GatherLink::finish).
    void finish() { link_.finish(); }

private:
    // Destruction in reverse: channel communicator (drains, waits for its consumers), channel, runtime, communicator.
    std::size_t capacity_ = 0;
    hipStream_t stream_ = nullptr;
    std::unique_ptr<dist::Communicator> comm_;
    Runtime runtime_;
    std::unique_ptr<Exchange> channel_;
    std::unique_ptr<ChannelComm> fast_;
    Link link_;
};

}  // namespace strix::xport::tp_gather
