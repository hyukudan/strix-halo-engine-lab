#pragma once
// HIP binding of the shared pair gather link (strix/pair/gather_link.hpp): the pair Communicator opened by
// dist::open_pair, the route, and the start-up link measurement (dist/link_cost.hpp) over the SAME gather form the
// decode runs, agreed by both ranks so both price trips with identical bits.
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

namespace strix::xport::tp_gather {

struct HipOps {
    using Stream = hipStream_t;
    static Stream stream(void* s) noexcept { return static_cast<hipStream_t>(s); }
    static void sync(Stream s) { ::strix::hip_check_message(hipStreamSynchronize(s), "tp gather: stream error"); }
    static dist::Tag tag(std::uint64_t call) noexcept { return dist::Tag{std::uint32_t(dist::TagKind::raw_slots), 0U, call}; }
};

class HipPairGather {
public:
    using Link = pair::GatherLink<HipOps, dist::Communicator, pair::NoExchange>;

    // `capacity`: the largest gather message in bytes (every rank the same).
    HipPairGather(dist::OpenedTransport opened, std::uint32_t rank, std::size_t capacity) : capacity_(capacity) {
        dist::Group grp;
        grp.rank = rank;
        const bool direct = opened.direct != nullptr;  // --tp-transport usb4gpu: device exchange when the rings allow it
        comm_ = direct ? std::make_unique<dist::Communicator>(grp, std::move(opened.host), std::move(opened.direct), capacity)
                       : std::make_unique<dist::Communicator>(grp, std::move(opened.host), capacity);
        link_ = Link(rank, direct ? pair::GatherRoute::device_direct : pair::GatherRoute::staged, comm_.get(), nullptr);
    }

    [[nodiscard]] Link& link() noexcept { return link_; }

    // Times the gather form on `stream` (each call ends with a stream wait, so it is complete) at the start-up sizes
    // that fit the capacity, fits latency + bytes / rate, and agrees field-wise with the peer. Restarts the counters.
    [[nodiscard]] dist::LinkCost measure_link(hipStream_t stream) {
        std::vector<std::uint64_t> sizes;
        for (const auto b : dist::default_link_sizes) if (b <= capacity_) sizes.push_back(b);
        if (sizes.empty()) throw std::invalid_argument("tp gather: capacity below the smallest link probe size");
        const std::size_t slot = std::size_t(sizes.back());
        void* buf = nullptr;
        ::strix::hip_check_message(hipMalloc(&buf, 2U * slot), "tp gather: probe buffer");
        dist::LinkCost mine;
        try {
            ::strix::hip_check_message(hipMemsetAsync(buf, 0, 2U * slot, stream), "tp gather: probe buffer");
            const auto samples = dist::measure([&](std::uint64_t bytes) {
                link_.gather(stream, buf, slot, std::size_t(bytes));
                HipOps::sync(stream);
            }, sizes, 5U, 35U);
            mine = dist::fit(samples);
        } catch (...) {
            (void)hipStreamSynchronize(stream);
            (void)hipFree(buf);
            throw;
        }
        ::strix::hip_check_message(hipFree(buf), "tp gather: probe buffer");
        link_.reset_counters();
        return dist::agree(comm_->transport(), mine);
    }

private:
    std::size_t capacity_ = 0;
    std::unique_ptr<dist::Communicator> comm_;
    Link link_;
};

}  // namespace strix::xport::tp_gather
