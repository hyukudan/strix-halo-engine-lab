#pragma once
#include "communicator_backend.hpp"
#include <type_traits>

namespace strix::xport {
// DS41 decode candidates are already device-resident, unframed payloads.
// Keep the ordinary start/finish route, even when async/staging is available.
template<class TagType> struct Ds41ArgmaxTraits {
    using Tag = TagType;
    static constexpr BackendPublish publish{false, false};
};

template<class StreamType> struct Ds41ExchangeRuntime {
    using Stream = StreamType;
    using Event = std::nullptr_t;
    Stream stream;
    // Legacy ordering belongs to Communicator. Prefill ChannelCommunicator
    // supplies its own device readiness and physical-slot lease fences.
    void record_source(Event, Stream, bool) {}
    void record_remote(Event, Stream) {}
};

template<class Communicator, class StreamType, class Candidate, class TagType>
void ds41_exchange_argmax(Communicator& communicator, StreamType exchange_stream,
                         Candidate* local_candidates, Candidate* remote_candidates,
                         std::size_t candidate_rows, const TagType& exchange_tag) {
    using Traits = Ds41ArgmaxTraits<TagType>;
    Ds41ExchangeRuntime<StreamType> exchange_runtime{exchange_stream};
    CommunicatorBackend<Communicator, decltype(exchange_runtime), typename Traits::Tag, true>
        exchange_backend{communicator, exchange_runtime};
    const auto exchange_ticket = exchange_backend.post(
        exchange_tag, {local_candidates, remote_candidates, AddressSpace::device},
        candidate_rows * sizeof(Candidate), Traits::publish);
    (void)exchange_backend.wait(exchange_ticket, exchange_stream);
    exchange_backend.release(exchange_ticket, exchange_stream);
}

// Ready views already contain the producer payload. Adapt only the Communicator
// vocabulary; retain its full consumer view until the stream-ordered release.
template<class Communicator, class StreamType> struct Ds41HalfSumReadyFacade {
    Communicator& ready_communicator;
    void* prepared_data;
    decltype(ready_communicator.finish_staged_ready()) received_view{};
    void* send_staging(std::size_t) { return prepared_data; }
    template<class TagType>
    void start_staged(StreamType ready_stream, std::size_t ready_bytes, const TagType& ready_tag) {
        ready_communicator.start_staged_ready(ready_stream, ready_bytes, ready_tag);
    }
    const void* finish_staged(StreamType) {
        received_view = ready_communicator.finish_staged_ready();
        return received_view.data;
    }
    void release_staged(StreamType ready_stream) {
        ready_communicator.release_staged_ready(ready_stream);
    }
    template<class TagType>
    void start(StreamType ready_stream, void* ready_payload, std::size_t ready_bytes, const TagType& ready_tag) {
        ready_communicator.start(ready_stream, ready_payload, ready_bytes, ready_tag);
    }
    template<class TagType>
    bool start_async(StreamType ready_stream, void* ready_payload, void*, std::size_t ready_bytes,
                     const TagType& ready_tag, std::nullptr_t, std::nullptr_t) {
        start(ready_stream, ready_payload, ready_bytes, ready_tag);
        return false;
    }
    void finish(StreamType ready_stream, void* ready_destination) {
        ready_communicator.finish(ready_stream, ready_destination);
    }
};

template<class StreamType> struct Ds41HalfSumReadyRuntime : Ds41ExchangeRuntime<StreamType> {
    void copy(void* ready_source, void* ready_destination, std::size_t, std::nullptr_t) {
        if (ready_source != ready_destination)
            throw std::logic_error("DS41 ready payload must already occupy the TX view");
    }
};

template<class Communicator, class StreamType, class TagType, class Overlap, class Consume>
void ds41_exchange_half_sum_ready(Communicator& half_communicator, StreamType half_stream,
                                 void* prepared_payload, std::size_t half_bytes,
                                 const TagType& half_tag, Overlap overlap_work, Consume consume_view) {
    Ds41HalfSumReadyFacade<Communicator, StreamType> ready_facade{half_communicator, prepared_payload};
    Ds41HalfSumReadyRuntime<StreamType> ready_runtime{{half_stream}};
    CommunicatorBackend<decltype(ready_facade), decltype(ready_runtime), TagType>
        ready_backend{ready_facade, ready_runtime};
    const auto ready_ticket = ready_backend.post(
        half_tag, {prepared_payload, nullptr, AddressSpace::device}, half_bytes, {true, false});
    overlap_work();
    (void)ready_backend.wait(ready_ticket, half_stream);
    consume_view(ready_facade.received_view);
    ready_backend.release(ready_ticket, half_stream);
}

template<class Communicator, class StreamType, class TagType, class Overlap>
void ds41_exchange_half_sum(Communicator& half_communicator, StreamType half_stream,
                           void* own_payload, void* other_payload, std::size_t half_bytes,
                           const TagType& half_tag, Overlap overlap_work) {
    Ds41ExchangeRuntime<StreamType> half_runtime{half_stream};
    CommunicatorBackend<Communicator, decltype(half_runtime), TagType, true>
        half_backend{half_communicator, half_runtime};
    const auto half_ticket = half_backend.post(
        half_tag, {own_payload, other_payload, AddressSpace::device}, half_bytes, {false, false});
    overlap_work();
    (void)half_backend.wait(half_ticket, half_stream);
    half_backend.release(half_ticket, half_stream);
}
// Raw-slot exchanges retain the caller's route and opaque byte/slot ordering.
// A separate lease supports split-shared producers/consumers in different ops.
template<class Communicator, class StreamType, class TagType>
class Ds41RawSlotsExchange {
    Ds41ExchangeRuntime<StreamType> raw_runtime_;
    using RawBackend = CommunicatorBackend<Communicator, decltype(raw_runtime_), TagType, true>;
    RawBackend raw_backend_;
    typename RawBackend::Ticket raw_ticket_;
public:
    Ds41RawSlotsExchange(Communicator& raw_communicator, StreamType raw_stream,
                         void* raw_local, void* raw_remote, std::size_t raw_bytes, const TagType& raw_tag)
        : raw_runtime_{raw_stream}, raw_backend_{raw_communicator, raw_runtime_},
          raw_ticket_{raw_backend_.post(raw_tag, {raw_local, raw_remote, AddressSpace::device},
                                        raw_bytes, {false, false})} {}
    Ds41RawSlotsExchange(const Ds41RawSlotsExchange&) = delete;
    Ds41RawSlotsExchange& operator=(const Ds41RawSlotsExchange&) = delete;
    void finish(StreamType completion_stream) {
        (void)raw_backend_.wait(raw_ticket_, completion_stream);
        raw_backend_.release(raw_ticket_, completion_stream);
    }
};

template<class Communicator, class StreamType, class TagType, class Overlap>
void ds41_exchange_raw_slots(Communicator& raw_communicator, StreamType raw_stream,
                             void* raw_local, void* raw_remote, std::size_t raw_bytes,
                             const TagType& raw_tag, Overlap raw_overlap) {
    Ds41RawSlotsExchange<Communicator, StreamType, TagType> raw_exchange{
        raw_communicator, raw_stream, raw_local, raw_remote, raw_bytes, raw_tag};
    raw_overlap();
    raw_exchange.finish(raw_stream);
}

// The producer already selected and populated this staging view. Never probe
// staging again, copy the payload, or select an asynchronous route here.
template<class Communicator, class StreamType>
struct Ds41RawSlotsStagedFacade : Ds41HalfSumReadyFacade<Communicator, StreamType> {
    template<class TagType>
    void start_staged(StreamType staged_stream, std::size_t staged_bytes, const TagType& staged_tag) {
        this->ready_communicator.start_staged(staged_stream, staged_bytes, staged_tag);
    }
    const void* finish_staged(StreamType staged_stream) {
        return this->ready_communicator.finish_staged(staged_stream);
    }
    void release_staged(StreamType staged_stream) {
        this->ready_communicator.release_staged(staged_stream);
    }
};

// Split-phase prefill exchanges keep independent work and staged consumers at
// their original call sites. The prepared view is borrowed until release().
template<class Communicator, class StreamType, class TagType>
class Ds41RawSlotsStagedExchange {
    using StagedFacade = Ds41RawSlotsStagedFacade<Communicator, StreamType>;
    using StagedRuntime = Ds41HalfSumReadyRuntime<StreamType>;
    using StagedBackend = CommunicatorBackend<StagedFacade, StagedRuntime, TagType>;
    StagedFacade staged_facade_;
    StagedRuntime staged_runtime_;
    StagedBackend staged_backend_;
    typename StagedBackend::Ticket staged_ticket_;
public:
    Ds41RawSlotsStagedExchange(Communicator& staged_communicator, StreamType staged_stream,
                               void* staged_prepared, std::size_t staged_bytes, const TagType& staged_tag)
        : staged_facade_{staged_communicator, staged_prepared}, staged_runtime_{{staged_stream}},
          staged_backend_{staged_facade_, staged_runtime_},
          staged_ticket_{staged_backend_.post(staged_tag, {staged_prepared, nullptr, AddressSpace::device},
                                               staged_bytes, {true, false})} {}
    Ds41RawSlotsStagedExchange(const Ds41RawSlotsStagedExchange&) = delete;
    Ds41RawSlotsStagedExchange& operator=(const Ds41RawSlotsStagedExchange&) = delete;
    const void* finish(StreamType staged_completion_stream) {
        return staged_backend_.wait(staged_ticket_, staged_completion_stream).remote;
    }
    void release(StreamType staged_release_stream) {
        staged_backend_.release(staged_ticket_, staged_release_stream);
    }
};

// Prefill half sums use the same opaque prepared-view lease: no route probe or
// copy, split start/finish for overlap, and explicit release after consumption.
template<class Communicator, class StreamType, class TagType>
using Ds41HalfSumStagedExchange = Ds41RawSlotsStagedExchange<Communicator, StreamType, TagType>;

template<bool Ready, class Communicator, class StreamType, class TagType, class Overlap, class Consume>
void ds41_exchange_raw_slots_staged(Communicator& raw_communicator, StreamType raw_stream,
                                    void* raw_prepared, std::size_t raw_bytes, const TagType& raw_tag,
                                    Overlap raw_overlap, Consume raw_consume) {
    if constexpr (!Ready) {
        Ds41RawSlotsStagedExchange<Communicator, StreamType, TagType> staged_exchange{
            raw_communicator, raw_stream, raw_prepared, raw_bytes, raw_tag};
        raw_overlap();
        raw_consume(staged_exchange.finish(raw_stream));
        staged_exchange.release(raw_stream);
    } else {
        using RawFacade = std::conditional_t<Ready, Ds41HalfSumReadyFacade<Communicator, StreamType>,
                                            Ds41RawSlotsStagedFacade<Communicator, StreamType>>;
        RawFacade raw_facade{raw_communicator, raw_prepared};
        Ds41HalfSumReadyRuntime<StreamType> raw_runtime{{raw_stream}};
        CommunicatorBackend<RawFacade, decltype(raw_runtime), TagType> raw_backend{raw_facade, raw_runtime};
        const auto raw_ticket = raw_backend.post(raw_tag, {raw_prepared, nullptr, AddressSpace::device},
                                                  raw_bytes, {true, false});
        raw_overlap();
        const auto raw_view = raw_backend.wait(raw_ticket, raw_stream);
        if constexpr (Ready) raw_consume(raw_facade.received_view);
        else raw_consume(raw_view.remote);
        raw_backend.release(raw_ticket, raw_stream);
    }
}

} // namespace strix::xport
