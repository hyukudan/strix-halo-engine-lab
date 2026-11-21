#pragma once
#include "strix/dist/communicator.hpp"
#include <hip/hip_runtime.h>

namespace strix::dist {

// One per rank. Collective calls are made in the same order on both ranks.
class Communicator {
public:
    Communicator(Group group, std::unique_ptr<Transport> transport, std::size_t max_bytes);
    // With a device exchange (GPU-direct): exchanges up to its capacity go device buffer to device buffer with no
    // host copy and no CPU wait on the caller's thread; larger ones (and the host-side collectives of the setup)
    // use `transport`. The same calls, in the same order on both ranks.
    Communicator(Group group, std::unique_ptr<Transport> transport, std::unique_ptr<DeviceExchange> direct, std::size_t max_bytes);
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    // Resize host capacity between collectives; preserves sequence numbers and the route table. grow_pipeline = false:
    // only the (fused) staging buffers grow; the chunked host pipeline keeps its capacity and messages above it take the
    // plain host path (for large staged messages only, without a second pair of pipeline buffers).
    void resize_bytes(std::size_t bytes, bool grow_pipeline = true);
    [[nodiscard]] std::size_t max_bytes() const noexcept { return max_bytes_; }

    // Sends `bytes` of device buffer `local` to the peer and writes the peer's `bytes` into device buffer
    // `remote`. The copy of `local` is ordered after its producer on `stream` and the CPU waits for it
    // before sending; the copy into `remote` is enqueued on `stream` (later work on the stream sees it),
    // and the CPU waits for that copy before the receive buffer is used again by the next exchange. Both
    // ranks pass the same size and tag.
    // STRIX_DIST_HOST_CHUNK_BYTES=N (default 0) pipelines full-host messages above N
    // on separate copy streams, with an independently checked trailer per chunk.
    // Both ranks must use the same N; direct/mixed routes and staged exchanges keep
    // their existing protocol. No consumer work is released after a failed chunk.
    // STRIX_DIST_HOST_CRC_OVERLAP=1 additionally prepares TX CRCs and validates RX
    // chunks on persistent workers while the progress thread moves other chunks.
    void exchange(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag);
    // One-way transfer of `bytes` of a device buffer: the sending rank passes sending = true and its device buffer, the
    // receiving rank sending = false and the buffer to fill; both at the same point with the same size and tag. Nothing
    // travels in the unused direction and no early receive is posted, on every route: host (the sender copies to the
    // pinned buffer and sends payload + trailer, the receiver receives, checks trailer and CRC, copies to the device),
    // pure device (the receiver's GPU checks the in-band identity and checksum; check_exchanges() reports a refusal) and
    // mixed (the host head and the device tail, each one way, at the same time). Deferred CRCs ride with the messages a
    // rank sends: the sender carries its pending one, the receiver checks the one the sender carries. Returns false, with
    // nothing sent, only when the route uses a device engine without a one-way form (the caller then uses exchange()).
    // Both ranks get the same answer (same route table and engine).
    [[nodiscard]] bool transfer(hipStream_t stream, bool sending, void* device, std::size_t bytes, const Tag& tag);
    // The same exchange in two halves, so that work enqueued in between overlaps the transfer. start()
    // enqueues the copy of `local` on `stream` and returns; the progress thread waits on the CPU for that
    // copy, then sends and receives. finish() waits for the peer's message, checks it, and enqueues its
    // copy into `remote` on `stream`. One exchange at a time: start, (enqueue other work), finish.
    void start(hipStream_t stream, const void* local, std::size_t bytes, const Tag& tag);
    void finish(hipStream_t stream, void* remote);
    // Host-resident payloads (control frames, forwarded messages): `local` goes straight into the pinned send buffer and
    // the peer's message is copied out of the pinned receive buffer into `remote` (empty: not wanted), with no device
    // buffer and no device copy on either side of the wire. The message on the wire (payload, trailer, sequence) is the
    // same as exchange()'s (identity, sequence, CRC, and the deferred-CRC carry in both directions), so a peer may use
    // either call for the same collective, also right after ready-view (deferred CRC) exchanges. The calling thread does
    // the transfer; not between start() and finish().
    void exchange_host(std::span<const std::byte> local, std::span<std::byte> remote, const Tag& tag);
    // Agree before an optional collective changes the pair's protocol. Both ranks
    // call this even when locally disabled; either refusal selects the old path.
    [[nodiscard]] bool agree_exchange(bool requested, std::size_t bytes, std::size_t scratch_bytes, const Tag& tag) {
        const std::byte local{static_cast<unsigned char>(requested && bytes != 0U && bytes <= scratch_bytes && bytes <= max_bytes())};
        std::byte peer{};
        exchange_host({&local, 1U}, {&peer, 1U}, tag);
        return local == std::byte{1} && peer == std::byte{1};
    }
    // Per-communicator opt-in, without changing other engines/process environment.
    void enable_async_start() { async_start_ = true; }
    // Enable before the first collective. Per-communicator scheduling policy for messages above GPU ring capacity:
    // start the first chunk before caller work; finish drains it and the tail.
    // Same bytes, chunk identities, route table and buffer lifetime as before.
    void enable_direct_chunk_overlap() { direct_chunk_overlap_ = true; }
    // Fatal teardown: stop and join host submission before caller events/buffers
    // can be retired. The communicator remains poisoned; caller must drain GPU work.
    void cancel_pending(const std::string& reason) noexcept;
    // STRIX_DIST_ASYNC_START=1 or enable_async_start(): full-host chunked path only. Publish remote at start;
    // destination_last_use must already be recorded after its last reader. Records
    // source_free after all off-stream D2H: wait on it before overwriting local.
    // finish only joins validation/copy submission and waits on the final H2D event.
    // One exchange at a time. Returns false (ordinary start contract) on other paths;
    // neither caller event is used then. Keep events/buffers alive until their
    // queued device work completes; finish joins submission, not device execution.
    bool start_async(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag,
                     hipEvent_t destination_last_use, hipEvent_t source_free);
    // GPU-direct: consumes a receive posted early for a guessed next exchange (DeviceExchange::quiesce), so that the
    // session can stay idle. Both ranks at the same point, between exchanges; nothing to do on the host path.
    void quiesce(hipStream_t stream);
    // Relinquish a borrowed transport to another progress owner between complete
    // exchanges: retire deferred validation, direct receives and device readers.
    void release_transport(hipStream_t stream);
    // Opt-in small host path, STRIX_DIST_STAGED_READY_BYTES=1..1048576 (0: unavailable).
    // One stream must outlive the communicator. Kernels write/read the coherent
    // views in place and call publish_staged_block. No HIP calls in start/finish/
    // release on this path. prepare returns {} for routes/sizes not admitted.
    [[nodiscard]] ProducerStage prepare_staged_ready(hipStream_t stream, std::size_t bytes);
    // The largest payload one ready view takes (0: no ready views); callers split larger exchanges into chunks.
    [[nodiscard]] std::size_t staged_ready_capacity() const noexcept { return staged_ready_bytes_ < max_bytes_ ? staged_ready_bytes_ : max_bytes_; }
    // Preflight all distinct sizes before preparing a view (prepare reserves it).
    // Capacity alone does not admit a size: trace and direct routes refuse views.
    // Keep these route guards consistent with prepare_staged_ready.
    [[nodiscard]] std::uint32_t staged_ready_chunk_rows(std::uint32_t rows, std::size_t row_bytes) const {
        if (!rows || !row_bytes || trace_) return 0U;
        const auto limit = staged_ready_capacity() / row_bytes;
        if (!limit) return 0U;
        auto chunk = limit < rows ? static_cast<std::uint32_t>(limit) : rows;
        const auto chunks = 1U + (rows - 1U) / chunk;
        chunk = 1U + (rows - 1U) / chunks;
        const auto tail = rows % chunk;
        if (direct_ && (route(std::size_t{chunk} * row_bytes).direct ||
                       (tail && route(std::size_t{tail} * row_bytes).direct))) return 0U;
        return chunk;
    }
    void start_staged_ready(hipStream_t stream, std::size_t bytes, const Tag& tag);
    [[nodiscard]] ConsumerStage finish_staged_ready();
    void release_staged_ready(hipStream_t stream);
    // Fused staging (host path only): send_staging(bytes) is the pinned send buffer as a device pointer when an exchange
    // of `bytes` would take the host path (nullptr otherwise: use start / finish). The caller's kernel writes the message
    // there on `stream`, then start_staged(stream, bytes, tag) sends it (no device-to-host copy); finish_staged(stream)
    // returns the peer's message in the pinned receive buffer (no host-to-device copy, checked as in finish), valid for
    // kernels enqueued on `stream` before release_staged(stream), which must follow before the next start.
    [[nodiscard]] void* send_staging(std::size_t bytes) const;
    void start_staged(hipStream_t stream, std::size_t bytes, const Tag& tag);
    [[nodiscard]] const void* finish_staged(hipStream_t stream);
    void release_staged(hipStream_t stream);
    [[nodiscard]] const CommunicatorStats& stats() const noexcept { return stats_; }
    // A borrower calls this only after draining its progress thread, with
    // deltas since its previous snapshot. Seconds are progress-thread elapsed
    // time (producer wait + wire + check), overlapping device execution; they
    // are not additional step wall time. The wait is also included in stage.
    void account_borrowed_transport(std::uint64_t exchanges, std::uint64_t bytes,
                                    double producer_wait, double wire, double check) noexcept {
        stats_.exchanges += exchanges;
        stats_.bytes_sent += bytes;
        stats_.seconds += producer_wait + wire + check;
        stats_.stage += producer_wait;
        stats_.wire += wire;
        stats_.check += check;
    }
    [[nodiscard]] const Group& group() const noexcept { return group_; }
    [[nodiscard]] std::string transport_name() const;
    // The transport, for a module that borrows it between this communicator's exchanges (both ranks at the same point,
    // nothing in flight here: e.g. an in-process A/B of XferExchange against CommExchange).
    [[nodiscard]] Transport& transport() noexcept { return *transport_; }
    [[nodiscard]] const DeviceExchange* device_exchange() const noexcept { return direct_.get(); }
    // Whether exchange() of `bytes` takes the host path (no device exchange, or the route table puts this size on the
    // host transport): then exchange_host() moves the same message with no device copy. Both ranks the same answer.
    [[nodiscard]] bool host_routed(std::size_t bytes) const { return !direct_ || !route(bytes).direct; }
    // The size router (with a device exchange): STRIX_DIST_ROUTE=table (default: the measured table; with host
    // streams, small messages on the host path, two rings up to 1 MB, mixed 35 % host above; without, one ring up to
    // 32 KB, two up to 256 KB, every ring above),
    // auto (measured at construction at idle, rank 0 decides), gpu (every size direct, the engine's ring rule), host
    // (the host path up to max_bytes), or an explicit table "UP_TO:host|gpu[:RINGS]|mix:RINGS:HOST_PERCENT,..." (UP_TO
    // in bytes or "inf"; mix: see Route).
    // The host path is never chosen when the transport is only a control channel (TCP rendezvous of the
    // all-GPU-direct mode).
    [[nodiscard]] const std::vector<Route>& routes() const noexcept { return routes_; }
    [[nodiscard]] std::string route_text() const;
    // Replaces the route table (same syntax as an explicit STRIX_DIST_ROUTE). Both ranks, same table, between exchanges.
    void set_routes(const std::string& table);
    // Throws if a device exchange has failed or was refused by the receiving GPU. A failed exchange already stopped
    // its stream (later work does not run and the stream's synchronisation reports an error); this names the cause.
    void check_exchanges() const;

private:
    Group group_;
    xport::HostWaitMetric slot_wait_{"communicator", "slot"};
    xport::HostWaitMetric peer_wait_{"communicator", "peer"};
    xport::HostWaitMetric worker_slot_wait_{"communicator-worker", "slot"};
    std::unique_ptr<Transport> transport_;
    std::size_t max_bytes_;
    std::byte* send_host_ = nullptr;
    std::byte* receive_host_ = nullptr;
    void* send_device_alias_ = nullptr;
    std::size_t d2h_kernel_bytes_ = 0; // STRIX_DIST_HOST_D2H_KERNEL_BYTES, default off; at most 1 MiB
    std::size_t staged_ready_bytes_ = 0, staged_prepared_bytes_ = 0;
    std::byte *staged_send_host_ = nullptr, *staged_receive_host_ = nullptr;
    void *staged_send_device_ = nullptr, *staged_receive_device_ = nullptr;
    StagedReady *producer_flag_ = nullptr, *consumer_flag_ = nullptr;
    StagedReady *producer_flag_device_ = nullptr, *consumer_flag_device_ = nullptr;
    unsigned* staged_counters_ = nullptr;
    std::uint64_t staged_token_ = 0, consumer_pending_token_ = 0;
    hipStream_t staged_stream_ = nullptr;
    bool staged_exposed_ = false, staged_prepared_ = false, staged_started_ = false, staged_receive_held_ = false;
    void wait_staged_flag(StagedReady* flag, std::uint64_t token, std::chrono::steady_clock::time_point deadline);
    // STRIX_DIST_CRC_DEFER (default on, =0 off; both ranks the same), ready-view exchanges only: the TX CRC is computed while
    // the message is on the wire and travels in the NEXT trailer; finish_staged_ready() returns once the trailer's identity
    // is checked, and the progress thread then computes the RX CRC, checked against the peer's next trailer. Every
    // exchange (deferred or not) carries and checks the previous deferred CRC, so a step that ends with an ordinary
    // exchange (the argmax / logits) has every payload checked before its result is returned; a mismatch fails it.
    bool crc_defer_ = false, deferring_ = false, tail_pending_ = false;
    std::uint64_t tx_carry_sequence_ = 0, rx_pending_sequence_ = 0; // sequence + 1; progress thread only
    std::uint32_t tx_carry_crc_ = 0, rx_pending_crc_ = 0;
    void drain_tail(); // waits for the previous job's deferred CRC work (it reads the staged buffers)
    std::uint64_t sequence_ = 0;
    hipEvent_t receive_done_ = nullptr;
    hipEvent_t send_ready_ = nullptr;
    bool receive_pending_ = false;
    bool started_ = false;
    std::size_t bytes_ = 0;
    Tag tag_;
    std::chrono::steady_clock::time_point started_at_{}, returned_at_{};
    // STRIX_DIST_TRACE=FILE: one line per exchange (FILE.r<rank>), wall-clock ns, written when the next exchange
    // starts. Host path ("H"): entry, producer done (seen by the CPU), staged (D2H + CRC done), the peer's staged
    // time, received, validated, consumer (H2D end on the GPU timeline). GPU-direct ("G", from the device exchange):
    // entry, scatter seen, TX submitted, RX complete, TX complete, consumer (gather end).
    std::FILE* trace_ = nullptr;
    hipEvent_t pre_copy_ = nullptr;
    std::uint64_t t_entry_ = 0, t_produced_ = 0, t_staged_ = 0, t_received_ = 0, t_validated_ = 0, peer_staged_ns_ = 0;
    hipEvent_t h2d_done_ = nullptr;
    struct PendingTrace {
        std::uint64_t sequence = 0;
        std::size_t bytes = 0;
        std::uint32_t tag = 0, layer = 0;
        std::uint64_t entry = 0, produced = 0, staged = 0, peer_staged = 0, received = 0, validated = 0;
    } pending_;
    bool trace_pending_ = false;
    void flush_trace();
    void validate_peer(bool deferred = false); // the peer's trailer and CRC of the received message (deferred: identity only)
    // The one trailer builder and peer check of every host message (start_host's job, exchange_host): identity, the
    // carried deferred CRC (taken: tx_carry_ is reset) and, for the peer, the carry against rx_pending_ (cleared).
    // Carry state belongs to the progress thread: callers on the caller thread drain_tail() first.
    [[nodiscard]] Trailer make_trailer(std::size_t bytes, const Tag& tag, std::uint32_t flags);
    void check_peer_trailer(const Trailer& peer, std::size_t bytes, const Tag& tag, bool deferred);
    // Bounded waits (STRIX_DIST_WAIT_TIMEOUT_MS, default 120000): the producer's copy, the reuse of the receive buffer,
    // and the progress job. On expiry (or any failed host exchange) the communicator stops for good, prints its state
    // (the collective in flight and its phase, the last collectives, the streams' progress) to stderr and closes the
    // transport so the peer's pending reads end at once.
    enum class Phase : int { idle = 0, reuse_wait = 1, producer = 2, transfer = 3, check = 4 };
    struct History {
        std::uint64_t sequence = 0, bytes = 0;
        std::uint32_t tag = 0, layer = 0;
        std::uint64_t request = 0, cmd_out = 0, cmd_in = 0; // the first 8 bytes sent / received (a command word)
        char path = 'H';
    };
    std::chrono::milliseconds wait_timeout_{120000};
    std::string terminal_;
    std::atomic<bool> cancel_{false};
    std::atomic<int> phase_{0};
    std::atomic<std::uint64_t> phase_since_{0}, cmd_out_{0}, cmd_in_{0};
    History current_;
    std::array<History, 32> history_{};
    std::size_t history_next_ = 0;
    void set_phase(Phase p);
    [[noreturn]] void stop(const std::string& why);
public:
    // The communicator's state for a failure report: phase, the collective in flight, the last collectives, the
    // transport's progress.
    [[nodiscard]] std::string diagnose() const;
private:
    static bool check_on_caller();
    class Progress;
    std::unique_ptr<Progress> progress_;
    class Pipeline;
    std::unique_ptr<Pipeline> pipeline_;
    bool pipeline_started_ = false;
    bool async_start_ = false;
    void start_pipeline(hipStream_t stream, const void* local, std::size_t bytes, void* remote = nullptr,
                        hipEvent_t destination_last_use = nullptr, hipEvent_t source_free = nullptr);
    void finish_pipeline(hipStream_t stream, void* remote);
    std::unique_ptr<DeviceExchange> direct_;
    bool direct_started_ = false;
    bool direct_chunk_overlap_ = false;
    std::vector<Route> routes_;
    void configure_routes();
    [[nodiscard]] const Route& route(std::size_t bytes) const;
    void start_host(hipStream_t stream, const void* local, std::size_t bytes);
    void finish_host(hipStream_t stream, void* remote);
    bool staged_ = false;  // the exchange in flight is a fused-staging one (start_staged)
    bool fused_ = true;    // fused staging (STRIX_DIST_STAGING=copy: off): staging buffers coherent, send_staging() available
    const void* direct_local_ = nullptr;
    // The device exchange's part: bytes, offset in the message (non-zero when a mixed route sent the head over the
    // host path), and the mixed route's rings.
    std::size_t direct_bytes_ = 0, direct_offset_ = 0;
    unsigned mix_rings_ = 0;
    // The exchange in flight is a one-way transfer(): its direction, else both.
    ExchangeDirection direction_ = ExchangeDirection::both;
    void direct_hint(std::size_t bytes, unsigned rings);
    bool one_way_probed_ = false, one_way_ = false;
    CommunicatorStats stats_;
};

} // namespace strix::dist
