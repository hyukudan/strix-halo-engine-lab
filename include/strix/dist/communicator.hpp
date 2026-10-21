#pragma once

// Two-rank communicator for the distributed runtime: whole-message transports between two ranks and a
// device-buffer exchange on top of them.
//
// Wire format: a message is its payload followed by a 64-byte trailer, so the trailer is the last
// thing to arrive; its sequence number tells a receiver (or, later, a device poll on the receive window)
// that the payload before it is complete, and a mismatch in epoch, plan, rank, tag, sequence, size or
// CRC-32C fails the exchange. Message sizes always come from data both ranks already share (routes, rows,
// plan); no shape metadata travels.
//
// Synchronization contract (also for every future backend): the bytes of a device buffer leave only
// after the producing work is complete as seen by the CPU, and a receive buffer is reused only after the
// work reading it is complete as seen by the CPU. Implicit device-side fences are not relied on: memory
// written by HIP kernels publishes none to the DMA engines.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strix::dist {

inline constexpr std::uint64_t trailer_magic = 0x31544c5254534944ULL; // "DISTRLT1"

struct Trailer {
    std::uint64_t magic = trailer_magic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t request = 0;
    std::uint64_t bytes = 0;
    std::uint32_t rank = 0;
    std::uint32_t tag = 0;
    std::uint32_t layer = 0;
    std::uint32_t crc32c = 0;
};
static_assert(sizeof(Trailer) == 64);

// CRC-32C (Castagnoli) of the payload; the SSE4.2 instruction when the CPU has it.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept;

// Moves whole messages (payload + trailer) between the two ranks, in order. Payloads are host memory.
// Backends: an in-process pair (two ranks in one process, the emulated two-node setup), TCP (bring-up and
// debugging) and the host path of USB4 streams (striped over the streams of a cable).
class Transport {
public:
    virtual ~Transport() = default;
    virtual void send(std::span<const std::byte> message) = 0;
    // Receives exactly message.size() bytes (the size is known from shared data).
    virtual void receive(std::span<std::byte> message) = 0;
    // Sends `out` and receives `in` at the same time (both ranks call it with their own message); the
    // backends do it with persistent progress threads, never one thread per call.
    virtual void exchange(std::span<const std::byte> out, std::span<std::byte> in) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

[[nodiscard]] std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> memory_pair();
// TCP with TCP_NODELAY; the listener accepts one peer. Timeouts bound the connection setup and every
// later blocking send or receive.
[[nodiscard]] std::unique_ptr<Transport> tcp_listen(std::uint16_t port, std::chrono::milliseconds timeout);
[[nodiscard]] std::unique_ptr<Transport> tcp_connect(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout);
// Host path of USB4 streams: `devices` are stream device files, opened read-write (both ranks list the
// paired streams in the same order). A message is split over the streams at 64-byte boundaries, in
// proportion to integer weights (equal when none are given; the trailer ends up in the last part); each
// stream has a persistent thread that writes its part and reads the peer's part of the same message.
[[nodiscard]] std::unique_ptr<Transport> usb4_host(const std::vector<std::string>& devices,
                                                   const std::vector<std::uint32_t>& weights = {});
// The same transport over descriptors that are already open (the streams of a finished pairing, or socket
// pairs in tests); it owns and closes them. Messages shorter than `min_part` bytes per stream use fewer
// streams; a transfer without progress for `timeout` fails the transport.
[[nodiscard]] std::unique_ptr<Transport> usb4_host_descriptors(std::vector<int> fds, std::vector<std::string> names,
                                                               std::vector<std::uint32_t> weights, std::size_t min_part,
                                                               std::chrono::milliseconds timeout);

// Automatic pairing of every live USB4 stream between the two ranks (any cabling: straight or crossed,
// any number of cables, 20 or 40 Gb/s links).
//   1. Candidates: the stream entries of the layout file (keys "<service>/<stream>", e.g. "0-2.0/data0",
//      with "device"); a stream is opened only if the layout does not mark it unattached or unopenable
//      and its service is present in /sys/bus/thunderbolt/devices (a peer is attached on that port).
//      Link speed, lanes and the peer host come from sysfs; the layout's v2 fields are used when present.
//   2. Handshake on every opened stream (repeated until answered): protocol version, rank, process
//      nonce, local stream index and device number, host name; then an acknowledgement echoing the
//      peer's nonce and stream index, which proves both directions. Streams that stay silent or answer
//      one way only are closed and reported; streams answered by a different peer process or the same
//      rank fail the pairing.
//   3. The accepted pairs are ordered by rank 0's stream index; streams of one cable (one XDomain) are
//      grouped, and each cable's throughput is measured with a short exchange on all its streams.
//   4. Rank 0 sends the plan (stream order and integer striping weights, proportional to the measured
//      cable throughput and shared equally among the cable's streams); rank 1 checks it against its own
//      pairing and echoes it.
// The detected topology is returned in `topology` (and its receipt() is the log line).
struct Usb4AutoOptions {
    std::uint32_t rank = 0;
    std::string layout;                          // empty: $STRIX_DIST_USB4_LAYOUT or /run/usb4stream-layout.json
    std::string sysfs = "/sys/bus/thunderbolt/devices";
    // Streams never to open (device files or layout keys); $STRIX_DIST_USB4_EXCLUDE (comma-separated) adds to it.
    std::vector<std::string> exclude;
    std::chrono::milliseconds timeout{60000};    // for the peer to appear on its first stream
    std::chrono::milliseconds grace{5000};       // for the remaining streams once one stream is paired
    std::size_t probe_bytes = std::size_t{16} << 20U; // per cable and direction; 0: weights from the link speed
    // Messages shorter than this per stream use fewer streams (rank 0's value, carried in the plan; 0 = always all;
    // $STRIX_DIST_USB4_MIN_PART overrides).
    std::size_t min_part_bytes = 0;
    // A transfer that makes no progress for this long fails the transport ($STRIX_DIST_USB4_TIMEOUT_MS overrides).
    std::chrono::milliseconds io_timeout{120000};
};
struct Usb4StreamPair {
    std::string key;           // layout key of the local stream
    std::string device;        // local device file
    std::string xdomain;       // local XDomain (one per cable)
    std::string cable;         // the layout's cable label, if any
    std::int32_t domain = -1;
    std::int32_t usb4_port = -1;
    double link_gbps = 0.0;    // raw link rate, min of both directions
    std::uint32_t rank0_index = 0;
    std::int32_t remote_device = -1; // the peer's tbstream number
    std::uint32_t remote_index = 0;
    double measured_gbps = 0.0; // the cable's exchange throughput, one direction
    std::uint32_t weight = 0;
};
struct Usb4Topology {
    std::string local_host, remote_host, remote_unique_id;
    std::vector<Usb4StreamPair> streams;  // in striping order
    std::vector<std::string> rejected;    // candidates not used, with the reason
    std::size_t min_part_bytes = 0;
    [[nodiscard]] std::string receipt() const;
};
[[nodiscard]] std::unique_ptr<Transport> usb4_host_auto(const Usb4AutoOptions& options, Usb4Topology* topology = nullptr);

// Device-memory transport (the zero-copy path of a USB4 stream with device buffers). Planned interface:
// windows are registered once; submit(window, bytes) may be called only after the CPU has seen the
// producer complete, and repost(window) only after the CPU has seen every reader of that window
// complete; completion of a receive is the trailer at the end of the window carrying the expected
// sequence. Not implemented yet.
class DirectTransport {
public:
    virtual ~DirectTransport() = default;
    virtual std::uint32_t register_window(void* device, std::size_t bytes) = 0;
    virtual void submit(std::uint32_t window, std::size_t bytes) = 0;
    virtual void repost(std::uint32_t window) = 0;
    virtual bool received(std::uint32_t window, std::uint64_t sequence) = 0;
};

struct Group {
    std::uint32_t rank = 0;
    std::uint32_t world = 2;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
};
struct Tag {
    std::uint32_t kind = 0;
    std::uint32_t layer = 0;
    std::uint64_t request = 0;
};
enum class TagKind : std::uint32_t { half_sum = 1, raw_slots = 2, argmax = 3, control = 4 };

struct CommunicatorStats {
    std::uint64_t exchanges = 0;
    std::uint64_t bytes_sent = 0;
    double seconds = 0.0;  // wall time inside exchange()
    double stage = 0.0;    // device to pinned host copy, with the CPU wait for it and the producer
    double wire = 0.0;     // transport exchange, including the wait for the peer
    double check = 0.0;    // trailer and CRC checks, and the host to device copy being enqueued
    double overlap = 0.0;  // between start() returning and finish() being called (work enqueued meanwhile)
    double finish_wait = 0.0; // inside finish(), waiting for the progress thread
    std::chrono::steady_clock::time_point last_entry{};
    std::chrono::steady_clock::time_point last_staged{};   // payload staged, sending starts
    std::chrono::steady_clock::time_point last_received{}; // peer message complete
};

// One per rank. Collective calls are made in the same order on both ranks.
class Communicator {
public:
    Communicator(Group group, std::unique_ptr<Transport> transport, std::size_t max_bytes);
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    // Sends `bytes` of device buffer `local` to the peer and writes the peer's `bytes` into device buffer
    // `remote`. The copy of `local` is ordered after its producer on `stream` and the CPU waits for it
    // before sending; the copy into `remote` is enqueued on `stream` (later work on the stream sees it),
    // and the CPU waits for that copy before the receive buffer is used again by the next exchange. Both
    // ranks pass the same size and tag.
    void exchange(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag);
    // The same exchange in two halves, so that work enqueued in between overlaps the transfer. start()
    // enqueues the copy of `local` on `stream` and returns; the progress thread waits on the CPU for that
    // copy, then sends and receives. finish() waits for the peer's message, checks it, and enqueues its
    // copy into `remote` on `stream`. One exchange at a time: start, (enqueue other work), finish.
    void start(hipStream_t stream, const void* local, std::size_t bytes, const Tag& tag);
    void finish(hipStream_t stream, void* remote);
    [[nodiscard]] const CommunicatorStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const Group& group() const noexcept { return group_; }
    [[nodiscard]] std::string transport_name() const { return transport_->name(); }

private:
    Group group_;
    std::unique_ptr<Transport> transport_;
    std::size_t max_bytes_;
    std::byte* send_host_ = nullptr;
    std::byte* receive_host_ = nullptr;
    std::uint64_t sequence_ = 0;
    hipEvent_t receive_done_ = nullptr;
    hipEvent_t send_ready_ = nullptr;
    bool receive_pending_ = false;
    bool started_ = false;
    std::size_t bytes_ = 0;
    Tag tag_;
    std::chrono::steady_clock::time_point started_at_{}, returned_at_{};
    class Progress;
    std::unique_ptr<Progress> progress_;
    CommunicatorStats stats_;
};

} // namespace strix::dist
