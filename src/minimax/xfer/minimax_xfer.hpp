#pragma once
// MiniMax M3.1 TP2 exchange module (qualification notes).
//
// MiniMax-specific layer above the shared USB4 infrastructure (strix::dist::Transport: the usb4stream host path,
// its streams, striping and eager mode). What is exchanged and when is MiniMax's schedule (MmSchedule below); this
// file is the engine that moves those messages with no CPU in the GPU's dependency chain:
//
//   producer kernel(s) write the message straight into a pinned, host-coherent TX slot (tx()), and the TX flag is
//   published after them on the stream (a 1-block publish kernel, a stream write, or the producer's own epilogue via
//   minimax_xfer_device.hpp); the progress thread, which already holds the job, sees the flag, appends the 64-byte
//   trailer (sequence, identity, CRC-32C), moves the message with Transport::exchange (eager on that thread for
//   decode sizes), checks the peer's trailer and writes the RX flag; the stream, which the CPU enqueued long before,
//   waits for the RX flag (hipStreamWaitValue64) and the consumer kernels read the peer's payload in place from the
//   RX slot (no host-to-device copy).
//
// The CPU never waits inside a decode step, so the whole token (all 60 layers) can be enqueued ahead and the GPU runs
// segment after segment with only the wire in between. Bytes are moved, never changed: reductions are the consumers'
// (fixed order h + (part0 + part1), part0 = rank 0's), so the residual is bitwise the one of the synchronous path.
//
// Rules (checked where possible, both ranks make the same calls in the same order with the same sizes and tags):
//   * one in-order stream (Options::stream) carries producers, publishes, waits and consumers;
//   * tx() of sequence s is called (and its producers enqueued) only after recv() of sequence s - depth;
//   * the view returned by recv() of sequence s is valid for work enqueued before send() of sequence s + depth;
//   * results are trusted only after check() (or drain()) at a step boundary: a failed exchange releases the stream
//     with garbage instead of hanging it, and check() names the cause.
// Environment (both ranks the same): STRIX_MMX_DEPTH (default 4), STRIX_MMX_PUBLISH=kernel|write_value (default
// kernel), STRIX_MMX_CRC=0|1|deferred (default 1 = inline), STRIX_MMX_TX_MEM / STRIX_MMX_RX_MEM=host|device
// (default host), STRIX_MMX_PROGRESS_CPU=N, STRIX_MMX_CRC_CPU=N, STRIX_MMX_IDLE_SPIN_US (default 3000),
// STRIX_MMX_TIMEOUT_MS (default 120000).

#include "strix/dist/communicator.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strix::minimax::xfer {

inline constexpr std::size_t kHidden = 6144;
inline constexpr std::size_t kRowF32 = kHidden * sizeof(float);  // one hidden-state partial row, FP32: 24576 B

// The MiniMax decode schedule (design and proof in design notes): per layer exactly two reductions of FP32
// partial rows (after o_proj, after the FFN) and one 8-byte (max, id) exchange per step for the vocabulary-split
// head; N users ride in one message (N rows). Router, routing and indexer are replicated: no metadata is exchanged.
struct MmSchedule {
    static constexpr unsigned kLayers = 60;
    static constexpr unsigned kPerLayer = 2;
    static constexpr unsigned kPerStep = kLayers * kPerLayer + 1;  // 121
    [[nodiscard]] static constexpr std::size_t row_bytes(unsigned rows) { return std::size_t{rows} * kRowF32; }
    static constexpr std::size_t kArgmaxBytes = 8;
};

enum class Kind : std::uint32_t { attn_sum = 1, ffn_sum = 2, argmax = 3, control = 4, gather = 5, test = 15 };
struct Tag {
    Kind kind = Kind::test;
    std::uint32_t layer = 0;
    std::uint64_t step = 0;
};

inline constexpr std::uint64_t kTrailerMagic = 0x3146584d58494d4dULL;  // "MMIXMXF1"
struct Trailer {
    std::uint64_t magic = kTrailerMagic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t step = 0;
    std::uint32_t bytes = 0;
    std::uint16_t kind = 0;
    std::uint16_t flags = 0;        // kTrailerDeferred, kTrailerPrev
    std::uint32_t layer = 0;
    std::uint32_t rank = 0;
    std::uint32_t crc32c = 0;       // this payload's CRC-32C (0 when deferred)
    std::uint32_t prev_crc32c = 0;  // the previous payload's CRC-32C when that one was deferred (kTrailerPrev)
};
static_assert(sizeof(Trailer) == 64);
inline constexpr std::uint16_t kTrailerDeferred = 1, kTrailerPrev = 2;
// Messages up to this size always carry their own CRC (the argmax / control messages); larger ones defer it when
// Options::crc_deferred: the CRC of message k travels in message k+1's trailer (or in a seal at drain()), computed by
// a worker thread while the wire moves the payload (TX) and after the stream was released (RX). A corrupted payload
// still fails the step (check() / drain()), one exchange later, before any result is trusted.
inline constexpr std::size_t kInlineCrcMax = 1024;

// Where the TX / RX slots live. host_coherent: hipHostMalloc coherent+mapped (CPU-cached, GPU uncached). device_fine:
// fine-grained device memory (hipExtMallocWithFlags hipDeviceMallocFinegrained), which libraries that only accept
// device pointers (strix_gemm) can write directly; the CPU reads / writes it in place.
// device_uncached: hipDeviceMallocUncached. Measurement found device_fine UNSAFE for the CPU side (stale TX bytes read
// after the flag; see design notes): only host_coherent is a default; the device kinds are for the coherence probe.
enum class Memory : std::uint32_t { host_coherent = 0, device_fine = 1, device_uncached = 2 };

// How the TX flag of a message is written once its producers are enqueued.
enum class Publish : std::uint32_t {
    kernel = 0,       // a 1-block kernel after the producers (system-scope release store)
    write_value = 1,  // hipStreamWriteValue64 after the producers
    producer = 2,     // the producer's own last block publishes (minimax_xfer_device.hpp), nothing is enqueued
};

// A received message for consumers that wait for it themselves (mmx_wait in minimax_xfer_device.hpp) instead of a
// stream wait packet: every block of the first consumer kernel calls mmx_wait(view) before reading the payload.
struct RxView {
    const void* payload = nullptr;
    const std::uint64_t* flag = nullptr;  // device pointer of the RX flag
    std::uint64_t sequence = 0;
    // Device-memory gate (x4: with every block polling the host flag, a 196 KB consumer's 192 pollers slowed the whole
    // exchange 2.4x): block 0 polls the host flag and opens the gate, the other blocks poll the gate in L2. nullptr:
    // every block polls the host flag.
    std::uint64_t* gate = nullptr;
};

// What a producer kernel needs to publish its own message (Publish::producer): see minimax_xfer_device.hpp.
struct ProducerTarget {
    std::uint64_t* flag = nullptr;    // device pointer of the TX flag
    unsigned* counter = nullptr;      // device scratch, reset by the last block
    std::uint64_t sequence = 0;       // value to publish
};

struct Options {
    std::uint32_t rank = 0;
    std::uint64_t epoch = 1;
    std::uint64_t plan = 0x46584d4dULL;  // "MMXF"
    std::size_t capacity = 0;            // largest payload in bytes
    unsigned depth = 4;                  // TX / RX slots
    hipStream_t stream = nullptr;
    Publish publish = Publish::kernel;   // default for send()
    bool crc = true;
    bool crc_deferred = true;             // CRC off the critical path (see kInlineCrcMax); x4: exact, never slower
                                          // (N=1 -24 %, N=8 -36 % exchange cost); STRIX_MMX_CRC=1 restores inline
    Memory tx_memory = Memory::host_coherent, rx_memory = Memory::host_coherent;
    int progress_cpu = -1;
    int crc_cpu = -1;
    std::uint64_t test_corrupt_sequence = 0;  // tests only (STRIX_MMX_TEST_CORRUPT): flip one received byte of this sequence
    unsigned idle_spin_us = 3000;        // progress thread: spin this long without a job before sleeping
    std::chrono::milliseconds timeout{120000};
    // Reads the STRIX_MMX_* variables over these values.
    void from_env();
};

struct Stats {
    std::uint64_t exchanges = 0, bytes = 0;
    double producer_wait = 0.0;  // progress thread: job taken -> TX flag seen (GPU work still running), seconds
    double wire = 0.0;           // CRC + Transport::exchange (includes the wait for the peer's message)
    double check = 0.0;          // peer trailer + CRC
};

class Exchanger {
public:
    Exchanger(std::unique_ptr<dist::Transport> owned, Options options);
    Exchanger(dist::Transport& borrowed, Options options);
    ~Exchanger();
    Exchanger(const Exchanger&) = delete;
    Exchanger& operator=(const Exchanger&) = delete;

    // Payload buffer (device-accessible, host-coherent) of the next exchange; producers write the message here.
    [[nodiscard]] void* tx();
    // For a producer that publishes itself (Publish::producer): the flag, its counter and the value, for the next send.
    [[nodiscard]] ProducerTarget producer_target();
    // The message of `bytes` is produced by the work enqueued so far on the stream: publish it (per `how`) and hand
    // the transfer to the progress thread. Never blocks.
    void send(std::size_t bytes, const Tag& tag, Publish how);
    void send(std::size_t bytes, const Tag& tag) { send(bytes, tag, options_.publish); }
    // Copy `bytes` of device buffer `local` into tx() with one kernel whose last block publishes, then send.
    void send_copy(const void* local, std::size_t bytes, const Tag& tag);
    // The stream waits for the peer's message of the oldest send() not yet received; returns its payload (RX slot).
    [[nodiscard]] const void* recv();
    // As recv() without the stream wait packet: the consumer kernels wait in-kernel (mmx_wait), saving the command
    // processor's wake-up and the launch behind it. Same lifetime rules as recv().
    [[nodiscard]] RxView recv_view();
    // The local payload of the exchange recv() just returned (its TX slot), for consumers of (part0, part1).
    [[nodiscard]] const void* last_local() const noexcept { return last_local_; }
    // CPU: waits until every posted exchange is complete, then check(). Between steps / requests only.
    void drain();
    // Throws with the cause if an exchange failed (the stream was released with garbage).
    void check() const;
    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::uint64_t sent() const noexcept { return next_; }
    [[nodiscard]] const Options& options() const noexcept { return options_; }
    [[nodiscard]] std::string name() const;

private:
    struct Job {
        std::uint64_t sequence = 0;
        std::size_t bytes = 0;
        Tag tag;
        bool seal = false;  // host-only 0-byte message carrying the last deferred CRC (drain())
    };
    class CrcWorker;
    static constexpr std::size_t kQueue = 256;
    void init();
    void run();
    void process(const Job& job);
    void post(const Job& job);
    [[noreturn]] void fail(const std::string& why);
    std::byte* tx_slot(std::uint64_t sequence) const;
    std::byte* rx_slot(std::uint64_t sequence) const;

    std::unique_ptr<dist::Transport> owned_;
    dist::Transport* transport_ = nullptr;
    Options options_;
    std::size_t stride_ = 0;
    std::byte* tx_host_ = nullptr;
    std::byte* rx_host_ = nullptr;
    std::byte* tx_dev_ = nullptr;
    std::byte* rx_dev_ = nullptr;
    void* tx_alloc_ = nullptr;
    void* rx_alloc_ = nullptr;
    std::uint64_t* flags_host_ = nullptr;  // [0] TX flag (GPU writes), [8] RX flag (CPU writes)
    std::uint64_t* flags_dev_ = nullptr;
    unsigned* counter_dev_ = nullptr;
    std::uint64_t* gate_dev_ = nullptr;
    std::uint64_t next_ = 0;       // sequences 1.. ; next_ = last sequence handed out by send()
    std::uint64_t received_ = 0;   // last sequence whose recv() was enqueued
    std::uint64_t tx_taken_ = 0;   // last sequence whose tx() was handed out
    std::uint64_t jobs_posted_ = 0;  // caller side job counter (sends + seals)
    bool tail_deferred_ = false;     // caller side: the last send's CRC is deferred and not yet sealed
    // progress thread, deferred CRCs
    bool prev_deferred_ = false;
    std::uint32_t prev_tx_crc_ = 0;
    std::uint64_t prev_rx_ticket_ = 0, prev_sequence_ = 0;
    std::byte seal_tx_[64 + 64] = {}, seal_rx_[64 + 64] = {};
    std::unique_ptr<CrcWorker> crc_worker_;
    const void* last_local_ = nullptr;
    Job jobs_[kQueue];
    std::atomic<std::uint64_t> posted_{0}, done_{0};
    std::atomic<bool> stop_{false}, failed_{false}, sleeping_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::string failure_;
    mutable std::mutex failure_mutex_;
    Stats stats_;
    std::thread thread_;
};

// Greedy head split by vocabulary (rank 0 owns ids [0, vocab0), rank 1 the rest): one exchange of `rows` x 8 bytes.
// Each rank's candidate per row is (max, id) over its logits with the first maximum winning (lowest id: the host scan
// `if (l[i] > l[best]) best = i`), written into the TX slot by one kernel that publishes itself; after the exchange a
// pick kernel writes per row the peer-agreed token, rank 0 winning ties (lower ids), into `out_ids` (device memory),
// so the next step's embedding can be read on the GPU without the CPU. A row with any non-finite logit on either rank
// gives kNonFinite (the caller treats it as an error after the step). `logits` is rows x ld, the first vn of each row.
inline constexpr std::uint32_t kNonFinite = 0xFFFFFFFFU;
void argmax_exchange(Exchanger& x, const float* logits, unsigned rows, unsigned vn, unsigned ld, std::uint32_t id0, std::uint64_t step,
                     std::uint32_t* out_ids);

}  // namespace strix::minimax::xfer
