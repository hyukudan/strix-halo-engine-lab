#pragma once

// strix/ds41-expert-rpc-v1: the wire between a parts-host (the device that runs the dense parts of every layer and the
// routed experts it keeps resident) and an expert server (devices that keep the other routed experts and compute them
// on request). Portable (std only; POSIX or Winsock sockets), so the CUDA parts-host on Windows and the HIP server on
// Linux share this one file.
//
// One request per MoE layer of a window (one or more rows): for every row its top_k expert ids and routing weights
// (router order), a mask of the slots the server computes, and the row's A8 MoE input. The answer keeps the combine
// of the one-device forward bit for bit, whatever experts are resident where (ds41_moe.hpp):
//   y = (shared + low) + high, low = fold over the row's slots with id < split in ascending id, from 0, each step
//   acc = acc + w * raw (one FP32 product then one FP32 add); high likewise for ids >= split.
// For every (half, row) the server sends, in this order:
//   - FOLD: the fold of the half's leading slots when they are all remote (the partial sum the parts-host continues),
//   - RAW: the unweighted vector of every other remote slot of the half (after the first local one), ascending id.
// Both sides derive the item list from the request alone (plan_response), so the answer carries no shape metadata.
// A fold of a prefix is exact; a fold of an interior run is not (the parts-host's partial sum enters before it), which
// is why later remote slots travel as raw vectors.
//
// Framing: a 32-byte little-endian header (magic "DS4X", version, type, request id, payload bytes, flags, check) and
// the payload. A session starts with HELLO (session token, geometry, model identity); every EXPERTS is answered by
// exactly one RESULT (or ERROR, which ends the session). Requests may be pipelined (several in flight; answers in
// order).

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::ds41::xrpc {

inline constexpr const char* kProtocol = "strix/ds41-expert-rpc-v1";
inline constexpr std::uint32_t kMagic = 0x58345344U;  // "DS4X"
inline constexpr std::uint16_t kVersion = 1;
inline constexpr std::size_t kHeaderBytes = 32;
inline constexpr std::uint64_t kMaxMessageBytes = std::uint64_t{1} << 30U;
inline constexpr std::uint32_t kMaxTopK = 16;

class WireError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class MsgType : std::uint16_t {
    hello = 1,      // parts-host -> server
    hello_ack = 2,  // server -> parts-host
    experts = 3,    // parts-host -> server: one layer of a window
    result = 4,     // server -> parts-host
    ping = 5,       // answered by pong (JSON status)
    pong = 6,
    close = 7,      // answered by ack; the session ends
    ack = 8,
    error = 9,      // server -> parts-host: UTF-8 text; the session ends
};
[[nodiscard]] const char* to_string(MsgType type) noexcept;

struct Header {
    std::uint32_t magic = kMagic;
    std::uint16_t version = kVersion;
    MsgType type = MsgType::error;
    std::uint64_t request = 0;
    std::uint64_t bytes = 0;  // payload bytes after the header
    std::uint32_t flags = 0;
};
[[nodiscard]] std::array<std::uint8_t, kHeaderBytes> encode_header(const Header& h);
// Throws WireError on a bad magic, version, check or a payload above max_bytes.
[[nodiscard]] Header decode_header(std::span<const std::uint8_t> bytes, std::uint64_t max_bytes = kMaxMessageBytes);

// Geometry of the routed experts (the model's; both sides compare it in HELLO).
struct Geometry {
    std::uint32_t layers = 40, hidden = 5120, inter = 2304, experts = 384, top_k = 6, split = 192;
    [[nodiscard]] std::uint32_t groups() const noexcept { return hidden / 32U; }  // A8 deltas per row
    [[nodiscard]] std::uint64_t a8_row_bytes() const noexcept { return std::uint64_t{hidden} + 4U * groups(); }
    [[nodiscard]] std::uint64_t vector_bytes() const noexcept { return std::uint64_t{hidden} * 4U; }
    friend bool operator==(const Geometry&, const Geometry&) = default;
};

// What one server process answers for: experts [expert_first, expert_end) of the layers in layer_mask (bit l).
struct Residency {
    std::uint32_t expert_first = 0, expert_end = 0;
    std::uint64_t layer_mask = 0;
    [[nodiscard]] bool serves(std::uint32_t layer, std::uint32_t expert) const noexcept {
        return layer < 64U && ((layer_mask >> layer) & 1U) != 0U && expert >= expert_first && expert < expert_end;
    }
};

struct Hello {
    std::array<std::uint8_t, 32> token{};  // session secret (--token-file)
    std::uint64_t model_identity = 0;      // the GGUF layout identity, both sides compare (0: not checked)
    Geometry geometry;
    std::string client, numeric_identity;  // e.g. "cuda-parts-host", "ds41-exact-v1"
};
struct HelloAck {
    std::uint32_t status = 0;  // 0 accepted, 1 token, 2 geometry, 3 model
    std::uint64_t model_identity = 0;
    Geometry geometry;
    Residency residency;
    std::uint32_t world = 1;     // devices behind this endpoint (relay: 2)
    std::uint32_t max_rows = 64; // rows per EXPERTS the server takes
    std::string engine, numeric_identity, message;
};

// EXPERTS flags
inline constexpr std::uint32_t kNullCompute = 1U;  // answer zeros of the planned size without computing (wire probes)
inline constexpr std::uint32_t kNoFold = 2U;       // every remote slot as RAW (no FOLD items)

// One layer of a window. Row r: ids[r * top_k + k], weights[r * top_k + k] (router order), remote[r] bit k = the
// server computes slot k; codes[r * hidden + i] and deltas[r * groups + g] its A8 MoE input.
struct ExpertsRequest {
    std::uint32_t layer = 0, rows = 0, top_k = 0, flags = 0;
    std::vector<std::uint32_t> ids;
    std::vector<float> weights;
    std::vector<std::uint32_t> remote;
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
};

// One vector of the answer.
struct PlanItem {
    enum class Kind : std::uint8_t { fold = 1, raw = 2 };
    Kind kind = Kind::raw;
    std::uint8_t half = 0;
    std::uint32_t row = 0;
    std::uint32_t first = 0, count = 1;  // fold: slots order[first .. first + count) of the row's half sequence; raw: count 1
    std::uint32_t slot = 0;              // raw: the slot (r * top_k + k); fold: the first slot
};
// The answer of a server for a request: items in (half, row, ascending id) order. order[r * top_k + j] are the row's
// slots in ascending expert id; low[r] how many have id < split. Every remote slot must be served by `residency`
// (WireError otherwise).
struct ResponsePlan {
    std::vector<PlanItem> items;
    std::vector<std::uint32_t> order, low;
    // fold items: the slots of each (in ascending id), flattened: item i uses fold_slots[fold_begin[i] ..
    // fold_begin[i] + count) (fold_begin is 0 for raw items).
    std::vector<std::uint32_t> fold_slots, fold_begin;
    // the distinct (row, slot) entries to compute, ascending (expert, row): the server's work list
    std::vector<std::uint32_t> entry_slots;
};
[[nodiscard]] ResponsePlan plan_response(const ExpertsRequest& req, const Geometry& g, const Residency& residency);

struct ResultTiming {
    std::uint64_t server_ns = 0;   // request fully received -> answer starts to leave
    std::uint64_t compute_ns = 0;  // device work (upload, kernels, download)
    std::uint64_t relay_ns = 0;    // relay mode: waiting for the peer's part
};
struct ExpertsResult {
    std::uint32_t layer = 0, items = 0, hidden = 0;
    ResultTiming timing;
    std::vector<float> vectors;  // items x hidden
};

[[nodiscard]] std::vector<std::uint8_t> encode(const Hello& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const HelloAck& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const ExpertsRequest& m, const Geometry& g);
// The result payload without the vectors (they follow; see Connection::send_parts).
[[nodiscard]] std::vector<std::uint8_t> encode_result_head(std::uint32_t layer, std::uint32_t items, std::uint32_t hidden,
                                                           const ResultTiming& t);
[[nodiscard]] Hello decode_hello(std::span<const std::uint8_t> p);
[[nodiscard]] HelloAck decode_hello_ack(std::span<const std::uint8_t> p);
[[nodiscard]] ExpertsRequest decode_experts(std::span<const std::uint8_t> p, const Geometry& g);
[[nodiscard]] ExpertsResult decode_result(std::span<const std::uint8_t> p);
inline constexpr std::size_t kResultHeadBytes = 40;
// Result fields in place (no copy of the vectors): the payload's vectors start at kResultHeadBytes.
struct ResultView {
    std::uint32_t layer = 0, items = 0, hidden = 0;
    ResultTiming timing;
    const float* vectors = nullptr;  // unaligned: copy before arithmetic on strict platforms
};
[[nodiscard]] ResultView view_result(std::span<const std::uint8_t> p);

// Bytes of a request / an answer (header included): what a layer costs on the wire.
[[nodiscard]] std::uint64_t request_bytes(std::uint32_t rows, std::uint32_t top_k, const Geometry& g) noexcept;
[[nodiscard]] std::uint64_t result_bytes(std::uint32_t items, const Geometry& g) noexcept;

// ---- the reference arithmetic (host): the server's FOLD items, and the parts-host's combine with the answer ----

// out = fold of w[slot] * raw[slot] over the item's slots (from 0). raw_of(slot) gives the slot's vector.
template <class RawOf>
void fold_item(const ResponsePlan& plan, std::size_t item, std::span<const float> weights, std::uint32_t hidden, RawOf raw_of,
               float* out) {
    const auto& it = plan.items[item];
    for (std::uint32_t o = 0; o < hidden; ++o) out[o] = 0.0F;
    for (std::uint32_t j = 0; j < it.count; ++j) {
        const auto slot = plan.fold_slots[plan.fold_begin[item] + j];
        const float w = weights[slot];
        const float* v = raw_of(slot);
        for (std::uint32_t o = 0; o < hidden; ++o) {
            const float term = w * v[o];
            out[o] = out[o] + term;
        }
    }
}

// The parts-host's combine of one window from its local raw vectors and the servers' answers. For every row and
// half, the seed is the half's FOLD item (if any) and the fold continues over the remaining slots with the local raw
// vector or the answer's RAW item. A request may carry a subset of the window's rows: row_map[request row] = window
// row (empty: the same rows).
struct Answer {
    const ResponsePlan* plan = nullptr;
    const float* vectors = nullptr;  // items x hidden
    std::span<const std::uint32_t> row_map;
};
// Per (row, half) and per slot of the window: where each vector comes from. Built once per window from the plans.
struct CombinePlan {
    std::uint32_t rows = 0, top_k = 0;
    std::vector<std::uint32_t> order, low;  // the window's: order[r * top_k + j] ascending id, low[r] ids < split
    // per (row, half) [r * 2 + h]: the seed (-1 none, else answer << 24 | item) and how many of the half's leading
    // slots it covers
    std::vector<std::int64_t> seed;
    std::vector<std::uint32_t> start;
    // per window slot: -1 local, else answer << 24 | item (its RAW vector)
    std::vector<std::int64_t> source;
};
// remote[r] bit k: slot k of window row r is computed by some server. Throws WireError unless every remote slot is
// covered exactly once and no local slot is.
[[nodiscard]] CombinePlan plan_combine(std::uint32_t rows, std::uint32_t top_k, std::span<const std::uint32_t> ids,
                                       std::span<const std::uint32_t> remote, std::uint32_t split, std::span<const Answer> answers);
// y[r][o] = (shared + low) + high; local_raw(slot) for local slots. Exact host reference of the parts-host combine.
template <class LocalRaw>
void combine_host(const CombinePlan& cp, std::span<const Answer> answers, std::span<const float> weights,
                  std::span<const float> shared, std::uint32_t hidden, LocalRaw local_raw, float* y) {
    const auto vec = [&](std::int64_t src) {
        const auto a = static_cast<std::size_t>(src >> 24), i = static_cast<std::size_t>(src & 0xffffff);
        return answers[a].vectors + i * hidden;
    };
    std::vector<float> half_sum(2U * hidden);
    for (std::uint32_t r = 0; r < cp.rows; ++r) {
        for (std::uint32_t h = 0; h < 2U; ++h) {
            float* acc = half_sum.data() + std::size_t{h} * hidden;
            const auto s = cp.seed[r * 2U + h];
            if (s >= 0) {
                const float* v = vec(s);
                for (std::uint32_t o = 0; o < hidden; ++o) acc[o] = v[o];
            } else {
                for (std::uint32_t o = 0; o < hidden; ++o) acc[o] = 0.0F;
            }
            const auto first = h == 0U ? 0U : cp.low[r], end = h == 0U ? cp.low[r] : cp.top_k;
            for (auto k = first + cp.start[r * 2U + h]; k < end; ++k) {
                const auto slot = cp.order[r * cp.top_k + k];
                const float w = weights[slot];
                const float* v = cp.source[slot] >= 0 ? vec(cp.source[slot]) : local_raw(slot);
                for (std::uint32_t o = 0; o < hidden; ++o) {
                    const float term = w * v[o];
                    acc[o] = acc[o] + term;
                }
            }
        }
        for (std::uint32_t o = 0; o < hidden; ++o) {
            const float a = shared[std::size_t{r} * hidden + o] + half_sum[o];
            y[std::size_t{r} * hidden + o] = a + half_sum[hidden + o];
        }
    }
}

// Session token: 32 bytes as 64 hex characters (whitespace ignored).
[[nodiscard]] std::array<std::uint8_t, 32> read_token_file(const std::string& path);
[[nodiscard]] bool same_token(const std::array<std::uint8_t, 32>& a, const std::array<std::uint8_t, 32>& b) noexcept;
[[nodiscard]] std::uint64_t fnv_floats(std::span<const float> v) noexcept;

// ---- TCP (POSIX or Winsock; TCP_NODELAY; optional busy-poll receive) ----

struct Message {
    Header header;
    std::vector<std::uint8_t> payload;
};

class Connection {
public:
    Connection(std::intptr_t fd, std::chrono::milliseconds timeout);
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    void send(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::uint32_t flags = 0);
    // One message from several buffers (header + parts, one gathered write where the platform has it).
    void send_parts(MsgType type, std::uint64_t request, std::span<const std::span<const std::uint8_t>> parts, std::uint32_t flags = 0);
    // A message sent in pieces (streamed answers): the header with the whole payload size and the first parts in one
    // gathered write, then the rest with send_raw until `bytes` have gone.
    void send_begin(MsgType type, std::uint64_t request, std::uint64_t bytes, std::span<const std::span<const std::uint8_t>> first,
                    std::uint32_t flags = 0);
    void send_raw(std::span<const std::uint8_t> bytes);
    // Waits for one whole message. busy_poll: spin on non-blocking reads (no sleep in the kernel) until data arrives;
    // the timeout still bounds the wait (0 = forever).
    [[nodiscard]] Message receive(std::uint64_t max_bytes = kMaxMessageBytes);
    // Receives into a caller buffer reused across calls (no allocation once large enough).
    Header receive_into(std::vector<std::uint8_t>& payload, std::uint64_t max_bytes = kMaxMessageBytes);
    // Busy-poll: the socket becomes non-blocking and reads/writes spin (no sleep in the kernel) within the timeout.
    void set_busy_poll(bool on);
    [[nodiscard]] std::string peer() const { return peer_; }
    [[nodiscard]] std::uint64_t bytes_sent() const noexcept { return sent_; }
    [[nodiscard]] std::uint64_t bytes_received() const noexcept { return received_; }

private:
    void write_gather(std::span<const std::span<const std::uint8_t>> parts);
    void read_all(std::uint8_t* p, std::size_t n);
    std::intptr_t fd_ = -1;
    std::chrono::milliseconds timeout_{0};
    bool busy_poll_ = false;
    std::string peer_;
    std::uint64_t sent_ = 0, received_ = 0;
};

class Listener {
public:
    Listener(const std::string& host, std::uint16_t port);  // port 0: any free port (port())
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    [[nodiscard]] std::unique_ptr<Connection> accept(std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout);
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

private:
    std::intptr_t fd_ = -1;
    std::uint16_t port_ = 0;
};

// "HOST:PORT"; retries until `wait` elapses (the server may still be loading).
[[nodiscard]] std::unique_ptr<Connection> connect(const std::string& host_port, std::chrono::milliseconds wait,
                                                  std::chrono::milliseconds io_timeout);
[[nodiscard]] std::pair<std::string, std::uint16_t> split_host_port(const std::string& s);

// ---- client ----

// One server session: HELLO, then EXPERTS requests (pipelined: submit several, collect answers in order).
class ExpertClient {
public:
    ExpertClient(const std::string& host_port, const Hello& hello, std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout,
                 bool busy_poll = false);
    ~ExpertClient();
    [[nodiscard]] const HelloAck& ack() const noexcept { return ack_; }
    // Sends one layer request; returns its request id.
    std::uint64_t submit(const ExpertsRequest& req);
    // The next answer (in submission order) into `payload` (reused); view_result(payload) reads it.
    ResultView collect(std::vector<std::uint8_t>& payload);
    std::string ping();
    void close();
    [[nodiscard]] Connection& connection() noexcept { return *conn_; }

private:
    std::unique_ptr<Connection> conn_;
    HelloAck ack_;
    Geometry geometry_;
    std::uint64_t next_ = 1, expected_ = 1;
    std::vector<std::uint8_t> scratch_;
};

}  // namespace strix::ds41::xrpc
