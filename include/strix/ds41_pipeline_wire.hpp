#pragma once

#include "strix/deepseek41_model.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// strix/ds41-pipeline-v1: the wire between two pipeline stages of the deepseek41 decode. The FRONT runs the embedding
// and target layers [0, K); the TAIL (role pipeline-stage) runs layers [K, n) and the head and answers with the greedy
// tokens. Portable (no device code): the engine (DeviceForward), the CPU reference and any other backend of the FRONT
// produce and consume the same messages.
//
// What crosses the cut, per step of R rows at positions p .. p + R - 1 (everything a layer >= K reads that a layer < K
// produced; derived from the model's attention organisation by plan_boundary):
//   - the token ids of the rows (Engram n-grams and the token history of the TAIL);
//   - the residual stream after layer K - 1: R x hc x hidden FP32;
//   - mix_ffn@K-1: the FFN site's hyper-connection mix of layer K - 1 (pre, post, comb; 2 hc + hc^2 FP32 per row): its
//     `pre` collapses the input of layer K's attention;
//   - kv@s for every KV source s < K whose compressed KV / index keys a layer >= K reads: the entries the step completed,
//     [p / ratio, (p + R) / ratio), each kv_entry + idx_entry bytes (FP4 KV + FP4 index key);
//   - topk@s for every index source s < K whose selection a layer >= K attends with: per row the count and the ids;
//   - cand@s when the candidate source is < K and a candidate user is >= K: the per-row candidate block bytes of the step.
// The TAIL keeps a mirror of the KV sources' caches it reads (filled by the kv@s sections), so a step only carries the
// new entries.
//
// Framing: every message is a 32-byte little-endian header (magic, version, type, request id, payload bytes, flags,
// header check) followed by the payload. The FRONT sends requests, the TAIL answers each with exactly one message
// (or ERROR). A session starts with HELLO (session token, cut, geometry, model identity); the TAIL refuses another token.
namespace strix::ds41::pipe {

inline constexpr const char* kProtocol = "strix/ds41-pipeline-v1";
inline constexpr std::uint32_t kMagic = 0x50345344U;  // "DS4P"
inline constexpr std::uint16_t kVersion = 1;
inline constexpr std::size_t kHeaderBytes = 32;
inline constexpr std::uint64_t kMaxMessageBytes = std::uint64_t{1} << 31U;  // state messages; steps are far smaller

class WireError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class MsgType : std::uint16_t {
    hello = 1,         // FRONT -> TAIL
    hello_ack = 2,     // TAIL -> FRONT
    step = 3,          // FRONT -> TAIL: a step of R rows
    result = 4,        // TAIL -> FRONT
    commit = 5,        // FRONT -> TAIL: keep rows [0, kept) of the last step (verify windows), answered by ack
    reset = 6,         // FRONT -> TAIL: new sequence (position 0), answered by ack
    export_state = 7,  // FRONT -> TAIL: answered by state
    state = 8,         // TAIL -> FRONT: manifest + attention state of the TAIL's layers
    import_state = 9,  // FRONT -> TAIL: manifest + attention state; answered by ack
    ack = 10,
    ping = 11,         // answered by pong (JSON status)
    pong = 12,
    close = 13,        // end of session, answered by ack
    error = 14,        // TAIL -> FRONT: UTF-8 message; the session ends
    // speculative extension (ds41_pipeline_dspark.hpp): verify windows go as STEP with kStepVerify
    spec_begin = 15,   // FRONT -> TAIL: start drafting at the committed position, answered by spec_accept (window 0)
    spec_accept = 16,  // TAIL -> FRONT: the decision of a verify window and the plan of the next one
    spec_commit = 17,  // FRONT -> TAIL: the last decision applied (end of a sequence), answered by ack
    // several requests (ds41_pipeline_multi.hpp, strix/ds41-pipeline-multi-v1): the TAIL drives a FRONT of sequence slots
    multi_hello = 18,   // FRONT -> TAIL: HELLO + the FRONT's slots and rows per step, answered by hello_ack
    seq_step = 19,      // TAIL -> FRONT: one slot's rows, answered by seq_boundary
    seq_boundary = 20,  // FRONT -> TAIL: the cut of that step (a STEP payload; header flags = the slot)
    seq_reset = 21,     // TAIL -> FRONT: a slot starts a fresh sequence at position 0, answered by ack
    seq_import = 22,    // TAIL -> FRONT: a slot continues a state (layers [0, cut) used), answered by ack
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
// Throws WireError on a bad magic, version, check or a payload above `max_bytes`.
[[nodiscard]] Header decode_header(std::span<const std::uint8_t> bytes, std::uint64_t max_bytes = kMaxMessageBytes);

// ---- the cut ----

enum class SectionKind : std::uint16_t { mix_ffn = 1, kv_entries = 2, topk = 3, candidates = 4 };
[[nodiscard]] const char* to_string(SectionKind kind) noexcept;

struct BoundaryGeometry {
    std::uint32_t layers = 0, hidden = 0, hc = 0, vocab = 0, topk = 0, candidate_block = 0;
    std::uint32_t kv_entry_bytes = 0, idx_entry_bytes = 0;  // one compressed position: FP4 KV + scales, FP4 index key + scales
    [[nodiscard]] std::uint32_t mix_floats() const noexcept { return 2U * hc + hc * hc; }
    [[nodiscard]] std::uint64_t stream_bytes_per_row() const noexcept { return std::uint64_t{hc} * hidden * 4U; }
};
[[nodiscard]] BoundaryGeometry boundary_geometry(const deepseek41::Config& config);

struct SourceLayer {
    std::uint32_t layer = 0, ratio = 0;
};
// What a cut after layer `cut - 1` carries (cut in [1, layers)).
struct BoundaryPlan {
    std::uint32_t cut = 0;
    std::vector<SourceLayer> kv_sources;     // KV sources < cut read by a layer >= cut (ascending)
    std::vector<SourceLayer> topk_sources;   // index sources < cut whose top-k a layer >= cut attends with
    std::int32_t candidate_source = -1;      // < cut with a candidate user >= cut, else -1
    std::uint32_t candidate_ratio = 0;       // compression ratio of the candidate source's KV source
};
[[nodiscard]] BoundaryPlan plan_boundary(const std::vector<deepseek41::LayerAttributes>& layers, std::uint32_t cut);
// Bytes of one step of `rows` rows at position `position` (header excluded): what the cut costs on the wire.
struct StepBytes {
    std::uint64_t tokens = 0, stream = 0, mix = 0, kv = 0, topk_max = 0, candidates = 0, framing = 0;
    [[nodiscard]] std::uint64_t total() const noexcept { return tokens + stream + mix + kv + topk_max + candidates + framing; }
    [[nodiscard]] std::uint64_t side() const noexcept { return mix + kv + topk_max + candidates; }
};
[[nodiscard]] StepBytes step_bytes(const BoundaryPlan& plan, const BoundaryGeometry& g, std::uint32_t position, std::uint32_t rows);

struct Section {
    SectionKind kind = SectionKind::mix_ffn;
    std::uint32_t layer = 0;
    std::uint32_t first = 0;  // kv_entries: first entry index; candidates: bytes per row
    std::uint32_t count = 0;  // kv_entries: entries; mix_ffn / topk / candidates: rows
    std::vector<std::uint8_t> bytes;
};

// The boundary a FRONT stage hands to the TAIL for one step.
struct Boundary {
    std::uint32_t position = 0, rows = 0;
    std::vector<std::uint32_t> tokens;  // rows
    std::vector<float> stream;          // rows x hc x hidden
    std::vector<Section> sections;
    // decode_step(..., view_stream = true): the stream stays in the received payload (no copy; `stream` is empty and the
    // payload must outlive the boundary). A TAIL receives into pinned memory and uploads the stream from there.
    const float* stream_view = nullptr;
    std::size_t stream_view_floats = 0;
    [[nodiscard]] const float* stream_data() const noexcept { return stream_view ? stream_view : stream.data(); }
    [[nodiscard]] std::size_t stream_floats() const noexcept { return stream_view ? stream_view_floats : stream.size(); }
    [[nodiscard]] const Section* find(SectionKind kind, std::uint32_t layer) const noexcept;
};
// Throws WireError naming the first difference between a boundary and what `plan` says the cut carries (sizes,
// layers, entry ranges, top-k counts).
void validate_boundary(const Boundary& b, const BoundaryPlan& plan, const BoundaryGeometry& g);

// ---- messages ----

inline constexpr std::uint32_t kStepWantLogits = 1U;  // RESULT carries every logit (R = 1 only)
// RESULT carries every row's full-logits FNV also for R > 1 (a TAIL with gathered logits gathers each row's vocabulary
// slices after the step; a check, not the production path)
inline constexpr std::uint32_t kStepWantRowFnv = 2U;

struct Hello {
    std::array<std::uint8_t, 32> token{};  // session secret (--stage-token-file)
    std::uint32_t cut = 0, layers = 0, hidden = 0, hc = 0, vocab = 0, max_positions = 0;
    std::uint64_t model_identity = 0;      // the GGUF layout identity (tensor table + metadata), both sides compare
    std::string backend, numeric_identity;  // the FRONT's ("hip", "host", "cuda"; "ds41-exact-v1", ...)
};
struct HelloAck {
    std::uint32_t status = 0;  // 0 accepted
    std::uint32_t first_layer = 0, end_layer = 0, world = 1, max_positions = 0;
    std::uint64_t model_identity = 0;
    std::string engine, numeric_identity, message;
};
struct StepMsg {
    std::uint32_t flags = 0;
    Boundary boundary;
};
struct Result {
    std::uint32_t position = 0;  // of row 0
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint64_t> logits_fnv;  // per row; 0 where the stage keeps no logits (R > 1)
    std::vector<float> logits;              // kStepWantLogits: vocab values of row 0
};
// state / import_state: a JSON manifest (layer range, position, next token, sampler, model identity) and the attention
// state bytes (deepseek41::serialize_attention_state).
struct StateMsg {
    std::string manifest;
    std::vector<std::uint8_t> state;
};

[[nodiscard]] std::vector<std::uint8_t> encode(const Hello& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const HelloAck& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const StepMsg& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const Result& m);
[[nodiscard]] std::vector<std::uint8_t> encode(const StateMsg& m);
[[nodiscard]] std::vector<std::uint8_t> encode_u32(std::uint32_t v);
[[nodiscard]] std::vector<std::uint8_t> encode_text(const std::string& s);
[[nodiscard]] Hello decode_hello(std::span<const std::uint8_t> p);
[[nodiscard]] HelloAck decode_hello_ack(std::span<const std::uint8_t> p);
// Bounds: at most `max_rows` rows and the geometry's sizes; every length is checked against the payload.
// view_stream: the boundary's stream points into `p` (Boundary::stream_view) instead of a copy.
[[nodiscard]] StepMsg decode_step(std::span<const std::uint8_t> p, const BoundaryGeometry& g, std::uint32_t max_rows, bool view_stream = false);
[[nodiscard]] Result decode_result(std::span<const std::uint8_t> p, std::uint32_t vocab, std::uint32_t max_rows);
[[nodiscard]] StateMsg decode_state(std::span<const std::uint8_t> p);
[[nodiscard]] std::uint32_t decode_u32(std::span<const std::uint8_t> p);
[[nodiscard]] std::string decode_text(std::span<const std::uint8_t> p, std::size_t max_bytes = 1U << 20U);

// Session token: 32 bytes, stored as 64 hex characters (whitespace ignored) in a file readable only by the owner.
[[nodiscard]] std::array<std::uint8_t, 32> read_token_file(const std::filesystem::path& path);
[[nodiscard]] bool same_token(const std::array<std::uint8_t, 32>& a, const std::array<std::uint8_t, 32>& b) noexcept;  // constant time

[[nodiscard]] std::uint64_t fnv_floats(std::span<const float> v) noexcept;  // FNV-1a over the bytes (as the apps' logits_fnv)

// ---- TCP (POSIX sockets; TCP_NODELAY, bounded waits) ----

struct Message {
    Header header;
    std::vector<std::uint8_t> payload;
};

class Connection {
public:
    explicit Connection(int fd, std::chrono::milliseconds timeout);
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    void send(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::uint32_t flags = 0);
    // Tests (a FRONT lost in the middle of a message): the header announces payload.size() bytes, only the first
    // `keep` bytes are sent.
    void send_truncated(MsgType type, std::uint64_t request, std::span<const std::uint8_t> payload, std::size_t keep, std::uint32_t flags = 0);
    // Waits for one whole message (the timeout bounds every blocking read; 0 = wait forever).
    [[nodiscard]] Message receive(std::uint64_t max_bytes = kMaxMessageBytes);
    // The same in two parts, so the caller chooses where the payload lands (e.g. pinned host memory): the header (when
    // `first_byte` is given, the time its first byte arrived), then exactly header.bytes payload bytes into `out`.
    [[nodiscard]] Header receive_header(std::uint64_t max_bytes = kMaxMessageBytes, std::chrono::steady_clock::time_point* first_byte = nullptr);
    void receive_payload(std::span<std::uint8_t> out);
    // Busy polling: reads spin on non-blocking recv (no sleep in the kernel's wait queue; one core spins while waiting)
    // and SO_BUSY_POLL is requested (best effort). The timeout still bounds each wait.
    void set_busy_poll(bool on);
    [[nodiscard]] bool busy_poll() const noexcept { return busy_; }
    [[nodiscard]] std::string peer() const { return peer_; }
    [[nodiscard]] std::uint64_t bytes_sent() const noexcept { return sent_; }
    [[nodiscard]] std::uint64_t bytes_received() const noexcept { return received_; }

private:
    void read_all(std::uint8_t* p, std::size_t n, std::chrono::steady_clock::time_point* first_byte);
    int fd_ = -1;
    std::string peer_;
    std::uint64_t sent_ = 0, received_ = 0;
    std::chrono::milliseconds timeout_{0};
    bool busy_ = false;
};

class Listener {
public:
    // host: an address to bind ("0.0.0.0", "127.0.0.1", a LAN address); port 0 picks a free one (port()).
    Listener(const std::string& host, std::uint16_t port);
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    // One connection; `timeout` bounds the wait for it (0 = forever) and is the connection's I/O timeout.
    [[nodiscard]] std::unique_ptr<Connection> accept(std::chrono::milliseconds wait, std::chrono::milliseconds io_timeout);
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

private:
    int fd_ = -1;
    std::uint16_t port_ = 0;
};

// "HOST:PORT"; retries until `wait` elapses (the TAIL may still be loading).
[[nodiscard]] std::unique_ptr<Connection> connect(const std::string& host_port, std::chrono::milliseconds wait,
                                                  std::chrono::milliseconds io_timeout);
// Splits "HOST:PORT" (the last colon); throws std::invalid_argument.
[[nodiscard]] std::pair<std::string, std::uint16_t> split_host_port(const std::string& s);

}  // namespace strix::ds41::pipe
