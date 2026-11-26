#pragma once
#include "strix/serve/pair_sampling.hpp"
// GLM TP2 server control channel (host-only, no HIP): rank 0 (lead) -> rank 1 (follower) command blocks.
//
// One fixed-size 512-byte block per round, both ranks exchange at the same time (the follower answers with its own
// block: an acknowledgement plus the echo of the tokens its last command produced). Prompt tokens of a prefill follow
// the prefill block in messages of kTokensPerMessage tokens. Both ranks then make the same engine calls (replay), so
// every engine collective pairs up.
//
// Checks on every round (both ranks): magic, version, the round number (no duplicate / stale round), the peer's
// acknowledgement of the previous round, and the echo: the round, count, last token and FNV-1a of the tokens the
// peer's last command produced must equal this rank's own (both ranks compute every token; a disagreement aborts both).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm/serve/glm_serve_import.hpp"
#include "glm/serve/glm_serve_prefix_cache.hpp"
#include "strix/pair/embed_frame.hpp"
#include "strix/pair/lockstep.hpp"

namespace strix::glm::serve {

inline constexpr std::uint32_t kControlMagic = 0x434D4C47U;  // "GLMC" (little-endian bytes)
inline constexpr std::uint32_t kControlVersion = 15;  // v15: the idle heartbeat can retire posted GPU-direct receives
inline constexpr std::uint32_t kPrefillMtp = 1U;   // prefill pflags: exact chunked MTP bootstrap (prefill_mtp)
inline constexpr std::uint32_t kPrefillLast = 2U;  // prefill pflags: the prompt's last chunk (MTP: its seed is kept)
// prefill pflags (with kPrefillMtp): a prefill_mtp session of layer units; `budget` = units to run (0: cancel it).
// kPrefillStart begins it over the plen tokens that follow at ppos; later calls carry no tokens.
inline constexpr std::uint32_t kPrefillUnits = 32U;
inline constexpr std::uint32_t kPrefillStart = 64U;
inline constexpr std::uint32_t kPrefillSample = 128U;  // final prefill chunk (plain or MTP bootstrap)
// prefill pflags: the chunk's tokens hold virtual ids (strix/session/embed_rows.hpp, image input). On the pair the
// embed frame of its spans (strix/pair/embed_frame.hpp) follows the tokens: one summary message (byte length, spans,
// rows, width, axes; checked before any allocation), then the bytes as 32-bit words. The follower re-interns the rows
// and remaps the ids to its own handles.
inline constexpr std::uint32_t kPrefillEmbeds = 256U;
inline constexpr std::uint32_t kPrefillTurbo = 512U;  // a whole prompt, negotiated in hello
inline constexpr std::uint32_t kEmbedFrameMaxBytes = 1U << 30U;
inline constexpr std::size_t kEmbedFrameMessage = std::size_t{1} << 19U;  // first message index of a frame (prompt messages stay below)
inline constexpr std::uint32_t kStepSample = 1U << 31;  // step pflags carries a sampled-row mask
inline constexpr std::uint32_t kHelloSamplingVote = 1U;
inline constexpr std::uint32_t kHelloTurboPrefillVote = 2U;
inline constexpr std::uint32_t kHeartbeatQuiesce = 1U;  // heartbeat pflags: retire paired receives before going idle
inline constexpr std::uint32_t kPrefillPending = 0xFFFFFFFEU;  // a prefill units call that did not complete the prompt
inline constexpr std::uint32_t kMaxMtpCycles = 64;
inline constexpr std::uint32_t kPrefixReply = 8U;     // prefix_prepare pflags: reply-end snapshot (default: prompt-end)
inline constexpr std::uint32_t kPrefixPressure = 16U; // prefix_evict pflags: eviction under memory pressure
// mtp_run / mtp_enter pflags: the request samples (temperature > 0). kRunSample: the decoder's verify rows are decided
// by the lead's sampler (strix::serve::decide_pair_window); kRunSampleDrafts (only with kRunSample): drafts are drawn
// from the drafter's rows too (strix::serve::pick_pair_draft). Both ranks run the same exchanges.
inline constexpr std::uint32_t kRunSample = 1U;
inline constexpr std::uint32_t kRunSampleDrafts = 2U;
// kRunRejection (only with kRunSampleDrafts): the sampler decides drafted rows by rejection sampling (it reads the
// drafts' distributions; the shared lookup arbiter is off, its drafts have none).
inline constexpr std::uint32_t kRunRejection = 8U;
inline constexpr std::uint32_t kEnterSeed = 4U;   // mtp_enter pflags: open the C1 decoder from the slot's batched-spec seed
inline constexpr std::uint32_t kImportFailed = ~0U;  // import_state's token on a rank whose restore failed
inline constexpr std::uint32_t kImportDsaOnce = 8U;  // import_state pflags: hand-off v2, rank 0 relays the DSA components to rank 1
inline constexpr std::size_t kControlBytes = 512;
inline constexpr unsigned kMaxRows = 8;                       // Engine::step takes 1..8 rows
inline constexpr std::size_t kTokensPerMessage = 16384;       // 64 KiB per prompt message
using strix::pair::kNoRound;

// mtp_enter: seed the slot's drafter from the last step's row pslot (row[0] = {slot, token it produced, its position}).
// mtp_run: up to `cycles` verify cycles of slot pslot (the first run opens the decoder from the seed with `budget`).
// mtp_leave: close the slot's decoder (its deferred KDA rows are applied: plain steps may follow).
// spec_verify: batched self-speculation, ONE verify of every decoding slot's window (row[] sorted by slot: anchor, then
//   drafts at consecutive positions; sbudget[i] = tokens window i may still emit), then per slot greedy acceptance,
//   commit and the drafter catch-up (both ranks, from the merged results). Produces per window: count, tokens.
// spec_draft: one more draft for slot pslot (an MTP extension step of its drafter chain).
enum class Cmd : std::uint32_t { reply = 0, hello = 1, prefill = 2, step = 3, heartbeat = 4, shutdown = 5,
                                 mtp_enter = 6, mtp_run = 7, mtp_leave = 8, spec_verify = 9, spec_draft = 10,
                                 import_state = 11, import_expect = 12, state_digest = 13, import_drop = 14, prefix_prepare = 15, prefix_commit = 16, prefix_restore = 17, prefix_clear = 18, prefix_evict = 19 };
// state_digest (import gate): both ranks digest slot pslot's state up to frontier ppos (Engine::state_bytes_upto, per
// component: fnv64 + size, as u32 words); rank 1 sends its digests to rank 0. No engine state changes.
// import_expect: every rank starts receiving the RTX hand-off for import_id (plen prompt tokens follow); its replies then
// carry that id and its staging status (import_id / import_status: ImportStatus). import_state commits only once both
// ranks report ready; a failure on either rank (import_abort: the slot reset, kImportFailed) is reported in the next
// reply (import_status restored / failed, import_token: the first token) and both fall back to the prefill.
// import_drop: every rank forgets what it staged (or is still receiving) under import_id (a fallback or a cancelled
// request), so its receiver and memory are freed at once.
// import_state: slot pslot restored from the payload each rank staged under import_id (the RTX producer's hand-off,
// verified on that rank), then its last prompt token (plen tokens follow, as for prefill) through prefill / prefill_mtp.

inline const char* cmd_name(Cmd c) {
    switch (c) {
        case Cmd::prefix_prepare: return "prefix_prepare";
        case Cmd::prefix_commit: return "prefix_commit";
        case Cmd::prefix_restore: return "prefix_restore";
        case Cmd::prefix_clear: return "prefix_clear";
        case Cmd::prefix_evict: return "prefix_evict";
        case Cmd::reply: return "reply";
        case Cmd::hello: return "hello";
        case Cmd::import_state: return "import_state";
        case Cmd::import_expect: return "import_expect";
        case Cmd::import_drop: return "import_drop";
        case Cmd::state_digest: return "state_digest";
        case Cmd::prefill: return "prefill";
        case Cmd::step: return "step";
        case Cmd::heartbeat: return "heartbeat";
        case Cmd::shutdown: return "shutdown";
        case Cmd::mtp_enter: return "mtp_enter";
        case Cmd::mtp_run: return "mtp_run";
        case Cmd::mtp_leave: return "mtp_leave";
        case Cmd::spec_verify: return "spec_verify";
        case Cmd::spec_draft: return "spec_draft";
    }
    return "?";
}

// Engine::Row on the wire (same field order; this header does not include the HIP engine header).
struct WireRow {
    std::uint32_t slot = 0, token = 0;
    std::uint64_t pos = 0;
    bool operator==(const WireRow&) const = default;
};

// hello: what both ranks must agree on before the first engine call.
struct Hello {
    std::uint64_t model_hash = 0;     // FNV-1a of the model directory's metadata (config, index, shard names and sizes)
    std::uint64_t identity_hash = 0;  // FNV-1a of the resolved engine configuration (identity, prefill, exchange, env)
    std::uint64_t capacity = 0;       // positions per slot
    std::uint32_t slots = 0;          // engine slots
    std::uint32_t max_rows = 0;       // prefill chunk rows (Options::max_rows)
    std::uint32_t step_rows = 0;      // rows per step (kMaxRows)
    std::uint32_t world = 0;
    std::uint32_t mtp = 0;            // served MTP self-speculation on (both ranks load and run it)
    std::uint32_t mtp_kmax = 0;       // drafts per cycle (adaptive below it)
    std::uint32_t g2 = 0;             // batched self-speculation (multi-slot verify) on
    // RTX import settings every rank must share (0 = import off): bit 0 receiver on, bit 1 'DSA once'; the expected
    // hand-off wire version; FNV-1a of the --import-model-sha attestation (operator-supplied, see the README).
    std::uint32_t import_flags = 0;
    std::uint32_t import_wire = 0;
    std::uint32_t route_fingerprint = 0;  // strix/route_receipt.hpp: 32-bit fold of the numerical-route fingerprint
    std::uint64_t import_model_hash = 0;
    std::uint64_t prefix_cache_bytes = 0;
    bool operator==(const Hello&) const = default;
};

struct Control {
    std::uint32_t magic = kControlMagic;
    std::uint32_t version = kControlVersion;
    std::uint64_t round = 0;
    std::uint64_t ack_round = kNoRound;  // the last round this rank completed (kNoRound: none)
    std::uint32_t cmd = 0;               // Cmd
    std::uint32_t resets = 0;            // bit s: reset slot s before the command (a prefill always sets its slot)
    std::uint32_t pslot = 0, plen = 0;   // prefill: slot, prompt tokens (they follow in kTokensPerMessage messages)
    std::uint32_t nrows = 0, rank = 0;   // step: rows used in row[]; sender's rank
    WireRow row[kMaxRows]{};
    // echo of the sender's last executed command that produced tokens
    std::uint64_t echo_round = kNoRound;
    std::uint32_t echo_count = 0, echo_token = 0, echo_hash = 0, pad0 = 0;
    Hello hello{};
    std::uint32_t pflags = 0;            // prefill: kPrefillMtp
    std::uint32_t budget = 0, cycles = 0, pad1 = 0;  // mtp_run; prefill units: budget = layer units of the call
    std::uint64_t ppos = 0;              // prefill: position of the chunk's first token (0: the slot is reset first)
    std::uint32_t sbudget[kMaxRows]{};   // spec_verify: per window
    std::uint32_t import_id = 0, import_status = 0;  // import_state / import_expect; replies: this rank's staging status
    std::uint32_t import_token = 0, pad2 = 0;          // replies: the first token of this rank's import_state
    // Prefix prepare: key tokens follow (plen), ppos = snapshot frontier, pflags = MTP.
    // Paired votes carry id / ok / charged bytes; commit accepts only after both votes match.
    // Restore names a committed id, identity, frontier and MTP setting; its paired vote precedes suffix work.
    std::uint64_t prefix_id = 0, prefix_identity = 0, prefix_bytes = 0;
    std::uint32_t prefix_ok = 0, prefix_pad = 0;
};
static_assert(sizeof(Control) <= kControlBytes);
static_assert(sizeof(WireRow) == 16);
// The block is a raw memcpy of Control: any change of these offsets is a wire-layout change and needs a new
// kControlVersion (decode checks magic and version before reading any other field, so a peer with another layout is
// refused there instead of misreading commands).
static_assert(offsetof(Control, magic) == 0 && offsetof(Control, version) == 4, "GLMC: magic / version lead the block");
static_assert(offsetof(Control, hello) == 200 && sizeof(Hello) == 80, "GLMC v7 layout: Hello");
static_assert(offsetof(Control, pflags) == 280 && offsetof(Control, budget) == 284 && offsetof(Control, cycles) == 288 &&
                  offsetof(Control, pad1) == 292 && offsetof(Control, ppos) == 296 && offsetof(Control, sbudget) == 304 &&
                  offsetof(Control, import_id) == 336 && offsetof(Control, import_status) == 340 &&
                  offsetof(Control, import_token) == 344 && offsetof(Control, pad2) == 348 &&
                  offsetof(Control, prefix_id) == 352 && offsetof(Control, prefix_identity) == 360 &&
                  offsetof(Control, prefix_bytes) == 368 && offsetof(Control, prefix_ok) == 376 &&
                  offsetof(Control, prefix_pad) == 380 && sizeof(Control) == 384,
              "GLMC v7 layout changed: bump kControlVersion and update these offsets");
static_assert(offsetof(Hello, model_hash) == 0 && offsetof(Hello, identity_hash) == 8 && offsetof(Hello, capacity) == 16 &&
                  offsetof(Hello, slots) == 24 && offsetof(Hello, max_rows) == 28 && offsetof(Hello, step_rows) == 32 &&
                  offsetof(Hello, world) == 36 && offsetof(Hello, mtp) == 40 && offsetof(Hello, mtp_kmax) == 44 &&
                  offsetof(Hello, g2) == 48 && offsetof(Hello, import_flags) == 52 && offsetof(Hello, import_wire) == 56 &&
                  offsetof(Hello, route_fingerprint) == 60 && offsetof(Hello, import_model_hash) == 64 && offsetof(Hello, prefix_cache_bytes) == 72,
              "GLMC v7 Hello layout changed: bump kControlVersion and update these offsets");

inline std::uint32_t fnv1a32(std::span<const std::uint32_t> tokens, std::uint32_t h = 2166136261U) {
    for (const std::uint32_t t : tokens)
        for (unsigned b = 0; b < 4; ++b) h = (h ^ ((t >> (8 * b)) & 255U)) * 16777619U;
    return h;
}
inline std::uint64_t fnv1a64(const void* p, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
inline std::uint64_t fnv1a64(const std::string& s, std::uint64_t h = 14695981039346656037ULL) { return fnv1a64(s.data(), s.size(), h); }

using strix::pair::ProtocolError;

using Block = std::array<std::byte, kControlBytes>;

inline Block encode(const Control& c) {
    Block b{};
    std::memcpy(b.data(), &c, sizeof(Control));
    return b;
}

// Structural validation of a received block (independent of the round state).
inline Control decode(const Block& b) {
    Control c;
    std::memcpy(&c, b.data(), sizeof(Control));
    if (c.magic != kControlMagic) throw ProtocolError("control channel: bad magic from the peer");
    if (c.version != kControlVersion)
        throw ProtocolError("control channel: protocol version " + std::to_string(c.version) + " from the peer, expected " +
                            std::to_string(kControlVersion));
    if (c.cmd > static_cast<std::uint32_t>(Cmd::prefix_evict)) throw ProtocolError("control channel: unknown command");
    if (c.cycles > kMaxMtpCycles) throw ProtocolError("control channel: too many MTP cycles");
    if (c.nrows > kMaxRows) throw ProtocolError("control channel: more than 8 rows");
    if (c.pslot >= 32) throw ProtocolError("control channel: prefill slot out of range");
    for (std::size_t b2 = sizeof(Control); b2 < kControlBytes; ++b2)
        if (b[b2] != std::byte{0}) throw ProtocolError("control channel: non-zero padding");
    return c;
}

// Empty when equal, otherwise the differing fields (local vs peer).
inline std::string hello_mismatch(const Hello& mine, const Hello& peer) {
    std::string s;
    auto f = [&](const char* name, std::uint64_t a, std::uint64_t b) {
        if (a == b) return;
        if (!s.empty()) s += ", ";
        s += std::string(name) + " " + std::to_string(a) + " vs peer " + std::to_string(b);
    };
    f("model_hash", mine.model_hash, peer.model_hash);
    f("identity_hash", mine.identity_hash, peer.identity_hash);
    f("capacity", mine.capacity, peer.capacity);
    f("slots", mine.slots, peer.slots);
    f("max_rows", mine.max_rows, peer.max_rows);
    f("step_rows", mine.step_rows, peer.step_rows);
    f("world", mine.world, peer.world);
    f("mtp", mine.mtp, peer.mtp);
    f("mtp_kmax", mine.mtp_kmax, peer.mtp_kmax);
    f("g2", mine.g2, peer.g2);
    f("route_fingerprint", mine.route_fingerprint, peer.route_fingerprint);
    f("import_flags", mine.import_flags, peer.import_flags);
    f("import_wire", mine.import_wire, peer.import_wire);
    f("import_model_hash", mine.import_model_hash, peer.import_model_hash);
    f("prefix_cache_bytes", mine.prefix_cache_bytes, peer.prefix_cache_bytes);
    return s;
}

// Step rows as the lead sends them: 1..8 rows, strictly ascending slots (one row per slot), each slot < slots, each
// position < capacity. The follower refuses anything else before calling the engine.
inline void check_rows(std::span<const WireRow> rows, std::uint32_t slots, std::uint64_t capacity) {
    if (rows.empty() || rows.size() > kMaxRows) throw ProtocolError("control channel: step needs 1..8 rows");
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].slot >= slots) throw ProtocolError("control channel: row slot out of range");
        if (rows[r].pos >= capacity) throw ProtocolError("control channel: row position beyond the capacity");
        if (r > 0 && rows[r].slot <= rows[r - 1].slot) throw ProtocolError("control channel: rows not in ascending slot order");
    }
}

// Windows of a spec_verify call: rows grouped by slot (sorted, one group per slot), positions consecutive per group.
struct Window {
    std::uint32_t slot = 0, row0 = 0, n = 0;
};
inline std::vector<Window> windows_of(std::span<const WireRow> rows, std::uint32_t slots, std::uint64_t capacity) {
    if (rows.empty() || rows.size() > kMaxRows) throw ProtocolError("control channel: spec_verify needs 1..8 rows");
    std::vector<Window> w;
    for (std::uint32_t r = 0; r < rows.size(); ++r) {
        if (rows[r].slot >= slots || rows[r].pos >= capacity) throw ProtocolError("control channel: spec row out of range");
        if (!w.empty() && rows[r].slot == w.back().slot) {
            if (rows[r].pos != rows[r - 1].pos + 1) throw ProtocolError("control channel: window positions not consecutive");
            ++w.back().n;
            continue;
        }
        if (!w.empty() && rows[r].slot < w.back().slot) throw ProtocolError("control channel: windows not in ascending slot order");
        w.push_back({rows[r].slot, r, 1});
    }
    return w;
}

// A slot's drafter state in batched self-speculation (both ranks hold the same; rank 0's lead reads it to size the
// windows): the drafts proposed for its next window and their merged confidences; `p` = the anchor position they follow.
struct SlotSpec {
    bool seeded = false, disabled = false;
    std::uint64_t p = 0;
    std::vector<std::uint32_t> drafts;
    std::vector<float> conf;
};

// KDA batches one call only when rows[r].slot == rows[0].slot + r (engine glm_fast_attn); otherwise per-row launches.
inline bool consecutive_slots(std::span<const WireRow> rows) {
    for (std::size_t r = 1; r < rows.size(); ++r)
        if (rows[r].slot != rows[0].slot + r) return false;
    return true;
}

// The two ranks' channel. `wire(out, in, layer, request)` moves one message of out.size() bytes each way (both ranks
// pass the same size and tag); world 1 passes an empty wire (the peer is this rank's own mirror).
class Channel {
public:
    using Wire = std::function<void(std::span<const std::byte>, std::span<std::byte>, std::uint32_t layer, std::uint64_t request)>;
    Channel(Wire wire, std::uint32_t rank) : wire_(std::move(wire)), lock_(rank) {}

    // One round: fills the bookkeeping fields of `mine` and returns the peer's block after every check.
    Control exchange(Control mine) {
        mine.magic = kControlMagic;
        mine.version = kControlVersion;
        const strix::pair::LockstepHeader head = lock_.header();
        mine.round = head.round;
        mine.ack_round = head.ack_round;
        mine.rank = head.rank;
        mine.echo_round = echo_round_;
        mine.echo_count = echo_count_;
        mine.echo_token = echo_token_;
        mine.echo_hash = echo_hash_;
        Control peer;
        if (!wire_) {
            peer = mine;
            peer.rank = head.rank ^ 1U;
        } else {
            const Block out = encode(mine);
            Block in{};
            wire_(out, in, 0, head.round);
            peer = decode(in);
        }
        lock_.check({peer.round, peer.ack_round, peer.rank});
        if (peer.echo_round != echo_round_ || peer.echo_count != echo_count_ || peer.echo_token != echo_token_ ||
            peer.echo_hash != echo_hash_)
            throw ProtocolError("cross-rank token mismatch: round " + std::to_string(echo_round_) + " local count " +
                                std::to_string(echo_count_) + " last " + std::to_string(echo_token_) + " hash " +
                                std::to_string(echo_hash_) + ", peer round " + std::to_string(peer.echo_round) + " count " +
                                std::to_string(peer.echo_count) + " last " + std::to_string(peer.echo_token) + " hash " +
                                std::to_string(peer.echo_hash));
        lock_.accept({peer.round, peer.ack_round, peer.rank});
        return peer;
    }

    // Prompt tokens after a prefill block: the lead sends `tokens`, the follower receives `n` of them.
    // k0: index of the first message inside the round (a second transfer of one round takes its own range).
    void tokens(std::vector<std::uint32_t>& tokens, std::size_t n, bool sender, std::size_t k0 = 0) {
        if (sender && tokens.size() != n) throw std::logic_error("control channel: prompt size");
        if (!wire_) return;
        if (!sender) tokens.assign(n, 0);
        std::vector<std::uint32_t> out, in;
        for (std::size_t c0 = 0, k = k0; c0 < n; c0 += kTokensPerMessage, ++k) {
            const std::size_t m = std::min(kTokensPerMessage, n - c0);
            out.assign(m, 0);
            in.assign(m, 0);
            if (sender) std::memcpy(out.data(), tokens.data() + c0, m * 4);
            wire_(std::as_bytes(std::span<const std::uint32_t>(out)), std::as_writable_bytes(std::span<std::uint32_t>(in)),
                  1, (lock_.round() << 20) | k);
            if (!sender) std::memcpy(tokens.data() + c0, in.data(), m * 4);
        }
    }

    // Tokens produced by the command of round `round` (the round already exchanged: round() - 1).
    void record(std::uint64_t round, std::span<const std::uint32_t> produced) {
        echo_round_ = round;
        echo_count_ = static_cast<std::uint32_t>(produced.size());
        echo_token_ = produced.empty() ? 0U : produced.back();
        echo_hash_ = fnv1a32(produced);
    }

    // hello on both ranks (each sends Cmd::hello with its own fields): throws on any mismatch.
    bool hello(const Hello& mine, bool sampling = false, bool* turbo_prefill = nullptr) {
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::hello);
        c.hello = mine;
        c.pflags = (sampling ? kHelloSamplingVote : 0U) |
            (turbo_prefill && *turbo_prefill ? kHelloTurboPrefillVote : 0U);
        const Control peer = exchange(c);
        if (peer.cmd != static_cast<std::uint32_t>(Cmd::hello)) throw ProtocolError("control channel: peer did not open with hello");
        const std::string m = hello_mismatch(mine, peer.hello);
        if (!m.empty()) throw ProtocolError("hello mismatch between the ranks: " + m);
        if (turbo_prefill) *turbo_prefill = (c.pflags & peer.pflags & kHelloTurboPrefillVote) != 0;
        return (c.pflags & peer.pflags & kHelloSamplingVote) != 0;
    }

    void exchange_sampling_logits(std::span<const float> out, std::span<float> in, std::uint64_t chunk) {
        wire_(std::as_bytes(out), std::as_writable_bytes(in), 2, (lock_.round() << 20) | chunk);
    }
    // Decision words of a sampled window or draft: one message index space with exchange_sampling_logits per round.
    void exchange_sampling_words(std::span<const std::uint32_t> out, std::span<std::uint32_t> in, std::uint64_t chunk) {
        wire_(std::as_bytes(out), std::as_writable_bytes(in), 2, (lock_.round() << 20) | chunk);
    }
    [[nodiscard]] bool paired() const noexcept { return static_cast<bool>(wire_); }
    [[nodiscard]] std::uint64_t round() const noexcept { return lock_.round(); }
    [[nodiscard]] std::uint32_t rank() const noexcept { return lock_.rank(); }

private:
    Wire wire_;
    strix::pair::Lockstep lock_;
    std::uint64_t echo_round_ = kNoRound;
    std::uint32_t echo_count_ = 0, echo_token_ = 0, echo_hash_ = 2166136261U;
};

// A sampled MTP command as the engine adapter sees it: the pair's channel (its sampling exchanges), the lead's sampler
// (null on the follower) and the kRunSample / kRunSampleDrafts flags.
struct SampledRun {
    Channel* ch = nullptr;
    strix::TokenSampler* sampler = nullptr;
    std::uint32_t flags = 0;
    // The request's effective stop ids (the lead's; empty on the follower): a verify window ends at the first one
    // decided and the decoder ends with it (the follower learns it from the window's kPairWindowStop decision).
    std::span<const std::uint32_t> stops{};
};

// The sampled-MTP flags of an mtp_run / mtp_enter command (kEnterSeed excluded), checked; with flags the lead must hold
// the request's sampler.
inline SampledRun sampled_run(const Control& c, Channel& ch, bool sender, std::span<const strix::serve::PairSamplingRow> choices,
                              std::uint32_t other_flags, const char* what) {
    SampledRun run{&ch, nullptr, c.pflags & ~other_flags};
    const bool drafts = (run.flags & kRunSampleDrafts) != 0, rejection = (run.flags & kRunRejection) != 0;
    if ((run.flags & ~(kRunSample | kRunSampleDrafts | kRunRejection)) != 0 ||
        ((drafts || rejection) && (run.flags & kRunSample) == 0) || (rejection && !drafts))
        throw ProtocolError(std::string("control channel: ") + what + " flags");
    if (run.flags != 0 && sender) {
        if (choices.empty() || choices[0].sampler == nullptr) throw std::logic_error("control channel: sampled MTP without a sampler");
        run.sampler = choices[0].sampler;
        run.stops = choices[0].stops;
    }
    return run;
}

// The embed frame of a prefill chunk with virtual ids (kPrefillEmbeds), after its tokens. Eng: embed_frame(ids) ->
// frame bytes (the lead: the rows of the spans its peer does not hold yet, for the table it was bound to);
// embed_frame_received(frame, ids) (the follower: re-interns the rows, binds its table and rewrites ids to its own
// handles); optional embed_row_width() (the language model width every row must have). World 1 sends nothing: the
// lead resolves the ids against its own table.
//
// Wire: one message of kEmbedFrameSummaryWords (frame bytes, spans, payload rows, widest row, most axes), then the
// bytes as 32-bit words. The follower checks the summary before it allocates anything: the spans are exactly the
// chunk's distinct virtual ids, the rows fit embed_frame_rows_bound (per span: the span limit and the context room
// after the chunk, not the whole slot capacity), the width is the model's, and the byte length is at most what those
// spans and rows can take (and kEmbedFrameMaxBytes). The received bytes are then reconciled with the summary
// (embed_frame_matches_summary: every entry's rows, width and axes, the exact byte length) before the parser sees them,
// which then runs on the received words (the interner holds its own copy of each new span).
// A refusal throws ProtocolError on the follower before the next collective; the lead, still in this exchange or the
// next one, sees the follower leave (wire error or timeout) and fails the pair; neither rank continues alone.
inline constexpr std::size_t kEmbedFrameSummaryWords = 5U;
inline constexpr std::uint32_t kEmbedFrameMaxAxes = 4U;
inline constexpr std::uint32_t kEmbedFrameMaxWidth = 1U << 16U;  // without embed_row_width()

// The summary of an encoded frame (the sender's own bytes): spans, rows of the payloads that follow, widest row, axes.
inline std::array<std::uint32_t, kEmbedFrameSummaryWords> embed_frame_summary(std::span<const std::byte> frame) {
    namespace fd = strix::pair::embed_frame_detail;
    std::array<std::uint32_t, kEmbedFrameSummaryWords> s{};
    s[0] = static_cast<std::uint32_t>(frame.size());
    s[1] = fd::get32(frame, 8U);
    std::uint64_t rows = 0;
    for (std::size_t e = 0; e < s[1]; ++e) {
        const std::size_t at = strix::pair::kEmbedFrameHeaderBytes + e * strix::pair::kEmbedFrameEntryBytes;
        if ((fd::get32(frame, at + 20U) & 1U) == 0U) continue;  // no payload: the peer holds the rows
        rows += fd::get32(frame, at + 4U);
        s[3] = std::max(s[3], fd::get32(frame, at + 8U));
        s[4] = std::max(s[4], fd::get32(frame, at + 16U));
    }
    if (rows > 0xFFFFFFFFU) throw std::logic_error("control channel: embed frame rows");
    s[2] = static_cast<std::uint32_t>(rows);
    return s;
}

// Most payload rows the frame of this chunk can carry, from data both ranks hold (the chunk's ids, the context room
// after it). A span has at most kEmbedMaxRows rows; its rows up to the highest one in this chunk are known, and the
// rest are placeholder ids of the same prompt that must fit the context after the chunk. A chunk of one handle is thus
// bounded by that span, not by the slot capacity. A frame only carries rows the peer lacks, so this is an upper bound.
inline std::uint64_t embed_frame_rows_bound(std::span<const std::uint32_t> ids, std::uint64_t room_after) {
    std::map<std::uint32_t, std::uint32_t> top;  // handle -> highest row index seen
    for (const auto id : ids)
        if (strix::session::is_embed_id(id)) {
            auto& t = top[strix::session::embed_handle(id)];
            t = std::max(t, strix::session::embed_row(id));
        }
    std::uint64_t bound = 0;
    for (const auto& kv : top)
        bound += std::min<std::uint64_t>(strix::session::kEmbedMaxRows, std::uint64_t{kv.second} + 1U + room_after);
    return bound;
}

// The received bytes against the summary both ranks agreed on: length, span count, payload rows, widest row, most axes
// and the exact byte length that the entries (rows, width, axes) imply. The caller has checked that the length holds
// the header and the entries. False on any difference; the parser then never sees the frame.
inline bool embed_frame_matches_summary(std::span<const std::byte> frame, const std::array<std::uint32_t, kEmbedFrameSummaryWords>& s,
                                        std::uint32_t max_width) {
    namespace fd = strix::pair::embed_frame_detail;
    const std::uint64_t head = strix::pair::kEmbedFrameHeaderBytes + std::uint64_t{s[1]} * strix::pair::kEmbedFrameEntryBytes;
    if (frame.size() != s[0] || frame.size() < head || fd::get32(frame, 8U) != s[1]) return false;
    std::uint64_t rows = 0, bytes = head;
    std::uint32_t width = 0, axes = 0;
    for (std::uint32_t e = 0; e < s[1]; ++e) {
        const std::size_t at = strix::pair::kEmbedFrameHeaderBytes + std::size_t{e} * strix::pair::kEmbedFrameEntryBytes;
        const std::uint32_t r = fd::get32(frame, at + 4U), w = fd::get32(frame, at + 8U), a = fd::get32(frame, at + 16U),
                            flags = fd::get32(frame, at + 20U);
        if (r == 0U || r > strix::session::kEmbedMaxRows || w == 0U || w > max_width || a > kEmbedFrameMaxAxes || (flags & ~1U) != 0U)
            return false;
        if ((flags & 1U) == 0U) continue;
        rows += r;
        bytes += fd::payload_bytes(r, w, a);
        width = std::max(width, w);
        axes = std::max(axes, a);
    }
    return rows == s[2] && width == s[3] && axes == s[4] && bytes == frame.size();
}

// room_after: context positions after this chunk (capacity - pos - length).
template<class Eng>
void embed_frame_round(Eng& eng, Channel& ch, std::vector<std::uint32_t>& prompt, bool sender, std::uint64_t capacity,
                       std::uint64_t room_after) {
    if constexpr (requires(std::span<const std::byte> f) { eng.embed_frame(std::span<const std::uint32_t>{}); eng.embed_frame_received(f, prompt); }) {
        if (!ch.paired()) return;
        std::vector<std::byte> frame;
        std::vector<std::uint32_t> summary(kEmbedFrameSummaryWords, 0), words;
        if (sender) {
            frame = eng.embed_frame(std::span<const std::uint32_t>(prompt));
            if (frame.empty() || frame.size() > kEmbedFrameMaxBytes) throw std::logic_error("control channel: embed frame size");
            const auto s = embed_frame_summary(frame);
            std::copy(s.begin(), s.end(), summary.begin());
        }
        ch.tokens(summary, kEmbedFrameSummaryWords, sender, kEmbedFrameMessage);
        const std::uint32_t length = summary[0], spans = summary[1], rows = summary[2], width = summary[3], axes = summary[4];
        std::uint32_t max_width = kEmbedFrameMaxWidth;
        if constexpr (requires { eng.embed_row_width(); }) max_width = static_cast<std::uint32_t>(eng.embed_row_width());
        const std::size_t expected_spans = strix::pair::embed_frame_handles(std::span<const std::uint32_t>(prompt)).size();
        if (length == 0 || length > kEmbedFrameMaxBytes || spans == 0 || spans != expected_spans || rows > capacity ||
            rows > embed_frame_rows_bound(std::span<const std::uint32_t>(prompt), room_after) ||
            width > max_width || axes > kEmbedFrameMaxAxes || (rows != 0 && width == 0))
            throw ProtocolError("control channel: embed frame summary out of range");
        const std::uint64_t most = std::uint64_t{strix::pair::kEmbedFrameHeaderBytes} +
                                   std::uint64_t{spans} * (strix::pair::kEmbedFrameEntryBytes + 3U) +
                                   strix::pair::embed_frame_detail::payload_bytes(rows, width, axes);
        if (length < strix::pair::kEmbedFrameHeaderBytes + std::uint64_t{spans} * strix::pair::kEmbedFrameEntryBytes ||
            length > most)
            throw ProtocolError("control channel: embed frame length does not match its summary");
        const std::size_t n = (std::size_t{length} + 3U) / 4U;
        words.assign(n, 0);
        if (sender) std::memcpy(words.data(), frame.data(), frame.size());
        ch.tokens(words, n, sender, kEmbedFrameMessage + 1U);
        if (!sender) {
            const auto bytes = std::as_bytes(std::span<const std::uint32_t>(words)).first(length);
            std::array<std::uint32_t, kEmbedFrameSummaryWords> agreed{};
            std::copy(summary.begin(), summary.end(), agreed.begin());
            bool same = false;
            try { same = embed_frame_matches_summary(bytes, agreed, max_width); } catch (const std::exception&) {}
            if (!same) throw ProtocolError("control channel: embed frame does not match its summary");
            try {
                eng.embed_frame_received(bytes, prompt);
            } catch (const std::exception& e) {  // a frame the follower cannot apply leaves the ranks apart: fatal for the pair
                throw ProtocolError(std::string("control channel: embed frame: ") + e.what());
            }
        }
    } else {
        (void)capacity;
        (void)room_after;
        throw ProtocolError("control channel: this engine takes no embedding rows");
    }
}

// The engine calls of one command, made identically by both ranks (the lead after sending it, the follower after
// receiving it). Eng: reset(slot); prefill(slot, tokens, pos0) -> next; step(span<const WireRow>) -> tokens;
// prefill_mtp(slot, tokens, pos0, last) -> next (last: seed kept); mtp_enter(row index, WireRow); mtp_run(slot, budget, cycles) -> tokens;
// mtp_leave(slot); spec_verify(rows, budgets) -> per window [count, tokens..]; spec_draft(slot) -> draft. `prompt`: the lead's prompt (sent), or empty on the follower (received). Returns the produced tokens.
template<class Eng>
std::vector<std::uint32_t> replay(Eng& eng, Channel& ch, const Control& c, std::vector<std::uint32_t>& prompt, bool sender,
                                  std::uint32_t slots, std::uint64_t capacity, PrefixCache* cache = nullptr,
                                  std::span<const strix::serve::PairSamplingRow> choices = {}) {
    const auto cmd = static_cast<Cmd>(c.cmd);
    const std::uint32_t sample_mask = cmd == Cmd::step && (c.pflags & kStepSample) != 0U ? c.pflags & ~kStepSample :
        (cmd == Cmd::prefill && (c.pflags & kPrefillSample) != 0U ? 1U : 0U);
    if (slots < 32 && (c.resets >> slots) != 0) throw ProtocolError("control channel: reset of a slot out of range");
    for (std::uint32_t s = 0; s < slots && s < 32; ++s)
        if ((c.resets >> s) & 1U) eng.reset(s);
    std::vector<std::uint32_t> out;
    switch (cmd) {
        case Cmd::prefix_evict:
            if (!cache || !cache->erase(c.prefix_id, (c.pflags & kPrefixPressure) != 0)) throw ProtocolError("prefix cache: eviction key missing");
            return out;
        case Cmd::prefix_clear:
            if (!cache) throw ProtocolError("prefix cache unavailable");
            cache->clear();
            return out;
        case Cmd::prefix_commit:
            if (!cache) throw ProtocolError("prefix cache unavailable");
            if (c.prefix_ok && (cache->pending_bytes() == 0 || !cache->pending_fits()))
                throw ProtocolError("prefix cache: commit without paired preparation / lead eviction");
            cache->commit(c.prefix_ok != 0);
            return out;
        case Cmd::prefix_prepare: {
            if (!cache || c.pslot >= slots || c.ppos == 0 || c.ppos >= capacity || c.prefix_id == 0 || c.plen != c.ppos)
                throw ProtocolError("prefix cache: invalid snapshot");
            ch.tokens(prompt, c.plen, sender);
            bool ok = false;
            std::string why;  // a refused store names its reason (one stderr line per refusal, counted as refused)
            bool keep_counted = false;
            try {
                if constexpr (requires { eng.state_bytes_upto(c.pslot, c.ppos); eng.prefix_state_bound(c.ppos); }) {
                    cache->commit(false);
                    const std::uint64_t bound = eng.prefix_state_bound(c.ppos);
                    bool available = false;  // prefix_store_fits: MemAvailable >= bound + this rank's RAM margin
                    if constexpr (requires { eng.prefix_store_allowed(bound); }) available = eng.prefix_store_allowed(bound);
                    else available = prefix_store_fits(prefix_memory_bytes(), bound, kPrefixMemoryMargin);
                    if (!available) {
                        const auto avail = prefix_memory_bytes();
                        why = "host memory (MemAvailable " + (avail ? std::to_string(*avail >> 20) + " MiB" : std::string("unknown")) +
                              " below the snapshot bound " + std::to_string(bound >> 20) + " MiB + the RAM margin)";
                    } else if (!cache->can_stage(bound, prompt.size())) {
                        why = "budget (bound " + std::to_string(bound) + " B, held " + std::to_string(cache->bytes()) + " of " +
                              std::to_string(cache->budget()) + " B)";
                    } else {
                        PrefixCache::Entry e;
                        e.id = c.prefix_id;
                        e.identity = c.prefix_identity;
                        e.mtp = (c.pflags & kPrefillMtp) != 0;
                        e.kind = (c.pflags & kPrefixReply) ? PrefixCache::Kind::reply_end : PrefixCache::Kind::prompt_end;
                        e.frontier = c.ppos;
                        e.ids = prompt;
                        e.state = eng.state_bytes_upto(c.pslot, c.ppos);
                        const std::size_t got = e.state.size();
                        ok = cache->prepare(std::move(e));
                        if (!ok) {  // the keep counts this refusal itself
                            why = "keep refused the entry (" + std::to_string(got) + " B, bound " + std::to_string(bound) + " B)";
                            keep_counted = true;
                        }
                    }
                } else {
                    why = "engine has no snapshot support";
                }
            } catch (const std::exception& x) {
                ok = false;
                why = std::string("snapshot failed: ") + x.what();
            } catch (...) {
                ok = false;
                why = "snapshot failed";
            }
            if (!ok) cache->commit(false);
            // Both ranks report before the lead's commit decision. No token echo changes.
            Control vote;
            vote.cmd = static_cast<std::uint32_t>(Cmd::reply);
            vote.prefix_id = c.prefix_id;
            vote.prefix_ok = ok ? 1U : 0U;
            vote.prefix_bytes = cache->pending_bytes();
            const auto peer = ch.exchange(vote);
            if (ok && (!peer.prefix_ok || peer.prefix_id != c.prefix_id || peer.prefix_bytes != vote.prefix_bytes)) {
                why = !peer.prefix_ok ? std::string("peer refused")
                                      : "rank layouts disagree (local " + std::to_string(vote.prefix_bytes) + " B, peer " +
                                            std::to_string(peer.prefix_bytes) + " B)";
                cache->commit(false);
            }
            if (!why.empty()) {
                if (!keep_counted) cache->note_refused();
                std::fprintf(stderr, "rank %u prefix store refused (id %llu, rows %llu): %s\n", ch.rank(),
                             static_cast<unsigned long long>(c.prefix_id), static_cast<unsigned long long>(c.ppos), why.c_str());
            }
            out.push_back(cache->pending_bytes() != 0 ? 1U : 0U);
            return out;
        }
        case Cmd::prefix_restore: {
            bool ok = false;
            try {
                if (cache && c.pslot < slots) {
                    const auto* e = cache->get(c.prefix_id);
                    const auto* state = cache->state(c.prefix_id);
                    if (e && state && e->frontier == c.ppos &&
                        e->identity == PrefixCache::key_of((c.pflags & kPrefillMtp) != 0, c.prefix_identity)) {
                        if constexpr (requires { eng.restore_state(c.pslot, *state, c.ppos); }) {
                            eng.reset(c.pslot);
                            eng.restore_state(c.pslot, *state, c.ppos);
                            cache->touch(c.prefix_id);
                            ok = true;
                        }
                    }
                }
            } catch (...) { ok = false; }
            Control vote;
            vote.cmd = static_cast<std::uint32_t>(Cmd::reply);
            vote.prefix_id = c.prefix_id;
            vote.prefix_ok = ok ? 1U : 0U;
            const auto peer = ch.exchange(vote);
            if (!ok || !peer.prefix_ok || peer.prefix_id != c.prefix_id)
                throw ProtocolError("prefix cache: paired restore failed or key missing");
            return out;
        }
        case Cmd::prefill: {
            // The MTP bootstrap (whole or in layer units) restores the prompt's last-row logits after seeding its
            // drafter, so the call that completes the prompt can be sampled on either route.
            const bool turbo = (c.pflags & kPrefillTurbo) != 0U;
            if (turbo && (c.ppos != 0 || c.pflags != (kPrefillMtp | kPrefillLast | kPrefillTurbo)))
                throw ProtocolError("control channel: TURBO requires a complete native MTP prefill");
            if (sample_mask && (c.pflags & kPrefillLast) == 0U)
                throw ProtocolError("control channel: sampling requires the final prefill chunk");
            if ((c.pflags & kPrefillUnits) != 0U && (c.pflags & kPrefillStart) == 0U) {  // resume / cancel: no tokens
                if (c.pslot >= slots || (c.pflags & kPrefillMtp) == 0U || c.plen != 0)
                    throw ProtocolError("control channel: prefill units resume out of range");
                if constexpr (requires { eng.prefill_units(c.pslot, std::span<const std::uint32_t>{}, c.ppos, false, c.budget); })
                    out.push_back(eng.prefill_units(c.pslot, std::span<const std::uint32_t>{}, c.ppos, false, c.budget));
                else
                    throw ProtocolError("control channel: this engine has no prefill units");
                break;
            }
            if (c.pslot >= slots || (c.ppos == 0 && ((c.resets >> c.pslot) & 1U) == 0))
                throw ProtocolError("control channel: prefill without its slot reset");
            if (c.plen == 0 || c.ppos >= capacity || c.plen >= capacity - c.ppos)
                throw ProtocolError("control channel: prompt chunk out of range");
            ch.tokens(prompt, c.plen, sender);
            if ((c.pflags & kPrefillEmbeds) != 0U) embed_frame_round(eng, ch, prompt, sender, capacity, capacity - c.ppos - c.plen);
            const std::span<const std::uint32_t> p(prompt);
            if ((c.pflags & kPrefillUnits) != 0U) {
                if ((c.pflags & kPrefillMtp) == 0U || c.budget == 0)
                    throw ProtocolError("control channel: prefill units start needs MTP and a unit budget");
                if constexpr (requires { eng.prefill_units(c.pslot, p, c.ppos, true, c.budget); })
                    out.push_back(eng.prefill_units(c.pslot, p, c.ppos, true, c.budget));
                else
                    throw ProtocolError("control channel: this engine has no prefill units");
                break;
            }
            if (turbo) {
                if constexpr (requires { eng.prefill_turbo(c.pslot, p); })
                    out.push_back(eng.prefill_turbo(c.pslot, p));
                else
                    throw ProtocolError("control channel: this engine has no TURBO prefill");
            } else {
                out.push_back((c.pflags & kPrefillMtp) != 0 ? eng.prefill_mtp(c.pslot, p, c.ppos, (c.pflags & kPrefillLast) != 0)
                                                            : eng.prefill(c.pslot, p, c.ppos));
            }
            break;
        }
        case Cmd::import_state: {
            if (c.pslot >= slots || ((c.resets >> c.pslot) & 1U) == 0) throw ProtocolError("control channel: import without its slot reset");
            if (c.plen < 2 || c.plen >= capacity) throw ProtocolError("control channel: import prompt length out of range");
            ch.tokens(prompt, c.plen, sender);
            // A failed restore / last-token call is local to this rank: the slot is reset (import_abort) and the lead
            // learns both ranks' outcome from the next reply. No echo: the ranks may differ here by design.
            std::uint32_t y = kImportFailed;
            try {
                y = eng.import_state(c.pslot, c.import_id, std::span<const std::uint32_t>(prompt), (c.pflags & kPrefillMtp) != 0,
                                     (c.pflags & kImportDsaOnce) != 0);
            } catch (const ImportFatal&) {
                throw;  // inside the engine collectives after both ranks restored: the engine's fatal contract
            } catch (const std::exception& e) {
                eng.import_abort(c.pslot, c.import_id, e.what());
                y = kImportFailed;
            }
            out.push_back(y);
            return out;
        }
        case Cmd::import_drop: {
            if constexpr (requires { eng.import_drop(c.import_id); }) eng.import_drop(c.import_id);
            return out;
        }
        case Cmd::import_expect: {
            if (c.plen < 2 || c.plen >= capacity) throw ProtocolError("control channel: import prompt length out of range");
            ch.tokens(prompt, c.plen, sender);
            eng.import_expect(c.import_id, std::span<const std::uint32_t>(prompt));
            return out;
        }
        case Cmd::step: {
            const std::span<const WireRow> rows(c.row, c.nrows);
            check_rows(rows, slots, capacity);
            if (sample_mask >> rows.size()) throw ProtocolError("control channel: sampling row mask");
            out = eng.step(rows);
            if (out.size() != rows.size()) throw std::runtime_error("engine step returned a wrong row count");
            break;
        }
        case Cmd::mtp_enter: {
            if (c.pslot >= kMaxRows || c.row[0].slot >= slots || c.row[0].pos + 1 >= capacity)
                throw ProtocolError("control channel: mtp_enter out of range");
            const SampledRun run = sampled_run(c, ch, sender, choices, kEnterSeed, "mtp_enter");
            if (run.flags == 0) {
                eng.mtp_enter(c.pslot, c.row[0], (c.pflags & kEnterSeed) != 0);
            } else if constexpr (requires { eng.mtp_enter(c.pslot, c.row[0], true, run); }) {
                eng.mtp_enter(c.pslot, c.row[0], (c.pflags & kEnterSeed) != 0, run);
            } else {
                throw ProtocolError("control channel: sampled MTP unavailable");
            }
            return out;
        }
        case Cmd::mtp_run: {
            if (c.pslot >= slots || c.cycles == 0 || c.budget == 0) throw ProtocolError("control channel: mtp_run out of range");
            const SampledRun run = sampled_run(c, ch, sender, choices, 0U, "mtp_run");
            if (run.flags == 0) {
                out = eng.mtp_run(c.pslot, c.budget, c.cycles);
            } else if constexpr (requires { eng.mtp_run(c.pslot, c.budget, c.cycles, run); }) {
                out = eng.mtp_run(c.pslot, c.budget, c.cycles, run);
            } else {
                throw ProtocolError("control channel: sampled MTP unavailable");
            }
            break;
        }
        case Cmd::spec_verify: {
            const std::span<const WireRow> rows(c.row, c.nrows);
            const auto w = windows_of(rows, slots, capacity);
            for (std::size_t i = 0; i < w.size(); ++i)
                if (c.sbudget[i] == 0) throw ProtocolError("control channel: spec window without budget");
            out = eng.spec_verify(rows, std::span<const std::uint32_t>(c.sbudget, w.size()));
            break;
        }
        case Cmd::spec_draft: {
            if (c.pslot >= slots) throw ProtocolError("control channel: spec_draft out of range");
            out.push_back(eng.spec_draft(c.pslot));
            break;
        }
        case Cmd::mtp_leave: {
            if (c.pslot >= slots) throw ProtocolError("control channel: mtp_leave out of range");
            eng.mtp_leave(c.pslot);
            return out;
        }
        case Cmd::state_digest: {
            if (c.pslot >= slots || c.ppos == 0 || c.ppos >= capacity) throw ProtocolError("control channel: state_digest out of range");
            if constexpr (requires { eng.state_digest(c.pslot, c.ppos); }) {
                std::vector<std::uint32_t> mine = eng.state_digest(c.pslot, c.ppos), peer;
                // the same layout on both ranks (hello): rank 1 sends its words, rank 0 receives them
                if (sender) ch.tokens(peer, mine.size(), false);
                else ch.tokens(mine, mine.size(), true);
                out = std::move(mine);  // rank 0: its digests, then rank 1's
                out.insert(out.end(), peer.begin(), peer.end());
                return out;
            } else {
                throw ProtocolError("control channel: state_digest not supported by this engine");
            }
        }
        case Cmd::heartbeat:
            if ((c.pflags & kHeartbeatQuiesce) != 0U) {
                if constexpr (requires { eng.quiesce_idle(); }) eng.quiesce_idle();
                else throw ProtocolError("control channel: idle transport quiesce unavailable");
            }
            return out;
        case Cmd::shutdown: return out;
        case Cmd::hello:
        case Cmd::reply: throw ProtocolError(std::string("control channel: unexpected ") + cmd_name(cmd));
    }
    // Both ranks still echo the paired head's result. Only the lead's returned tokens are replaced.
    ch.record(ch.round() - 1, out);
    // A units call the engine did not complete keeps kPrefillPending on both ranks: the lead's unit arithmetic check
    // then fails loudly instead of emitting a token sampled from rows that are not the prompt's last.
    const bool pending = cmd == Cmd::prefill && !out.empty() && out[0] == kPrefillPending;
    if (sample_mask && !pending) {
        if constexpr (requires { eng.sample_outputs(ch, sample_mask, choices, std::span(out)); })
            eng.sample_outputs(ch, sample_mask, choices, std::span(out));
        else
            throw ProtocolError("control channel: sampling unavailable");
    }
    return out;
}

}  // namespace strix::glm::serve
