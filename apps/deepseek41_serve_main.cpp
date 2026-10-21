// strix-deepseek41-serve: DeepSeek V4.1 Flash served by a two-node engine group (EP experts, TP attention).
// Rank 0 runs the OpenAI/Anthropic-compatible HTTP front end (strix_serve) and the continuous-batching scheduler; rank
// 1 has no API and is driven by rank 0 over the communicator: before every step rank 0 sends a control block (the
// rows of the step and the slots to reset), both ranks run BatchForward::step on the same rows, and both hold the
// same greedy tokens. Requests are admitted between steps while slots are free (continuous admission); their prompts
// are read in chunks of rows next to the decoding requests. Greedy decoding only: every request's tokens are those
// of running it alone.
//
//   strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto|tcp-listen:P|tcp-connect:H:P|emulate
//       [--group-manifest FILE] [--layers 40] [--context 4096] [--slots 4] [--prompt-rows 8] [--tp-attention]
//       [--pool-positions N] (a shared KV pool of N positions over all slots: a request holds prompt + max_tokens of it)
//       [--engram-direct] [--http-bind 127.0.0.1] [--http-port 0] [--http-api-key-file F] [--http-max-queue 64]
//       [--http-max-connections 64] [--model-id deepseek-v4.1-flash] [--default-max-tokens 1024]
//       [--connect-timeout 60] [--peer-timeout 60]
//       [--prefill-path rows|device] [--prefill-rows 4096] [--prefill-min 0]
//       [--prefix-cache-mb 0] [--prefix-block 256] [--prefix-reserve-mb 4096]
//   strix-deepseek41-serve MODEL.gguf --describe-json [--context N] [--slots N] [--layers 40]
//
// Events (one JSON line each on stdout): {"event":"READY",...} when a rank serves (rank 0 after its HTTP listener is
// up), {"event":"POISONED","reason":...} before a non-zero exit (peer lost, engine failure, group mismatch),
// {"event":"STOPPED"} after a clean stop (SIGINT/SIGTERM: stop admitting, finish the running requests, stop rank 1).
// The group manifest (TensorMeld, strix/group-manifest-v1) gives group_id, epoch, rank, transport, http, context and
// slots; both ranks check group_id, epoch, the engine executable's SHA-256 and the model file size in a hello.
//
// --prefill-path device: an admitted request's prompt (of at least --prefill-min tokens) is run by the two-rank device
// prefill (DevicePrefill, both ranks; rank 0 sends a prefill command and the prompt over the control channel) instead
// of rows of the batch steps; the prefill reads the BatchForward's weights in place (one copy per node), runs pieces
// of at most --prefill-rows rows, and its attention state is imported into the request's slot, which then decodes as
// usual. Same tokens as the rows path (the prefill's rows equal the decode step's). The other requests wait during the
// prefill (no decode steps run).
// --prefix-cache-mb M (device path): a prefix cache of attention states on the host, M MiB (deepseek41_prefix_cache.hpp):
// a prompt whose beginning was prefilled before resumes from the longest cached block boundary; each rank keeps its own
// copy (the state is replicated), rank 0 decides hits and inserts (an insert is skipped while the node's MemAvailable is
// below --prefix-reserve-mb) and rank 1 checks it holds the same entry. The cache lives on the host, apart from the KV
// pool: it never takes room from live requests.
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;
using strix::deepseek41::BatchSequence;
namespace serve = strix::serve;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string model, transport = "usb4auto", manifest, http_bind = "127.0.0.1", api_key_file, model_id = "deepseek-v4.1-flash";
    std::string group_id = "local";
    std::uint64_t epoch = 0;
    std::uint32_t rank = 0, layers = 40, context = 4096, slots = 4, prompt_rows = 8, http_port = 0, max_queue = 64, max_connections = 64;
    std::uint32_t pool = 0;  // BatchForwardOptions::pool_positions (0: every slot owns `context` positions)
    std::uint32_t default_max_tokens = 1024;
    unsigned connect_timeout = 60, peer_timeout = 60;
    bool tp = true, engram_direct = true, describe = false;
    bool device_prefill = true;                      // --prefill-path device (default since P15; rows = the batch steps)
    std::uint32_t prefill_rows = 4096, prefill_min = 0;
    std::uint64_t prefix_cache_mb = 0, prefix_reserve_mb = 4096;
    std::uint32_t prefix_block = 256;
};

void event(const serve::Json& e) {
    std::printf("%s\n", e.dump().c_str());
    std::fflush(stdout);
}
[[noreturn]] void poisoned(std::uint32_t rank, const std::string& reason) {
    serve::Json e = serve::Json::object();
    e.set("event", serve::Json("POISONED"));
    e.set("rank", serve::Json(static_cast<std::uint64_t>(rank)));
    e.set("reason", serve::Json(reason));
    event(e);
    std::_Exit(3);
}

std::string file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::uint64_t fnv(std::string_view s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
    return h;
}

std::string executable_sha256() {
    static const std::string sha = strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe"));
    return sha;
}

// strix/group-manifest-v1 (TensorMeld): the fields this engine uses override the command line.
void apply_manifest(Args& a) {
    const auto m = serve::parse_json(file_text(a.manifest));
    const auto str = [&](const serve::Json* v) { return v && v->is_string() ? v->as_string() : std::string(); };
    const auto num = [&](const serve::Json* v, std::uint64_t fallback) { return v && v->is_number() ? static_cast<std::uint64_t>(v->as_int()) : fallback; };
    if (const auto schema = str(m.find("schema")); schema != "strix/group-manifest-v1")
        throw std::invalid_argument("group manifest: schema must be strix/group-manifest-v1, not '" + schema + "'");
    a.group_id = str(m.find("group_id"));
    if (a.group_id.empty()) throw std::invalid_argument("group manifest: group_id is required");
    a.epoch = num(m.find("epoch"), 0U);
    a.rank = static_cast<std::uint32_t>(num(m.find("rank"), a.rank));
    if (num(m.find("world"), 2U) != 2U) throw std::invalid_argument("group manifest: this engine group has world 2");
    if (const auto* t = m.find("transport"); t && t->is_object()) {
        const auto kind = str(t->find("kind"));
        if (kind == "usb4_host_auto") a.transport = "usb4auto";
        else if (!kind.empty()) throw std::invalid_argument("group manifest: unsupported transport kind " + kind);
        a.connect_timeout = static_cast<unsigned>(num(t->find("connect_timeout_s"), a.connect_timeout));
    }
    if (const auto* h = m.find("http"); h && h->is_object()) {
        if (const auto b = str(h->find("bind")); !b.empty()) a.http_bind = b;
        a.http_port = static_cast<std::uint32_t>(num(h->find("port"), a.http_port));
        if (const auto k = str(h->find("api_key_file")); !k.empty()) a.api_key_file = k;
    }
    a.context = static_cast<std::uint32_t>(num(m.find("context"), a.context));
    a.slots = static_cast<std::uint32_t>(num(m.find("slots"), a.slots));
    a.pool = static_cast<std::uint32_t>(num(m.find("pool_positions"), a.pool));
    if (const auto* model = m.find("model"); model && model->is_object())
        if (const auto id = str(model->find("id")); !id.empty()) a.model_id = id;
}

strix::deepseek41::BatchForwardOptions forward_options(const Args& a, strix::dist::Communicator* comm, std::uint32_t rank) {
    strix::deepseek41::BatchForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.context;
    o.slots = a.slots;
    o.pool_positions = a.pool;
    o.rank = rank;
    o.world = 2U;
    o.communicator = comm;
    o.tp_attention = a.tp;
    o.engram_direct_io = a.engram_direct;
    return o;
}

// ---- the control channel: one fixed-size block per round, rank 0 -> rank 1 (rank 1 answers with its own) ----
enum class Command : std::uint32_t { hello = 1, step = 2, heartbeat = 3, shutdown = 4, prefill = 5 };
struct Control {
    std::uint32_t magic = 0x43425344U;  // "DSBC"
    std::uint32_t command = 0;
    std::uint64_t round = 0;
    std::uint32_t rows = 0, resets = 0;  // resets: bit s = reset slot s before the step
    BatchRow row[BatchForward::max_rows]{};
    // hello
    std::uint64_t group_hash = 0, epoch = 0, model_bytes = 0, exe_hash = 0;
    std::uint32_t layers = 0, slots = 0, context = 0, tp = 0;
    // KV pool: rank 0's reserve/release operations since the last step, in order (op = kind << 30 | slot << 24 |
    // positions; kind 1 release, 2 reserve), replayed by rank 1 before the step
    std::uint32_t pool = 0, ops = 0;
    std::uint32_t op[64]{};
    // prefill: slot, prompt tokens (they follow in exchanges of kTokensPerMessage), the cache's resume boundary, shared
    // boundary and key as rank 0 found them, flags (bit 0: insert the boundaries); hello: the prefill configuration
    std::uint32_t pslot = 0, plen = 0, presume = 0, pshared = 0, pflags = 0;
    std::uint64_t pkey = 0;
    std::uint32_t prefill_mode = 0, prefill_rows = 0, prefix_block = 0;
    std::uint64_t prefix_bytes = 0;
};
constexpr std::uint32_t kTokensPerMessage = 16384U;  // 64 KiB per exchange_host
constexpr std::uint32_t kPoolRelease = 1U, kPoolReserve = 2U;
static_assert(sizeof(Control) <= 512U);

Control exchange(BatchForward& fwd, const Control& out) {
    std::array<std::byte, 512> o{}, i{};
    std::memcpy(o.data(), &out, sizeof(Control));
    fwd.exchange_host(o, i);
    Control in;
    std::memcpy(&in, i.data(), sizeof(Control));
    if (in.magic != out.magic) throw std::runtime_error("control channel: bad magic from the peer");
    return in;
}

Control hello(const Args& a) {
    Control c;
    c.command = static_cast<std::uint32_t>(Command::hello);
    c.group_hash = fnv(a.group_id);
    c.epoch = a.epoch;
    c.model_bytes = std::filesystem::file_size(a.model);
    c.exe_hash = fnv(executable_sha256());
    c.layers = a.layers;
    c.slots = a.slots;
    c.context = a.context;
    c.tp = a.tp ? 1U : 0U;
    c.pool = a.pool;
    c.prefill_mode = a.device_prefill ? 1U : 0U;
    c.prefill_rows = a.prefill_rows;
    c.prefix_block = a.prefix_block;
    c.prefix_bytes = a.prefix_cache_mb << 20U;
    return c;
}

std::string hello_mismatch(const Control& mine, const Control& peer) {
    if (peer.command != static_cast<std::uint32_t>(Command::hello)) return "the peer did not answer the hello";
    if (peer.group_hash != mine.group_hash) return "group_id differs between the ranks";
    if (peer.epoch != mine.epoch) return "epoch differs between the ranks";
    if (peer.exe_hash != mine.exe_hash) return "the engine executables differ (SHA-256)";
    if (peer.model_bytes != mine.model_bytes) return "the model files differ (size)";
    if (peer.layers != mine.layers || peer.slots != mine.slots || peer.context != mine.context || peer.tp != mine.tp || peer.pool != mine.pool)
        return "layers, slots, context, KV pool or tensor-parallel attention differ between the ranks";
    if (peer.prefill_mode != mine.prefill_mode || peer.prefill_rows != mine.prefill_rows || peer.prefix_block != mine.prefix_block ||
        peer.prefix_bytes != mine.prefix_bytes)
        return "the prefill path or the prefix cache differ between the ranks";
    return {};
}

// Watchdog: the process ends (POISONED) if a round with the peer has not completed for `limit` seconds.
class Watchdog {
public:
    Watchdog(std::uint32_t rank, unsigned limit) : rank_(rank), limit_(limit), thread_([this] { run(); }) {}
    ~Watchdog() {
        stop_.store(true);
        thread_.join();
    }
    void progress() { last_.store(Clock::now().time_since_epoch().count()); }
    void pause(bool p) {
        paused_.store(p);
        progress();
    }

private:
    void run() {
        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (paused_.load()) continue;
            const auto idle = Clock::now().time_since_epoch().count() - last_.load();
            if (std::chrono::duration<double>(Clock::duration(idle)).count() > limit_)
                poisoned(rank_, "no round with the peer for " + std::to_string(limit_) + " s");
        }
    }
    std::uint32_t rank_;
    unsigned limit_;
    std::atomic<std::int64_t> last_{Clock::now().time_since_epoch().count()};
    std::atomic<bool> stop_{false}, paused_{false};
    std::thread thread_;
};

// ---- the device prefill of a prompt into a request's slot: both ranks run it with the same arguments ----
std::uint64_t fnv_floats(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

// MemAvailable of this node in MiB (0 when unknown).
std::uint64_t mem_available_mb() {
    std::ifstream in("/proc/meminfo");
    for (std::string line; std::getline(in, line);)
        if (line.rfind("MemAvailable:", 0) == 0) return std::stoull(line.substr(13)) / 1024U;
    return 0;
}

struct PrefillOutcome {
    std::uint32_t next_token = 0, resume = 0, pieces = 0, inserted = 0;
    std::uint64_t logits = 0;
    double state_seconds = 0.0, prefill_seconds = 0.0, handoff_seconds = 0.0, seconds = 0.0;
};

// The row-split prefill of two ranks gives each 8-row window to one rank alternately: a run needs more than 8 rows so
// that each rank has a row. Shorter prompts take the rows path; pieces and resume points leave at least this many rows.
constexpr std::uint32_t kMinRows = 9U;

class SlotPrefill {
public:
    SlotPrefill(const Args& a, BatchForward& fwd, strix::dist::Communicator& comm, std::uint32_t rank) : a_(a), fwd_(fwd) {
        strix::deepseek41::PrefillOptions o;
        o.layers = a.layers;
        o.max_positions = a.context;
        o.max_rows = std::min(a.prefill_rows, a.context);
        o.rank = rank;
        o.world = 2U;
        o.communicator = &comm;
        o.engram_direct_io = a.engram_direct;
        o.shared_weights = fwd.file_form_weights();
        prefill_ = std::make_unique<strix::deepseek41::DevicePrefill>(a.model, o);
        if (a.prefix_cache_mb != 0U) {
            strix::deepseek41::PrefixCacheOptions co;
            co.block = a.prefix_block;
            co.budget_bytes = a.prefix_cache_mb << 20U;
            cache_ = std::make_unique<strix::deepseek41::PrefixCache>(co);
        }
        const std::string cache = cache_ ? ", prefix cache " + std::to_string(a.prefix_cache_mb) + " MiB in blocks of " + std::to_string(a.prefix_block) : "";
        std::fprintf(stderr, "rank %u: device prefill (pieces of %u rows) reads %.2f GiB of the batch forward's weights in place, loaded %.2f GiB%s\n", rank,
                     o.max_rows, static_cast<double>(prefill_->shared_weight_bytes()) / 1073741824.0,
                     static_cast<double>(prefill_->weight_bytes()) / 1073741824.0, cache.c_str());
    }
    [[nodiscard]] bool has_cache() const noexcept { return cache_ != nullptr; }
    strix::deepseek41::PrefixLookup lookup(std::span<const std::uint32_t> prompt) {
        return cache_ ? cache_->lookup(prompt, kMinRows) : strix::deepseek41::PrefixLookup{};
    }
    [[nodiscard]] strix::deepseek41::PrefixCacheStats cache_stats() const { return cache_ ? cache_->stats() : strix::deepseek41::PrefixCacheStats{}; }

    // The prompt's state into `slot` (reserved and reset): resumed from `hit` (a lookup of this cache), then the
    // prefill pieces; every piece end on a block boundary inserted when `insert`.
    PrefillOutcome run(std::uint32_t slot, const std::vector<std::uint32_t>& prompt, const strix::deepseek41::PrefixLookup& hit, bool insert) {
        namespace ds = strix::deepseek41;
        PrefillOutcome out;
        const auto t0 = Clock::now();
        const auto N = static_cast<std::uint32_t>(prompt.size());
        const std::uint32_t rows = std::min(a_.prefill_rows, a_.context);
        out.resume = hit.resume;
        ds::AttentionState state;
        if (hit.resume == N) {  // the whole prompt is a cached boundary
            state = cache_->state(prompt, hit.resume, hit.key);
            out.next_token = hit.next_token;
            out.logits = hit.logits_fnv;
            out.state_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        } else {
            if (hit.resume > 0U) prefill_->import_state(cache_->state(prompt, hit.resume, hit.key));
            out.state_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            std::vector<std::uint32_t> ends;
            ends = cache_ ? ds::plan_prefill_pieces(hit.resume, hit.shared, N, a_.prefix_block, rows, kMinRows)
                          : ds::plan_prefill_pieces(0U, 0U, N, rows, rows, kMinRows);
            const auto t1 = Clock::now();
            ds::PrefillResult r;
            std::uint32_t at = hit.resume;
            for (const auto end : ends) {
                const std::span<const std::uint32_t> piece(prompt.data() + at, end - at);
                r = at == 0U ? prefill_->run(piece) : prefill_->extend(piece);
                at = end;
                if (cache_ && insert && end % a_.prefix_block == 0U) {
                    auto s = prefill_->export_state();
                    s.tokens.assign(prompt.begin(), prompt.begin() + end);
                    s.next_token = r.next_token;
                    out.inserted += cache_->insert(s, fnv_floats(r.logits)) ? 1U : 0U;
                    if (end == N) state = std::move(s);
                }
            }
            out.pieces = static_cast<std::uint32_t>(ends.size());
            out.next_token = r.next_token;
            out.logits = fnv_floats(r.logits);
            out.prefill_seconds = std::chrono::duration<double>(Clock::now() - t1).count();
            if (state.layers.empty()) {
                state = prefill_->export_state();
                state.tokens = prompt;
                state.next_token = r.next_token;
            }
        }
        const auto t2 = Clock::now();
        fwd_.import_state(slot, state);
        out.handoff_seconds = std::chrono::duration<double>(Clock::now() - t2).count();
        out.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        return out;
    }

private:
    const Args& a_;
    BatchForward& fwd_;
    std::unique_ptr<strix::deepseek41::DevicePrefill> prefill_;
    std::unique_ptr<strix::deepseek41::PrefixCache> cache_;
};

// The prompt of a prefill command, rank 0 -> rank 1 in messages of kTokensPerMessage tokens (rank 1: received).
void send_prompt(BatchForward& fwd, std::span<const std::uint32_t> prompt, std::vector<std::uint32_t>* received) {
    std::vector<std::byte> o(std::size_t{kTokensPerMessage} * 4U), i(o.size());
    const std::size_t n = received ? received->size() : prompt.size();
    for (std::size_t at = 0; at < n; at += kTokensPerMessage) {
        const std::size_t take = std::min<std::size_t>(kTokensPerMessage, n - at);
        std::fill(o.begin(), o.end(), std::byte{0});
        if (!received) std::memcpy(o.data(), prompt.data() + at, take * 4U);
        fwd.exchange_host(o, i);
        if (received) std::memcpy(received->data() + at, i.data(), take * 4U);
    }
}

std::atomic<bool>& stop_requested() {
    static std::atomic<bool> s{false};
    return s;
}
extern "C" void on_signal(int) { stop_requested().store(true); }

// ---- rank 1: no API; runs what rank 0 sends ----
void follow(const Args& a, BatchForward& fwd, Watchdog& dog, SlotPrefill* prefill) {
    const auto mine = hello(a);
    const auto first = exchange(fwd, mine);
    dog.progress();
    if (const auto why = hello_mismatch(mine, first); !why.empty()) poisoned(1U, why);
    serve::Json ready = serve::Json::object();
    ready.set("event", serve::Json("READY"));
    ready.set("rank", serve::Json(1));
    ready.set("group_id", serve::Json(a.group_id));
    ready.set("epoch", serve::Json(a.epoch));
    event(ready);
    Control reply;
    for (;;) {
        reply.round += 1U;
        const auto c = exchange(fwd, reply);
        dog.progress();
        if (c.command == static_cast<std::uint32_t>(Command::shutdown)) return;
        if (c.command == static_cast<std::uint32_t>(Command::heartbeat)) continue;
        const bool is_prefill = c.command == static_cast<std::uint32_t>(Command::prefill);
        if (is_prefill ? (prefill == nullptr || c.plen == 0U || c.plen >= a.context || c.pslot >= a.slots)
                       : (c.command != static_cast<std::uint32_t>(Command::step) || c.rows == 0U || c.rows > BatchForward::max_rows))
            throw std::runtime_error("control channel: unexpected command");
        if (c.ops > std::size(c.op)) throw std::runtime_error("control channel: too many pool operations");
        for (std::uint32_t i = 0; i < c.ops; ++i) {  // the pool operations of rank 0, in its order
            const std::uint32_t kind = c.op[i] >> 30U, slot = (c.op[i] >> 24U) & 63U, positions = c.op[i] & 0xFFFFFFU;
            if (kind == kPoolRelease) fwd.release(slot);
            else if (kind != kPoolReserve || !fwd.reserve(slot, positions)) throw std::runtime_error("control channel: pool operation failed on rank 1");
        }
        for (std::uint32_t s = 0; s < a.slots; ++s)
            if ((c.resets >> s) & 1U) fwd.reset(s);
        if (is_prefill) {  // the prompt, then the same prefill as rank 0 (the cache must give rank 0's hit)
            std::vector<std::uint32_t> prompt(c.plen);
            send_prompt(fwd, {}, &prompt);
            const auto hit = prefill->lookup(prompt);
            if (hit.resume != c.presume || hit.shared != c.pshared || hit.key != c.pkey)
                throw std::runtime_error("prefix cache: rank 1 holds other entries than rank 0 (resume " + std::to_string(hit.resume) + " vs " +
                                         std::to_string(c.presume) + ")");
            dog.pause(true);  // a long prompt's prefill is one round
            (void)prefill->run(c.pslot, prompt, hit, (c.pflags & 1U) != 0U);
            dog.pause(false);
            continue;
        }
        fwd.step(std::span<const BatchRow>(c.row, c.rows));
        dog.progress();
    }
}

// ---- rank 0: HTTP + the scheduler ----
struct Running {
    serve::TakenJob job;
    BatchSequence seq;
    std::size_t published = 0;
    Clock::time_point start, first_token;
    bool has_first = false;
    bool device_prefill = false;  // the prompt goes through the device prefill before the next step
    std::uint32_t cached = 0;     // prompt tokens resumed from the prefix cache
};

struct ServeState {
    std::mutex mutex;
    std::uint64_t steps = 0, rows = 0, decode_rows = 0, prompt_rows = 0, admitted = 0;
    double step_seconds = 0.0;
    std::uint64_t prefills = 0, prefill_tokens = 0, cached_tokens = 0;
    double prefill_seconds = 0.0;
    strix::deepseek41::PrefixCacheStats cache;
    std::uint32_t busy = 0;
    std::atomic<bool> draining{false}, failed{false};
};

void lead(const Args& a, BatchForward& fwd, strix::dist::Communicator& comm, Watchdog& dog, SlotPrefill* prefill) {
    const auto mine = hello(a);
    const auto peer = exchange(fwd, mine);
    dog.progress();
    if (const auto why = hello_mismatch(mine, peer); !why.empty()) poisoned(0U, why);

    const auto tokenizer = serve::BpeTokenizer::load_gguf(a.model);
    serve::ApiConfig config;
    config.model_id = a.model_id;
    config.default_max_tokens = a.default_max_tokens;
    config.default_thinking = false;  // chat mode unless the request enables thinking
    config.sampling_policy = "greedy";
    config.engine_description = "deepseek41 engine group (2 ranks, EP experts" + std::string(a.tp ? " + TP attention" : "") + "), " +
                                std::to_string(a.slots) + " slots, greedy, exact per request";
    config.stop_tokens = {tokenizer->eos_token()};
    config.chat_renderer = [](const serve::ChatTemplateInput& in) { return serve::render_deepseek_v41_chat(in); };
    config.output_parser = [](const std::string& t, bool thinking, const serve::Json& tools) { return serve::parse_deepseek_v41_output(t, thinking, tools); };
    config.tool_call_marker = serve::deepseek_v41_tool_marker;
    if (!a.api_key_file.empty()) {
        auto key = file_text(a.api_key_file);
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) key.pop_back();
        if (key.empty()) throw std::invalid_argument("--http-api-key-file is empty");
        config.api_key = key;
    }
    ServeState state;
    serve::GenerationQueue queue(a.max_queue);
    config.on_drain = [&] {
        state.draining.store(true);
        queue.shutdown();  // queued requests end, new ones are refused; running ones finish
    };
    const auto weights = fwd.weight_bytes(), per_slot = fwd.state_bytes_per_slot(), workspace = fwd.workspace_bytes();
    config.status_extra = [&](serve::Json& out) {
        const auto [free, total] = fwd.device_memory();
        serve::Json mem = serve::Json::object();
        mem.set("weights_bytes", serve::Json(weights));
        mem.set("state_bytes_per_slot", serve::Json(per_slot));
        mem.set("state_bytes", serve::Json(per_slot * a.slots));
        mem.set("workspace_bytes", serve::Json(workspace));
        mem.set("device_free_bytes", serve::Json(static_cast<std::uint64_t>(free)));
        mem.set("device_total_bytes", serve::Json(static_cast<std::uint64_t>(total)));
        out.set("memory", std::move(mem));
        const std::lock_guard lock(state.mutex);
        serve::Json slots = serve::Json::object();
        slots.set("total", serve::Json(static_cast<std::uint64_t>(a.slots)));
        slots.set("busy", serve::Json(static_cast<std::uint64_t>(state.busy)));
        slots.set("free", serve::Json(static_cast<std::uint64_t>(a.slots - state.busy)));
        slots.set("context_tokens", serve::Json(static_cast<std::uint64_t>(a.context)));
        if (a.pool != 0U) {
            slots.set("pool_positions", serve::Json(static_cast<std::uint64_t>(a.pool)));
            slots.set("pool_bytes", serve::Json(fwd.pool_bytes()));
            slots.set("pool_largest_free", serve::Json(fwd.pool_free_positions()));
        }
        out.set("slots", std::move(slots));
        serve::Json batch = serve::Json::object();
        batch.set("steps", serve::Json(state.steps));
        batch.set("rows", serve::Json(state.rows));
        batch.set("decode_rows", serve::Json(state.decode_rows));
        batch.set("prompt_rows", serve::Json(state.prompt_rows));
        batch.set("mean_rows_per_step", serve::Json(state.steps ? static_cast<double>(state.rows) / static_cast<double>(state.steps) : 0.0));
        batch.set("mean_step_ms", serve::Json(state.steps ? state.step_seconds * 1e3 / static_cast<double>(state.steps) : 0.0));
        out.set("batch", std::move(batch));
        serve::Json pf = serve::Json::object();
        pf.set("path", serve::Json(a.device_prefill ? "device" : "rows"));
        pf.set("prefills", serve::Json(state.prefills));
        pf.set("prefill_tokens", serve::Json(state.prefill_tokens));
        pf.set("cached_tokens", serve::Json(state.cached_tokens));
        pf.set("prefill_seconds", serve::Json(state.prefill_seconds));
        if (a.prefix_cache_mb != 0U) {
            const auto& cs = state.cache;
            serve::Json pc = serve::Json::object();
            pc.set("budget_bytes", serve::Json(a.prefix_cache_mb << 20U));
            pc.set("bytes", serve::Json(cs.bytes));
            pc.set("blocks", serve::Json(cs.blocks));
            pc.set("snapshots", serve::Json(cs.snapshots));
            pc.set("lookups", serve::Json(cs.lookups));
            pc.set("hits", serve::Json(cs.hits));
            pc.set("hit_tokens", serve::Json(cs.hit_tokens));
            pc.set("inserts", serve::Json(cs.inserts));
            pc.set("evicted_snapshots", serve::Json(cs.evicted_snapshots));
            pc.set("refused", serve::Json(cs.refused));
            pf.set("prefix_cache", std::move(pc));
        }
        out.set("prefill", std::move(pf));
        const auto& cs = comm.stats();
        serve::Json group = serve::Json::object();
        group.set("group_id", serve::Json(a.group_id));
        group.set("epoch", serve::Json(a.epoch));
        group.set("rank", serve::Json(0));
        group.set("world", serve::Json(2));
        group.set("state", serve::Json(state.failed.load() ? "poisoned" : state.draining.load() ? "draining" : "ok"));
        group.set("transport", serve::Json(comm.transport_name()));
        group.set("exchanges", serve::Json(cs.exchanges));
        group.set("bytes_sent", serve::Json(cs.bytes_sent));
        group.set("exchange_seconds", serve::Json(cs.seconds));
        group.set("mean_exchange_us", serve::Json(cs.exchanges ? cs.seconds * 1e6 / static_cast<double>(cs.exchanges) : 0.0));
        out.set("group", std::move(group));
    };
    config.metrics_extra = [&] {
        std::ostringstream m;
        const std::lock_guard lock(state.mutex);
        m << "# HELP strix_slots_busy Request slots in use.\n# TYPE strix_slots_busy gauge\nstrix_slots_busy " << state.busy << '\n';
        m << "# HELP strix_batch_steps_total Forward steps.\n# TYPE strix_batch_steps_total counter\nstrix_batch_steps_total " << state.steps << '\n';
        m << "# HELP strix_batch_rows_total Rows of the forward steps.\n# TYPE strix_batch_rows_total counter\nstrix_batch_rows_total " << state.rows << '\n';
        m << "# HELP strix_group_exchanges_total Exchanges with the peer rank.\n# TYPE strix_group_exchanges_total counter\nstrix_group_exchanges_total "
          << comm.stats().exchanges << '\n';
        return m.str();
    };
    serve::ApiService api(config, tokenizer, queue, a.context, false);
    serve::HttpServerConfig http;
    http.bind_address = a.http_bind;
    http.port = static_cast<std::uint16_t>(a.http_port);
    http.max_connections = a.max_connections;
    serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) { api.handle(request, response); });
    server.listen();
    api.set_state("ready");
    std::cerr << "strix serve: listening on " << a.http_bind << ':' << server.port() << " model " << a.model_id << std::endl;
    {
        serve::Json ready = serve::Json::object();
        ready.set("event", serve::Json("READY"));
        ready.set("rank", serve::Json(0));
        ready.set("http", serve::Json(a.http_bind + ":" + std::to_string(server.port())));
        ready.set("model", serve::Json(a.model_id));
        ready.set("slots", serve::Json(static_cast<std::uint64_t>(a.slots)));
        ready.set("context", serve::Json(static_cast<std::uint64_t>(a.context)));
        ready.set("group_id", serve::Json(a.group_id));
        ready.set("epoch", serve::Json(a.epoch));
        event(ready);
    }
    std::thread http_thread([&server] { server.serve(); });

    std::vector<std::unique_ptr<Running>> running;
    std::vector<std::uint32_t> free_slots;
    for (std::uint32_t s = a.slots; s-- > 0U;) free_slots.push_back(s);
    std::uint32_t pending_resets = 0;
    std::vector<std::uint32_t> pool_ops;              // KV pool operations since the last step (sent to rank 1)
    std::deque<serve::TakenJob> pool_waiting;         // taken from the queue, waiting for room in the KV pool (FIFO)
    const auto pool_release = [&](std::uint32_t slot) {
        if (a.pool == 0U) return;
        fwd.release(slot);
        pool_ops.push_back(kPoolRelease << 30U | slot << 24U);
    };
    auto last_round = Clock::now();
    const auto finish = [&](std::size_t i, const std::string& reason, const std::string& error) {
        auto& r = *running[i];
        serve::GenerationStats st;
        st.prompt_tokens = r.seq.prompt.size();
        st.cached_tokens = r.cached;
        st.generated_tokens = r.seq.generated.size();
        st.cycles = r.seq.generated.size();
        const auto now = Clock::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>((r.has_first ? r.first_token : now) - r.start).count();
        st.decode_ms = r.has_first ? std::chrono::duration<double, std::milli>(now - r.first_token).count() : 0.0;
        st.finish = reason;
        st.sampling = "greedy";
        queue.finish(r.job, st, error);
        pool_release(r.seq.slot);
        free_slots.push_back(r.seq.slot);
        running.erase(running.begin() + static_cast<std::ptrdiff_t>(i));
        const std::lock_guard lock(state.mutex);
        state.busy = static_cast<std::uint32_t>(running.size());
    };
    // publish the new tokens, then end the requests that are done
    const auto publish = [&] {
        for (std::size_t i = running.size(); i-- > 0U;) {
            auto& r = *running[i];
            if (r.seq.generated.size() == r.published) continue;
            if (!r.has_first) {
                r.has_first = true;
                r.first_token = Clock::now();
            }
            const std::span<const std::uint32_t> fresh(r.seq.generated.data() + r.published, r.seq.generated.size() - r.published);
            r.published = r.seq.generated.size();
            const bool more = r.job.job.on_tokens ? r.job.job.on_tokens(fresh) : true;
            const auto& stops = r.job.job.stop_tokens;
            const bool stop = std::find(stops.begin(), stops.end(), fresh.back()) != stops.end();
            if (!more) finish(i, "cancelled", {});
            else if (stop) finish(i, "stop", {});
            else if (r.seq.generated.size() >= r.job.job.max_tokens) finish(i, "length", {});
            else if (r.seq.position + 1U >= a.context) finish(i, "context", {});
        }
    };
    // the device prefill of a request's prompt on both ranks (rank 1 gets the command with the pending resets and pool
    // operations, then the prompt)
    const auto device_prefill = [&](Running& r) {
        const auto& prompt = r.seq.prompt;
        const auto t0 = Clock::now();
        const auto hit = prefill->lookup(prompt);
        const bool insert = prefill->has_cache() && mem_available_mb() >= a.prefix_reserve_mb;
        Control c;
        c.command = static_cast<std::uint32_t>(Command::prefill);
        c.resets = pending_resets;
        if (pool_ops.size() > std::size(c.op)) throw std::runtime_error("too many KV pool operations in one round");
        c.ops = static_cast<std::uint32_t>(pool_ops.size());
        std::copy(pool_ops.begin(), pool_ops.end(), c.op);
        pool_ops.clear();
        c.pslot = r.seq.slot;
        c.plen = static_cast<std::uint32_t>(prompt.size());
        c.presume = hit.resume;
        c.pshared = hit.shared;
        c.pkey = hit.key;
        c.pflags = insert ? 1U : 0U;
        (void)exchange(fwd, c);
        for (std::uint32_t s = 0; s < a.slots; ++s)
            if ((pending_resets >> s) & 1U) fwd.reset(s);
        pending_resets = 0;
        send_prompt(fwd, prompt, nullptr);
        dog.pause(true);
        const auto o = prefill->run(r.seq.slot, prompt, hit, insert);
        dog.pause(false);
        last_round = Clock::now();
        r.seq.position = c.plen;
        r.seq.fed = prompt.size();
        r.seq.generated.assign(1U, o.next_token);
        r.device_prefill = false;
        r.cached = hit.resume;
        const double total = std::chrono::duration<double>(Clock::now() - t0).count();
        std::fprintf(stderr, "prefill slot %u: %u tokens, resume %u (shared %u), %u pieces, %u inserted; token %u logits_fnv %016llx; %.3f s (state %.3f, "
                     "prefill %.3f, slot import %.3f, control + prompt %.3f)\n",
                     r.seq.slot, c.plen, hit.resume, hit.shared, o.pieces, o.inserted, o.next_token, static_cast<unsigned long long>(o.logits), total,
                     o.state_seconds, o.prefill_seconds, o.handoff_seconds, total - o.seconds);
        const std::lock_guard lock(state.mutex);
        ++state.prefills;
        state.prefill_tokens += c.plen;
        state.cached_tokens += hit.resume;
        state.prefill_seconds += total;
        state.cache = prefill->cache_stats();
    };
    std::string failure;
    try {
        for (;;) {
            const bool stopping = stop_requested().load();
            if (stopping && !state.draining.load()) config.on_drain();
            if (stopping && running.empty()) break;  // SIGINT/SIGTERM: drained
            // admission between steps: oldest queued jobs into free slots
            while (!state.draining.load() && !free_slots.empty()) {
                std::optional<serve::TakenJob> job;
                if (!pool_waiting.empty()) {
                    job = std::move(pool_waiting.front());
                    pool_waiting.pop_front();
                } else {
                    job = queue.take(false);
                }
                if (!job) break;
                auto r = std::make_unique<Running>();
                r->job = std::move(*job);
                r->start = Clock::now();
                const auto& prompt = r->job.job.prompt;
                if (prompt.empty() || prompt.size() + 1U >= a.context) {
                    serve::GenerationStats st;
                    st.prompt_tokens = prompt.size();
                    st.finish = "context";
                    queue.finish(r->job, st, "prompt is empty or fills the context");
                    continue;
                }
                if (a.pool != 0U) {  // room for the prompt and every token it may generate, or wait (FIFO)
                    const auto want = static_cast<std::uint32_t>(std::min<std::uint64_t>(a.context, prompt.size() + std::uint64_t{r->job.job.max_tokens} + 1U));
                    if (!fwd.reserve(free_slots.back(), want)) {
                        if (want > a.pool) {  // never fits
                            serve::GenerationStats st;
                            st.prompt_tokens = prompt.size();
                            st.finish = "context";
                            queue.finish(r->job, st, "prompt + max_tokens exceed the KV pool");
                            continue;
                        }
                        pool_waiting.push_front(std::move(r->job));
                        break;
                    }
                    pool_ops.push_back(kPoolReserve << 30U | free_slots.back() << 24U | want);
                }
                r->seq.slot = free_slots.back();
                free_slots.pop_back();
                r->seq.prompt = prompt;
                r->device_prefill = prefill != nullptr && prompt.size() >= std::max(a.prefill_min, kMinRows);
                pending_resets |= 1U << r->seq.slot;
                running.push_back(std::move(r));
                const std::lock_guard lock(state.mutex);
                state.busy = static_cast<std::uint32_t>(running.size());
                ++state.admitted;
            }
            if (state.draining.load())  // taken jobs still waiting for pool room end with the drain
                for (; !pool_waiting.empty(); pool_waiting.pop_front()) {
                    serve::GenerationStats st;
                    st.prompt_tokens = pool_waiting.front().job.prompt.size();
                    st.finish = "cancelled";
                    queue.finish(pool_waiting.front(), st, "draining");
                }
            // cancelled requests leave before the step
            for (std::size_t i = running.size(); i-- > 0U;)
                if (running[i]->job.ticket->cancelled()) finish(i, "cancelled", {});
            // device prefills of the admitted prompts, then their first tokens
            bool prefilled = false;
            for (auto& r : running)
                if (r->device_prefill) {
                    device_prefill(*r);
                    prefilled = true;
                }
            if (prefilled) {
                publish();
                continue;  // admit again, then step the requests that are decoding
            }
            if (running.empty()) {
                if (std::chrono::duration<double>(Clock::now() - last_round).count() > 1.0) {  // keep rank 1 (and the link) alive
                    Control hb;
                    hb.command = static_cast<std::uint32_t>(Command::heartbeat);
                    (void)exchange(fwd, hb);
                    dog.progress();
                    last_round = Clock::now();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::vector<BatchSequence*> seqs;
            for (auto& r : running) seqs.push_back(&r->seq);
            const auto plan = strix::deepseek41::plan_step(seqs, a.prompt_rows);
            Control c;
            c.command = static_cast<std::uint32_t>(Command::step);
            c.rows = static_cast<std::uint32_t>(plan.rows.size());
            c.resets = pending_resets;
            std::copy(plan.rows.begin(), plan.rows.end(), c.row);
            if (pool_ops.size() > std::size(c.op)) throw std::runtime_error("too many KV pool operations in one round");
            c.ops = static_cast<std::uint32_t>(pool_ops.size());
            std::copy(pool_ops.begin(), pool_ops.end(), c.op);
            pool_ops.clear();
            (void)exchange(fwd, c);
            for (std::uint32_t s = 0; s < a.slots; ++s)
                if ((pending_resets >> s) & 1U) fwd.reset(s);
            pending_resets = 0;
            const auto t0 = Clock::now();
            const auto& tokens = fwd.step(plan.rows);
            dog.progress();
            last_round = Clock::now();
            std::uint32_t decode = 0;
            for (std::size_t r = 0; r < plan.rows.size(); ++r) decode += seqs[plan.owner[r]]->reading_prompt() ? 0U : 1U;
            strix::deepseek41::apply_step(seqs, plan, tokens);
            {
                const std::lock_guard lock(state.mutex);
                ++state.steps;
                state.rows += plan.rows.size();
                state.decode_rows += decode;
                state.prompt_rows += plan.rows.size() - decode;
                state.step_seconds += std::chrono::duration<double>(last_round - t0).count();
            }
            publish();
        }
    } catch (const std::exception& e) {
        failure = e.what();
    }
    if (!failure.empty()) {
        state.failed.store(true);
        api.set_state("failed");
        while (!running.empty()) finish(running.size() - 1U, "length", "engine failure: " + failure);
    } else {
        Control bye;
        bye.command = static_cast<std::uint32_t>(Command::shutdown);
        try {
            (void)exchange(fwd, bye);
        } catch (const std::exception& e) {
            failure = std::string("shutdown of rank 1: ") + e.what();
        }
    }
    server.stop();
    queue.shutdown();
    http_thread.join();
    if (!failure.empty()) poisoned(0U, failure);
}

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4auto") {
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::fprintf(stderr, "usb4 topology: %s\n", topology.receipt().c_str());
        return t;
    }
    throw std::invalid_argument("unknown or unavailable transport " + spec);
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport) {
    // the device prefill exchanges a chunk's MoE half sums (<= 21 MB) in one message
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, a.epoch, fnv(a.group_id)}, std::move(transport),
                                   a.device_prefill ? std::size_t{22} << 20U : std::size_t{4} << 20U);
    Watchdog dog(rank, a.peer_timeout);
    dog.pause(true);  // loading the weights is not a round
    BatchForward fwd(a.model, forward_options(a, &comm, rank));
    std::fprintf(stderr, "rank %u: weights %.2f GiB loaded in %.1f s, %u slots of %u positions (%.1f MiB state each), workspace %.1f MiB\n", rank,
                 static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds(), a.slots, a.context,
                 static_cast<double>(fwd.state_bytes_per_slot()) / 1048576.0, static_cast<double>(fwd.workspace_bytes()) / 1048576.0);
    std::unique_ptr<SlotPrefill> prefill;
    if (a.device_prefill) prefill = std::make_unique<SlotPrefill>(a, fwd, comm, rank);
    dog.pause(false);
    if (rank == 0U) lead(a, fwd, comm, dog, prefill.get());
    else follow(a, fwd, dog, prefill.get());
    serve::Json stopped = serve::Json::object();
    stopped.set("event", serve::Json("STOPPED"));
    stopped.set("rank", serve::Json(static_cast<std::uint64_t>(rank)));
    event(stopped);
}

void describe(const Args& a) {
    namespace fs = std::filesystem;
    serve::Json out = serve::Json::object();
    out.set("schema", serve::Json("strix/engine-capabilities-v1"));
    out.set("engine", serve::Json("strix-deepseek41-serve"));
    out.set("executable_sha256", serve::Json(executable_sha256()));
    serve::Json model = serve::Json::object();
    model.set("id", serve::Json(a.model_id));
    model.set("architecture", serve::Json("deepseek41"));
    model.set("file", serve::Json(fs::path(a.model).filename().string()));
    model.set("bytes", serve::Json(static_cast<std::uint64_t>(fs::file_size(a.model))));
    out.set("model", std::move(model));
    serve::Json modes = serve::Json::array();
    serve::Json mode = serve::Json::object();
    mode.set("name", serve::Json(a.tp ? "ep2-tp" : "ep2"));
    mode.set("world", serve::Json(2));
    mode.set("numerical_profile", serve::Json("ds41-exact-v1 (A8 x Q8_0 sites, K-split wo_b, canonical (shared + low) + high)"));
    mode.set("transports", serve::Json(serve::Json::Array{serve::Json("usb4_host_auto")}));
    serve::Json ranks = serve::Json::array();
    for (std::uint32_t r = 0; r < 2U; ++r) {
        auto o = forward_options(a, nullptr, r);
        const auto f = BatchForward::footprint(a.model, o);
        serve::Json mem = serve::Json::object();
        mem.set("rank", serve::Json(static_cast<std::uint64_t>(r)));
        mem.set("resident_weights_bytes", serve::Json(f.weights));
        mem.set("state_bytes_per_slot", serve::Json(f.state_per_slot));
        mem.set("workspace_bytes", serve::Json(f.workspace));
        mem.set("kv_pool_bytes", serve::Json(f.pool));
        mem.set("total_bytes", serve::Json(f.weights + f.state_per_slot * a.slots + f.workspace + f.pool));
        ranks.push(std::move(mem));
    }
    mode.set("memory", std::move(ranks));
    mode.set("context_tokens", serve::Json(static_cast<std::uint64_t>(a.context)));
    mode.set("slots", serve::Json(static_cast<std::uint64_t>(a.slots)));
    mode.set("max_slots", serve::Json(static_cast<std::uint64_t>(BatchForward::max_rows)));
    modes.push(std::move(mode));
    out.set("modes", std::move(modes));
    serve::Json api = serve::Json::object();
    api.set("openai", serve::Json(true));
    api.set("anthropic", serve::Json(true));
    api.set("sampling", serve::Json(serve::Json::Array{serve::Json("greedy")}));
    api.set("endpoints", serve::Json(serve::Json::Array{serve::Json("/v1/chat/completions"), serve::Json("/v1/completions"), serve::Json("/v1/messages"),
                                                        serve::Json("/v1/models"), serve::Json("/health"), serve::Json("/metrics"),
                                                        serve::Json("/api/status"), serve::Json("/admin/drain"), serve::Json("/tokenize"),
                                                        serve::Json("/detokenize")}));
    out.set("api", std::move(api));
    out.set("argv_template", serve::Json(serve::Json::Array{serve::Json("{executable}"), serve::Json("{model}"), serve::Json("--group-manifest"),
                                                            serve::Json("{manifest}")}));
    out.set("events", serve::Json(serve::Json::Array{serve::Json("READY"), serve::Json("POISONED"), serve::Json("STOPPED")}));
    std::printf("%s\n", out.dump().c_str());
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    bool tp_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--rank") a.rank = u32();
        else if (s == "--transport") a.transport = next();
        else if (s == "--group-manifest") a.manifest = next();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--context") a.context = u32();
        else if (s == "--slots") a.slots = u32();
        else if (s == "--pool-positions") a.pool = u32();
        else if (s == "--prompt-rows") a.prompt_rows = u32();
        else if (s == "--tp-attention") a.tp = tp_set = true;
        else if (s == "--no-tp-attention") a.tp = false, tp_set = true;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--engram-mapped") a.engram_direct = false;
        else if (s == "--http-bind") a.http_bind = next();
        else if (s == "--http-port") a.http_port = u32();
        else if (s == "--http-api-key-file") a.api_key_file = next();
        else if (s == "--http-max-queue") a.max_queue = u32();
        else if (s == "--http-max-connections") a.max_connections = u32();
        else if (s == "--model-id") a.model_id = next();
        else if (s == "--default-max-tokens") a.default_max_tokens = u32();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--peer-timeout") a.peer_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--describe-json") a.describe = true;
        else if (s == "--prefill-path") {
            const auto v = next();
            if (v != "rows" && v != "device") throw std::invalid_argument("--prefill-path is rows or device");
            a.device_prefill = v == "device";
        }
        else if (s == "--prefill-rows") a.prefill_rows = u32();
        else if (s == "--prefill-min") a.prefill_min = u32();
        else if (s == "--prefix-cache-mb") a.prefix_cache_mb = std::stoull(next());
        else if (s == "--prefix-block") a.prefix_block = u32();
        else if (s == "--prefix-reserve-mb") a.prefix_reserve_mb = std::stoull(next());
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    (void)tp_set;
    if (a.model.empty()) throw std::invalid_argument("usage: strix-deepseek41-serve MODEL.gguf --rank R --transport T | --group-manifest FILE | --describe-json");
    if (!a.manifest.empty()) apply_manifest(a);
    if (a.slots == 0U || a.slots > BatchForward::max_rows) throw std::invalid_argument("--slots must be in 1..8");
    if (a.prompt_rows == 0U || a.prompt_rows > BatchForward::max_rows) throw std::invalid_argument("--prompt-rows must be in 1..8");
    if (a.rank > 1U) throw std::invalid_argument("--rank is 0 or 1");
    if (a.prefill_rows < 32U || a.prefill_rows % 8U != 0U) throw std::invalid_argument("--prefill-rows must be a multiple of 8, at least 32");
    if (a.prefix_block == 0U || a.prefix_block % 8U != 0U) throw std::invalid_argument("--prefix-block must be a positive multiple of 8");
    if (a.prefix_cache_mb != 0U && !a.device_prefill) throw std::invalid_argument("--prefix-cache-mb needs --prefill-path device");
    if (a.describe) {
        describe(a);
        return 0;
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (a.transport == "emulate") {  // both ranks in this process over the in-memory pair (one-node tests, few layers)
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t));
            } catch (...) {
                failure = std::current_exception();
            }
        });
        run_rank(a, 0U, std::move(t0));
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    ::alarm(a.connect_timeout);
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout);
    ::alarm(0);
    std::fprintf(stderr, "connected (%s)\n", transport->name().c_str());
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport));
    return 0;
} catch (const std::exception& e) {
    serve::Json ev = serve::Json::object();
    ev.set("event", serve::Json("POISONED"));
    ev.set("reason", serve::Json(std::string(e.what())));
    std::printf("%s\n", ev.dump().c_str());
    std::fflush(stdout);
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
