// strix-minimax-serve: the MiniMax M3.1 Flash TP2 engine behind an OpenAI-compatible HTTP API (both ranks run it;
// rank 0 serves HTTP). The DS serve recipe: rank 0 owns the queue and a continuous-batching scheduler over the
// engine's sequence slots and sends one fixed-size control block per round (prefill / unit / step / draft / heartbeat /
// shutdown); rank 1 replays it, so both ranks make the same engine calls and produce the same tokens (greedy, exact).
//
// usage: strix-minimax-serve MODEL_DIR --rank R --transport T [--port 8080] [--bind 127.0.0.1] [--slots 8]
//        [--capacity 8256] [--max-rows 4096] [--model-id minimax-m3.1-flash] [--max-tokens 4096]
//        [--dspark DIR [--dspark-exl3 DIR]] [--admin] [--resume-check P.u32[:k/k...][,...] [--check-only]]
//   TRANSPORT: usb4-auto | tcp-listen:PORT | tcp-connect:HOST:PORT
// Routes (rank 0): POST /v1/chat/completions, POST /v1/completions (stream or not), GET /v1/models, GET /health,
// GET /metrics (metrics text), GET /api/status[?detail=1] (strix-serve-status-v1), POST /tokenize; with --admin,
// GET / POST /admin/sched (scheduler policy at runtime, for A/B in one load). The API and its observability are in
// src/minimax/serve/minimax_api.* and minimax_observe.* (host only, HTTP threads); this file gives them the scheduler's
// per-request timings, its mm_* lines and /admin/sched. Sampling parameters are accepted and ignored: decoding is
// greedy (said in the reply).
//
// Scheduler (qualification notes), rank 0's decisions only (rank 1 runs the commands):
//   STRIX_MM_SERVE_SCHED=off|interleave (default off until measured): off = an admitted prompt's whole prefill runs at
//     once and the decoding requests wait for all of it (the first scheduler). interleave = the prefill runs as a
//     resumable task in LAYER units (Engine::prefill_begin / prefill_resume: the same kernel calls and rows as one
//     prefill, so the bits are those of the request alone; row units or batched prompts are not exact here: the prefill
//     split-K policy follows the call's row count) between decode steps, when its predicted time exceeds the token-gap
//     target STRIX_MM_SERVE_GAP_MS (default 1500); a unit's layers are chosen so that step + unit stays within the gap.
//   --dspark: DSpark windows inside the batch steps, sharing the step's 8 rows (n decoding rows + drafts <= 8: C1 up to 7
//     drafts, C8 none). Requests admitted while <= STRIX_MM_SERVE_DSPARK_INFLIGHT (default 1: measurement C2 measured slower) are in flight keep a
//     drafter context (prefill taps + every committed row); above it appends stop and those contexts are stale for good.
//     Each cycle a gate per decoding count decides whether to draft; the drafting slots' windows are allocated jointly
//     (expected tokens per measured cycle cost over every feasible total). Drafts change acceptance only, never output.
//     OFF by default (measurement: C1 +8 % aggregate but prose -10 %, C2 -3 %): STRIX_MM_SERVE_DSPARK=1 (or /admin/sched) turns
//     the loaded drafter on.
#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_observe.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/minimax/engine.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/xfer/minimax_xfer.hpp"
#include "hip_minimax_dspark.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
constexpr std::uint32_t kStepRows = 8;  // Engine::step's row limit

std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
double env_num(const char* name, double def) {
    const char* v = std::getenv(name);
    return v == nullptr || *v == 0 ? def : std::atof(v);
}

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    const std::chrono::seconds limit(300);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto") {
        strix::dist::Usb4AutoOptions o;
        o.minimax_compat = true;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

// The transport shared in turn by the communicator and the exchange module (as strix-minimax-rank).
class Borrowed final : public strix::dist::Transport {
public:
    explicit Borrowed(strix::dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    [[nodiscard]] std::string name() const override { return t_.name(); }
    void abort(const std::string& why) override { t_.abort(why); }

private:
    strix::dist::Transport& t_;
};

// ---- control channel: one 512-byte block per round, rank 0 -> rank 1 (rank 1 answers with its own) ----
// prefill: opens the resumable prefill of `plen` tokens (they follow in kTokensPerMessage chunks) in `pslot`; pflags bit 0
//   = the prefill's taps go to the slot's drafter context. unit: the next `layers` layers of the open prefill. step: the
//   rows (one row per slot, or a slot's verify window: its anchor then its drafts at consecutive positions); `append` bit
//   s = slot s's committed rows (anchor + accepted drafts, from the step's tokens on each rank) are appended to its drafter
//   context. draft: the 7-row DSpark block + 7 Markov steps of each of `dslots` slots from its anchor at its position.
//   taps: the engine's draft taps on (pflags 1) / off (0), only with no prefill open (DSpark switched at runtime).
enum class Command : std::uint32_t { heartbeat = 1, prefill = 2, step = 3, shutdown = 4, unit = 5, cancel = 6, draft = 7, taps = 8 };
struct Control {
    std::uint32_t magic = 0x43534d4dU;  // "MMSC"
    std::uint32_t command = 0;
    std::uint32_t rows = 0;             // step: rows used in row[]
    std::uint32_t pslot = 0, plen = 0;  // prefill: slot and prompt length
    std::uint32_t pflags = 0;
    std::uint64_t round = 0;
    mm::Row row[kStepRows]{};
    std::uint32_t layers = 0;  // unit
    std::uint32_t append = 0;  // step
    std::uint32_t dslots = 0;  // draft
    std::uint32_t dslot[kStepRows]{}, danchor[kStepRows]{};
    std::uint64_t dpos[kStepRows]{};
};
static_assert(sizeof(Control) <= 512);
constexpr std::size_t kTokensPerMessage = 16384;  // 64 KiB per control exchange

Control exchange(mm::Engine& eng, const Control& out) {
    std::array<std::byte, 512> o{}, i{};
    std::memcpy(o.data(), &out, sizeof(Control));
    eng.exchange_bytes(o.data(), i.data(), o.size());
    Control in;
    std::memcpy(&in, i.data(), sizeof(Control));
    if (in.magic != out.magic) throw std::runtime_error("control channel: bad magic from the peer");
    return in;
}
// Prompt tokens of a prefill, rank 0 -> rank 1.
void exchange_tokens(mm::Engine& eng, std::vector<std::uint32_t>& tokens, std::size_t n, bool sender) {
    if (!sender) tokens.assign(n, 0);
    std::vector<std::uint32_t> in(kTokensPerMessage);
    for (std::size_t c0 = 0; c0 < n; c0 += kTokensPerMessage) {
        const std::size_t m = std::min(kTokensPerMessage, n - c0);
        std::vector<std::uint32_t> out(m, 0);
        if (sender) std::copy_n(tokens.begin() + static_cast<long>(c0), m, out.begin());
        eng.exchange_bytes(out.data(), in.data(), m * 4);
        if (!sender) std::copy_n(in.begin(), m, tokens.begin() + static_cast<long>(c0));
    }
}

// ---- the engine calls of every command, made identically by both ranks ----
struct Drafts {
    std::array<std::uint32_t, ds::block> tokens{};
    std::array<float, ds::block> confidence{};
};
struct Runner {
    mm::Engine& eng;
    ds::DeviceDrafter* dd = nullptr;  // DSpark drafter (both ranks), nullptr = none
    std::uint32_t v0 = 0, vn = 0;     // this rank's vocabulary half (drafter Markov steps)

    void prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> prompt, bool context) {
        if (dd != nullptr) {
            if (context)
                eng.set_prefill_tap_sink([this](const float* taps, std::size_t n, std::uint32_t s, std::uint64_t pos0) { dd->append(s, taps, n, pos0, nullptr); });
            else
                eng.set_prefill_tap_sink({});
        }
        eng.prefill_begin(slot, prompt, 0);
    }
    bool prefill_unit(unsigned layers, std::uint32_t* next) {
        const bool done = eng.prefill_resume(layers, next);
        if (done && dd != nullptr) eng.set_prefill_tap_sink({});  // scoped to the request it was set for
        return done;
    }
    void prefill_cancel() {
        eng.prefill_cancel();
        if (dd != nullptr) eng.set_prefill_tap_sink({});
    }
    // DSpark block + Markov chain (one 8-byte candidate exchange per step) + confidences, as strix-minimax-rank --spec auto.
    Drafts draft(std::uint32_t slot, std::uint32_t anchor, std::uint64_t pos) {
        Drafts out;
        dd->forward_block(slot, eng.embed_row(anchor), eng.embed_row(ds::noise_token), pos, nullptr);
        const float* logits = eng.head_logits(dd->y(), ds::block);
        std::array<std::uint32_t, ds::block> prev{};
        std::uint32_t t = anchor;
        for (std::uint32_t i = 0; i < ds::block; ++i) {
            prev[i] = t;
            const ds::Candidate mine = dd->markov_step_host(t, logits + static_cast<std::size_t>(i) * vn, v0, v0 + vn, nullptr);
            const auto theirs = eng.peer2(__builtin_bit_cast(std::uint32_t, mine.value), mine.id);
            t = ds::combine(mine, ds::Candidate{__builtin_bit_cast(float, theirs[0]), theirs[1]}).id;
            out.tokens[i] = t;
        }
        out.confidence = dd->confidence_host(prev, nullptr);
        return out;
    }
    // One step; then each slot of `append` gets its committed rows appended (its anchor + the drafts the target accepted:
    // the window's contiguous tap rows, before the next call overwrites them).
    std::vector<std::uint32_t> step(std::span<const mm::Row> rows, std::uint32_t append) {
        auto g = eng.step(rows);
        if (dd != nullptr && append != 0) {
            const float* taps = eng.last_taps();
            for (std::size_t i0 = 0; i0 < rows.size();) {
                std::size_t i1 = i0 + 1;
                while (i1 < rows.size() && rows[i1].slot == rows[i0].slot) ++i1;
                if ((append >> rows[i0].slot) & 1U) {
                    std::vector<std::uint32_t> d;
                    for (std::size_t i = i0 + 1; i < i1; ++i) d.push_back(rows[i].token);
                    const auto v = ds::verify_greedy(d, std::span<const std::uint32_t>(g.data() + i0, i1 - i0));
                    dd->append(rows[i0].slot, taps + i0 * ds::taps * 6144, v.accepted + 1, rows[i0].pos, nullptr);
                }
                i0 = i1;
            }
        }
        return g;
    }
};

// ---- rank 0: scheduler policy (runtime switchable) and counters ----
enum class Sched : std::uint32_t { off = 0, interleave = 1 };
struct Policy {
    Sched sched = Sched::off;
    double gap_ms = 1500;         // token-gap target of the decoding requests (interleave)
    bool dspark = true;           // with a drafter loaded
    std::uint32_t inflight = 1;   // DSpark contexts kept while <= this many requests are in flight (measurement: C2 slower)
    std::uint32_t draft_slots = 8;  // at most this many slots draft in one cycle
    std::uint64_t epoch = 0;      // bumped by every change (the lead resets its cost tables / gates)
};
struct Counters {
    double step_s = 0, prefill_s = 0, draft_s = 0, idle_s = 0, stall_s = 0;
    std::uint64_t cancels = 0;  // open prefills abandoned (client gone)
    // DSpark coverage: cycles with >= 2 windows (k >= 1), of which with unequal k; windows with a rejected draft
    std::uint64_t multi_windows = 0, unequal_windows = 0, rejections = 0;
    std::uint64_t steps = 0, step_rows = 0, units = 0, prefills = 0, drafts = 0, drafted = 0, accepted = 0, plain_cycles = 0,
                  draft_cycles = 0;
    std::array<std::uint64_t, kStepRows + 1> rows_hist{};
};
// Per-request extras beyond sv::GenerationStats, for the reply's timings and the request log.
using Extra = ms::RequestExtra;
struct Shared {
    std::mutex m;
    Policy policy;
    Counters counters;
    std::unordered_map<const sv::GenerationTicket*, Extra> extras;
};

struct Running {
    sv::TakenJob job;
    std::uint32_t slot = 0;
    std::uint64_t pos = 0;  // position of `last` (the next input token)
    std::uint32_t last = 0;
    sv::GenerationStats stats;
    Extra x;
    Clock::time_point decode_start, last_emit;
    bool prefilling = false;
    bool context = false;  // DSpark: the drafter context of this slot is complete
    ds::SurvivalController survival{16};
    double tau = 1.0;      // EMA of emitted tokens per drafting cycle (which slots draft first)
};

bool is_stop(const Running& r, std::uint32_t id) {
    return std::find(r.job.job.stop_tokens.begin(), r.job.job.stop_tokens.end(), id) != r.job.job.stop_tokens.end();
}

// Cycle costs measured by rank 0 (seconds), by total rows and step shape: plain (one row per slot), one slot's window
// (fused MSA), mixed (several slots, at least one window: per-row MSA). Unmeasured row counts are scaled from the
// measured ones of the same shape by 1 + 0.12 (R - r) (the measurement verify slope), then from plain.
struct Costs {
    std::array<std::array<double, kStepRows + 1>, 3> v{};
    double draft = 0;  // one slot's draft (block + head + Markov chain)
    void reset() { *this = Costs{}; }
    static void ema(double& c, double x) { c = c <= 0 ? x : 0.8 * c + 0.2 * x; }
    void observe(unsigned shape, std::uint32_t rows, double s) { ema(v[shape][rows], s); }
    [[nodiscard]] double at(unsigned shape, std::uint32_t rows) const {
        for (unsigned sh : {shape, 0U}) {
            double base = 0;
            int n = 0;
            for (std::uint32_t r = 1; r <= kStepRows; ++r)
                if (v[sh][r] > 0) {
                    base += v[sh][r] / (1.0 + 0.12 * (r - 1));
                    ++n;
                }
            if (v[sh][rows] > 0) return v[sh][rows];
            if (n > 0) return base / n * (1.0 + 0.12 * (rows - 1));
        }
        return 0.05 * (1.0 + 0.12 * (rows - 1));
    }
};

// Prefill layer-time model of rank 0 (interleave): per rows bucket (log2), EMA of ms per layer from the measured units,
// plus a fixed cost per unit (restore, cache table, drain). Prior 10 ms + 0.1 ms x rows per layer (measurement: chats ~25 ms,
// 1K 108 ms, 3K 314 ms per layer).
struct LayerModel {
    std::array<double, 16> per_layer{};
    double fixed_ms = 5.0;
    static unsigned bucket(std::uint64_t rows) {
        unsigned b = 0;
        while ((std::uint64_t{1} << b) < rows && b < 15) ++b;
        return b;
    }
    [[nodiscard]] double layer_ms(std::uint64_t rows) const {
        const double v = per_layer[bucket(rows)];
        return v > 0 ? v : 10.0 + 0.1 * static_cast<double>(rows);
    }
    void observe(std::uint64_t rows, unsigned layers, double ms) {
        if (layers < 4) return;
        const double x = std::max(0.0, ms - fixed_ms) / layers;
        double& c = per_layer[bucket(rows)];
        c = c <= 0 ? x : 0.7 * c + 0.3 * x;
    }
};

void log_request(const Running& r, const char* finish) {
    std::fprintf(stderr,
                 "{\"event\":\"mmserve-req\",\"slot\":%u,\"prompt_tokens\":%llu,\"prompt_fnv\":\"%s\",\"tokens\":%llu,\"ids_fnv\":\"%s\","
                 "\"finish\":\"%s\",\"queue_ms\":%.1f,\"ttft_ms\":%.1f,\"prefill_ms\":%.1f,\"prefill_units\":%llu,\"decode_ms\":%.1f,"
                 "\"max_gap_ms\":%.1f,\"stall_ms\":%.1f,\"cycles\":%llu,\"drafted\":%llu,\"accepted\":%llu,\"context\":%s}\n",
                 r.slot, static_cast<unsigned long long>(r.stats.prompt_tokens), hex(r.x.prompt_fnv).c_str(),
                 static_cast<unsigned long long>(r.stats.generated_tokens), hex(r.x.ids_fnv).c_str(), finish,
                 std::chrono::duration<double, std::milli>(r.job.started - r.job.submitted).count(), r.x.ttft_ms, r.x.prefill_wall_ms,
                 static_cast<unsigned long long>(r.x.units), r.stats.decode_ms, r.x.max_gap_ms, r.x.stall_ms,
                 static_cast<unsigned long long>(r.x.cycles), static_cast<unsigned long long>(r.x.drafted),
                 static_cast<unsigned long long>(r.x.accepted), r.x.context ? "true" : "false");
}

// ---- rank 0: the scheduler ----
class Lead {
public:
    Lead(mm::Engine& eng, Runner& run, sv::GenerationQueue& queue, Shared& shared, std::uint64_t max_rows, bool taps)
        : eng_(eng), run_(run), queue_(queue), sh_(shared), slots_(eng.slots()), capacity_(eng.capacity()), max_rows_(max_rows),
          run_slots_(slots_), taps_(taps) {}

    void loop() {
        auto last_round = Clock::now();
        while (!g_stop.load() && !queue_.stopping()) {
            refresh_policy();
            sync_taps();
            bool worked = false;
            // a client gone while its prefill is open: the task is abandoned at this unit boundary (both ranks)
            if (eng_.prefill_open() && run_slots_[prefill_slot_]->job.ticket->cancelled()) {
                Control c;
                c.command = static_cast<std::uint32_t>(Command::cancel);
                c.round = round_++;
                (void)exchange(eng_, c);
                run_.prefill_cancel();
                auto& r = run_slots_[prefill_slot_];
                r->decode_start = Clock::now();
                r->stats.prefill_ms = ms_since(prefill_start_);
                finish(r, "cancelled");
                add([](Counters& k) { ++k.cancels; });
            }
            // admission: one open prefill at a time
            if (!eng_.prefill_open()) worked |= admit();
            if (eng_.prefill_open()) {
                const bool off = pol_.sched == Sched::off;
                const bool fresh = run_slots_[prefill_slot_]->x.units == 0;  // a started task keeps interleaving
                if (off || decoding_count() == 0 || (fresh && predicted_rest_ms() <= pol_.gap_ms)) {
                    prefill_unit(~0U);
                    last_round = Clock::now();
                    // off (today's order) or nothing decoding: the next admission comes before any step; interleave
                    // with requests decoding: one decode step first, so successive whole prefills do not add up in one gap
                    if (off || decoding_count() == 0) continue;
                    worked = true;
                }
            }
            if (decoding_count() > 0) {
                cycle();
                worked = true;
            }
            if (eng_.prefill_open()) {
                prefill_unit(unit_layers());
                worked = true;
            }
            if (worked) {
                last_round = Clock::now();
                continue;
            }
            const auto idle0 = Clock::now();
            if (std::chrono::duration<double>(Clock::now() - last_round).count() > 1.0) {  // keep rank 1 and the link alive
                Control hb;
                hb.command = static_cast<std::uint32_t>(Command::heartbeat);
                hb.round = round_++;
                (void)exchange(eng_, hb);
                last_round = Clock::now();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            add([&](Counters& c) { c.idle_s += secs(idle0, Clock::now()); });
        }
        if (eng_.prefill_open()) {
            Control c;
            c.command = static_cast<std::uint32_t>(Command::cancel);
            c.round = round_++;
            (void)exchange(eng_, c);
            run_.prefill_cancel();
        }
        for (auto& r : run_slots_)
            if (r) finish(r, "cancelled");
        Control bye;
        bye.command = static_cast<std::uint32_t>(Command::shutdown);
        (void)exchange(eng_, bye);
    }

private:
    template <class F> void add(F f) {
        std::lock_guard<std::mutex> l(sh_.m);
        f(sh_.counters);
    }
    // The engine's draft taps follow DSpark on / off (taps cost capture work in every call): switched on both ranks with
    // no prefill open; requests admitted while they were off have no drafter context.
    void sync_taps() {
        const bool want = run_.dd != nullptr && pol_.dspark;
        if (want == taps_ || eng_.prefill_open()) return;
        Control c;
        c.command = static_cast<std::uint32_t>(Command::taps);
        c.round = round_++;
        c.pflags = want ? 1U : 0U;
        (void)exchange(eng_, c);
        eng_.set_taps(want);
        taps_ = want;
    }
    void refresh_policy() {
        std::lock_guard<std::mutex> l(sh_.m);
        if (sh_.policy.epoch != pol_.epoch) {  // a new arm: the same initial state as a fresh load
            costs_.reset();
            layer_ = LayerModel{};
            gates_.assign(kStepRows + 1, ds::DraftGate());
            explore_ = {};
        }
        pol_ = sh_.policy;
    }
    std::uint32_t decoding_count() const {
        std::uint32_t n = 0;
        for (const auto& r : run_slots_) n += r && !r->prefilling ? 1U : 0U;
        return n;
    }
    std::uint32_t inflight() const {
        std::uint32_t n = 0;
        for (const auto& r : run_slots_) n += r ? 1U : 0U;
        return n;
    }

    void finish(std::optional<Running>& r, const std::string& why) {
        r->stats.finish = why;
        r->stats.decode_ms = ms_since(r->decode_start);
        r->stats.cycles = r->x.cycles;
        r->stats.proposed = r->x.drafted;
        r->stats.accepted = r->x.accepted;
        // armed instrumentation (requests the queue marked observed at submission): submit -> first emitted token, the
        // time emit() already took right before that token's publish; none when no token was emitted
        if (r->job.job.observed && r->stats.generated_tokens > 0) r->stats.first_token_ms = r->x.ttft_ms;
        log_request(*r, why.c_str());
        {
            std::lock_guard<std::mutex> l(sh_.m);
            sh_.extras[r->job.ticket.get()] = r->x;
        }
        queue_.finish(r->job, r->stats);
        r.reset();
    }

    // Emits `ids` of request r in order until a stop token, a cancellation or its token limit; returns false when r ended.
    bool emit(std::optional<Running>& r, std::span<const std::uint32_t> ids, std::uint32_t* emitted = nullptr) {
        for (const auto id : ids) {
            if (emitted != nullptr) ++*emitted;
            const auto now = Clock::now();
            if (r->stats.generated_tokens == 0) r->x.ttft_ms = std::chrono::duration<double, std::milli>(now - r->job.submitted).count();
            else r->x.max_gap_ms = std::max(r->x.max_gap_ms, std::chrono::duration<double, std::milli>(now - r->last_emit).count());
            r->last_emit = now;
            r->x.ids_fnv = fnv(&id, 4, r->x.ids_fnv);
            ++r->stats.generated_tokens;
            const bool more = r->job.job.on_tokens(std::span(&id, 1));
            if (is_stop(*r, id)) {
                finish(r, "stop");
                return false;
            }
            if (!more) {
                finish(r, "cancelled");
                return false;
            }
            if (r->stats.generated_tokens >= r->job.job.max_tokens) {
                finish(r, "length");
                return false;
            }
        }
        return true;
    }

    bool admit() {
        std::uint32_t free_slot = slots_;
        for (std::uint32_t s = 0; s < slots_; ++s)
            if (!run_slots_[s]) {
                free_slot = s;
                break;
            }
        if (free_slot >= slots_) return false;
        auto t = queue_.take(false);
        if (!t) return false;
        auto& prompt = t->job.prompt;
        if (prompt.empty() || prompt.size() + 1 >= capacity_) {
            queue_.finish(*t, {}, "prompt of " + std::to_string(prompt.size()) + " tokens does not fit the context (" +
                                      std::to_string(capacity_) + ")");
            return true;
        }
        Running r;
        r.slot = free_slot;
        r.prefilling = true;
        r.stats.prompt_tokens = prompt.size();
        r.x.prompt_fnv = fnv(prompt.data(), prompt.size() * 4);
        r.x.ids_fnv = 14695981039346656037ULL;
        // DSpark context: a drafter, DSpark on, few enough requests in flight, a one-chunk prompt (longer ones keep the
        // grouped prefill), room for the 7-row block
        r.context = run_.dd != nullptr && pol_.dspark && taps_ && inflight() + 1 <= pol_.inflight && prompt.size() <= max_rows_ &&
                    prompt.size() + ds::block + 1 < capacity_;
        r.x.context = r.context;
        Control c;
        c.command = static_cast<std::uint32_t>(Command::prefill);
        c.round = round_++;
        c.pslot = free_slot;
        c.plen = static_cast<std::uint32_t>(prompt.size());
        c.pflags = r.context ? 1U : 0U;
        (void)exchange(eng_, c);
        exchange_tokens(eng_, prompt, prompt.size(), true);
        run_.prefill_begin(free_slot, prompt, r.context);
        r.job = std::move(*t);
        prefill_start_ = Clock::now();
        run_slots_[free_slot] = std::move(r);
        prefill_slot_ = free_slot;
        add([](Counters& k) { ++k.prefills; });
        return true;
    }

    // predicted rest of the open prefill (ms): the open unit's remaining layers, then 60 layers of each later unit (groups
    // of up to two max_rows chunks, the engine's default pf_group)
    double predicted_rest_ms() const {
        const std::uint64_t unit = eng_.prefill_unit_rows();
        double ms = layer_.fixed_ms + static_cast<double>(60 - eng_.prefill_layer()) * layer_.layer_ms(unit);
        for (std::uint64_t left = eng_.prefill_rows_left() - unit; left > 0;) {
            const std::uint64_t u = std::min<std::uint64_t>(left, 2 * max_rows_);
            ms += layer_.fixed_ms + 60.0 * layer_.layer_ms(u);
            left -= u;
        }
        return ms;
    }
    unsigned unit_layers() const {
        if (decoding_count() == 0) return ~0U;
        const double step_ms = 1e3 * costs_.at(0, decoding_count());
        const double budget = std::max(1.0, pol_.gap_ms - step_ms - layer_.fixed_ms);
        const double per = layer_.layer_ms(std::max<std::uint64_t>(1, eng_.prefill_unit_rows()));
        return std::max(1U, static_cast<unsigned>(budget / per));
    }

    void prefill_unit(unsigned layers) {
        Control c;
        c.command = static_cast<std::uint32_t>(Command::unit);
        c.round = round_++;
        c.layers = layers;
        (void)exchange(eng_, c);
        const std::uint64_t unit_rows = eng_.prefill_unit_rows(), left0 = eng_.prefill_rows_left();
        const unsigned layer0 = eng_.prefill_layer();
        const auto t0 = Clock::now();
        std::uint32_t next = 0;
        const bool done = run_.prefill_unit(layers, &next);
        const double ms = ms_since(t0);
        // layer time of units that stayed inside one chunk / group (or ended the last one)
        if (!done && eng_.prefill_rows_left() == left0 && eng_.prefill_layer() > layer0) layer_.observe(unit_rows, eng_.prefill_layer() - layer0, ms);
        if (done && left0 == unit_rows) layer_.observe(unit_rows, 60 - layer0, ms);
        auto& r = run_slots_[prefill_slot_];
        ++r->x.units;
        // the decoding requests stalled for this unit
        double stalled = 0;
        for (auto& o : run_slots_)
            if (o && !o->prefilling) {
                o->x.stall_ms += ms;
                stalled += ms / 1e3;
            }
        add([&](Counters& k) {
            k.prefill_s += ms / 1e3;
            k.stall_s += stalled;
            ++k.units;
        });
        if (!done) return;
        r->prefilling = false;
        r->stats.prefill_ms = ms_since(prefill_start_);
        r->x.prefill_wall_ms = r->stats.prefill_ms;
        r->pos = r->stats.prompt_tokens;
        r->last = next;
        r->decode_start = Clock::now();
        (void)emit(r, std::span(&next, 1));
    }

    // One decode cycle over every decoding request: a plain batch step, or DSpark drafts + verify windows within 8 rows.
    void cycle() {
        std::vector<std::uint32_t> order;
        for (std::uint32_t s = 0; s < slots_; ++s)
            if (run_slots_[s] && !run_slots_[s]->prefilling) {
                if (run_slots_[s]->pos + 1 >= capacity_) {
                    finish(run_slots_[s], "context");
                    continue;
                }
                order.push_back(s);
            }
        if (order.empty()) return;
        const auto n = static_cast<std::uint32_t>(order.size());
        // DSpark contexts: kept only while few requests are in flight (each append reads the 94 MB fc)
        const bool keep = run_.dd != nullptr && pol_.dspark && taps_ && inflight() <= pol_.inflight;
        std::uint32_t append = 0;
        for (auto s : order) {
            auto& r = run_slots_[s];
            if (!keep) r->context = false;  // stale for good (acceptance only)
            if (r->context) append |= 1U << s;
        }
        // drafting slots: contexts, room for the block, more than one token to go; the budget is the step's free rows
        const std::uint32_t budget = kStepRows - n;
        std::vector<std::uint32_t> cand;
        if (keep && budget > 0)
            for (auto s : order) {
                const auto& r = run_slots_[s];
                if (r->context && r->pos + ds::block + 1 < capacity_ && r->stats.generated_tokens + 1 < r->job.job.max_tokens) cand.push_back(s);
            }
        bool drafted = false;
        std::map<std::uint32_t, Drafts> drafts;
        const auto c0 = Clock::now();
        double t_draft = 0;
        if (!cand.empty() && gates_[n].draft()) {
            std::sort(cand.begin(), cand.end(), [&](auto a, auto b) { return run_slots_[a]->tau > run_slots_[b]->tau; });
            cand.resize(std::min<std::size_t>({cand.size(), budget, pol_.draft_slots}));
            Control c;
            c.command = static_cast<std::uint32_t>(Command::draft);
            c.round = round_++;
            c.dslots = static_cast<std::uint32_t>(cand.size());
            for (std::size_t i = 0; i < cand.size(); ++i) {
                c.dslot[i] = cand[i];
                c.danchor[i] = run_slots_[cand[i]]->last;
                c.dpos[i] = run_slots_[cand[i]]->pos;
            }
            (void)exchange(eng_, c);
            for (auto s : cand) drafts[s] = run_.draft(s, run_slots_[s]->last, run_slots_[s]->pos);
            drafted = true;
            t_draft = secs(c0, Clock::now());
            Costs::ema(costs_.draft, t_draft / static_cast<double>(cand.size()));
        }
        // windows: every feasible total of draft rows j = 0..budget from the per-slot survival increments (each slot's
        // increments do not increase with depth, so the j best increments are the best j-row allocation); the j with the
        // most expected tokens per cycle cost, lowest on ties; every 16th drafting cycle per n takes the largest
        std::map<std::uint32_t, std::uint32_t> k;
        if (drafted) {
            struct Inc {
                double gain;
                std::uint32_t slot, depth;
            };
            std::vector<Inc> inc;
            for (auto& [s, d] : drafts) {
                const auto& r = run_slots_[s];
                const std::uint64_t room = std::min<std::uint64_t>(r->job.job.max_tokens - r->stats.generated_tokens - 1,
                                                                   capacity_ - r->pos - 1);
                double surv = 1.0;
                for (std::uint32_t i = 0; i < ds::block && i < room; ++i) {
                    surv *= r->survival.probability(i, d.confidence[i]);
                    inc.push_back({surv, s, i});
                }
            }
            std::stable_sort(inc.begin(), inc.end(), [](const Inc& a, const Inc& b) { return a.gain > b.gain; });
            const std::size_t jmax = std::min<std::size_t>(inc.size(), budget);
            const unsigned shape = n == 1 ? 1U : 2U;
            std::size_t best = 0;
            double best_rate = static_cast<double>(n) / costs_.at(0, n), g = 0;
            for (std::size_t j = 1; j <= jmax; ++j) {
                g += inc[j - 1].gain;
                const double cost = costs_.draft * static_cast<double>(drafts.size()) + costs_.at(shape, n + static_cast<std::uint32_t>(j));
                const double rate = (static_cast<double>(n) + g) / cost;
                if (rate > best_rate) {
                    best_rate = rate;
                    best = j;
                }
            }
            if (++explore_[n] % 16 == 1) best = jmax;
            for (std::size_t j = 0; j < best; ++j) ++k[inc[j].slot];
        }
        // the step rows: every decoding slot's anchor, then its window's drafts
        Control c;
        c.command = static_cast<std::uint32_t>(Command::step);
        c.round = round_++;
        c.append = append;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> span_of;  // slot -> (first row, rows)
        for (auto s : order) {
            auto& r = run_slots_[s];
            const std::uint32_t first = c.rows;
            c.row[c.rows++] = mm::Row{s, r->last, r->pos};
            const std::uint32_t ks = k.count(s) ? k[s] : 0U;
            for (std::uint32_t i = 0; i < ks; ++i) c.row[c.rows++] = mm::Row{s, drafts[s].tokens[i], r->pos + 1 + i};
            span_of.push_back({first, c.rows - first});
        }
        const auto t0 = Clock::now();
        (void)exchange(eng_, c);
        const auto next = run_.step(std::span(c.row, c.rows), append);
        if (append != 0) ds::DeviceDrafter::device_synchronize();  // the context appends are part of this cycle's work
        const auto t_done = Clock::now();
        const double t_step = secs(t0, t_done);  // control exchange + step + appends, completed
        const bool windows = c.rows > n;
        costs_.observe(windows ? (n == 1 ? 1U : 2U) : 0U, c.rows, t_step);
        std::uint32_t emitted_total = 0, accepted_total = 0, drafted_total = 0, windows_n = 0, rejected_n = 0;
        std::uint32_t k_min = ~0U, k_max = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            const std::uint32_t s = order[i];
            auto& r = run_slots_[s];
            const auto [first, rows] = span_of[i];
            std::vector<std::uint32_t> d;
            for (std::uint32_t j = 1; j < rows; ++j) d.push_back(c.row[first + j].token);
            const auto v = ds::verify_greedy(d, std::span<const std::uint32_t>(next.data() + first, rows));
            if (drafted && drafts.count(s)) {
                const std::uint32_t ks = rows - 1;
                r->survival.observe(std::span<const float>(drafts[s].confidence.data(), ks), ks, v.accepted);
                r->x.drafted += ks;
                r->x.accepted += v.accepted;
                Costs::ema(r->tau, static_cast<double>(v.emitted.size()));
                drafted_total += ks;
                accepted_total += v.accepted;
                if (ks > 0) {
                    ++windows_n;
                    k_min = std::min(k_min, ks);
                    k_max = std::max(k_max, ks);
                    rejected_n += v.accepted < ks ? 1U : 0U;
                }
            }
            ++r->x.cycles;
            r->pos += v.accepted + 1;
            r->last = v.emitted.back();
            (void)emit(r, v.emitted, &emitted_total);  // the tokens actually emitted (a stop / limit drops the rest)
        }
        // the gate on one boundary for both kinds of cycle: from the cycle's start (before any draft) to its completed
        // step + appends; plain cycles as seconds per token of the n rows
        const double t_cycle = secs(c0, t_done);
        if (drafted) gates_[n].observe(true, emitted_total, t_cycle, costs_.at(0, n) / n);
        else if (!cand.empty() || budget > 0) gates_[n].observe(false, n, t_cycle / n, 0.0);
        add([&](Counters& k2) {
            k2.step_s += t_step;
            k2.draft_s += t_draft;
            ++k2.steps;
            k2.step_rows += c.rows;
            ++k2.rows_hist[c.rows];
            if (windows_n >= 2) ++k2.multi_windows;
            if (windows_n >= 2 && k_min != k_max) ++k2.unequal_windows;
            k2.rejections += rejected_n;
            if (drafted) {
                ++k2.draft_cycles;
                k2.drafts += drafts.size();
                k2.drafted += drafted_total;
                k2.accepted += accepted_total;
            } else {
                ++k2.plain_cycles;
            }
        });
    }

    mm::Engine& eng_;
    Runner& run_;
    sv::GenerationQueue& queue_;
    Shared& sh_;
    const std::uint32_t slots_;
    const std::uint64_t capacity_;
    const std::uint64_t max_rows_;
    std::vector<std::optional<Running>> run_slots_;
    std::uint64_t round_ = 0;
    std::uint32_t prefill_slot_ = 0;
    Clock::time_point prefill_start_;
    Policy pol_;
    Costs costs_;
    LayerModel layer_;
    bool taps_ = false;  // the engine's taps (set to the startup policy before READY, then sync_taps)
    std::vector<ds::DraftGate> gates_ = std::vector<ds::DraftGate>(kStepRows + 1);  // one per decoding count
    std::array<std::uint64_t, kStepRows + 1> explore_{};
};

void follow(mm::Engine& eng, Runner& run) {
    for (;;) {
        Control mine;
        const Control c = exchange(eng, mine);
        switch (static_cast<Command>(c.command)) {
        case Command::heartbeat:
            break;
        case Command::shutdown:
            if (eng.prefill_open()) run.prefill_cancel();
            return;
        case Command::prefill: {
            std::vector<std::uint32_t> prompt;
            exchange_tokens(eng, prompt, c.plen, false);
            run.prefill_begin(c.pslot, prompt, (c.pflags & 1U) != 0);
            break;
        }
        case Command::unit: {
            std::uint32_t next = 0;
            (void)run.prefill_unit(c.layers, &next);
            break;
        }
        case Command::cancel:
            run.prefill_cancel();
            break;
        case Command::taps:
            eng.set_taps((c.pflags & 1U) != 0);
            break;
        case Command::draft:
            if (run.dd == nullptr) throw std::runtime_error("control channel: draft command without a drafter on rank 1");
            for (std::uint32_t i = 0; i < c.dslots; ++i) (void)run.draft(c.dslot[i], c.danchor[i], c.dpos[i]);
            break;
        case Command::step:
            (void)run.step(std::span(c.row, c.rows), c.append);
            break;
        default:
            throw std::runtime_error("control channel: unknown command " + std::to_string(c.command));
        }
    }
}

// ---- both ranks, before READY: the resumable prefill vs prefill() with decode steps of other slots in between ----
// For each prompt file: reference = slots 1, 2 prefilled with the prompt's first 64 tokens, then S decode steps on them,
// then the prompt prefilled whole in slot 0; arm k = the same, but slot 0's prefill opened after the first step and
// resumed k layers at a time between the steps. Compared: slot 0's state FNV (every layer's K / V / index-key rows), the
// final rows' FNV, the next token, the decode tokens of slots 1 and 2, and (with a drafter) a draft from slot 0.
int resume_check(mm::Engine& eng, Runner& run, const std::string& files) {
    int bad = 0;
    std::size_t at = 0;
    while (at <= files.size()) {
        const auto comma = files.find(',', at);
        std::string f = files.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? files.size() + 1 : comma + 1;
        if (f.empty()) continue;
        std::vector<unsigned> ks{1, 7, 60};  // FILE[:k/k/...]
        if (const auto colon = f.rfind(':'); colon != std::string::npos) {
            ks.clear();
            for (std::size_t a = colon + 1; a < f.size();) {
                const auto slash = f.find('/', a);
                ks.push_back(static_cast<unsigned>(std::stoul(f.substr(a, slash == std::string::npos ? std::string::npos : slash - a))));
                a = slash == std::string::npos ? f.size() : slash + 1;
            }
            f.resize(colon);
        }
        std::ifstream in(f, std::ios::binary);
        std::vector<std::uint32_t> p;
        for (std::uint32_t t; in.read(reinterpret_cast<char*>(&t), 4);) p.push_back(t);
        if (p.size() < 80) throw std::invalid_argument("resume-check: " + f + " has fewer than 80 tokens");
        const std::vector<std::uint32_t> q(p.begin(), p.begin() + 64);
        constexpr int kSteps = 24;
        const bool ctx = run.dd != nullptr && p.size() <= 4096;
        struct Result {
            std::uint64_t state = 0, out = 0;
            std::uint32_t next = 0;
            std::vector<std::uint32_t> dec;
            std::array<std::uint32_t, ds::block> draft{};
            double wall = 0;
            unsigned units = 0;
        };
        const auto arm = [&](unsigned k) {
            Result r;
            std::array<std::uint32_t, 3> last{};
            for (std::uint32_t s = 1; s <= 2; ++s) {
                run.prefill_begin(s, q, false);
                (void)run.prefill_unit(~0U, &last[s]);
            }
            std::array<std::uint64_t, 3> pos{0, q.size(), q.size()};
            const auto step = [&] {
                const std::array<mm::Row, 2> rows{mm::Row{1, last[1], pos[1]}, mm::Row{2, last[2], pos[2]}};
                const auto g = run.step(rows, 0);
                for (std::uint32_t s = 1; s <= 2; ++s) {
                    last[s] = g[s - 1];
                    ++pos[s];
                    r.dec.push_back(g[s - 1]);
                }
            };
            const auto t0 = Clock::now();
            if (k == 0) {
                for (int i = 0; i < kSteps; ++i) step();
                run.prefill_begin(0, p, ctx);
                (void)run.prefill_unit(~0U, &r.next);
                r.out = eng.prefill_out_fnv();  // before any later call overwrites the final rows (they alias hn)
                r.units = 1;
            } else {
                step();
                run.prefill_begin(0, p, ctx);
                int steps = 1;
                for (bool done = false; !done;) {
                    done = run.prefill_unit(k, &r.next);
                    if (done) r.out = eng.prefill_out_fnv();
                    ++r.units;
                    if (steps < kSteps) {
                        step();
                        ++steps;
                    }
                }
                for (; steps < kSteps; ++steps) step();
            }
            r.wall = secs(t0, Clock::now());
            r.state = eng.state_fnv(0, p.size());
            if (ctx) r.draft = run.draft(0, r.next, p.size()).tokens;
            return r;
        };
        const Result ref = arm(0);
        for (const unsigned k : ks) {
            if (k == 0) throw std::invalid_argument("resume-check: k >= 1");
            const Result a = arm(k);
            const bool ok = a.state == ref.state && a.out == ref.out && a.next == ref.next && a.dec == ref.dec && a.draft == ref.draft;
            bad += ok ? 0 : 1;
            std::printf("rank %u resume-check %s (%zu tokens%s) k=%u units %u: state %016llx / %016llx out %016llx / %016llx next %u / %u "
                        "decode %s draft %s wall %.2f / %.2f s -> %s\n",
                        eng.rank(), f.c_str(), p.size(), ctx ? ", drafter context" : "", k, a.units, static_cast<unsigned long long>(a.state),
                        static_cast<unsigned long long>(ref.state), static_cast<unsigned long long>(a.out),
                        static_cast<unsigned long long>(ref.out), a.next, ref.next, a.dec == ref.dec ? "==" : "DIFFER",
                        ctx ? (a.draft == ref.draft ? "==" : "DIFFER") : "-", a.wall, ref.wall, ok ? "IDENTICAL" : "MISMATCH");
            std::fflush(stdout);
        }
    }
    // either rank's mismatch stops both (the peer's count over the control channel)
    return bad + static_cast<int>(eng.peer(static_cast<std::uint32_t>(bad)));
}

// ---- rank 0: the HTTP API's scheduler side (ms::MinimaxApi serves the routes) ----
sv::Json policy_json(Shared& shared, bool drafter) {
    std::lock_guard<std::mutex> l(shared.m);
    const Policy& p = shared.policy;
    const Counters& c = shared.counters;
    sv::Json b = sv::Json::object();
    b.set("sched", sv::Json(p.sched == Sched::interleave ? "interleave" : "off"));
    b.set("gap_ms", sv::Json(p.gap_ms));
    b.set("dspark", sv::Json(p.dspark && drafter));
    b.set("dspark_inflight", sv::Json(static_cast<std::int64_t>(p.inflight)));
    b.set("draft_slots", sv::Json(static_cast<std::int64_t>(p.draft_slots)));
    b.set("epoch", sv::Json(static_cast<std::int64_t>(p.epoch)));
    sv::Json k = sv::Json::object();
    k.set("step_s", sv::Json(c.step_s));
    k.set("prefill_s", sv::Json(c.prefill_s));
    k.set("draft_s", sv::Json(c.draft_s));
    k.set("idle_s", sv::Json(c.idle_s));
    k.set("stall_s", sv::Json(c.stall_s));
    k.set("steps", sv::Json(static_cast<std::int64_t>(c.steps)));
    k.set("step_rows", sv::Json(static_cast<std::int64_t>(c.step_rows)));
    k.set("units", sv::Json(static_cast<std::int64_t>(c.units)));
    k.set("prefills", sv::Json(static_cast<std::int64_t>(c.prefills)));
    k.set("cancels", sv::Json(static_cast<std::int64_t>(c.cancels)));
    k.set("multi_windows", sv::Json(static_cast<std::int64_t>(c.multi_windows)));
    k.set("unequal_windows", sv::Json(static_cast<std::int64_t>(c.unequal_windows)));
    k.set("rejections", sv::Json(static_cast<std::int64_t>(c.rejections)));
    k.set("draft_cycles", sv::Json(static_cast<std::int64_t>(c.draft_cycles)));
    k.set("plain_cycles", sv::Json(static_cast<std::int64_t>(c.plain_cycles)));
    k.set("drafts", sv::Json(static_cast<std::int64_t>(c.drafts)));
    k.set("drafted", sv::Json(static_cast<std::int64_t>(c.drafted)));
    k.set("accepted", sv::Json(static_cast<std::int64_t>(c.accepted)));
    sv::Json h = sv::Json::array();
    for (auto v : c.rows_hist) h.as_array().push_back(sv::Json(static_cast<std::int64_t>(v)));
    k.set("rows_hist", std::move(h));
    b.set("counters", std::move(k));
    return b;
}

// The mm_* value lines of GET /metrics: the queue's (from the snapshot the observer renders too) and the scheduler's
// counters (the observer adds their HELP/TYPE).
std::string mm_metric_lines(const sv::QueueSnapshot& s, Shared& shared) {
    Counters c;
    {
        std::lock_guard<std::mutex> l(shared.m);
        c = shared.counters;
    }
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "mm_queued %llu\nmm_running %llu\nmm_completed %llu\nmm_failed %llu\nmm_prompt_tokens %llu\n"
                  "mm_generated_tokens %llu\nmm_busy_seconds %.3f\nmm_step_seconds %.3f\nmm_prefill_seconds %.3f\n"
                  "mm_draft_seconds %.3f\nmm_idle_seconds %.3f\nmm_decode_stall_seconds %.3f\nmm_steps %llu\nmm_step_rows %llu\n"
                  "mm_prefill_units %llu\nmm_draft_cycles %llu\nmm_drafted %llu\nmm_accepted %llu\n",
                  static_cast<unsigned long long>(s.queued), static_cast<unsigned long long>(s.running),
                  static_cast<unsigned long long>(s.completed), static_cast<unsigned long long>(s.failed),
                  static_cast<unsigned long long>(s.prompt_tokens), static_cast<unsigned long long>(s.generated_tokens),
                  s.busy_seconds, c.step_s, c.prefill_s, c.draft_s, c.idle_s, c.stall_s, static_cast<unsigned long long>(c.steps),
                  static_cast<unsigned long long>(c.step_rows), static_cast<unsigned long long>(c.units),
                  static_cast<unsigned long long>(c.draft_cycles), static_cast<unsigned long long>(c.drafted),
                  static_cast<unsigned long long>(c.accepted));
    return buf;
}

// GET / POST /admin/sched (with --admin): the policy and counters; POST changes them (a bad value throws: a 400).
sv::Json admin_sched(const sv::HttpRequest& req, Shared& shared, bool drafter) {
    if (req.method == "POST") {
        const sv::Json body = sv::parse_json(req.body);
        std::lock_guard<std::mutex> l(shared.m);
        Policy& p = shared.policy;
        if (const auto* v = body.find("sched"); v && v->is_string()) {
            if (v->as_string() == "off") p.sched = Sched::off;
            else if (v->as_string() == "interleave") p.sched = Sched::interleave;
            else throw std::invalid_argument("sched: off | interleave");
        }
        if (const auto* v = body.find("gap_ms"); v && v->is_number()) p.gap_ms = std::max(1.0, v->as_double());
        if (const auto* v = body.find("dspark"); v && v->is_bool()) p.dspark = v->as_bool();
        if (const auto* v = body.find("dspark_inflight"); v && v->is_number()) p.inflight = static_cast<std::uint32_t>(std::max<std::int64_t>(0, v->as_int()));
        if (const auto* v = body.find("draft_slots"); v && v->is_number()) p.draft_slots = static_cast<std::uint32_t>(std::clamp<std::int64_t>(v->as_int(), 1, 8));
        if (body.find("reset_counters")) shared.counters = Counters{};
        ++p.epoch;
    }
    return policy_json(shared, drafter);
}

}  // namespace

int main(int argc, char** argv) {
    // Defaults-on resolver before anything reads a switch; --print-effective-config prints the resolved configuration.
    if (strix::minimax::defaults::print_or_apply("serve", argc, argv)) return 0;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR --rank R --transport T [--port P] [--bind ADDR] [--slots S] [--capacity C] "
                             "[--max-rows R] [--model-id ID] [--max-tokens N] [--dspark DIR [--dspark-exl3 DIR]] [--admin] "
                             "[--resume-check P.u32[,...]]\n", argv[0]);
        return 2;
    }
    try {
        (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "60000", 0);
        (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "60000", 0);
        // DS production host-path settings (CRC on parallel workers, CRC overlap, 1 MiB host chunks): transport
        // only, bits unchanged; measurement ABA x3 TTFT 1K -12.8 %, decode -1 %. An explicit value wins (=0 disables each).
        (void)::setenv("STRIX_DIST_CRC_PARALLEL", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CRC_OVERLAP", "1", 0);
        (void)::setenv("STRIX_DIST_HOST_CHUNK_BYTES", "1048576", 0);
        // deferred CRC ON by default (measurement: 7 processes inline/deferred alternated, 384-step 1K
        // sessions, tokens identical, each deferred arm below its two inline neighbours by 0.04-0.12 ms/token).
        // STRIX_MMX_CRC: 0 = off, 1 = inline, deferred = deferred check.
        (void)::setenv("STRIX_MMX_CRC", "deferred", 0);
        mm::Options opt;
        opt.model_dir = argv[1];
        opt.slots = 8;
        std::string transport_spec = "usb4-auto", bind = "127.0.0.1", model_id = "minimax-m3.1-flash", dspark_dir, dspark_exl3, check;
        std::uint16_t port = 8080;
        std::uint32_t max_tokens = 4096;
        bool admin = false, check_only = false;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--rank") opt.rank = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--transport") transport_spec = next();
            else if (a == "--port") port = static_cast<std::uint16_t>(std::stoul(next()));
            else if (a == "--bind") bind = next();
            else if (a == "--slots") opt.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--max-rows") opt.max_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--model-id") model_id = next();
            else if (a == "--max-tokens") max_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--dspark") dspark_dir = next();
            else if (a == "--dspark-exl3") dspark_exl3 = next();
            else if (a == "--admin") admin = true;
            else if (a == "--resume-check") check = next();
            else if (a == "--check-only") check_only = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        if (opt.slots == 0 || opt.slots > 8) throw std::invalid_argument("--slots 1..8");
        if (!dspark_exl3.empty() && dspark_dir.empty()) dspark_dir = opt.model_dir + "/dspark";
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::string receipt;
        auto transport = open_transport(transport_spec, opt.rank, receipt);
        strix::dist::Group group;
        group.minimax_compat = true;
        group.rank = opt.rank;
        group.world = 2;
        group.plan = 0x4d4d5331ULL;  // "MMS1"
        const std::size_t max_bytes = std::max<std::size_t>(std::size_t{opt.max_rows} * 6144, 256 * 782 * 128) * sizeof(float);
        // exchange module for the decode steps (default on, STRIX_MM_XFER=off disables)
        const char* xfer_env = std::getenv("STRIX_MM_XFER");
        std::unique_ptr<strix::minimax::xfer::Exchanger> exchanger;
        if (xfer_env == nullptr || std::string(xfer_env) != "off") {
            strix::minimax::xfer::Options xo;
            xo.rank = opt.rank;
            xo.capacity = strix::minimax::xfer::MmSchedule::row_bytes(8);
            xo.stream = nullptr;
            xo.from_env();
            exchanger = std::make_unique<strix::minimax::xfer::Exchanger>(std::make_unique<Borrowed>(*transport), xo);
        }
        strix::dist::Communicator comm(group, std::make_unique<Borrowed>(*transport), max_bytes);
        std::printf("connected: rank %u transport %s\n", opt.rank, comm.transport_name().c_str());
        std::fflush(stdout);
        mm::Engine eng(opt, comm);
        if (exchanger) eng.set_exchanger(exchanger.get(), true);
        const auto lr = eng.load();
        std::printf("rank %u: load %.1f s, device used %.1f GiB\n", opt.rank, lr.seconds, lr.device_gib);
        std::fflush(stdout);
        // DSpark drafter on both ranks (both run every draft command; the target's residual taps are replicated)
        std::unique_ptr<ds::DraftFile> draft_file;
        std::unique_ptr<ds::DeviceDrafter> drafter;
        Runner run{eng};
        if (!dspark_dir.empty()) {
            const auto t0 = Clock::now();
            draft_file = ds::DraftFile::open(dspark_dir, dspark_exl3.empty());
            ds::DeviceDrafter::Options dopt;
            dopt.slots = opt.slots;
            dopt.capacity = opt.capacity;
            if (const char* e = std::getenv("STRIX_MM_DSPARK_ATTN")) dopt.attn_variant = static_cast<std::uint32_t>(std::atoi(e));  // as the rank app
            drafter = std::make_unique<ds::DeviceDrafter>(draft_file->weights(), dopt);
            if (!dspark_exl3.empty()) eng.attach_draft_exl3(dspark_exl3, *drafter);
            eng.set_taps(true);  // for the drafter-context gate; then the startup policy below (both ranks)
            run.dd = drafter.get();
            std::tie(run.v0, run.vn) = eng.vocab_range();
            std::printf("rank %u: dspark %s%s loaded in %.1f s (%.2f GB device, rings %.0f MB, %u slots)\n", opt.rank, dspark_dir.c_str(),
                        dspark_exl3.empty() ? "" : (" + EXL3 " + dspark_exl3).c_str(), secs(t0, Clock::now()),
                        static_cast<double>(drafter->weight_bytes()) / 1e9, static_cast<double>(drafter->ring_bytes()) / 1e6, opt.slots);
            std::fflush(stdout);
        }
        if (!check.empty()) {
            const int bad = resume_check(eng, run, check);
            std::printf("rank %u resume-check: %s\n", opt.rank, bad == 0 ? "ALL IDENTICAL" : "MISMATCH");
            std::fflush(stdout);
            if (bad != 0) return 4;
            if (check_only) {  // structural gates only (e.g. a small --max-rows: groups then a single chunk), no serving
                std::printf("rank %u: check only, STOPPED\n", opt.rank);
                return 0;
            }
        }
        // taps only while DSpark is on (both ranks read the same STRIX_MM_SERVE_DSPARK; runtime switches via Command::taps)
        const bool dspark_on = drafter != nullptr && env_num("STRIX_MM_SERVE_DSPARK", 0) != 0;
        if (drafter != nullptr) eng.set_taps(dspark_on);
        if (opt.rank != 0) {
            follow(eng, run);
            std::printf("rank 1: STOPPED\n");
            return 0;
        }
        Shared shared;
        {
            const char* s = std::getenv("STRIX_MM_SERVE_SCHED");
            const std::string m = s ? s : "off";
            if (m == "interleave" || m == "1") shared.policy.sched = Sched::interleave;
            else if (m != "off" && m != "0") throw std::invalid_argument("STRIX_MM_SERVE_SCHED: off | interleave");
            shared.policy.gap_ms = env_num("STRIX_MM_SERVE_GAP_MS", 1500);
            shared.policy.dspark = dspark_on;
            shared.policy.inflight = static_cast<std::uint32_t>(env_num("STRIX_MM_SERVE_DSPARK_INFLIGHT", 1));
            shared.policy.draft_slots = static_cast<std::uint32_t>(std::clamp(env_num("STRIX_MM_SERVE_DRAFT_SLOTS", 8), 1.0, 8.0));
        }
        sv::GenerationQueue queue(64);
        ms::ObserverInfo oi;
        oi.model = model_id;
        oi.context_tokens = opt.capacity;
        oi.slots = opt.slots;
        oi.transport = comm.transport_name();
        oi.load_seconds = lr.seconds;
        oi.device_used_bytes = static_cast<std::uint64_t>(std::max(0.0, lr.device_gib) * 1073741824.0);
        oi.draft = drafter != nullptr;
        ms::Observer observer(std::move(oi));
        ms::MinimaxApiOptions ao;
        ao.model_id = model_id;
        ao.default_max_tokens = max_tokens;
        ao.context_tokens = opt.capacity;
        // the scheduler's side: its per-request extras (inserted by Lead::finish before the ticket finishes), its mm_*
        // lines and, with --admin, /admin/sched
        ao.take_extra = [&shared](const sv::GenerationTicket& t) -> std::optional<Extra> {
            std::lock_guard<std::mutex> l(shared.m);
            const auto it = shared.extras.find(&t);
            if (it == shared.extras.end()) return std::nullopt;
            Extra x = it->second;
            shared.extras.erase(it);
            return x;
        };
        ao.legacy_metrics = [&shared](const sv::QueueSnapshot& q) { return mm_metric_lines(q, shared); };
        if (admin) {
            const bool has_drafter = drafter != nullptr;
            ao.admin_sched = [&shared, has_drafter](const sv::HttpRequest& r) { return admin_sched(r, shared, has_drafter); };
        }
        ms::MinimaxApi api(ms::api_tokenizer(ms::MinimaxTokenizer::load(opt.model_dir + "/tokenizer.json")), queue, ao, observer);
        sv::HttpServerConfig hc;
        hc.bind_address = bind;
        hc.port = port;
        sv::HttpServer http(hc, [&](const sv::HttpRequest& r, sv::HttpResponseWriter& w) { api.handle(r, w); });
        http.listen();
        std::thread http_thread([&] { http.serve(); });
        std::printf("READY rank 0 http://%s:%u model %s slots %u context %llu sched %s gap %.0f ms dspark %s\n", bind.c_str(), http.port(),
                    model_id.c_str(), opt.slots, static_cast<unsigned long long>(opt.capacity),
                    shared.policy.sched == Sched::interleave ? "interleave" : "off", shared.policy.gap_ms, shared.policy.dspark ? "on" : "off");
        std::fflush(stdout);
        Lead lead(eng, run, queue, shared, opt.max_rows, dspark_on);
        lead.loop();
        queue.shutdown();
        http.stop();
        http_thread.join();
        std::printf("rank 0: STOPPED\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-minimax-serve error: %s\n", e.what());
        return 2;
    }
}
