// strix-glm5-serve host-only tests: control block encode/decode, hello / row checks, finish decisions, the slot table,
// and the lead + follower protocol over an in-process loopback with a deterministic fake engine (batched streams must
// equal their alone runs; serial mode; world 1; context limit; a cross-rank token mismatch aborts both ranks).
#include "strix/session/prefill_session.hpp"
#include "glm/serve/glm_serve_control.hpp"
#include "glm/serve/glm_serve_lead.hpp"
#include "glm/serve/glm_serve_import.hpp"
#include "glm/serve/glm_serve_selftest.hpp"
#include "glm/serve/glm_serve_vision.hpp"
#include "glm/rtxpf/import/glm_import_socket.hpp"

#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <atomic>
#include <functional>
#include <map>
#include <optional>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace gs = strix::glm::serve;
namespace sv = strix::serve;

namespace {

int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

template<class F>
bool throws_with(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Deterministic per-slot sequence model: a slot's tokens depend only on its own history (slot id excluded), as the
// engine's batch invariance promises. Positions are checked against the slot's frontier. The fake MTP path emits the
// same chain 1..3 tokens per verify cycle, with the real adapter's rules: a decoder opens only from a fresh seed (MTP
// prefill or the last step's row, checked against that step), no other engine call while it is open. Every call is
// logged so both ranks' call sequences can be compared.
// In-process pair exchange for the 'DSA once' relay: each side posts its block and waits for the other's (lock-step,
// as Communicator::exchange_host); remote empty = not wanted.
struct Pipe {
    std::mutex m;
    std::condition_variable cv;
    bool corrupt_dsa = false;  // one payload chunk, after successful prepare/header exchanges
    std::deque<std::vector<std::byte>> q[2];  // q[r]: blocks sent by rank r, not yet taken
    void xchg(unsigned rank, std::span<const std::byte> local, std::span<std::byte> remote) {
        std::unique_lock<std::mutex> l(m);
        q[rank].emplace_back(local.begin(), local.end());
        if (corrupt_dsa && rank == 0 && local.size() == 64) {
            q[rank].back()[0] ^= std::byte{1};
            corrupt_dsa = false;
        }
        cv.notify_all();
        if (!cv.wait_for(l, std::chrono::seconds(10), [&] { return !q[1 - rank].empty(); }))
            throw std::runtime_error("pipe: the peer never exchanged");
        auto b = std::move(q[1 - rank].front());
        q[1 - rank].pop_front();
        if (b.size() != local.size()) throw std::runtime_error("pipe: block sizes differ");
        if (!remote.empty()) std::copy(b.begin(), b.end(), remote.begin());
    }
};
// A synthetic per-rank layout (KDA, DSA, KDA, DSA, DSA) and its payloads: KDA bytes differ per rank, DSA bytes are the
// same for both (as the RTX producer's).
const std::vector<gs::StateComponent> kRelayLayout = {{false, 40}, {true, 100}, {false, 24}, {true, 300}, {true, 7}};
std::vector<std::uint8_t> relay_part(bool dsa, unsigned rank) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(gs::component_bytes(kRelayLayout, dsa)));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(dsa ? 0x80U + i * 7U : rank * 16U + i * 3U);
    return v;
}
std::vector<std::uint8_t> relay_full(unsigned rank) {
    return gs::merge_dsa(relay_part(false, rank), relay_part(true, 0), kRelayLayout);
}

std::uint32_t fake_frame_crc(std::span<const std::byte> bytes) noexcept {
    std::uint32_t h = 2166136261U;
    for (const std::byte b : bytes) h = (h ^ static_cast<std::uint32_t>(b)) * 16777619U;
    return h;
}

struct FakeEng {
    std::uint32_t slots;
    // Image input, with the real adapter's PromptEmbeds: this rank's table and frames. A provided row enters the
    // slot's chain as a hash of its data instead of its token, so the ranks' tokens agree only when both hold the
    // same rows at the same positions.
    gs::PromptEmbeds embeds{&fake_frame_crc};
    std::uint64_t embed_rows = 0, embed_frames = 0, embed_frame_bytes = 0;
    std::vector<std::uint32_t> effective;
    void bind_embeds(std::shared_ptr<const strix::session::EmbedTable> table) { embeds.bind(std::move(table)); }
    std::vector<std::byte> embed_frame(std::span<const std::uint32_t> ids) {
        auto frame = embeds.frame(ids);
        ++embed_frames;
        embed_frame_bytes += frame.size();
        return frame;
    }
    void embed_frame_received(std::span<const std::byte> frame, std::vector<std::uint32_t>& ids) {
        embeds.received(frame, ids);
        ++embed_frames;
        embed_frame_bytes += frame.size();
    }
    std::span<const std::uint32_t> provide(std::span<const std::uint32_t> t) {
        if (!gs::PromptEmbeds::holds_virtual_ids(t)) return t;
        const auto run = embeds.resolve(t);
        effective = run.tokens;
        for (const auto& seg : run.splice.segments)
            for (std::uint32_t j = 0; j < seg.rows; ++j) {
                const auto* row = seg.data + std::size_t{j} * run.splice.width;
                effective[seg.run_row + j] = 0x40000000U | (fake_frame_crc(std::as_bytes(std::span(row, run.splice.width))) >> 2U);
                ++embed_rows;
            }
        return effective;
    }
    bool stops = true;
    std::uint32_t corrupt_call = 0;  // > 0: the n-th step returns a different first token
    unsigned delay_us = 0;           // per step / verify cycle
    std::vector<std::uint64_t> h, frontier;
    std::vector<bool> ready;
    std::uint64_t calls = 0, steps = 0, cycles = 0, prefills = 0;
    std::vector<std::size_t> step_rows;
    std::vector<std::string> log;
    struct Seed { std::uint32_t slot, y0; std::uint64_t p; };
    std::optional<Seed> seed;
    struct Dec { std::uint32_t slot, last, budget, emitted; };
    std::optional<Dec> dec;
    std::vector<gs::WireRow> last_rows;
    std::vector<std::uint32_t> last_out;
    // batched self-speculation: per slot drafter state + the slot's anchor (its last emitted token)
    std::vector<gs::SlotSpec> spec;
    std::vector<std::uint32_t> anchor;
    std::uint64_t spec_verifies = 0, spec_drafts = 0, multi_slot_verifies = 0;
    explicit FakeEng(std::uint32_t s) : slots(s), h(s, 0), frontier(s, 0), ready(s, false), spec(s), anchor(s, 0) {}
    // the fake drafter: the true next token after the chain state x, wrong for a quarter of the states
    std::pair<std::uint32_t, float> guess(std::uint64_t x) const {
        const std::uint32_t t = tok(x);
        const bool ok = ((x >> 17) % 4U) != 0;
        return {ok ? t : t ^ 0x2BU, ok ? 0.85F : 0.35F};
    }
    std::uint64_t chain(std::uint32_t s) const {  // state after the anchor and the slot's drafts
        std::uint64_t x = fold(h[s], anchor[s]);
        for (const auto d : spec[s].drafts) x = fold(x, d);
        return x;
    }
    void seed_spec(std::uint32_t s) {
        spec[s] = {};
        spec[s].seeded = true;
        spec[s].p = frontier[s];
        const auto g = guess(fold(h[s], anchor[s]));
        spec[s].drafts.push_back(g.first);
        spec[s].conf.push_back(g.second);
    }
    [[nodiscard]] const gs::SlotSpec& spec_state(std::uint32_t s) const { return spec.at(s); }
    std::uint32_t spec_draft(std::uint32_t s) {
        plain_call();
        if (!spec[s].seeded || spec[s].drafts.empty()) throw std::runtime_error("fake spec_draft: no chain");
        const auto g = guess(chain(s));
        spec[s].drafts.push_back(g.first);
        spec[s].conf.push_back(g.second);
        ++spec_drafts;
        log.push_back("draft " + std::to_string(s));
        return g.first;
    }
    std::vector<std::uint32_t> spec_verify(std::span<const gs::WireRow> rows, std::span<const std::uint32_t> budgets) {
        plain_call();
        const auto win = gs::windows_of(rows, slots, 1U << 30);
        if (budgets.size() != win.size()) throw std::runtime_error("fake spec_verify: budgets");
        std::vector<std::uint32_t> out;
        std::string l = "verify";
        for (std::size_t i = 0; i < win.size(); ++i) {
            const auto& w = win[i];
            const auto s = w.slot;
            if (rows[w.row0].pos != frontier[s] || rows[w.row0].token != anchor[s]) throw std::runtime_error("fake spec_verify: anchor");
            std::vector<std::uint32_t> drafts, target;
            for (std::uint32_t j = 1; j < w.n; ++j) {
                drafts.push_back(rows[w.row0 + j].token);
                if (j > spec[s].drafts.size() || spec[s].drafts[j - 1] != drafts.back()) throw std::runtime_error("fake spec_verify: drafts");
            }
            std::uint64_t x = h[s];
            for (std::uint32_t j = 0; j < w.n; ++j) {
                x = fold(x, rows[w.row0 + j].token);
                target.push_back(tok(x));
            }
            const auto acc = strix::glm::spec::accept_greedy(drafts, target, strix::glm::spec::kNoFlag,
                                                             strix::glm::spec::Terminal{strix::glm::spec::kNoFlag, budgets[i]});
            for (unsigned j = 0; j < acc.rows; ++j) (void)row(s, rows[w.row0 + j].token);
            std::vector<std::uint32_t> emitted(drafts.begin(), drafts.begin() + acc.m);
            if (acc.has_bonus) emitted.push_back(acc.bonus);
            anchor[s] = emitted.back();
            out.push_back(static_cast<std::uint32_t>(emitted.size()));
            out.insert(out.end(), emitted.begin(), emitted.end());
            if (acc.stop) spec[s] = {};
            else seed_spec(s);
            l += " " + std::to_string(s) + "@" + std::to_string(rows[w.row0].pos) + "+" + std::to_string(w.n);
        }
        ++spec_verifies;
        if (win.size() > 1) ++multi_slot_verifies;
        log.push_back(l);
        wait();
        return out;
    }
    std::uint32_t tok(std::uint64_t x) const {
        if (stops && x % 53 == 7) return 154820U;
        return static_cast<std::uint32_t>((x >> 11) % 150000U) + 1U;
    }
    static std::uint64_t fold(std::uint64_t x, std::uint32_t t) { return (x ^ t) * 1099511628211ULL + 0x9e3779b97f4a7c15ULL; }
    void plain_call() {
        if (dec) throw std::runtime_error("fake: engine call while an MTP decoder is open");
        seed.reset();
        last_rows.clear();
        last_out.clear();
    }
    std::function<void()> on_first_decode;
    void wait() {
        if (auto hook = std::exchange(on_first_decode, {})) hook();
        if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
    // one trunk row of `slot` at its frontier
    std::uint32_t row(std::uint32_t s, std::uint32_t token) {
        h[s] = fold(h[s], token);
        ++frontier[s];
        return tok(h[s]);
    }
    bool refuse_prefix = false, fail_prefix_restore = false;
    std::uint64_t snapshot_bound = sizeof(std::uint64_t), snapshot_copies = 0;
    std::uint64_t prefix_state_bound(std::uint64_t) const { return snapshot_bound; }
    bool prefix_store_allowed(std::uint64_t) const { return !refuse_prefix; }
    std::vector<std::uint8_t> state_bytes_upto(std::uint32_t s, std::uint64_t p) {
        ++snapshot_copies;
        if (frontier.at(s) != p) throw std::runtime_error("fake snapshot frontier");
        std::vector<std::uint8_t> bytes(sizeof(std::uint64_t));
        std::memcpy(bytes.data(), &h.at(s), bytes.size());
        return bytes;
    }
    void restore_state(std::uint32_t s, const std::vector<std::uint8_t>& bytes, std::uint64_t p) {
        if (fail_prefix_restore) throw std::runtime_error("fake prefix restore failure");
        if (!ready.at(s) || frontier[s] != 0 || bytes.size() != sizeof(std::uint64_t)) throw std::runtime_error("fake prefix restore");
        std::memcpy(&h.at(s), bytes.data(), bytes.size());
        frontier[s] = p;
        spec[s] = {};
        log.push_back("restore " + std::to_string(s) + " @" + std::to_string(p));
    }
    void reset(std::uint32_t s) {
        plain_call();
        spec.at(s) = {};
        h.at(s) = 14695981039346656037ULL;
        frontier.at(s) = 0;
        ready.at(s) = true;
        ++calls;
        log.push_back("reset " + std::to_string(s));
    }
    std::uint32_t prefill(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0) {
        plain_call();
        if (units_open) throw std::runtime_error("fake prefill: another prompt's prefill units session is open");
        if (!ready.at(s) || pos0 != frontier[s]) throw std::runtime_error("fake prefill: not at the slot frontier / without reset");
        t = provide(t);
        for (const auto x : t) h[s] = fold(h[s], x);
        frontier[s] = pos0 + t.size();
        ++calls;
        ++prefills;
        log.push_back("prefill " + std::to_string(s) + " @" + std::to_string(pos0) + " +" + std::to_string(t.size()));
        spec[s] = {};
        anchor[s] = tok(h[s]);
        if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us * (t.size() / 32 + 1)));
        return tok(h[s]);
    }
    std::vector<std::uint32_t> step(std::span<const gs::WireRow> rows) {
        plain_call();
        std::vector<std::uint32_t> out;
        ++steps;
        step_rows.push_back(rows.size());
        std::string l = "step";
        for (const auto& r : rows) {
            if (r.pos != frontier.at(r.slot)) throw std::runtime_error("fake step: position not at the slot frontier");
            out.push_back(row(r.slot, r.token));
            spec[r.slot] = {};
            l += " " + std::to_string(r.slot) + "@" + std::to_string(r.pos);
        }
        if (corrupt_call != 0 && steps == corrupt_call) out[0] ^= 1U;
        for (std::size_t i = 0; i < rows.size(); ++i) anchor[rows[i].slot] = out[i];
        ++calls;
        log.push_back(l);
        last_rows.assign(rows.begin(), rows.end());
        last_out = out;
        wait();
        return out;
    }
    // import: a staged payload = the state of a prefix (hash + frontier), as Engine::restore_state would write it.
    struct Staged { std::uint64_t h, p; };
    std::map<std::uint32_t, Staged> staged;
    void stage(std::uint32_t id, std::span<const std::uint32_t> prefix) {
        std::uint64_t x = 14695981039346656037ULL;
        for (const auto t : prefix) x = fold(x, t);
        staged[id] = Staged{x, prefix.size()};
    }
    // import_expect: the fake "receives" at once (the prefix state of the announced prompt); fail_expect makes it fail.
    std::uint32_t expected_id = 0, expected_status = 0;
    bool fail_expect = false, fail_restore = false;
    std::optional<gs::ImportReport> import_done;
    std::string abort_why;
    void import_expect(std::uint32_t id, std::span<const std::uint32_t> t) {
        log.push_back("expect " + std::to_string(id) + " " + std::to_string(t.size()));
        expected_id = id;
        import_done.reset();
        if (fail_expect) {
            expected_status = static_cast<std::uint32_t>(gs::ImportStatus::failed);
            return;
        }
        stage(id, t.first(t.size() - 1));
        expected_status = static_cast<std::uint32_t>(gs::ImportStatus::ready);
    }
    const std::atomic<bool>* hold = nullptr;  // while true, a staged import reports listening (the RTX still sending)
    gs::ImportReport import_report() const {
        if (import_done && import_done->id == expected_id) return *import_done;
        if (hold && hold->load() && expected_status == static_cast<std::uint32_t>(gs::ImportStatus::ready))
            return {expected_id, static_cast<std::uint32_t>(gs::ImportStatus::listening), 0};
        return {expected_id, expected_status, 0};
    }
    // state_digest (import gate): the slot's state words at a frontier (no state change)
    std::vector<std::uint32_t> state_digest(std::uint32_t s, std::uint64_t f) const {
        if (frontier.at(s) != f) throw std::runtime_error("fake state_digest: not at the slot frontier");
        return {static_cast<std::uint32_t>(h[s]), static_cast<std::uint32_t>(h[s] >> 32), static_cast<std::uint32_t>(f), 0U};
    }
    // import_abort (replay, after import_state threw): the slot reset, the failure reported.
    void import_abort(std::uint32_t s, std::uint32_t id, const std::string& why) {
        abort_why = why;
        seed.reset();
        reset(s);
        log.back() += " (import abort)";
        import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::failed), 0};
    }
    // 'DSA once': both fake ranks share `pipe`; rank 0 stages the full synthetic payload, rank 1 its KDA part only.
    Pipe* pipe = nullptr;
    unsigned rank = 0;
    bool relay_fail = false;  // this rank cannot take part in the relay (nothing usable staged)
    gs::RelayStats last_relay{};
    std::vector<std::uint8_t> relayed;  // rank 1: the merged payload it restored
    std::uint8_t binding_salt = 0;  // != 0: this rank staged a hand-off with another binding (both must abort)
    unsigned suffix_collectives = 0;  // pair exchanges made by the suffix (the engine's cross-rank argmax, modelled)
    // As EngineOps: gs::import_transaction (prepare vote with the binding, relay, restore, restored vote, then the
    // suffix, which pairs up with the peer like the engine's argmax collective). Votes ride `pipe` (none: one rank).
    std::uint32_t import_state(std::uint32_t s, std::uint32_t id, std::span<const std::uint32_t> t, bool mtp, bool dsa) {
        plain_call();
        bool prepared = true;
        std::string why;
        gs::Binding b{};
        const auto it = staged.find(id);
        try {
            if (!ready.at(s) || frontier[s] != 0) throw std::runtime_error("fake import without reset");
            if (it == staged.end()) throw std::runtime_error("fake import: nothing staged under this id");
            if (it->second.p + 1 != t.size()) throw std::runtime_error("fake import: staged frontier != prompt - 1");
            for (std::size_t k = 0; k < 8; ++k) b[k] = static_cast<std::uint8_t>(it->second.h >> (8 * k));
            b[31] = binding_salt;
        } catch (const std::exception& e) {
            prepared = false;
            why = e.what();
        }
        std::function<void(std::span<const std::byte>, std::span<std::byte>, std::uint64_t)> xchg;
        if (pipe) xchg = [this](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe->xchg(rank, a, r); };
        return gs::import_transaction(
            prepared, why, b, t.size() >= 1 ? t.size() - 1 : 0, xchg,
            [&] {
                if (!dsa) return;
                if (!pipe) throw std::runtime_error("fake relay: no pipe");
                const auto mine = rank == 0 ? relay_full(0) : relay_part(false, 1);
                relayed = gs::dsa_relay(rank, !relay_fail, mine, kRelayLayout, 64, xchg, &last_relay);
                if (rank == 1 && relayed != relay_full(1)) throw std::runtime_error("fake relay: rank 1 payload differs from its full state");
            },
            [&] {
                if (fail_restore) {  // a half-written slot, as a restore_state that throws midway
                    h[s] = 12345;
                    frontier[s] = 7;
                    throw std::runtime_error("fake restore failure");
                }
                h[s] = fold(it->second.h, t.back());
                frontier[s] = t.size();
            },
            [&]() -> std::uint32_t {
                if (mtp && units_open) throw std::runtime_error("fake import suffix: prefill session still open");
                if (pipe) {  // the engine's argmax over both ranks: a rank alone here would wait for its peer forever
                    const std::uint32_t mine = tok(h[s]);
                    std::uint32_t peer = 0;
                    pipe->xchg(rank, std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
                    ++suffix_collectives;
                    if (peer != mine) throw std::runtime_error("fake suffix: the ranks' first tokens differ");
                }
                ++calls;
                log.push_back("import " + std::to_string(s) + " " + std::to_string(t.size()) + (mtp ? " mtp" : "") + (dsa ? " dsa" : ""));
                spec[s] = {};
                anchor[s] = tok(h[s]);
                if (mtp) {
                    seed = Seed{s, tok(h[s]), t.size()};
                    seed_spec(s);  // batched self-speculation starts from the import's seed, as after the slot's own prefill
                }
                import_done = gs::ImportReport{id, static_cast<std::uint32_t>(gs::ImportStatus::restored), tok(h[s])};
                return tok(h[s]);
            });
    }
    std::vector<std::uint32_t> dropped;  // import_drop ids (fallback / cancel)
    void import_drop(std::uint32_t id) {
        dropped.push_back(id);
        staged.erase(id);
        if (expected_id == id) expected_status = static_cast<std::uint32_t>(gs::ImportStatus::failed);
    }
    std::uint32_t prefill_mtp(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0, bool last) {
        if (units_open) throw std::runtime_error("fake prefill: prefill session still open");
        const auto y0 = prefill(s, t, pos0);
        log.back() += last ? " mtp last" : " mtp";
        if (last) {
            seed = Seed{s, y0, pos0 + t.size()};
            seed_spec(s);
        }
        return y0;
    }
    // Prefill units (the engine's prefill_mtp session): the whole-prompt call runs when the last unit arrives, so the
    // tokens equal the idle run by construction; while a session is open no other prompt may prefill (the engine's
    // row buffers hold the paused piece).
    std::uint32_t unit_rows = 0, unit_layers = 0;
    std::vector<std::uint32_t> units_tokens, unit_calls;
    std::uint32_t units_slot = 0;
    std::uint64_t units_pos = 0, units_done = 0, units_total = 0;
    bool units_open = false;
    unsigned units_cancelled = 0, units_delay_us = 0;
    std::function<void()> units_advanced;
    std::uint32_t prefill_units(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0, bool start, std::uint32_t n) {
        if (n == 0) {
            if (!units_open) throw std::runtime_error("fake units: cancel without a session");
            units_open = false;
            ++units_cancelled;
            log.push_back("units cancel " + std::to_string(s));
            return gs::kPrefillPending;
        }
        if (start) {
            if (units_open) throw std::runtime_error("fake units: a session is already open");
            units_open = true;
            t = provide(t);  // as the real adapter: resolved once, at the start of the session
            units_tokens.assign(t.begin(), t.end());
            units_slot = s;
            units_pos = pos0;
            units_done = 0;
            units_total = strix::session::LayerUnits{t.size(), unit_rows, unit_layers}.total();
            if (units_total == 0) throw std::runtime_error("fake units: no unit geometry");
        } else if (!units_open || s != units_slot || pos0 != units_pos) {
            throw std::runtime_error("fake units: resume without its session");
        }
        unit_calls.push_back(n);
        log.push_back("units " + std::to_string(s) + " +" + std::to_string(n));
        if (units_delay_us) std::this_thread::sleep_for(std::chrono::microseconds(units_delay_us));
        units_done = std::min<std::uint64_t>(units_done + n, units_total);
        if (units_advanced) units_advanced();
        if (units_done < units_total) return gs::kPrefillPending;
        units_open = false;
        return prefill_mtp(units_slot, units_tokens, units_pos, true);
    }
    void mtp_enter(std::uint32_t r, const gs::WireRow& w, bool from_seed) {
        if (from_seed) {
            const auto& st = spec.at(w.slot);
            if (!st.seeded || st.drafts.size() != 1 || st.p != w.pos || anchor[w.slot] != w.token || frontier[w.slot] != w.pos)
                throw std::runtime_error("fake mtp_enter: no fresh batched seed");
            plain_call();
            seed = Seed{w.slot, w.token, w.pos};
            spec[w.slot] = {};
            log.push_back("enter-seed " + std::to_string(w.slot) + "@" + std::to_string(w.pos));
            return;
        }
        if (r >= last_rows.size() || last_rows[r].slot != w.slot || last_rows[r].pos != w.pos || last_out[r] != w.token)
            throw std::runtime_error("fake mtp_enter: not the last step's row");
        plain_call();
        if (frontier[w.slot] != w.pos + 1) throw std::runtime_error("fake mtp_enter: frontier");
        seed = Seed{w.slot, w.token, w.pos + 1};
        log.push_back("enter " + std::to_string(w.slot) + "@" + std::to_string(w.pos));
    }
    std::vector<std::uint32_t> mtp_run(std::uint32_t s, std::uint32_t budget, std::uint32_t n) {
        if (!dec) {
            if (!seed || seed->slot != s || frontier[s] != seed->p) throw std::runtime_error("fake mtp_run without a fresh seed");
            dec = Dec{s, seed->y0, budget, 1};
            seed.reset();
            spec[s] = {};
        } else if (dec->slot != s) {
            throw std::runtime_error("fake mtp_run: other slot");
        }
        std::vector<std::uint32_t> out;
        for (std::uint32_t c = 0; c < n && dec->emitted < dec->budget; ++c) {
            const unsigned k = 1U + static_cast<unsigned>(h[s] % 3U);
            for (unsigned i = 0; i < k && dec->emitted < dec->budget; ++i) {
                dec->last = row(s, dec->last);
                out.push_back(dec->last);
                ++dec->emitted;
            }
            ++cycles;
            wait();
        }
        anchor[s] = dec->last;
        log.push_back("run " + std::to_string(s) + " -> " + std::to_string(out.size()));
        return out;
    }
    void mtp_leave(std::uint32_t s) {
        if (dec && dec->slot != s) throw std::runtime_error("fake mtp_leave: other slot");
        dec.reset();
        seed.reset();
        spec.at(s) = {};
        log.push_back("leave " + std::to_string(s));
    }
};

// In-process wire: one message queue per direction; exchange = send, then receive (5 s timeout).
class Loopback {
public:
    gs::Channel::Wire side(int s) {
        return [this, s](std::span<const std::byte> out, std::span<std::byte> in, std::uint32_t, std::uint64_t) {
            {
                std::lock_guard<std::mutex> l(m_);
                q_[s].emplace_back(out.begin(), out.end());
                cv_.notify_all();
            }
            std::unique_lock<std::mutex> l(m_);
            if (!cv_.wait_for(l, std::chrono::seconds(5), [&] { return !q_[1 - s].empty(); }))
                throw std::runtime_error("loopback: peer timeout");
            auto msg = std::move(q_[1 - s].front());
            q_[1 - s].pop_front();
            if (msg.size() != in.size()) throw std::runtime_error("loopback: size mismatch");
            std::copy(msg.begin(), msg.end(), in.begin());
        };
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::vector<std::byte>> q_[2];
};

std::vector<std::vector<std::uint32_t>> prompts(unsigned n) {
    std::vector<std::vector<std::uint32_t>> p;
    for (unsigned i = 0; i < n; ++i) {
        std::vector<std::uint32_t> t;
        for (unsigned k = 0; k < 5 + 7 * i; ++k) t.push_back(1000U + 13U * i + k);
        p.push_back(std::move(t));
    }
    return p;
}

gs::Hello hello_for(std::uint32_t slots, std::uint64_t cap, std::uint32_t world) {
    gs::Hello h;
    h.model_hash = 0x1234;
    h.identity_hash = 0x5678;
    h.capacity = cap;
    h.slots = slots;
    h.max_rows = 4096;
    h.step_rows = gs::kMaxRows;
    h.world = world;
    return h;
}

struct PairRun {
    gs::SelftestResult result;
    std::string lead_error, follow_error, driver_error;
    gs::LeadStats stats;
    std::uint64_t follower_snapshot_copies = 0;
    std::vector<std::size_t> rows0, rows1;
    std::vector<std::string> log0, log1;
};

PairRun run_pair(unsigned n, std::uint32_t slots, bool batch, std::uint32_t tokens, std::uint32_t corrupt = 0,
                 gs::Hello h1 = hello_for(8, 4096, 2), bool mtp = false, unsigned chunk = 0, unsigned long_n = 0,
                 unsigned delay_us = 0, bool skip_basic = false, bool g2 = false, std::uint64_t* multi = nullptr,
                 std::uint64_t import_min = 0, bool fail_import = false, bool fail_restore = false, std::uint8_t binding_salt = 0, bool prefix = false, bool refuse_prefix = false, std::uint64_t prefix_budget = 1ULL << 20, bool reply_keep = false, bool refuse_reply = false, bool oversized = false) {
    PairRun out;
    Loopback lb;
    FakeEng e0(slots), e1(slots);
    Pipe pipe;  // the import commit's votes and the suffix collective pair up through it
    e0.pipe = e1.pipe = &pipe;
    e1.rank = 1;
    e1.corrupt_call = corrupt;
    e0.delay_us = e1.delay_us = delay_us;
    e1.fail_expect = fail_import;  // the follower's staging fails -> every import falls back to its own prefill
    e1.fail_restore = fail_restore;  // the follower's restore fails after both staged -> reset + fallback, still exact
    e1.binding_salt = binding_salt;  // the follower staged another hand-off -> the prepare vote aborts both
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    gs::Hello h0 = hello_for(slots, 4096, 2);
    if (prefix) h0.prefix_cache_bytes = h1.prefix_cache_bytes = prefix_budget;
    e1.refuse_prefix = refuse_prefix;
    if (oversized) e1.snapshot_bound = prefix_budget + 1;
    if (h1.slots == 8 && slots != 8) h1.slots = slots;
    std::thread follower([&] {
        try {
            gs::follow(e1, c1, h1, slots, 4096);
        } catch (const std::exception& e) {
            out.follow_error = e.what();
        }
    });
    std::atomic<bool> stop{false};
    sv::GenerationQueue queue(64);
    gs::LeadConfig lc;
    lc.prefix_cache_bytes = h0.prefix_cache_bytes;
    lc.prefix_reply_store = reply_keep && !refuse_reply;
    lc.slots = slots;
    lc.capacity = 4096;
    lc.batch = batch;
    lc.heartbeat_seconds = 0.01;
    lc.mtp = mtp;
    lc.mtp_cycles = 2;
    lc.prefill_chunk = chunk;
    // These levels assert the base interleave (one chunk per decode round); the grouped form has its own case below.
    lc.interleave.select = sv::PrefillInterleave::Select::off;
    lc.g2 = g2;
    if (import_min != 0) {  // route on: the fake ranks stage at import_expect, the 'RTX' always accepts
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = import_min;
        lc.import_timeout_s = 5.0;
    }
    try {
        c0.hello(h0);
        gs::Lead<FakeEng> lead(e0, c0, queue, lc, stop);
        lead.set_trace(true);
        std::thread driver([&] {
            try {
                gs::SelftestHooks hooks;
                if (mtp) hooks.set_mtp = [&lead](bool on) { lead.set_mtp(on); };
                hooks.take_trace = [&lead] { return lead.take_trace(); };
                hooks.set_prefill_chunk = [&lead](unsigned c) { lead.set_prefill_chunk(c); };
                hooks.progress = [&lead] { return lead.progress(); };
                hooks.prefill_chunk = chunk;
                hooks.extra_chunks = {64U};
                hooks.skip_basic = skip_basic;
                if (g2) hooks.set_g2 = [&lead](bool on) { lead.set_g2(on); };
                for (unsigned i = 0; i < long_n; ++i) {
                    std::vector<std::uint32_t> t;
                    for (unsigned k = 0; k < 1000 + 37 * i; ++k) t.push_back(2000U + 7U * i + k % 911U);
                    hooks.long_prompts.push_back(std::move(t));
                }
                if (prefix) out.result.exact = gs::run_prefix_selftest(queue, prompts(n), tokens, gs::default_stop_ids(), {17, 23},
                    [&lead] { lead.clear_prefix_cache(); }, [&lead] { return lead.stats().prefix_hits; },
                    [](const std::string& line) { std::printf("%s\n", line.c_str()); }, reply_keep && !mtp);
                else out.result = gs::run_selftest(queue, prompts(n), tokens, gs::default_stop_ids(), slots, [](const std::string& l) {
                    std::printf("  %s\n", l.c_str());
                }, hooks);
            } catch (const std::exception& e) {
                out.driver_error = e.what();
            }
            stop.store(true);
        });
        try {
            lead.loop();
        } catch (const std::exception& e) {
            out.lead_error = e.what();
            queue.shutdown();
        }
        driver.join();
        out.stats = lead.stats();
    } catch (const std::exception& e) {
        out.lead_error = e.what();
    }
    follower.join();
    out.rows0 = e0.step_rows;
    out.rows1 = e1.step_rows;
    out.log0 = e0.log;
    out.log1 = e1.log;
    out.follower_snapshot_copies = e1.snapshot_copies;
    if (multi) *multi = e0.multi_slot_verifies;
    return out;
}

void queued_successor_interleave() {
    // Final grouped chunks must pay their debt before a queued successor prefills (also when slots are reused).
    for (const bool grouped : {false, true}) {
        FakeEng e(4);
        e.stops = false;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(4, 4096, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.prefill_chunk = 512;
        lc.prefill_chunk_cap = 4096;
        lc.interleave.select = grouped ? sv::PrefillInterleave::Select::automatic : sv::PrefillInterleave::Select::off;
        lc.interleave.max_stall_ms = 1e9;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::vector<std::shared_ptr<sv::GenerationTicket>> tickets;
        // Queue everything before running, so prompt completion cannot race successor submission.
        for (unsigned i = 0; i < 4; ++i) {
            sv::GenerationJob j;
            j.prompt.assign(i == 0 ? 8U : 2560U, 5U);
            j.max_tokens = i == 0 ? 100U : 1U;
            tickets.push_back(queue.submit(std::move(j)));
        }
        std::thread runner([&] { lead.loop(); });
        for (const auto& ticket : tickets) {
            bool done = false;
            while (!done) (void)ticket->wait(std::chrono::milliseconds(100), done);
        }
        stop.store(true);
        runner.join();
        unsigned chunks = 0, steps = 0, owed = 0, violations = 0, successors = 0;
        for (const auto& line : e.log) {
            if (line.rfind("step", 0) == 0) ++steps;
            if (line.rfind("prefill 1 ", 0) != 0) continue;
            if (chunks != 0) {
                if (steps != owed) ++violations;
                if (line.find(" @0 ") != std::string::npos) ++successors;
            }
            owed = grouped && line.find(" +2048") != std::string::npos ? 4U : 1U;
            steps = 0;
            ++chunks;
        }
        CHECK(chunks == (grouped ? 6U : 15U) && successors == 2 && violations == 0);
        std::printf("queued-successor interleave (%s): %u chunks, %u successors, %u debt violations\n",
                    grouped ? "grouped" : "single", chunks, successors, violations);
    }

}



// FakeEng with logits for sampled requests: a row's logits are a function of the chain state after its input (both
// ranks hold the same states), over a 64-id vocabulary split 32 / 32 between the ranks, with one strong peak (the
// row's argmax). Plain sampled rows go through sample_pair_rows, sampled MTP windows through decide_pair_window, so
// a sampled request's tokens must not depend on the route.
constexpr std::uint32_t kFakeVocab = 64, kFakeSlice = 32;
struct SampledFake : FakeEng {
    using FakeEng::FakeEng;
    using FakeEng::mtp_run;
    struct RowAt { std::uint32_t slot; std::uint32_t token; std::uint64_t pos; std::uint64_t x; };
    std::vector<RowAt> last_sampled;  // rows of the last step / prefill call (sample_outputs reads them)
    bool last_prefill = false;
    std::uint64_t sampled_runs = 0, window_rows = 0, draft_picks = 0, sampled_enters = 0;
    // windows that ended at an accepted stop draft in row 0 with more rows allowed; the frontier at each mtp_leave
    std::uint64_t stop_cuts_row0 = 0;
    std::vector<std::uint64_t> leave_frontier;
    std::uint64_t lead_accepted = 0, lead_rejected = 0;
    bool lead_coupled = true;
    std::uint32_t dec_flags = 0;
    // the fake drafter's rows: the target's peak a little lower and other noise, so q is close to p but not equal
    static float draft_logit(std::uint64_t x, std::uint32_t t) {
        if (t == peak(x)) return 6.0F;
        const std::uint64_t m = (x ^ (std::uint64_t{t} * 0xD6E8FEB86659FD93ULL)) * 0x94D049BB133111EBULL;
        return static_cast<float>((m >> 40) % 1000U) / 400.0F;
    }
    std::vector<float> draft_slice(std::uint64_t x, bool paired) const {
        const std::uint32_t lo = paired ? rank * kFakeSlice : 0U, n = paired ? kFakeSlice : kFakeVocab;
        std::vector<float> v(n);
        for (std::uint32_t i = 0; i < n; ++i) v[i] = draft_logit(x, lo + i);
        return v;
    }
    static std::uint32_t peak(std::uint64_t x) { return static_cast<std::uint32_t>((x >> 13) % kFakeVocab); }
    static float logit(std::uint64_t x, std::uint32_t t) {
        if (t == peak(x)) return 7.0F;
        const std::uint64_t m = (x ^ (std::uint64_t{t} * 0x9E3779B97F4A7C15ULL)) * 0xBF58476D1CE4E5B9ULL;
        return static_cast<float>((m >> 40) % 1000U) / 400.0F;
    }
    std::vector<float> slice(std::uint64_t x, bool paired) const {
        const std::uint32_t lo = paired ? rank * kFakeSlice : 0U, n = paired ? kFakeSlice : kFakeVocab;
        std::vector<float> v(n);
        for (std::uint32_t i = 0; i < n; ++i) v[i] = logit(x, lo + i);
        return v;
    }
    std::vector<std::uint32_t> step(std::span<const gs::WireRow> rows) {
        auto out = FakeEng::step(rows);
        last_sampled.clear();
        for (std::size_t i = 0; i < rows.size(); ++i) last_sampled.push_back({rows[i].slot, rows[i].token, rows[i].pos, h[rows[i].slot]});
        last_prefill = false;
        return out;
    }
    std::uint32_t prefill(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0) {
        const auto y = FakeEng::prefill(s, t, pos0);
        last_sampled = {{s, t.empty() ? 0U : t.back(), pos0 + t.size() - 1, h[s]}};
        last_prefill = true;
        return y;
    }
    std::uint32_t prefill_mtp(std::uint32_t s, std::span<const std::uint32_t> t, std::uint64_t pos0, bool last) {
        const auto y0 = prefill(s, t, pos0);
        log.back() += last ? " mtp last" : " mtp";
        if (last) {
            seed = Seed{s, y0, pos0 + t.size()};
            seed_spec(s);
        }
        return y0;
    }
    void sample_outputs(gs::Channel& ch, std::uint32_t mask, std::span<const sv::PairSamplingRow> choices,
                        std::span<std::uint32_t> out) {
        std::uint64_t chunk = 0;
        std::vector<float> local;
        sv::sample_pair_rows(mask, ch.rank() == 0, ch.paired() ? kFakeSlice : 0U, kFakeVocab, choices, out,
            [&](std::size_t r) { local = slice(last_sampled.at(r).x, ch.paired()); return std::span<const float>(local); },
            [&](std::span<const float> a, std::span<float> b) { ch.exchange_sampling_logits(a, b, chunk++); });
        // the sampled token is what the row produced (mtp_enter seeds from it); the follower learns it from the lead's
        // next command, so only the lead's record can hold it here: both records are rebuilt from mtp_enter's row
        if (ch.rank() != 0) return;
        if (last_prefill) {
            last_rows = {gs::WireRow{last_sampled[0].slot, last_sampled[0].token, last_sampled[0].pos}};
            last_out = {out[0]};
        } else {
            for (std::size_t r = 0; r < out.size() && r < last_out.size(); ++r) last_out[r] = out[r];
        }
    }
    // mtp_enter from a sampled row: the follower's record does not hold the sampled token, so the row is checked
    // against the state (slot, position) only; the token is taken from the command.
    void mtp_enter(std::uint32_t r, const gs::WireRow& w, bool from_seed) {
        if (!from_seed && r < last_sampled.size() && last_sampled[r].slot == w.slot && last_sampled[r].pos == w.pos &&
            frontier[w.slot] == w.pos + 1) {
            last_rows = {gs::WireRow{w.slot, last_sampled[r].token, w.pos}};
            last_out.assign(r + 1, 0);
            last_rows.resize(r + 1, last_rows[0]);
            last_rows[r] = gs::WireRow{w.slot, last_sampled[r].token, w.pos};
            last_out[r] = w.token;
        }
        FakeEng::mtp_enter(r, w, from_seed);
    }
    void mtp_enter(std::uint32_t r, const gs::WireRow& w, bool from_seed, gs::SampledRun run) {
        if (from_seed || (run.flags & gs::kRunSampleDrafts) == 0) throw std::runtime_error("fake sampled mtp_enter: flags");
        ++sampled_enters;
        mtp_enter(r, w, from_seed);
    }
    std::vector<std::uint32_t> mtp_run(std::uint32_t s, std::uint32_t budget, std::uint32_t n, gs::SampledRun run) {
        if (dec && dec_flags != run.flags) throw std::runtime_error("fake sampled mtp_run: flags changed under an open decoder");
        if (!dec) {
            if (!seed || seed->slot != s || frontier[s] != seed->p) throw std::runtime_error("fake sampled mtp_run without a fresh seed");
            dec = Dec{s, seed->y0, budget, 1};
            dec_flags = run.flags;
            seed.reset();
            spec[s] = {};
        } else if (dec->slot != s) {
            throw std::runtime_error("fake sampled mtp_run: other slot");
        }
        const bool paired = run.ch->paired();
        std::uint64_t chunk = 0;
        std::vector<std::uint32_t> out;
        for (std::uint32_t c = 0; c < n && dec->emitted < dec->budget && !dec_stopped; ++c) {
            const unsigned remaining = dec->budget - dec->emitted;
            const unsigned k = std::min(1U + static_cast<unsigned>(h[s] % 3U), remaining - 1U);
            std::vector<std::uint32_t> inputs{dec->last}, drafts, ids;
            std::vector<std::uint64_t> xs{fold(h[s], dec->last)};
            const std::uint64_t first = frontier[s] + 1;
            std::vector<strix::TokenDraft> td;
            std::vector<float> local;
            for (unsigned i = 0; i < k; ++i) {  // the fake drafter: the row's argmax, wrong for a quarter of the states
                const auto x = xs.back();
                std::uint32_t d = ((x >> 17) % 4U) != 0 ? peak(x) : (peak(x) + 1U) % kFakeVocab;
                strix::TokenDraft kept;
                if ((run.flags & gs::kRunSampleDrafts) != 0) {  // drawn from the drafter's row on the lead
                    d = sv::pick_pair_draft(run.sampler != nullptr, paired ? kFakeSlice : 0U, kFakeVocab, run.sampler, first + i,
                        [&] { local = draft_slice(x, paired); return std::span<const float>(local); },
                        [&](std::span<const float> a, std::span<float> b) { run.ch->exchange_sampling_logits(a, b, chunk++); },
                        [&](std::span<const std::uint32_t> a, std::span<std::uint32_t> b) { run.ch->exchange_sampling_words(a, b, chunk++); },
                        &kept);
                    ++draft_picks;
                }
                if (run.sampler == nullptr || (run.flags & gs::kRunSampleDrafts) == 0) {
                    kept = strix::TokenDraft{};
                    kept.token = d;
                }
                kept.position = first + i;
                td.push_back(std::move(kept));
                drafts.push_back(d);
                inputs.push_back(d);
                xs.push_back(fold(x, d));
            }
            for (const auto x : xs) ids.push_back(peak(x));
            const auto window = sv::decide_pair_window(run.sampler != nullptr, paired ? kFakeSlice : 0U, kFakeVocab, run.sampler,
                first, td, std::min(k + 1U, remaining), run.stops,
                [&](std::size_t r) { local = slice(xs.at(r), paired); return std::span<const float>(local); },
                [&](std::span<const float> a, std::span<float> b) { run.ch->exchange_sampling_logits(a, b, chunk++); },
                [&](std::span<const std::uint32_t> a, std::span<std::uint32_t> b) { run.ch->exchange_sampling_words(a, b, chunk++); });
            window_rows += window.tokens.size();
            if (window.stop && window.accepted == 1U && window.tokens.size() == 1U && std::min(k + 1U, remaining) > 1U) ++stop_cuts_row0;
            std::copy(window.tokens.begin(), window.tokens.end(), ids.begin());
            // as the engine's cycle (glm_spec_run): commit by the window's accepted count, not by token equality; a
            // decided stop id is terminal
            const auto acc = strix::glm::spec::accept_decided(drafts, ids, window.accepted, strix::glm::spec::kNoFlag,
                strix::glm::spec::Terminal{window.stop ? window.tokens.back() : strix::glm::spec::kNoFlag, remaining});
            if (acc.stop) dec_stopped = true;
            for (unsigned j = 0; j < acc.rows; ++j) (void)row(s, inputs[j]);
            if (run.sampler) {
                lead_accepted = run.sampler->stats().accepted;
                lead_rejected = run.sampler->stats().rejected;
                lead_coupled = run.sampler->coupled();
            }
            for (unsigned j = 0; j < acc.m; ++j) out.push_back(drafts[j]);
            if (acc.has_bonus) out.push_back(acc.bonus);
            dec->emitted += acc.m + (acc.has_bonus ? 1U : 0U);
            dec->last = out.back();
            ++cycles;
            ++sampled_runs;
        }
        anchor[s] = dec->last;
        log.push_back("sampled run " + std::to_string(s) + " -> " + std::to_string(out.size()));
        return out;
    }
    bool dec_stopped = false;
    void mtp_leave(std::uint32_t s) {
        leave_frontier.push_back(frontier.at(s));
        FakeEng::mtp_leave(s);
        dec_flags = 0;
        dec_stopped = false;
    }
};

struct SampledServe {
    std::vector<std::vector<std::uint32_t>> tokens;
    std::vector<std::string> log0;
    std::uint64_t mtp_rounds = 0, sampled_runs = 0, steps = 0, g2_cycles = 0, draft_picks = 0, sampled_enters = 0;
    std::uint64_t lead_accepted = 0, lead_rejected = 0;
    bool lead_coupled = true;
    std::vector<std::string> receipts;  // each request's sampling_receipt ("" when none)
    std::string error;
    std::uint64_t window_rows = 0, stop_cuts_row0 = 0;
    std::vector<std::uint64_t> leave_frontier0, leave_frontier1;  // the committed frontier at each mtp_leave, per rank
};
// One pair (loopback) serving `jobs`; sequential = one request at a time (the C1 route), otherwise all queued at once.
SampledServe serve_sampled(const std::vector<sv::GenerationJob>& jobs, bool sampled_mtp, bool sequential, bool g2 = false,
                           strix::SampledAcceptance accept = strix::SampledAcceptance::coupled,
                           std::uint32_t force_reject_row = strix::TokenSamplerOptions::forced_rejection_off) {
    SampledServe out;
    Loopback lb;
    SampledFake e0(4), e1(4);
    e1.rank = 1;
    e0.stops = e1.stops = false;
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    const gs::Hello h = hello_for(4, 4096, 2);
    std::thread follower([&] {
        try {
            gs::follow(e1, c1, h, 4, 4096, true);
        } catch (const std::exception& e) {
            out.error += std::string("follower: ") + e.what();
        }
    });
    std::atomic<bool> stop{false};
    sv::GenerationQueue queue(16);
    gs::LeadConfig lc;
    lc.slots = 4;
    lc.capacity = 4096;
    lc.heartbeat_seconds = 0.01;
    lc.mtp = true;
    lc.mtp_cycles = 2;
    lc.prefill_chunk = 0;
    lc.g2 = g2;
    lc.sampling = c0.hello(h, true);
    lc.sampled_mtp = sampled_mtp;
    lc.sampler_options.acceptance = accept;
    lc.sampler_options.force_reject_row = force_reject_row;
    try {
        gs::Lead<SampledFake> lead(e0, c0, queue, lc, stop);
        std::thread runner([&] {
            try {
                lead.loop();
            } catch (const std::exception& e) {
                out.error += std::string("lead: ") + e.what();
                queue.shutdown();
            }
        });
        std::vector<std::shared_ptr<sv::GenerationTicket>> tickets;
        out.tokens.resize(jobs.size());
        const auto drain = [&](std::size_t i) {
            if (!tickets[i]) {
                out.error += " request " + std::to_string(i) + " not queued";
                return;
            }
            bool done = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            while (!done && std::chrono::steady_clock::now() < deadline) {
                const auto t = tickets[i]->wait(std::chrono::milliseconds(100), done);
                out.tokens[i].insert(out.tokens[i].end(), t.begin(), t.end());
            }
            if (!done) out.error += " request " + std::to_string(i) + " timed out";
            std::string receipt;
            for (const auto& [k, v] : tickets[i]->stats().engine_extra)
                if (k == "sampling_receipt") receipt = v;
            out.receipts.push_back(receipt);
        };
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            tickets.push_back(queue.submit(sv::GenerationJob(jobs[i])));
            if (sequential) drain(i);
        }
        if (!sequential)
            for (std::size_t i = 0; i < jobs.size(); ++i) drain(i);
        stop.store(true);
        runner.join();
        out.mtp_rounds = lead.stats().mtp_rounds;
        out.steps = lead.stats().steps;
        out.g2_cycles = lead.stats().g2_cycles;
    } catch (const std::exception& e) {
        out.error += std::string("setup: ") + e.what();
    }
    follower.join();
    out.log0 = e0.log;
    out.sampled_runs = e0.sampled_runs;
    out.draft_picks = e0.draft_picks;
    out.sampled_enters = e0.sampled_enters;
    out.lead_accepted = e0.lead_accepted;
    out.lead_rejected = e0.lead_rejected;
    out.lead_coupled = e0.lead_coupled;
    out.window_rows = e0.window_rows;
    out.stop_cuts_row0 = e0.stop_cuts_row0;
    out.leave_frontier0 = e0.leave_frontier;
    out.leave_frontier1 = e1.leave_frontier;
    return out;
}

sv::GenerationJob sampled_job(unsigned i, double temperature, std::uint32_t max_tokens) {
    sv::GenerationJob j;
    for (unsigned k = 0; k < 9 + 5 * i; ++k) j.prompt.push_back(100U + 13U * i + k);
    j.max_tokens = max_tokens;
    j.sampling.temperature = temperature;
    j.sampling.seed = 1000U + i;
    j.sampling.seeded = true;
    return j;
}

void test_sampled_drafts_serve() {
    // coupled, temperature 0.9 (above the argmax gate): drafts drawn from the drafter rows, tokens of plain sampling
    std::vector<sv::GenerationJob> hot, cool;
    for (unsigned i = 0; i < 3; ++i) {
        hot.push_back(sampled_job(i, 0.9, 40));
        cool.push_back(sampled_job(i, 0.5, 40));
    }
    const auto hot_plain = serve_sampled(hot, false, true), hot_spec = serve_sampled(hot, true, true);
    CHECK(hot_plain.error.empty() && hot_spec.error.empty() && hot_plain.tokens == hot_spec.tokens);
    CHECK(hot_spec.draft_picks > 0 && hot_spec.sampled_enters > 0 && hot_spec.lead_accepted > 0);
    // temperature 0.5 (below the gate): greedy drafts, no drafter row crosses the pair
    const auto cool_plain = serve_sampled(cool, false, true), cool_spec = serve_sampled(cool, true, true);
    CHECK(cool_plain.error.empty() && cool_spec.error.empty() && cool_plain.tokens == cool_spec.tokens);
    CHECK(cool_spec.mtp_rounds > 0 && cool_spec.draft_picks == 0 && cool_spec.sampled_enters == 0);
    // rejection (opt-in): exact in distribution, not token-equal to plain sampling; every request completes
    const auto rej = serve_sampled(hot, true, true, false, strix::SampledAcceptance::rejection);
    CHECK(rej.error.empty() && !rej.lead_coupled && rej.draft_picks > 0 && rej.lead_accepted + rej.lead_rejected > 0);
    for (const auto& t : rej.tokens) CHECK(t.size() == 40);
    // every sampled response states its acceptance rule and what its drafts did (plain steps: no drafted rows)
    const auto has = [](const std::vector<std::string>& r, const std::string& needle) {
        return r.size() == 3 && std::all_of(r.begin(), r.end(), [&](const std::string& x) { return x.find(needle) != std::string::npos; });
    };
    CHECK(has(hot_plain.receipts, "\"acceptance\":\"coupled\"") && has(hot_plain.receipts, "\"accepted\":0,\"rejected\":0"));
    CHECK(has(hot_spec.receipts, "\"acceptance\":\"coupled\"") && !has(hot_spec.receipts, "\"accepted\":0,"));
    CHECK(has(rej.receipts, "\"acceptance\":\"rejection\""));
    if (!hot_spec.error.empty() || !cool_spec.error.empty() || !rej.error.empty())
        std::fprintf(stderr, "sampled drafts errors: %s | %s | %s\n", hot_spec.error.c_str(), cool_spec.error.c_str(), rej.error.c_str());
    std::printf("sampled drafts: T0.9 coupled == plain (%llu draft picks, %llu accepted), T0.5 no draft exchange, "
                "rejection completes (%llu accepted / %llu rejected)\n",
                static_cast<unsigned long long>(hot_spec.draft_picks), static_cast<unsigned long long>(hot_spec.lead_accepted),
                static_cast<unsigned long long>(rej.lead_accepted), static_cast<unsigned long long>(rej.lead_rejected));
}

void test_sampled_mtp_serve() {
    std::vector<sv::GenerationJob> jobs;
    for (unsigned i = 0; i < 3; ++i) jobs.push_back(sampled_job(i, 0.9, 40));
    const auto plain = serve_sampled(jobs, false, true);
    const auto spec = serve_sampled(jobs, true, true);
    CHECK(plain.error.empty() && spec.error.empty());
    if (!plain.error.empty() || !spec.error.empty()) std::fprintf(stderr, "sampled serve errors: %s | %s\n", plain.error.c_str(), spec.error.c_str());
    CHECK(plain.tokens == spec.tokens);
    for (const auto& t : spec.tokens) CHECK(t.size() == 40);
    CHECK(plain.mtp_rounds == 0 && plain.sampled_runs == 0);  // switch off: sampled requests decode by plain steps
    // a rejection whose token equals its draft (coupled, forced at row 0 of every window): the row after it was never
    // decided, so nothing past the rejected row may be committed; the tokens stay those of plain sampling
    const auto forced = serve_sampled(jobs, true, true, false, strix::SampledAcceptance::coupled, 0);
    CHECK(forced.error.empty() && forced.tokens == plain.tokens && forced.lead_rejected > 0);
    if (!forced.error.empty()) std::fprintf(stderr, "forced rejection serve error: %s\n", forced.error.c_str());
    CHECK(spec.mtp_rounds > 0 && spec.sampled_runs > 0);
    // the sampled prefill's seed (drawn from its greedy first token) is never used: the decoder opens from mtp_enter
    unsigned enters = 0, seed_enters = 0;
    for (const auto& l : spec.log0) {
        enters += l.rfind("enter ", 0) == 0;
        seed_enters += l.rfind("enter-seed", 0) == 0;
    }
    CHECK(enters >= 3 && seed_enters == 0);
    // a greedy request beside sampled ones with batched verify on: the sampled tokens stay those of plain sampling
    std::vector<sv::GenerationJob> mixed = jobs;
    sv::GenerationJob greedy = sampled_job(5, 0.0, 40);
    mixed.insert(mixed.begin() + 1, greedy);
    const auto both = serve_sampled(mixed, true, false, true);
    CHECK(both.error.empty());
    if (!both.error.empty()) std::fprintf(stderr, "mixed serve error: %s\n", both.error.c_str());
    CHECK(both.tokens.size() == 4 && both.tokens[0] == plain.tokens[0] && both.tokens[2] == plain.tokens[1] && both.tokens[3] == plain.tokens[2]);
    std::printf("sampled MTP serve: tokens == plain sampling (3 x 40), %llu MTP rounds / %llu sampled cycles, %u re-seeds; "
                "mixed with greedy + G2: sampled tokens unchanged (%llu G2 cycles)\n",
                static_cast<unsigned long long>(spec.mtp_rounds), static_cast<unsigned long long>(spec.sampled_runs), enters,
                static_cast<unsigned long long>(both.g2_cycles));
}

// A sampled MTP window never decides, commits or catches up a row past a stop id of the request, on either rank: the
// request's stop token is taken from its free-running output, so some window accepts it as the draft of row 0 while
// more rows were allowed. The tokens are those of plain sampling (up to and including the stop), the lead decides no
// row after it, and the frontier each rank holds when the decoder leaves never passes prompt + emitted - 1 (the stop
// token is never processed as input).
void test_sampled_mtp_stop() {
    const sv::GenerationJob base = sampled_job(0, 0.9, 40);
    const auto free_run = serve_sampled({base}, false, true);
    CHECK(free_run.error.empty() && free_run.tokens.size() == 1 && free_run.tokens[0].size() == 40);
    if (free_run.tokens.size() != 1 || free_run.tokens[0].size() != 40) return;
    const auto& all = free_run.tokens[0];
    unsigned tried = 0, cut_at = 0;
    for (std::size_t k = 1; k + 1 < all.size() && cut_at == 0; ++k) {
        if (std::find(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all[k]) != all.begin() + static_cast<std::ptrdiff_t>(k))
            continue;  // the stop must end the output at its first occurrence
        auto job = base;
        job.stop_tokens = {all[k]};
        const std::vector<std::uint32_t> want(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k) + 1);
        const auto plain = serve_sampled({job}, false, true), spec = serve_sampled({job}, true, true);
        ++tried;
        CHECK(plain.error.empty() && spec.error.empty());
        if (!plain.error.empty() || !spec.error.empty())
            std::fprintf(stderr, "sampled stop errors: %s | %s\n", plain.error.c_str(), spec.error.c_str());
        CHECK(plain.tokens.size() == 1 && plain.tokens[0] == want && spec.tokens.size() == 1 && spec.tokens[0] == want);
        CHECK(spec.window_rows <= want.size());
        const std::uint64_t bound = job.prompt.size() + want.size() - 1U;
        CHECK(!spec.leave_frontier0.empty() && !spec.leave_frontier1.empty());
        for (const auto f : spec.leave_frontier0) CHECK(f <= bound);
        for (const auto f : spec.leave_frontier1) CHECK(f <= bound);
        if (spec.stop_cuts_row0 > 0) cut_at = static_cast<unsigned>(k);
    }
    CHECK(cut_at > 0);
    std::printf("sampled MTP stop: an accepted stop draft in row 0 ends the window (stop at output %u, %u stop ids tried); "
                "tokens == plain sampling, no row past the stop on either rank\n", cut_at, tried);
}

// FakeEng with the sampled MTP entry points: records what replay hands them.
struct SampledProbe : FakeEng {
    using FakeEng::FakeEng;
    using FakeEng::mtp_enter;
    using FakeEng::mtp_run;
    std::vector<gs::SampledRun> runs, enters;
    std::vector<std::uint32_t> mtp_run(std::uint32_t, std::uint32_t, std::uint32_t, gs::SampledRun run) {
        runs.push_back(run);
        return {5, 6};
    }
    void mtp_enter(std::uint32_t, const gs::WireRow&, bool, gs::SampledRun run) { enters.push_back(run); }
};

void test_sampled_mtp_replay() {
    gs::Channel solo({}, 0);
    solo.hello(hello_for(2, 64, 1));
    strix::TokenSamplingPolicy policy;
    policy.temperature = 0.8F;
    strix::TokenSampler sampler(policy, strix::TokenSamplerOptions{});
    const std::array<sv::PairSamplingRow, 1> choice{{{&sampler, 0}}};
    std::vector<std::uint32_t> none;
    gs::Control run;
    run.cmd = static_cast<std::uint32_t>(gs::Cmd::mtp_run);
    run.pslot = 1;
    run.budget = 9;
    run.cycles = 2;
    run.pflags = gs::kRunSample;
    CHECK(gs::decode(gs::encode(run)).pflags == gs::kRunSample);
    FakeEng plain(2);  // no sampled entry points: a sampled command is a protocol error, not a greedy run
    CHECK(throws_with([&] { (void)gs::replay(plain, solo, run, none, true, 2, 64, nullptr, choice); }, "sampled MTP unavailable"));
    SampledProbe probe(2);
    const auto out = gs::replay(probe, solo, run, none, true, 2, 64, nullptr, choice);
    CHECK(out == (std::vector<std::uint32_t>{5, 6}) && probe.runs.size() == 1);
    CHECK(probe.runs.at(0).ch == &solo && probe.runs.at(0).sampler == &sampler && probe.runs.at(0).flags == gs::kRunSample);
    (void)gs::replay(probe, solo, run, none, false, 2, 64);  // the follower runs the same command without a sampler
    CHECK(probe.runs.size() == 2 && probe.runs.at(1).sampler == nullptr && probe.runs.at(1).flags == gs::kRunSample);
    CHECK(throws_with([&] { (void)gs::replay(probe, solo, run, none, true, 2, 64); }, "sampled MTP without a sampler"));
    run.pflags = gs::kRunSample | 16U;
    CHECK(throws_with([&] { (void)gs::replay(probe, solo, run, none, true, 2, 64, nullptr, choice); }, "mtp_run flags"));
    run.pflags = gs::kRunSampleDrafts;  // sampled drafts need sampled targets
    CHECK(throws_with([&] { (void)gs::replay(probe, solo, run, none, true, 2, 64, nullptr, choice); }, "mtp_run flags"));
    run.pflags = gs::kRunSample | gs::kRunRejection;  // rejection needs the drafts' distributions: sampled drafts
    CHECK(throws_with([&] { (void)gs::replay(probe, solo, run, none, true, 2, 64, nullptr, choice); }, "mtp_run flags"));
    gs::Control enter;
    enter.cmd = static_cast<std::uint32_t>(gs::Cmd::mtp_enter);
    enter.pslot = 0;
    enter.row[0] = {1, 4, 7};
    enter.pflags = gs::kRunSample | gs::kRunSampleDrafts;
    (void)gs::replay(probe, solo, enter, none, true, 2, 64, nullptr, choice);
    CHECK(probe.enters.size() == 1 && probe.enters.at(0).flags == (gs::kRunSample | gs::kRunSampleDrafts) &&
          probe.enters.at(0).sampler == &sampler);
    // A sampled final units call that the engine did not complete keeps kPrefillPending (the lead's unit check then
    // fails loudly) instead of a token sampled from rows that are not the prompt's last.
    {
        SampledFake units(2);
        units.unit_rows = 4;
        units.unit_layers = 2;
        gs::Control pf;
        pf.cmd = static_cast<std::uint32_t>(gs::Cmd::prefill);
        pf.pslot = 1;
        pf.resets = 1U << 1;
        pf.ppos = 0;
        pf.pflags = gs::kPrefillMtp | gs::kPrefillUnits | gs::kPrefillStart | gs::kPrefillLast | gs::kPrefillSample;
        pf.budget = 1;  // one unit of several: the engine is still pending
        std::vector<std::uint32_t> prompt{5, 6, 7, 8, 9, 10, 11, 12, 13};
        pf.plen = static_cast<std::uint32_t>(prompt.size());
        std::vector<std::uint32_t> first;
        const bool threw = throws_with([&] { first = gs::replay(units, solo, pf, prompt, true, 2, 64, nullptr, choice); }, "");
        CHECK(!threw && first.size() == 1 && first[0] == gs::kPrefillPending);
    }
    CHECK(gs::kControlVersion == 14);
    std::printf("sampled MTP replay: flags and the lead's sampler reach the engine; unknown flags and engines refused\n");
}

}  // namespace

// Host protocol regressions: geometry decline and imports ready inside/between pieces.
// These checks cover ordering/accounting, not native residual/logit/state equality.
void audit601_regressions(bool imports) {
    const unsigned cases = imports ? 2U : 4U;
    for (unsigned variant = 0; variant < cases; ++variant) {
        Loopback lb;
        FakeEng e(4), e1(4);
        e.stops = e1.stops = false;
        e.delay_us = e1.delay_us = 500;
        e.unit_rows = e1.unit_rows = imports ? 256U : 8192U;
        e.unit_layers = e1.unit_layers = imports ? 4U : 45U;
        std::atomic<bool> hold{true};
        std::atomic<unsigned> ready_ranks{0};
        if (imports) {
            e.hold = e1.hold = &hold;
            const auto hook = [&](FakeEng& f) {
                // One unit: inside a piece; four units: exactly between pieces, still active.
                if (f.unit_calls.size() == (variant == 0 ? 1U : 4U) && ready_ranks.fetch_add(1) == 1)
                    hold.store(false);
            };
            // Callbacks outlive this block: retain the hook by value.
            e.units_advanced = [&, hook] { hook(e); };
            e1.units_advanced = [&, hook] { hook(e1); };
        }
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(4, 32768, 2), 4, 32768); });
        ch.hello(hello_for(4, 32768, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 32768;
        lc.mtp = true;
        lc.short_first = false;
        lc.prefill_chunk = imports ? 64U : 512U;
        lc.prefill_units_rows = e.unit_rows;
        lc.prefill_units_layers = e.unit_layers;
        lc.prefill_units_admit = [&](std::uint64_t rows) {
            const strix::session::LayerUnits lu{rows, e.unit_rows, e.unit_layers};
            // Pipeline 2: seqpar pauses only even pieces; disabled seqpar cannot pause any.
            return lu.all_pieces_pausable([&](std::uint32_t r) { return imports || (variant != 2 && r >= 2 && r % 2 == 0); });
        };
        if (imports) {
            lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
            lc.import_min_tokens = 100;
            lc.import_listen_grace_s = 0;
        }
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto submit = [&](std::size_t rows, std::uint32_t token, std::uint32_t count) {
            sv::GenerationJob j;
            j.prompt.assign(rows, token);
            j.max_tokens = count;
            return queue.submit(std::move(j));
        };
        auto a = submit(16, 3, 400);
        bool done_a = false;
        auto tokens_a = a->wait(std::chrono::milliseconds(100), done_a);
        while (tokens_a.empty() && !done_a) tokens_a = a->wait(std::chrono::milliseconds(100), done_a);
        auto b = imports ? submit(300, 7, 6) : nullptr;
        // Odd single piece; odd multi-piece tail; seqpar disabled on even rows; supported control.
        const std::size_t rows = imports ? 640U : variant == 0 ? 8191U : variant == 1 ? 16383U : 8192U;
        auto c = submit(rows, 9, 6);
        auto drain = [](const auto& ticket) {
            std::vector<std::uint32_t> tokens;
            bool done = false;
            while (!done) {
                auto next = ticket->wait(std::chrono::milliseconds(100), done);
                tokens.insert(tokens.end(), next.begin(), next.end());
            }
            return tokens;
        };
        const auto tokens_c = drain(c);
        const auto tokens_b = b ? drain(b) : std::vector<std::uint32_t>{};
        const auto rest_a = drain(a);
        tokens_a.insert(tokens_a.end(), rest_a.begin(), rest_a.end());
        stop.store(true);
        runner.join();
        follower.join();
        auto plain = [](std::size_t n, std::uint32_t token, std::size_t count) {
            FakeEng f(1);
            f.stops = false;
            f.reset(0);
            const std::vector<std::uint32_t> prompt(n, token);
            std::vector<std::uint32_t> out{f.prefill(0, prompt, 0)};
            while (out.size() < count) {
                const gs::WireRow row{0, out.back(), f.frontier[0]};
                out.push_back(f.step({&row, 1})[0]);
            }
            return out;
        };
        CHECK(tokens_a == plain(16, 3, 400));
        CHECK(tokens_c == plain(rows, 9, 6));
        CHECK(e.log == e1.log && !e.units_open && !e1.units_open);
        if (imports) {
            CHECK(ready_ranks.load() == 2 && !hold.load());
            CHECK(tokens_b == plain(300, 7, 6));
            CHECK(b->stats().imported_tokens == 299 && lead.stats().imports == 1);
            CHECK(!e.unit_calls.empty() && e.dropped.empty() && e1.dropped.empty());
            std::size_t last_unit = 0, imported = 0;
            unsigned decode_between = 0;
            for (std::size_t i = 0; i < e.log.size(); ++i) {
                if (e.log[i].rfind("units ", 0) == 0) last_unit = i;
                if (e.log[i].rfind("import ", 0) == 0) imported = i;
                if (last_unit != 0 && imported == 0 && e.log[i].rfind("step", 0) == 0) ++decode_between;
            }
            CHECK(imported > last_unit && decode_between > 0);
        } else {
            CHECK(e.unit_calls.empty() == (variant != 3));
        }
        std::printf("audit601 %s variant %u: completed both ranks\n", imports ? "import deferral" : "geometry admission", variant);
    }
}

void import_route_receipts() {
    // ---- per-request route receipts: an imported prompt reports imported_tokens + route rtx; a refused one, its fallback
    for (const int variant : {0, 1, 2, 3, 4}) {  // success, staging failure, relay success, unavailable, corrupt payload
        const bool fail = variant == 1, dsa = variant >= 2, relay_fail = variant >= 3;
        Loopback lb;
        FakeEng e(4), e1(4);
        Pipe pipe;
        e.stops = e1.stops = false;
        e.fail_expect = fail;
        e.pipe = e1.pipe = &pipe;
        e1.rank = 1;
        e1.relay_fail = variant == 3;
        pipe.corrupt_dsa = variant == 4;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(4, 4096, 2), 4, 4096); });
        ch.hello(hello_for(4, 4096, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 100;
        lc.state_digest = true;
        lc.import_dsa_once = dsa;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        sv::GenerationJob j;
        j.prompt.assign(300, 9U);
        j.max_tokens = 4;
        auto t = queue.submit(std::move(j));
        if (variant == 4) {
            bool completed = false;
            while (!completed) (void)t->wait(std::chrono::milliseconds(100), completed);
        }
        sv::GenerationJob k;
        k.prompt = prompts(1)[0];  // short: never routed
        k.max_tokens = 4;
        auto u = queue.submit(std::move(k));
        bool done = false, done_u = false;
        while (!done) (void)t->wait(std::chrono::milliseconds(100), done);
        while (!done_u) (void)u->wait(std::chrono::milliseconds(100), done_u);
        stop.store(true);
        runner.join();
        follower.join();
        const auto st = t->stats(), su = u->stats();
        std::string route;
        for (const auto& [name, value] : st.engine_extra)
            if (name == "prefill_route") route = value;
        if (!fail && !relay_fail) CHECK(st.imported_tokens == 299 && route.find("\"route\":\"rtx\"") != std::string::npos && lead.stats().imported_tokens == 299);
        if (fail) {  // the fallback drops the hand-off on both ranks at once (import_drop)
            CHECK(st.imported_tokens == 0 && route.find("\"fallback\":\"rank0_staging_failed\"") != std::string::npos);
            CHECK(!e.dropped.empty() && e.dropped == e1.dropped);
        }
        if (dsa && !relay_fail) {  // both ranks relayed inside import_state's replay; rank 1 restored its full state
            CHECK(route.find("\"dsa_once\":true") != std::string::npos && route.find("\"relay_bytes\":407") != std::string::npos);
            CHECK(e1.relayed == relay_full(1) && e1.last_relay.chunks == 7);
            bool logged = false;
            for (const auto& line : e1.log) logged = logged || (line.rfind("import ", 0) == 0 && line.find(" dsa") != std::string::npos);
            CHECK(logged);
        }
        if (relay_fail) {  // header or checksum failure: both abort, then local prefill and later requests remain usable
            CHECK(st.imported_tokens == 0 && route.find("\"fallback\":\"both_restore_failed\"") != std::string::npos);
            CHECK(lead.stats().imports == 0 && lead.stats().import_fallbacks == 1);
            bool aborted0 = false, aborted1 = false;
            for (const auto& line : e.log) aborted0 = aborted0 || line.find("(import abort)") != std::string::npos;
            for (const auto& line : e1.log) aborted1 = aborted1 || line.find("(import abort)") != std::string::npos;
            CHECK(aborted0 && aborted1);
            CHECK(e.suffix_collectives == 0 && e1.suffix_collectives == 0);
            CHECK(t->error().empty() && u->error().empty());
        }
        CHECK(su.imported_tokens == 0 && su.engine_extra.empty());
        const auto dg = lead.take_digests();  // one per request, right after its prefill, rank 1's gathered by rank 0
        CHECK(dg.size() == 2 && dg[0].rank0.size() == 4 && dg[0].rank0 == dg[0].rank1 && dg[1].rank0 == dg[1].rank1);
        CHECK(!route.empty() && sv::parse_json(route).is_object());
        static const char* names[] = {"imported", "staging fails", "imported, DSA once", "DSA once, rank 1 cannot relay", "DSA once, corrupt payload"};
        std::printf("route receipt (%s): %s\n", names[variant], route.c_str());
    }
}

// Root store (LeadConfig::prefix_root): a producer whose leading system prompt (stable_prefix) repeats keeps its state at
// root_mark(stable, n, 1024, 2) on its way through the prefill (the fake snapshot throws unless the slot is exactly at
// that frontier); a sibling (same system prompt, other user turn) restores there and its tokens equal its cold run;
// the producer's tokens equal its run without a root; a restore that reached the root stores none.
// Image prompts on the pair: the lead flags each prefill chunk that holds virtual ids, the embed frame of its spans
// follows the tokens, and the follower prefills the same rows. Tokens must not depend on chunking or on MTP, image
// prompts must stay out of the prefix cache, and the rows must matter (the same ids as plain tokens give other text).
struct ImageRun {
    std::vector<std::uint32_t> image, text, repeat;
    std::string error, session_error;
    std::uint64_t rows0 = 0, rows1 = 0, frames0 = 0, frames1 = 0, bytes0 = 0, prefix_stores = 0, prefix_lookups = 0;
    std::uint64_t first_frame_bytes = 0;
    bool logs_equal = false, chunked = false;
};
ImageRun image_run(bool mtp, unsigned chunk, bool neighbour, unsigned units_rows = 0) {
    ImageRun out;
    const std::uint32_t slots = 2, width = strix::glm::kLanguageHidden, rows = 40, image_token = strix::glm::kImageToken;
    strix::session::EmbedInterner interner;
    strix::session::EmbedRows span;
    span.rows = rows;
    span.width = width;
    span.real_token_id = image_token;
    span.digest.fill(7);
    span.data.resize(std::size_t{rows} * width);
    for (std::size_t i = 0; i < span.data.size(); ++i) span.data[i] = static_cast<std::uint16_t>(i * 2654435761U >> 13U);
    const auto lease = interner.intern(std::move(span));
    const auto table = std::make_shared<const strix::session::EmbedTable>(std::vector<strix::session::EmbedLease>{lease});
    std::vector<std::uint32_t> prompt, plain;
    for (std::uint32_t k = 0; k < 30; ++k) prompt.push_back(500U + k);
    prompt.push_back(strix::glm::kImageBeginToken);
    for (const auto id : lease.ids()) prompt.push_back(id);
    prompt.push_back(strix::glm::kImageEndToken);
    for (std::uint32_t k = 0; k < 25; ++k) prompt.push_back(900U + k);
    for (const auto id : prompt) plain.push_back(strix::session::is_embed_id(id) ? image_token : id);

    Loopback lb;
    FakeEng e0(slots), e1(slots);
    Pipe pipe;
    e0.pipe = e1.pipe = &pipe;
    e1.rank = 1;
    e0.stops = e1.stops = false;
    e0.unit_rows = e1.unit_rows = units_rows;
    e0.unit_layers = e1.unit_layers = units_rows ? 4U : 0U;
    std::binary_semaphore neighbour_decoding(0), image_queued(0);
    if (neighbour) {
        e0.on_first_decode = [&] {
            neighbour_decoding.release();
            if (!image_queued.try_acquire_for(std::chrono::seconds(3)))
                throw std::runtime_error("image: timed out waiting for the image request");
        };
    }
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    gs::Hello h0 = hello_for(slots, 4096, 2), h1 = hello_for(slots, 4096, 2);
    h0.prefix_cache_bytes = h1.prefix_cache_bytes = 1U << 22;
    h0.mtp = h1.mtp = mtp ? 1U : 0U;
    std::string follow_error, lead_error, driver_error;
    std::thread follower([&] {
        try { gs::follow(e1, c1, h1, slots, 4096); } catch (const std::exception& e) { follow_error = e.what(); }
    });
    std::atomic<bool> stop{false};
    sv::GenerationQueue queue(16);
    gs::LeadConfig lc;
    lc.prefix_cache_bytes = h0.prefix_cache_bytes;
    lc.slots = slots;
    lc.capacity = 4096;
    lc.heartbeat_seconds = 0.01;
    lc.mtp = mtp;
    lc.mtp_cycles = 2;
    lc.prefill_chunk = chunk;
    lc.interleave.select = sv::PrefillInterleave::Select::off;
    if (units_rows) {
        lc.prefill_units_rows = units_rows;
        lc.prefill_units_layers = 4;
        lc.prefill_units_admit = [](std::uint64_t) { return true; };
    }
    try {
        c0.hello(h0);
        gs::Lead<FakeEng> lead(e0, c0, queue, lc, stop);
        std::thread driver([&] {
            try {
                const auto submit = [&](const std::vector<std::uint32_t>& p, bool image, std::uint32_t max_tokens, const std::string& session = {}) {
                    sv::GenerationJob job;
                    job.prompt = p;
                    if (image) job.embeds = table;
                    job.session_id = session;
                    job.max_tokens = max_tokens;
                    auto t = queue.submit(std::move(job));
                    if (!t) throw std::runtime_error("image: queue refused a job");
                    return t;
                };
                const auto collect = [](const std::shared_ptr<sv::GenerationTicket>& t, std::string* error = nullptr) {
                    std::vector<std::uint32_t> got_all;
                    for (bool done = false; !done;) {
                        const auto got = t->wait(std::chrono::milliseconds(200), done);
                        got_all.insert(got_all.end(), got.begin(), got.end());
                    }
                    if (error) *error = t->error();
                    return got_all;
                };
                // A neighbour that keeps decoding makes the scheduler cut the image prompt into chunks.
                std::shared_ptr<sv::GenerationTicket> busy;
                if (neighbour) {
                    std::vector<std::uint32_t> other;
                    for (std::uint32_t k = 0; k < 9; ++k) other.push_back(40U + k);
                    busy = submit(other, false, 400);
                    if (!neighbour_decoding.try_acquire_for(std::chrono::seconds(3)))
                        throw std::runtime_error("image: neighbour did not start decoding");
                }
                const auto before = lead.stats();
                const auto first_image = submit(prompt, true, 12);
                if (neighbour) image_queued.release();
                out.image = collect(first_image, &out.error);
                out.first_frame_bytes = e0.embed_frame_bytes;
                out.prefix_stores = lead.stats().prefix_stores - before.prefix_stores;
                out.prefix_lookups = lead.stats().prefix_hits + lead.stats().prefix_misses - before.prefix_hits - before.prefix_misses;
                out.repeat = collect(submit(prompt, true, 12));
                if (busy) (void)collect(busy);
                out.text = collect(submit(plain, false, 12));
                (void)collect(submit(prompt, true, 12, "conversation-1"), &out.session_error);
            } catch (const std::exception& e) {
                driver_error = e.what();
            }
            stop.store(true);
        });
        try {
            lead.loop();
        } catch (const std::exception& e) {
            lead_error = e.what();
            queue.shutdown();
        }
        driver.join();
    } catch (const std::exception& e) {
        lead_error = e.what();
    }
    follower.join();
    if (!lead_error.empty() || !follow_error.empty() || !driver_error.empty()) {
        std::printf("image: lead '%s' follow '%s' driver '%s'\n", lead_error.c_str(), follow_error.c_str(), driver_error.c_str());
        out.error = "pair failed";
    }
    out.rows0 = e0.embed_rows;
    out.rows1 = e1.embed_rows;
    out.frames0 = e0.embed_frames;
    out.frames1 = e1.embed_frames;
    out.bytes0 = e0.embed_frame_bytes;
    out.logs_equal = e0.log == e1.log;
    unsigned image_calls = 0;  // prefill calls of the first image request's slot before it decodes
    for (const auto& l : e0.log) image_calls += l.rfind("prefill ", 0) == 0 || l.rfind("units ", 0) == 0 ? 1U : 0U;
    out.chunked = image_calls > 4;
    return out;
}
void image_prompts() {
    const std::uint64_t span_bytes = 40ULL * strix::glm::kLanguageHidden * 2U;
    const ImageRun whole = image_run(false, 0, false);
    CHECK(whole.error.empty() && whole.image.size() == 12 && whole.image == whole.repeat);
    CHECK(whole.image != whole.text);                       // the provided rows are what the model saw
    CHECK(whole.logs_equal && whole.rows0 == 80 && whole.rows1 == 80);  // two image requests, 40 rows each, on both ranks
    CHECK(whole.frames0 == 2 && whole.frames1 == 2);
    CHECK(whole.first_frame_bytes > span_bytes && whole.bytes0 < whole.first_frame_bytes + 1024);  // the rows travel once
    CHECK(whole.prefix_stores == 0 && whole.prefix_lookups == 0);       // an image prompt has no cache key
    CHECK(whole.session_error.find("image input does not take sessions") != std::string::npos);
    struct Case { bool mtp; unsigned chunk; bool neighbour; unsigned units; };
    for (const Case k : {Case{true, 0, false, 0}, Case{false, 16, true, 0}, Case{true, 16, true, 0}, Case{true, 16, true, 64}}) {
        const ImageRun r = image_run(k.mtp, k.chunk, k.neighbour, k.units);
        CHECK(r.error.empty() && r.image == whole.image && r.repeat == whole.image && r.text == whole.text);
        CHECK(r.logs_equal && r.rows0 == 80 && r.rows1 == 80 && r.frames0 == r.frames1);
        CHECK(r.prefix_stores == 0 && r.prefix_lookups == 0);
        CHECK(r.first_frame_bytes > span_bytes && r.bytes0 < r.first_frame_bytes + 8192);
        if (k.neighbour && k.units == 0) CHECK(r.chunked && r.frames0 > 2);  // one frame per chunk that holds image rows
    }
    std::printf("image prompts: %s\n", failures == 0 ? "PASS" : "FAIL");
}

void prefix_root_sibling() {
    CHECK(gs::prefix_root_row(2100, 2300) == 2048);
    CHECK(gs::prefix_root_row(1024, 5000) == 0);     // strictly inside the system prompt
    CHECK(gs::prefix_root_row(1025, 5000) == 1024);
    CHECK(gs::prefix_root_row(5000, 4097) == 3072);  // at least 2 rows before the end
    CHECK(gs::prefix_root_row(0, 5000) == 0);
    std::vector<std::uint32_t> system, a, b;
    for (std::uint32_t k = 0; k < 2100; ++k) system.push_back(3000U + k % 977U);
    a = system;
    b = system;
    for (std::uint32_t k = 0; k < 150; ++k) {
        a.push_back(7000U + k);
        b.push_back(9000U + 3U * k);
    }
    const std::uint32_t slots = 2;
    Loopback lb;
    FakeEng e0(slots), e1(slots);
    Pipe pipe;
    e0.pipe = e1.pipe = &pipe;
    e1.rank = 1;
    gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
    gs::Hello h0 = hello_for(slots, 4096, 2), h1 = hello_for(slots, 4096, 2);
    h0.prefix_cache_bytes = h1.prefix_cache_bytes = 1U << 20;
    std::string follow_error, lead_error, driver_error;
    std::thread follower([&] {
        try { gs::follow(e1, c1, h1, slots, 4096); } catch (const std::exception& e) { follow_error = e.what(); }
    });
    std::atomic<bool> stop{false};
    sv::GenerationQueue queue(16);
    gs::LeadConfig lc;
    lc.prefix_cache_bytes = h0.prefix_cache_bytes;
    lc.slots = slots;
    lc.capacity = 4096;
    lc.heartbeat_seconds = 0.01;
    lc.interleave.select = sv::PrefillInterleave::Select::off;
    try {
        c0.hello(h0);
        gs::Lead<FakeEng> lead(e0, c0, queue, lc, stop);
        std::thread driver([&] {
            try {
                const auto run = [&](const std::vector<std::uint32_t>& prompt, std::uint32_t stable) {
                    sv::GenerationJob job;
                    job.prompt = prompt;
                    job.stable_prefix = stable;
                    job.max_tokens = 12;
                    job.stop_tokens = gs::default_stop_ids();
                    auto t = queue.submit(std::move(job));
                    if (!t) throw std::runtime_error("root: queue refused a job");
                    std::vector<std::uint32_t> out;
                    for (bool done = false; !done;) {
                        const auto got = t->wait(std::chrono::milliseconds(200), done);
                        out.insert(out.end(), got.begin(), got.end());
                    }
                    return out;
                };
                const auto producer = run(a, 2100);
                auto s = lead.stats();
                CHECK(s.prefix_root_stores == 1 && s.prefix_root_refused == 0 && s.prefix_stores == 2 && s.prefix_hits == 0);
                const auto sibling = run(b, 2100);
                s = lead.stats();
                CHECK(s.prefix_hits == 1 && s.prefix_hit_tokens_prompt == 2048);  // restored at the root
                CHECK(s.prefix_root_stores == 1 && s.prefix_stores == 3);         // reached the root: no second root
                lead.clear_prefix_cache();
                const auto sibling_cold = run(b, 0);  // no stable prefix: no root, cold prefill
                const auto producer_cold = run(a, 0);
                s = lead.stats();
                CHECK(s.prefix_root_stores == 1 && s.prefix_hits == 1);
                CHECK(!sibling.empty() && sibling == sibling_cold);
                CHECK(!producer.empty() && producer == producer_cold);
            } catch (const std::exception& e) {
                driver_error = e.what();
            }
            stop.store(true);
        });
        try {
            lead.loop();
        } catch (const std::exception& e) {
            lead_error = e.what();
            queue.shutdown();
        }
        driver.join();
    } catch (const std::exception& e) {
        lead_error = e.what();
    }
    follower.join();
    CHECK(lead_error.empty() && follow_error.empty() && driver_error.empty());
    if (!lead_error.empty() || !follow_error.empty() || !driver_error.empty())
        std::printf("root: lead '%s' follow '%s' driver '%s'\n", lead_error.c_str(), follow_error.c_str(), driver_error.c_str());
    bool restored_at_root = false;
    for (const auto& l : e0.log) restored_at_root = restored_at_root || l.find("@2048") != std::string::npos;
    CHECK(restored_at_root);
    std::printf("prefix root sibling: %s\n", failures == 0 ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    test_sampled_mtp_replay();
    test_sampled_mtp_serve();
    test_sampled_mtp_stop();
    test_sampled_drafts_serve();
    if (argc == 2 && std::string(argv[1]) == "--prefix-root") {
        prefix_root_sibling();
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--audit656-import") {
        import_route_receipts();
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && (std::string(argv[1]) == "--audit601-geometry" || std::string(argv[1]) == "--audit601-import")) {
        audit601_regressions(std::string(argv[1]) == "--audit601-import");
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--interleave-only") {
        queued_successor_interleave();
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--image-only") {
        image_prompts();
        return failures == 0 ? 0 : 1;
    }
    const bool prefix_only = argc == 2 && std::string(argv[1]) == "--prefix-only";
    image_prompts();
    prefix_root_sibling();
    // ---- admission order: LeadConfig::short_first feeds SessionRuntime::reserve_fifo(wait, short_first), which takes
    // from the shared queue with this selection; off keeps FIFO
    {
        CHECK(gs::LeadConfig{}.short_first);
        for (const bool on : {true, false}) {
            strix::serve::GenerationQueue q;
            strix::serve::GenerationJob lng, shrt;
            lng.prompt.resize(2000);
            shrt.prompt.resize(10);
            const auto tl = q.submit(std::move(lng));
            const auto ts = q.submit(std::move(shrt));
            auto first = q.take(false, on);
            CHECK(first && first->ticket == (on ? ts : tl));
        }
    }
    // ---- prefix key semantics, byte LRU, full-token collision checks and admission parsing
    {
        auto entry = [](std::uint64_t id, std::vector<std::uint32_t> ids, bool mtp = false, std::uint64_t identity = 7) {
            gs::PrefixCache::Entry e;
            e.id = id; e.ids = std::move(ids);
            e.mtp = mtp; e.identity = identity; e.frontier = e.ids.size();
            e.state.assign(16, 42);
            return e;
        };
        const auto one = [](std::size_t key) { return strix::serve::ConversationKeep::charged_bytes(16, key); };
        gs::PrefixCache c(4096);
        CHECK(c.prepare(entry(1, {1, 2}))); c.commit(true);
        CHECK(c.prepare(entry(2, {1, 2, 3}))); c.commit(true);
        const std::vector<std::uint32_t> p{1, 2, 3, 4};
        CHECK(c.find(p, false, 7)->id == 2);
        CHECK(c.find(std::span(p).first(3), false, 7)->id == 1);  // exact key forbidden
        CHECK(!c.find(std::span(p).first(2), false, 7));
        CHECK(!c.find(p, true, 7) && !c.find(p, false, 8));
        CHECK(c.prepare(entry(3, {1, 2, 3}, true))); c.commit(true);
        CHECK(c.prepare(entry(4, {1, 2}, true))); c.commit(true);
        CHECK(c.find(p, true, 7)->id == 3 && c.find(p, true, 7)->frontier == 3);  // MTP key = the prompt; restore at |A|
        // a follow-up whose token after the prompt is not the first generated one (template turn) still restores
        CHECK(c.find(std::vector<std::uint32_t>{1, 2, 9, 4}, true, 7)->id == 4);
        CHECK(c.find(std::span(p).first(3), true, 7)->id == 4);  // exact MTP key forbidden: one token must remain
        CHECK(c.state(3) && c.state(3)->size() == 16 && !c.state(99));
        // a weak hash makes every key collide: only the full token compare decides
        gs::PrefixCache weak(4096, [](std::uint64_t, std::span<const std::uint32_t>) -> std::uint64_t { return 5; });
        CHECK(weak.prepare(entry(1, {9, 9, 9}))); weak.commit(true);
        CHECK(weak.prepare(entry(2, {1, 2, 3}))); weak.commit(true);
        CHECK(weak.find(p, false, 7)->id == 2 && !weak.find(std::vector<std::uint32_t>{9, 9, 8, 1}, false, 7));
        gs::PrefixCache lru(one(2) * 2);
        CHECK(lru.prepare(entry(1, {1, 2}))); lru.commit(true);
        CHECK(lru.prepare(entry(2, {2, 3}))); lru.commit(true);
        lru.touch(1);
        // A full grant refuses before staging; explicit paired pressure reclaim makes room.
        CHECK(!lru.prepare(entry(3, {3, 4})));
        CHECK(lru.oldest() == 2 && lru.erase(2));
        CHECK(lru.prepare(entry(3, {3, 4})));
        std::vector<std::uint64_t> victims;
        CHECK(lru.victims(victims) && victims.empty() && lru.pending_fits());
        for (const auto v : victims) CHECK(lru.erase(v));
        lru.commit(true);
        CHECK(lru.get(1) && !lru.get(2) && !lru.state(2) && lru.get(3) && lru.stats().evictions == 1);
        auto huge = entry(4, {4, 5}); huge.state.resize(4096);
        CHECK(!lru.prepare(std::move(huge)) && lru.bytes() == one(2) * 2);
        lru.set_budget(one(2) * 3);
        CHECK(lru.prepare(entry(5, {5, 6}))); lru.commit(false);
        CHECK(!lru.get(5) && lru.get(1));
        CHECK(lru.prepare(entry(6, {1, 2})));
        CHECK(lru.replacement_id() == 1 && lru.victims(victims) && victims == std::vector<std::uint64_t>{1});
        CHECK(lru.erase(1)); lru.commit(true);
        CHECK(!lru.get(1) && lru.get(6) && lru.get(3) && lru.bytes() == one(2) * 2);
        CHECK(gs::prefix_budget_bytes("0") == 0 && gs::prefix_budget_bytes("0.5") == (1ULL << 29));
        for (const auto* v : {"-1", "nan", "inf", "1x", "17179869184", "1e-20"})
            CHECK(throws_with([&] { (void)gs::prefix_budget_bytes(v); }, "--prefix-cache-gib"));
        for (const auto* v : {"", "MemAvailable: -1 kB", "MemAvailable: bad kB", "MemAvailable: 12 MB"}) {
            std::istringstream in(v); CHECK(!gs::prefix_memory_bytes(in));
        }
        // Store rule = MemAvailable >= snapshot bound + the GLM load admission's RAM margin (8 GiB default).
        constexpr std::uint64_t GiB = 1ULL << 30, MiB = 1ULL << 20;
        CHECK(gs::kPrefixMemoryMargin == 8 * GiB);
        CHECK(gs::prefix_snapshot_bound(0) == 152'633'400ULL);
        const auto bound2k = gs::prefix_snapshot_bound(1994);
        CHECK(bound2k == 192'449'592ULL);
        CHECK(gs::prefix_snapshot_bound(UINT64_MAX) == UINT64_MAX);
        // Measured on the GLM 4.05 bpw pair (rank 1 serving, 2K prompt): MemAvailable 14976 MiB. The former fixed
        // 16 GiB floor refused every store there; the snapshot (< 200 MiB) plus the margin fits.
        std::istringstream rank1("MemFree: 0 kB\nMemAvailable: 15335424 kB\n");
        const auto avail = gs::prefix_memory_bytes(rank1);
        CHECK(avail && *avail == 14976 * MiB && bound2k < 200 * MiB);
        CHECK(gs::prefix_store_fits(avail, bound2k, gs::kPrefixMemoryMargin));
        CHECK(gs::prefix_store_fits(8 * GiB + bound2k, bound2k, 8 * GiB));
        CHECK(!gs::prefix_store_fits(8 * GiB + bound2k - 1, bound2k, 8 * GiB));
        CHECK(!gs::prefix_store_fits(std::nullopt, 1, 0));               // no reading: refused
        CHECK(!gs::prefix_store_fits(GiB, 2 * GiB, 0));                  // bound above MemAvailable: no wrap
        CHECK(!gs::prefix_store_fits(UINT64_MAX, UINT64_MAX, 1));
        // Pressure rule: only below the margin, by the missing bytes; a store the rule admitted never causes it.
        CHECK(gs::prefix_pressure_shortfall(std::nullopt, 8 * GiB) == 0);
        CHECK(gs::prefix_pressure_shortfall(8 * GiB, 8 * GiB) == 0);
        CHECK(gs::prefix_pressure_shortfall(*avail, 8 * GiB) == 0);      // the measured rank keeps its entries
        CHECK(gs::prefix_pressure_shortfall(7 * GiB, 8 * GiB) == GiB);
        for (const std::uint64_t a : {8 * GiB + bound2k, 9 * GiB, *avail, 64 * GiB})
            if (gs::prefix_store_fits(a, bound2k, 8 * GiB)) CHECK(gs::prefix_pressure_shortfall(a - bound2k, 8 * GiB) == 0);
        auto hello = hello_for(2, 4096, 2), peer = hello;
        peer.prefix_cache_bytes = 1;
        CHECK(gs::hello_mismatch(hello, peer).find("prefix_cache_bytes") != std::string::npos);
    }
    {
        // A grant holding only one snapshot refuses new stores without a transient over-allocation.
        const auto r = run_pair(3, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                                nullptr, 0, false, false, 0, true, false, 256);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.stats.prefix_stores > 0 && r.stats.prefix_evictions == 0 && r.stats.prefix_bytes <= 256 && r.log0 == r.log1);
        // Native prefix caching must not replace the optional imported numerical identity.
        const auto imported = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                                       nullptr, 1, false, false, 0, true);
        CHECK(imported.lead_error.empty() && imported.follow_error.empty() && imported.driver_error.empty());
        CHECK(imported.stats.imports > 0 && imported.stats.prefix_hits == 0 && imported.stats.prefix_stores == 0);
        // Nor may the opt-in reply-end store keep an imported slot under the native identity: a later native-route
        // request (import_route false, e.g. while another import is in flight) would restore FAST import rows.
        const auto reply = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                                    nullptr, 1, false, false, 0, true, false, 1ULL << 20, true);
        CHECK(reply.lead_error.empty() && reply.follow_error.empty() && reply.driver_error.empty());
        CHECK(reply.stats.imports > 0 && reply.stats.prefix_hits == 0 && reply.stats.prefix_stores == 0 &&
              reply.stats.prefix_stores_reply == 0 && reply.stats.prefix_reply_skipped > 0);
    }
    // Missing peer key and a local restore exception must be observed on BOTH ranks before suffix collectives.
    for (const bool missing : {false, true}) {
        Loopback lb;
        gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
        FakeEng e0(1), e1(1);
        gs::PrefixCache k0(4096), k1(4096);
        gs::PrefixCache::Entry e;
        e.id = 1; e.identity = 7; e.frontier = 2; e.ids = {11, 22};
        e0.reset(0); (void)e0.prefill(0, e.ids, 0);
        e.state = e0.state_bytes_upto(0, 2);
        CHECK(k0.prepare(e)); k0.commit(true);
        if (!missing) { CHECK(k1.prepare(e)); k1.commit(true); e1.fail_prefix_restore = true; }
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::prefix_restore);
        c.prefix_id = 1; c.prefix_identity = 7; c.ppos = 2;
        bool failed0 = false, failed1 = false;
        std::thread peer([&] {
            gs::Control reply;
            const auto cmd = c1.exchange(reply);
            std::vector<std::uint32_t> p;
            failed1 = throws_with([&] { (void)gs::replay(e1, c1, cmd, p, false, 1, 64, &k1); }, "paired restore failed");
        });
        (void)c0.exchange(c);
        std::vector<std::uint32_t> p;
        failed0 = throws_with([&] { (void)gs::replay(e0, c0, c, p, true, 1, 64, &k0); }, "paired restore failed");
        peer.join();
        CHECK(failed0 && failed1 && e1.prefills == 0);
    }
    // PC2 through the actual lead/follower replay, plain and MTP/G2; independent rank-1 refusal must discard both.
    for (const bool mtp : {false, true}) {
        auto r = run_pair(3, 4, true, 16, 0, hello_for(4, 4096, 2), mtp, 7, 0, 0, false, mtp,
                          nullptr, 0, false, false, 0, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.stats.prefix_hits >= 3 && r.stats.prefix_stores > 0);
        CHECK(r.log0 == r.log1);
        auto f = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), mtp, 7, 0, 0, false, mtp,
                          nullptr, 0, false, false, 0, true, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(!f.result.exact && f.stats.prefix_hits == 0 && f.stats.prefix_stores == 0);
        // the lead counts the peer's refusal (it was invisible before: no keep-level refusal on either rank)
        CHECK(r.stats.prefix_refused == 0 && f.stats.prefix_refused > 0);
    }

    // A prompt-only run must fail the reply gate even when prompt hits and output equality succeed.
    for (const bool refuse_reply : {false, true}) {
        auto r = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                          nullptr, 0, false, false, 0, true, false, 1ULL << 20, true, refuse_reply);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.stats.prefix_hits > 0);
        CHECK(r.result.exact == !refuse_reply);
        CHECK(refuse_reply ? r.stats.prefix_hits_reply == 0 : r.stats.prefix_hits_reply > 0);
    }

    {
        auto r = run_pair(2, 4, true, 16, 0, hello_for(4, 4096, 2), false, 7, 0, 0, false, false,
                          nullptr, 0, false, false, 0, true, false, 1ULL << 20, false, false, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.follower_snapshot_copies == 0 && r.stats.prefix_stores == 0 && !r.result.exact && r.stats.prefix_refused > 0);
    }

    if (prefix_only) return failures ? 1 : 0;

    // ---- import commit votes: both ranks fail together on any rank's failure, binding or frontier mismatch
    {
        std::printf("import commit votes:\n");
        auto vote_pair = [](bool ok0, bool ok1, std::uint8_t salt1, std::uint64_t f1) {
            Pipe pipe;
            gs::Binding b0{}, b1{};
            b0[0] = b1[0] = 42;
            b1[1] = salt1;
            std::string e[2];
            std::thread t([&] {
                try {
                    gs::import_vote("prepare", ok1, "local", b1, f1, [&](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe.xchg(1, a, r); }, 1);
                } catch (const std::exception& x) { e[1] = x.what(); }
            });
            try {
                gs::import_vote("prepare", ok0, "local", b0, 100, [&](std::span<const std::byte> a, std::span<std::byte> r, std::uint64_t) { pipe.xchg(0, a, r); }, 1);
            } catch (const std::exception& x) { e[0] = x.what(); }
            t.join();
            return std::pair<std::string, std::string>{e[0], e[1]};
        };
        {
            const auto [a, b] = vote_pair(true, true, 0, 100);
            CHECK(a.empty() && b.empty());
        }
        {
            const auto [a, b] = vote_pair(true, false, 0, 100);
            CHECK(a.find("peer rank failed") != std::string::npos && b.find("this rank: local") != std::string::npos);
        }
        {
            const auto [a, b] = vote_pair(true, true, 7, 100);  // different binding: both refuse
            CHECK(a.find("different hand-offs") != std::string::npos && b.find("different hand-offs") != std::string::npos);
        }
        {
            const auto [a, b] = vote_pair(true, true, 0, 99);  // different frontier: both refuse
            CHECK(a.find("different hand-offs") != std::string::npos && b.find("different hand-offs") != std::string::npos);
        }
        // the transaction: a one-rank restore failure aborts both before the suffix (no rank alone in a collective);
        // a failure inside the suffix is ImportFatal (the engine's fatal contract), never an import failure.
        auto txn = [](bool fail_restore1, bool fail_suffix0) {
            Pipe pipe;
            const gs::Binding b{};
            std::string e[2];
            bool fatal[2] = {false, false};
            unsigned suffix[2] = {0, 0};
            auto run = [&](unsigned r) {
                try {
                    (void)gs::import_transaction(
                        true, "", b, 10, [&](std::span<const std::byte> a, std::span<std::byte> x, std::uint64_t) { pipe.xchg(r, a, x); },
                        [] {}, [&] { if (r == 1 && fail_restore1) throw std::runtime_error("restore broke"); },
                        [&]() -> std::uint32_t {
                            ++suffix[r];
                            if (r == 0 && fail_suffix0) throw std::runtime_error("collective broke");
                            return 5;
                        });
                } catch (const gs::ImportFatal& x) {
                    fatal[r] = true;
                    e[r] = x.what();
                } catch (const std::exception& x) {
                    e[r] = x.what();
                }
            };
            std::thread t([&] { run(1); });
            run(0);
            t.join();
            return std::make_tuple(e[0], e[1], fatal[0], fatal[1], suffix[0], suffix[1]);
        };
        {
            const auto [a, b, f0, f1, s0, s1] = txn(true, false);
            CHECK(a.find("restore: the peer rank failed") != std::string::npos && b.find("restore: this rank: restore broke") != std::string::npos);
            CHECK(!f0 && !f1 && s0 == 0 && s1 == 0);
            std::printf("  rank 1 restore fails: both abort before the suffix (rank 0: %s)\n", a.c_str());
        }
        {
            const auto [a, b, f0, f1, s0, s1] = txn(false, true);
            CHECK(f0 && !f1 && a.find("fatal") != std::string::npos && b.empty() && s0 == 1 && s1 == 1);
        }
        {
            const auto [a, b, f0, f1, s0, s1] = txn(false, false);
            CHECK(a.empty() && b.empty() && s0 == 1 && s1 == 1);
        }
    }
#ifndef _WIN32
    // ---- import receiver: a peer trickling bytes is cut by cancellation and by the aggregate deadline
    {
        std::printf("import receiver, slow-drip peer:\n");
        namespace gi = strix::glm::import;
        const gi::Bytes key(32, std::byte{7});
        gi::Begin expect;
        expect.frontier = 1;
        expect.total = 2;
        for (const int mode : {0, 1}) {  // 0: cancelled after 400 ms, 1: aggregate deadline 1 s
            const std::string ep = mode == 0 ? "127.0.0.1:47991" : "127.0.0.1:47992";
            std::atomic<bool> cancel{false}, bound{false}, stop_drip{false};
            std::string err;
            double secs = 0;
            std::thread rx([&] {
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    (void)gi::receive(ep, key, expect, &cancel, [&] { bound.store(true); }, mode == 0 ? 900U : 1U);
                } catch (const std::exception& e) {
                    err = e.what();
                }
                secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
            for (int k = 0; k < 200 && !bound.load(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<std::uint16_t>(mode == 0 ? 47991 : 47992));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool connected = bound.load() && ::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0;
            CHECK(connected);
            std::thread drip([&] {  // reads the nonce, then one byte of the length field every 150 ms
                char nonce[32];
                std::size_t got = 0;
                while (connected && got < sizeof nonce) {
                    const auto n = ::recv(fd, nonce + got, sizeof nonce - got, 0);
                    if (n <= 0) break;
                    got += static_cast<std::size_t>(n);
                }
                const char one = 0;
                for (int k = 0; k < 40 && !stop_drip.load(); ++k) {
                    if (k < 3) (void)::send(fd, &one, 1, MSG_NOSIGNAL);  // 3 of the 4 length bytes, then nothing more
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
            });
            if (mode == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                cancel.store(true);
            }
            rx.join();
            stop_drip.store(true);
            drip.join();
            ::close(fd);
            CHECK(mode == 0 ? err.find("cancelled") != std::string::npos : err.find("deadline") != std::string::npos);
            CHECK(secs < 3.0);
            std::printf("  %s: '%s' after %.2f s\n", mode == 0 ? "cancel" : "deadline 1 s", err.c_str(), secs);
        }
    }
#endif
    // ---- 'DSA once' relay protocol: rank 0's DSA components reach rank 1 in chunks, merged in layout order
    {
        std::printf("dsa relay protocol:\n");
        const auto run = [](bool ok0, bool ok1, std::size_t chunk, std::vector<std::uint8_t> mine0) {
            Pipe pipe;
            std::vector<std::uint8_t> out[2];
            std::string err[2];
            gs::RelayStats st[2];
            const std::vector<std::uint8_t> mine[2] = {std::move(mine0), relay_part(false, 1)};
            std::thread t1([&] {
                try {
                    out[1] = gs::dsa_relay(1, ok1, mine[1], kRelayLayout, chunk,
                                           [&](std::span<const std::byte> a, std::span<std::byte> b, std::uint64_t) { pipe.xchg(1, a, b); }, &st[1]);
                } catch (const std::exception& e) {
                    err[1] = e.what();
                }
            });
            try {
                out[0] = gs::dsa_relay(0, ok0, mine[0], kRelayLayout, chunk,
                                       [&](std::span<const std::byte> a, std::span<std::byte> b, std::uint64_t) { pipe.xchg(0, a, b); }, &st[0]);
            } catch (const std::exception& e) {
                err[0] = e.what();
            }
            t1.join();
            return std::make_tuple(out[0], out[1], err[0], err[1], st[1]);
        };
        {  // chunks smaller than the components, a partial last chunk: rank 1 ends with its KDA + rank 0's DSA, in order
            const auto [o0, o1, e0, e1, s1] = run(true, true, 64, relay_full(0));
            CHECK(e0.empty() && e1.empty() && o0.empty() && o1 == relay_full(1));
            CHECK(s1.bytes == 407 && s1.chunks == 7);
            CHECK(gs::gather_dsa(relay_full(1), kRelayLayout) == relay_part(true, 0));
            std::printf("  407 DSA bytes in %llu chunks of <= 64: rank 1 == its full state: %s\n",
                        static_cast<unsigned long long>(s1.chunks), o1 == relay_full(1) ? "yes" : "no");
        }
        {  // one chunk
            const auto [o0, o1, e0, e1, s1] = run(true, true, 1U << 20, relay_full(0));
            CHECK(e0.empty() && e1.empty() && o1 == relay_full(1) && s1.chunks == 1);
        }
        {  // rank 1 has nothing usable: both fail at the header (neither waits alone)
            const auto [o0, o1, e0, e1, s1] = run(true, false, 64, relay_full(0));
            CHECK(!e0.empty() && !e1.empty() && o1.empty());
            std::printf("  rank 1 not ready: rank 0 '%s' / rank 1 '%s'\n", e0.c_str(), e1.c_str());
        }
        {  // rank 0's staged payload has the wrong size: both fail
            auto bad = relay_full(0);
            bad.pop_back();
            const auto [o0, o1, e0, e1, s1] = run(true, true, 64, bad);
            CHECK(!e0.empty() && !e1.empty());
        }
    }
    // ---- encode / decode
    {
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::step);
        c.nrows = 2;
        c.row[0] = {0, 5, 10};
        c.row[1] = {1, 6, 11};
        c.resets = 0x5;
        c.hello = hello_for(8, 8192, 2);
        const auto b = gs::encode(c);
        const auto d = gs::decode(b);
        CHECK(d.cmd == c.cmd && d.nrows == 2 && d.row[1] == c.row[1] && d.resets == 0x5 && d.hello == c.hello);
        auto bad = b;
        bad[0] = std::byte{0};
        CHECK(throws_with([&] { (void)gs::decode(bad); }, "bad magic"));
        gs::Control v = c;
        v.version = 99;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(v)); }, "protocol version"));
        gs::Control r = c;
        r.nrows = 9;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(r)); }, "more than 8 rows"));
        auto pad = b;
        pad[gs::kControlBytes - 1] = std::byte{1};
        CHECK(throws_with([&] { (void)gs::decode(pad); }, "padding"));
        gs::Control k = c;
        k.cmd = 77;
        CHECK(throws_with([&] { (void)gs::decode(gs::encode(k)); }, "unknown command"));
    }
    // ---- hello, rows
    {
        const auto a = hello_for(8, 8192, 2);
        auto b = a;
        CHECK(gs::hello_mismatch(a, b).empty());
        b.slots = 4;
        b.identity_hash = 1;
        const auto m = gs::hello_mismatch(a, b);
        CHECK(m.find("slots") != std::string::npos && m.find("identity_hash") != std::string::npos);
        gs::Hello c = a;
        c.route_fingerprint = a.route_fingerprint + 1U;  // the peer resolved another numerical route
        CHECK(gs::hello_mismatch(a, c).find("route_fingerprint") != std::string::npos);
        // RTX import settings: receiver on / 'DSA once', wire version and the attested model SHA must agree on every rank
        {
            auto i0 = a;
            i0.import_flags = 3;
            i0.import_wire = 2;
            i0.import_model_hash = gs::fnv1a64(std::string("ab12"));
            auto i1 = i0;
            CHECK(gs::hello_mismatch(i0, i1).empty());
            i1.import_flags = 1;  // the peer has no 'DSA once'
            i1.import_wire = 1;
            CHECK(gs::hello_mismatch(i0, i1).find("import_flags") != std::string::npos &&
                  gs::hello_mismatch(i0, i1).find("import_wire") != std::string::npos);
            i1 = i0;
            i1.import_model_hash = gs::fnv1a64(std::string("cd34"));  // another attested model
            CHECK(gs::hello_mismatch(i0, i1).find("import_model_hash") != std::string::npos);
            i1 = a;  // the peer has no import receiver at all
            CHECK(gs::hello_mismatch(i0, i1).find("import_flags") != std::string::npos);
            // the fields travel in the control block: a mismatch over the loopback refuses on both ranks at hello
            Loopback lb;
            gs::Channel c0(lb.side(0), 0), c1(lb.side(1), 1);
            std::string e1;
            std::thread t([&] {
                try {
                    c1.hello(i1);
                } catch (const std::exception& e) {
                    e1 = e.what();
                }
            });
            std::string e0;
            try {
                c0.hello(i0);
            } catch (const std::exception& e) {
                e0 = e.what();
            }
            t.join();
            CHECK(e0.find("import_flags") != std::string::npos && e1.find("import_flags") != std::string::npos);
            std::printf("hello with different import settings: refused on both ranks (%s)\n", e0.c_str());
            // the attested model SHA is hashed canonically: the same digest in upper / lower case agrees
            const std::string lower = "d4b733a5d05d587c65a5a0f3d73b65c6bd8f62cb6fc2b6e48f6646826e3d1d3a";
            std::string upper = lower;
            for (auto& ch : upper) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            CHECK(gs::attested_model_hash(lower) == gs::attested_model_hash(upper) && upper != lower);
            CHECK(gs::attested_model_hash(lower) != gs::attested_model_hash(std::string(64, '0')));
            CHECK(throws_with([] { (void)gs::attested_model_hash("abc"); }, "64 hex digits"));
            CHECK(throws_with([&] { (void)gs::attested_model_hash(std::string(63, 'a') + "g"); }, "not a hex digit"));
        }
        // a peer with the previous (v5) block layout is refused at the version check, before any later field is read
        {
            gs::Control old_peer;
            old_peer.cmd = static_cast<std::uint32_t>(gs::Cmd::heartbeat);
            auto blk = gs::encode(old_peer);
            const std::uint32_t v5 = 5;
            std::memcpy(blk.data() + offsetof(gs::Control, version), &v5, sizeof v5);
            CHECK(throws_with([&] { (void)gs::decode(blk); },
                              "protocol version 5 from the peer, expected " + std::to_string(gs::kControlVersion)));
            CHECK(gs::decode(gs::encode(old_peer)).version == gs::kControlVersion);
            std::printf("v5-layout peer block: refused at the version check\n");
        }
        // Builds of earlier layouts (v13: no stop decision in a sampled window; v12: the 10-word sampled decision; v11: the
        // contributor's; v9: main before this change) are refused at the first exchange on either rank, before any later
        // field is read.
        for (const std::uint32_t old : {13U, 12U, 11U, 9U})
            for (const std::uint32_t rank : {0U, 1U}) {
                gs::Channel ch(
                    [&](std::span<const std::byte> out, std::span<std::byte> in, std::uint32_t, std::uint64_t) {
                        std::memcpy(in.data(), out.data(), in.size());
                        std::memcpy(in.data() + offsetof(gs::Control, version), &old, sizeof old);
                        const std::uint32_t peer_rank = rank ^ 1U;
                        std::memcpy(in.data() + offsetof(gs::Control, rank), &peer_rank, sizeof peer_rank);
                    },
                    rank);
                gs::Control hello;
                hello.cmd = static_cast<std::uint32_t>(gs::Cmd::heartbeat);
                CHECK(throws_with([&] { (void)ch.exchange(hello); },
                                  "protocol version " + std::to_string(old) + " from the peer, expected 14"));
            }
        std::printf("v13 / v12 / v11 / v9 peers: refused at the first exchange on rank 0 and rank 1\n");
        std::vector<gs::WireRow> rows = {{0, 1, 3}, {1, 1, 3}, {2, 1, 9}};
        CHECK(gs::consecutive_slots(rows));
        gs::check_rows(rows, 8, 16);
        rows[2].slot = 4;
        CHECK(!gs::consecutive_slots(rows));
        rows[2].slot = 1;
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "ascending"));
        rows[2] = {9, 1, 0};
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "slot out of range"));
        rows[2] = {3, 1, 16};
        CHECK(throws_with([&] { gs::check_rows(rows, 8, 16); }, "capacity"));
        CHECK(throws_with([&] { gs::check_rows({}, 8, 16); }, "1..8"));
    }
    // ---- decide, slot table
    {
        const auto& st = gs::default_stop_ids();
        CHECK(gs::decide(154827U, 1, 10, 5, 100, st) == gs::Next::stop);
        CHECK(gs::decide(42U, 10, 10, 5, 100, st) == gs::Next::length);
        CHECK(gs::decide(42U, 3, 10, 100, 100, st) == gs::Next::context);
        CHECK(gs::decide(42U, 3, 10, 99, 100, st) == gs::Next::decode);
        gs::SlotTable t(4);
        CHECK(*t.alloc() == 0 && *t.alloc() == 1 && *t.alloc() == 2);
        t.release(1);
        CHECK(*t.alloc() == 1);
        CHECK(*t.alloc() == 3 && !t.alloc() && t.full());
        t.release(0);
        CHECK(t.live() == 3 && *t.alloc() == 0);
    }
    // ---- channel: rounds, tokens, echo (world 1 mirror and loopback)
    {
        gs::Channel solo({}, 0);
        solo.hello(hello_for(1, 64, 1));
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::heartbeat);
        (void)solo.exchange(c);
        CHECK(solo.round() == 2);
        Loopback lb;
        gs::Channel a(lb.side(0), 0), b(lb.side(1), 1);
        std::vector<std::uint32_t> big(40000), got;
        for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<std::uint32_t>(i * 7);
        std::thread t([&] { b.tokens(got, big.size(), false); });
        a.tokens(big, big.size(), true);
        t.join();
        CHECK(got == big);
    }
    // ---- import_state: replay restores the staged prefix and runs the last token; == the slot's own prefill
    {
        const std::vector<std::uint32_t> prompt{11, 22, 33, 44, 55};
        for (const bool mtp : {false, true}) {
            FakeEng own(2), imp(2);
            own.reset(1);
            const auto y_own = mtp ? own.prefill_mtp(1, prompt, 0, true) : own.prefill(1, prompt, 0);
            imp.stage(7, std::span<const std::uint32_t>(prompt).first(prompt.size() - 1));
            imp.expected_id = 7;
            gs::Channel solo({}, 0);
            solo.hello(hello_for(2, 64, 1));
            gs::Control c;
            c.cmd = static_cast<std::uint32_t>(gs::Cmd::import_state);
            c.pslot = 1;
            c.resets = 1U << 1;
            c.plen = static_cast<std::uint32_t>(prompt.size());
            c.import_id = 7;
            c.pflags = mtp ? gs::kPrefillMtp : 0U;
            CHECK(gs::decode(gs::encode(c)).import_id == 7);
            auto p = prompt;
            const auto out = gs::replay(imp, solo, c, p, true, 2, 64);
            CHECK(out.size() == 1 && out[0] == y_own && imp.h[1] == own.h[1] && imp.frontier[1] == own.frontier[1]);
            CHECK(!mtp || (imp.seed && imp.seed->y0 == y_own && imp.seed->p == prompt.size()));
            CHECK(imp.import_report().status == static_cast<std::uint32_t>(gs::ImportStatus::restored) && imp.import_report().token == y_own);
            c.import_id = 8;  // nothing staged under it: the rank's failure is local (slot reset, kImportFailed, reported)
            auto q = prompt;
            imp.expected_id = 8;
            const auto bad = gs::replay(imp, solo, c, q, true, 2, 64);
            CHECK(bad.size() == 1 && bad[0] == gs::kImportFailed && imp.abort_why.find("nothing staged") != std::string::npos);
            CHECK(imp.frontier[1] == 0 && imp.ready[1] && !imp.seed);
            CHECK(imp.import_report().status == static_cast<std::uint32_t>(gs::ImportStatus::failed));
            c.import_id = 7;
            c.resets = 0;  // the same command must reset the slot
            CHECK(throws_with([&] { (void)gs::replay(imp, solo, c, q, true, 2, 64); }, "without its slot reset"));
        }
        FakeEng short_one(2);
        short_one.stage(3, std::vector<std::uint32_t>{1, 2});
        gs::Control c;
        c.cmd = static_cast<std::uint32_t>(gs::Cmd::import_state);
        c.pslot = 0;
        c.resets = 1;
        c.plen = 4;  // staged frontier 2 != 4 - 1
        c.import_id = 3;
        gs::Channel solo({}, 0);
        solo.hello(hello_for(2, 64, 1));
        std::vector<std::uint32_t> p{1, 2, 3, 4};
        CHECK(gs::replay(short_one, solo, c, p, true, 2, 64).at(0) == gs::kImportFailed && short_one.abort_why.find("staged frontier") != std::string::npos);
    }
    // ---- ImportStager : receives off the caller's thread; ready / failed per id; payload only when ready
    {
        std::mutex gm;
        std::condition_variable gcv;
        bool go = false;
        gs::ImportStager st([&](std::uint32_t id, std::span<const std::uint32_t> t, const std::atomic<bool>& cancel,
                                const std::function<void()>& listening) {
            if (id == 3) {  // never connected: waits until cancelled
                listening();
                while (!cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                throw std::runtime_error("import cancelled");
            }
            std::unique_lock<std::mutex> l(gm);
            if (id == 1) listening();
            gcv.wait(l, [&] { return go; });
            if (id == 2) throw std::runtime_error("model hash mismatch");
            return std::vector<std::uint8_t>(t.size() * 3, static_cast<std::uint8_t>(id));
        });
        st.expect(1, {5, 6, 7});
        st.expect(2, {5, 6});
        for (int i = 0; i < 2000 && st.status(1) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(1) == gs::ImportStatus::listening && st.status(2) == gs::ImportStatus::pending && st.status(9) == gs::ImportStatus::none);
        CHECK(throws_with([&] { (void)st.payload(1); }, "nothing staged"));
        CHECK(throws_with([&] { st.expect(1, {1, 2}); }, "already expected"));
        {
            std::lock_guard<std::mutex> l(gm);
            go = true;
        }
        gcv.notify_all();
        for (int i = 0; i < 2000 &&
             (st.status(1) == gs::ImportStatus::listening || st.status(2) == gs::ImportStatus::pending); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(1) == gs::ImportStatus::ready && st.payload(1).size() == 9 && st.payload(1)[0] == 1);
        CHECK(st.status(2) == gs::ImportStatus::failed && st.error(2).find("model hash") != std::string::npos);
        st.drop(1);
        CHECK(st.status(1) == gs::ImportStatus::none);
        // dropping a receive that never connects returns at once (cancelled, reaped later); its id never reappears
        st.expect(3, {1, 2, 3});
        for (int i = 0; i < 2000 && st.status(3) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto t0 = std::chrono::steady_clock::now();
        st.drop(3);
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50) && st.status(3) == gs::ImportStatus::none);
        for (int i = 0; i < 2000 && st.retired() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st.drop(99);  // reaps
        }
        CHECK(st.retired() == 0 && st.status(3) == gs::ImportStatus::none);
    }
    // ---- drop never stalls the replay; a new import is refused (falls back) while a dropped receive still closes the
    // endpoint, and binds again once that receive has ended
    {
        std::atomic<int> started{0};
        gs::ImportStager st([&](std::uint32_t, std::span<const std::uint32_t>, const std::atomic<bool>& cancel,
                                const std::function<void()>& listening) -> std::vector<std::uint8_t> {
            ++started;
            listening();
            while (!cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::this_thread::sleep_for(std::chrono::milliseconds(300));  // a slow close of the endpoint
            throw std::runtime_error("import cancelled");
        });
        st.expect(1, {1, 2, 3});
        for (int i = 0; i < 2000 && st.status(1) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto t0 = std::chrono::steady_clock::now();
        st.drop(1);
        st.expect(2, {1, 2, 3});  // immediately: the dropped receive still holds the endpoint
        const auto dt = std::chrono::steady_clock::now() - t0;
        CHECK(dt < std::chrono::milliseconds(50));
        CHECK(st.status(2) == gs::ImportStatus::failed && st.error(2).find("still closing") != std::string::npos && started.load() == 1);
        st.drop(2);
        for (int i = 0; i < 3000 && st.retired() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st.drop(99);  // reaps
        }
        CHECK(st.retired() == 0);
        st.expect(3, {1, 2, 3});  // the endpoint is free again: a new receive starts
        for (int i = 0; i < 2000 && st.status(3) != gs::ImportStatus::listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(st.status(3) == gs::ImportStatus::listening && started.load() == 2);
        std::printf("drop + immediate expect: refused in %.1f ms (no stall), next expect after close: %s\n",
                    std::chrono::duration<double, std::milli>(dt).count(), st.status(3) == gs::ImportStatus::listening ? "listening" : "NO");
        st.drop(3);
    }
    // ---- import route over the loopback: imported slots decode exactly as prefilled ones; failures fall back
    for (const bool mtp : {false, true}) {
        std::printf("import route, 6 prompts, 4 slots%s:\n", mtp ? ", MTP" : "");
        const auto r = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 0, 0, 0, false, false, nullptr, 1);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.rows0 == r.rows1);
        CHECK(r.stats.prefills > 0 && r.stats.imports == r.stats.prefills && r.stats.import_fallbacks == 0);  // every prompt imported
        const auto f = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 0, 0, 0, false, false, nullptr, 1, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(f.result.exact && f.stats.prefills > 0 && f.stats.imports == 0 && f.stats.import_fallbacks == f.stats.prefills);
        // both ranks staged, the follower's restore fails: its slot is reset, both fall back, the server keeps serving
        const auto g = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 128, 0, 0, false, false, nullptr, 1, false, true);
        CHECK(g.lead_error.empty() && g.follow_error.empty() && g.driver_error.empty());
        CHECK(g.result.exact && g.stats.prefills > 0 && g.stats.imports == 0 && g.stats.import_fallbacks == g.stats.prefills);
        bool aborted = false, aborted0 = false;
        for (const auto& line : g.log1) aborted = aborted || line.find("(import abort)") != std::string::npos;
        for (const auto& line : g.log0) aborted0 = aborted0 || line.find("(import abort)") != std::string::npos;
        CHECK(aborted && aborted0);  // the restored vote: rank 0 restored but aborts too, before any suffix collective
        // the ranks staged different hand-offs (binding): the prepare vote aborts both, every prompt prefills itself
        const auto m = run_pair(6, 4, true, 40, 0, hello_for(8, 4096, 2), mtp, 128, 0, 0, false, false, nullptr, 1, false, false, 1);
        CHECK(m.lead_error.empty() && m.follow_error.empty() && m.driver_error.empty());
        CHECK(m.result.exact && m.stats.prefills > 0 && m.stats.imports == 0 && m.stats.import_fallbacks == m.stats.prefills);
    }
    // ---- import route beside the chunked prefill: long prompts import (no chunk), short ones prefill in chunks
    for (const bool mtp : {false, true}) {
        std::printf("import route + prefill interleave (chunk 128, long prompts import)%s:\n", mtp ? ", MTP" : "");
        const auto r = run_pair(4, 4, true, 64, 0, hello_for(8, 4096, 2), mtp, 128, 3, 0, false, false, nullptr, 500);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.imports > 0 && r.stats.import_fallbacks == 0 && r.stats.imports < r.stats.prefills);
        bool long_chunked = false, imported = false;
        for (const auto& line : r.log0) {
            long_chunked = long_chunked || line.find(" @128 +128") != std::string::npos;
            imported = imported || line.rfind("import ", 0) == 0;
        }
        CHECK(imported && !long_chunked);  // the long prompts skipped the chunked prefill
        const auto f = run_pair(4, 4, true, 64, 0, hello_for(8, 4096, 2), mtp, 128, 3, 0, false, false, nullptr, 500, true);
        CHECK(f.lead_error.empty() && f.follow_error.empty() && f.driver_error.empty());
        CHECK(f.result.exact && f.log0 == f.log1 && f.stats.imports == 0 && f.stats.import_fallbacks > 0);
        // a failed import prefills on the pair: in chunks while others decode, whole when it is alone (adaptive chunking)
        bool fell_back_prefilled = false;
        for (const auto& line : f.log0) {
            if (line.find(" @128 +128") != std::string::npos) fell_back_prefilled = true;
            const auto at = line.find(" @0 +");
            if (line.rfind("prefill ", 0) == 0 && at != std::string::npos && std::stoul(line.substr(at + 5)) >= 500) fell_back_prefilled = true;
        }
        CHECK(fell_back_prefilled);
    }
    // ---- lead + follower over the loopback
    std::printf("batched, 6 prompts, 4 slots:\n");
    {
        const auto r = run_pair(6, 4, true, 40);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.rows0 == r.rows1);  // the follower made exactly the lead's step calls
        bool multi = false;
        for (const auto n : r.rows0) multi = multi || n > 1;
        CHECK(multi);  // steps really batched rows
        CHECK(r.stats.active == 0 && r.stats.finished > 0);
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
    }
    std::printf("batched, 8 prompts, 8 slots, no early stops in some streams:\n");
    {
        const auto r = run_pair(8, 8, true, 64);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact && r.rows0 == r.rows1);
        for (const auto& l : r.result.levels) {
            if (!l.window.valid) continue;  // early stops can end a stream before the last prefill
            CHECK(l.window.tokens > 0 && l.window.seconds > 0);
            std::uint64_t sum = 0;
            for (const auto n : l.window.per_stream) sum += n;
            CHECK(sum == l.window.tokens && l.window.per_stream.size() == l.streams.size());
            if (l.c > 1) CHECK(l.window.steps == 0 || l.window.rows >= l.window.steps);
        }
    }
    std::printf("serial (OFF switch), 4 prompts:\n");
    {
        const auto r = run_pair(4, 4, false, 24);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact);
        for (const auto n : r.rows0) CHECK(n == 1);
    }
    std::printf("cross-rank mismatch (rank 1 corrupts its 5th step):\n");
    {
        const auto r = run_pair(3, 4, true, 30, 5);
        CHECK(r.lead_error.find("cross-rank token mismatch") != std::string::npos);
        CHECK(r.follow_error.find("cross-rank token mismatch") != std::string::npos);
        CHECK(!r.result.exact);
        std::printf("  lead: %s\n  follower: %s\n", r.lead_error.c_str(), r.follow_error.c_str());
    }
    std::printf("hello mismatch (capacity):\n");
    {
        auto h1 = hello_for(4, 8192, 2);
        const auto r = run_pair(2, 4, true, 8, 0, h1);
        CHECK(r.lead_error.find("hello mismatch") != std::string::npos && r.follow_error.find("hello mismatch") != std::string::npos);
    }
    std::printf("C1 MTP + batched, 6 prompts, 4 slots:\n");
    {
        const auto r = run_pair(6, 4, true, 48, 0, hello_for(8, 4096, 2), true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);  // the follower made exactly the lead's engine calls
        CHECK(r.stats.mtp_rounds > 0 && r.stats.mtp_bootstraps > 0 && r.stats.mtp_enters > 0 && r.stats.mtp_leaves > 0);
        bool multi = false;
        for (const auto x : r.rows0) multi = multi || x > 1;
        CHECK(multi);
        std::printf("  mtp rounds %llu tokens %llu enters %llu leaves %llu, steps %llu\n",
                    static_cast<unsigned long long>(r.stats.mtp_rounds), static_cast<unsigned long long>(r.stats.mtp_tokens),
                    static_cast<unsigned long long>(r.stats.mtp_enters), static_cast<unsigned long long>(r.stats.mtp_leaves),
                    static_cast<unsigned long long>(r.stats.steps));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
    }
    std::printf("C1 MTP, serial, 3 prompts:\n");
    {
        const auto r = run_pair(3, 4, false, 40, 0, hello_for(8, 4096, 2), true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.mtp_rounds > 0);
    }
    std::printf("prefill interleave: chunk 128, long prompts, MTP, 4 slots:\n");
    {
        const auto r = run_pair(4, 4, true, 256, 0, hello_for(8, 4096, 2), true, 128, 4, 200);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);
        bool chunked = false;
        for (const auto& line : r.log0) chunked = chunked || line.find(" @128 +128") != std::string::npos;
        CHECK(chunked);  // prompts really went in chunks at pos0 > 0
        CHECK(r.result.stall_whole_ms > 2.0 * r.result.stall_chunked_ms && r.result.stall_chunked_ms > 0);
        // adaptive: a prompt admitted with nothing decoding goes in one call (the first long C=1 runs)
        bool whole_alone = false;
        for (const auto& line : r.log0) whole_alone = whole_alone || line.rfind("prefill 0 @0 +1000", 0) == 0;
        CHECK(whole_alone);
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("G2 batched self-speculation, 6 prompts, 4 slots, MTP:\n");
    {
        std::uint64_t multi = 0;
        const auto r = run_pair(6, 4, true, 64, 0, hello_for(8, 4096, 2), true, 0, 0, 0, false, true, &multi);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty());
        CHECK(r.result.exact);
        CHECK(r.log0 == r.log1);  // the follower made exactly the lead's engine calls
        CHECK(r.stats.g2_cycles > 0 && multi > 0 && r.stats.g2_drafts > 0 && r.stats.g2_accepted > 0);
        std::printf("  g2 cycles %llu rows %llu (%.2f/cycle) drafts %llu accepted %llu extension calls %llu, multi-slot verifies %llu\n",
                    static_cast<unsigned long long>(r.stats.g2_cycles), static_cast<unsigned long long>(r.stats.g2_rows),
                    r.stats.g2_cycles ? static_cast<double>(r.stats.g2_rows) / static_cast<double>(r.stats.g2_cycles) : 0.0,
                    static_cast<unsigned long long>(r.stats.g2_drafts), static_cast<unsigned long long>(r.stats.g2_accepted),
                    static_cast<unsigned long long>(r.stats.g2_draft_calls), static_cast<unsigned long long>(multi));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.follow_error.empty()) std::printf("  follow error: %s\n", r.follow_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("G2 + chunked long prompts + staggered arrivals, 8 slots:\n");
    {
        const auto r = run_pair(8, 8, true, 96, 0, hello_for(8, 4096, 2), true, 128, 4, 0, false, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.stats.g2_cycles > 0);
        // C=6 / C=8: plain steps (no room for drafts), G2 again with fewer slots after a re-seed
        CHECK(r.stats.g2_plain > 0 && r.stats.g2_reseeds > 0);
        std::size_t wide_spec = 0;
        for (const auto& line : r.log0) {
            if (line.rfind("verify", 0) != 0) continue;
            std::size_t windows = 0;
            for (std::size_t p = line.find('@'); p != std::string::npos; p = line.find('@', p + 1)) ++windows;
            if (windows > 4) ++wide_spec;
        }
        CHECK(wide_spec == 0);  // never a windowed verify of more than 4 slots
        std::printf("  g2 cycles %llu, handover plain steps %llu, re-seeds %llu, steps %llu\n",
                    static_cast<unsigned long long>(r.stats.g2_cycles), static_cast<unsigned long long>(r.stats.g2_plain),
                    static_cast<unsigned long long>(r.stats.g2_reseeds), static_cast<unsigned long long>(r.stats.steps));
        if (!r.lead_error.empty()) std::printf("  lead error: %s\n", r.lead_error.c_str());
        if (!r.driver_error.empty()) std::printf("  driver error: %s\n", r.driver_error.c_str());
    }
    std::printf("skip basic: only the long reference and the staggered arms:\n");
    {
        const auto r = run_pair(3, 4, true, 40, 0, hello_for(8, 4096, 2), true, 128, 3, 0, true);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
        CHECK(r.result.levels.empty());  // no short levels ran
    }
    std::printf("prefill interleave: chunk 100 (not a divisor), plain:\n");
    {
        const auto r = run_pair(3, 4, true, 40, 0, hello_for(8, 4096, 2), false, 100, 3);
        CHECK(r.lead_error.empty() && r.follow_error.empty() && r.driver_error.empty() && r.result.exact && r.log0 == r.log1);
    }
    queued_successor_interleave();

    // ---- decode-first: a running stream gets one decode round between consecutive prefill chunks of a new prompt
    // (interleave off), or the same rounds per prompt token around grouped chunks (the shared PrefillInterleave policy:
    // 64, then 4 x 64 while one request decodes)
    for (const bool grouped : {false, true}) {
        FakeEng e(4);
        e.stops = false;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(4, 4096, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        lc.interleave.select = grouped ? sv::PrefillInterleave::Select::automatic : sv::PrefillInterleave::Select::off;
        lc.interleave.max_stall_ms = 1e9;
        lc.prefill_timeline = grouped ? 1 : 0;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::vector<std::uint32_t> p, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt = std::move(p);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        auto ta = job(prompts(1)[0], 400);
        std::vector<std::uint32_t> a;
        bool done = false;
        while (a.size() < 5 && !done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        std::vector<std::uint32_t> big(640, 5U);
        auto tb = job(big, 4);
        bool done_b = false;
        while (!done_b) (void)tb->wait(std::chrono::milliseconds(100), done_b);
        while (!done) (void)ta->wait(std::chrono::milliseconds(100), done);
        stop.store(true);
        runner.join();
        // between two chunk calls of the big prompt there is exactly one step; grouped: 64 + 256 + 256 + 64 tokens with
        // 1, 4 and 4 steps after the first three chunks (the same 9 steps the 10 single chunks get)
        int last_chunk = -1, violations = 0, chunks = 0, steps_between = 0;
        const int want_grouped[3] = {1, 4, 4};
        for (std::size_t i = 0; i < e.log.size(); ++i) {
            if (e.log[i].rfind("prefill 1 ", 0) == 0) {
                if (last_chunk >= 0) {
                    int st = 0;
                    for (std::size_t j = static_cast<std::size_t>(last_chunk) + 1; j < i; ++j) st += e.log[j].rfind("step", 0) == 0 ? 1 : 0;
                    if (st != (grouped ? want_grouped[std::min(chunks - 1, 2)] : 1)) ++violations;
                    steps_between += st;
                }
                last_chunk = static_cast<int>(i);
                ++chunks;
            }
        }
        CHECK(chunks == (grouped ? 4 : 10) && violations == 0 && steps_between == 9);
        std::printf("decode-first interleave (%s, %d steps between chunks): %d chunks of the 640-token prompt, expected step pattern: %s\n", grouped ? "grouped" : "single", steps_between, chunks,
                    violations == 0 ? "yes" : "NO");
    }
    import_route_receipts();
    // ---- a prompt waiting for its RTX state does not hold the admission: a short prompt behind it is served meanwhile
    {
        Loopback lb;
        FakeEng e(4), e1(4);
        e.stops = e1.stops = false;
        std::atomic<bool> hold{true};
        e.hold = e1.hold = &hold;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(4, 4096, 2), 4, 4096); });
        ch.hello(hello_for(4, 4096, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 64;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 100;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        sv::GenerationJob j;
        j.prompt.assign(300, 9U);
        j.max_tokens = 4;
        auto t = queue.submit(std::move(j));
        sv::GenerationJob k;
        k.prompt.assign(90, 4U);  // two chunks, not routed
        k.max_tokens = 4;
        auto u = queue.submit(std::move(k));
        bool done = false, done_u = false;
        while (!done_u) (void)u->wait(std::chrono::milliseconds(100), done_u);
        (void)t->wait(std::chrono::milliseconds(1), done);
        CHECK(!done);  // the routed prompt still waits for its state
        hold.store(false);
        while (!done) (void)t->wait(std::chrono::milliseconds(100), done);
        stop.store(true);
        runner.join();
        follower.join();
        CHECK(t->stats().imported_tokens == 299 && u->stats().imported_tokens == 0 && e.log == e1.log);
        std::printf("import wait beside the prefill: short prompt served meanwhile: %s\n", done_u ? "yes" : "NO");
    }
    // ---- import gate shape (8192-token prompt, chunked admission on, the slot reused across arms): the imported arm's
    // tokens equal the own-prefill arm's from the first one on (no extra / stale first token), plain and MTP
    for (const bool mtp : {false, true}) {
        Loopback lb;
        FakeEng e(2), e1(2);
        e.stops = e1.stops = false;
        gs::Channel ch(lb.side(0), 0), c1(lb.side(1), 1);
        std::thread follower([&] { gs::follow(e1, c1, hello_for(2, 16384, 2), 2, 16384); });
        ch.hello(hello_for(2, 16384, 2));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 2;
        lc.capacity = 16384;
        lc.heartbeat_seconds = 0.01;
        lc.prefill_chunk = 512;
        lc.mtp = mtp;
        lc.mtp_cycles = 2;
        lc.state_digest = true;
        lc.rtx_prefill = [](std::span<const std::uint32_t>, std::uint32_t) { return true; };
        lc.import_min_tokens = 512;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        std::vector<std::uint32_t> q8k;
        for (unsigned i = 0; q8k.size() < 8188; ++i) q8k.insert(q8k.end(), {11U, 100000U + i % 997U});
        for (const std::uint32_t t : {102340U, 11U, 102573U, 22U}) q8k.push_back(t);  // not insert({...}): gcc 14 -O3 reports a false -Wstringop-overflow
        auto run = [&](bool routed) {
            lead.set_route(routed);
            sv::GenerationJob j;
            j.prompt = q8k;
            j.max_tokens = 32;
            auto t = queue.submit(std::move(j));
            std::vector<std::uint32_t> got;
            bool done = false;
            while (!done) {
                auto g = t->wait(std::chrono::milliseconds(100), done);
                got.insert(got.end(), g.begin(), g.end());
            }
            return std::make_pair(got, t->stats());
        };
        const auto b0 = run(false), a = run(true), b1 = run(false);
        stop.store(true);
        runner.join();
        follower.join();
        CHECK(a.second.imported_tokens == 8191 && b0.second.imported_tokens == 0);
        CHECK(a.first.size() == 32 && a.first == b0.first && b0.first == b1.first);
        CHECK(e.log == e1.log);
        bool import_line = false;
        for (const auto& line : e.log) import_line = import_line || line.rfind("import 0 8192", 0) == 0 || line.rfind("import 1 8192", 0) == 0;
        CHECK(import_line);
        const auto dg = lead.take_digests();
        CHECK(dg.size() == 3 && dg[0].rank0 == dg[1].rank0 && dg[1].rank0 == dg[2].rank0 && dg[1].frontier == 8192);
        std::printf("import gate shape, 8192-token prompt%s: imported == own prefill from the first token: %s\n", mtp ? ", MTP" : "",
                    a.first == b0.first ? "yes" : "NO");
    }
    // ---- a second request arriving while the first decodes on the MTP path: leave, batched steps, enter again
    {
        FakeEng e(4);
        e.stops = false;
        e.delay_us = 300;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(4, 4096, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 4;
        lc.capacity = 4096;
        lc.heartbeat_seconds = 0.01;
        lc.mtp = true;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::vector<std::uint32_t> p, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt = std::move(p);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        const auto pa = prompts(2)[0], pb = prompts(2)[1];
        auto ta = job(pa, 300);
        std::vector<std::uint32_t> a;
        bool done = false;
        while (a.size() < 20 && !done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        auto tb = job(pb, 40);
        std::vector<std::uint32_t> b;
        bool done_b = false;
        while (!done_b) {
            auto g = tb->wait(std::chrono::milliseconds(100), done_b);
            b.insert(b.end(), g.begin(), g.end());
        }
        while (!done) {
            auto g = ta->wait(std::chrono::milliseconds(100), done);
            a.insert(a.end(), g.begin(), g.end());
        }
        stop.store(true);
        runner.join();
        // reference: the plain chain of each prompt alone
        auto plain = [](const std::vector<std::uint32_t>& p, std::size_t n) {
            FakeEng f(1);
            f.stops = false;
            f.reset(0);
            std::vector<std::uint32_t> t{f.prefill(0, p, 0)};
            while (t.size() < n) {
                const gs::WireRow r{0, t.back(), f.frontier[0]};
                t.push_back(f.step({&r, 1})[0]);
            }
            return t;
        };
        CHECK(a == plain(pa, 300));
        CHECK(b == plain(pb, 40));
        const auto st = lead.stats();
        CHECK(st.mtp_leaves >= 1 && st.mtp_enters >= 1 && st.steps > 0 && st.mtp_rounds > 0);
        std::printf("arrival during MTP: A %zu tokens, B %zu tokens, mtp rounds %llu enters %llu leaves %llu, steps %llu\n", a.size(),
                    b.size(), static_cast<unsigned long long>(st.mtp_rounds), static_cast<unsigned long long>(st.mtp_enters),
                    static_cast<unsigned long long>(st.mtp_leaves), static_cast<unsigned long long>(st.steps));
    }
    // ---- prefill units (a long MTP prompt beside a decoding request, cut by LAYERS): one engine session paused at layer
    // boundaries of its 256-row pieces; every call carries the 64-token base chunk's work in layer units of the current
    // piece (LayerUnits::base_units; grouping capped at one base-equivalent), a decode round separates the calls, the
    // tokens equal the idle run; a cancelled request closes the session on the engine and the next prompt prefills
    {
        namespace ss = strix::session;
        static_assert(ss::LayerUnits{640, 256, 45}.total() == 135 && ss::LayerUnits{640, 256, 45}.pieces() == 3);
        static_assert(ss::LayerUnits{640, 256, 45}.piece_rows_at(89) == 256 && ss::LayerUnits{640, 256, 45}.piece_rows_at(90) == 128);
        static_assert(ss::LayerUnits{640, 256, 45}.base_units(64, 0) == 12);    // 64 x 45 / 256 = 11.25, rounded up
        static_assert(ss::LayerUnits{640, 256, 45}.base_units(64, 96) == 23);   // the 128-row piece: 22.5
        static_assert(ss::LayerUnits{640, 256, 45}.base_units(64, 130) == 5);   // the units left
        static_assert(ss::LayerUnits{640, 256, 45}.base_units(0, 3) == 132 && ss::LayerUnits{640, 256, 45}.base_units(64, 135) == 0);
        static_assert(ss::LayerUnits{0, 256, 45}.total() == 0 && ss::LayerUnits{1, 4096, 45}.base_units(512, 0) == 45);
        for (const bool cancel : {false, true}) {
            FakeEng e(4);
            e.stops = false;
            e.unit_rows = 256;
            e.unit_layers = 45;
            if (cancel) e.units_delay_us = 20000;
            gs::Channel ch({}, 0);
            ch.hello(hello_for(4, 4096, 1));
            sv::GenerationQueue queue(8);
            gs::LeadConfig lc;
            lc.slots = 4;
            lc.capacity = 4096;
            lc.heartbeat_seconds = 0.01;
            lc.mtp = true;
            lc.prefill_chunk = 64;
            lc.prefill_units_rows = 256;
            lc.prefill_units_layers = 45;
            lc.prefill_units_admit = [](std::uint64_t) { return true; };
            lc.prefill_timeline = 1;
            std::atomic<bool> stop{false};
            gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
            std::thread runner([&] { lead.loop(); });
            auto job = [&](std::vector<std::uint32_t> p, std::uint32_t max) {
                sv::GenerationJob j;
                j.prompt = std::move(p);
                j.max_tokens = max;
                return queue.submit(std::move(j));
            };
            const auto pa = prompts(3)[0], pd = prompts(3)[2];
            auto ta = job(pa, 400);
            std::vector<std::uint32_t> a, b, d;
            bool done = false;
            while (a.size() < 5 && !done) {
                auto g = ta->wait(std::chrono::milliseconds(100), done);
                a.insert(a.end(), g.begin(), g.end());
            }
            std::vector<std::uint32_t> big(640);
            for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<std::uint32_t>(3 + i % 97);
            auto tb = job(big, 6);
            bool done_b = false;
            if (cancel) {
                std::this_thread::sleep_for(std::chrono::milliseconds(70));
                tb->cancel();
            }
            while (!done_b) {
                auto g = tb->wait(std::chrono::milliseconds(100), done_b);
                b.insert(b.end(), g.begin(), g.end());
            }
            auto td = job(pd, 8);
            bool done_d = false;
            while (!done_d) {
                auto g = td->wait(std::chrono::milliseconds(100), done_d);
                d.insert(d.end(), g.begin(), g.end());
            }
            while (!done) {
                auto g = ta->wait(std::chrono::milliseconds(100), done);
                a.insert(a.end(), g.begin(), g.end());
            }
            stop.store(true);
            runner.join();
            auto plain = [](const std::vector<std::uint32_t>& p, std::size_t n) {
                FakeEng f(1);
                f.stops = false;
                f.reset(0);
                std::vector<std::uint32_t> t{f.prefill(0, p, 0)};
                while (t.size() < n) {
                    const gs::WireRow r{0, t.back(), f.frontier[0]};
                    t.push_back(f.step({&r, 1})[0]);
                }
                return t;
            };
            CHECK(a == plain(pa, 400));
            CHECK(d == plain(pd, 8));
            int back_to_back = 0;
            for (std::size_t i = 1; i < e.log.size(); ++i)
                if (e.log[i].rfind("units 1 +", 0) == 0 && e.log[i - 1].rfind("units 1 +", 0) == 0) ++back_to_back;
            if (!cancel) {
                const std::vector<std::uint32_t> want{12, 12, 12, 12, 12, 12, 12, 12, 23, 16};
                CHECK(b == plain(big, 6));
                CHECK(e.unit_calls == want);
                CHECK(back_to_back == 0 && e.units_cancelled == 0);
            } else {
                CHECK(e.units_cancelled == 1 && !e.units_open && e.unit_calls.size() < 10);
            }
            std::string calls;
            for (const auto u : e.unit_calls) calls += (calls.empty() ? "" : " ") + std::to_string(u);
            std::printf("prefill units%s: %zu calls (%s), decode between every call: %s, tokens == idle run: %s\n",
                        cancel ? " (cancelled)" : "", e.unit_calls.size(), calls.c_str(),
                        back_to_back == 0 ? "yes" : "NO", a == plain(pa, 400) && d == plain(pd, 8) ? "yes" : "NO");
        }
    }
    // ---- world 1, context limit, rejected prompt, max_tokens
    {
        FakeEng e(2);
        e.stops = false;
        gs::Channel ch({}, 0);
        ch.hello(hello_for(2, 64, 1));
        sv::GenerationQueue queue(8);
        gs::LeadConfig lc;
        lc.slots = 2;
        lc.capacity = 64;
        lc.heartbeat_seconds = 0.01;
        std::atomic<bool> stop{false};
        gs::Lead<FakeEng> lead(e, ch, queue, lc, stop);
        std::thread runner([&] { lead.loop(); });
        auto job = [&](std::size_t n, std::uint32_t max) {
            sv::GenerationJob j;
            j.prompt.assign(n, 7U);
            j.max_tokens = max;
            return queue.submit(std::move(j));
        };
        auto drain = [](const std::shared_ptr<sv::GenerationTicket>& t) {
            std::vector<std::uint32_t> all;
            bool done = false;
            while (!done) {
                auto g = t->wait(std::chrono::milliseconds(100), done);
                all.insert(all.end(), g.begin(), g.end());
            }
            return all;
        };
        auto t1 = job(40, 1000);
        auto t2 = job(64, 10);
        auto t3 = job(10, 5);
        const auto a1 = drain(t1), a2 = drain(t2), a3 = drain(t3);
        CHECK(a1.size() == 25 && t1->stats().finish == "context");  // first token + rows at positions 40..63
        CHECK(a2.empty() && t2->stats().error_code == "context_length_exceeded");
        CHECK(a3.size() == 5 && t3->stats().finish == "length");
        stop.store(true);
        runner.join();
    }
    {  // Embed frame round: receive bounds and paired refusals (host only). The follower must fail the pair before it
       // enters a further collective: its wire-call count shows how far it got (1: summary only, 2: summary and bytes).
        namespace sess = strix::session;
        sess::EmbedInterner interner;
        sess::EmbedRows span;
        span.rows = 6;
        span.width = 16;
        span.real_token_id = strix::glm::kImageToken;
        span.digest.fill(9);
        span.data.assign(std::size_t{6} * 16U, 0x3F80);
        const auto lease = interner.intern(std::move(span));
        const auto table = std::make_shared<const sess::EmbedTable>(std::vector<sess::EmbedLease>{lease});
        const auto all_ids = lease.ids();
        CHECK(all_ids.size() == 6);

        // The row bound: per handle the span limit and the rows the chunk and the room after it can hold.
        const std::vector<std::uint32_t> head3(all_ids.begin(), all_ids.begin() + 3);
        CHECK(gs::embed_frame_rows_bound(head3, 0) == 3);
        CHECK(gs::embed_frame_rows_bound(head3, 3) == 6);
        CHECK(gs::embed_frame_rows_bound(head3, std::uint64_t{1} << 40U) == sess::kEmbedMaxRows);
        CHECK(gs::embed_frame_rows_bound(std::vector<std::uint32_t>{11, 12}, 100) == 0);

        struct RefuseEng {
            std::vector<std::byte> embed_frame(std::span<const std::uint32_t>) { return {}; }
            void embed_frame_received(std::span<const std::byte>, std::vector<std::uint32_t>&) { throw std::runtime_error("refused here"); }
        };
        struct Outcome {
            std::string lead_error, follow_error;
            unsigned follower_calls = 0;
        };
        // lead_fn drives rank 0 on its channel; the follower runs the real round on `eng` for `chunk`.
        const auto drive = [&](const std::function<void(gs::Channel&)>& lead_fn, auto& eng, const std::vector<std::uint32_t>& chunk,
                               std::uint64_t capacity, std::uint64_t room_after) {
            Outcome o;
            Loopback lb;
            std::atomic<unsigned> calls{0};
            auto w1 = lb.side(1);
            gs::Channel c0(lb.side(0), 0);
            gs::Channel c1([&](std::span<const std::byte> out, std::span<std::byte> in, std::uint32_t layer, std::uint64_t request) {
                ++calls;
                w1(out, in, layer, request);
            }, 1);
            std::thread lead([&] {
                try { lead_fn(c0); } catch (const std::exception& e) { o.lead_error = e.what(); }
            });
            std::vector<std::uint32_t> prompt1 = chunk;
            try { gs::embed_frame_round(eng, c1, prompt1, false, capacity, room_after); } catch (const std::exception& e) { o.follow_error = e.what(); }
            lead.join();
            o.follower_calls = calls.load();
            return o;
        };
        const auto frame_of = [&](const std::vector<std::uint32_t>& chunk) {  // a fresh ledger each time: rows always travel
            FakeEng probe(2);
            probe.bind_embeds(table);
            return probe.embed_frame(chunk);
        };
        const auto send_summary = [](gs::Channel& ch, std::array<std::uint32_t, gs::kEmbedFrameSummaryWords> s) {
            std::vector<std::uint32_t> v(s.begin(), s.end());
            ch.tokens(v, v.size(), true, gs::kEmbedFrameMessage);
        };
        const auto send_bytes = [](gs::Channel& ch, const std::vector<std::byte>& frame) {
            std::vector<std::uint32_t> w((frame.size() + 3U) / 4U, 0U);
            std::memcpy(w.data(), frame.data(), frame.size());
            ch.tokens(w, w.size(), true, gs::kEmbedFrameMessage + 1U);
        };
        const std::uint64_t cap = std::uint64_t{1} << 17U;

        {  // the whole span in the chunk, bound met: applied
            FakeEng e0(2), e1(2);
            e0.bind_embeds(table);
            std::vector<std::uint32_t> p0 = all_ids;
            const auto o = drive([&](gs::Channel& ch) { gs::embed_frame_round(e0, ch, p0, true, cap, 0); }, e1, all_ids, cap, 0);
            CHECK(o.lead_error.empty() && o.follow_error.empty() && o.follower_calls == 2 && e1.embed_frames == 1);
        }
        {  // a chunk of one handle whose summary claims the whole span (6 rows), but the chunk holds 3 and the context room
           // after it is zero: the bound is 3 rows, so the follower refuses on the summary, before any payload storage
            FakeEng e1(2);
            const auto frame = frame_of(head3);
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, gs::embed_frame_summary(frame)); }, e1, head3, cap, 0);
            CHECK(o.follow_error.find("summary out of range") != std::string::npos);
            CHECK(o.follower_calls == 1 && e1.embed_frames == 0);
        }
        {  // spans of the summary differ from the chunk's distinct handles
            FakeEng e1(2);
            auto s = gs::embed_frame_summary(frame_of(all_ids));
            s[1] = 2;
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, s); }, e1, all_ids, cap, 0);
            CHECK(o.follow_error.find("summary out of range") != std::string::npos && o.follower_calls == 1);
        }
        {  // width above the model's
            FakeEng e1(2);
            auto s = gs::embed_frame_summary(frame_of(all_ids));
            s[3] = 1U << 17U;
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, s); }, e1, all_ids, cap, 0);
            CHECK(o.follow_error.find("summary out of range") != std::string::npos && o.follower_calls == 1);
        }
        {  // the bytes disagree with the summary: an entry claims other rows / width / axes than the summary says
            const auto good = frame_of(all_ids);
            const std::size_t at = strix::pair::kEmbedFrameHeaderBytes;  // first entry; rows at +4, width at +8, axes at +16
            for (const std::size_t field : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
                FakeEng e1(2);
                auto bad = good;
                std::uint32_t v = 0;
                std::memcpy(&v, bad.data() + at + field, 4);
                v = field == 16U ? 1U : v - 1U;
                std::memcpy(bad.data() + at + field, &v, 4);
                const auto o = drive([&](gs::Channel& ch) { send_summary(ch, gs::embed_frame_summary(good)); send_bytes(ch, bad); }, e1,
                                     all_ids, cap, 0);
                CHECK(o.follow_error.find("does not match its summary") != std::string::npos);
                CHECK(o.follower_calls == 2 && e1.embed_frames == 0);
            }
        }
        {  // a byte length that the entries do not imply (two trailing bytes under a summary that states them)
            FakeEng e1(2);
            auto longer = frame_of(all_ids);
            auto s = gs::embed_frame_summary(longer);
            longer.insert(longer.end(), 2, std::byte{0});
            s[0] = static_cast<std::uint32_t>(longer.size());
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, s); send_bytes(ch, longer); }, e1, all_ids, cap, 0);
            CHECK(o.follow_error.find("does not match its summary") != std::string::npos);
            CHECK(o.follower_calls == 2 && e1.embed_frames == 0);
        }
        {  // receiver refusal: the engine cannot apply a frame that matches its summary; the pair fails on the follower
            RefuseEng refuse;
            const auto good = frame_of(all_ids);
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, gs::embed_frame_summary(good)); send_bytes(ch, good); }, refuse,
                                 all_ids, cap, 0);
            CHECK(o.follow_error.find("control channel: embed frame: refused here") != std::string::npos);
            CHECK(o.follower_calls == 2);
        }
        {  // peer teardown: the lead leaves after the summary; the follower's byte exchange fails (loopback timeout, 5 s)
           // and the round throws instead of returning a half-received frame
            FakeEng e1(2);
            const auto good = frame_of(all_ids);
            const auto o = drive([&](gs::Channel& ch) { send_summary(ch, gs::embed_frame_summary(good)); }, e1, all_ids, cap, 0);
            CHECK(!o.follow_error.empty() && e1.embed_frames == 0);
        }
    }
    if (failures == 0) std::printf("glm serve cpu tests: ALL PASS\n");
    else std::printf("glm serve cpu tests: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
