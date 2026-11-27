// N-rank layer pipeline (strix/pair/layer_pipeline.hpp) on a small CPU layer graph: every rank holds only its layers'
// sequence state (a recurrence like a GDN layer and a causal history like an attention layer), ranks run in threads
// connected by in-memory ordered links carrying the real shared seam pipe (strix/pair/pipe.hpp). Invariant: for any
// number of ranks and any cuts, the composed prefill logits and every decoded id equal the single-rank run byte for byte.
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "strix/pair/layer_pipeline.hpp"

namespace {
using namespace strix::pair;
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
}

constexpr unsigned H = 8, V = 11;
struct Weights {
    unsigned layers;
    std::vector<float> w, embed, head;
    explicit Weights(unsigned l) : layers(l), w(std::size_t(l) * H * H), embed(std::size_t(V) * H), head(std::size_t(V) * H) {
        std::uint32_t s = 12345U;
        const auto next = [&] { s = s * 1664525U + 1013904223U; return float(int(s >> 9) % 2001 - 1000) / 1000.0F; };
        for (auto* v : {&w, &embed, &head}) for (auto& x : *v) x = next();
    }
};
// One rank's engine: owns layers [begin, end); state of other layers is never touched.
struct Toy {
    const Weights& m;
    unsigned begin, end;
    std::vector<float> rec, hist, x, logits;
    unsigned committed = 0;
    unsigned stage_end;   // windows commit at the stage end (a relay decoder owns more layers than its stage runs)
    Toy(const Weights& w, unsigned b, unsigned e, unsigned max_rows)
        : m(w), begin(b), end(e), rec(std::size_t(w.layers) * H), hist(std::size_t(w.layers) * H), x(std::size_t(max_rows) * H), logits(V), stage_end(e) {}
    void range(unsigned first, unsigned last, const std::uint32_t* tokens, unsigned rows, bool head) {
        if (first < begin || last > end) throw std::logic_error("toy: layer outside the owned range");
        if (first == 0U)
            for (unsigned r = 0; r < rows; ++r) std::memcpy(&x[std::size_t(r) * H], &m.embed[std::size_t(tokens[r]) * H], H * 4U);
        for (unsigned l = first; l < last; ++l)
            for (unsigned r = 0; r < rows; ++r) {
                float* xr = &x[std::size_t(r) * H];
                float* re = &rec[std::size_t(l) * H];
                float* hi = &hist[std::size_t(l) * H];
                float y[H];
                for (unsigned j = 0; j < H; ++j) {
                    re[j] = 0.75F * re[j] + 0.25F * xr[j];
                    hi[j] += xr[j];
                    float acc = 0.0F;
                    for (unsigned i = 0; i < H; ++i) acc += m.w[(std::size_t(l) * H + j) * H + i] * xr[i];
                    y[j] = acc + 0.5F * re[j] + 0.25F * hi[j] / float(committed + r + 1U);
                }
                for (unsigned j = 0; j < H; ++j) xr[j] += 0.5F * std::tanh(y[j]);
            }
        if (head) {
            const float* xl = &x[std::size_t(rows - 1U) * H];
            for (unsigned v = 0; v < V; ++v) {
                float acc = 0.0F;
                for (unsigned j = 0; j < H; ++j) acc += m.head[std::size_t(v) * H + j] * xl[j];
                logits[v] = acc;
            }
        }
        if (last == stage_end) committed += rows;
    }
    // Layer state of [first, last): recurrent + history rows (the engine's export_state / import_state shape).
    std::vector<std::byte> save(unsigned first, unsigned last) const {
        std::vector<std::byte> b(std::size_t(last - first) * H * 8U);
        std::memcpy(b.data(), &rec[std::size_t(first) * H], std::size_t(last - first) * H * 4U);
        std::memcpy(b.data() + std::size_t(last - first) * H * 4U, &hist[std::size_t(first) * H], std::size_t(last - first) * H * 4U);
        return b;
    }
    void load(unsigned first, unsigned last, std::span<const std::byte> b) {
        if (first < begin || last > end || b.size() != std::size_t(last - first) * H * 8U) throw std::logic_error("toy: bad state load");
        std::memcpy(&rec[std::size_t(first) * H], b.data(), std::size_t(last - first) * H * 4U);
        std::memcpy(&hist[std::size_t(first) * H], b.data() + std::size_t(last - first) * H * 4U, std::size_t(last - first) * H * 4U);
    }
    std::uint32_t top() const {
        std::uint32_t best = 0;
        for (std::uint32_t v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    }
};

// In-memory ordered duplex link; close() wakes every waiter and makes later calls throw (the pipe's abort).
struct Channel {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::byte> q[2];
    bool closed = false;
    void send(int dir, std::span<const std::byte> b) {
        std::lock_guard lock(mu);
        if (closed) throw std::runtime_error("link closed");
        q[dir].insert(q[dir].end(), b.begin(), b.end());
        cv.notify_all();
    }
    void receive(int dir, std::span<std::byte> b) {
        std::unique_lock lock(mu);
        cv.wait(lock, [&] { return closed || q[dir].size() >= b.size(); });
        if (q[dir].size() < b.size()) throw std::runtime_error("link closed");
        std::copy(q[dir].begin(), q[dir].begin() + std::ptrdiff_t(b.size()), b.begin());
        q[dir].erase(q[dir].begin(), q[dir].begin() + std::ptrdiff_t(b.size()));
    }
    void close() { std::lock_guard lock(mu); closed = true; cv.notify_all(); }
};

struct Result {
    std::vector<float> logits, dec_logits;  // last prompt row / last decode step (head stage)
    std::vector<std::uint32_t> ids;
    std::string error;
    std::uint64_t guesses = 0, hits = 0;
};
// First-stage guess policies for pipelined speculation (DecodeOps::guess / rewind).
enum class Guess { off, oracle, wrong, alternate, none };

// Plain single-rank loop (not the driver): the reference.
Result reference(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen) {
    Toy t(m, 0, m.layers, width);
    Result r;
    for (const auto& w : layers::windows(prompt.size(), width))
        t.range(0, m.layers, prompt.data() + w.start, w.rows, w.start + w.rows == prompt.size());
    r.logits = t.logits;
    r.ids.push_back(t.top());
    for (unsigned s = 1; s < gen; ++s) {
        const auto tok = r.ids.back();
        t.range(0, m.layers, &tok, 1U, true);
        r.ids.push_back(t.top());
    }
    r.dec_logits = t.logits;
    return r;
}

std::vector<Result> run_chain(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen,
                              const std::vector<std::uint32_t>& cuts, std::uint64_t bad_identity_stage = ~0ULL,
                              std::uint32_t relay_decoder = ~0U, Guess guess = Guess::off,
                              const std::vector<std::uint32_t>* truth = nullptr, bool broken_rewind = false) {
    const unsigned stages = unsigned(cuts.size() - 1U);
    std::vector<std::unique_ptr<Channel>> links;
    for (unsigned i = 0; i + 1U < stages; ++i) links.push_back(std::make_unique<Channel>());
    std::vector<Result> out(stages);
    const auto count = layers::windows(prompt.size(), width).size();
    const std::size_t row_bytes = H * 4U;
    std::vector<std::thread> threads;
    for (unsigned s = 0; s < stages; ++s)
        threads.emplace_back([&, s] {
            try {
                layers::Placement p{cuts, s};
                const bool decoder = s == relay_decoder;   // relay: the decoder owns every layer, runs its stage
                Toy t(m, decoder ? 0U : cuts[s], decoder ? m.layers : cuts[s + 1U], width);
                t.stage_end = cuts[s + 1U];
                const std::uint64_t identity = s == bad_identity_stage ? 2U : 1U;
                std::unique_ptr<Pipe> in, outp;
                Channel* up = s ? links[s - 1U].get() : nullptr;      // dir 0: down the line, dir 1: up the line
                Channel* down = s + 1U < stages ? links[s].get() : nullptr;
                const auto options = [&](std::uint32_t cut) {
                    PipeOptions o; o.validate = layers::validator(identity, count, cut, width, row_bytes); return o; };
                if (up)
                    in = std::make_unique<Pipe>(1U, count, [up](std::span<const std::byte> b) { up->send(1, b); },
                        [up](std::span<std::byte> b) { up->receive(0, b); }, [up] { up->close(); }, options(cuts[s]));
                if (down)
                    outp = std::make_unique<Pipe>(0U, count, [down](std::span<const std::byte> b) { down->send(0, b); },
                        [down](std::span<std::byte> b) { down->receive(1, b); }, [down] { down->close(); }, options(cuts[s + 1U]));
                layers::StageOps ops;
                ops.row_bytes = row_bytes;
                ops.forward = [&](std::span<const std::uint32_t> tok, const layers::Window& w, bool head) {
                    t.range(p.first(), p.last(), tok.data(), w.rows, head); };
                ops.seam = [&](PipeFrame& f, bool save, std::size_t bytes) {
                    if (save) std::memcpy(f.payload.data(), t.x.data(), bytes); else std::memcpy(t.x.data(), f.payload.data(), bytes); };
                ops.fence = [] {};
                (void)layers::prefill(p, prompt, width, identity, in.get(), outp.get(), ops);
                if (p.head()) out[s].logits = t.logits;
                layers::Link lu, ld;
                if (up) { lu.send = [up](std::span<const std::byte> b) { up->send(1, b); }; lu.receive = [up](std::span<std::byte> b) { up->receive(0, b); }; }
                if (down) { ld.send = [down](std::span<const std::byte> b) { down->send(0, b); }; ld.receive = [down](std::span<std::byte> b) { down->receive(1, b); }; }
                layers::Placement dp = p;
                std::uint32_t first = p.head() ? t.top() : 0U;
                if (relay_decoder != ~0U) {
                    layers::RelayOps r;
                    r.save = [&](std::uint32_t layer) { return t.save(layer, layer + 1U); };
                    r.load = [&](std::uint32_t layer, std::span<const std::byte> b) { t.load(layer, layer + 1U, b); };
                    if (relay_decoder % 2U) r.checksum = [](std::span<const std::byte> b) {   // an injected checksum (odd decoders)
                        std::uint32_t h = 2166136261U; for (auto v : b) { h ^= std::to_integer<std::uint32_t>(v); h *= 16777619U; } return h; };
                    const auto rs = layers::relay_state(p, relay_decoder, first, up ? &lu : nullptr, down ? &ld : nullptr, r);
                    if (!decoder) return;
                    first = rs.token;
                    t.stage_end = m.layers;
                    dp = layers::Placement{{0U, m.layers}, 0U};
                    up = nullptr; down = nullptr;
                }
                layers::DecodeOps d;
                d.row_bytes = row_bytes;
                d.forward = [&](std::uint32_t tok, bool head) { t.range(dp.first(), dp.last(), &tok, 1U, head); };
                d.top = [&] { return t.top(); };
                d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), t.x.data(), row_bytes); };
                d.import_row = [&](std::span<const std::byte> b) { std::memcpy(t.x.data(), b.data(), row_bytes); };
                // The toy's state is recurrent (not position-indexed), so its rewind restores a snapshot taken before
                // every forward; broken_rewind keeps the guessed row's state (negative control: must be detected).
                std::vector<float> snap_rec, snap_hist;
                unsigned snap_committed = 0, calls = 0;
                if (s == 0U && guess != Guess::off) {
                    d.forward = [&](std::uint32_t tok, bool head) {
                        snap_rec = t.rec; snap_hist = t.hist; snap_committed = t.committed;
                        t.range(dp.first(), dp.last(), &tok, 1U, head);
                    };
                    d.guess = [&](std::uint32_t, std::uint32_t& next) {
                        const unsigned c = calls++;   // call c guesses ids[c + 1]
                        const std::uint32_t right = truth->at(c + 1U);
                        switch (guess) {
                        case Guess::oracle: next = right; return true;
                        case Guess::wrong: next = (right + 1U) % V; return true;
                        case Guess::alternate: next = c % 2U == 0U ? right : (right + 3U) % V; return true;
                        default: return false;
                        }
                    };
                    d.rewind = [&] { if (!broken_rewind) { t.rec = snap_rec; t.hist = snap_hist; t.committed = snap_committed; } };
                }
                const auto st = layers::decode(dp, first, gen, up ? &lu : nullptr, down ? &ld : nullptr, d);
                out[s].ids = st.ids;
                out[s].guesses = st.guesses;
                out[s].hits = st.hits;
                if (dp.head()) out[s].dec_logits = t.logits;
            } catch (const std::exception& e) {
                out[s].error = e.what();
                if (s) links[s - 1U]->close();
                if (s + 1U < stages) links[s]->close();
            }
        });
    for (auto& t : threads) t.join();
    return out;
}

// Two-stage prefill handed to a decode that runs on EVERY rank (share_state) with the head's per-window rows returned to
// stage 0 (return_rows): each rank then decodes alone with the whole model; ids, logits bytes and the returned rows must
// equal one rank's.
struct Shared {
    Result res[2];
    std::vector<float> head_rows, first_rows;   // the final rows of every window: computed on the head, received on stage 0
    std::uint64_t state_bytes[2]{}, tail_bytes[2]{};
};
Shared run_share(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen, std::uint32_t cut) {
    Shared out;
    const std::vector<std::uint32_t> cuts{0, cut, m.layers};
    Channel link;
    const auto wins = layers::windows(prompt.size(), width);
    const auto count = wins.size();
    const std::size_t row_bytes = H * 4U;
    std::atomic<bool> head_done{false};
    std::vector<std::thread> threads;
    for (unsigned s = 0; s < 2U; ++s)
        threads.emplace_back([&, s] {
            try {
                layers::Placement p{cuts, s};
                Toy t(m, 0, m.layers, width);   // TP-style: every rank holds the whole model
                t.stage_end = cuts[s + 1U];
                PipeOptions po;
                po.validate = layers::validator(1U, count, cut, width, row_bytes);
                std::unique_ptr<Pipe> pipe;
                if (s == 1U)
                    pipe = std::make_unique<Pipe>(1U, count, [&](std::span<const std::byte> b) { link.send(1, b); },
                        [&](std::span<std::byte> b) { link.receive(0, b); }, [&] { link.close(); }, po);
                else
                    pipe = std::make_unique<Pipe>(0U, count, [&](std::span<const std::byte> b) { link.send(0, b); },
                        [&](std::span<std::byte> b) { link.receive(1, b); }, [&] { link.close(); }, po);
                std::vector<float> rows;   // head: the final rows of each window, in order
                layers::StageOps ops;
                ops.row_bytes = row_bytes;
                ops.forward = [&](std::span<const std::uint32_t> tok, const layers::Window& w, bool head) {
                    t.range(p.first(), p.last(), tok.data(), w.rows, head);
                    if (p.head()) rows.insert(rows.end(), t.x.begin(), t.x.begin() + std::ptrdiff_t(w.rows * H));
                };
                ops.seam = [&](PipeFrame& f, bool save, std::size_t bytes) {
                    if (save) std::memcpy(f.payload.data(), t.x.data(), bytes); else std::memcpy(t.x.data(), f.payload.data(), bytes); };
                ops.fence = [] {};
                (void)layers::prefill(p, prompt, width, 1U, p.embeds() ? nullptr : pipe.get(), p.head() ? nullptr : pipe.get(), ops);
                pipe.reset();
                layers::Link up, down;
                if (s == 1U) { up.send = [&](std::span<const std::byte> b) { link.send(1, b); }; up.receive = [&](std::span<std::byte> b) { link.receive(0, b); }; }
                else { down.send = [&](std::span<const std::byte> b) { link.send(0, b); }; down.receive = [&](std::span<std::byte> b) { link.receive(1, b); }; }
                // the hand-off waits for BOTH stages: the head finishes late, stage 0 must not pass the barrier before that
                if (p.head()) { std::this_thread::sleep_for(std::chrono::milliseconds(60)); head_done.store(true); }
                layers::line_barrier(s ? &up : nullptr, s ? nullptr : &down);
                if (!p.head() && !head_done.load()) out.res[s].error = "hand-off started before the head stage finished";
                const std::uint32_t first = p.head() ? t.top() : 0U;
                layers::RelayOps r;
                r.save = [&](std::uint32_t layer) { return t.save(layer, layer + 1U); };
                r.load = [&](std::uint32_t layer, std::span<const std::byte> b) { t.load(layer, layer + 1U, b); };
                const auto rs = layers::share_state(p, first, s ? &up : nullptr, s ? nullptr : &down, r);
                out.state_bytes[s] = rs.bytes;
                layers::TailOps tail;
                tail.row_bytes = row_bytes;
                tail.save = [&](std::uint32_t w, std::span<std::byte> b) {
                    std::size_t at = 0;
                    for (std::uint32_t i = 0; i < w; ++i) at += std::size_t(wins[i].rows) * H;
                    std::memcpy(b.data(), &rows[at], b.size());
                };
                std::vector<float> got;
                tail.load = [&](std::uint32_t, const layers::Window&, std::span<const std::byte> b) {
                    const auto n = got.size(); got.resize(n + b.size() / 4U); std::memcpy(&got[n], b.data(), b.size()); };
                out.tail_bytes[s] = layers::return_rows(p, prompt.size(), width, s ? &up : nullptr, s ? nullptr : &down, tail);
                if (s == 0U) out.first_rows = got; else out.head_rows = rows;
                // decode alone on the whole model (no links)
                t.stage_end = m.layers;
                layers::Placement dp{{0U, m.layers}, 0U};
                layers::DecodeOps d;
                d.row_bytes = row_bytes;
                d.forward = [&](std::uint32_t tok, bool head) { t.range(0, m.layers, &tok, 1U, head); };
                d.top = [&] { return t.top(); };
                d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), t.x.data(), row_bytes); };
                d.import_row = [&](std::span<const std::byte> b) { std::memcpy(t.x.data(), b.data(), row_bytes); };
                const auto st = layers::decode(dp, rs.token, gen, nullptr, nullptr, d);
                out.res[s].ids = st.ids;
                out.res[s].dec_logits = t.logits;
            } catch (const std::exception& e) {
                out.res[s].error = e.what();
                link.close();
            }
        });
    for (auto& t : threads) t.join();
    return out;
}

std::string text(const std::vector<std::uint32_t>& cuts) {
    std::string s;
    for (auto c : cuts) s += (s.empty() ? "" : ",") + std::to_string(c);
    return s;
}
} // namespace

int main() {
    const Weights m(7);
    std::vector<std::uint32_t> prompt;
    for (unsigned i = 0; i < 10; ++i) prompt.push_back((i * 7U + 3U) % V);
    const unsigned width = 3, gen = 6;   // windows 3,3,3,1
    const auto ref = reference(m, prompt, width, gen);

    // windows partition the prompt in order
    const auto ws = layers::windows(10, 3);
    expect(ws.size() == 4 && ws[3].start == 9 && ws[3].rows == 1, "windows 10/3 = 3,3,3,1");

    std::vector<std::vector<std::uint32_t>> plans{{0, 7}};
    for (std::uint32_t k = 1; k < 7; ++k) plans.push_back({0, k, 7});                     // N = 2, every cut
    for (std::uint32_t a = 1; a < 7; ++a) for (std::uint32_t b = a + 1U; b < 7; ++b) plans.push_back({0, a, b, 7});  // N = 3
    plans.push_back({0, 1, 2, 3, 7}); plans.push_back({0, 2, 4, 6, 7});                    // N = 4
    unsigned checked = 0;
    for (const auto& cuts : plans) {
        const auto r = run_chain(m, prompt, width, gen, cuts);
        for (std::size_t s = 0; s < r.size(); ++s) {
            expect(r[s].error.empty(), "cuts " + text(cuts) + " stage " + std::to_string(s) + " error: " + r[s].error);
            expect(r[s].ids == ref.ids, "cuts " + text(cuts) + " stage " + std::to_string(s) + " ids differ from one rank");
        }
        const auto& last = r.back().logits;
        expect(last.size() == ref.logits.size() && std::memcmp(last.data(), ref.logits.data(), last.size() * 4U) == 0,
               "cuts " + text(cuts) + " last-row logits bytes differ from one rank");
        ++checked;
    }
    // relay to one decoder (every stage of every plan): the decoder's ids equal one rank, the others decode nothing
    unsigned relayed = 0;
    for (const auto& cuts : plans) {
        for (std::uint32_t dec = 0; dec + 1U < cuts.size(); ++dec) {
            const auto r = run_chain(m, prompt, width, gen, cuts, ~0ULL, dec);
            for (std::size_t s = 0; s < r.size(); ++s) {
                expect(r[s].error.empty(), "relay cuts " + text(cuts) + " decoder " + std::to_string(dec) + " stage " + std::to_string(s) + " error: " + r[s].error);
                if (s == dec) expect(r[s].ids == ref.ids, "relay cuts " + text(cuts) + " decoder " + std::to_string(dec) + " ids differ from one rank");
                else expect(r[s].ids.empty(), "relay: a non-decoder stage decoded");
            }
            ++relayed;
        }
    }
    // two-stage prefill shared with a decode on every rank: both ranks' ids / decode logits equal one rank; the head's
    // per-window rows reach stage 0 byte for byte; the state moved is every layer once per direction
    unsigned shared = 0;
    for (std::uint32_t k = 1; k < 7; ++k) {
        const auto r = run_share(m, prompt, width, gen, k);
        const std::string tag = "share cut " + std::to_string(k);
        for (unsigned s = 0; s < 2U; ++s) {
            expect(r.res[s].error.empty(), tag + " stage " + std::to_string(s) + " error: " + r.res[s].error);
            expect(r.res[s].ids == ref.ids, tag + " stage " + std::to_string(s) + " ids differ from one rank");
            expect(r.res[s].dec_logits.size() == ref.dec_logits.size() &&
                   std::memcmp(r.res[s].dec_logits.data(), ref.dec_logits.data(), ref.dec_logits.size() * 4U) == 0,
                   tag + " stage " + std::to_string(s) + " decode logits bytes differ from one rank");
        }
        expect(r.state_bytes[0] == 7U * H * 8U && r.state_bytes[1] == 7U * H * 8U,
               tag + " state bytes: every layer once per rank (sent own + received peer's)");
        expect(r.head_rows.size() == prompt.size() * H && r.first_rows == r.head_rows, tag + " returned rows differ from the head's");
        expect(r.tail_bytes[0] == prompt.size() * H * 4U && r.tail_bytes[1] == r.tail_bytes[0], tag + " tail bytes");
        ++shared;
    }
    // pipelined speculation on the first stage: every guess policy gives the one-rank ids and decode logits bytes;
    // the stage-0 counters match the policy; a rewind that keeps the guessed state is detected.
    const unsigned sgen = 12;
    const auto sref = reference(m, prompt, width, sgen);
    std::vector<std::vector<std::uint32_t>> spec_plans;
    for (std::uint32_t k = 1; k < 7; ++k) spec_plans.push_back({0, k, 7});
    spec_plans.push_back({0, 2, 5, 7});
    unsigned spec_checked = 0;
    for (const auto& cuts : spec_plans)
        for (const auto g : {Guess::oracle, Guess::wrong, Guess::alternate, Guess::none}) {
            const auto r = run_chain(m, prompt, width, sgen, cuts, ~0ULL, ~0U, g, &sref.ids);
            const std::string tag = "spec cuts " + text(cuts) + " policy " + std::to_string(int(g));
            for (std::size_t s = 0; s < r.size(); ++s) {
                expect(r[s].error.empty(), tag + " stage " + std::to_string(s) + " error: " + r[s].error);
                expect(r[s].ids == sref.ids, tag + " stage " + std::to_string(s) + " ids differ from one rank");
            }
            const auto& dl = r.back().dec_logits;
            expect(dl.size() == sref.dec_logits.size() && std::memcmp(dl.data(), sref.dec_logits.data(), dl.size() * 4U) == 0,
                   tag + " last decode logits bytes differ from one rank");
            const std::uint64_t n = sgen - 2U;  // guesses: steps 1 .. sgen-2
            const std::uint64_t want_g = g == Guess::none ? 0U : n;
            const std::uint64_t want_h = g == Guess::oracle ? n : g == Guess::alternate ? (n + 1U) / 2U : 0U;
            expect(r[0].guesses == want_g && r[0].hits == want_h,
                   tag + " counters guesses " + std::to_string(r[0].guesses) + " hits " + std::to_string(r[0].hits));
            ++spec_checked;
        }
    {
        const auto r = run_chain(m, prompt, width, sgen, {0, 4, 7}, ~0ULL, ~0U, Guess::wrong, &sref.ids, true);
        const bool same = r.back().ids == sref.ids && r.back().dec_logits.size() == sref.dec_logits.size() &&
                          std::memcmp(r.back().dec_logits.data(), sref.dec_logits.data(), sref.dec_logits.size() * 4U) == 0;
        expect(!same, "negative control: a rewind that keeps the guessed row's state must change ids or logits");
    }
    // a stage with another model/plan identity is refused by name (and nothing hangs)
    const auto bad = run_chain(m, prompt, width, gen, {0, 3, 7}, 1U);
    expect(bad[1].error.find("seam header mismatch") != std::string::npos, "identity mismatch refused: got '" + bad[1].error + "'");
    // stage/cut validation
    bool threw = false;
    try { layers::Placement{{0, 3, 3, 7}, 0}.check(); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "non-increasing cuts refused");
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("PASS: layer pipeline over N ranks (%u plans, N=1..4, ids and logits bytes equal to one rank; "
                "%u relayed decodes equal; %u shared-state decodes + tail rows equal; %u pipelined-speculation runs equal, "
                "negative control detected)\n",
                checked, relayed, shared, spec_checked);
    return 0;
}
