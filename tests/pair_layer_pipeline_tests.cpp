// N-rank layer pipeline (strix/pair/layer_pipeline.hpp) on a small CPU layer graph: every rank holds only its layers'
// sequence state (a recurrence like a GDN layer and a causal history like an attention layer), ranks run in threads
// connected by in-memory ordered links carrying the real shared seam pipe (strix/pair/pipe.hpp). Invariant: for any
// number of ranks and any cuts, the composed prefill logits and every decoded id equal the single-rank run byte for byte.
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
    Toy(const Weights& w, unsigned b, unsigned e, unsigned max_rows)
        : m(w), begin(b), end(e), rec(std::size_t(w.layers) * H), hist(std::size_t(w.layers) * H), x(std::size_t(max_rows) * H), logits(V) {}
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
        if (last == end) committed += rows;
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

struct Result { std::vector<float> logits; std::vector<std::uint32_t> ids; std::string error; };

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
    return r;
}

std::vector<Result> run_chain(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen,
                              const std::vector<std::uint32_t>& cuts, std::uint64_t bad_identity_stage = ~0ULL) {
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
                Toy t(m, cuts[s], cuts[s + 1U], width);
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
                layers::DecodeOps d;
                d.row_bytes = row_bytes;
                d.forward = [&](std::uint32_t tok, bool head) { t.range(p.first(), p.last(), &tok, 1U, head); };
                d.top = [&] { return t.top(); };
                d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), t.x.data(), row_bytes); };
                d.import_row = [&](std::span<const std::byte> b) { std::memcpy(t.x.data(), b.data(), row_bytes); };
                out[s].ids = layers::decode(p, p.head() ? t.top() : 0U, gen, up ? &lu : nullptr, down ? &ld : nullptr, d).ids;
            } catch (const std::exception& e) {
                out[s].error = e.what();
                if (s) links[s - 1U]->close();
                if (s + 1U < stages) links[s]->close();
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
    // a stage with another model/plan identity is refused by name (and nothing hangs)
    const auto bad = run_chain(m, prompt, width, gen, {0, 3, 7}, 1U);
    expect(bad[1].error.find("seam header mismatch") != std::string::npos, "identity mismatch refused: got '" + bad[1].error + "'");
    // stage/cut validation
    bool threw = false;
    try { layers::Placement{{0, 3, 3, 7}, 0}.check(); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "non-increasing cuts refused");
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("PASS: layer pipeline over N ranks (%u plans, N=1..4, ids and logits bytes equal to one rank)\n", checked);
    return 0;
}
