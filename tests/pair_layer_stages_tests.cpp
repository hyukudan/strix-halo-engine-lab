// Interleaved virtual stages of the shared layer pipeline (include/strix/pair/layer_pipeline.hpp prefill / decode over a
// StageMap): two ranks over loopback transports run 2v alternating stages of a toy engine whose every layer keeps its
// own sequence state; the head logits and the greedy ids must be BITWISE those of the single-rank loop at the same
// window width, for v = 1..3, uneven cuts and 1..9 chunks (more than one StagePipes group). Also: a half cut and a
// missing range hook are refused, and a seam from another model/plan identity stops both ranks.
#include "strix/pair/layer_pipeline.hpp"
#include "strix/pair/link_mux.hpp"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strix::pair;
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
}

constexpr unsigned H = 8, V = 13;
struct Weights {
    unsigned layers;
    std::vector<float> w, embed, head;
    explicit Weights(unsigned l) : layers(l), w(std::size_t(l) * H * H), embed(std::size_t(V) * H), head(std::size_t(V) * H) {
        std::uint32_t s = 777U;
        const auto next = [&] { s = s * 1664525U + 1013904223U; return float(int(s >> 9) % 2001 - 1000) / 1000.0F; };
        for (auto* v : {&w, &embed, &head}) for (auto& x : *v) x = next();
    }
};
// One rank's engine: touches only the layers it owns; every layer keeps its own recurrent state and row count.
struct Toy {
    const Weights& m;
    std::vector<bool> owned;
    std::vector<float> rec, hist, x, logits;
    std::vector<unsigned> committed;
    Toy(const Weights& w, std::vector<bool> own, unsigned max_rows)
        : m(w), owned(std::move(own)), rec(std::size_t(w.layers) * H), hist(std::size_t(w.layers) * H),
          x(std::size_t(max_rows) * H), logits(V), committed(w.layers) {}
    void range(unsigned first, unsigned last, const std::uint32_t* tokens, unsigned rows, bool head) {
        if (first == 0U)
            for (unsigned r = 0; r < rows; ++r) std::memcpy(&x[std::size_t(r) * H], &m.embed[std::size_t(tokens[r]) * H], H * 4U);
        for (unsigned l = first; l < last; ++l) {
            if (!owned[l]) throw std::logic_error("toy: layer not owned by this rank");
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
                    y[j] = acc + 0.5F * re[j] + 0.25F * hi[j] / float(committed[l] + r + 1U);
                }
                for (unsigned j = 0; j < H; ++j) xr[j] += 0.5F * std::tanh(y[j]);
            }
            committed[l] += rows;
        }
        if (head) {
            const float* xl = &x[std::size_t(rows - 1U) * H];
            for (unsigned v = 0; v < V; ++v) {
                float acc = 0.0F;
                for (unsigned j = 0; j < H; ++j) acc += m.head[std::size_t(v) * H + j] * xl[j];
                logits[v] = acc;
            }
        }
    }
    std::uint32_t top() const {
        std::uint32_t best = 0;
        for (std::uint32_t v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        return best;
    }
};

class Stream {   // one ordered byte stream of a loopback transport
public:
    void send(std::span<const std::byte> d) {
        std::lock_guard lock(mu_);
        if (aborted_) throw std::runtime_error("loopback aborted");
        q_.insert(q_.end(), d.begin(), d.end());
        cv_.notify_all();
    }
    void receive(std::span<std::byte> d) {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return aborted_ || q_.size() >= d.size(); });
        if (q_.size() < d.size()) throw std::runtime_error("loopback aborted");
        std::copy_n(q_.begin(), d.size(), d.begin());
        q_.erase(q_.begin(), q_.begin() + static_cast<std::ptrdiff_t>(d.size()));
    }
    void abort() { std::lock_guard lock(mu_); aborted_ = true; cv_.notify_all(); }
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::byte> q_;
    bool aborted_ = false;
};
struct Channel {   // full duplex: side 0 sends on ab, side 1 on ba
    Stream ab, ba;
    void abort() { ab.abort(); ba.abort(); }
    PipeLink end(unsigned side) {
        auto& out = side == 0U ? ab : ba;
        auto& in = side == 0U ? ba : ab;
        return {[&out](std::span<const std::byte> d) { out.send(d); }, [&in](std::span<std::byte> d) { in.receive(d); },
            [this] { abort(); }};
    }
    layers::Link link(unsigned side) {
        auto& out = side == 0U ? ab : ba;
        auto& in = side == 0U ? ba : ab;
        return {[&out](std::span<const std::byte> d) { out.send(d); }, [&in](std::span<std::byte> d) { in.receive(d); }};
    }
};

struct Result { std::vector<float> logits; std::vector<std::uint32_t> ids; std::string error; };

Result reference(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen) {
    Toy t(m, std::vector<bool>(m.layers, true), width);
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

// muxed: both directions of the stage pipes ride ONE link through strix::pair::LinkMux (channel 0 forward, 1 reverse,
// small pieces so frames are cut); after close() the decode runs on the same raw link.
std::vector<Result> run_pair(const Weights& m, const std::vector<std::uint32_t>& prompt, unsigned width, unsigned gen,
                             const StageMap& map, unsigned bad_identity_rank = 2U, bool muxed = false) {
    Channel forward, reverse;
    std::vector<Result> out(2);
    const auto chunks = layers::windows(prompt.size(), width).size();
    const std::size_t row_bytes = H * 4U;
    const auto body = [&](unsigned rank) {
        try {
            std::vector<bool> own(m.layers);
            for (unsigned l = 0; l < m.layers; ++l) own[l] = map.owns(rank, l);
            Toy t(m, own, width);
            const std::uint64_t identity = rank == bad_identity_rank ? 2U : 1U;
            PipeOptions options;
            options.validate = layers::stage_validator(identity, map, width, row_bytes);
            std::unique_ptr<strix::pair::LinkMux> mux;
            PipeLink fl = forward.end(rank), rl = reverse.end(rank);
            if (muxed) {
                const auto raw = forward.end(rank);
                mux = std::make_unique<strix::pair::LinkMux>(raw.send, raw.receive, [&](const std::string&) { forward.abort(); },
                                                             2U, std::size_t{24});
                for (std::uint32_t ch = 0; ch < 2U; ++ch) {
                    auto& l = ch == 0U ? fl : rl;
                    l.send = [&mux, ch](std::span<const std::byte> d) { mux->send(ch, d); };
                    l.receive = [&mux, ch](std::span<std::byte> d) { mux->receive(ch, d); };
                    l.abort = [&mux] { mux->abort("stage pipe abort"); };
                }
            }
            auto pipes = std::make_unique<StagePipes>(rank, map, chunks, fl, rl, options);
            layers::StageOps ops;
            ops.row_bytes = row_bytes;
            ops.range = [&](std::span<const std::uint32_t> tok, const layers::Window& w, std::uint32_t a, std::uint32_t b, bool head) {
                t.range(a, b, tok.data(), w.rows, head); };
            ops.seam = [&](PipeFrame& f, bool save, std::size_t bytes) {
                if (save) std::memcpy(f.payload.data(), t.x.data(), bytes); else std::memcpy(t.x.data(), f.payload.data(), bytes); };
            ops.fence = [] {};
            (void)layers::prefill(map, rank, prompt, width, identity, *pipes, ops);
            pipes.reset();
            if (mux) mux->close();
            const bool head_rank = map.owner(map.stages() - 1U) == rank;
            if (head_rank) out[rank].logits = t.logits;
            layers::DecodeOps d;
            d.row_bytes = row_bytes;
            d.range = [&](std::uint32_t tok, std::uint32_t a, std::uint32_t b, bool head) { t.range(a, b, &tok, 1U, head); };
            d.top = [&] { return t.top(); };
            d.export_row = [&](std::span<std::byte> b) { std::memcpy(b.data(), t.x.data(), row_bytes); };
            d.import_row = [&](std::span<const std::byte> b) { std::memcpy(t.x.data(), b.data(), row_bytes); };
            out[rank].ids = layers::decode(map, rank, head_rank ? t.top() : 0U, gen, forward.link(rank), d).ids;
        } catch (const std::exception& e) {
            out[rank].error = e.what();
            forward.abort();
            reverse.abort();
        }
    };
    std::thread r0(body, 0U), r1(body, 1U);
    r0.join();
    r1.join();
    return out;
}

std::vector<std::uint32_t> make_prompt(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint32_t> p(n);
    for (auto& t : p) { seed = seed * 22695477U + 1U; t = (seed >> 16) % V; }
    return p;
}
} // namespace

int main() {
    constexpr unsigned L = 12, gen = 6;
    const Weights m(L);
    const char* maps[] = {"0,6,12", "0,2,7,9,12", "0,1,3,6,8,11,12", "0,5,6,7,8,11,12"};
    struct Case { std::size_t prompt; unsigned width; };
    const Case cases[] = {{5, 8}, {37, 8}, {64, 16}, {70, 8}, {33, 4}};
    for (const char* text : maps) {
        const auto map = StageMap::parse(text, L);
        for (const auto& c : cases) {
            const auto prompt = make_prompt(c.prompt, static_cast<std::uint32_t>(c.prompt * 31U + c.width));
            const auto ref = reference(m, prompt, c.width, gen);
            const auto got = run_pair(m, prompt, c.width, gen, map);
            const std::string tag = std::string(text) + " prompt " + std::to_string(c.prompt) + " width " + std::to_string(c.width);
            expect(got[0].error.empty() && got[1].error.empty(), tag + ": ran (" + got[0].error + " | " + got[1].error + ")");
            expect(got[1].logits.size() == ref.logits.size() &&
                       std::memcmp(got[1].logits.data(), ref.logits.data(), ref.logits.size() * sizeof(float)) == 0,
                   tag + ": head logits bitwise == single rank");
            expect(got[0].ids == ref.ids && got[1].ids == ref.ids, tag + ": greedy ids == single rank on both ranks");
            const auto mux = run_pair(m, prompt, c.width, gen, map, 2U, true);
            expect(mux[0].error.empty() && mux[1].error.empty(), tag + " muxed: ran (" + mux[0].error + " | " + mux[1].error + ")");
            expect(mux[1].logits.size() == ref.logits.size() &&
                       std::memcmp(mux[1].logits.data(), ref.logits.data(), ref.logits.size() * sizeof(float)) == 0 &&
                       mux[0].ids == ref.ids && mux[1].ids == ref.ids,
                   tag + " muxed: one link carries both directions, logits bitwise and ids == single rank");
        }
    }
    {   // a seam from another identity stops both ranks (no hang)
        const auto map = StageMap::parse("0,2,7,9,12", L);
        const auto prompt = make_prompt(37, 5U);
        const auto got = run_pair(m, prompt, 8, gen, map, 1U);
        expect(!got[0].error.empty() && !got[1].error.empty(), "identity mismatch stops both ranks");
    }
    {   // the shared default map: v = 3 on uniform layers, v = 1 for one window, v lowered for few layers
        expect(strix::pair::default_stage_map(42U, 4U).text() == "0,7,14,21,28,35,42", "default 42 layers 4 windows = v3 uniform");
        expect(strix::pair::default_stage_map(42U, 1U).text() == "0,21,42", "default one window = two-stage cut");
        expect(strix::pair::default_stage_map(12U, 9U).virtual_stages() == 1U, "12 layers: stages of >= 4 layers -> v = 1");
        expect(strix::pair::default_stage_map(16U, 9U).virtual_stages() == 2U, "16 layers -> v = 2");
    }
    {   // refusals: half cut, missing range hook
        StageMap half;
        half.cuts = {StageCut(0U), StageCut(5U, true), StageCut(12U)};
        bool threw = false;
        try { layers::check_stage_map(half, 0U, true); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "half cut refused");
        threw = false;
        try { layers::check_stage_map(StageMap::parse("0,6,12", L), 0U, false); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "missing range hook refused");
    }
    if (failures == 0) std::printf("PASS pair layer stages (4 maps x 5 prompt/width cases, separate and muxed links + identity + refusals)\n");
    return failures == 0 ? 0 : 1;
}
