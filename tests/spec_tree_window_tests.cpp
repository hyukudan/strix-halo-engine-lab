// Token-tree window contract (strix/spec/tree_window.hpp): random trees from the shared controller's layout
// The serial and tree *host* arithmetic is a scheduling oracle: compile with strict non-contracted
// FP expressions (CMakeLists.txt and cmake/Qwen35Spec.cmake). With x86 FMA contraction, the same
// attend() expression can round one ULP differently after inlining in serial_row vs run_window;
// greedy_of intentionally hashes the float bits, turning this host artifact into a token mismatch.
// Device CUDA/HIP bitwise A/B checks remain a separate qualification gate.
// (Block::tree_rows) and accept walk (accept_tree), run through a host model of the target rules the CUDA core implements
// (RowMap::self K/V index, gdn_window leaf mask, gdn_commit_path, kv_move_row). Invariant: every row's output and the
// committed state after each step are bitwise those of SERIAL decoding of the same tokens -> emitted ids == serial.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/spec/block_drafter.hpp"
#include "strix/spec/tree_window.hpp"

namespace sp = strix::spec;

#define TW_CHECK(c, ...)                                         \
    do {                                                         \
        if (!(c)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                   \
            std::fprintf(stderr, "\n");                          \
            std::exit(1);                                        \
        }                                                        \
    } while (0)

namespace {

constexpr std::uint32_t kVocab = 23;
constexpr std::size_t kMaxPos = 4096;

float embed(std::uint32_t t) { return 0.25f + 0.013f * static_cast<float>(t) + 0.0007f * static_cast<float>((t * 7U) % 5U); }
std::uint32_t bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

// Committed target state: recurrent S + conv taps (oldest first) + one attention K cache. Order-sensitive float arithmetic,
// so any change of operand or order shows in the bits.
struct State {
    float S = 0.1f;
    std::array<float, 3> taps{0.f, 0.f, 0.f};
    std::vector<float> kc = std::vector<float>(kMaxPos + 64, 0.f);
    std::uint32_t pos = 0;
};

float conv(const std::array<float, 3>& t, float x) { return ((0.31f * t[0] + 0.27f * t[1]) + 0.23f * t[2]) + 0.19f * x; }
float step_s(float S, float c) { return S * 0.93f + c * 0.41f; }
float key_of(float s1, float c) { return s1 * 1.7f + c; }
// Attention over keys 0..len-1 in ascending order; key len-1 (the row's own) read from cache index own.
float attend(const std::vector<float>& kc, std::uint32_t len, std::uint32_t own) {
    float acc = 0.f;
    for (std::uint32_t k = 0; k < len; ++k) acc = acc * 0.9995f + kc[k == len - 1 ? own : k] * (1.0f + 0.001f * static_cast<float>(k % 13U));
    return acc;
}
std::uint32_t greedy_of(float y) { return (bits(y) * 2654435761U >> 7) % kVocab; }

// Serial decode of one token: output + state advance (the reference).
std::uint32_t serial_row(State& st, std::uint32_t tok) {
    const float x = embed(tok), c = conv(st.taps, x), s1 = step_s(st.S, c);
    st.kc[st.pos] = key_of(s1, c);
    const float y = attend(st.kc, st.pos + 1, st.pos) + s1;
    st.S = s1;
    st.taps = {st.taps[1], st.taps[2], x};
    ++st.pos;
    return greedy_of(y);
}

// Tree window under the target rules; returns greedy per row and leaves the window's own state (after non-leaf rows) in shadow.
struct Window {
    float shadow = 0.f;
    std::vector<float> x;  // raw inputs per row (the conv commit reads them)
};
std::vector<std::uint32_t> run_window(State& st, const sp::TreeRows& t, const sp::TreeWindow& w, Window& win) {
    std::vector<std::uint32_t> g(w.rows);
    float S = st.S;
    auto taps = st.taps;
    win.x.assign(w.rows, 0.f);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const float x = embed(t.tokens[r]), c = conv(taps, x), s1 = step_s(S, c);
        win.x[r] = x;
        st.kc[st.pos + w.index[r]] = key_of(s1, c);  // kv_store at the row's own index
        const float y = attend(st.kc, st.pos + w.pos[r] + 1, st.pos + w.index[r]) + s1;
        g[r] = greedy_of(y);
        if (!(w.leaf >> r & 1U)) {  // leaf: nothing advanced
            S = s1;
            taps = {taps[1], taps[2], x};
        }
    }
    win.shadow = S;
    return g;
}

void commit(State& st, const Window& win, const sp::PathCommit& pc) {
    if (pc.replay) {
        float S = st.S;
        auto taps = st.taps;
        for (std::uint32_t r = 0; r < pc.end; ++r) {
            if (pc.skip >> r & 1U) continue;
            S = step_s(S, conv(taps, win.x[r]));
            taps = {taps[1], taps[2], win.x[r]};
        }
        st.S = S;
    } else {
        st.S = win.shadow;
    }
    std::vector<float> all{st.taps[0], st.taps[1], st.taps[2]};  // conv taps: the last three of (old taps, accepted inputs)
    for (std::uint32_t r = 0; r < pc.end; ++r)
        if (!(pc.skip >> r & 1U)) all.push_back(win.x[r]);
    st.taps = {all[all.size() - 3], all[all.size() - 2], all[all.size() - 1]};
    if (pc.move) st.kc[st.pos + pc.to] = st.kc[st.pos + pc.from];  // kv_move_row
    st.pos += pc.accepted;
}

bool same_state(const State& a, const State& b) {
    if (bits(a.S) != bits(b.S) || a.pos != b.pos) return false;
    for (int i = 0; i < 3; ++i)
        if (bits(a.taps[static_cast<std::size_t>(i)]) != bits(b.taps[static_cast<std::size_t>(i)])) return false;
    for (std::uint32_t p = 0; p < a.pos; ++p)
        if (bits(a.kc[p]) != bits(b.kc[p])) return false;
    return true;
}

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    std::mt19937 rng(20261119U);
    auto draw = [&rng](std::uint32_t m) { return static_cast<std::uint32_t>(rng() % m); };  // 32-bit draw (GCC -Wconversion)
    std::uint64_t cases = 0, rows_total = 0, sib_rows = 0, sib_taken = 0, replays = 0, moves = 0;
    for (int seq = 0; seq < 60; ++seq) {
        // serial reference of the whole stream
        const std::uint32_t n_tok = 300;
        State ref;
        std::vector<std::uint32_t> ids{static_cast<std::uint32_t>(seq) % kVocab};
        for (std::uint32_t i = 0; i < n_tok + 40; ++i) ids.push_back(serial_row(ref, ids.back()));
        // speculative stream with random trees
        State st, ser;  // ser: serial decode of the emitted tokens, advanced with the stream
        std::vector<std::uint32_t> out{ids[0]};
        const double p_ok = std::uniform_real_distribution<double>(0.3, 0.95)(rng);
        const double q_alt = std::uniform_real_distribution<double>(0.0, 0.9)(rng);
        while (out.size() < n_tok) {
            const std::size_t at = out.size();  // ids[at] = target's next token after the anchor out.back()
            sp::Block b;
            const std::uint32_t n = 1U + draw(15U);
            bool on_path = true;
            for (std::uint32_t j = 0; j < n; ++j) {
                const std::uint32_t truth = ids[at + j];
                const bool hit = on_path && std::bernoulli_distribution(p_ok)(rng);
                const std::uint32_t d = hit ? truth : (truth + 1U + draw(kVocab - 1U)) % kVocab;
                std::uint32_t alt = (d + 1U + draw(kVocab - 1U)) % kVocab;
                if (!hit && on_path && std::bernoulli_distribution(q_alt)(rng)) alt = truth;
                if (alt == d) alt = (d + 1U) % kVocab;
                on_path = on_path && hit;
                b.drafts.push_back(d);
                b.alts.push_back(alt);
                const bool s = (n + 1U + b.siblings) < sp::kTreeWindowMaxRows && draw(2U) == 0U;
                b.sib.push_back(s ? 1U : 0U);
                b.siblings += s ? 1U : 0U;
            }
            sp::TreeRows t;
            b.tree_rows(out.back(), t);
            const auto w = sp::tree_window(t);
            TW_CHECK(w.rows == b.rows() && w.chain == n + 1U, "rows %u chain %u", w.rows, w.chain);
            // descriptor invariants: chain rows at their depth; leaves after the chain, distinct; ancestors contiguous
            std::uint32_t seen = 0;
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                TW_CHECK(w.index[r] < w.rows, "index %u >= rows", w.index[r]);
                TW_CHECK(!(seen >> w.index[r] & 1U), "K/V index %u used twice", w.index[r]);
                seen |= 1U << w.index[r];
                if (w.leaf >> r & 1U) TW_CHECK(w.index[r] >= w.chain && t.node[r] < 0, "leaf index");
                else TW_CHECK(w.index[r] == w.pos[r] && t.node[r] >= 0, "chain index != depth");
                std::uint32_t depth = 0;
                for (std::int32_t a = t.parent[r]; a >= 0; a = t.parent[static_cast<std::size_t>(a)]) {
                    TW_CHECK(!(w.leaf >> a & 1U), "a leaf is an ancestor");
                    TW_CHECK(w.index[static_cast<std::size_t>(a)] == w.pos[r] - 1U - depth, "ancestors not contiguous");
                    ++depth;
                }
                TW_CHECK(depth == w.pos[r], "depth %u != pos %u", depth, w.pos[r]);
            }
            // verify + accept + commit under the target rules
            Window win;
            const auto g = run_window(st, t, w, win);
            const auto acc = sp::accept_tree(t, g);
            const auto pc = sp::path_commit(w, acc.path);
            TW_CHECK(pc.accepted == acc.emitted.size(), "accepted %u emitted %zu", pc.accepted, acc.emitted.size());
            commit(st, win, pc);
            for (const auto e : acc.emitted) out.push_back(e);
            // ids == serial and the committed state == the serial state at the same length
            for (std::size_t i = at; i < out.size(); ++i) TW_CHECK(out[i] == ids[i], "seq %d id %zu: %u != serial %u", seq, i, out[i], ids[i]);
            for (std::size_t i = at - 1; i + 1 < out.size(); ++i) (void)serial_row(ser, out[i]);
            TW_CHECK(same_state(st, ser), "seq %d: committed state != serial after %zu tokens", seq, out.size());
            ++cases;
            rows_total += w.rows;
            sib_rows += b.siblings;
            sib_taken += acc.sibling ? 1U : 0U;
            replays += pc.replay ? 1U : 0U;
            moves += pc.move ? 1U : 0U;
        }
    }
    TW_CHECK(sib_taken > 50 && moves == sib_taken && replays > 0, "coverage: taken %llu moves %llu replays %llu",
             static_cast<unsigned long long>(sib_taken), static_cast<unsigned long long>(moves), static_cast<unsigned long long>(replays));

    // contract violations are refused
    {
        sp::Block b;
        b.drafts = {3, 4, 5};
        b.alts = {6, 7, 8};
        b.sib = {1, 0, 1};
        b.siblings = 2;
        sp::TreeRows t;
        b.tree_rows(1, t);  // rows: 0 a, 1 s1, 2 d1, 3 d2, 4 s3, 5 d3
        const auto w = sp::tree_window(t);
        TW_CHECK(w.leaf == ((1U << 1) | (1U << 4)) && w.index[1] == 4 && w.index[4] == 5 && w.index[5] == 3, "layout");
        const std::uint32_t full[] = {0, 2, 3, 5}, leaf1[] = {0, 1}, leaf3[] = {0, 2, 3, 4};
        auto pf = sp::path_commit(w, full);
        TW_CHECK(!pf.replay && !pf.move && pf.accepted == 4, "full chain path: keep the window state");
        auto p1 = sp::path_commit(w, leaf1);
        TW_CHECK(p1.replay && p1.move && p1.from == 4 && p1.to == 1 && p1.end == 2 && p1.skip == 0, "leaf 1 path");
        auto p3 = sp::path_commit(w, leaf3);
        TW_CHECK(p3.replay && p3.move && p3.from == 5 && p3.to == 3 && p3.end == 5 && p3.skip == 2U, "leaf 3 path");
        const std::uint32_t bad1[] = {0, 1, 2}, bad2[] = {0, 3}, bad3[] = {1};
        TW_CHECK(throws([&] { (void)sp::path_commit(w, bad1); }), "path through a leaf accepted");
        TW_CHECK(throws([&] { (void)sp::path_commit(w, bad2); }), "path skipping a chain row accepted");
        TW_CHECK(throws([&] { (void)sp::path_commit(w, bad3); }), "path not from the anchor accepted");
        sp::TreeRows sub = t;  // a child under a leaf (subtree): not this contract
        sub.tokens.push_back(9);
        sub.parent.push_back(4);
        sub.node.push_back(4);
        sub.depth.push_back(4);
        TW_CHECK(throws([&] { (void)sp::tree_window(sub); }), "subtree accepted");
    }
    std::printf("tree window: %llu random tree steps (%llu rows, %llu sibling rows, %llu taken, %llu replays) == serial: PASS\n",
                static_cast<unsigned long long>(cases), static_cast<unsigned long long>(rows_total),
                static_cast<unsigned long long>(sib_rows), static_cast<unsigned long long>(sib_taken),
                static_cast<unsigned long long>(replays));
    return 0;
}
