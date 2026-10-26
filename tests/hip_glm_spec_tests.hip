// S1 (development smoke of gate K for the spec kernels; NOT the full P1 K gate, which needs both ranks' slices of the
// admitted download): verify-row device kernels of ONE slot == the host reference bit for bit
// (design notes section 5).
//   * kda_verify (deferred replay), real L4 rank-0 KDA weights (32 heads): a stream of verify cycles with an enforced
//     schedule (every R = 1..8 with full and single-row commits, replay counts 0..8) then random cycles; rejected rows
//     carry junk inputs. Every new row's y (accepted or not) == spec::KdaVerify, the committed S + conv after every call
//     == host, the final flush == the plain kda_step sequence. Inactive rows of every buffer are re-poisoned (NaN) before
//     each call and must stay NaN.
//   * flag attribution: NaN in a rejected new row -> only row_flag[that row] (committed flag clear, committed state exact);
//     NaN in a row that is then committed -> row_flag at verify AND the committed flag at its replay.
//   * dsa_append_rows, real L3 indexer weights: prefill to 2,040 positions in enforced R = 1..8 chunks, then verify cycles
//     to beyond 2,240 (P > 512: sparse selection) with rollbacks on every pool phase; caches (latent, key, gate and pooled,
//     memcmp) == host after every commit; every row's token list and absorbed-MLA output (32 heads, real L3 kv_b) == host
//     while the later rows of the block are already stored; inactive input rows NaN-poisoned, scratch rows beyond R
//     stay NaN. C3 (a finite fp32 latent that rounds to bf16 inf) and C4 (a pool closed over an injected NaN gate)
//     attribute to exactly the row that caused them.
// Mutant builds (-DGLMSPEC_MUTANT=1..4, see glm_spec_dev.hip) must exit nonzero.
// usage: hip_glm_spec_tests CASES_DIR [all|kda|dsaflags]   (CASES_DIR = tools/glm/export_host_case.py output:
//        kda_l4_r0, mla_l3_r0; the mutant runs use the part their mutation touches)
#include "glm/host/glm_host.hpp"
#include "glm/spec/glm_spec_dev.hpp"
#include "glm/spec/glm_spec_host.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;
namespace D = strix::glm::dev;
namespace S = strix::glm::spec;

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> T* up(const std::vector<T>& h) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> void put(T* d, const std::vector<T>& h) { hok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d"); }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
template <class T> T* zalloc(std::size_t n) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    hok(hipMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "memset");
    return p;
}
template <class T> bool same(const T* a, const T* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(T)) == 0; }
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
bool all_nan(const float* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isnan(p[i])) return false;
    return true;
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
const float kNaN = __builtin_nanf("");

struct CaseDir {
    std::map<std::string, std::pair<std::string, std::vector<char>>> raw;  // name -> (dtype, bytes)
    explicit CaseDir(const std::string& dir) {
        std::ifstream m(dir + "/manifest.txt");
        if (!m) throw std::runtime_error("no manifest in " + dir);
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream ss(line);
            std::string name, dt;
            std::size_t count = 0;
            ss >> name >> dt >> count;
            const std::size_t es = dt == "u16" ? 2 : (dt == "f64" ? 8 : 4);
            std::vector<char> b(count * es);
            std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
            f.read(b.data(), static_cast<std::streamsize>(b.size()));
            if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + name);  // fail closed
            raw.emplace(name, std::make_pair(dt, std::move(b)));
        }
    }
    // fail closed on a missing tensor, a different dtype or a different element count than the test indexes
    template <class T> std::vector<T> v(const std::string& n, const char* dt, std::size_t count) const {
        const auto it = raw.find(n);
        if (it == raw.end()) throw std::runtime_error("fixture missing " + n);
        if (it->second.first != dt || it->second.second.size() != count * sizeof(T))
            throw std::runtime_error("fixture " + n + " has dtype " + it->second.first + " / " +
                                     std::to_string(it->second.second.size()) + " B, expected " + dt + " x " + std::to_string(count));
        std::vector<T> out(count);
        std::memcpy(out.data(), it->second.second.data(), it->second.second.size());
        return out;
    }
};

// ---------------------------------------------------------------- KDA verify
struct KdaFix {
    static constexpr unsigned Hh = 32, C = Hh * kKdaDim, MR = 8, SCR = Hh * 4 * kKdaDim;
    std::vector<std::uint16_t> conv, onorm;
    std::vector<float> A_log, dt, qkv, f, gate, b;
    D::KdaDev kd{};
    H::KdaWeights hw{};
    explicit KdaFix(const std::string& cases) {
        CaseDir c(cases + "/kda_l4_r0");
        conv = c.v<std::uint16_t>("conv", "u16", static_cast<std::size_t>(3) * C * kConvK);
        onorm = c.v<std::uint16_t>("o_norm", "u16", kKdaDim);
        A_log = c.v<float>("A_log", "f32", Hh);
        dt = c.v<float>("dt_bias", "f32", C);
        qkv = c.v<float>("qkv", "f32", static_cast<std::size_t>(12) * 3 * C);
        f = c.v<float>("f", "f32", static_cast<std::size_t>(12) * C);
        gate = c.v<float>("gate", "f32", static_cast<std::size_t>(12) * C);
        b = c.v<float>("b", "f32", static_cast<std::size_t>(12) * Hh);
        kd = D::KdaDev{Hh, up(conv), up(A_log), up(dt), up(onorm)};
        hw = H::KdaWeights{conv.data(), A_log.data(), dt.data(), H::Vec{onorm.data(), H::WType::BF16, kKdaDim}};
    }
    S::KdaRow row(unsigned t, bool real, std::mt19937& g) const {  // stream row t (12 real steps, then random) or junk
        S::KdaRow r;
        if (real && t < 12) {
            r.qkv.assign(qkv.begin() + static_cast<std::ptrdiff_t>(t) * 3 * C, qkv.begin() + static_cast<std::ptrdiff_t>(t + 1) * 3 * C);
            r.f.assign(f.begin() + static_cast<std::ptrdiff_t>(t) * C, f.begin() + static_cast<std::ptrdiff_t>(t + 1) * C);
            r.gate.assign(gate.begin() + static_cast<std::ptrdiff_t>(t) * C, gate.begin() + static_cast<std::ptrdiff_t>(t + 1) * C);
            r.b.assign(b.begin() + static_cast<std::ptrdiff_t>(t) * Hh, b.begin() + static_cast<std::ptrdiff_t>(t + 1) * Hh);
        } else {
            r.qkv = rnd(3 * C, g, 1.0F);
            r.f = rnd(C, g, 1.0F);
            r.gate = rnd(C, g, 1.0F);
            r.b = rnd(Hh, g, 1.0F);
        }
        return r;
    }
};

// Device-side buffers of one slot with the ping-pong replay sets.
struct KdaDev2 {
    static constexpr unsigned Hh = KdaFix::Hh, C = KdaFix::C, MR = KdaFix::MR, SCR = KdaFix::SCR;
    float *S = nullptr, *conv = nullptr, *qkv[2]{}, *scr[2]{}, *beta[2]{}, *f = nullptr, *g = nullptr, *b = nullptr, *o = nullptr, *y = nullptr;
    unsigned *flag = nullptr, *rflag = nullptr;
    explicit KdaDev2(const H::KdaState& s) {
        S = up(s.S);
        conv = up(s.conv);
        for (int i = 0; i < 2; ++i) {
            qkv[i] = zalloc<float>(static_cast<std::size_t>(MR) * 3 * C);
            scr[i] = zalloc<float>(static_cast<std::size_t>(MR) * SCR);
            beta[i] = zalloc<float>(MR * Hh);
        }
        f = zalloc<float>(static_cast<std::size_t>(MR) * C);
        g = zalloc<float>(static_cast<std::size_t>(MR) * C);
        b = zalloc<float>(MR * Hh);
        o = zalloc<float>(static_cast<std::size_t>(MR) * C);
        y = zalloc<float>(static_cast<std::size_t>(MR) * C);
        flag = zalloc<unsigned>(1);
        rflag = zalloc<unsigned>(MR);
    }
    // writes the new rows into set nxt with NaN in every inactive row (inputs and the outputs/scratch it must not touch)
    void load(unsigned nxt, const std::vector<S::KdaRow>& rows) {
        std::vector<float> iq(static_cast<std::size_t>(MR) * 3 * C, kNaN), ifv(static_cast<std::size_t>(MR) * C, kNaN), ig(ifv), ib(MR * Hh, kNaN);
        for (std::size_t j = 0; j < rows.size(); ++j) {
            std::copy(rows[j].qkv.begin(), rows[j].qkv.end(), iq.begin() + static_cast<std::ptrdiff_t>(j) * 3 * C);
            std::copy(rows[j].f.begin(), rows[j].f.end(), ifv.begin() + static_cast<std::ptrdiff_t>(j) * C);
            std::copy(rows[j].gate.begin(), rows[j].gate.end(), ig.begin() + static_cast<std::ptrdiff_t>(j) * C);
            std::copy(rows[j].b.begin(), rows[j].b.end(), ib.begin() + static_cast<std::ptrdiff_t>(j) * Hh);
        }
        put(qkv[nxt], iq);
        put(f, ifv);
        put(g, ig);
        put(b, ib);
        put(scr[nxt], std::vector<float>(static_cast<std::size_t>(MR) * SCR, kNaN));  // re-poison: stale rows of a larger R
        put(beta[nxt], std::vector<float>(MR * Hh, kNaN));
        put(y, std::vector<float>(static_cast<std::size_t>(MR) * C, kNaN));
        hok(hipMemset(rflag, 0, MR * 4), "rflag0");
    }
};

void test_kda_verify(const std::string& cases, std::mt19937& g) {
    const KdaFix fx(cases);
    constexpr unsigned C = KdaFix::C, MR = KdaFix::MR, SCR = KdaFix::SCR;
    std::vector<std::pair<unsigned, unsigned>> sched;  // (R, rows committed): every R with full and single commits
    for (unsigned R = 1; R <= MR; ++R) {
        sched.push_back({R, R});
        sched.push_back({R, 1});
    }
    unsigned N = 0;
    for (auto& s : sched) N += s.second;
    N += 40;  // then random cycles
    std::vector<S::KdaRow> stream;
    for (unsigned t = 0; t < N; ++t) stream.push_back(fx.row(t, true, g));
    const H::KdaState fresh{KdaFix::Hh, std::vector<float>(static_cast<std::size_t>(KdaFix::Hh) * kKdaDim * kKdaDim, 0.0F),
                            std::vector<float>(static_cast<std::size_t>(3) * C * 3, 0.0F)};
    S::KdaVerify hv(fresh);
    KdaDev2 d(fresh);
    unsigned pos = 0, cur = 0, n_rep = 0, cycles = 0, junk = 0;
    bool ys = true, states = true, pad = true, flags = true;
    std::set<unsigned> seenR, seenRep;
    while (pos < N) {
        unsigned R, m1;
        if (cycles < sched.size()) {
            R = sched[cycles].first;
            m1 = sched[cycles].second;
        } else {
            R = 1 + static_cast<unsigned>(g() % MR);
            m1 = 1 + static_cast<unsigned>(g() % R);
        }
        m1 = std::min(m1, N - pos);
        std::vector<S::KdaRow> rows;
        for (unsigned j = 0; j < R; ++j) rows.push_back(j < m1 ? stream[pos + j] : fx.row(0, false, g));
        junk += R - m1;
        const unsigned nxt = cur ^ 1U;
        d.load(nxt, rows);
        D::kda_verify(fx.kd, d.qkv[cur], d.scr[cur], d.beta[cur], n_rep, d.qkv[nxt], d.f, d.g, d.b, R, d.S, d.conv, d.scr[nxt],
                      d.beta[nxt], d.o, d.y, d.flag, d.rflag, nullptr);
        std::vector<std::vector<float>> yh;
        const unsigned first = hv.run(rows, fx.hw, yh);
        const auto yd = down(d.y, static_cast<std::size_t>(MR) * C);
        ys &= first == R;
        for (unsigned j = 0; j < R && j < first; ++j) ys &= same(yh[j].data(), yd.data() + static_cast<std::size_t>(j) * C, C);
        pad &= all_nan(yd.data() + static_cast<std::size_t>(R) * C, static_cast<std::size_t>(MR - R) * C);
        const auto sd = down(d.scr[nxt], static_cast<std::size_t>(MR) * SCR);
        pad &= all_nan(sd.data() + static_cast<std::size_t>(R) * SCR, static_cast<std::size_t>(MR - R) * SCR);
        const auto bd = down(d.beta[nxt], MR * KdaFix::Hh);
        pad &= all_nan(bd.data() + static_cast<std::size_t>(R) * KdaFix::Hh, static_cast<std::size_t>(MR - R) * KdaFix::Hh);
        states &= same(down(d.S, hv.committed().S.size()), hv.committed().S) && same(down(d.conv, hv.committed().conv.size()), hv.committed().conv);
        const auto rf = down(d.rflag, MR);
        flags &= std::all_of(rf.begin(), rf.end(), [](unsigned v) { return v == 0; }) && down(d.flag, 1)[0] == 0;
        seenR.insert(R);
        seenRep.insert(n_rep);
        hv.commit(m1);
        n_rep = m1;
        cur = nxt;
        pos += m1;
        ++cycles;
    }
    D::kda_verify(fx.kd, d.qkv[cur], d.scr[cur], d.beta[cur], n_rep, nullptr, nullptr, nullptr, nullptr, 0, d.S, d.conv, d.scr[cur ^ 1U],
                  d.beta[cur ^ 1U], d.o, d.y, d.flag, d.rflag, nullptr);  // flush
    hv.flush(fx.hw);
    H::KdaState plain = fresh;
    std::vector<float> y(C);
    for (unsigned t = 0; t < N; ++t) H::kda_step(plain, stream[t].in(), fx.hw, y);
    const bool final_ok = same(down(d.S, plain.S.size()), plain.S) && same(down(d.conv, plain.conv.size()), plain.conv) &&
                          same(hv.committed().S, plain.S) && same(hv.committed().conv, plain.conv);
    char bf[240];
    std::snprintf(bf, sizeof bf, "%u cycles, %u positions, %u junk rows, R seen %zu/8, replay counts seen %zu/9, real L4 rank-0 weights",
                  cycles, N, junk, seenR.size(), seenRep.size());
    check(ys && seenR.size() == 8 && seenRep.size() == 9, std::string("kda_verify: every new row's y == host (accepted and rejected; ") + bf + ")");
    check(states, "kda_verify: committed S + conv after every call == host KdaVerify (replay of the previous commit)");
    check(final_ok, "kda_verify: flush (n_new = 0) == host flush == plain kda_step sequence");
    check(pad, "kda_verify: inactive rows (re-poisoned before every call) untouched: y, prep scratch, beta stay NaN");
    check(flags, "kda_verify: no flag on finite inputs (NaN padding rows do not leak)");

    // query-only NaN: Q never reaches the recurrent state -> y non-finite, S finite, no C2 flag (row or committed)
    {
        std::vector<S::KdaRow> rows{fx.row(0, false, g)};
        rows[0].qkv[3] = kNaN;  // a Q channel of head 0
        d.load(0, rows);
        hok(hipMemset(d.flag, 0, 4), "f0");
        D::kda_verify(fx.kd, nullptr, nullptr, nullptr, 0, d.qkv[0], d.f, d.g, d.b, 1, d.S, d.conv, d.scr[0], d.beta[0], d.o, d.y,
                      d.flag, d.rflag, nullptr);
        const auto rf = down(d.rflag, MR);
        const auto yq = down(d.y, C);
        const bool y_nan = std::any_of(yq.begin(), yq.end(), [](float v) { return std::isnan(v); });
        const bool st = same(down(d.S, plain.S.size()), plain.S) && same(down(d.conv, plain.conv.size()), plain.conv);
        check(rf[0] == 0 && down(d.flag, 1)[0] == 0 && y_nan && st,
              "kda_verify: Q-only NaN -> non-finite y but no C2 flag, stored state untouched (the row is not committed)");
    }
    // flag attribution 1: NaN in rejected row 1 of 3 -> row_flag[1] (and the later row 2, fed by its conv state), the
    // committed flag stays clear; committing row 0 only leaves the state exact
    {
        std::vector<S::KdaRow> rows{fx.row(0, false, g), fx.row(0, false, g), fx.row(0, false, g)};
        rows[1].qkv[C + 5] = kNaN;  // a K channel: K reaches the state update (Q does not)
        d.load(0, rows);
        hok(hipMemset(d.flag, 0, 4), "f0");
        D::kda_verify(fx.kd, nullptr, nullptr, nullptr, 0, d.qkv[0], d.f, d.g, d.b, 3, d.S, d.conv, d.scr[0], d.beta[0], d.o, d.y,
                      d.flag, d.rflag, nullptr);
        const auto rf = down(d.rflag, MR);
        const unsigned cflag = down(d.flag, 1)[0];
        D::kda_verify(fx.kd, d.qkv[0], d.scr[0], d.beta[0], 1, nullptr, nullptr, nullptr, nullptr, 0, d.S, d.conv, d.scr[1], d.beta[1], d.o,
                      d.y, d.flag, d.rflag, nullptr);
        H::kda_step(plain, rows[0].in(), fx.hw, y);
        const bool st = same(down(d.S, plain.S.size()), plain.S) && same(down(d.conv, plain.conv.size()), plain.conv);
        check(rf[0] == 0 && rf[1] != 0 && cflag == 0 && down(d.flag, 1)[0] == 0 && st,
              "kda_verify: NaN in rejected row 1 -> row_flag[1] only, committed flag clear, committed row 0 state exact");
    }
    // flag attribution 2: NaN in row 0, which is then committed -> row_flag[0] at verify and the committed flag at replay
    {
        std::vector<S::KdaRow> rows{fx.row(0, false, g), fx.row(0, false, g)};
        rows[0].qkv[C + 7] = kNaN;  // a K channel
        d.load(0, rows);
        hok(hipMemset(d.flag, 0, 4), "f0");
        D::kda_verify(fx.kd, nullptr, nullptr, nullptr, 0, d.qkv[0], d.f, d.g, d.b, 2, d.S, d.conv, d.scr[0], d.beta[0], d.o, d.y,
                      d.flag, d.rflag, nullptr);
        const auto rf = down(d.rflag, MR);
        const unsigned before = down(d.flag, 1)[0];
        D::kda_verify(fx.kd, d.qkv[0], d.scr[0], d.beta[0], 1, nullptr, nullptr, nullptr, nullptr, 0, d.S, d.conv, d.scr[1], d.beta[1], d.o,
                      d.y, d.flag, d.rflag, nullptr);
        check(rf[0] != 0 && before == 0 && down(d.flag, 1)[0] != 0,
              "kda_verify: NaN in row 0 -> row_flag[0] at verify, committed flag (C2) set when the row is replayed");
    }
}

// ---------------------------------------------------------------- DSA rows of one slot
struct DsaFix {
    static constexpr unsigned Hh = 32;
    std::vector<std::uint16_t> wk, gate, wp, knw, knb, kvb;
    std::vector<float> ape;
    H::IndexerWeights hw{};
    D::IndexerDev dw{};
    explicit DsaFix(const std::string& cases) {
        CaseDir c(cases + "/mla_l3_r0");
        wk = c.v<std::uint16_t>("indexer.wk", "u16", static_cast<std::size_t>(kIdxDim) * kHidden);
        gate = c.v<std::uint16_t>("indexer.index_kpool_compress_gate", "u16", static_cast<std::size_t>(kIdxDim) * kHidden);
        wp = c.v<std::uint16_t>("indexer.weights_proj", "u16", static_cast<std::size_t>(kIdxHeads) * kHidden);
        knw = c.v<std::uint16_t>("indexer.k_norm", "u16", kIdxDim);
        knb = c.v<std::uint16_t>("indexer.k_norm.bias", "u16", kIdxDim);
        ape = c.v<float>("indexer.index_kpool_compress_ape", "f32", static_cast<std::size_t>(kKpool) * kIdxDim);
        const auto kvb_all = c.v<std::uint16_t>("kv_b_proj", "u16", static_cast<std::size_t>(kMlaHeads) * 2 * kMlaDim * kKvLora);
        kvb.assign(kvb_all.begin(), kvb_all.begin() + static_cast<std::ptrdiff_t>(Hh) * 2 * kMlaDim * kKvLora);
        hw = H::IndexerWeights{H::Mat{wk.data(), H::WType::F16, kIdxDim, kHidden}, H::Mat{gate.data(), H::WType::F16, kIdxDim, kHidden},
                               H::Mat{wp.data(), H::WType::F16, kIdxHeads, kHidden}, H::Vec{knw.data(), H::WType::F16, kIdxDim},
                               H::Vec{knb.data(), H::WType::F16, kIdxDim}, ape.data()};
        dw = D::IndexerDev{up(wk), up(gate), up(wp), up(knw), up(knb), up(ape)};
    }
};
D::DsaSlot new_slot(unsigned cap) {
    return D::DsaSlot{zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kKvLora), zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim),
                      zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim), zalloc<float>(static_cast<std::size_t>(cap / kKpool) * kIdxDim), cap};
}

void test_dsa_rows(const std::string& cases, std::mt19937& g) {
    const DsaFix fx(cases);
    constexpr unsigned Hh = DsaFix::Hh, cap = 3072, PRE = 2040, END = 2240, MR = 8;
    const D::DsaSlot slot = new_slot(cap);
    float *dx = zalloc<float>(static_cast<std::size_t>(MR) * kHidden), *dl = zalloc<float>(static_cast<std::size_t>(MR) * kKvLora);
    float *dks = zalloc<float>(MR * kIdxDim), *dgs = zalloc<float>(MR * kIdxDim);
    unsigned* rflag = zalloc<unsigned>(MR);
    float *dwh = zalloc<float>(kIdxHeads), *dsc = zalloc<float>(cap / kKpool), *dq = zalloc<float>(Hh * kMlaDim);
    float *dqt = zalloc<float>(Hh * kKvLora), *dqI = zalloc<float>(kIdxHeads * kIdxDim), *dxr = zalloc<float>(kHidden), *dout = zalloc<float>(Hh * kMlaDim);
    std::uint32_t *dwork = zalloc<std::uint32_t>(cap), *dtok = zalloc<std::uint32_t>(kIdxTopk + 3), *dn = zalloc<std::uint32_t>(1);
    std::uint16_t* dkvb = up(fx.kvb);
    const H::Mat hkvb{fx.kvb.data(), H::WType::F16, Hh * 2 * kMlaDim, kKvLora};
    H::DsaCache hc;
    auto cache_eq = [&](unsigned n) {
        return hc.tokens() == n && hc.pools() == n / kKpool && same(down(slot.latent, static_cast<std::size_t>(n) * kKvLora), hc.latent) &&
               same(down(slot.ikey, static_cast<std::size_t>(n) * kIdxDim), hc.ikey) &&
               same(down(slot.igate, static_cast<std::size_t>(n) * kIdxDim), hc.igate) &&
               same(down(slot.pooled, static_cast<std::size_t>(n / kKpool) * kIdxDim), hc.pooled);  // memcmp: -0 != +0
    };
    unsigned t = 0, cycles = 0, chunks = 0, rows_checked = 0, cross_pool = 0, sparse_rows = 0;
    unsigned phase[4] = {0, 0, 0, 0};
    std::set<unsigned> seenR, seenSpecR;
    bool caches = true, lists = true, outs = true, pad = true, flags = true;
    while (t < END) {
        const bool spec = t >= PRE;
        const unsigned R = spec ? (cycles < MR ? cycles + 1 : 1 + static_cast<unsigned>(g() % MR))
                                : (chunks < MR ? chunks + 1 : 1 + static_cast<unsigned>(g() % MR));
        const unsigned m1 = spec ? (cycles < MR ? (cycles % 2 ? R : 1) : 1 + static_cast<unsigned>(g() % R)) : R;
        std::vector<float> X(static_cast<std::size_t>(MR) * kHidden, kNaN), L(static_cast<std::size_t>(MR) * kKvLora, kNaN);
        for (unsigned j = 0; j < R; ++j) {
            const auto x = rnd(kHidden, g, 0.5F), l = rnd(kKvLora, g, 1.0F);
            std::copy(x.begin(), x.end(), X.begin() + static_cast<std::ptrdiff_t>(j) * kHidden);
            std::copy(l.begin(), l.end(), L.begin() + static_cast<std::ptrdiff_t>(j) * kKvLora);
        }
        put(dx, X);  // rows >= R are NaN: they must not be read
        put(dl, L);
        put(dks, std::vector<float>(MR * kIdxDim, kNaN));
        put(dgs, std::vector<float>(MR * kIdxDim, kNaN));
        D::dsa_append_rows(slot, t, R, dl, dx, fx.dw, dks, dgs, rflag, nullptr);
        const unsigned hf = S::dsa_append_rows(hc, L, X, kHidden, fx.hw, R);
        const auto ks = down(dks, MR * kIdxDim), gs = down(dgs, MR * kIdxDim);
        pad &= hf == R && all_nan(ks.data() + R * kIdxDim, (MR - R) * kIdxDim) && all_nan(gs.data() + R * kIdxDim, (MR - R) * kIdxDim);
        if (spec) {  // every row's query sees exactly positions <= t + j although rows > j are already stored
            for (unsigned j = 0; j < R; ++j) {
                const auto qI = rnd(kIdxHeads * kIdxDim, g, 1.0F), q = rnd(Hh * kMlaDim, g, 1.0F);
                const std::vector<float> xr(X.begin() + static_cast<std::ptrdiff_t>(j) * kHidden, X.begin() + static_cast<std::ptrdiff_t>(j + 1) * kHidden);
                put(dqI, qI);
                put(dxr, xr);
                put(dq, q);
                D::indexer_select(slot, t + j, dqI, dxr, fx.dw, dwh, dsc, dwork, dtok, dn, nullptr);
                const auto n = down(dn, 1)[0];
                const auto ref = H::indexer_select(hc, t + j, qI, xr, fx.hw);
                lists &= down(dtok, n) == ref;
                sparse_rows += (t + j + 1) / kKpool > kIdxPools;
                D::mla_attention(slot, dtok, dn, dq, dkvb, Hh, dqt, dout, nullptr);
                std::vector<float> oh(Hh * kMlaDim);
                H::mla_attention(hc, ref, q, hkvb, Hh, oh);
                outs &= same(down(dout, oh.size()).data(), oh.data(), oh.size());
                ++rows_checked;
            }
            if (m1 < R && (t + R - 1) / kKpool > (t + m1) / kKpool) ++cross_pool;
            hc.truncate(t + m1);
            ++phase[(t + m1) % kKpool];
            seenSpecR.insert(R);
            ++cycles;
        } else {
            seenR.insert(R);
            ++chunks;
        }
        t += m1;
        caches &= cache_eq(t);
    }
    const auto rf = down(rflag, MR);
    flags = std::all_of(rf.begin(), rf.end(), [](unsigned v) { return v == 0; });
    char bf[260];
    std::snprintf(bf, sizeof bf, "(prefill to %u in %u chunks, %u verify cycles to %u, R seen %zu/8 + %zu/8, %u rollbacks across a pool "
                                 "boundary, frontier phases %u/%u/%u/%u)",
                  PRE, chunks, cycles, t, seenR.size(), seenSpecR.size(), cross_pool, phase[0], phase[1], phase[2], phase[3]);
    check(caches && seenR.size() == 8 && seenSpecR.size() == 8 && cross_pool > 0 && phase[0] && phase[1] && phase[2] && phase[3],
          std::string("dsa_append_rows: latent/key/gate/pooled bytes == host after every commit ") + bf);
    check(lists && sparse_rows > 0, "rows of one slot: token list of row j at t0 + j == host (" + std::to_string(rows_checked) + " rows, " +
                                         std::to_string(sparse_rows) + " with P > 512 pools: sparse selection)");
    check(outs, "rows of one slot: absorbed MLA output (32 rank heads, real L3 kv_b) == host for every row");
    check(pad, "dsa_append_rows: NaN input rows beyond R not read; key/gate scratch rows beyond R stay NaN");
    check(flags, "dsa_append_rows: no row flag on finite inputs");
}

void test_dsa_flags(const std::string& cases, std::mt19937& g) {
    const DsaFix fx(cases);
    constexpr unsigned MR = 8;
    float *dx = zalloc<float>(static_cast<std::size_t>(MR) * kHidden), *dl = zalloc<float>(static_cast<std::size_t>(MR) * kKvLora);
    float *dks = zalloc<float>(MR * kIdxDim), *dgs = zalloc<float>(MR * kIdxDim);
    unsigned* rflag = zalloc<unsigned>(MR);
    // C3 attribution: row 2 of 4 has a finite fp32 latent that rounds to bf16 +inf; only row_flag[2]; rows 0-1 == host
    {
        const D::DsaSlot s2 = new_slot(64);
        H::DsaCache h2;
        std::vector<float> X(static_cast<std::size_t>(MR) * kHidden, kNaN), L(static_cast<std::size_t>(MR) * kKvLora, kNaN);
        for (unsigned j = 0; j < 4; ++j) {
            const auto x = rnd(kHidden, g, 0.5F), l = rnd(kKvLora, g, 1.0F);
            std::copy(x.begin(), x.end(), X.begin() + static_cast<std::ptrdiff_t>(j) * kHidden);
            std::copy(l.begin(), l.end(), L.begin() + static_cast<std::ptrdiff_t>(j) * kKvLora);
        }
        L[static_cast<std::size_t>(2) * kKvLora + 9] = 3.4E38F;  // finite fp32 above the largest bf16
        put(dx, X);
        put(dl, L);
        hok(hipMemset(rflag, 0, MR * 4), "rf0");
        D::dsa_append_rows(s2, 0, 4, dl, dx, fx.dw, dks, dgs, rflag, nullptr);
        const unsigned hf = S::dsa_append_rows(h2, L, X, kHidden, fx.hw, 4);
        const auto r2 = down(rflag, MR);
        check(hf == 2 && r2[0] == 0 && r2[1] == 0 && r2[2] != 0 && r2[3] == 0 && same(down(s2.latent, 2 * kKvLora), h2.latent) &&
                  same(down(s2.ikey, 2 * kIdxDim), h2.ikey),
              "C3: a latent that rounds to bf16 inf flags exactly its row (host stops there); rows before it == host");
    }
    // C4 attribution: pool 0 closed by row 3 of a block over an injected NaN gate at position 1 -> row_flag[3] only
    {
        const D::DsaSlot s3 = new_slot(64);
        std::vector<float> X(static_cast<std::size_t>(MR) * kHidden, kNaN), L(static_cast<std::size_t>(MR) * kKvLora, kNaN);
        for (unsigned j = 0; j < 4; ++j) {
            const auto x = rnd(kHidden, g, 0.5F), l = rnd(kKvLora, g, 1.0F);
            std::copy(x.begin(), x.end(), X.begin() + static_cast<std::ptrdiff_t>(j) * kHidden);
            std::copy(l.begin(), l.end(), L.begin() + static_cast<std::ptrdiff_t>(j) * kKvLora);
        }
        put(dx, X);
        put(dl, L);
        hok(hipMemset(rflag, 0, MR * 4), "rf0");
        D::dsa_append_rows(s3, 0, 3, dl, dx, fx.dw, dks, dgs, rflag, nullptr);  // positions 0..2, no pool closed
        const std::uint16_t nan16 = 0x7FC0;
        hok(hipMemcpy(s3.igate + kIdxDim + 5, &nan16, 2, hipMemcpyHostToDevice), "inject");
        const auto r0 = down(rflag, MR);
        put(dx, std::vector<float>(X.begin() + 3 * kHidden, X.end()));
        put(dl, std::vector<float>(L.begin() + 3 * kKvLora, L.end()));
        D::dsa_append_rows(s3, 3, 1, dl, dx, fx.dw, dks, dgs, rflag + 3, nullptr);  // position 3 closes pool 0; its flag slot is 3
        const auto r1 = down(rflag, MR);
        check(std::all_of(r0.begin(), r0.end(), [](unsigned v) { return v == 0; }) && r1[3] != 0 && r1[0] == 0 && r1[1] == 0 && r1[2] == 0,
              "C4: a pool closed over a non-finite gate flags exactly the closing row");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR\n", argv[0]);
        return 2;
    }
    std::mt19937 g(20261025);
    try {
        const std::string part = argc > 2 ? argv[2] : "all";
        if (part != "all" && part != "kda" && part != "dsaflags") throw std::invalid_argument("part: all|kda|dsaflags");
        if (part == "all" || part == "kda") test_kda_verify(argv[1], g);
        if (part == "all") test_dsa_rows(argv[1], g);
        if (part == "all" || part == "dsaflags") test_dsa_flags(argv[1], g);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 2;
    }
    std::printf("checks: %d failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}
