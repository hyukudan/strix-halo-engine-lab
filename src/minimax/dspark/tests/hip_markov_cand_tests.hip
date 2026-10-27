// ds::CandMarkov qualification on one GPU. Two instances (rank halves [0, 100096) and [100096, 200064), the
// engine's vocab_range) run in two threads with an in-memory symmetric exchange, on the real W1/W2 of a pack and random
// base logits. Gates per draw: (1) both ranks return the same 7 tokens; (2) they equal a CPU replica of the candidate-set
// chain (canonical top-C per half, H from the same float norms, markov_kernel's per-candidate arithmetic, `better`);
// (3) a nonfinite logit in one half makes BOTH ranks return false. Also prints how often the chain equals the full chain.
// Gates (1)-(3) run with both top-C kernels (STRIX_MM_MARKOV_TOPC v1 and v2, CandMarkov::set_topc).
// (4) select payload, v1 and v2 (CandMarkov::select_payload): per row the counts (C, or the fallback flag when the row has
// a nonfinite logit), the SET of ids with each id's score bits (ids sorted; v2 must emit them ascending) and the owned-H
// values equal a CPU replica of the canonical set, and the v1 and v2 normalized payloads are byte-identical. Fixtures:
// random; integer-quantized (ties everywhere); signed zeros at the threshold; all-equal; all +-0; many exact ties at the
// C-th value; random finite bit patterns (denormals, extremes) of both signs / negative only; NaN / +inf / -inf rows. At
// C (both halves), C = 1 (both halves) and C = vn (rank-1 half; rank 0's vn exceeds the peer half the constructor allows).
// Usage: strix-hip-mm-markov-cand-tests --pack model-dspark-exl3.safetensors [--draws N] [--C N] [--H N]
#include "markov_cand.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace ds = strix::minimax::dspark;

namespace {
constexpr std::uint32_t kVocab = 200064, kSplit = 782 * 128, kR = 256, kT = 7;
#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP %s at %d\n", hipGetErrorString(e_), __LINE__);               \
            std::exit(2);                                                                          \
        }                                                                                          \
    } while (0)

float bf(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }
bool better(float v, std::uint32_t i, float b, std::uint32_t bi) { return v > b || (v == b && i < bi); }
std::vector<std::uint16_t> read_tensor(const std::string& file, const std::string& name, std::size_t n) {
    std::ifstream f(file, std::ios::binary);
    std::uint64_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 8);
    std::string h(hl, '\0');
    f.read(h.data(), static_cast<std::streamsize>(hl));
    const auto p = h.find("\"" + name + "\"");
    if (p == std::string::npos) throw std::runtime_error("tensor not found: " + name);
    const auto q = h.find("data_offsets", p);
    const auto lb = h.find('[', q), cm = h.find(',', lb), rb = h.find(']', cm);
    const std::uint64_t a = std::stoull(h.substr(lb + 1, cm - lb - 1)), b = std::stoull(h.substr(cm + 1, rb - cm - 1));
    if (b - a != n * 2) throw std::runtime_error("tensor size: " + name);
    std::vector<std::uint16_t> v(n);
    f.seekg(static_cast<std::streamoff>(8 + hl + a));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 2));
    return v;
}
// lane chains of 8 + xor butterfly (lane 0), the device order
float lanes_reduce(const float (&lanes_in)[32]) {
    float lanes[32];
    std::memcpy(lanes, lanes_in, sizeof lanes);
    for (int d = 16; d != 0; d >>= 1) {
        float nx[32];
        for (unsigned l = 0; l < 32; ++l) nx[l] = lanes[l] + lanes[l ^ static_cast<unsigned>(d)];
        std::memcpy(lanes, nx, sizeof nx);
    }
    return lanes[0];
}
float score(const std::vector<std::uint16_t>& w1, const std::vector<std::uint16_t>& w2, std::uint32_t prev, std::uint32_t v, float base) {
    float lanes[32];
    for (unsigned l = 0; l < 32; ++l) {
        float acc = 0.0F;
        for (unsigned e = 0; e < 8; ++e) acc = acc + bf(w1[std::size_t{prev} * kR + 8 * l + e]) * bf(w2[std::size_t{v} * kR + 8 * l + e]);
        lanes[l] = acc;
    }
    return base + lanes_reduce(lanes);
}
std::vector<std::uint32_t> topc(const float* row, std::uint32_t v0, std::uint32_t v1, std::uint32_t C) {
    std::vector<std::uint32_t> ids;
    for (std::uint32_t v = v0; v < v1; ++v) ids.push_back(v);
    auto val = [&](std::uint32_t v) { return row[v] == 0.0F ? 0.0F : row[v]; };
    std::partial_sort(ids.begin(), ids.begin() + C, ids.end(),
                      [&](std::uint32_t x, std::uint32_t y) { return val(x) > val(y) || (val(x) == val(y) && x < y); });
    ids.resize(C);
    return ids;
}

std::uint32_t fbits(float x) { return __builtin_bit_cast(std::uint32_t, x); }
// Normalized select payload: per row counts, (id, score bits) sorted by id (rows with counts == C), hval bits.
struct NormPayload {
    std::uint32_t counts[kT];
    std::vector<std::pair<std::uint32_t, std::uint32_t>> sel[kT];
    std::vector<std::uint32_t> hval;
    bool operator==(const NormPayload& o) const {
        for (unsigned k = 0; k < kT; ++k)
            if (counts[k] != o.counts[k] || sel[k] != o.sel[k]) return false;
        return hval == o.hval;
    }
};
NormPayload normalize(const std::vector<unsigned char>& p, unsigned C, unsigned hmax, bool& ascending) {
    std::vector<std::uint32_t> w(p.size() / 4);
    std::memcpy(w.data(), p.data(), w.size() * 4);
    const std::size_t co = std::size_t{kT} * C, so = co + kT, ho = so + std::size_t{kT} * C;
    NormPayload n;
    ascending = true;
    for (unsigned k = 0; k < kT; ++k) {
        n.counts[k] = w[co + k];
        if (n.counts[k] != C) continue;
        for (unsigned i = 0; i < C; ++i) {
            n.sel[k].emplace_back(w[std::size_t{k} * C + i], w[so + std::size_t{k} * C + i]);
            if (i > 0 && w[std::size_t{k} * C + i] <= w[std::size_t{k} * C + i - 1]) ascending = false;
        }
        std::sort(n.sel[k].begin(), n.sel[k].end());
    }
    n.hval.assign(w.begin() + static_cast<std::ptrdiff_t>(ho), w.begin() + static_cast<std::ptrdiff_t>(ho + std::size_t{kT} * hmax));
    return n;
}
// CPU replica: rows [7][vn] of the local half starting at v0.
NormPayload expect_select(const std::vector<float>& rows, std::uint32_t v0, std::uint32_t vn, unsigned C, const std::vector<std::uint32_t>& own,
                          unsigned hmax) {
    NormPayload n;
    for (unsigned k = 0; k < kT; ++k) {
        const float* row = rows.data() + std::size_t{k} * vn;
        bool fin = true;
        for (std::uint32_t v = 0; v < vn; ++v) fin = fin && std::isfinite(row[v]);
        n.counts[k] = fin ? C : 0xffffffffU;
        if (fin) {
            const std::vector<std::uint32_t> ids = topc(row, 0, vn, C);
            for (const auto v : ids) n.sel[k].emplace_back(v0 + v, fbits(row[v]));
            std::sort(n.sel[k].begin(), n.sel[k].end());
        }
        for (unsigned i = 0; i < hmax; ++i) n.hval.push_back(i < own.size() ? fbits(row[own[i] - v0]) : fbits(0.0F));
    }
    return n;
}

struct Pipe {  // in-memory symmetric exchange between the two rank threads
    std::mutex m;
    std::condition_variable cv;
    std::vector<unsigned char> box[2];
    bool full[2] = {false, false};
    void exchange(int r, const void* out, void* in, std::size_t bytes) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !full[r]; });
        box[r].assign(static_cast<const unsigned char*>(out), static_cast<const unsigned char*>(out) + bytes);
        full[r] = true;
        cv.notify_all();
        cv.wait(l, [&] { return full[1 - r]; });
        std::memcpy(in, box[1 - r].data(), bytes);
        full[1 - r] = false;
        cv.notify_all();
    }
};
// Fixture fx (0 random, 1 adversarial, 2 nonfinite) for a half of n logits at set size Cc: rows [7][n].
const char* const kFxName[3] = {"random", "adversarial", "nonfinite"};
std::vector<float> fixture(int fx, int r, std::uint32_t n, unsigned Cc) {
    std::mt19937 g(100U + 7U * static_cast<unsigned>(fx) + static_cast<unsigned>(r));
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> rows(std::size_t{kT} * n);
    auto at = [&](unsigned k, std::uint32_t v) -> float& { return rows[std::size_t{k} * n + v]; };
    auto rbits = [&](bool neg) {
        std::uint32_t u = static_cast<std::uint32_t>(g());
        if (((u >> 23U) & 255U) == 255U) u &= ~(1U << 30U);  // finite
        if (neg) u |= 0x80000000U;
        return __builtin_bit_cast(float, u);
    };
    for (unsigned k = 0; k < kT; ++k)
        for (std::uint32_t v = 0; v < n; ++v) {
            float x = 4.0F * nd(g);
            if (fx == 0 && k == 3) x = 0.25F * nd(g);
            if (fx == 0 && k == 6) x = 100.0F * nd(g);
            if (fx == 1) {
                switch (k) {
                case 0: x = std::floor(2.0F * nd(g)); break;                              // ties everywhere
                case 1: x = v % 3 == 0 ? 0.0F : v % 3 == 1 ? -0.0F : -1.0F - std::fabs(nd(g)); break;  // +-0 at the threshold
                case 2: x = 1.5F; break;                                                    // all equal
                case 3: x = (v & 1U) ? -0.0F : 0.0F; break;                                 // all +-0
                case 5: x = rbits(false); break;                                            // any finite bits
                case 6: x = rbits(true); break;                                             // negative only
                default: break;
                }
            }
            at(k, v) = x;
        }
    if (fx == 1) {
        for (unsigned i = 0; i < Cc / 2; ++i) at(1, static_cast<std::uint32_t>((std::uint64_t{i} * 7919U) % n)) = 1.0F + std::fabs(nd(g));
        std::vector<float> c(rows.begin() + static_cast<std::ptrdiff_t>(4 * std::size_t{n}), rows.begin() + static_cast<std::ptrdiff_t>(5 * std::size_t{n}));
        std::nth_element(c.begin(), c.begin() + (Cc - 1), c.end(), std::greater<float>());
        const float X = c[Cc - 1];  // the C-th largest value; then many more exact ties with it
        for (std::uint32_t v = 0; v < n; v += 17) at(4, v) = X;
    }
    if (fx == 2) {
        at(0, n / 2) = -INFINITY;
        at(2, 5) = NAN;
        at(5, n - 1) = INFINITY;
    }
    return rows;
}
int select_gates(ds::CandMarkov& m0, ds::CandMarkov& m1, float* const* dl, const hipStream_t* st, const std::uint32_t* v0,
                 const std::uint32_t* vn, const std::vector<std::uint32_t>& H, unsigned C, unsigned NH, const std::uint16_t* dw1,
                 const std::uint16_t* dw2) {
    ds::CandMarkov c1a(dw1, dw2, kVocab, v0[0], vn[0], 1, NH), c1b(dw1, dw2, kVocab, v0[1], vn[1], 1, NH),
        cvn(dw1, dw2, kVocab, v0[1], vn[1], vn[1], NH);
    struct Inst {
        ds::CandMarkov* m;
        int r;
        unsigned c;
    };
    const Inst inst[5] = {{&m0, 0, C}, {&m1, 1, C}, {&c1a, 0, 1}, {&c1b, 1, 1}, {&cvn, 1, vn[1]}};
    int fails = 0;
    for (const Inst& in : inst) {
        std::vector<std::uint32_t> own;
        for (const auto h : H)
            if (h >= v0[in.r] && h < v0[in.r] + vn[in.r]) own.push_back(h);
        const unsigned hmax = in.m->hmax();
        if (hmax < own.size()) {
            std::printf("FAIL select r%d C %u: hmax %u < owned H %zu\n", in.r, in.c, hmax, own.size());
            ++fails;
            continue;
        }
        for (int fx = 0; fx < 3; ++fx) {
            const std::vector<float> rows = fixture(fx, in.r, vn[in.r], in.c);
            CK(hipMemcpy(dl[in.r], rows.data(), rows.size() * 4, hipMemcpyHostToDevice));
            const NormPayload want = expect_select(rows, v0[in.r], vn[in.r], in.c, own, hmax);
            NormPayload got[2];
            bool asc[2] = {false, false}, okk[2] = {false, false};
            for (int impl = 0; impl < 2; ++impl) {
                in.m->set_topc(impl ? ds::CandMarkov::TopC::v2 : ds::CandMarkov::TopC::v1);
                std::vector<unsigned char> p(in.m->payload_bytes());
                in.m->select_payload(dl[in.r], p.data(), st[in.r]);
                got[impl] = normalize(p, in.c, hmax, asc[impl]);
                okk[impl] = got[impl] == want;
                if (!okk[impl])
                    for (unsigned k = 0; k < kT; ++k)
                        if (got[impl].counts[k] != want.counts[k] || got[impl].sel[k] != want.sel[k]) {
                            std::size_t d = 0;
                            while (d < want.sel[k].size() && d < got[impl].sel[k].size() && want.sel[k][d] == got[impl].sel[k][d]) ++d;
                            std::printf("  v%d r%d C %u %s row %u: counts %u want %u, first differing sorted entry %zu\n", impl + 1, in.r, in.c,
                                        kFxName[fx], k, got[impl].counts[k], want.counts[k], d);
                            break;
                        }
            }
            const bool same = got[0] == got[1], good = okk[0] && okk[1] && same && asc[1];
            std::printf("%s select r%d C %u %s: v1 == cpu %d, v2 == cpu %d, v1 == v2 %d, v2 ids ascending %d\n", good ? "PASS" : "FAIL", in.r,
                        in.c, kFxName[fx], okk[0], okk[1], same, asc[1]);
            if (!good) ++fails;
        }
        in.m->set_topc(ds::CandMarkov::TopC::v1);
    }
    return fails;
}
}  // namespace

int main(int argc, char** argv) {
    std::string pack;
    int draws = 8;
    unsigned C = 1024, NH = 2048;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--pack") pack = argv[++i];
        else if (s == "--draws") draws = std::atoi(argv[++i]);
        else if (s == "--C") C = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (s == "--H") NH = static_cast<unsigned>(std::atoi(argv[++i]));
    }
    const std::string pre = "language_model.model.dspark.markov_head.";
    const auto w1 = read_tensor(pack, pre + "markov_w1.weight", std::size_t{kVocab} * kR);
    const auto w2 = read_tensor(pack, pre + "markov_w2.weight", std::size_t{kVocab} * kR);
    std::uint16_t *dw1, *dw2;
    CK(hipMalloc(&dw1, w1.size() * 2));
    CK(hipMalloc(&dw2, w2.size() * 2));
    CK(hipMemcpy(dw1, w1.data(), w1.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw2, w2.data(), w2.size() * 2, hipMemcpyHostToDevice));
    const std::uint32_t v0[2] = {0, kSplit}, vn[2] = {kSplit, kVocab - kSplit};
    ds::CandMarkov m0(dw1, dw2, kVocab, v0[0], vn[0], C, NH), m1(dw1, dw2, kVocab, v0[1], vn[1], C, NH);
    // CPU H: the device norm order (lane chains of 8 squares, butterfly), sorted (desc, id asc)
    std::vector<float> nrm(kVocab);
    for (std::uint32_t v = 0; v < kVocab; ++v) {
        float lanes[32];
        for (unsigned l = 0; l < 32; ++l) {
            float acc = 0.0F;
            for (unsigned e = 0; e < 8; ++e) {
                const float x = bf(w2[std::size_t{v} * kR + 8 * l + e]);
                acc = acc + x * x;
            }
            lanes[l] = acc;
        }
        nrm[v] = lanes_reduce(lanes);
    }
    std::vector<std::uint32_t> order(kVocab);
    for (std::uint32_t v = 0; v < kVocab; ++v) order[v] = v;
    std::partial_sort(order.begin(), order.begin() + NH, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return nrm[a] > nrm[b] || (nrm[a] == nrm[b] && a < b); });
    const std::vector<std::uint32_t> H(order.begin(), order.begin() + NH);
    float* dl[2];
    for (int r = 0; r < 2; ++r) CK(hipMalloc(&dl[r], std::size_t{kT} * vn[r] * 4));
    hipStream_t st[2];
    for (auto& s : st) CK(hipStreamCreate(&s));
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> base(std::size_t{kT} * kVocab);
    int fails = 0, same_full = 0;
    for (int impl = 0; impl < 2; ++impl) {
    const auto tc = impl ? ds::CandMarkov::TopC::v2 : ds::CandMarkov::TopC::v1;
    m0.set_topc(tc);
    m1.set_topc(tc);
    std::mt19937 g(5);  // same draws for both kernels
    for (int draw = 0; draw <= draws; ++draw) {
        const bool poison = draw == draws;  // last draw: a NaN in rank 1's half, row 3
        for (auto& x : base) x = 4.0F * nd(g);
        if (poison) base[std::size_t{3} * kVocab + kSplit + 17] = NAN;
        for (int r = 0; r < 2; ++r)
            for (unsigned k = 0; k < kT; ++k)
                CK(hipMemcpy(dl[r] + std::size_t{k} * vn[r], base.data() + std::size_t{k} * kVocab + v0[r], std::size_t{vn[r]} * 4, hipMemcpyHostToDevice));
        const std::uint32_t anchor = g() % kVocab;
        Pipe pipe;
        std::array<std::uint32_t, kT> out[2]{};
        bool ok[2] = {false, false};
        std::thread th[2];
        for (int r = 0; r < 2; ++r)
            th[r] = std::thread([&, r] {
                ds::CandMarkov& mc = r ? m1 : m0;
                ok[r] = mc.draft(dl[r], anchor, [&](const void* o, void* in, std::size_t b) { pipe.exchange(r, o, in, b); }, out[r], st[r]);
            });
        for (auto& t : th) t.join();
        if (poison) {
            if (ok[0] || ok[1]) {
                std::printf("FAIL topc v%d nonfinite: returns %d %d (both must be false)\n", impl + 1, ok[0], ok[1]);
                ++fails;
            } else {
                std::printf("PASS topc v%d nonfinite logit -> both ranks fall back\n", impl + 1);
            }
            break;
        }
        if (!ok[0] || !ok[1] || out[0] != out[1]) {
            std::printf("FAIL topc v%d draw %d: ranks disagree or fell back (%d %d)\n", impl + 1, draw, ok[0], ok[1]);
            ++fails;
            continue;
        }
        std::uint32_t t = anchor, tf = anchor;
        bool full_eq = true;
        for (unsigned k = 0; k < kT; ++k) {
            const float* row = base.data() + std::size_t{k} * kVocab;
            std::vector<std::uint32_t> cand = topc(row, 0, kSplit, C), c1 = topc(row, kSplit, kVocab, C);
            cand.insert(cand.end(), c1.begin(), c1.end());
            cand.insert(cand.end(), H.begin(), H.end());
            float bv = -INFINITY;
            std::uint32_t bi = 0xffffffffU;
            for (const auto v : cand) {
                const float s = score(w1, w2, t, v, row[v]);
                if (better(s, v, bv, bi)) { bv = s; bi = v; }
            }
            if (bi != out[0][k]) {
                std::printf("FAIL topc v%d draw %d step %u: device %u cpu %u\n", impl + 1, draw, k, out[0][k], bi);
                ++fails;
            }
            t = out[0][k];
            if (draw < 2 && impl == 0) {  // full chain (both halves) for the agreement statistic
                float fv = -INFINITY;
                std::uint32_t fi = 0xffffffffU;
                for (std::uint32_t v = 0; v < kVocab; ++v) {
                    const float s = score(w1, w2, tf, v, row[v]);
                    if (better(s, v, fv, fi)) { fv = s; fi = v; }
                }
                full_eq = full_eq && fi == out[0][k];
                tf = fi;
            }
        }
        if (draw < 2 && impl == 0 && full_eq) ++same_full;
    }
    std::printf("%s topc v%d chain gates\n", fails ? "FAIL-SO-FAR" : "PASS", impl + 1);
    }
    fails += select_gates(m0, m1, dl, st, v0, vn, H, C, NH, dw1, dw2);
    std::printf("info: candidate chain == full chain on %d of 2 synthetic draws (synthetic logits; acceptance is measured offline)\n", same_full);
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
