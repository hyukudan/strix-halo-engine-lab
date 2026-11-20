// Host tests of the transposed butterfly halving (include/strix/cuda_core/attention_score_reduce.hpp), the dot reduction of
// the CUDA decode attention scores kernel (attention.cu attn_scores_kernel):
// 1. 32 host threads run reduce<N> in lockstep (a barrier per shuffle = a warp shuffle xor); every slot of every lane holds
//    exactly the bits of the per-dot xor butterfly of the contract (offsets 16, 8, 4, 2, 1, own + partner) for N = 1, 2, 3,
//    8, 12, 16, 24, 32 over random partials of mixed magnitude, signed zeros and infinities;
// 2. the slot map covers every dot, lanes of one replica group hold equal bits, and the scores kernel's PairMap (N = 2 GT,
//    GT 4 / 6 / 8) gives position = lane bit 16 and a fold lane / slot for every head;
// 3. sensitivity control: the data makes a different summation order (lane-sequential) change bits in most dots, so (1)
//    would catch a tree change.
#include <barrier>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "strix/cuda_core/attention_score_reduce.hpp"

namespace sr = strix::cuda_core::score_reduce;

namespace {

int g_fail = 0;
void check(bool ok, const char* what, int n, int c, unsigned lane) {
    if (!ok) {
        if (g_fail < 20) std::fprintf(stderr, "FAIL %s (N %d case %d lane %u)\n", what, n, c, lane);
        ++g_fail;
    }
}
std::uint32_t bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// Partials of one dot over the 32 lanes: products of mixed magnitude (as q * k sums), some exact zeros of both signs and,
// in a few cases, an infinity.
std::vector<float> partials(std::mt19937& rng, int c, int dot) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::uniform_int_distribution<int> e(-12, 12);
    std::vector<float> p(32);
    for (int l = 0; l < 32; ++l) p[static_cast<std::size_t>(l)] = std::ldexp(u(rng), e(rng));
    if (c % 7 == 3) p[static_cast<std::size_t>(dot % 32)] = (dot & 1) ? 0.0f : -0.0f;
    if (c % 11 == 5) {
        for (float& x : p) x = (dot & 1) ? -0.0f : 0.0f;
    }
    if (c % 13 == 7 && dot == 0) p[5] = std::numeric_limits<float>::infinity();
    return p;
}

float butterfly(std::vector<float> v) {  // the contract: lane l += lane l ^ off, off = 16 .. 1; all lanes end equal
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
        std::vector<float> nv(32);
        for (std::uint32_t l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ off];
        v = nv;
    }
    for (std::uint32_t l = 1; l < 32; ++l)
        if (bits(v[l]) != bits(v[0])) return std::numeric_limits<float>::quiet_NaN();
    return v[0];
}

template <int N>
int run(int cases) {
    std::mt19937 rng(0x5eed + N);
    // in[c][dot][lane], expected[c][dot]
    std::vector<std::vector<std::vector<float>>> in(static_cast<std::size_t>(cases));
    std::vector<std::vector<float>> want(static_cast<std::size_t>(cases), std::vector<float>(N));
    int order_sensitive = 0, compared = 0;
    for (int c = 0; c < cases; ++c) {
        for (int d = 0; d < N; ++d) {
            in[static_cast<std::size_t>(c)].push_back(partials(rng, c, d));
            const std::vector<float>& p = in[static_cast<std::size_t>(c)].back();
            want[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] = butterfly(p);
            float seq = 0.0f;
            for (float x : p) seq += x;
            ++compared;
            if (bits(seq) != bits(want[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)])) ++order_sensitive;
        }
    }
    // got[c][lane][slot]
    constexpr int C = sr::remaining(N);
    std::vector<float> got(static_cast<std::size_t>(cases) * 32 * C);
    float exch[32] = {};
    std::barrier sync(32);
    auto lane_body = [&](std::uint32_t lane) {
        auto xr = [&](float x, std::uint32_t off) {
            exch[lane] = x;
            sync.arrive_and_wait();
            const float r = exch[lane ^ off];
            sync.arrive_and_wait();
            return r;
        };
        auto add = [](float a, float b) { return a + b; };
        for (int c = 0; c < cases; ++c) {
            float v[N];
            for (int d = 0; d < N; ++d) v[d] = in[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)][lane];
            sr::reduce<N>(v, lane, xr, add);
            for (int i = 0; i < C; ++i) got[(static_cast<std::size_t>(c) * 32 + lane) * C + static_cast<std::size_t>(i)] = v[i];
        }
    };
    std::vector<std::thread> th;
    for (std::uint32_t l = 0; l < 32; ++l) th.emplace_back(lane_body, l);
    for (std::thread& t : th) t.join();

    for (int c = 0; c < cases; ++c) {
        std::vector<int> seen(N, 0);
        for (std::uint32_t lane = 0; lane < 32; ++lane)
            for (int i = 0; i < C; ++i) {
                const std::uint32_t idx = sr::slot_index(N, lane, i);
                check(idx < static_cast<std::uint32_t>(N), "slot index in range", N, c, lane);
                if (idx >= static_cast<std::uint32_t>(N)) continue;
                ++seen[idx];
                const float g = got[(static_cast<std::size_t>(c) * 32 + lane) * C + static_cast<std::size_t>(i)];
                check(bits(g) == bits(want[static_cast<std::size_t>(c)][idx]), "slot bits == per-dot butterfly", N, c, lane);
                const std::uint32_t rep = lane & ~sr::replica_mask(N);
                check(sr::slot_index(N, rep, i) == idx, "replica group holds the same dot", N, c, lane);
            }
        for (int d = 0; d < N; ++d) check(seen[static_cast<std::size_t>(d)] == 32 / (N / C), "every dot in 32 / (N / C) lanes", N, c, 0);
    }
    std::printf("score reduce N %2d: halvings %d, %d values per lane, %d cases x 32 lanes bitwise == butterfly; "
                "sequential order differs in %d / %d dots\n",
                N, sr::halvings(N), C, cases, order_sensitive, compared);
    if constexpr (N >= 8) check(order_sensitive * 2 > compared, "control: data is order-sensitive", N, 0, 0);
    return 0;
}

template <int GT, int P>
void score_map() {
    using M = sr::ScoreMap<GT, P>;
    for (std::uint32_t lane = 0; lane < 32; ++lane)
        for (int i = 0; i < M::C; ++i) {
            const std::uint32_t idx = sr::slot_index(M::N, lane, i);
            check(idx / GT == M::position(lane), "step position (lane bit 16 when P = 2)", GT, i, lane);
            check(idx % GT == M::head(lane, i), "step head", GT, i, lane);
        }
    for (int g = 0; g < GT; ++g) {
        check(M::fold_lane(g) < 32u && M::fold_slot(g) >= 0, "fold source exists", GT, g, 0);
        if (M::fold_lane(g) < 32u) check(M::head(M::fold_lane(g), M::fold_slot(g)) == static_cast<std::uint32_t>(g), "fold source holds g", GT, g, 0);
    }
    std::printf("score map GT %d P %d: N %d, %d slots per lane, replica mask %u, fold sources ok\n", GT, P, M::N, M::C, sr::replica_mask(M::N));
}

}  // namespace

int main() {
    run<1>(40);
    run<2>(40);
    run<3>(40);
    run<8>(120);
    run<12>(120);
    run<16>(120);
    run<24>(60);
    run<32>(60);
    score_map<4, 1>();
    score_map<4, 2>();
    score_map<6, 1>();
    score_map<6, 2>();
    score_map<8, 1>();
    score_map<8, 2>();
    static_assert(sr::score_positions(4) == 2 && sr::score_positions(6) == 2 && sr::score_positions(8) == 1);
    if (g_fail) {
        std::fprintf(stderr, "attention score reduce host tests: %d failures\n", g_fail);
        return 1;
    }
    std::printf("attention score reduce host tests: PASS\n");
    return 0;
}
