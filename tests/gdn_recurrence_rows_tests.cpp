// Invariant of binding the shared chunked GDN recurrence core (qpf2k_gdn::gdn_qpf2k_cols_kernel) in the dims window
// path (hip_gdn_decode_step_dims): for the dims buffers (qkv / beta-gate / core with separate row strides), running the
// core's arithmetic on the first multiple-of-chunk rows (include/strix/gdn_recurrence_rows.hpp split / advance) and the
// incumbent recurrent_rows_kernel on the rest gives the same bits, output and state, as recurrent_rows_kernel on the
// whole window. Host models of both kernels (per value column, the kernels' operations in their order):
//   rows: 128 lanes = keys; stride-halving tree p[i] + p[i + s], s = 64..1; delta = (v - sum) * beta;
//         retained = fma(k, delta, retained) (the kernel's contracted `+=`).
//   cols: 4-lane groups, lane `sub` holds keys sub + 4 j (j < 32); in-lane tree p[j] + p[j + h], h = 16..1, then DPP
//         row_xmask 2 and 1 (own + partner); retained = fmaf(k, delta, retained).
#include "strix/gdn_recurrence_rows.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace gr = strix::gdn_recurrence;
constexpr std::uint32_t kD = 128, kG = 4, kK = kD / kG, kChunk = 8;
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
struct Dims {
    std::uint32_t key_heads, value_heads;
    std::uint32_t channels() const { return (2U * key_heads + value_heads) * kD; }
};
// Value head vh, key head vh % key_heads; qkv row layout [q | k | v].
const float* q_of(const gr::Rows& r, const Dims& m, std::uint32_t row, std::uint32_t vh) {
    return r.qkv + row * r.qkv_stride + std::uint64_t(vh % m.key_heads) * kD;
}
const float* k_of(const gr::Rows& r, const Dims& m, std::uint32_t row, std::uint32_t vh) {
    return q_of(r, m, row, vh) + std::uint64_t(m.key_heads) * kD;
}
float v_of(const gr::Rows& r, const Dims& m, std::uint32_t row, std::uint32_t vh, std::uint32_t col) {
    return r.qkv[row * r.qkv_stride + std::uint64_t(m.key_heads) * kD * 2U + std::uint64_t(vh) * kD + col];
}
// state layout key_value: [vh][key][col]
float& st(std::vector<float>& s, std::uint32_t vh, std::uint32_t key, std::uint32_t col) {
    return s[(std::size_t(vh) * kD + key) * kD + col];
}

float rows_tree(std::vector<float> p) {
    for (std::uint32_t s = kD / 2U; s > 0; s >>= 1U)
        for (std::uint32_t i = 0; i < s; ++i) p[i] = p[i] + p[i + s];
    return p[0];
}
void run_rows(const gr::Rows& r, const Dims& m, std::uint32_t rows, std::vector<float>& state) {
    std::vector<float> p(kD);
    for (std::uint32_t vh = 0; vh < m.value_heads; ++vh)
        for (std::uint32_t col = 0; col < kD; ++col)
            for (std::uint32_t row = 0; row < rows; ++row) {
                const float* q = q_of(r, m, row, vh);
                const float* k = k_of(r, m, row, vh);
                const float decay = std::exp(r.gate[row * r.scalar_stride + vh]);
                for (std::uint32_t key = 0; key < kD; ++key) {
                    float& s = st(state, vh, key, col);
                    s = s * decay;
                    p[key] = s * k[key];
                }
                const float delta = (v_of(r, m, row, vh, col) - rows_tree(p)) * r.beta[row * r.scalar_stride + vh];
                for (std::uint32_t key = 0; key < kD; ++key) {
                    float& s = st(state, vh, key, col);
                    s = std::fma(k[key], delta, s);
                    p[key] = s * q[key];
                }
                r.core[row * r.core_stride + std::uint64_t(vh) * kD + col] = rows_tree(p);
            }
}

// Sums of the 4 lanes of a group after the in-lane tree, then row_xmask 2 and 1: lane l adds its partner to its own.
void group_sum(float (&t)[kG]) {
    for (std::uint32_t x = kG / 2U; x > 0; x >>= 1U) {
        float n[kG];
        for (std::uint32_t l = 0; l < kG; ++l) n[l] = t[l] + t[l ^ x];
        std::memcpy(t, n, sizeof(n));
    }
}
float lane_tree(float (&p)[kK]) {
    for (std::uint32_t h = kK / 2U; h > 0; h >>= 1U)
        for (std::uint32_t j = 0; j < h; ++j) p[j] = p[j] + p[j + h];
    return p[0];
}
void run_cols(const gr::Rows& r, const Dims& m, std::uint32_t rows, std::vector<float>& state) {
    for (std::uint32_t vh = 0; vh < m.value_heads; ++vh)
        for (std::uint32_t col = 0; col < kD; ++col)
            for (std::uint32_t row = 0; row < rows; ++row) {
                const float* q = q_of(r, m, row, vh);
                const float* k = k_of(r, m, row, vh);
                const float decay = std::exp(r.gate[row * r.scalar_stride + vh]);
                const float beta = r.beta[row * r.scalar_stride + vh];
                float t[kG], p[kK];
                for (std::uint32_t sub = 0; sub < kG; ++sub) {
                    for (std::uint32_t j = 0; j < kK; ++j) {
                        float& s = st(state, vh, sub + j * kG, col);
                        s = s * decay;
                        p[j] = s * k[sub + j * kG];
                    }
                    t[sub] = lane_tree(p);
                }
                group_sum(t);
                for (std::uint32_t sub = 0; sub < kG; ++sub) {
                    const float delta = (v_of(r, m, row, vh, col) - t[sub]) * beta;  // each lane its own sum
                    for (std::uint32_t j = 0; j < kK; ++j) {
                        float& s = st(state, vh, sub + j * kG, col);
                        s = std::fma(k[sub + j * kG], delta, s);
                        p[j] = s * q[sub + j * kG];
                    }
                    t[sub] = lane_tree(p);
                }
                group_sum(t);
                r.core[row * r.core_stride + std::uint64_t(vh) * kD + col] = t[0];  // lane sub == 0 stores
            }
}

struct Window {
    std::vector<float> qkv, beta, gate, core, state;
};
Window make(const Dims& m, std::uint32_t rows, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    Window w;
    w.qkv.resize(std::size_t(rows) * m.channels());
    w.beta.resize(std::size_t(rows) * m.value_heads);
    w.gate.resize(std::size_t(rows) * m.value_heads);
    w.core.assign(std::size_t(rows) * m.value_heads * kD, 0.0F);
    w.state.resize(std::size_t(m.value_heads) * kD * kD);
    for (auto& x : w.qkv) x = n(rng);
    for (auto& x : w.beta) x = 1.0F / (1.0F + std::exp(-n(rng)));
    for (auto& x : w.gate) x = -std::fabs(n(rng));
    for (auto& x : w.state) x = 0.25F * n(rng);
    w.state[7] = -0.0F;
    return w;
}
gr::Rows view(Window& w, const Dims& m) {
    return gr::Rows{w.qkv.data(), w.beta.data(), w.gate.data(), w.core.data(), m.channels(), m.value_heads,
                    std::uint64_t(m.value_heads) * kD};
}

void split_rule() {
    check(gr::split(512, kChunk).bulk == 512 && gr::split(512, kChunk).tail == 0, "split 512: all chunked");
    check(gr::split(37, kChunk).bulk == 32 && gr::split(37, kChunk).tail == 5, "split 37: 32 + 5");
    check(gr::split(7, kChunk).bulk == 0 && gr::split(7, kChunk).tail == 7, "split 7: all per-row");
    check(gr::split(0, kChunk).bulk == 0 && gr::split(0, kChunk).tail == 0, "split 0 (switch off): nothing chunked");
    float buf[64] = {};
    const auto u = gr::uniform(buf, buf + 1, buf + 2, buf + 3, 10);
    const auto a = gr::advance(u, 3);
    check(u.qkv_stride == 10 && u.scalar_stride == 10 && u.core_stride == 10, "uniform: one stride for the four tensors");
    check(a.qkv == buf + 30 && a.beta == buf + 31 && a.gate == buf + 32 && a.core == buf + 33, "advance: rows first..");
}

void same_bits(const Dims& m, std::uint32_t rows, std::uint32_t seed) {
    const Window base = make(m, rows, seed);
    Window ref = base, cut = base;
    run_rows(view(ref, m), m, rows, ref.state);
    const auto part = gr::split(rows, kChunk);
    const auto all = view(cut, m);
    run_cols(all, m, part.bulk, cut.state);
    run_rows(gr::advance(all, part.bulk), m, part.tail, cut.state);
    check(std::memcmp(ref.core.data(), cut.core.data(), ref.core.size() * sizeof(float)) == 0, "core bits equal");
    check(std::memcmp(ref.state.data(), cut.state.data(), ref.state.size() * sizeof(float)) == 0, "state bits equal");
    bool moved = false;
    for (std::size_t i = 0; i < ref.state.size(); ++i) moved = moved || ref.state[i] != base.state[i];
    check(moved, "the model evolves the state (non-vacuous)");
}
}  // namespace

int main() {
    split_rule();
    same_bits(Dims{2, 4}, 37, 1118);  // 32 chunked + 5 per-row, value head -> key head vh % 2
    same_bits(Dims{1, 3}, 16, 1119);  // multiple of the chunk: no per-row tail
    same_bits(Dims{2, 2}, 5, 1120);   // shorter than a chunk: per-row only
    if (failures) return EXIT_FAILURE;
    std::puts("gdn_recurrence_rows_tests: PASS");
    return EXIT_SUCCESS;
}
