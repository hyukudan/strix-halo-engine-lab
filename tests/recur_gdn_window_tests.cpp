// F9 gated delta window form (src/ops/recur/gated_delta_window.hpp): host emulation of its reduction trees and of the
// deferred commit, against the shared serial reference gated_delta_step_ref (ScalarGate).
// 1. key_sum (lane l holds keys l, l+32, l+64, l+96; in-lane (p0 + p2) + (p1 + p3), then cross-lane 16..1) is the
//    128-leaf halving tree, bit for bit.
// 2. sum128 (LDS strides 64 and 32, shuffles 16..1) is the same tree.
// 3. The scatter variant (column_sums: xor32 exchanges 16, 8, 4 with the lower-key subtree first, then 2..1, column c
//    rooted in lane (c&1)*16 + (c&2)*4 + (c&4)) gives every column the same tree, for 8 and 4 columns.
// 4. A run of windows with deferred commit (each window commits the previous window's rows, without output, before its
//    own rows) leaves the state and the outputs of a serial row-by-row run.
// Build: -ffp-contract=off -fno-fast-math (device contraction is proven by ISA comparison of the binders, not here).
#include "ops/recur/gated_delta_policy.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace recur = strix::ops::recur;

constexpr std::uint32_t kD = 128;
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}
bool same(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

float halving_tree(std::array<float, kD> p) {
    for (std::uint32_t h = kD / 2U; h > 0; h >>= 1U)
        for (std::uint32_t l = 0; l < h; ++l) p[l] = p[l] + p[l + h];
    return p[0];
}
// Cross-lane read of lane l + d inside a 16-lane row (DPP row_shl), lane l ^ 16 for d = 16 (row exchange); a lane whose
// source is outside the row keeps its own value (only the root lanes are consumed).
float down(const std::array<float, 32>& v, std::uint32_t lane, std::uint32_t d) {
    if (d == 16U) return v[lane ^ 16U];
    return (lane % 16U) + d < 16U ? v[lane + d] : v[lane];
}
float key_sum(const std::array<float, kD>& p) {
    std::array<float, 32> s{};
    for (std::uint32_t l = 0; l < 32U; ++l) s[l] = (p[l] + p[l + 64U]) + (p[l + 32U] + p[l + 96U]);
    for (std::uint32_t d = 16U; d; d /= 2U) {
        std::array<float, 32> n{};
        for (std::uint32_t l = 0; l < 32U; ++l) n[l] = s[l] + down(s, l, d);
        s = n;
    }
    return s[0];
}
float sum128(const std::array<float, kD>& p) {
    std::array<float, kD> scratch = p;
    for (std::uint32_t l = 0; l < 64U; ++l) scratch[l] = scratch[l] + scratch[l + 64U];
    std::array<float, 32> s{};
    for (std::uint32_t l = 0; l < 32U; ++l) s[l] = scratch[l] + scratch[l + 32U];
    for (std::uint32_t d = 16U; d; d /= 2U) {
        std::array<float, 32> n{};
        for (std::uint32_t l = 0; l < 32U; ++l) n[l] = l + d < 32U ? s[l] + s[l + d] : s[l];
        s = n;
    }
    return s[0];
}
// columns[c][key] -> per-column roots through the scatter tree.
template <std::uint32_t C> std::array<float, C> column_sums(const std::vector<std::array<float, kD>>& columns) {
    std::array<std::array<float, C>, 32> sums{};
    for (std::uint32_t l = 0; l < 32U; ++l)
        for (std::uint32_t c = 0; c < C; ++c) {
            const auto& p = columns[c];
            sums[l][c] = (p[l] + p[l + 64U]) + (p[l + 32U] + p[l + 96U]);
        }
    const auto scatter = [&](std::uint32_t distance, std::uint32_t count) {
        auto next = sums;
        for (std::uint32_t l = 0; l < 32U; ++l)
            for (std::uint32_t c = 0; c < count / 2U; ++c) {
                const bool upper = (l & distance) != 0U;
                const float local = sums[l][2U * c + (upper ? 1U : 0U)];
                // xor32 delivers the value the source lane (l ^ distance, opposite half) selected with its own flag.
                const float peer = sums[l ^ distance][2U * c + (upper ? 1U : 0U)];
                next[l][c] = upper ? peer + local : local + peer;
            }
        sums = next;
    };
    scatter(16U, C);
    scatter(8U, C / 2U);
    if constexpr (C == 8U) scatter(4U, 2U);
    std::array<float, 32> sum{};
    for (std::uint32_t l = 0; l < 32U; ++l) sum[l] = sums[l][0];
    for (std::uint32_t d = (C == 4U ? 4U : 2U); d; d /= 2U) {
        std::array<float, 32> n{};
        for (std::uint32_t l = 0; l < 32U; ++l) n[l] = sum[l] + down(sum, l, d);
        sum = n;
    }
    std::array<float, C> out{};
    for (std::uint32_t c = 0; c < C; ++c) out[c] = sum[(c & 1U) * 16U + (c & 2U) * 4U + (c & 4U)];
    return out;
}

struct Row {
    std::vector<float> q, k, v;
    float beta, gate;
};
// Window step on one value head (state [key][column], all columns): the device expressions with key_sum.
void window_step(std::vector<float>& state, const Row& r, float* out) {
    const float decay = std::exp(r.gate);
    for (std::uint32_t col = 0; col < kD; ++col) {
        std::array<float, kD> p{};
        for (std::uint32_t l = 0; l < kD; ++l) {
            state[l * kD + col] = state[l * kD + col] * decay;
            p[l] = state[l * kD + col] * r.k[l];
        }
        const float delta = (r.v[col] - key_sum(p)) * r.beta;
        for (std::uint32_t l = 0; l < kD; ++l) {
            state[l * kD + col] += r.k[l] * delta;
            p[l] = state[l * kD + col] * r.q[l];
        }
        if (out) out[col] = key_sum(p);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int trial = 0; trial < 2000; ++trial) {
        std::array<float, kD> p{};
        const float scale = std::ldexp(1.0F, trial % 40 - 20);
        for (auto& x : p) x = normal(rng) * scale;
        const float ref = halving_tree(p);
        check(same(key_sum(p), ref), "key_sum == 128-leaf halving tree");
        check(same(sum128(p), ref), "sum128 == 128-leaf halving tree");
    }
    for (int trial = 0; trial < 400; ++trial) {
        std::vector<std::array<float, kD>> columns(8);
        for (auto& c : columns)
            for (auto& x : c) x = normal(rng);
        const auto eight = column_sums<8>(columns);
        const auto four = column_sums<4>(columns);
        for (std::uint32_t c = 0; c < 8U; ++c) check(same(eight[c], halving_tree(columns[c])), "scatter 8: column tree");
        for (std::uint32_t c = 0; c < 4U; ++c) check(same(four[c], halving_tree(columns[c])), "scatter 4: column tree");
    }
    // Deferred commit over windows of 3, 1, 4 and 2 rows (one value head, key head 0).
    const std::uint32_t sizes[] = {3U, 1U, 4U, 2U};
    std::vector<Row> rows;
    for (std::uint32_t w : sizes)
        for (std::uint32_t i = 0; i < w; ++i) {
            Row r{std::vector<float>(kD), std::vector<float>(kD), std::vector<float>(kD), 0.0F, 0.0F};
            for (std::uint32_t l = 0; l < kD; ++l) {
                r.q[l] = normal(rng) * 0.08F;
                r.k[l] = normal(rng) * 0.08F;
                r.v[l] = normal(rng);
            }
            r.beta = 1.0F / (1.0F + std::exp(-normal(rng)));
            r.gate = -std::fabs(normal(rng)) * 0.3F;
            rows.push_back(r);
        }
    std::vector<float> serial(kD * kD), windowed(kD * kD);
    for (auto& x : serial) x = normal(rng) * 0.1F;
    windowed = serial;
    std::vector<float> serial_out(rows.size() * kD), window_out(rows.size() * kD);
    const recur::ScalarGate policy{};
    for (std::size_t i = 0; i < rows.size(); ++i)
        recur::gated_delta_step_ref(policy, rows[i].q.data(), rows[i].k.data(), rows[i].v.data(), rows[i].beta,
                                    &rows[i].gate, 0U, kD, serial.data(), serial_out.data() + i * kD);
    // Window w runs the previous window's rows in a register copy (no output), then its own rows; the committed state is
    // the copy after the pending rows only (the window's own rows are committed by the next window).
    std::size_t begin = 0, pending_begin = 0, pending_count = 0;
    for (std::uint32_t w : sizes) {
        std::vector<float> regs = windowed;
        for (std::size_t i = 0; i < pending_count; ++i) window_step(regs, rows[pending_begin + i], nullptr);
        if (pending_count) windowed = regs;
        for (std::uint32_t i = 0; i < w; ++i) window_step(regs, rows[begin + i], window_out.data() + (begin + i) * kD);
        pending_begin = begin;
        pending_count = w;
        begin += w;
    }
    for (std::size_t i = 0; i < pending_count; ++i) window_step(windowed, rows[pending_begin + i], nullptr);
    bool outputs = true, state = true;
    for (std::size_t i = 0; i < serial_out.size(); ++i) outputs = outputs && same(serial_out[i], window_out[i]);
    for (std::size_t i = 0; i < serial.size(); ++i) state = state && same(serial[i], windowed[i]);
    check(outputs, "deferred windows: outputs == serial reference");
    check(state, "deferred windows: committed state == serial reference");
    if (failures) {
        std::printf("recur gdn window tests: %d failures\n", failures);
        return 1;
    }
    std::printf("recur gdn window tests: PASS\n");
    return 0;
}
