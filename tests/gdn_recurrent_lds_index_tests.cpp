// Invariant of the GDN recurrence LDS forms (include/strix/gdn_recurrent_lds_index.hpp): every form is a bijection of
// the 128 x 32 (lane, column) elements into its buffer, the column-major form puts the 32 lanes of a wave on 32
// distinct banks for every column (the lane-major form puts them on one), and a host model of recurrent_rows_kernel
// (same per-row operations, tree pairs and order) gives bitwise equal outputs and state in every form.
#include "strix/gdn_recurrent_lds_index.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
using strix::gdn_recurrent_lds::Form;
using strix::gdn_recurrent_lds::extent;
using strix::gdn_recurrent_lds::index;
using strix::gdn_recurrent_lds::kColumns;
using strix::gdn_recurrent_lds::kLanes;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr Form kForms[] = {Form::lane_major, Form::lane_major_pad, Form::column_major};

void bijection() {
    for (const Form f : kForms) {
        std::vector<int> hits(extent(f), 0);
        bool in_range = true;
        for (std::uint32_t lane = 0; lane < kLanes; ++lane)
            for (std::uint32_t column = 0; column < kColumns; ++column) {
                const auto i = index(f, lane, column);
                if (i >= hits.size()) in_range = false;
                else ++hits[i];
            }
        check(in_range, "index inside extent");
        std::size_t used = 0;
        bool once = true;
        for (const int h : hits) {
            used += h != 0 ? 1U : 0U;
            once = once && h <= 1;
        }
        check(once && used == std::size_t(kLanes) * kColumns, "each element has its own slot");
    }
}

// Banks (32 x 4 bytes) touched by the 32 lanes of each wave for one column.
unsigned banks(Form f, std::uint32_t wave, std::uint32_t column) {
    std::uint32_t mask = 0;
    for (std::uint32_t l = 0; l < 32U; ++l) mask |= 1U << (index(f, 32U * wave + l, column) % 32U);
    unsigned n = 0;
    for (; mask; mask &= mask - 1U) ++n;
    return n;
}

void bank_spread() {
    bool column_major_free = true, lane_major_one = true;
    for (std::uint32_t w = 0; w < kLanes / 32U; ++w)
        for (std::uint32_t c = 0; c < kColumns; ++c) {
            column_major_free = column_major_free && banks(Form::column_major, w, c) == 32U;
            lane_major_one = lane_major_one && banks(Form::lane_major, w, c) == 1U;
        }
    check(column_major_free, "column_major: 32 distinct banks per wave and column");
    check(lane_major_one, "lane_major: one bank per wave and column (the measured conflict)");
}

// Host model of recurrent_rows_kernel for one (value head, tile): lanes are the outer loop where the kernel runs them in
// parallel; each barrier phase is completed for all lanes before the next, as in the kernel.
struct Rows {
    std::vector<float> qkv, beta, gate, state, core;
};
void run(Form f, std::uint32_t rows, Rows& t) {
    const std::uint32_t d = kLanes, key_heads = 1, value_heads = 1, value_head = 0, value_base = 0;
    const std::uint64_t q_base = 0, k_base = std::uint64_t(key_heads) * d, v_base = std::uint64_t(key_heads) * d * 2U;
    const std::uint64_t qkv_stride = (2U * key_heads + value_heads) * d, core_stride = std::uint64_t(value_heads) * d;
    std::vector<float> red(extent(f), std::nanf("")), deltas(kColumns);
    std::vector<float> retained(std::size_t(d) * kColumns);
    for (std::uint32_t lane = 0; lane < d; ++lane)
        for (std::uint32_t c = 0; c < kColumns; ++c) retained[lane * kColumns + c] = t.state[std::size_t(lane) * d + value_base + c];
    const auto tree = [&] {
        for (std::uint32_t stride = d / 2U; stride > 0; stride >>= 1U)
            for (std::uint32_t lane = 0; lane < stride; ++lane)
                for (std::uint32_t c = 0; c < kColumns; ++c) red[index(f, lane, c)] += red[index(f, lane + stride, c)];
    };
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* q = t.qkv.data() + r * qkv_stride;
        const float decay = std::exp(t.gate[r * value_heads + value_head]);
        for (std::uint32_t lane = 0; lane < d; ++lane)
            for (std::uint32_t c = 0; c < kColumns; ++c) {
                float& s = retained[lane * kColumns + c];
                s = s * decay;
                red[index(f, lane, c)] = s * q[k_base + lane];
            }
        tree();
        for (std::uint32_t c = 0; c < kColumns; ++c)
            deltas[c] = (q[v_base + value_base + c] - red[index(f, 0U, c)]) * t.beta[r * value_heads + value_head];
        for (std::uint32_t lane = 0; lane < d; ++lane)
            for (std::uint32_t c = 0; c < kColumns; ++c) {
                float& s = retained[lane * kColumns + c];
                s += q[k_base + lane] * deltas[c];
                red[index(f, lane, c)] = s * q[q_base + lane];
            }
        tree();
        for (std::uint32_t c = 0; c < kColumns; ++c) t.core[r * core_stride + value_base + c] = red[index(f, 0U, c)];
    }
    for (std::uint32_t lane = 0; lane < d; ++lane)
        for (std::uint32_t c = 0; c < kColumns; ++c) t.state[std::size_t(lane) * d + value_base + c] = retained[lane * kColumns + c];
}

void same_bits() {
    const std::uint32_t rows = 24, d = kLanes;
    std::mt19937 rng(1117);
    std::normal_distribution<float> n(0.0F, 1.0F);
    Rows base;
    base.qkv.resize(std::size_t(rows) * 3U * d);
    base.beta.resize(rows);
    base.gate.resize(rows);
    base.state.resize(std::size_t(d) * d);
    base.core.assign(std::size_t(rows) * d, 0.0F);
    for (auto& x : base.qkv) x = n(rng);
    for (auto& x : base.beta) x = 1.0F / (1.0F + std::exp(-n(rng)));
    for (auto& x : base.gate) x = -std::fabs(n(rng));
    for (auto& x : base.state) x = 0.25F * n(rng);
    base.state[5] = -0.0F;
    Rows ref = base;
    run(Form::lane_major, rows, ref);
    for (const Form f : {Form::lane_major_pad, Form::column_major}) {
        Rows t = base;
        run(f, rows, t);
        check(std::memcmp(t.core.data(), ref.core.data(), t.core.size() * sizeof(float)) == 0, "core bits equal");
        check(std::memcmp(t.state.data(), ref.state.data(), t.state.size() * sizeof(float)) == 0, "state bits equal");
    }
    bool moved = false;
    for (std::size_t i = 0; i < ref.state.size(); ++i) moved = moved || ref.state[i] != base.state[i];
    check(moved, "the model evolves the state (non-vacuous)");
}
}  // namespace

int main() {
    bijection();
    bank_spread();
    same_bits();
    if (failures) return EXIT_FAILURE;
    std::puts("gdn_recurrent_lds_index_tests: PASS");
    return EXIT_SUCCESS;
}
