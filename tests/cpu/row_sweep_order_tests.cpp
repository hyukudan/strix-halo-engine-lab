// The declared row-sweep accumulation order (core/dense_linear.hpp + core/gemv_rows_direct.inc) as the host
// reference computes it (cpu::chunk16_kernel / cpu::parallel_row_sweep_matvec). The expected values come from an
// independent literal emulation of the device wave: 32 lane accumulators, lane l visits chunks l, l+32, ... of 16
// elements in ascending order as acc = acc + (w * x) with the product rounded first (no FMA), then the xor butterfly
// 16, 8, 4, 2, 1 over the lanes.
#include "core/host_lane.hpp"
#include "cpu/core/dense.hpp"
#include "cpu/core/ordered_linear_reference.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
using namespace strix;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
std::uint32_t bits(float v) { return std::bit_cast<std::uint32_t>(v); }

float literal_wave(const float* w, const float* x, unsigned width) {
    float acc[32] = {};
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned c = lane; c * 16U < width; c += 32U)
            for (unsigned e = 0; e < 16U; ++e) {
                volatile float product = w[c * 16U + e] * x[c * 16U + e];  // rounded before the add
                const float p = product;
                acc[lane] = acc[lane] + p;
            }
    for (unsigned m = 16; m > 0; m >>= 1U) {
        float next[32];
        for (unsigned lane = 0; lane < 32; ++lane) next[lane] = acc[lane] + acc[lane ^ m];
        for (unsigned lane = 0; lane < 32; ++lane) acc[lane] = next[lane];
    }
    return acc[0];
}

void hand_case() {
    // Elements 0 and 1 of lane 0: 1 * (1 + 2^-12) = exact, then (1 + 2^-13) * -(1 + 2^-13) whose exact product
    // -(1 + 2^-12 + 2^-26) rounds to -(1 + 2^-12): the chain ends at +0 without FMA (-2^-26 with FMA).
    std::vector<float> w(32, 0.0F), x(32, 0.0F);
    const float a = 1.0F + 0x1p-12F, b = 1.0F + 0x1p-13F;
    w[0] = 1.0F; x[0] = a;
    w[1] = b; x[1] = -b;
    for (const auto kernel : {cpu::chunk16_kernel(cpu::Isa::scalar), cpu::chunk16_kernel()})
        check(bits(kernel.run(w.data(), x.data(), 32)) == 0U, "hand case: product rounded before the add");
    check(bits(literal_wave(w.data(), x.data(), 32)) == 0U, "hand case: literal emulation");
    // Lane assignment: lane 1 owns chunks 1 and 33 (2^24 then +1 is lost, tie to even), lane 2 owns chunk 2
    // (-2^24); the butterfly cancels them to +0. A different lane for the +1 would give 1.
    std::vector<float> w2(1024, 0.0F), x2(1024, 1.0F);
    w2[16] = 0x1p24F;
    w2[528] = 1.0F;
    w2[32] = -0x1p24F;
    check(bits(cpu::chunk16_kernel().run(w2.data(), x2.data(), 1024)) == 0U, "hand case: lane and chunk assignment");
    check(bits(literal_wave(w2.data(), x2.data(), 1024)) == 0U, "hand case: literal lane assignment");
}

void random_cases() {
    std::mt19937 rng(20251127);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    const auto scalar = cpu::chunk16_kernel(cpu::Isa::scalar);
    const auto native = cpu::chunk16_kernel();
    for (const unsigned width : {16U, 32U, 512U, 528U, 1024U, 2560U, 3072U, 8192U}) {
        std::vector<float> w(width), x(width);
        for (int trial = 0; trial < 6; ++trial) {
            for (auto& v : w) v = dist(rng);
            for (auto& v : x) v = dist(rng);
            const auto want = bits(literal_wave(w.data(), x.data(), width));
            check(bits(scalar.run(w.data(), x.data(), width)) == want, "scalar kernel equals the literal wave");
            check(bits(native.run(w.data(), x.data(), width)) == want, "native kernel equals the literal wave");
        }
    }
}

void matvec_groups() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    const unsigned width = 96, group_rows = 5, groups = 3, outputs = group_rows * groups;
    std::vector<float> matrix(std::size_t{outputs} * width), in(std::size_t{groups} * width), out(outputs);
    for (auto& v : matrix) v = dist(rng);
    for (auto& v : in) v = dist(rng);
    core::HostLane lane;
    for (const unsigned threads : {1U, 3U}) {
        cpu::parallel_row_sweep_matvec(cpu::chunk16_kernel(), lane, threads, matrix, in, out, groups);
        for (unsigned o = 0; o < outputs; ++o)
            check(bits(out[o]) == bits(literal_wave(matrix.data() + std::size_t{o} * width,
                                                    in.data() + std::size_t{o / group_rows} * width, width)),
                  "grouped matvec equals the literal wave per output");
    }
}
}  // namespace

int main() {
    hand_case();
    random_cases();
    matvec_groups();
    if (failures) return 1;
    std::puts("row-sweep order: PASS");
    return 0;
}
