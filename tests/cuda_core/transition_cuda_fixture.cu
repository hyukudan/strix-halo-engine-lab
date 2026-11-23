// TRANSITION fixture (sm_120a, no model): the residual.transition decode leaf (residual add prologue + bank readout +
// RMSNorm with BF16 gain and output + a8g32-v1 image, one block) against the incumbent decode leaves it replaces, all
// bound through the production decode catalogue (strix_exec_cuda_decode_catalog: same translation unit and flags as
// the engines' step programs).
//   identity: arena A runs residual.add + residual.bank (K8c readout) + rms.bf16_gain.f32_image per normed row;
//             arena B the same with the tiled readout (residual.bank.tiled); arena C the one transition leaf. The three
//             arenas (every image plus guard words around each image) must be byte-identical, and the transition's
//             a8g32 images must equal the shared CPU reference (strix/core/a8g32.hpp) applied to arena A's norm rows.
//   --bench N: graph replay of N copies of each arm (incumbent tiled chain vs transition) per case; prints us/chain.
// Device execution is an operator task: never registered with CTest.
#include "exec/catalog_cuda_decode.hpp"
#include "strix/core/a8g32.hpp"
#include "strix/device/runtime.hpp"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <initializer_list>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace residual = strix::core::residual_mix;
namespace decode = strix::exec::cuda_decode;
namespace native = strix::device::native;
using native::check;

constexpr unsigned pad = 32U;
constexpr float guard = -1234.25F;
constexpr std::int8_t code_guard = 0x5A;
constexpr std::int32_t sum_guard = 0x7EADBEEF;

std::uint32_t next(std::uint32_t& seed) { return seed = seed * 1664525U + 1013904223U; }

float sample(std::uint32_t& seed, float scale) {
    return static_cast<float>(static_cast<int>(next(seed) >> 9U) - 4194304) * 0x1p-22F * scale;
}

// Seeds complete before any arm runs (pageable copies return before their DMA lands).
template <class T>
void upload(T* to, const std::vector<T>& values) {
    check(cudaMemcpy(to, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "seed fixture image");
    check(cudaDeviceSynchronize(), "seeded image resident");
}

template <class T>
std::vector<T> download(const T* from, std::size_t count) {
    std::vector<T> values(count);
    check(cudaMemcpy(values.data(), from, count * sizeof(T), cudaMemcpyDeviceToHost), "read fixture image");
    return values;
}

void run(const strix::exec::KernelBinding& binding, cudaStream_t stream) {
    binding.fn(binding.args, binding.grid, stream);
}

struct Case {
    const char* name;
    unsigned width, banks, queries;
    bool prologue, round, in_place, rhs_is_output, normalized;
    unsigned normed;  // bit q: query q has a norm
    unsigned a8;      // bit q: that norm also stores its a8g32 image
    bool unit_gain = false;
};

// Slots of one arena (each image `width` floats between `pad` guard words on both sides).
struct Layout {
    unsigned width, banks, queries;
    [[nodiscard]] std::size_t stride() const { return std::size_t{width} + 2U * pad; }
    [[nodiscard]] unsigned slots() const { return banks + 2U + 2U * queries; }
    [[nodiscard]] std::size_t floats() const { return stride() * slots(); }
    [[nodiscard]] std::size_t at(unsigned slot) const { return slot * stride() + pad; }
    [[nodiscard]] unsigned bank(unsigned b) const { return b; }
    [[nodiscard]] unsigned lhs() const { return banks; }
    [[nodiscard]] unsigned rhs() const { return banks + 1U; }
    [[nodiscard]] unsigned raw(unsigned q) const { return banks + 2U + q; }
    [[nodiscard]] unsigned normed(unsigned q) const { return banks + 2U + queries + q; }
    [[nodiscard]] std::size_t groups() const { return width / 32U; }
};

struct Arena {
    explicit Arena(const Layout& l)
        : values(l.floats()), codes(std::size_t{l.queries} * (l.width + 2U * pad)),
          deltas(std::size_t{l.queries} * (l.groups() + 2U * pad)), sums(std::size_t{l.queries} * (l.groups() + 2U * pad)) {}
    native::Buffer<float> values;
    native::Buffer<std::int8_t> codes;
    native::Buffer<float> deltas;
    native::Buffer<std::int32_t> sums;
};

struct Bound {
    residual::AddArgs add{};
    residual::MixArgs mix{};
    decode::RmsBf16 norms[residual::kMaxQueries]{};
    residual::TransitionArgs transition{};
};

Bound bind_arena(const Case& c, const Layout& l, const Arena& arena, const float* queries,
                 const std::vector<const std::uint16_t*>& gains) {
    Bound b{};
    float* base = arena.values.data();
    const auto image = [&](unsigned slot) { return base + l.at(slot); };
    auto& mix = b.mix;
    for (unsigned bank = 0; bank < c.banks; ++bank) mix.banks[bank] = image(l.bank(bank));
    mix.queries = queries;
    mix.rows = 1U;
    mix.width = c.width;
    mix.bank_count = c.banks;
    mix.query_count = c.queries;
    mix.epsilon = 1e-6F;
    mix.normalized = c.normalized;
    for (unsigned q = 0; q < c.queries; ++q) mix.output[q] = image(l.raw(q));
    auto* current = image(l.bank(c.banks - 1U));
    if (c.prologue) {
        const float* lhs = c.in_place ? current : image(l.lhs());
        const float* rhs = c.rhs_is_output ? mix.output[0] : image(l.rhs());
        b.add = {lhs, rhs, current, c.width, c.round};
    }
    b.transition.mix = mix;
    b.transition.add = b.add;
    b.transition.norm_epsilon = 1e-6F;
    for (unsigned q = 0; q < c.queries; ++q) {
        if (!(c.normed >> q & 1U)) continue;
        b.norms[q] = {mix.output[q], gains[q], image(l.normed(q)), 1U, c.width, 1e-6F};
        auto& norm = b.transition.norm[q];
        norm.gain = gains[q];
        norm.output = image(l.normed(q));
        if (c.a8 >> q & 1U) {
            norm.codes = arena.codes.data() + std::size_t{q} * (c.width + 2U * pad) + pad;
            norm.deltas = arena.deltas.data() + std::size_t{q} * (l.groups() + 2U * pad) + pad;
            norm.sums = arena.sums.data() + std::size_t{q} * (l.groups() + 2U * pad) + pad;
        }
    }
    return b;
}

void seed(const Case& c, const Layout& l, const Arena& arena, std::uint32_t seed_value) {
    std::vector<float> values(l.floats(), guard);
    std::uint32_t s = seed_value;
    for (unsigned slot = 0; slot < l.raw(0); ++slot) {
        // Mixed magnitudes: residual rows of a decoder are O(1..100) with a few large features.
        const float scale = slot % 3U == 0U ? 24.0F : slot % 3U == 1U ? 3.0F : 0.5F;
        for (unsigned i = 0; i < c.width; ++i) values[l.at(slot) + i] = sample(s, i % 97U == 0U ? 8.0F * scale : scale);
    }
    if (c.normalized) {
        // Pre-normalized banks: unit RMS rows (any values are legal; this is the captured-bank regime).
        for (unsigned bank = 0; bank < c.banks; ++bank) {
            double squares = 0.0;
            for (unsigned i = 0; i < c.width; ++i) squares += double(values[l.at(bank) + i]) * values[l.at(bank) + i];
            const auto inverse = static_cast<float>(1.0 / std::sqrt(squares / c.width + 1e-6));
            for (unsigned i = 0; i < c.width; ++i) values[l.at(bank) + i] *= inverse;
        }
    }
    upload(arena.values.data(), values);
    upload(arena.codes.data(), std::vector<std::int8_t>(arena.codes.span().size(), code_guard));
    upload(arena.deltas.data(), std::vector<float>(arena.deltas.span().size(), guard));
    upload(arena.sums.data(), std::vector<std::int32_t>(arena.sums.span().size(), sum_guard));
}

void incumbent(const Bound& b, const Case& c, bool tiled, cudaStream_t stream) {
    if (c.prologue) run(decode::bind(b.add), stream);
    if (tiled) run(decode::bind(decode::BankReadoutTiled{b.mix}), stream);
    else run(decode::bind(b.mix), stream);
    for (unsigned q = 0; q < c.queries; ++q)
        if (c.normed >> q & 1U) run(decode::bind(b.norms[q]), stream);
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void compare_arenas(const Case& c, const char* arm, const Arena& expected, const Arena& actual, const Layout& l) {
    const auto x = download(expected.values.data(), l.floats());
    const auto y = download(actual.values.data(), l.floats());
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::bit_cast<std::uint32_t>(x[i]) != std::bit_cast<std::uint32_t>(y[i])) {
            char text[256];
            std::snprintf(text, sizeof(text), "%s %s: slot %zu word %zd: %08x != %08x", c.name, arm, i / l.stride(),
                          static_cast<std::ptrdiff_t>(i % l.stride()) - static_cast<std::ptrdiff_t>(pad),
                          std::bit_cast<std::uint32_t>(x[i]), std::bit_cast<std::uint32_t>(y[i]));
            fail(text);
        }
}

void compare_a8(const Case& c, const Layout& l, const Arena& incumbent_arena, const Arena& fused) {
    const auto rows = download(incumbent_arena.values.data(), l.floats());
    const auto codes = download(fused.codes.data(), fused.codes.span().size());
    const auto deltas = download(fused.deltas.data(), fused.deltas.span().size());
    const auto sums = download(fused.sums.data(), fused.sums.span().size());
    const auto code_stride = std::size_t{c.width} + 2U * pad, group_stride = l.groups() + 2U * pad;
    for (unsigned q = 0; q < c.queries; ++q) {
        const bool stored = (c.normed >> q & 1U) && (c.a8 >> q & 1U);
        std::vector<std::int8_t> ref_codes(c.width, code_guard);
        std::vector<float> ref_deltas(l.groups(), guard);
        std::vector<std::int32_t> ref_sums(l.groups(), sum_guard);
        if (stored)
            strix::core::a8g32::quantize_row(std::span<const float>(rows.data() + l.at(l.normed(q)), c.width),
                                             ref_codes.data(), ref_deltas.data(), ref_sums.data());
        for (std::size_t i = 0; i < code_stride; ++i) {
            const bool inside = i >= pad && i < pad + c.width;
            const auto want = inside ? ref_codes[i - pad] : code_guard;
            if (codes[q * code_stride + i] != want) fail(std::string(c.name) + ": a8g32 code differs from the CPU reference");
        }
        for (std::size_t g = 0; g < group_stride; ++g) {
            const bool inside = g >= pad && g < pad + l.groups();
            const float want_delta = inside ? ref_deltas[g - pad] : guard;
            const std::int32_t want_sum = inside ? ref_sums[g - pad] : sum_guard;
            if (std::bit_cast<std::uint32_t>(deltas[q * group_stride + g]) != std::bit_cast<std::uint32_t>(want_delta) ||
                sums[q * group_stride + g] != want_sum)
                fail(std::string(c.name) + ": a8g32 delta or sum differs from the CPU reference");
        }
    }
}

struct Fixture {
    const Case& c;
    Layout l{c.width, c.banks, c.queries};
    native::Buffer<float> queries{std::size_t{c.queries} * c.width};
    native::Buffer<std::uint16_t> gains{std::size_t{c.queries} * c.width};
    std::vector<const std::uint16_t*> gain_rows;

    explicit Fixture(const Case& test) : c(test) {
        std::uint32_t s = 0x9E3779B9U ^ c.width ^ (c.banks << 16U);
        std::vector<float> q(std::size_t{c.queries} * c.width);
        for (auto& v : q) v = sample(s, 0.25F);
        upload(queries.data(), q);
        std::vector<std::uint16_t> g(std::size_t{c.queries} * c.width);
        for (auto& v : g) v = static_cast<std::uint16_t>((next(s) >> 16U & 0x8000U) | (0x3E00U + (next(s) >> 24U)));
        upload(gains.data(), g);
        for (unsigned row = 0; row < c.queries; ++row)
            gain_rows.push_back(c.unit_gain ? nullptr : gains.data() + std::size_t{row} * c.width);
    }
};

void check_case(const Case& c, cudaStream_t stream) {
    const Fixture f(c);
    const Arena k8(f.l), tiled(f.l), fused(f.l);
    for (const auto* arena : {&k8, &tiled, &fused}) seed(c, f.l, *arena, 17U + c.width);
    const auto a = bind_arena(c, f.l, k8, f.queries.data(), f.gain_rows);
    const auto b = bind_arena(c, f.l, tiled, f.queries.data(), f.gain_rows);
    const auto t = bind_arena(c, f.l, fused, f.queries.data(), f.gain_rows);
    if (!residual::fits(t.transition)) fail(std::string(c.name) + ": transition operands refused by fits()");
    const auto leaf = decode::bind(t.transition);
    if (leaf.identity != "cuda-sm120-residual-transition-r1-v1" || leaf.pdl.wait != strix::exec::PdlWait::before_step_io)
        fail(std::string(c.name) + ": transition binding identity or PDL contract");
    incumbent(a, c, false, stream);
    incumbent(b, c, true, stream);
    run(leaf, stream);
    native::synchronize(stream);
    compare_arenas(c, "K8c chain vs transition", k8, fused, f.l);
    compare_arenas(c, "tiled chain vs transition", tiled, fused, f.l);
    compare_a8(c, f.l, k8, fused);
    std::printf("PASS TRANSITION %s width=%u banks=%u queries=%u normed=%x a8=%x prologue=%d\n", c.name, c.width, c.banks,
                c.queries, c.normed, c.a8, c.prologue ? 1 : 0);
}

float replay_us(cudaStream_t stream, unsigned copies, const auto& enqueue) {
    cudaGraph_t graph{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (unsigned i = 0; i < copies; ++i) enqueue();
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    cudaGraphExec_t exec{};
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    check(cudaGraphLaunch(exec, stream), "warm replay");
    check(cudaEventRecord(start, stream), "record");
    constexpr unsigned replays = 5;
    for (unsigned r = 0; r < replays; ++r) check(cudaGraphLaunch(exec, stream), "replay");
    check(cudaEventRecord(stop, stream), "record");
    check(cudaEventSynchronize(stop), "sync");
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
    (void)cudaGraphExecDestroy(exec);
    (void)cudaGraphDestroy(graph);
    return ms * 1000.0F / static_cast<float>(replays * copies);
}

void bench_case(const Case& c, unsigned copies, cudaStream_t stream) {
    const Fixture f(c);
    const Arena chain(f.l), fused(f.l);
    seed(c, f.l, chain, 5U);
    seed(c, f.l, fused, 5U);
    const auto a = bind_arena(c, f.l, chain, f.queries.data(), f.gain_rows);
    const auto t = bind_arena(c, f.l, fused, f.queries.data(), f.gain_rows);
    const auto leaf = decode::bind(t.transition);
    const float before = replay_us(stream, copies, [&] { incumbent(a, c, true, stream); });
    const float after = replay_us(stream, copies, [&] { run(leaf, stream); });
    unsigned leaves = 1U + (c.prologue ? 1U : 0U);
    for (unsigned q = 0; q < c.queries; ++q) leaves += c.normed >> q & 1U;
    std::printf("BENCH TRANSITION %s incumbent %u leaves %.2f us transition 1 leaf %.2f us saved %.2f us\n", c.name,
                leaves, static_cast<double>(before), static_cast<double>(after), static_cast<double>(before - after));
}

// Decoder shapes first (width 2304: banks = live blocks + current; 1 or 3 queries), then edge geometry.
const Case cases[] = {
    {"attn-read.full.pass0", 2304, 5, 3, true, true, false, true, false, 0x7, 0x0},
    {"attn-read.local.pass0", 2304, 3, 1, true, true, false, true, false, 0x1, 0x0},
    {"attn-read.pass1.raw", 2304, 4, 3, true, true, false, true, false, 0x0, 0x0},
    {"mlp-read.pass0.inplace", 2304, 2, 1, true, false, true, false, false, 0x1, 0x0},
    {"mlp-read.pass1.raw", 2304, 5, 1, true, false, true, false, false, 0x0, 0x0},
    {"read+norm.no-prologue", 2304, 5, 1, false, false, false, false, false, 0x1, 0x0},
    {"a8g32.all", 2304, 5, 3, true, true, false, true, false, 0x7, 0x7},
    {"a8g32.subset", 2304, 4, 3, true, false, false, false, false, 0x5, 0x4},
    {"one-bank", 2304, 1, 2, true, true, false, false, false, 0x3, 0x1},
    {"eight-banks", 2304, 8, 8, true, true, false, true, false, 0xA5, 0x21},
    {"normalized", 2304, 3, 2, true, false, false, false, true, 0x3, 0x2},
    {"unit-gain", 2304, 4, 1, true, true, false, false, false, 0x1, 0x1, true},
    {"width4096", 4096, 6, 3, true, true, true, false, false, 0x7, 0x2},
    {"width96", 96, 3, 2, true, true, false, true, false, 0x3, 0x3},
    {"width1000", 1000, 5, 2, true, false, false, true, false, 0x3, 0x0},
    {"width513", 513, 8, 3, false, false, false, false, false, 0x6, 0x0},
};

}  // namespace

int main(int argc, char** argv) {
    try {
        unsigned bench = 0;
        if (argc == 3 && std::string(argv[1]) == "--bench") bench = static_cast<unsigned>(std::stoul(argv[2]));
        else if (argc != 1) throw std::invalid_argument("usage: strix-transition-cuda-fixture [--bench N]");
        const native::StreamOwner stream;
        for (const auto& c : cases) check_case(c, stream.get());
        std::puts("PASS TRANSITION fused add + bank readout + RMS bf16 + a8g32 bitwise vs incumbent leaves");
        if (bench != 0U) {
            for (const auto& c : cases)
                if (c.width == 2304U && c.a8 == 0U) bench_case(c, bench, stream.get());
            std::puts("PASS TRANSITION bench");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL TRANSITION %s\n", error.what());
        return 1;
    }
}
