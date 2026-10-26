#include "../src/hip_ds41_staged_ready.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_rn_table.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void expect(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t n) { check(hipMalloc(&p, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& x) : Dev(x.size()) { check(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice)); }
    ~Dev() { (void)hipFree(p); }
};
struct Coherent {
    void* cpu = nullptr; void* gpu = nullptr;
    explicit Coherent(std::size_t n) {
        check(hipHostMalloc(&cpu, n, hipHostMallocMapped | hipHostMallocCoherent));
        check(hipHostGetDevicePointer(&gpu, cpu, 0));
    }
    ~Coherent() { (void)hipHostFree(cpu); }
};
void wait(strix::dist::StagedReady* f, std::uint64_t token) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::atomic_ref(f->done).load(std::memory_order_acquire) != token)
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("ready token timeout");
}
void producer(unsigned rows, unsigned count, unsigned packed, hipStream_t s, bool lean = false) {
    using namespace strix;
    constexpr unsigned width = 4096, row_width = 8192;
    std::vector<std::uint8_t> w(std::size_t{count} * row_width / 32 * 34);
    for (std::size_t b = 0; b < w.size() / 34; ++b) {
        w[b * 34] = 0; w[b * 34 + 1] = 0x24; // exact f16 scale
        for (unsigned k = 0; k < 32; ++k) w[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<int>((b * 13 + k * 7) % 16) - 8);
    }
    ds41::PackedChoice choice{static_cast<ds41::PackedKind>(packed), {}};
    for (unsigned i = 0; i < 16; ++i) choice.table[i] = static_cast<std::int8_t>(static_cast<int>(i) - 8);
    const auto weights = packed ? ds41::pack_rows(w, row_width, choice, ds41::PackedLayout::interleaved) : w;
    std::vector<std::int8_t> x(std::size_t{rows} * width);
    std::vector<float> d(std::size_t{rows} * width / 32);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<std::int8_t>(static_cast<int>((i * 17) % 255) - 127);
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = static_cast<float>((i % 23) + 1) / 256.0F;
    Dev<std::uint8_t> dw(weights); Dev<std::int8_t> dx(x); Dev<float> dd(d), baseline(std::size_t{rows} * count);
    Dev<unsigned> counter(1); check(hipMemset(counter.p, 0, sizeof(unsigned)));
    Coherent output(std::size_t{rows} * count * 4), control(sizeof(dist::StagedReady));
    auto* f = new (control.cpu) dist::StagedReady{}; f->block_counter = counter.p;
    ds41::Q8A8View v;
    v.weight = dw.p; v.width = width; v.count = count; v.per_group = count;
    v.codes = dx.p; v.deltas = dd.p; v.rows = rows; v.x_stride = width; v.out_stride = count;
    v.row_width = row_width; v.k_offset = width; v.packed = packed; v.out = baseline.p;
    std::memcpy(v.table, choice.table.data(), 16);
    ds41::enqueue_q8_0_a8(v, s);
    std::vector<float> expected(std::size_t{rows} * count);
    check(hipMemcpyAsync(expected.data(), baseline.p, expected.size() * 4, hipMemcpyDeviceToHost, s));
    check(hipStreamSynchronize(s));
    v.out = static_cast<float*>(output.gpu);
    for (std::uint64_t gen = 1; gen <= 4; ++gen) {
        std::memset(output.cpu, 0xa5, expected.size() * 4);
        std::atomic_ref(f->ready).store(gen, std::memory_order_release);
        expect(ds41::enqueue_q8_0_a8_ready(v, {output.gpu, static_cast<dist::StagedReady*>(control.gpu), gen, lean ? counter.p : nullptr}, s),
               "missing producer epilogue");
        wait(f, gen); // Deliberately no HIP event or stream sync before CPU reads TX.
        expect(std::memcmp(expected.data(), output.cpu, expected.size() * 4) == 0, "producer bytes/early publication");
    }
    v.a8_codes = dx.p;
    expect(!ds41::enqueue_q8_0_a8_ready(v, {output.gpu, static_cast<dist::StagedReady*>(control.gpu), 5}, s), "unsupported A8 admitted");
    check(hipStreamSynchronize(s));
}
template <bool ReadsOnly, bool Lean> void consumer(hipStream_t s) {
    using namespace strix;
    constexpr unsigned n = 513, generations = 64;
    Coherent rx(n * 4), control(sizeof(dist::StagedReady));
    Dev<unsigned> counter(1); check(hipMemset(counter.p, 0, sizeof(unsigned)));
    auto* f = new (control.cpu) dist::StagedReady{}; f->block_counter = counter.p;
    std::vector<float> own(n, 0.5F), expected(std::size_t{n} * generations);
    Dev<float> a(own), out(expected.size());
    for (unsigned gen = 1; gen <= generations; ++gen) {
        for (unsigned i = 0; i < n; ++i) {
            const float value = static_cast<float>(gen * 17 + i) / 32.0F;
            static_cast<float*>(rx.cpu)[i] = value;
            expected[std::size_t{gen - 1} * n + i] = value + own[i];
        }
        std::atomic_ref(f->ready).store(gen, std::memory_order_release);
        // Exactly the production add, with padding lanes and asynchronous RX reuse.
        deepseek41::ready_detail::add_halves_ready<ReadsOnly, Lean><<<3, 256, 0, s>>>(a.p, static_cast<float*>(rx.gpu),
            out.p + std::size_t{gen - 1} * n, n, static_cast<dist::StagedReady*>(control.gpu), gen, counter.p);
        check(hipGetLastError()); wait(f, gen);
    }
    std::vector<float> actual(expected.size());
    check(hipMemcpyAsync(actual.data(), out.p, actual.size() * 4, hipMemcpyDeviceToHost, s)); check(hipStreamSynchronize(s));
    expect(std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0, "consumer output/RX reuse");
}
// The MoE ready pair: half sums published into coherent TX (read by the CPU right after the token, no
// sync) == enqueue_half_sum's bytes, both halves; the combine from coherent RX == enqueue_canonical_sum's,
// over asynchronous generations with RX rewritten after each consumed token.
void moe(unsigned rows, hipStream_t s) {
    using namespace strix;
    constexpr unsigned hidden = 5120 + 37, top_k = 8, generations = 32;  // a partial last block
    std::vector<float> raw(std::size_t{rows} * top_k * hidden), weights(std::size_t{rows} * top_k);
    std::vector<std::uint32_t> order(std::size_t{rows} * top_k), low(rows);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<float>(static_cast<int>((i * 7919) % 2001) - 1000) / 997.0F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>((i * 37) % 101 + 1) / 103.0F;
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned k = 0; k < top_k; ++k) order[r * top_k + k] = r * top_k + (k * 3 + r) % top_k;
        low[r] = (r * 3) % (top_k + 1);
    }
    Dev<float> draw(raw), dw(weights), shared(std::size_t{rows} * hidden), half(std::size_t{rows} * hidden);
    Dev<std::uint32_t> dorder(order), dlow(low);
    Dev<unsigned> counter(2); check(hipMemset(counter.p, 0, 2 * sizeof(unsigned)));
    std::vector<float> sh(std::size_t{rows} * hidden);
    for (std::size_t i = 0; i < sh.size(); ++i) sh[i] = static_cast<float>(static_cast<int>(i % 977) - 488) / 211.0F;
    check(hipMemcpy(shared.p, sh.data(), sh.size() * 4, hipMemcpyHostToDevice));
    Coherent tx(sh.size() * 4), rx(sh.size() * 4), control(2 * sizeof(dist::StagedReady));
    auto* f = static_cast<dist::StagedReady*>(control.cpu);
    new (f) dist::StagedReady{}; new (f + 1) dist::StagedReady{};
    auto* fd = static_cast<dist::StagedReady*>(control.gpu);
    f[0].block_counter = counter.p; f[1].block_counter = counter.p + 1;
    ds41::HalfSumView v;
    v.raw = draw.p; v.weights = dw.p; v.order = dorder.p; v.low_counts = dlow.p; v.rows = rows; v.top_k = top_k; v.hidden = hidden;
    std::vector<float> expected(sh.size()), actual(sh.size());
    std::uint64_t token = 0;
    for (unsigned h = 0; h < 2; ++h) {
        v.half = h; v.out = half.p;
        ds41::enqueue_half_sum(v, s);
        check(hipMemcpyAsync(expected.data(), half.p, expected.size() * 4, hipMemcpyDeviceToHost, s)); check(hipStreamSynchronize(s));
        for (unsigned g = 0; g < 4; ++g) {
            std::memset(tx.cpu, 0xa5, expected.size() * 4);
            std::atomic_ref(f[0].ready).store(++token, std::memory_order_release);
            v.out = static_cast<float*>(tx.gpu);
            ds41::enqueue_half_sum_ready(v, {tx.gpu, fd, token, counter.p}, s);
            wait(&f[0], token);  // no HIP event or stream sync before the CPU reads TX
            expect(std::memcmp(expected.data(), tx.cpu, expected.size() * 4) == 0, "MoE half sum ready bytes/early publication");
        }
    }
    // combine: low = this rank's TX half (half 0 sums), high = RX rewritten per generation
    Dev<float> out(sh.size() * generations), ref(sh.size() * generations), high(sh.size());
    for (unsigned g = 0; g < generations; ++g) {
        if (g) wait(&f[1], token);  // RX reuse only after the consumer's token
        auto* r = static_cast<float*>(rx.cpu);
        for (std::size_t i = 0; i < sh.size(); ++i) r[i] = static_cast<float>(static_cast<int>((i * 31 + g * 7) % 1999) - 999) / 509.0F;
        check(hipMemcpyAsync(high.p, r, sh.size() * 4, hipMemcpyHostToDevice, s));
        ds41::enqueue_canonical_sum(shared.p, static_cast<float*>(tx.gpu), high.p, sh.size(), ref.p + sh.size() * g, s);
        check(hipStreamSynchronize(s));  // the reference's own copy of RX completes before RX is handed over
        std::atomic_ref(f[1].ready).store(++token, std::memory_order_release);
        ds41::enqueue_canonical_sum_ready(shared.p, static_cast<float*>(tx.gpu), static_cast<float*>(rx.gpu), sh.size(), out.p + sh.size() * g,
                                          {rx.gpu, fd + 1, token, counter.p + 1}, s);
    }
    wait(&f[1], token);
    std::vector<float> a(sh.size() * generations), b(a.size());
    check(hipMemcpyAsync(a.data(), out.p, a.size() * 4, hipMemcpyDeviceToHost, s));
    check(hipMemcpyAsync(b.data(), ref.p, b.size() * 4, hipMemcpyDeviceToHost, s)); check(hipStreamSynchronize(s));
    expect(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "MoE canonical sum ready output/RX reuse");
}
}
int main() try {
    hipStream_t s{}; check(hipStreamCreate(&s));
    for (bool lean : {false, true}) {
        for (unsigned p : {0U, 4U, 5U, 6U})
            for (unsigned r : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) producer(r, 5120, p, s, lean);
        // Base forms, including partial waves and a partial last block. No RN override
        // means that the same binary also exercises the production table above.
        { strix::ds41::RnWideOff off;
          for (unsigned p : {0U, 6U}) for (unsigned r : {1U, 4U, 6U}) producer(r, 513, p, s, lean);
          for (unsigned p : {0U, 6U}) for (unsigned r : {6U, 8U}) producer(r, 1056, p, s, lean); }
    }
    consumer<false, false>(s); consumer<true, false>(s); consumer<false, true>(s); consumer<true, true>(s);
    for (unsigned rows : {1U, 6U, 8U}) moe(rows, s);
    check(hipStreamDestroy(s));
    std::cout << "ready epilogue: 42 shapes x 4 generations, general and lean releases; four consumer releases x 64; "
                 "MoE half sum ready (R1/R6/R8, both halves x 4) and combine ready x 32, bitwise PASS\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
