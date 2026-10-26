// Root cause of the slower fused o_b ready epilogue (measured): device time per launch of
// (a) the real R1/R6 o_b products: plain, plain + one-block publish, general epilogue, lean epilogue;
// (b) the real consumer add: plain, general reads-only, lean reads-only;
// (c) a synthetic product (R1 o_b grid: shared activations, own weight rows, one coherent store per wave)
//     with each release ingredient alone: seq_cst vs release-only lane fence, per-block control-line loads.
// Each case with and without a CPU thread spinning on the control line (the progress thread's wait).
// Timing only: back-to-back launches between two events, the same token (no CPU step per launch).
#include "../src/hip_ds41_staged_ready.hpp"
#include "strix/ds41_packed.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace {
using namespace strix;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t n) { check(hipMalloc(&p, n * sizeof(T))); check(hipMemset(p, 0, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& x) : Dev(x.size()) { check(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice)); }
    ~Dev() { (void)hipFree(p); }
};
struct Coherent {
    void* cpu = nullptr; void* gpu = nullptr;
    explicit Coherent(std::size_t n) {
        check(hipHostMalloc(&cpu, n, hipHostMallocMapped | hipHostMallocCoherent));
        check(hipHostGetDevicePointer(&gpu, cpu, 0));
        std::memset(cpu, 0, n);
    }
    ~Coherent() { (void)hipHostFree(cpu); }
};
// Synthetic release variants: Fence 0 none, 1 seq_cst (general), 2 release-only; Loads: the general
// thread-0 chain (ready + counter pointer from the control line) or the lean one (view counter, last block).
template <int Fence, bool Loads>
__device__ __forceinline__ void release(dist::StagedReady* flag, unsigned* counter, std::uint64_t token) {
    if constexpr (Fence == 1) __threadfence_system();
    if constexpr (Fence == 2) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    if constexpr (Loads) {
        // publish_staged_block without its lane fence
        __syncthreads();
        if (threadIdx.x == 0) {
            if (__hip_atomic_load(&flag->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != token) __builtin_trap();
            auto* c = flag->block_counter;
            const auto old = __hip_atomic_fetch_add(c, 1U, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (old + 1U == gridDim.x) {
                __hip_atomic_store(c, 0U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&flag->done, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    } else {
        dist::publish_staged_block_lean_impl<false>(flag, counter, token, gridDim.x);
    }
}
template <int Fence, bool Loads, bool Publish>
__global__ __launch_bounds__(256) void synthetic(const int4* __restrict__ x, const int4* __restrict__ w, float* out,
                                                 dist::StagedReady* flag, unsigned* counter, std::uint64_t token) {
    // 8 waves x one output: lane l reads activation chunks l, l + 32, ... (256 x 16 B, shared by every block)
    // and the wave's weight row (8 KiB), as the R1 Q8 product's loads.
    const unsigned wv = threadIdx.x / 32U, l = threadIdx.x % 32U, o = blockIdx.x * 8U + wv;
    int acc = 0;
    for (unsigned k = l; k < 256U; k += 32U) {
        const int4 a = x[k], b = w[static_cast<std::size_t>(o) * 512U + k], c = w[static_cast<std::size_t>(o) * 512U + 256U + k];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + a.x * c.y + a.y * c.z + a.z * c.w + a.w * c.x;
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor(acc, s, 32);
    if (l == 0) out[o] = static_cast<float>(acc);
    if constexpr (Publish) release<Fence, Loads>(flag, counter, token);
}
__global__ void add_plain(const float* a, const float* b, float* __restrict__ out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

struct Spinner {  // the progress thread's wait: acquire loads of done, as Communicator::wait_staged_flag
    std::atomic<bool> stop{false};
    std::thread t;
    Spinner(dist::StagedReady* f, bool on) {
        if (on) t = std::thread([this, f] { while (!stop.load(std::memory_order_relaxed)) (void)std::atomic_ref(f->done).load(std::memory_order_acquire); });
    }
    ~Spinner() { stop = true; if (t.joinable()) t.join(); }
};
double time_us(hipStream_t s, const std::function<void()>& launch, unsigned n = 400) {
    for (unsigned i = 0; i < 20; ++i) launch();
    hipEvent_t a, b; check(hipEventCreate(&a)); check(hipEventCreate(&b));
    check(hipEventRecord(a, s));
    for (unsigned i = 0; i < n; ++i) launch();
    check(hipEventRecord(b, s)); check(hipEventSynchronize(b));
    float ms = 0; check(hipEventElapsedTime(&ms, a, b));
    check(hipEventDestroy(a)); check(hipEventDestroy(b));
    check(hipGetLastError());
    return 1e3 * ms / n;
}
}

int main() try {
    hipStream_t s{}; check(hipStreamCreate(&s));
    Coherent control(sizeof(dist::StagedReady)), tx(1u << 20), rx(1u << 20);
    Dev<unsigned> counter(2);
    auto* f = new (control.cpu) dist::StagedReady{};
    f->block_counter = counter.p;
    f->ready = 7;
    auto* flag = static_cast<dist::StagedReady*>(control.gpu);
    const dist::ProducerStage general{tx.gpu, flag, 7, nullptr}, lean{tx.gpu, flag, 7, counter.p};

    // (a) the real products, R1 (1 row) and R6 (6 rows) o_b forms: U6 packed, 5120 outputs, K half 4096 of 8192
    constexpr unsigned count = 5120, width = 4096, row_width = 8192;
    std::vector<std::uint8_t> q8(std::size_t{count} * row_width / 32 * 34);
    for (std::size_t b = 0; b < q8.size() / 34; ++b) {
        q8[b * 34] = 0; q8[b * 34 + 1] = 0x24;
        for (unsigned k = 0; k < 32; ++k) q8[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<int>((b * 13 + k * 7) % 16) - 8);
    }
    ds41::PackedChoice choice{static_cast<ds41::PackedKind>(6), {}};
    for (unsigned i = 0; i < 16; ++i) choice.table[i] = static_cast<std::int8_t>(static_cast<int>(i) - 8);
    Dev<std::uint8_t> dw(ds41::pack_rows(q8, row_width, choice, ds41::PackedLayout::interleaved));
    std::vector<std::int8_t> x(std::size_t{8} * width);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<std::int8_t>(static_cast<int>((i * 17) % 255) - 127);
    std::vector<float> d(std::size_t{8} * width / 32, 1.0F / 64.0F);
    Dev<std::int8_t> dx(x); Dev<float> dd(d);
    for (bool spin : {false, true}) {
        Spinner spinner(f, spin);
        for (unsigned rows : {1U, 6U}) {
            ds41::Q8A8View v;
            v.weight = dw.p; v.width = width; v.count = count; v.per_group = count;
            v.codes = dx.p; v.deltas = dd.p; v.rows = rows; v.x_stride = width; v.out_stride = count;
            v.row_width = row_width; v.k_offset = width; v.packed = 6; v.out = static_cast<float*>(tx.gpu);
            std::memcpy(v.table, choice.table.data(), 16);
            const double plain = time_us(s, [&] { ds41::enqueue_q8_0_a8(v, s); });
            const double separate = time_us(s, [&] { ds41::enqueue_q8_0_a8(v, s); deepseek41::ready_detail::publish_ready_kernel<<<1, 64, 0, s>>>(general); });
            const double gen = time_us(s, [&] { (void)ds41::enqueue_q8_0_a8_ready(v, general, s); });
            const double ln = time_us(s, [&] { (void)ds41::enqueue_q8_0_a8_ready(v, lean, s); });
            std::printf("product R%u%s: plain %.2f, plain+publish %.2f, general epilogue %.2f, lean epilogue %.2f us/launch\n",
                        rows, spin ? " (CPU spin)" : "", plain, separate, gen, ln);
        }
        // (b) the consumer add, 5120 floats (20 blocks)
        Dev<float> a(count), out(count);
        const auto* r = static_cast<const float*>(rx.gpu);
        const double add0 = time_us(s, [&] { add_plain<<<20, 256, 0, s>>>(a.p, r, out.p, count); });
        const double addg = time_us(s, [&] { deepseek41::ready_detail::add_halves_ready<true, false><<<20, 256, 0, s>>>(a.p, r, out.p, count, flag, 7, counter.p + 1); });
        const double addl = time_us(s, [&] { deepseek41::ready_detail::add_halves_ready<true, true><<<20, 256, 0, s>>>(a.p, r, out.p, count, flag, 7, counter.p + 1); });
        std::printf("consumer add 5120%s: plain %.2f, general reads-only %.2f, lean reads-only %.2f us/launch\n", spin ? " (CPU spin)" : "", add0, addg, addl);
        // (c) synthetic R1 o_b grid: 640 blocks x 8 waves
        Dev<int4> sx(256), sw(std::size_t{count} * 512);
        auto* so = static_cast<float*>(tx.gpu);
        const auto syn = [&](auto kernel) { return time_us(s, [&] { hipLaunchKernelGGL(kernel, dim3(640), dim3(256), 0, s, sx.p, sw.p, so, flag, counter.p, std::uint64_t{7}); }); };
        std::printf("synthetic%s: no release %.2f; seq_cst+loads (general) %.2f; release-only+loads %.2f; seq_cst+lean chain %.2f; "
                    "release-only+lean chain (lean) %.2f; no fence+lean chain %.2f us/launch\n", spin ? " (CPU spin)" : "",
                    syn(synthetic<0, false, false>), syn(synthetic<1, true, true>), syn(synthetic<2, true, true>),
                    syn(synthetic<1, false, true>), syn(synthetic<2, false, true>), syn(synthetic<0, false, true>));
    }
    check(hipStreamSynchronize(s));
    check(hipStreamDestroy(s));
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
