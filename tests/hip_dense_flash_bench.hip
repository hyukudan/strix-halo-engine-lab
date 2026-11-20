// Kernel bench of the shared dense flash prefill core (src/attn_core/dense_flash_prefill.hpp) on the shapes the engines
// launch, synthetic data, no model. Per case: one warm-up launch, then REPS launches timed with hip events (median ms),
// TFLOP/s of the visited work (QK + PV over the visible keys) and a 64-bit FNV digest of the output bits (two builds of the
// core with the same digest per case produce the same bits per row).
//   mha      MHA sliding form: D 256, bf16, one head x 64 positions per work-group, 64 heads, window 512 (Hy4-flash shape)
//   g4s      Gemma4 E4B sliding layer form: D 256, f16, GQA group 4 (8 waves), window 512
//   g4g      Gemma4 E4B global layer form: D 512, f16, GQA group 4 (16 waves), full causal
//   qw       Qwen3.5 27B wide-window form (dense_decode_flash.hpp <256, 6>): D 256, bf16, group 6 (12 waves)
//   v80 v64  vision encoder form: D 80 / 64, f16 two-term (hi + lo) operands, 64 positions, 1024 rows from position
//            4095 (each row sees 4096-5119 keys, the size of a bidirectional encoder row)
// Usage: strix-dense-flash-bench [REPS] [CASE]   (CASE: one name above; default all)
#include "attn_core/dense_flash_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace tile = strix::attn_core::tile;

struct BArgs {
    const float* q;
    float* out;
    const std::uint16_t *k, *vt;
    unsigned* failure;
    unsigned rows, p0, k0, window, vstride, heads, kv_heads;
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Common {
    using Args = BArgs;
    using Shape = dense::Shape<D, G, (D >= 128U ? D / 128U : 1U), Pos>;
    static constexpr float kScaleLog2 = (D == 512U ? 0.044194173824159220F : 0.0625F) * 1.4426950408889634F;
    static constexpr tile::Operand kOp = Op;
    __device__ static bool valid(const Args& a) { return a.rows != 0U && a.p0 >= a.k0; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.p0; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * a.heads + h) * D; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * a.heads + h) * D; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Classic : Common<D, G, Pos, Op> {
    using Args = BArgs;
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
template <unsigned D>  // two-term operands (lo terms = the same arrays: timing only)
struct Classic2 : Classic<D, 1U, 64U, tile::Operand::f16> {
    using Args = BArgs;
    using Shape = dense::Shape<D, 1U, 1U, 64U>;
    static constexpr unsigned kTerms = 2U;
    __device__ static const std::uint16_t* key_lo(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t_lo(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
constexpr unsigned kN = 192U;  // MHA case: own columns per head before the shared 64

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t round_op(float x, bool f16) {
    if (f16) {
        const _Float16 h = static_cast<_Float16>(x);
        std::uint16_t b;
        std::memcpy(&b, &h, 2);
        return b;
    }
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ok(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

struct Case {
    const char* name;
    unsigned D, heads, kv_heads, rows, p0, window;
    bool f16;
};

template <class P> void run(const Case& c, unsigned reps) {
    const unsigned D = c.D, H = c.heads, KH = c.kv_heads;
    const unsigned k0 = (c.window && c.p0 >= c.window ? c.p0 - c.window + 1U : 0U) & ~15U, keys = c.p0 + c.rows - k0 + 16U, vstride = keys + 16U;
    std::mt19937 g(c.rows * 131U + c.p0 + D);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> q(std::size_t(c.rows) * H * D);
    for (auto& x : q) x = u(g);
    std::vector<std::uint16_t> kfull(std::size_t(keys) * KH * D), vt(std::size_t(KH) * D * vstride, 0U);
    for (auto& x : kfull) x = round_op(u(g), c.f16);
    std::vector<std::uint16_t> vrows(std::size_t(keys) * KH * D);
    for (auto& x : vrows) x = round_op(u(g), c.f16);
    for (unsigned p = 0; p < keys; ++p)
        for (unsigned h = 0; h < KH; ++h)
            for (unsigned d = 0; d < D; ++d) vt[std::size_t(h * D + d) * vstride + p] = vrows[(std::size_t(p) * KH + h) * D + d];
    if (D == 256U && KH == H)  // MHA case: columns 192..255 of every head = head 0's (the shared rope key of an MLA form)
        for (unsigned p = 0; p < keys; ++p)
            for (unsigned h = 1; h < H; ++h)
                for (unsigned d = kN; d < D; ++d) kfull[(std::size_t(p) * KH + h) * D + d] = kfull[std::size_t(p) * KH * D + d];
    BArgs a{};
    a.q = up(q);
    a.k = up(kfull);
    a.vt = up(vt);
    a.failure = up(std::vector<unsigned>{0U});
    a.rows = c.rows;
    a.p0 = c.p0;
    a.k0 = k0;
    a.window = c.window;
    a.vstride = vstride;
    a.heads = H;
    a.kv_heads = KH;
    const std::size_t n = std::size_t(c.rows) * H * D;
    float* out = nullptr;
    ok(hipMalloc(&out, n * 4U), "malloc out");
    a.out = out;
    hipEvent_t e0, e1;
    ok(hipEventCreate(&e0), "event");
    ok(hipEventCreate(&e1), "event");
    dense::launch<P>(a, H, nullptr);
    ok(hipDeviceSynchronize(), "warm-up");
    std::vector<float> ms(reps);
    for (unsigned r = 0; r < reps; ++r) {
        ok(hipEventRecord(e0, nullptr), "record");
        dense::launch<P>(a, H, nullptr);
        ok(hipEventRecord(e1, nullptr), "record");
        ok(hipEventSynchronize(e1), "sync");
        ok(hipEventElapsedTime(&ms[r], e0, e1), "elapsed");
    }
    std::vector<float> host(n);
    ok(hipMemcpy(host.data(), out, n * 4U, hipMemcpyDeviceToHost), "download");
    unsigned failure = 0U;
    ok(hipMemcpy(&failure, a.failure, 4U, hipMemcpyDeviceToHost), "failure");
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const float x : host) {
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        for (unsigned i = 0; i < 4U; ++i) h = (h ^ ((b >> (8U * i)) & 0xffU)) * 0x100000001b3ULL;
    }
    double work = 0.0;  // visible (row, key) pairs x heads x D x 2 products x 2 flop
    for (unsigned r = 0; r < c.rows; ++r) {
        const unsigned pos = c.p0 + r, lo = c.window && pos + 1U >= c.window ? pos + 1U - c.window : 0U;
        work += double(pos - lo + 1U);
    }
    work *= double(H) * D * 4.0;
    std::sort(ms.begin(), ms.end());
    const float med = ms[reps / 2U];
    std::printf("case %-5s D %u heads %u/%u rows %u p0 %u window %u: median %.3f ms min %.3f max %.3f  %.2f TF/s  digest %016llx failure %u\n",
                c.name, D, H, KH, c.rows, c.p0, c.window, double(med), double(ms.front()), double(ms.back()), work / (double(med) * 1e9),
                static_cast<unsigned long long>(h), failure);
    for (const void* p : {static_cast<const void*>(a.q), static_cast<const void*>(a.k), static_cast<const void*>(a.vt),
                          static_cast<const void*>(a.failure), static_cast<const void*>(out)})
        if (p) ok(hipFree(const_cast<void*>(p)), "free");
    ok(hipEventDestroy(e0), "event");
    ok(hipEventDestroy(e1), "event");
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned reps = argc > 1 ? unsigned(std::atoi(argv[1])) : 7U;
    const std::string only = argc > 2 ? argv[2] : "";
    const auto want = [&](const char* n) { return only.empty() || only == n; };
    try {
        // MHA sliding layer-window: 2048 rows, 64 heads, window 512, the halo keys before the rows.
        if (want("mha")) run<Classic<256U, 1U, 64U, tile::Operand::bf16>>({"mha", 256U, 64U, 64U, 2048U, 2048U, 512U, false}, reps);
        if (want("g4s")) run<Classic<256U, 4U, 16U, tile::Operand::f16>>({"g4s", 256U, 8U, 2U, 2048U, 2048U, 512U, true}, reps);
        if (want("g4g")) run<Classic<512U, 4U, 16U, tile::Operand::f16>>({"g4g", 512U, 8U, 2U, 2048U, 2048U, 0U, true}, reps);
        if (want("qw")) run<Classic<256U, 6U, 16U, tile::Operand::bf16>>({"qw", 256U, 24U, 4U, 2048U, 2048U, 0U, false}, reps);
        if (want("v80")) run<Classic2<80U>>({"v80", 80U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("v64")) run<Classic2<64U>>({"v64", 64U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
