// FAST lever ds41-dense-f16-v1 (shared policy core::f16_dense_tall::q8a8, "dense-q8a8-f16-v1"): the dense Q8_0 x A8
// products in one FP16 WMMA chain against the exact integer products (relative L2 per case) and the time of both.
// Lines "fd ...": rows in count parts acc | exact ms | fast ms at 128-row CTAs | at 256-row CTAs | rule | rel_l2.
// --no-timing: one repetition (the ctest form).
#include "core/f16_dense_q8a8.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(hipError_t e, const char* what) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
}
template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n;
    explicit Dev(std::size_t count) : n(count) { ck(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) { ck(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> down() {
        std::vector<T> h(n);
        ck(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
};
struct Case {
    unsigned rows, in, count, parts;
    bool acc;
};
std::mt19937 rng(20261114);

template <class F> double timed(F&& f, unsigned reps) {
    f();
    ck(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < reps; ++i) f();
    ck(hipDeviceSynchronize(), "sync");
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// The DS41 flash prologue's query math (hip_ds41_attention_flash.inl query16: FP32 q, rope512 on the pairs of dims
// 448..511 of each 512-wide head; then frag_from_f32 = to_bf16 per value), one thread per 16-dim chunk.
__global__ void prologue_image(const float* q, const strix::ds41::attn::RopeRow* rope, unsigned rows, unsigned count, std::uint16_t* out) {
    const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
    if (i >= std::size_t{rows} * count / 16U) return;
    const unsigned r = static_cast<unsigned>(i * 16U / count), chunk = static_cast<unsigned>(i * 16U % count % 512U / 16U);
    float x[16];
    for (unsigned k = 0; k < 16U; ++k) x[k] = q[i * 16U + k];
    if (chunk >= 28U)
        for (unsigned k = 0; k < 8U; ++k) {
            const unsigned p = 8U * (chunk - 28U) + k;
            strix::ds41::attn::rope_pair(x[2U * k], x[2U * k + 1U], rope[r].c[p], rope[r].s[p], false);
        }
    for (unsigned k = 0; k < 16U; ++k) out[i * 16U + k] = strix::attn_core::tile::to_operand<strix::attn_core::tile::Operand::bf16>(x[k]);
}
// enqueue_q_b_rope_image (q_b's image for the flash core's query_op) against the prologue on the FAST FP32 output:
// bitwise equal, taken from the FAST dense rule's rows on, declined (nothing launched) below.
bool run_image(strix::ds41::Q8A8DensePrefillView v, const Dev<float>& fast, const Case& c, unsigned reps) {
    namespace attn = strix::ds41::attn;
    std::vector<attn::RopeRow> hr(c.rows);
    std::uniform_real_distribution<float> th(-3.14159F, 3.14159F);
    for (auto& row : hr)
        for (unsigned p = 0; p < attn::kRopePairs; ++p) {
            const float t = th(rng);
            row.c[p] = std::cos(t), row.s[p] = std::sin(t);
        }
    Dev<attn::RopeRow> dr(c.rows);
    dr.up(hr);
    Dev<std::uint16_t> image(std::size_t{c.rows} * c.count), ref(std::size_t{c.rows} * c.count);
    strix::ds41::set_fast_dense(1);
    v.out = fast.p;  // run()'s FAST FP32 output; the image form does not write it
    const bool want = c.rows >= 128U, taken = strix::ds41::enqueue_q_b_rope_image(v, image.p, dr.p, nullptr);
    if (!taken) {
        std::printf("img %5u %5u | declined %s\n", c.rows, c.count, want ? "FAIL" : "ok");
        return !want;
    }
    const double ti = timed([&] { (void)strix::ds41::enqueue_q_b_rope_image(v, image.p, dr.p, nullptr); }, reps);
    const unsigned chunks = c.rows * c.count / 16U;
    hipLaunchKernelGGL(prologue_image, dim3((chunks + 255U) / 256U), dim3(256), 0, nullptr, fast.p, dr.p, c.rows, c.count, ref.p);
    ck(hipDeviceSynchronize(), "image sync");
    const auto hi = image.down(), hp = ref.down();
    std::size_t diff = 0;
    for (std::size_t i = 0; i < hi.size(); ++i) diff += hi[i] != hp[i];
    const bool ok = want && diff == 0U;
    std::printf("img %5u %5u | image %8.4f ms | differing %zu of %zu %s\n", c.rows, c.count, ti, diff, hi.size(), ok ? "ok" : "FAIL");
    return ok;
}

bool run(const Case& c, unsigned reps) {
    const unsigned row_bytes = c.in / 32U * 34U, groups = c.in / 32U;
    std::vector<std::uint8_t> w(std::size_t{c.count} * row_bytes);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.0005F, 0.02F), step(0.002F, 0.05F);
    for (std::size_t b = 0; b < std::size_t{c.count} * groups; ++b) {
        const _Float16 d = static_cast<_Float16>(scale(rng));
        std::memcpy(&w[b * 34U], &d, 2);
        for (unsigned i = 0; i < 32U; ++i) w[b * 34U + 2U + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    std::vector<std::int8_t> a(std::size_t{c.rows} * c.in);
    for (auto& x : a) x = static_cast<std::int8_t>(code(rng));
    std::vector<float> dl(std::size_t{c.rows} * groups);
    for (auto& x : dl) x = step(rng);
    Dev<std::uint8_t> dw(w.size() + 64U);  // the stagers read 68 bytes from a block pair: slack past the last row
    Dev<std::int8_t> da(a.size());
    Dev<float> dd(dl.size()), exact(std::size_t{c.rows} * c.count), fast(std::size_t{c.rows} * c.count);
    w.resize(w.size() + 64U);
    dw.up(w), da.up(a), dd.up(dl);
    const std::vector<float> ones(std::size_t{c.rows} * c.count, 1.0F);

    strix::ds41::Q8A8DensePrefillView v;
    v.weight = dw.p, v.codes = da.p, v.deltas = dd.p;
    v.width = c.in / c.parts, v.count = c.count, v.rows = c.rows, v.x_stride = c.in, v.out_stride = c.count;
    if (c.parts == 2U) v.row_width = c.in, v.k_parts = 2U;
    v.accumulate = c.acc;
    v.tall = 1U;
    const auto call = [&](int mode, float* out) {
        strix::ds41::set_fast_dense(mode);
        v.out = out;
        (void)strix::ds41::enqueue_q8_0_a8_dense_prefill(v, nullptr);
    };
    const unsigned r = c.acc ? 1U : reps;  // an accumulating call is not repeatable
    if (c.acc) exact.up(ones), fast.up(ones);
    const double te = c.acc ? (call(0, exact.p), 0.0) : timed([&] { call(0, exact.p); }, r);
    const double tf = c.acc ? (call(1, fast.p), 0.0) : timed([&] { call(1, fast.p); }, r);
    ck(hipDeviceSynchronize(), "sync");
    const auto he = exact.down(), hf = fast.down();
    // Gate both CTA heights even in --no-timing mode; timing alone cannot establish identity.
    namespace q8a8 = strix::core::f16_dense_tall::q8a8;
    double tw[2] = {0.0, 0.0};
    bool geometry_equal = true;
    if (!c.acc) {
        Dev<float> scratch(std::size_t{c.rows} * c.count);
        q8a8::View f{};
        f.weight = dw.p, f.codes = da.p, f.deltas = dd.p, f.out = scratch.p;
        f.row_bytes = row_bytes, f.in = c.in, f.count = c.count, f.rows = c.rows, f.x_stride = c.in, f.out_stride = c.count;
        std::vector<float> wave8;
        for (unsigned i = 0; i < 2U; ++i) {
            tw[i] = timed([&] { q8a8::enqueue(f, nullptr, i ? 16U : 8U); }, r);
            ck(hipDeviceSynchronize(), "geometry sync");
            const auto result = scratch.down();
            if (i == 0U) wave8 = result;
            else geometry_equal = std::memcmp(wave8.data(), result.data(), result.size() * sizeof(float)) == 0;
        }
    }
    double num = 0.0, den = 0.0;
    unsigned bad = 0;
    for (std::size_t i = 0; i < he.size(); ++i) {
        if (!std::isfinite(hf[i])) ++bad;
        const double d = double(hf[i]) - double(he[i]);
        num += d * d, den += double(he[i]) * double(he[i]);
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const bool lever = c.rows >= 128U;  // below: the exact kernels (rel must be 0)
    const bool ok = geometry_equal && bad == 0U && (lever ? rel > 0.0 && rel < 3e-3 : rel == 0.0);
    std::printf("fd %5u %5u %5u parts %u acc %d | exact %8.4f | fast %8.4f | w8 %8.4f w16 %8.4f | rule %2u | rel_l2 %.3e %s\n", c.rows,
                c.in, c.count, c.parts, int(c.acc), te, tf, tw[0], tw[1], q8a8::waves_for(c.rows, c.in), rel, ok ? "ok" : "FAIL");
    bool image_ok = true;
    if (c.parts == 1U && !c.acc && c.count % 512U == 0U) image_ok = run_image(v, fast, c, reps);
    return ok && image_ok;
}
}  // namespace

int main(int argc, char** argv) {
    const bool timing = !(argc > 1 && std::string(argv[1]) == "--no-timing");
    const unsigned reps = timing ? 20U : 1U;
    // q_b-like (in 1024 / 1536, many outputs), o-like (in 4096, two K parts of 4096), partial last tile, accumulate, small
    const Case cases[] = {
        {64, 1024, 512, 1, false},    {128, 1024, 2048, 1, false},  {256, 1024, 2048, 1, false},  {512, 1024, 4096, 1, false},
        {1024, 1024, 4096, 1, false}, {2048, 1024, 8192, 1, false}, {4096, 1024, 8192, 1, false}, {1024, 1536, 4096, 1, false},
        {1024, 4096, 1024, 1, false}, {1024, 8192, 1024, 2, false}, {2048, 8192, 1024, 2, false}, {1024, 4096, 4096, 1, false},
        {300, 1024, 1000, 1, false},  {513, 256, 40, 1, false},    {256, 1024, 512, 1, true},
    };
    bool ok = true;
    for (const auto& c : cases) ok = run(c, reps) && ok;
    strix::ds41::set_fast_dense(-1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
