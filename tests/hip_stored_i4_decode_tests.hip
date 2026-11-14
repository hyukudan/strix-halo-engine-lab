// Stored W4A16: integer-valued fixture, every live-row count, split offsets,
// ring refill, two output groups and sentinel checks on inactive rows.
// v_dot2(acc)_f32_f16 on gfx1151 is not an exactly rounded fma chain: integer-valued operands can come back off by
// a few F32 ulps of the operand magnitudes (measured 2^-20 on a true 0, 2^-18 on 51). Live outputs are therefore
// checked against the exact sum with a relative bound on the sum of |terms|; inactive slots stay exact sentinels.
#include "core/iu4.hpp"
#include "core/w4a16.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
constexpr unsigned K = 256, N = 128, R = 8, S = 4;
void ok(hipError_t e) { if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::abort(); } }
__global__ void decode(const std::uint8_t* q, const std::uint16_t* scales, const _Float16* x, float* out,
                       unsigned live, unsigned k0) {
    const unsigned lane = threadIdx.x;
    strix::core::w4a16::wave_rows<R, 4U, 4U>(q, scales, N, K, k0, K / 16U, blockIdx.x * 4U,
        x + k0 * 16U, K, lane, live, [&](unsigned s, unsigned r, unsigned col, float v) {
            out[(s * R + r) * N + col] = v;
        });
}
}
int main() {
    std::vector<std::uint8_t> q(K * N / 2);
    std::vector<_Float16> scales(N * K / 64), x(R * K);
    for (unsigned n = 0; n < N; ++n) for (unsigned g = 0; g < K / 64; ++g)
        scales[n * (K / 64) + g] = static_cast<_Float16>(g % 2 ? 0.5f : 1.0f);
    for (unsigned k = 0; k < K; ++k) for (unsigned n = 0; n < N; ++n)
        strix::core::iu4_put(q[strix::core::iu4_weight_byte(static_cast<int>(k), static_cast<int>(n), N)], k & 1U,
                            static_cast<int>((k + 3 * n) % 15) - 7);
    for (unsigned r = 0; r < R; ++r) for (unsigned k = 0; k < K; ++k)
        x[r * K + k] = static_cast<_Float16>(static_cast<int>((r + k) % 7) - 3);
    std::uint8_t* dq = nullptr; std::uint16_t* ds = nullptr; _Float16* dx = nullptr; float* dout = nullptr;
    ok(hipMalloc(&dq, q.size())); ok(hipMalloc(&ds, scales.size() * 2));
    ok(hipMalloc(&dx, x.size() * 2)); ok(hipMalloc(&dout, S * R * N * 4));
    ok(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
    ok(hipMemcpy(ds, scales.data(), scales.size() * 2, hipMemcpyHostToDevice));
    ok(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    constexpr float sentinel = -123456.0f;
    for (unsigned live = 0; live <= R; ++live) for (unsigned k0 : {0U, 4U}) {
        std::vector<float> out(S * R * N, sentinel);
        ok(hipMemcpy(dout, out.data(), out.size() * 4, hipMemcpyHostToDevice));
        decode<<<2, 32>>>(dq, ds, dx, dout, live, k0);
        ok(hipGetLastError()); ok(hipDeviceSynchronize());
        ok(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
        for (unsigned s = 0; s < S; ++s) for (unsigned r = 0; r < R; ++r) for (unsigned n = 0; n < N; ++n) {
            float expected = sentinel, tol = 0.0f;
            if (r < live && s < S - k0 / 4U) {
                expected = 0;
                const unsigned first = k0 * 16U + s * 64U;
                for (unsigned k = first; k < first + 64; ++k) {
                    const float term = float(static_cast<int>((k + 3 * n) % 15) - 7) * float(x[r * K + k]) * float(scales[n * (K / 64) + k / 64]);
                    expected += term;
                    tol += term < 0 ? -term : term;
                }
                tol *= 0x1p-16f;
            }
            const float got = out[(s * R + r) * N + n], err = got > expected ? got - expected : expected - got;
            if (!(tol == 0.0f ? got == expected : err <= tol)) {
                std::fprintf(stderr, "FAIL stored-i4 decode live=%u k0=%u split=%u row=%u col=%u: %g != %g\n",
                             live, k0, s, r, n, double(out[(s * R + r) * N + n]), double(expected));
                return 1;
            }
        }
    }
    for (void* p : {static_cast<void*>(dq), static_cast<void*>(ds), static_cast<void*>(dx), static_cast<void*>(dout)}) ok(hipFree(p));
    std::puts("stored-i4 decode: PASS live=0..8 split-offsets ring-refill sentinels");
}
