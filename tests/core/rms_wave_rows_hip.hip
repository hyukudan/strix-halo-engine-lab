// Rows of at most 256 values in enqueue_rms_bf16_rows (parallel) take one wave per row (rmsnorm_wave_rows_kernel): every
// output must be byte-equal to the 256-lane block form (rmsnorm_strided_kernel<Bf16ScaleSqrtDivTraits, 256>) on the same
// rows. Cases: widths 1 .. 256 around each register count (J = 1, 2, 4, 8), row counts around the 8-row block, with and
// without the BF16 scale, in place and at a row stride, rows holding +0 / -0 / subnormal / overflowing values.
// Width 257 keeps the block form (checked equal too). --bench: ms of both forms on 65,888 rows of 128 values in place.
// Rounded numerator (enqueue_rms_bf16_rows_rounded_numerator): the serial form equals the host B((B(x) * r) * w) with r
// from the FP32 row bit for bit, the parallel form is within 1 BF16 ulp, and on BF16-valued rows both forms equal the
// plain entry's bits.
// Exit 1 on a failed check.
#include <hip/hip_runtime.h>
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"

#include "ops/norm/rms_bf16_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace norm = strix::ops::norm;
constexpr float kEpsilon = 1.0e-6F;

void require(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::cerr << "HIP failure: " << what << ": " << hipGetErrorString(status) << '\n';
        std::exit(2);
    }
}

template <class T>
T* device(std::size_t n) {
    T* p = nullptr;
    require(hipMalloc(&p, std::max<std::size_t>(n, 1U) * sizeof(T)), "malloc");
    return p;
}

std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

void block_form(const norm::Bf16RmsRows& a, hipStream_t stream) {
    norm::rmsnorm_strided_kernel<norm::Bf16ScaleSqrtDivTraits, 256><<<dim3(a.rows), 256, 0, stream>>>(
        a.input, a.input_row, a.scale, 0, a.width, kEpsilon, a.output, a.output_row);
}

float bf16_value(float x) {
    const std::uint32_t u = std::uint32_t{bf16(x)} << 16U;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<float> run_rows(const std::vector<float>& x, const std::vector<std::uint16_t>& w, unsigned rows,
                            unsigned width, bool rounded, norm::RmsRowsReduction reduction) {
    float* in = device<float>(x.size());
    float* out = device<float>(x.size());
    std::uint16_t* scale = device<std::uint16_t>(w.size());
    require(hipMemcpy(in, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x");
    require(hipMemcpy(scale, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w");
    const norm::Bf16RmsRows a{in, scale, out, rows, width, width, width};
    require(rounded ? norm::enqueue_rms_bf16_rows_rounded_numerator(a, kEpsilon, reduction, nullptr)
                    : norm::enqueue_rms_bf16_rows(a, kEpsilon, reduction, nullptr), "rounded rms");
    std::vector<float> y(x.size());
    require(hipMemcpy(y.data(), out, y.size() * 4U, hipMemcpyDeviceToHost), "y");
    require(hipFree(in), "free");
    require(hipFree(out), "free");
    require(hipFree(scale), "free");
    return y;
}

bool rounded_numerator_case() {
    std::mt19937 random(1127U);
    std::uniform_real_distribution<float> value(-3.0F, 3.0F);
    bool ok = true;
    for (const unsigned width : {64U, 200U, 4096U}) {
        constexpr unsigned rows = 3;
        std::vector<float> x(std::size_t{rows} * width), xb(x.size());
        std::vector<std::uint16_t> w(width);
        for (auto& v : x) v = value(random);  // mantissas below BF16: the numerator rounding matters
        for (std::size_t i = 0; i < x.size(); ++i) xb[i] = bf16_value(x[i]);
        for (auto& v : w) v = bf16(value(random));
        std::vector<float> host(x.size());
        for (unsigned r = 0; r < rows; ++r) {
            float ss = 0.0F;
            for (unsigned i = 0; i < width; ++i) ss = std::fma(x[r * width + i], x[r * width + i], ss);
            const float inv = 1.0F / std::sqrt(ss / float(width) + kEpsilon);
            for (unsigned i = 0; i < width; ++i) {
                const std::uint32_t wb = std::uint32_t{w[i]} << 16U;
                float wf;
                std::memcpy(&wf, &wb, 4);
                host[r * width + i] = bf16_value((bf16_value(x[r * width + i]) * inv) * wf);
            }
        }
        const auto serial = run_rows(x, w, rows, width, true, norm::RmsRowsReduction::serial_reference);
        const auto parallel = run_rows(x, w, rows, width, true, norm::RmsRowsReduction::parallel);
        unsigned max_ulp = 0;
        for (std::size_t i = 0; i < host.size(); ++i) {
            std::uint32_t a, b;
            std::memcpy(&a, &parallel[i], 4);
            std::memcpy(&b, &host[i], 4);
            a >>= 16U;
            b >>= 16U;
            max_ulp = std::max(max_ulp, a > b ? a - b : b - a);
        }
        const bool serial_ok = std::memcmp(serial.data(), host.data(), host.size() * 4U) == 0;
        bool plain_ok = true;
        for (const auto reduction : {norm::RmsRowsReduction::serial_reference, norm::RmsRowsReduction::parallel}) {
            const auto a = run_rows(xb, w, rows, width, true, reduction), b = run_rows(xb, w, rows, width, false, reduction);
            plain_ok = plain_ok && std::memcmp(a.data(), b.data(), a.size() * 4U) == 0;
        }
        std::cout << "rounded numerator width " << width << ": serial " << (serial_ok ? "bitwise" : "DIFFERS")
                  << ", parallel max BF16 ulp " << max_ulp << ", BF16 rows equal the plain entry "
                  << (plain_ok ? "yes" : "NO") << std::endl;
        ok = ok && serial_ok && max_ulp <= 1U && plain_ok;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(20261124U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const std::vector<unsigned> widths = {1U, 5U, 31U, 32U, 33U, 64U, 100U, 128U, 129U, 200U, 255U, 256U, 257U};
    const std::vector<unsigned> heights = {1U, 7U, 8U, 9U, 1000U};
    bool ok = true;
    unsigned cases = 0;
    for (const unsigned width : widths) {
        for (const unsigned rows : heights) {
            for (unsigned form = 0; form < 4U; ++form) {  // bit 0: scale, bit 1: in place (else a row stride)
                const bool scaled = (form & 1U) != 0U, in_place = (form & 2U) != 0U;
                const std::size_t stride = in_place ? width : width + 3U;
                std::vector<float> host(rows * stride);
                for (auto& v : host) v = nd(rng);
                if (rows > 1U) {
                    host[stride + 0] = 0.0F;
                    if (width > 1U) host[stride + 1] = -0.0F;
                    if (width > 2U) host[stride + 2] = 1.0e-40F;
                }
                if (rows > 2U) host[2U * stride] = 3.0e19F;  // its square overflows: r = 0
                if (rows > 3U)
                    for (unsigned i = 0; i < width; ++i) host[3U * stride + i] = 0.0F;  // all zero: r = 1 / sqrt(eps)
                std::vector<std::uint16_t> scale(width);
                for (auto& s : scale) s = bf16(1.0F + 0.1F * nd(rng));
                auto* in = device<float>(host.size());
                auto* out_ref = device<float>(host.size());
                auto* out_new = device<float>(host.size());
                auto* sc = device<std::uint16_t>(width);
                require(hipMemcpy(sc, scale.data(), width * 2U, hipMemcpyHostToDevice), "scale");
                const auto run = [&](float* out, bool dispatch) {
                    require(hipMemcpy(in, host.data(), host.size() * 4U, hipMemcpyHostToDevice), "in");
                    require(hipMemset(out, 0xFF, host.size() * 4U), "seed");
                    float* target = in_place ? in : out;
                    const norm::Bf16RmsRows a{in, scaled ? sc : nullptr, target, rows, width, stride, stride};
                    if (dispatch) require(norm::enqueue_rms_bf16_rows(a, kEpsilon, norm::RmsRowsReduction::parallel, nullptr), "rms");
                    else block_form(a, nullptr);
                    require(hipDeviceSynchronize(), "sync");
                    std::vector<float> h(host.size());
                    require(hipMemcpy(h.data(), target, h.size() * 4U, hipMemcpyDeviceToHost), "d2h");
                    return h;
                };
                const auto x = run(out_ref, false), y = run(out_new, true);
                const bool same = std::memcmp(x.data(), y.data(), x.size() * 4U) == 0;
                ok = ok && same;
                ++cases;
                if (!same)
                    std::cout << "check width " << width << " rows " << rows << " scaled " << scaled << " in_place "
                              << in_place << " DIFFER FAIL\n";
                for (auto* p : {in, out_ref, out_new}) require(hipFree(p), "free");
                require(hipFree(sc), "free");
            }
        }
    }
    std::cout << "rms wave rows: " << (ok ? "EXACT PASS" : "FAILED") << " cases=" << cases << '\n';
    if (bench) {
        const unsigned rows = 65888U, width = 128U;
        auto* v = device<float>(std::size_t{rows} * width);
        std::vector<float> host(std::size_t{rows} * width);
        for (auto& x : host) x = nd(rng);
        require(hipMemcpy(v, host.data(), host.size() * 4U, hipMemcpyHostToDevice), "in");
        const norm::Bf16RmsRows a{v, nullptr, v, rows, width, width, width};
        hipEvent_t e0, e1;
        require(hipEventCreate(&e0), "e");
        require(hipEventCreate(&e1), "e");
        double t[2];
        for (unsigned arm = 0; arm < 2U; ++arm) {
            std::vector<float> ms;
            for (unsigned rep = 0; rep < 12U; ++rep) {
                require(hipEventRecord(e0, nullptr), "r");
                if (arm == 0U) block_form(a, nullptr);
                else require(norm::enqueue_rms_bf16_rows(a, kEpsilon, norm::RmsRowsReduction::parallel, nullptr), "b");
                require(hipEventRecord(e1, nullptr), "r");
                require(hipEventSynchronize(e1), "s");
                float m = 0.0F;
                require(hipEventElapsedTime(&m, e0, e1), "el");
                if (rep > 1U) ms.push_back(m);
            }
            std::sort(ms.begin(), ms.end());
            t[arm] = double(ms[ms.size() / 2U]);
        }
        std::cout << "bench rows " << rows << " width " << width << " block_ms " << t[0] << " wave_ms " << t[1]
                  << " ratio " << t[1] / t[0] << " wave_GBps " << double(rows) * width * 8.0 / (t[1] * 1.0e6) << '\n';
        require(hipFree(v), "free");
    }
    ok = rounded_numerator_case() && ok;
    return ok ? 0 : 1;
}
