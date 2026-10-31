// Explicit matrix diagnostics; not a ctest or a model qualification gate.
#include "gemma4/prefill/device.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>

namespace pf = strix::gemma4::prefill;
namespace gpu = strix::gemma4::gpu;
using namespace strix::gemma4;

int main(int argc, char** argv) try {
    gpu::require_device();
    const bool counters = argc == 2 && std::string_view(argv[1]) == "--counter";
    if (argc > 1 && !counters) throw std::invalid_argument("matrix bench accepts only --counter");
    bool warm = false;
    for (auto format : {WeightFormat::q4_k, WeightFormat::q6_k, WeightFormat::q8_0, WeightFormat::bf16}) {
        if (counters && format != WeightFormat::q4_k) continue;
        for (auto shape : {std::array<unsigned, 2>{2560, 10240}, {10240, 2560}, {2560, 256}, {256, 2560}, {2560, 512}, {2560, 2048}}) {
            if (counters && shape != std::array<unsigned, 2>{2560, 10240}) continue;
            const auto k = shape[0], n = shape[1]; constexpr unsigned m = 512;
            const auto packed = fixture::packed(format, n, k);
            gpu::Buffer w(packed.size()), x(std::size_t(m) * k * 4), y(std::size_t(m) * n * 4), ref(y.size());
            gpu::Buffer scratch(std::size_t(m) * k * 2), failure(4), maximum(4);
            std::vector<float> input(std::size_t(m) * k);
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 199) - 99) / 256.0F;
            w.upload(packed.data(), packed.size()); x.upload(input.data(), x.size()); failure.zero(); maximum.zero();
            LinearView view{w.as<std::uint8_t>(), n, k, packed_row_bytes(format, k), format};
            pf::PreparedLinear prepared(view);
            hipEvent_t begin, converted, end;
            CHECK(hipEventCreate(&begin) == hipSuccess); CHECK(hipEventCreate(&converted) == hipSuccess); CHECK(hipEventCreate(&end) == hipSuccess);
            for (unsigned route = counters ? 1U : 0U; route < 2; ++route) {
                auto run = [&](void* event = nullptr) { pf::linear(view, x.as<float>(), m, y.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(),
                                            nullptr, true, route ? &prepared : nullptr, event); };
                if (!warm && !counters) {
                    const auto begin_warm = std::chrono::steady_clock::now();
                    do { for (unsigned i = 0; i < 10; ++i) run(); gpu::synchronize(); }
                    while (std::chrono::steady_clock::now() - begin_warm < std::chrono::seconds(3));
                    warm = true;
                }
                if (!counters) { run(); gpu::synchronize(); }
                const unsigned repetitions = counters ? 1U : 5U;
                float ms = 0, conversion_ms = 0, matrix_ms = 0;
                for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
                    CHECK(hipEventRecord(begin) == hipSuccess); run(converted);
                    CHECK(hipEventRecord(end) == hipSuccess); CHECK(hipEventSynchronize(end) == hipSuccess);
                    float elapsed = 0;
                    CHECK(hipEventElapsedTime(&elapsed, begin, end) == hipSuccess); ms += elapsed;
                    CHECK(hipEventElapsedTime(&elapsed, begin, converted) == hipSuccess); conversion_ms += elapsed;
                    CHECK(hipEventElapsedTime(&elapsed, converted, end) == hipSuccess); matrix_ms += elapsed;
                }
                ms /= float(repetitions); conversion_ms /= float(repetitions); matrix_ms /= float(repetitions);
                std::printf("{\"matrix_bench\":true,\"format\":%u,\"M\":%u,\"N\":%u,\"K\":%u,\"prepared\":%u,\"ms\":%.6f,\"tflops\":%.6f,\"minimum_gb_s\":%.6f,\"conversion_ms\":%.6f,\"matrix_ms\":%.6f,\"matrix_tflops\":%.6f,\"matrix_minimum_gb_s\":%.6f}\n",
                            unsigned(format), m, n, k, route, double(ms), 2.0 * m * n * k / (ms * 1e9),
                            (double(m) * (n + k) * 4 + (route ? double(n) * k * 2 : double(packed.size()))) / (ms * 1e6),
                            double(conversion_ms), double(matrix_ms), 2.0 * m * n * k / (matrix_ms * 1e9),
                            (double(m) * (2 * k + 4 * n) + (route ? double(n) * k * 2 : double(packed.size()))) / (matrix_ms * 1e6));
                if (!route) CHECK(hipMemcpy(ref.data(), y.data(), y.size(), hipMemcpyDeviceToDevice) == hipSuccess);
                else if (!counters) {
                    std::vector<float> a(std::size_t(m) * n), b(a.size());
                    y.download(a.data(), y.size()); ref.download(b.data(), ref.size()); bytes(a, b);
                }
            }
            unsigned flag = 0; failure.download(&flag, 4); CHECK(flag == 0);
            CHECK(hipEventDestroy(begin) == hipSuccess); CHECK(hipEventDestroy(converted) == hipSuccess); CHECK(hipEventDestroy(end) == hipSuccess);
        }
    }
    std::puts(counters ? "matrix counter diagnostic complete (not a byte gate)" : "matrix byte equality PASS");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
