// Byte gate of the EXL3 many-row cores against the eight-row device linears
// under the process identity (STRIX_G4_EXL3_IDENTITY=v1|v2|v3).
#include <hip/hip_runtime.h>
#include "gemma4/exl3/device.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>

namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
namespace gpu = g4::gpu;
namespace {
// Synthetic pack: random trellis bits and finite FP16 scales in [0.5, 2).
std::vector<std::uint8_t> pack(unsigned inputs, unsigned outputs, unsigned k, unsigned seed) {
    std::mt19937 rng(seed);
    const std::size_t bytes = 2ULL * (inputs + outputs) + std::size_t(inputs / 16) * (outputs / 16) * 32 * k;
    std::vector<std::uint8_t> data(bytes);
    for (unsigned i = 0; i < inputs + outputs; ++i) {
        const auto h = std::uint16_t(0x3800U + rng() % 0x0800U);
        data[2ULL * i] = std::uint8_t(h); data[2ULL * i + 1] = std::uint8_t(h >> 8U);
    }
    for (std::size_t i = 2ULL * (inputs + outputs); i < bytes; ++i) data[i] = std::uint8_t(rng());
    return data;
}
std::vector<float> inputs(std::size_t n, unsigned seed) {
    std::vector<float> x(n);
    std::mt19937 rng(seed);
    for (auto& a : x) a = (rng() % 16U == 0) ? ((rng() & 1U) ? -0.0F : 0.0F)
                                             : std::ldexp(float(int(rng() % 2000001U) - 1000000), int(rng() % 8U) - 22);
    return x;
}
struct Pack {
    std::vector<std::uint8_t> host;
    gpu::Buffer data;
    g4::LinearView view;
    Pack(unsigned in, unsigned out, unsigned k, unsigned seed) : host(pack(in, out, k, seed)), data(host.size()) {
        data.upload(host.data(), host.size());
        view = {data.as<std::uint8_t>(), out, in, std::size_t(in) * k / 8, g4::WeightFormat::exl3_mul1};
    }
};
std::size_t cases = 0;
void compare(const Pack& w, unsigned rows, unsigned seed) {
    const auto x = inputs(std::size_t(rows) * w.view.inputs, seed);
    gpu::Buffer dx(x.size() * 4); dx.upload(x.data(), x.size() * 4);
    const auto count = std::size_t(rows) * w.view.outputs;
    gpu::Buffer reference(count * 4), candidate(count * 4), failure(4);
    gpu::Buffer scratch(std::max(ex::device_scratch_bytes, ex::rows_scratch_bytes(w.view, rows)));
    const std::vector<float> poison(count, g4::bit_cast<float>(0x7fc0deadU));
    reference.upload(poison.data(), count * 4); candidate.upload(poison.data(), count * 4);
    failure.zero();
    for (unsigned r = 0; r < rows; r += 8)
        ex::device_linear(w.view, dx.as<float>() + std::size_t(r) * w.view.inputs, std::min(8U, rows - r),
                          reference.as<float>() + std::size_t(r) * w.view.outputs, scratch.data(), failure.as<unsigned>());
    gpu::synchronize();
    unsigned flag = 1; failure.download(&flag, 4);
    if (flag) throw std::runtime_error("reference arithmetic failure");
    if (!ex::device_linear_rows(w.view, dx.as<float>(), rows, candidate.as<float>(), scratch.data(), scratch.size(),
                                failure.as<unsigned>())) {
        if (rows > 8) throw std::runtime_error("many-row route declined");
        return;
    }
    gpu::synchronize();
    failure.download(&flag, 4);
    if (flag) throw std::runtime_error("many-row arithmetic failure");
    std::vector<float> a(count), b(count);
    reference.download(a.data(), count * 4); candidate.download(b.data(), count * 4);
    for (std::size_t i = 0; i < count; ++i)
        if (std::memcmp(&a[i], &b[i], 4)) {
            std::cerr << "mismatch " << ex::active_identity() << " I " << w.view.inputs << " O " << w.view.outputs << " K "
                      << ex::bits(w.view) << " rows " << rows << " at " << i << ": " << std::hexfloat << a[i] << " vs " << b[i] << '\n';
            throw std::runtime_error("EXL3 many-row byte mismatch");
        }
    ++cases;
}
void tests() {
    struct Shape { unsigned in, out; };
    // Family shapes (q, k/v, o, gate/up, down, PLE) and split edge cases.
    const Shape small[] = {{256, 128}, {256, 2560}, {2560, 256}, {2560, 512}, {2048, 2560}, {2560, 2048}, {384, 384}};
    const Shape large[] = {{2560, 10240}, {10240, 2560}};
    const unsigned many[] = {9, 15, 16, 17, 31, 32, 33, 47, 64, 100, 129, 255, 512};
    const unsigned few[] = {9, 33, 100, 512};
    unsigned seed = 1;
    for (const unsigned k : {4U, 5U, 6U, 8U}) {
        for (const auto s : small) {
            const Pack w(s.in, s.out, k, seed++);
            for (const auto rows : many) compare(w, rows, seed * 131 + rows);
        }
        for (const auto s : large) {
            const Pack w(s.in, s.out, k, seed++);
            for (const auto rows : few) compare(w, rows, seed * 131 + rows);
        }
        std::cout << "{\"identity\":\"" << ex::active_identity() << "\",\"k\":" << k << ",\"cases\":" << cases << "}\n";
    }
}
void bench() {
    struct Shape { const char* name; unsigned in, out, k; };
    const Shape shapes[] = {{"gate", 2560, 10240, 4}, {"down", 10240, 2560, 4}, {"q", 2560, 2048, 4}, {"kv", 2560, 512, 5},
                            {"o", 2048, 2560, 4}};
    for (const auto& s : shapes) {
        const Pack w(s.in, s.out, s.k, 7);
        const unsigned rows = 512;
        const auto x = inputs(std::size_t(rows) * s.in, 3);
        gpu::Buffer dx(x.size() * 4), out(std::size_t(rows) * s.out * 4), failure(4);
        gpu::Buffer scratch(std::max(ex::device_scratch_bytes, ex::rows_scratch_bytes(w.view, rows)));
        dx.upload(x.data(), x.size() * 4);
        for (const bool many : {false, true}) {
            const auto run = [&] {
                if (many) (void)ex::device_linear_rows(w.view, dx.as<float>(), rows, out.as<float>(), scratch.data(), scratch.size(), failure.as<unsigned>());
                else for (unsigned r = 0; r < rows; r += 8)
                    ex::device_linear(w.view, dx.as<float>() + std::size_t(r) * s.in, 8, out.as<float>() + std::size_t(r) * s.out,
                                      scratch.data(), failure.as<unsigned>());
            };
            run(); gpu::synchronize();
            const auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 3; ++i) run();
            gpu::synchronize();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 3;
            std::cout << std::setprecision(5) << "{\"bench\":\"" << s.name << "\",\"identity\":\"" << ex::active_identity()
                      << "\",\"route\":\"" << (many ? "many-row" : "rows8") << "\",\"ms\":" << ms << ",\"gmac_s\":"
                      << double(rows) * s.in * s.out / (ms * 1e6) << "}\n";
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        gpu::require_device();
        if (argc == 2 && std::string_view(argv[1]) == "bench") { bench(); return 0; }
        tests();
        std::cout << "{\"gate\":\"exl3-rows-vs-rows8\",\"identity\":\"" << ex::active_identity() << "\",\"cases\":" << cases
                  << ",\"verdict\":\"EXACT\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
