// Byte gate of the prompt-prefill fused epilogues (STRIX_G4_PREFILL_FUSE_EPILOGUE)
// against the separate launches they replace, for odd and chunk-sized row counts:
// norm_residual_rows == norm + residual + check_finite [+ norm of the new residual],
// gelu_product_finite == gelu_product + check_finite. Values, residuals and the
// failure word must match bit for bit, including rows that overflow.
#include <hip/hip_runtime.h>
#include "gemma4/kernels/ops.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
template <class T> Buffer device(const std::vector<T>& values) {
    Buffer b(values.size() * sizeof(T));
    b.upload(values.data(), b.size());
    return b;
}
template <class T> std::vector<T> read(const Buffer& b, std::size_t count) {
    std::vector<T> v(count);
    b.download(v.data(), count * sizeof(T));
    return v;
}
std::vector<float> values(std::size_t n, unsigned seed, float scale) {
    std::vector<float> x(n);
    std::mt19937 rng(seed);
    for (auto& a : x) {
        const auto kind = rng() % 16U;
        a = kind == 0 ? ((rng() & 1U) ? -0.0F : 0.0F) : std::ldexp(float(int(rng() % 2000001U) - 1000000), int(rng() % 12U) - 20) * scale;
    }
    return x;
}
// 1 x width norm weights in the formats checkpoints carry (F32 GGUF, BF16 safetensors).
struct Weights {
    std::vector<std::uint8_t> host;
    Buffer data;
    LinearView view;
    Weights(unsigned width, WeightFormat format, unsigned seed) {
        const auto x = values(width, seed, 1.0F);
        const unsigned bytes = format == WeightFormat::bf16 ? 2U : 4U;
        host.resize(std::size_t(width) * bytes);
        for (unsigned i = 0; i < width; ++i) {
            const float w = 1.0F + x[i];
            std::uint32_t u = 0; std::memcpy(&u, &w, 4);
            if (bytes == 2) { const auto h = std::uint16_t(u >> 16U); std::memcpy(host.data() + 2ULL * i, &h, 2); }
            else std::memcpy(host.data() + 4ULL * i, &u, 4);
        }
        data = device(host);
        view = {data.as<std::uint8_t>(), 1, width, std::uint32_t(host.size()), format};
    }
};
void residual_(Buffer& r, const Buffer& branch, unsigned rows, unsigned width, LinearView scalar) {
    residual(r.as<float>(), branch.as<float>(), rows * width, scalar);
}
bool equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
void norm_residual_case(unsigned rows, unsigned width, WeightFormat format, bool scaled, bool next, bool overflow) {
    const auto n = std::size_t(rows) * width;
    auto branch = values(n, rows * 7U + width, 4.0F), residual = values(n, rows * 13U + width, 8.0F);
    if (overflow) residual[n / 2] = INFINITY; // one nonfinite residual element
    Weights w(width, format, 3U), following(width, format, 5U), scalar(1, WeightFormat::f32, 9U);
    const LinearView s = scaled ? scalar.view : LinearView{};
    Buffer b = device(branch), r1 = device(residual), r2 = device(residual), x1(n * 4), x2(n * 4), f1(4), f2(4), tmp(n * 4);
    f1.zero(); f2.zero(); x1.zero(); x2.zero();
    // Separate launches as DevicePrefill issues them.
    norm(b.as<float>(), w.view, tmp.as<float>(), rows, width, nullptr, f1.as<std::uint32_t>());
    residual_(r1, tmp, rows, width, s);
    check_finite(r1.as<float>(), n, f1.as<std::uint32_t>());
    if (next) norm(r1.as<float>(), following.view, x1.as<float>(), rows, width, nullptr, f1.as<std::uint32_t>());
    const bool fused = norm_residual_rows(b.as<float>(), w.view, r2.as<float>(), rows, width, s, next ? x2.as<float>() : nullptr,
                                          next ? following.view : LinearView{}, f2.as<std::uint32_t>());
    synchronize();
    CHECK(fused);
    CHECK(equal(read<float>(r1, n), read<float>(r2, n)));
    if (next) CHECK(equal(read<float>(x1, n), read<float>(x2, n)));
    const auto a = read<std::uint32_t>(f1, 1)[0], c = read<std::uint32_t>(f2, 1)[0];
    CHECK(a == c && (a != 0) == overflow);
}
void gelu_case(unsigned count, bool overflow) {
    auto g = values(count, count * 3U, 2.0F), u = values(count, count * 5U, 2.0F);
    if (overflow) g[count - 1] = 1.0e20F, u[count - 1] = 1.0e20F;
    Buffer g1 = device(g), g2 = device(g), up = device(u), f1(4), f2(4);
    f1.zero(); f2.zero();
    gelu_product(g1.as<float>(), up.as<float>(), count);
    check_finite(g1.as<float>(), count, f1.as<std::uint32_t>());
    gelu_product_finite(g2.as<float>(), up.as<float>(), count, f2.as<std::uint32_t>());
    synchronize();
    CHECK(equal(read<float>(g1, count), read<float>(g2, count)));
    const auto a = read<std::uint32_t>(f1, 1)[0], c = read<std::uint32_t>(f2, 1)[0];
    CHECK(a == c && (a != 0) == overflow);
}
} // namespace
int main() try {
    require_device();
    unsigned cases = 0;
    for (const unsigned rows : {1U, 7U, 8U, 9U, 511U, 512U, 513U, 1023U, 1024U})
        for (const auto format : {WeightFormat::f32, WeightFormat::bf16})
            for (const bool scaled : {false, true})
                for (const bool next : {false, true}) { norm_residual_case(rows, 2560, format, scaled, next, false); ++cases; }
    for (const unsigned rows : {1U, 7U, 513U}) { norm_residual_case(rows, 2560, WeightFormat::f32, true, true, true); ++cases; }
    for (const unsigned count : {1U, 7U, 255U, 256U, 257U, 513U * 10240U, 1024U * 10240U}) { gelu_case(count, false); ++cases; }
    for (const unsigned count : {1U, 7U, 513U * 256U}) { gelu_case(count, true); ++cases; }
    std::cout << "fused prefill epilogues: " << cases << " cases byte-equal to the separate launches (values, residuals, failure word)\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
