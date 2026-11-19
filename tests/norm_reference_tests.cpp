#include "../src/ops/norm/reference.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {
enum class Format { f32, bf16, f16 };

struct HostTraits {
    using Input = float;
    using Output = float;
    using Scale = void;
    using ScaleFormat = Format;
    using Stream = void*;
    static constexpr float epsilon = 1e-5F;
    static constexpr auto reduction_order = strix::ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned n) { return n >= 512 ? 256U : 128U; }
    static float inv_sqrt(float x) { return 1.0F / std::sqrt(x); }
    static void launch(const float* x, const void* w, Format format, unsigned n, float eps,
                       float* y, unsigned rows, void* stream);
    static float load(const void* w, Format format, std::size_t i) {
        if (format == Format::f32) return static_cast<const float*>(w)[i];
        const auto bits = static_cast<const std::uint16_t*>(w)[i];
        if (format == Format::f16) {
            // Fixtures use finite, normal half scales, including both signs.
            const float magnitude = std::ldexp(1.0F + static_cast<float>(bits & 1023U) / 1024.0F,
                                              static_cast<int>((bits >> 10U) & 31U) - 15);
            return (bits & 32768U) ? -magnitude : magnitude;
        }
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }
};

using Norm = strix::ops::norm::Norm<HostTraits>;
unsigned checks = 0;

void require(bool condition, const char* what) {
    ++checks;
    if (!condition) throw std::runtime_error(what);
}

bool bits_equal(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

void HostTraits::launch(const float* x, const void* w, Format format, unsigned n, float eps,
                        float* y, unsigned rows, void* stream) {
    require(stream == &checks, "stream forwarding");
    strix::ops::norm::reference<HostTraits>(x, w, format, n, eps, y, rows);
}

// Independent recursive evaluation of GLM's halving tree (bit-reversed leaves).
float tree(const std::vector<float>& partial, unsigned first, unsigned stride) {
    if (stride == partial.size()) return partial[first];
    return tree(partial, first, stride * 2U) + tree(partial, first + stride, stride * 2U);
}

void numeric_case(unsigned n, Format format, bool weighted, float eps) {
    constexpr unsigned rows = 3;
    const std::size_t count = static_cast<std::size_t>(rows) * n;
    std::vector<float> x(count), scales(n), y(count + 2U, -123.0F), expected(count);
    std::vector<std::uint16_t> packed(n);
    for (std::size_t i = 0; i < count; ++i)
        x[i] = std::ldexp(static_cast<float>(static_cast<int>((i * 37U) % 101U) - 50),
                          static_cast<int>(i % 9U) - 8);
    for (unsigned i = 0; i < n; ++i) {
        scales[i] = static_cast<float>(static_cast<int>(i % 13U) - 6) / 8.0F;
        packed[i] = static_cast<std::uint16_t>((format == Format::bf16 ? 0x3e00U : 0x3400U)
                                             + (i % 7U) * 16U + ((i & 1U) ? 0x8000U : 0U));
    }
    const void* w = weighted ? (format == Format::f32 ? static_cast<const void*>(scales.data())
                                                     : static_cast<const void*>(packed.data())) : nullptr;
    for (unsigned row = 0; row < rows; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * n;
        std::vector<float> partial(Norm::lanes(n), 0.0F);
        double exact_ss = 0;
        for (unsigned i = 0; i < n; ++i) {
            const float value = x[offset + i];
            partial[i % partial.size()] = std::fma(value, value, partial[i % partial.size()]);
            exact_ss += static_cast<double>(value) * value;
        }
        const float r = 1.0F / std::sqrt(tree(partial, 0, 1) / static_cast<float>(n) + eps);
        for (unsigned i = 0; i < n; ++i) {
            const float weight = w ? HostTraits::load(w, format, i) : 1.0F;
            expected[offset + i] = w ? (x[offset + i] * r) * weight : x[offset + i] * r;
            const double oracle = static_cast<double>(x[offset + i]) /
                                  std::sqrt(exact_ss / n + static_cast<double>(eps)) * weight;
            require(std::abs(static_cast<double>(expected[offset + i]) - oracle) <=
                        2e-6 * (1.0 + std::abs(oracle)), "mathematical RMSNorm oracle");
        }
    }
    strix::ops::norm::reference<HostTraits>(x.data(), w, format, n, eps, y.data() + 1, rows);
    require(y.front() == -123.0F && y.back() == -123.0F, "row bounds");
    for (std::size_t i = 0; i < count; ++i) require(bits_equal(y[i + 1U], expected[i]), "reduction/output bits");
    Norm::run(x.data(), w, format, n, eps, x.data(), rows, &checks);
    for (std::size_t i = 0; i < count; ++i) require(bits_equal(x[i], expected[i]), "in-place bits");
}
struct Bf16DoubleTraits : HostTraits {
    static constexpr auto output_form = strix::ops::norm::OutputForm::bf16_double;
};

// Independent BF16 round-to-nearest-even of a finite float (fixture values are finite).
float rne(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    u = (u + 0x7FFFU + ((u >> 16U) & 1U)) & 0xFFFF0000U;
    float r;
    std::memcpy(&r, &u, sizeof(r));
    return r;
}

// bf16_double: y = R(w * R(x*r)) and unit scale y = R(x*r), with x*r the plain unit-scale output (same reduction).
void bf16_double_case(unsigned n, bool weighted) {
    std::vector<float> x(n), w(n), plain(n), y(n);
    for (unsigned i = 0; i < n; ++i) {
        x[i] = std::sin(0.37F * static_cast<float>(i) + 0.1F) * 3.0F;
        w[i] = 1.0F + 0.01F * static_cast<float>(i % 17U);
    }
    strix::ops::norm::reference<HostTraits>(x.data(), nullptr, Format::f32, n, 1e-5F, plain.data(), 1);
    strix::ops::norm::reference<Bf16DoubleTraits>(x.data(), weighted ? w.data() : nullptr, Format::f32, n, 1e-5F, y.data(), 1);
    for (unsigned i = 0; i < n; ++i) {
        const float want = weighted ? rne(w[i] * rne(plain[i])) : rne(plain[i]);
        require(bits_equal(y[i], want), "bf16_double form");
    }
}

}  // namespace

int main() {
    try {
        static_assert(Norm::epsilon == 1e-5F);
        static_assert(Norm::lanes(511) == 128 && Norm::lanes(512) == 256);
        unsigned cases = 0;
        for (unsigned n : {1U, 7U, 127U, 128U, 129U, 255U, 256U, 511U, 512U, 513U, 1024U, 1536U, 4096U})
            for (Format format : {Format::f32, Format::bf16, Format::f16})
                for (bool weighted : {false, true})
                    for (float eps : {0.0F, Norm::epsilon, 1e-20F}) {
                        numeric_case(n, format, weighted, eps);
                        ++cases;
                    }
        static_assert(strix::ops::norm::output_form_of<HostTraits>() == strix::ops::norm::OutputForm::plain);
        for (unsigned n : {1U, 96U, 512U, 1536U, 3072U})
            for (bool weighted : {false, true}) bf16_double_case(n, weighted);
        float zero[4] = {}, y[4] = {1, 1, 1, 1};
        strix::ops::norm::reference<HostTraits>(zero, nullptr, Format::f32, 4, Norm::epsilon, y, 1);
        for (float value : y) require(value == 0.0F, "zero row");
        strix::ops::norm::reference<HostTraits>(nullptr, nullptr, Format::f32, 4, Norm::epsilon, y, 0);
        require(y[0] == 0.0F, "zero rows");
        bool rejected = false;
        try { strix::ops::norm::reference<HostTraits>(zero, nullptr, Format::f32, 0, Norm::epsilon, y, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "zero width rejected");
        std::cout << "PASS: " << cases << " RMSNorm cases, " << checks << " checks (FP32/BF16/F16/unit scale, tails, rows, in-place)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
