#include "cuda/core/dense/rows_f32_input_major.cuh"
#include "mmidx_frozen.cuh"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kIn = mmidx_frozen::kHidden;
constexpr unsigned kGuard = 11;
constexpr std::uint32_t kCanary = 0x7fc00039U;

struct Case {
    unsigned rows, out, stride, offset;
    bool arithmetic = false;
};

constexpr Case cases[] = {
    {1, 1, 1, 0}, {15, 127, 131, 2}, {16, 128, 128, 0}, {17, 129, 137, 3},
    {33, 256, 512, 256}, {65, 128, 128, 0}, {3, 3, 7, 1, true},
};

void check(cudaError_t status, const char* step) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(step) + ": " + cudaGetErrorString(status));
}

class DeviceFloats {
public:
    explicit DeviceFloats(const std::vector<float>& data) : bytes_(data.size() * sizeof(float)) {
        check(cudaMalloc(&data_, bytes_), "allocate");
        const auto status = cudaMemcpy(data_, data.data(), bytes_, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) cudaFree(data_);
        check(status, "upload");
    }
    ~DeviceFloats() { cudaFree(data_); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;

    float* get() const { return data_; }
    void read(std::vector<float>& data) const {
        check(cudaMemcpy(data.data(), data_, bytes_, cudaMemcpyDeviceToHost), "download");
    }

private:
    std::size_t bytes_;
    float* data_ = nullptr;
};

float random_finite(std::mt19937& rng) {
    const auto bits = static_cast<std::uint32_t>(rng());
    return std::bit_cast<float>((bits & 0x807fffffU) | ((124U + (bits >> 24U) % 6U) << 23U));
}

void run(const Case& c, std::mt19937& rng) {
    std::vector<float> x(static_cast<std::size_t>(c.rows) * kIn), w(static_cast<std::size_t>(kIn) * c.out);
    if (c.arithmetic) {
        x[0] = -1.0F;
        x[1] = 0x1.000002p0F;
        w[0] = 1.0F;
        w[c.out] = 0x1.fffffcp-1F;
        x[kIn] = 0x1p24F;
        x[kIn + 1] = 1.0F;
        x[kIn + 2] = -0x1p24F;
        w[1] = w[c.out + 1] = w[2 * c.out + 1] = 1.0F;
        x[2 * kIn] = std::bit_cast<float>(1U);
        w[2] = 1.0F;
    } else {
        for (auto& v : x) v = random_finite(rng);
        for (auto& v : w) v = random_finite(rng);
    }
    std::vector<float> expected(static_cast<std::size_t>(c.rows) * c.stride + 2 * kGuard,
                                std::bit_cast<float>(kCanary));
    auto actual = expected;
    DeviceFloats dx(x), dw(w), d_expected(expected), d_actual(actual);
    const auto begin = kGuard + c.offset;
    mmidx_frozen::dense_t_kernel<<<dim3((c.rows + 15) / 16, (c.out + 127) / 128), 128>>>(
        dx.get(), dw.get(), d_expected.get() + begin, c.rows, c.out, c.stride);
    check(cudaGetLastError(), "frozen kernel launch");
    check(strix::cuda_core::dense_f32_input_major_rows(dx.get(), dw.get(), d_actual.get() + begin,
        c.rows, kIn, c.out, c.stride, nullptr), "shared kernel launch");
    d_expected.read(expected);
    d_actual.read(actual);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool inside = i >= kGuard && i < actual.size() - kGuard;
        const auto col = inside ? (i - kGuard) % c.stride : 0;
        const bool live = inside && col >= c.offset && col < c.offset + c.out;
        const auto a = std::bit_cast<std::uint32_t>(actual[i]);
        const auto b = std::bit_cast<std::uint32_t>(expected[i]);
        if (a != b || (live ? a == kCanary : a != kCanary))
            throw std::runtime_error("bit or canary mismatch at element " + std::to_string(i) +
                                     " for rows=" + std::to_string(c.rows) + " out=" + std::to_string(c.out));
    }
    if (c.arithmetic && (std::bit_cast<std::uint32_t>(actual[begin]) != std::bit_cast<std::uint32_t>(-0x1p-46F) ||
        std::bit_cast<std::uint32_t>(actual[begin + c.stride + 1]) != 0U ||
        std::bit_cast<std::uint32_t>(actual[begin + 2 * c.stride + 2]) != 1U))
        throw std::runtime_error("fused, ascending, or subnormal arithmetic discriminator failed");
}

}  // namespace

int main() {
    try {
        std::mt19937 rng(0x4d4d4944U);
        std::size_t live = 0;
        for (const auto& c : cases) {
            run(c, rng);
            live += static_cast<std::size_t>(c.rows) * c.out;
        }
        std::cout << "MMIDX EXACT PASS cases=" << sizeof(cases) / sizeof(*cases) << " live=" << live
                  << " canaries=PASS arithmetic=PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "MMIDX EXACT FAIL: " << error.what() << '\n';
        return 1;
    }
}
