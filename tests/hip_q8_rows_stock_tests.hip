// Stock Q8_0 sites (src/hip_q8_rows_sites.hpp) of the shared q8_rows core: LinearRowsF32<R> (R = 2..8, one launch)
// against LinearF32 launched once per row, bitwise, at verify-window shapes and edge shapes, with smooth, edge-scale and
// NaN/Inf inputs and the BF16 store epilogue. Run once with the default launch (R = 4 / 8 on the VOPD coefficient
// kernels for FP32 widths >= 1024, R = 6 / 7 padded to the 8-row one from 1536, R = 3 stepped at 6144 -> 2560) and
// once with STRIX_Q8_ROWS_VOPD=0 (project_rows_kernel, stepped at 6144 -> 2560 R = 3 / 6 / 7).
// STRIX_Q8_ROWS_BENCH=1 adds warm / cold median timings of R = 1..8 at the projection shapes (us and weight GB/s).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include <hip/hip_runtime.h>
#include "hip_q8_rows_sites.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strix;

void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) { check(hipMalloc(&p, size), "allocate"); }
    ~Buffer() { if (hipFree(p) != hipSuccess) std::_Exit(3); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    void upload(const void* data) const { check(hipMemcpy(p, data, bytes, hipMemcpyHostToDevice), "upload"); }
    std::vector<std::uint32_t> words() const {
        std::vector<std::uint32_t> v(bytes / 4U);
        check(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost), "download");
        return v;
    }
};

// Flavor 0: smooth scales / codes; 1: edge scales (zero, signed zero, subnormal, large) and extreme codes.
std::vector<std::uint8_t> weights_bytes(std::uint32_t width, std::uint32_t outputs, unsigned flavor) {
    const auto blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::uint8_t> result(blocks * 34U);
    constexpr std::array<std::uint16_t, 8> scales{0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x2000, 0xa000};
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto scale = flavor == 1U ? scales[block % scales.size()] : static_cast<std::uint16_t>(0x2400U + (block % 3U) * 0x400U);
        result[block * 34U] = static_cast<std::uint8_t>(scale & 0xffU);
        result[block * 34U + 1U] = static_cast<std::uint8_t>(scale >> 8U);
        for (std::size_t lane = 0; lane < 32U; ++lane) {
            const int code = flavor == 1U ? (lane % 2U ? -128 : 127) : static_cast<int>((block * 7U + lane * 3U) % 29U) - 14;
            result[block * 34U + 2U + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        }
    }
    return result;
}
// Flavor 0: FP32 values using the low mantissa; 1: BF16-edge bit patterns; 2: sparse NaN / +-Inf.
std::vector<float> input_values(std::uint32_t width, unsigned row, unsigned flavor) {
    std::vector<float> result(width);
    constexpr std::array<std::uint16_t, 8> edges{0, 0x8000, 1, 0x8001, 0x0080, 0x8080, 0x3e80, 0xbe80};
    for (std::uint32_t i = 0; i < width; ++i) {
        const float value = static_cast<float>(static_cast<int>((i * 5U + row * 11U) % 31U) - 15) * 0.015625F * static_cast<float>(row + 1U);
        std::uint32_t bits = std::bit_cast<std::uint32_t>(value) ^ ((i * 2654435761U + row) & 0x7fffU);
        if (flavor == 1U) bits = static_cast<std::uint32_t>(edges[(i + row) % edges.size()]) << 16U;
        if (flavor == 2U && i % 97U == row) bits = (i / 97U) % 3U == 0U ? 0x7f800000U : (i / 97U) % 3U == 1U ? 0xff800000U : 0x7fc01234U + row;
        result[i] = std::bit_cast<float>(bits);
    }
    return result;
}

struct Case {
    std::uint32_t width, outputs;
    std::unique_ptr<Buffer> weights, x, bias, y;
    Case(std::uint32_t w, std::uint32_t o, unsigned flavor, unsigned rows, unsigned input_flavor)
        : width(w), outputs(o) {
        const auto bytes = weights_bytes(w, o, flavor);
        weights = std::make_unique<Buffer>(bytes.size());
        weights->upload(bytes.data());
        std::vector<float> host;
        for (unsigned r = 0; r < rows; ++r) {
            const auto values = input_values(w, r, input_flavor);
            host.insert(host.end(), values.begin(), values.end());
        }
        x = std::make_unique<Buffer>(host.size() * 4U);
        x->upload(host.data());
        std::vector<float> b(o);
        for (std::uint32_t i = 0; i < o; ++i) b[i] = static_cast<float>(static_cast<int>(i % 13U) - 6) * 0.0078125F;
        bias = std::make_unique<Buffer>(b.size() * 4U);
        bias->upload(b.data());
        y = std::make_unique<Buffer>(static_cast<std::size_t>(rows) * o * 4U);
    }
    const float* row(unsigned r) const { return x->as<float>() + static_cast<std::size_t>(r) * width; }
};

void launch_r1(const Case& c, unsigned r, float* out, bool round, hipStream_t s) {
    q8_rows::LinearF32 site{};
    site.input[0] = c.row(r);
    site.segments = q8_rows::segments({{c.weights->p, c.outputs}});
    site.width = c.width;
    site.outputs = c.outputs;
    site.out = out;
    site.bias = round ? c.bias->as<float>() : nullptr;
    site.round = round;
    q8_rows::launch(site, false, s);
}
template <unsigned R> void launch_rows(const Case& c, float* out, bool round, hipStream_t s) {
    q8_rows::LinearRowsF32<R> site{};
    for (unsigned r = 0; r < R; ++r) site.input[r] = c.row(r);
    site.segments = q8_rows::segments({{c.weights->p, c.outputs}});
    site.width = c.width;
    site.outputs = c.outputs;
    site.out = out;
    site.bias = round ? c.bias->as<float>() : nullptr;
    site.round = round;
    q8_rows::launch(site, false, s);
}
template <unsigned... Rs> void dispatch(unsigned rows, const Case& c, float* out, bool round, hipStream_t s,
                                         std::integer_sequence<unsigned, Rs...>) {
    const bool hit = ((rows == Rs + 1U ? (launch_rows<Rs + 1U>(c, out, round, s), true) : false) || ...);
    require(hit, "rows out of range");
}
void launch_any(unsigned rows, const Case& c, float* out, bool round, hipStream_t s) {
    if (rows == 1U) launch_r1(c, 0U, out, round, s);
    else dispatch(rows, c, out, round, s, std::make_integer_sequence<unsigned, 8U>{});
}

// One window against R single-row launches; returns the number of compared words.
std::size_t compare(hipStream_t s, unsigned rows, std::uint32_t w, std::uint32_t o, unsigned flavor, unsigned input_flavor,
                    bool round) {
    Case c(w, o, flavor, rows, input_flavor);
    Buffer reference(static_cast<std::size_t>(rows) * o * 4U);
    check(hipMemsetAsync(c.y->p, 0xa5, c.y->bytes, s), "poison");
    check(hipMemsetAsync(reference.p, 0x5a, reference.bytes, s), "poison");
    for (unsigned r = 0; r < rows; ++r) launch_r1(c, r, reference.as<float>() + static_cast<std::size_t>(r) * o, round, s);
    launch_any(rows, c, c.y->as<float>(), round, s);
    check(hipGetLastError(), "launch");
    check(hipStreamSynchronize(s), "sync");
    const auto a = c.y->words(), b = reference.words();
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            throw std::runtime_error("R=" + std::to_string(rows) + " " + std::to_string(w) + "->" + std::to_string(o) + " flavor " +
                std::to_string(flavor) + "/" + std::to_string(input_flavor) + (round ? " round" : "") + ": row " +
                std::to_string(i / o) + " out " + std::to_string(i % o) + " differs from LinearF32");
        }
    return a.size();
}

__global__ void flush_cache(const uint4* data, std::size_t count, unsigned* sink) {
    uint4 acc{};
    for (std::size_t i = blockIdx.x * 256U + threadIdx.x; i < count; i += static_cast<std::size_t>(gridDim.x) * 256U) {
        const auto v = data[i];
        acc.x ^= v.x;
        acc.y ^= v.w;
    }
    if (acc.x == 0x9e3779b9U && acc.y == 1U) *sink = acc.x;
}
float median_us(const Case& c, unsigned rows, bool cold, const Buffer& flush, hipStream_t s) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event");
    check(hipEventCreate(&stop), "event");
    std::vector<float> samples;
    for (unsigned it = 0; it < 28U; ++it) {
        if (cold) {
            STRIX_HIP_LAUNCH_GGL(Rdna3Layout, flush_cache, dim3(1024), dim3(256), 0, s, flush.as<const uint4>(), flush.bytes / 16U, flush.as<unsigned>());
            check(hipStreamSynchronize(s), "flush");
        }
        check(hipEventRecord(start, s), "record");
        launch_any(rows, c, c.y->as<float>(), true, s);
        check(hipEventRecord(stop, s), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 4U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(start), "event");
    check(hipEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() try {
    hipStream_t s = nullptr;
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    const char* vopd = std::getenv("STRIX_Q8_ROWS_VOPD");
    // Verify-window projection shapes (FP32 inputs 3072 / 1024 / 1536 / 8192 wide) and edge shapes: narrow outputs, a
    // partial last 256-column chunk, widths under the VOPD threshold.
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 12> shapes{{{3072U, 1024U}, {1024U, 2048U}, {3072U, 1536U},
        {1536U, 3072U}, {8192U, 3072U}, {3072U, 576U}, {3072U, 128U}, {1056U, 33U}, {1568U, 33U}, {1024U, 7U}, {512U, 5U}, {6144U, 2560U}}};
    std::size_t words = 0, cases = 0;
    for (unsigned rows = 2U; rows <= 8U; ++rows)
        for (const auto& [w, o] : shapes)
            for (unsigned flavor = 0; flavor < 2U; ++flavor)
                for (unsigned input_flavor = 0; input_flavor < 3U; ++input_flavor) {
                    words += compare(s, rows, w, o, flavor, input_flavor, input_flavor == 0U);
                    ++cases;
                }
    std::cout << "q8_rows stock sites: LinearRowsF32<2..8> equal LinearF32 per row bitwise: " << cases << " cases, " << words
              << " words (STRIX_Q8_ROWS_VOPD=" << (vopd ? vopd : "default") << ")\n";
    if (std::getenv("STRIX_Q8_ROWS_BENCH")) {
        Buffer flush(256ULL << 20U);
        check(hipMemset(flush.p, 1, flush.bytes), "flush init");
        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 8> bench{{{3072U, 8192U}, {1024U, 8192U}, {8192U, 3072U},
            {3072U, 1024U}, {3072U, 1536U}, {1536U, 3072U}, {3072U, 576U}, {6144U, 2560U}}};
        for (const bool cold : {false, true})
            for (const auto& [w, o] : bench) {
                Case c(w, o, 0U, 8U, 0U);
                const double bytes = static_cast<double>(w / 32U) * o * 34.0;
                std::cout << "BENCH " << (cold ? "cold" : "warm") << " " << w << "->" << o << " vopd=" << (vopd ? vopd : "default");
                for (unsigned rows = 1U; rows <= 8U; ++rows) {
                    const float us = median_us(c, rows, cold, flush, s);
                    std::cout << " R" << rows << "=" << std::fixed << std::setprecision(1) << us << "us/" << std::setprecision(0)
                              << bytes / (static_cast<double>(us) * 1e3) << "GBs";
                }
                std::cout << '\n';
            }
    }
    check(hipStreamDestroy(s), "stream");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "q8_rows stock fixture failed: " << error.what() << '\n';
    return 1;
}
