// Operator benchmark and route-equality check for EXL3 decode linears at the
// family's real shapes. Synthetic packed data; no model files. Node-only CLI.
#include <hip/hip_runtime.h>
#include "gemma4/exl3/device.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
namespace gpu = g4::gpu;
namespace {
void hip(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
// Comparison kernel with the cooperative tile layout of the strix-trellis
// GEMV (Apache-2.0, see src/gemma4/exl3/notices): lane t decodes states
// 8t..8t+7, packed FP16 dot2 accumulation, quad reduction. Timing only; it
// is a different reduction order and is never used for model arithmetic.
using Pair = _Float16 __attribute__((ext_vector_type(2)));
template <int Bits> struct Aligned {
    std::uint32_t high, low, tail;
    __device__ explicit Aligned(const std::uint32_t* data) {
        constexpr int count = Bits * 8;
        const int start = ((threadIdx.x & 31) * 8 + 1) * Bits + 256 * Bits - 16;
        const int index = start / 32, shift = start & 31;
        const auto a = data[index % count], b = data[(index + 1) % count], c = data[(index + 2) % count];
        high = (a << shift) | (b >> (32 - shift)); low = (b << shift) | (c >> (32 - shift)); tail = c << shift;
    }
    template <int J> __device__ std::uint32_t state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535U;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535U;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535U;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535U;
    }
};
__device__ Pair mul1_pair(std::uint32_t a, std::uint32_t b) {
    const unsigned lo = __builtin_amdgcn_udot4(a * 0x83dcd12dU, 0x01010101U, 0x6400U, false);
    const unsigned hi = __builtin_amdgcn_udot4(b * 0x83dcd12dU, 0x01010101U, 0x6400U, false);
    const auto s = __builtin_bit_cast(_Float16, std::uint16_t(0x1eee)), o = __builtin_bit_cast(_Float16, std::uint16_t(0xc931));
    return __builtin_elementwise_fma(__builtin_bit_cast(Pair, lo | (hi << 16U)), Pair{s, s}, Pair{o, o});
}
template <int Bits, int Rows>
__global__ __launch_bounds__(128) void cooperative(const std::uint32_t* packed, const _Float16* x, float* y, int input, int output) {
    const int tile_n = blockIdx.x * 4 + threadIdx.x / 32;
    if (tile_n >= output / 16) return;
    const int lane = threadIdx.x & 31;
    float sums[Rows][2] = {};
    for (int k = 0; k < input / 16; ++k) {
        Aligned<Bits> tile(packed + (std::size_t(k) * (output / 16) + tile_n) * Bits * 8);
        Pair act[Rows][2];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int channel = k * 16 + 2 * (lane & 3);
            __builtin_memcpy(&act[r][0], x + std::size_t(r) * input + channel, sizeof(Pair));
            __builtin_memcpy(&act[r][1], x + std::size_t(r) * input + channel + 8, sizeof(Pair));
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t a, b;
            if (pair == 0) { a = tile.template state<0>(); b = tile.template state<1>(); }
            else if (pair == 1) { a = tile.template state<2>(); b = tile.template state<3>(); }
            else if (pair == 2) { a = tile.template state<4>(); b = tile.template state<5>(); }
            else { a = tile.template state<6>(); b = tile.template state<7>(); }
            const Pair w = mul1_pair(a, b);
#pragma unroll
            for (int r = 0; r < Rows; ++r) sums[r][pair / 2] = __builtin_amdgcn_fdot2(w, act[r][pair & 1], sums[r][pair / 2], false);
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float v = sums[r][h]; v += __shfl_down(v, 2, 4); v += __shfl_down(v, 1, 4);
            if (!(lane & 3)) y[std::size_t(r) * output + tile_n * 16 + lane / 4 + h * 8] = v;
        }
}
template <int Rows> void launch_cooperative(g4::LinearView w, const void* x, float* y) {
    const auto* packed = reinterpret_cast<const std::uint32_t*>(w.data + 2ULL * (w.inputs + w.outputs));
    const dim3 grid((w.outputs / 16 + 3) / 4);
    switch (ex::bits(w)) {
#define CASE(B) case B: hipLaunchKernelGGL((cooperative<B, Rows>), grid, dim3(128), 0, nullptr, packed, static_cast<const _Float16*>(x), y, int(w.inputs), int(w.outputs)); break
        CASE(4); CASE(5); CASE(6); CASE(8);
#undef CASE
    }
    hip(hipGetLastError());
}
struct Shape { const char* name; unsigned inputs, outputs; std::vector<unsigned> bits; };
template <class F> double time_us(F&& f, unsigned iterations) {
    f(); hip(hipDeviceSynchronize());
    hipEvent_t a, b; hip(hipEventCreate(&a)); hip(hipEventCreate(&b));
    hip(hipEventRecord(a, nullptr));
    for (unsigned i = 0; i < iterations; ++i) f();
    hip(hipEventRecord(b, nullptr)); hip(hipEventSynchronize(b));
    float ms = 0; hip(hipEventElapsedTime(&ms, a, b));
    hip(hipEventDestroy(a)); hip(hipEventDestroy(b));
    return 1000.0 * ms / iterations;
}
}
int main(int argc, char** argv) {
    try {
        gpu::require_device();
        const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        bool failed = false;
        // All lane-tile decodes of the 65,536 states equal the declared host codebook.
        {
            gpu::Buffer table(65536 * 2); ex::device_codebook_table(table.data()); gpu::synchronize();
            std::vector<std::uint16_t> bits(65536); table.download(bits.data(), bits.size() * 2);
            for (unsigned s = 0; s < 65536; ++s)
                if (g4::bit_cast<unsigned>(g4::fp16_value(bits[s])) != g4::bit_cast<unsigned>(ex::codebook(s)))
                    throw std::runtime_error("packed codebook state " + std::to_string(s));
            std::cout << "{\"check\":\"codebook-65536\",\"verdict\":\"EXACT\"}" << std::endl;
        }
        std::mt19937 rng(787);
        {
            // Grouped transforms (q/k/v-like members with mixed K and outputs) equal separate linears.
            const unsigned inputs = 2560, outs[3] = {2048, 512, 512}, ks[3] = {5, 6, 8};
            std::vector<gpu::Buffer> packed, single, grouped;
            g4::LinearView views[3];
            float* targets[3];
            for (unsigned t = 0; t < 3; ++t) {
                g4::LinearView v{nullptr, outs[t], inputs, std::size_t(inputs) * ks[t] / 8, g4::WeightFormat::exl3_mul1};
                std::vector<std::uint8_t> data(g4::weight_storage_bytes(v));
                for (auto& b : data) b = std::uint8_t(rng());
                for (unsigned i = 0; i < inputs + outs[t]; ++i) {
                    const std::uint16_t h = (i < inputs ? 0x3c00U : 0x2000U) | (rng() & 1U ? 0x8000U : 0U);
                    std::memcpy(data.data() + 2ULL * i, &h, 2);
                }
                packed.emplace_back(data.size()); packed.back().upload(data.data(), data.size());
                v.data = packed.back().as<std::uint8_t>(); views[t] = v;
                single.emplace_back(8ULL * outs[t] * 4); grouped.emplace_back(8ULL * outs[t] * 4);
                targets[t] = grouped.back().as<float>();
            }
            std::vector<float> input(8ULL * inputs);
            std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
            for (auto& v : input) v = uniform(rng);
            gpu::Buffer x(input.size() * 4), scratch(std::max<std::size_t>(ex::device_scratch_bytes, 3ULL * 8 * inputs * 2)), failure(4);
            x.upload(input.data(), input.size() * 4); failure.zero();
            for (unsigned rows = 1; rows <= 8; ++rows) {
                for (unsigned t = 0; t < 3; ++t)
                    ex::device_linear(views[t], x.as<float>(), rows, single[t].as<float>(), scratch.data(), failure.as<unsigned>());
                ex::device_linear_group(views, 3, x.as<float>(), rows, targets, scratch.data(), failure.as<unsigned>());
                gpu::synchronize();
                for (unsigned t = 0; t < 3; ++t) {
                    std::vector<float> a(rows * outs[t]), b(rows * outs[t]);
                    single[t].download(a.data(), a.size() * 4); grouped[t].download(b.data(), b.size() * 4);
                    if (std::memcmp(a.data(), b.data(), a.size() * 4)) throw std::runtime_error("group mismatch R" + std::to_string(rows));
                }
            }
            unsigned flag = 0; failure.download(&flag, 4);
            if (flag) throw std::runtime_error("group failure flag");
            std::cout << "{\"check\":\"group-transforms\",\"verdict\":\"EXACT\"}" << std::endl;
        }
        if (ex::v2_active()) {
            // g4-exl3-v2: device linear == host reference (all non-head shapes, R 1,2,3,8),
            // head R1..4 == serial rows; then DRAM-cold timing of the v2 linear.
            struct V2Shape { const char* name; unsigned inputs, outputs; std::vector<unsigned> bits; };
            const std::vector<V2Shape> v2shapes = {
                {"gate_up", 2560, 10240, {4, 5, 6}}, {"down", 10240, 2560, {4, 5, 6}}, {"q_sliding", 2560, 2048, {5, 6}},
                {"q_global", 2560, 4096, {5, 6}}, {"o_sliding", 2048, 2560, {5, 6}}, {"o_global", 4096, 2560, {5, 6}},
                {"kv_sliding", 2560, 512, {6, 8}}, {"kv_global", 2560, 1024, {6, 8}}, {"ple_gate", 2560, 256, {5}},
                {"ple_projection", 256, 2560, {5}}, {"head", 2560, 262144, {6, 8}}};
            for (const auto& shape : v2shapes)
                for (unsigned k : shape.bits) {
                    g4::LinearView host{nullptr, shape.outputs, shape.inputs, std::size_t(shape.inputs) * k / 8, g4::WeightFormat::exl3_mul1};
                    std::vector<std::uint8_t> data(g4::weight_storage_bytes(host));
                    for (auto& b : data) b = std::uint8_t(rng());
                    for (unsigned i = 0; i < shape.inputs + shape.outputs; ++i) {
                        const std::uint16_t h = (i < shape.inputs ? 0x3c00U : 0x2000U) | (rng() & 1U ? 0x8000U : 0U);
                        std::memcpy(data.data() + 2ULL * i, &h, 2);
                    }
                    host.data = data.data();
                    gpu::Buffer packed(data.size()); packed.upload(data.data(), data.size());
                    auto w = host; w.data = packed.as<std::uint8_t>();
                    std::vector<float> input(8ULL * shape.inputs);
                    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
                    for (auto& v : input) v = uniform(rng);
                    gpu::Buffer x(input.size() * 4), scratch(ex::device_scratch_bytes), out(8ULL * shape.outputs * 4), failure(4);
                    x.upload(input.data(), input.size() * 4); failure.zero();
                    const bool head = shape.outputs > 100000;
                    std::vector<float> serial;
                    for (unsigned rows : head ? std::vector<unsigned>{1, 4} : std::vector<unsigned>{1, 2, 3, 8}) {
                        for (bool shared : {true, false}) {
                            out.zero();
                            ex::device_linear(w, x.as<float>(), rows, out.as<float>(), scratch.data(), failure.as<unsigned>(), shared);
                            gpu::synchronize();
                            std::vector<float> actual(std::size_t(rows) * shape.outputs);
                            out.download(actual.data(), actual.size() * 4);
                            std::vector<float> expected;
                            if (head) {
                                if (rows == 1) { serial = actual; continue; }
                                expected.resize(actual.size());
                                for (unsigned r = 0; r < rows; ++r) {
                                    ex::device_linear(w, x.as<float>() + std::size_t(r) * shape.inputs, 1, out.as<float>(), scratch.data(), failure.as<unsigned>());
                                    gpu::synchronize();
                                    out.download(expected.data() + std::size_t(r) * shape.outputs, std::size_t(shape.outputs) * 4);
                                }
                            } else {
                                expected.resize(actual.size());
                                ex::linear(host, input.data(), rows, expected.data());
                            }
                            for (std::size_t i = 0; i < actual.size(); ++i)
                                if (std::memcmp(&actual[i], &expected[i], 4)) {
                                    std::cout << "{\"v2_mismatch\":\"" << shape.name << "\",\"K\":" << k << ",\"rows\":" << rows
                                              << ",\"shared\":" << shared << ",\"index\":" << i << "}" << std::endl;
                                    failed = true;
                                    break;
                                }
                        }
                    }
                    unsigned flag = 0; failure.download(&flag, 4);
                    if (flag) throw std::runtime_error("v2 arithmetic failure flag");
                    std::vector<gpu::Buffer> copies;
                    std::vector<g4::LinearView> cold;
                    for (std::size_t total = 0; total < (160ULL << 20) || cold.empty(); total += data.size()) {
                        copies.emplace_back(data.size()); copies.back().upload(data.data(), data.size());
                        cold.push_back(host); cold.back().data = copies.back().as<std::uint8_t>();
                    }
                    unsigned it = 0;
                    for (unsigned rows : {1U, 4U, 8U}) {
                        const double us = time_us([&] { ex::device_linear(cold[it++ % cold.size()], x.as<float>(), rows, out.as<float>(), scratch.data(), failure.as<unsigned>()); },
                                                  head ? 8 : 40);
                        std::cout << "{\"v2_shape\":\"" << shape.name << "\",\"K\":" << k << ",\"rows\":" << rows << ",\"splits\":"
                                  << ex::v2_splits(shape.inputs, shape.outputs) << ",\"us\":" << us << ",\"core_gbs\":"
                                  << double(shape.inputs) * shape.outputs * k / 8 / us / 1000.0 << "}" << std::endl;
                    }
                }
            if (failed) throw std::runtime_error("v2 mismatch");
            std::cout << "{\"check\":\"v2-device-host\",\"verdict\":\"EXACT\"}" << std::endl;
            return 0;
        }
        const std::vector<Shape> shapes = {
            {"gate_up", 2560, 10240, {4, 5, 6}}, {"down", 10240, 2560, {4, 5, 6}},
            {"q_sliding", 2560, 2048, {5, 6}},   {"q_global", 2560, 4096, {5, 6}},
            {"o_sliding", 2048, 2560, {5, 6}},   {"o_global", 4096, 2560, {5, 6}},
            {"kv_sliding", 2560, 512, {6, 8}},   {"kv_global", 2560, 1024, {6, 8}},
            {"head", 2560, 262144, {6, 8}},
        };
        for (const auto& shape : shapes)
            for (unsigned k : shape.bits) {
                g4::LinearView host{nullptr, shape.outputs, shape.inputs, std::size_t(shape.inputs) * k / 8, g4::WeightFormat::exl3_mul1};
                std::vector<std::uint8_t> data(g4::weight_storage_bytes(host));
                for (auto& b : data) b = std::uint8_t(rng());
                for (unsigned i = 0; i < shape.inputs + shape.outputs; ++i) {
                    const std::uint16_t h = (i < shape.inputs ? 0x3c00U : 0x2000U) | (rng() & 1U ? 0x8000U : 0U);
                    std::memcpy(data.data() + 2ULL * i, &h, 2);
                }
                gpu::Buffer packed(data.size()); packed.upload(data.data(), data.size());
                auto w = host; w.data = packed.as<std::uint8_t>();
                // Timing cycles through copies totalling >= 160 MB so weights stream from DRAM,
                // not the 32 MB last-level cache, as in a full decode pass.
                std::vector<gpu::Buffer> copies;
                std::vector<g4::LinearView> cold;
                for (std::size_t total = 0; total < (160ULL << 20) || cold.empty(); total += data.size()) {
                    copies.emplace_back(data.size()); copies.back().upload(data.data(), data.size());
                    cold.push_back(host); cold.back().data = copies.back().as<std::uint8_t>();
                }
                unsigned it = 0;
                const auto next = [&]() -> const g4::LinearView& { return cold[it++ % cold.size()]; };
                std::vector<float> input(8ULL * shape.inputs);
                std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
                for (auto& v : input) v = uniform(rng);
                gpu::Buffer x(input.size() * 4), scratch(std::max<std::size_t>(ex::device_scratch_bytes, 8ULL * shape.inputs * 2)), out(8ULL * shape.outputs * 4), failure(4);
                x.upload(input.data(), input.size() * 4); failure.zero();
                ex::device_input(w, x.as<float>(), 8, scratch.data(), failure.as<unsigned>());
                gpu::synchronize();
                // Route equality on the core (pre output transform) bytes for every row count.
                for (unsigned rows = 1; rows <= 8; ++rows) {
                    if (quick && rows != 1 && rows != 8) continue;
                    std::vector<std::vector<float>> results;
                    for (auto route : {ex::CoreRoute::legacy, ex::CoreRoute::lane8, ex::CoreRoute::lane4, ex::CoreRoute::staged8, ex::CoreRoute::staged4}) {
                        out.zero();
                        ex::device_core(w, scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), route);
                        gpu::synchronize();
                        results.emplace_back(std::size_t(rows) * shape.outputs);
                        out.download(results.back().data(), results.back().size() * 4);
                    }
                    for (std::size_t r = 1; r < results.size(); ++r)
                        for (std::size_t i = 0; i < results[0].size(); ++i)
                            if (std::memcmp(&results[0][i], &results[r][i], 4)) {
                                std::cout << "{\"mismatch\":\"" << shape.name << "\",\"K\":" << k << ",\"rows\":" << rows << ",\"route\":" << r
                                          << ",\"index\":" << i << ",\"legacy\":" << results[0][i] << ",\"route_value\":" << results[r][i] << "}" << std::endl;
                                failed = true;
                                break;
                            }
                }
                unsigned flag = 0; failure.download(&flag, 4);
                if (flag) throw std::runtime_error("arithmetic failure flag");
                const double core_bytes = double(shape.inputs) * shape.outputs * k / 8;
                const unsigned iterations = shape.outputs > 100000 ? 8 : 40;
                for (unsigned rows : {1U, 2U, 4U, 8U}) {
                    std::string line = std::string("{\"shape\":\"") + shape.name + "\",\"I\":" + std::to_string(shape.inputs) +
                        ",\"O\":" + std::to_string(shape.outputs) + ",\"K\":" + std::to_string(k) + ",\"rows\":" + std::to_string(rows);
                    const auto report = [&](const char* name, double us) {
                        line += std::string(",\"") + name + "_us\":" + std::to_string(us) + ",\"" + name + "_gbs\":" + std::to_string(core_bytes / us / 1000.0);
                    };
                    report("legacy", time_us([&] { ex::device_core(next(), scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), ex::CoreRoute::legacy); }, iterations));
                    report("lane8", time_us([&] { ex::device_core(next(), scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), ex::CoreRoute::lane8); }, iterations));
                    report("lane4", time_us([&] { ex::device_core(next(), scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), ex::CoreRoute::lane4); }, iterations));
                    report("staged8", time_us([&] { ex::device_core(next(), scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), ex::CoreRoute::staged8); }, iterations));
                    report("staged4", time_us([&] { ex::device_core(next(), scratch.data(), rows, out.as<float>(), failure.as<unsigned>(), ex::CoreRoute::staged4); }, iterations));
                    report("auto_linear", time_us([&] { ex::device_linear(next(), x.as<float>(), rows, out.as<float>(), scratch.data(), failure.as<unsigned>()); }, iterations));
                    if (rows == 1) report("coop", time_us([&] { launch_cooperative<1>(next(), scratch.data(), out.as<float>()); }, iterations));
                    if (rows == 4) report("coop", time_us([&] { launch_cooperative<4>(next(), scratch.data(), out.as<float>()); }, iterations));
                    if (rows == 8) report("coop", time_us([&] { launch_cooperative<8>(next(), scratch.data(), out.as<float>()); }, iterations));
                    std::cout << line << "}" << std::endl;
                }
            }
        if (failed) throw std::runtime_error("route mismatch");
        std::cout << "{\"check\":\"lane-tile-routes\",\"verdict\":\"EXACT\"}" << std::endl;
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
