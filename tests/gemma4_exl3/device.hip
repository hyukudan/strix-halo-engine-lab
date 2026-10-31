#include <hip/hip_runtime.h>
#include "gemma4/exl3/device.hpp"
#include "gemma4/exl3/emulator.hpp"
#include "gemma4/prefill/device.hpp"
#include <cstring>
#include <fstream>
#include <iostream>

namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
namespace gpu = g4::gpu;
namespace {
template <class T> std::vector<T> read(const std::string& name) {
    std::ifstream f(std::filesystem::path(G4_EXL3_FIXTURES) / name, std::ios::binary | std::ios::ate);
    if (!f || std::size_t(f.tellg()) % sizeof(T)) throw std::runtime_error("fixture read");
    std::vector<T> data(std::size_t(f.tellg()) / sizeof(T));
    f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(T)));
    if (!f) throw std::runtime_error("fixture truncated");
    return data;
}
void same(std::span<const float> a, std::span<const float> b, const char* name) {
    if (a.size() != b.size()) throw std::runtime_error("shape mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(a.data() + i, b.data() + i, 4))
            throw std::runtime_error(std::string(name) + " mismatch at " + std::to_string(i));
}
__global__ void decode(g4::LinearView w, float* out) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.inputs * w.outputs) out[i] = ex::rotated_at(w, i / w.outputs, i % w.outputs);
}
}
int main() {
    try {
        gpu::require_device();
        const auto input = read<float>("input.f32");
        gpu::Buffer x(17ULL * 256 * 4), out(256ULL * 128 * 4), scratch(17ULL * 256 * 2), failure(4), maximum(4);
        for (unsigned k : {4U, 5U, 6U, 8U}) {
            auto data = read<std::uint8_t>("k" + std::to_string(k) + ".bin");
            g4::LinearView host{data.data(), 128, 256, 256ULL * k / 8, g4::WeightFormat::exl3_mul1};
            gpu::Buffer packed(data.size()); packed.upload(data.data(), data.size());
            auto device = host; device.data = packed.as<std::uint8_t>();
            const auto rotated = read<std::uint16_t>("k" + std::to_string(k) + "-rotated.f16");
            hipLaunchKernelGGL(decode, dim3(128), dim3(256), 0, nullptr, device, out.as<float>());
            gpu::synchronize();
            std::vector<float> decoded(rotated.size()); out.download(decoded.data(), decoded.size() * 4);
            for (std::size_t i = 0; i < decoded.size(); ++i)
                if (g4::bit_cast<unsigned>(decoded[i]) != g4::bit_cast<unsigned>(g4::fp16_value(rotated[i]))) throw std::runtime_error("device rotated decode");
            x.upload(input.data(), input.size() * 4);
            const auto expected = read<float>("k" + std::to_string(k) + "-output.f32");
            for (unsigned rows = 1; rows <= 8; ++rows)
                for (bool shared : {false, true}) {
                    failure.zero();
                    ex::device_linear(device, x.as<float>(), rows, out.as<float>(), scratch.data(), failure.as<unsigned>(), shared);
                    gpu::synchronize();
                    unsigned flag = 0; failure.download(&flag, 4); if (flag) throw std::runtime_error("device arithmetic failure");
                    std::vector<float> actual(rows * 128); out.download(actual.data(), actual.size() * 4);
                    same(actual, std::span(expected).first(actual.size()), "decode R1..8 shared/plain");
                }
            g4::prefill::PreparedLinear panels(device);
            for (unsigned rows : {1U, 8U, 17U}) {
                std::vector<float> values(rows * 256);
                for (std::size_t i = 0; i < values.size(); ++i) values[i] = input[i % input.size()];
                x.upload(values.data(), values.size() * 4);
                const auto reference = ex::emulate_prefill(host, values, rows);
                for (const void* p : {static_cast<const void*>(nullptr), static_cast<const void*>(panels.panels().data())}) {
                    failure.zero(); maximum.zero();
                    ex::device_prefill(device, x.as<float>(), rows, out.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(), p);
                    gpu::synchronize();
                    unsigned flag = 0; failure.download(&flag, 4); if (flag) throw std::runtime_error("prefill arithmetic failure");
                    std::vector<float> actual(rows * 128); out.download(actual.data(), actual.size() * 4);
                    same(actual, reference, "WMMA packed/prepared/emulator");
                }
            }
            std::vector<float> overflow(256, g4::bit_cast<float>(0x7f7fffffU));
            x.upload(overflow.data(), overflow.size() * 4); failure.zero();
            ex::device_linear(device, x.as<float>(), 1, out.as<float>(), scratch.data(), failure.as<unsigned>());
            gpu::synchronize();
            unsigned flag = 0; failure.download(&flag, 4);
            if (!(flag & 4U)) throw std::runtime_error("device FP16 overflow accepted");
        }
        std::cout << "EXL3 K4/K5/K6/K8 decoder, R1..8, shared OFF, WMMA panels/emulator EXACT\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
