#include "strix/hip_shared_expert_launcher.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace strix;
void check(hipError_t e, const char *op) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string(op) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Owner {
    hipStream_t stream{};
    std::vector<void *> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() { check(hipStreamCreate(&stream), "shared projection stream"); }
    ~Owner() {
        (void)hipStreamSynchronize(stream);
        for (auto e : executions)
            (void)hipGraphExecDestroy(e);
        for (auto g : graphs)
            (void)hipGraphDestroy(g);
        for (auto p : allocations)
            (void)hipFree(p);
        (void)hipStreamDestroy(stream);
    }
    HipSharedQ8ProjectionView allocate(std::uint64_t n) {
        void *p = nullptr;
        check(hipMalloc(&p, n + 128U), "guarded shared projection allocation");
        allocations.push_back(p);
        check(hipMemset(p, 0xa5, n + 128U), "poison shared owner");
        return {reinterpret_cast<std::uintptr_t>(p) + 64U, n};
    }
};
void upload(HipSharedQ8ProjectionView v, const void *data) {
    check(hipMemcpy(reinterpret_cast<void *>(v.address), data, v.bytes, hipMemcpyHostToDevice),
          "upload shared probe");
}
void guard(HipSharedQ8ProjectionView v) {
    std::array<unsigned char, 64> bytes{};
    for (auto addr : {v.address - 64U, v.address + v.bytes}) {
        check(hipMemcpy(bytes.data(), reinterpret_cast<void *>(addr), 64, hipMemcpyDeviceToHost),
              "shared probe guard");
        for (auto b : bytes)
            require(b == 0xa5, "shared probe guard modified");
    }
}
float half(unsigned bits) {
    const unsigned e = (bits >> 10U) & 31U, m = bits & 1023U;
    const float x =
        e == 0U ? std::ldexp(static_cast<float>(m), -24)
                : std::ldexp(1.0F + static_cast<float>(m) / 1024.0F, static_cast<int>(e) - 15);
    return bits & 0x8000U ? -x : x;
}
unsigned short bf16(float x) {
    auto bits = std::bit_cast<unsigned>(x);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<unsigned short>(bits >> 16U);
}
std::vector<unsigned char> oracle(const std::vector<float> &input,
                                  const std::vector<unsigned char> &weights, unsigned outputs,
                                  bool down, bool bf16_output, float shared_gate,
                                  bool wrong_association = false) {
    const auto width = static_cast<unsigned>(input.size());
    const unsigned bytes = bf16_output ? 2U : 4U;
    std::vector<unsigned char> result(static_cast<std::size_t>(outputs) * bytes);
    for (unsigned row = 0; row < outputs; ++row) {
        std::array<float, 256> partial{};
        for (unsigned leaf = 0; leaf < 256; ++leaf)
            for (unsigned feature = leaf; feature < width; feature += 256U) {
                const auto offset =
                    (static_cast<std::uint64_t>(row) * (width / 32U) + feature / 32U) * 34U;
                const float scale =
                    half(weights[offset] | static_cast<unsigned>(weights[offset + 1U]) << 8U);
                const int code = std::bit_cast<signed char>(weights[offset + 2U + feature % 32U]);
                if (wrong_association) {
                    volatile float first = input[feature] * scale;
                    partial[leaf] = std::fma(first, static_cast<float>(code), partial[leaf]);
                } else {
                    volatile float decoded = scale * static_cast<float>(code);
                    partial[leaf] = std::fma(input[feature], decoded, partial[leaf]);
                }
            }
        for (unsigned stride = 128; stride; stride /= 2)
            for (unsigned leaf = 0; leaf < stride; ++leaf) {
                volatile float sum = partial[leaf] + partial[leaf + stride];
                partial[leaf] = sum;
            }
        float value = partial[0];
        if (down) {
            volatile float multiplied = value * shared_gate;
            value = multiplied;
        }
        if (bf16_output) {
            const auto bits = bf16(value);
            std::memcpy(result.data() + row * 2U, &bits, 2);
        } else
            std::memcpy(result.data() + row * 4U, &value, 4);
    }
    return result;
}
void family(unsigned width, unsigned outputs, HipSharedQ8ProjectionRole role,
            HipActivationStorage storage, unsigned flavor) {
    const bool down = role == HipSharedQ8ProjectionRole::down,
               output_bf16 = down && storage == HipActivationStorage::bf16;
    const auto element = static_cast<unsigned>(storage);
    Owner o;
    const auto input = o.allocate(static_cast<std::uint64_t>(width) * (down ? 4U : element)),
               weights = o.allocate(shared_expert_prefetch::required_weight_bytes(width, outputs)),
               gate = down ? o.allocate(4) : HipSharedQ8ProjectionView{},
               output = o.allocate(static_cast<std::uint64_t>(outputs) * (output_bf16 ? 2U : 4U));
    std::vector<float> values(width);
    std::vector<unsigned short> hidden(width);
    std::vector<unsigned char> packed(weights.bytes);
    unsigned seed = width * 65537U + outputs;
    for (auto &value : values) {
        seed = seed * 1664525U + 1013904223U;
        value = std::bit_cast<float>((seed & 0x807fffffU) | 0x3e800000U);
    }
    constexpr std::array<unsigned short, 7> scales{0,       0x8000U, 1,      0x03ffU,
                                                   0x0400U, 0x7bffU, 0xfbffU};
    constexpr std::array<unsigned short, 6> nonfinite_scales{
        0x7c00U, 0xfc00U, 0x7e01U, 0xfe55U, 0x3c00U, 0xbc00U};
    constexpr std::array<signed char, 5> codes{-128, -1, 0, 1, 127};
    constexpr std::array<unsigned short, 8> activation_bits{0,       0x8000U, 1,       0x8001U,
                                                            0x0080U, 0x8080U, 0x3f80U, 0xbf80U};
    for (unsigned row = 0; row < outputs; ++row)
        for (unsigned group = 0; group < width / 32U; ++group) {
            const auto offset = (static_cast<std::uint64_t>(row) * (width / 32U) + group) * 34U;
            seed = seed * 1664525U + 1013904223U;
            const unsigned scale = flavor == 4 ? nonfinite_scales[(row + group) % nonfinite_scales.size()]
                                    : flavor == 1 ? scales[row % scales.size()]
                                               : (flavor == 2 ? 1U : 0x3000U | (seed & 0x83ffU));
            packed[offset] = static_cast<unsigned char>(scale);
            packed[offset + 1] = static_cast<unsigned char>(scale >> 8U);
            for (unsigned j = 0; j < 32; ++j) {
                seed = seed * 1664525U + 1013904223U;
                packed[offset + 2U + j] = flavor == 2 ? 127U
                                          : flavor == 1
                                              ? static_cast<unsigned char>(codes[j % codes.size()])
                                              : static_cast<unsigned char>(seed >> 24U);
            }
        }
    if (flavor == 1)
        for (unsigned i = 0; i < width; ++i)
            values[i] = std::bit_cast<float>(
                static_cast<unsigned>(activation_bits[i % activation_bits.size()]) << 16U);
    if (flavor == 2)
        values.assign(width, std::ldexp(1.0F, -126));
    if (flavor == 4) {
        constexpr std::array<unsigned, 8> payloads{
            0x7fc10000U, 0xffc20000U, 0x7fc30000U, 0xffc40000U,
            0x7f800000U, 0xff800000U, 0x3f800000U, 0xbf800000U};
        for (unsigned i = 0; i < width; ++i)
            values[i] = std::bit_cast<float>(payloads[i % payloads.size()]);
    }
    upload(weights, packed.data());
    const std::array modes{HipSharedQ8ProjectionMode::legacy256,
                           HipSharedQ8ProjectionMode::virtual256,
                           HipSharedQ8ProjectionMode::prefetch8};
    for (auto mode : modes) {
        hipGraph_t graph{};
        check(hipStreamBeginCapture(o.stream, hipStreamCaptureModeThreadLocal),
              "capture shared projection");
        try {
            launch_hip_shared_q8_projection_probe(role, mode, storage, input, weights, gate, output,
                                                  width, outputs, o.stream);
        } catch (...) {
            (void)hipStreamEndCapture(o.stream, &graph);
            if (graph)
                (void)hipGraphDestroy(graph);
            throw;
        }
        check(hipStreamEndCapture(o.stream, &graph), "end shared capture");
        o.graphs.push_back(graph);
        hipGraphExec_t executable{};
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "instantiate shared graph");
        o.executions.push_back(executable);
        std::size_t count = 0;
        check(hipGraphGetNodes(graph, nullptr, &count), "shared graph count");
        require(count == 1, "shared probe graph must have1kernel");
        hipGraphNode_t node{};
        check(hipGraphGetNodes(graph, &node, &count), "shared graph node");
        hipKernelNodeParams params{};
        check(hipGraphKernelNodeGetParams(node, &params), "shared graph parameters");
        require(params.blockDim.x == (mode == HipSharedQ8ProjectionMode::virtual256 ? 32U : 256U) &&
                    params.gridDim.x == (mode == HipSharedQ8ProjectionMode::prefetch8
                                             ? outputs / 8U + (outputs % 8U != 0U)
                                             : outputs),
                "shared projection geometry");
    }
    for (unsigned replay = 0; replay < 4; ++replay) {
        if (replay)
            for (auto &value : values)
                value = -value;
        if (!down && storage == HipActivationStorage::bf16) {
            for (unsigned i = 0; i < width; ++i) {
                hidden[i] = static_cast<unsigned short>(std::bit_cast<unsigned>(values[i]) >> 16U);
                values[i] = std::bit_cast<float>(static_cast<unsigned>(hidden[i]) << 16U);
            }
            upload(input, hidden.data());
        } else
            upload(input, values.data());
        const float shared_gate = flavor == 3 ? (replay == 0   ? 0.0F
                                                 : replay == 1 ? -0.0F
                                                 : replay == 2 ? 0.5F
                                                               : -0.5F)
                                              : 1.0F;
        if (down)
            upload(gate, &shared_gate);
        // Nonfinite payload propagation is compared across device modes only:
        // the host FMA implementation is not an AMD NaN-payload oracle.
        const auto expected = flavor == 4 ? std::vector<unsigned char>{}
            : oracle(values, packed, outputs, down, output_bf16, shared_gate);
        if (flavor == 2)
            require(expected !=
                        oracle(values, packed, outputs, down, output_bf16, shared_gate, true),
                    "decode-association mutant not detected");
        std::vector<unsigned char> first;
        for (unsigned mode = 0; mode < 3; ++mode) {
            check(hipMemsetAsync(reinterpret_cast<void *>(output.address), 0xff, output.bytes,
                                 o.stream),
                  "poison shared output");
            if (replay == 0)
                launch_hip_shared_q8_projection_probe(role, modes[mode], storage, input, weights,
                                                      gate, output, width, outputs, o.stream);
            else
                check(hipGraphLaunch(o.executions[mode], o.stream), "replay shared projection");
            check(hipStreamSynchronize(o.stream), "drain shared projection");
            std::vector<unsigned char> actual(output.bytes);
            check(hipMemcpy(actual.data(), reinterpret_cast<void *>(output.address), output.bytes,
                            hipMemcpyDeviceToHost),
                  "read shared output bits");
            require(flavor == 4 || actual == expected,
                    "shared projection differs from independent fixed256 FMA/store oracle");
            if (mode == 0)
                first = actual;
            else
                require(actual == first, "shared legacy/virtual/prefetch bits differ");
            guard(input);
            guard(weights);
            if (down)
                guard(gate);
            guard(output);
        }
    }
    check(hipMemset(reinterpret_cast<void *>(output.address), 0xff, output.bytes),
          "poison rejected shared output");
    for (unsigned kind = 0; kind < 6; ++kind) {
        auto in = input, w = weights, g = gate, out = output;
        unsigned n = width;
        if (kind == 0)
            --w.bytes;
        if (kind == 1)
            w.address += 2U;
        if (kind == 2)
            out.address = in.address;
        if (kind == 3)
            in.address += 2U;
        if (kind == 4)
            n = 33;
        if (kind == 5) {
            if (down)
                --g.bytes;
            else
                g = {0x1000, 4};
        }
        bool rejected = false;
        try {
            launch_hip_shared_q8_projection_probe(role, HipSharedQ8ProjectionMode::prefetch8,
                                                  storage, in, w, g, out, n, outputs, o.stream);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid shared probe not rejected before enqueue");
    }
    check(hipStreamSynchronize(o.stream), "drain rejected shared");
    std::vector<unsigned char> unchanged(output.bytes);
    check(hipMemcpy(unchanged.data(), reinterpret_cast<void *>(output.address), output.bytes,
                    hipMemcpyDeviceToHost),
          "read unchanged shared output");
    for (auto byte : unchanged)
        require(byte == 0xffU, "invalid shared probe changed semantic destination");
    guard(input);
    guard(weights);
    if (down)
        guard(gate);
    guard(output);
    std::cout << "Shared prefetch width=" << width << " outputs=" << outputs << " down=" << down
              << " storage=" << element << " flavor=" << flavor << " PASS\n";
}
} // namespace
int main() try {
    for (auto storage : {HipActivationStorage::bf16, HipActivationStorage::fp32}) {
        for (auto role : {HipSharedQ8ProjectionRole::gate_up, HipSharedQ8ProjectionRole::down})
            for (auto shape : {std::pair{32U, 1U},
                               {96U, 7U},
                               {288U, 9U},
                               {640U, 33U},
                               {2560U, 65U},
                               {2592U, 31U}})
                family(shape.first, shape.second, role, storage, 0);
        family(2560, 640, HipSharedQ8ProjectionRole::gate_up, storage, 0);
        family(640, 2560, HipSharedQ8ProjectionRole::down, storage, 0);
        for (auto role : {HipSharedQ8ProjectionRole::gate_up, HipSharedQ8ProjectionRole::down}) {
            family(2560, 7, role, storage, 1);
            family(role == HipSharedQ8ProjectionRole::down ? 640U : 2560U, 7, role, storage, 2);
            family(640, 9, role, storage, 4);
        }
        family(640, 9, HipSharedQ8ProjectionRole::down, storage, 3);
    }
    std::cout << "Shared expert prefetch exact diagnostic fixtures PASS\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
