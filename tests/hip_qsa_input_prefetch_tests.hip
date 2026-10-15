#include "strix/hip_indexed_attention_launcher.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
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
    Owner() { check(hipStreamCreate(&stream), "create projection stream"); }
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
    HipQsaInputProjectionView allocate(std::uint64_t bytes) {
        void *p = nullptr;
        check(hipMalloc(&p, bytes + 128U), "allocate guarded projection");
        allocations.push_back(p);
        check(hipMemset(p, 0xa5, bytes + 128U), "poison guarded owner");
        return {reinterpret_cast<std::uintptr_t>(p) + 64U, bytes};
    }
};
void upload(HipQsaInputProjectionView v, const void *p) {
    check(hipMemcpy(reinterpret_cast<void *>(v.address), p, v.bytes, hipMemcpyHostToDevice),
          "upload exact view");
}
void guard(HipQsaInputProjectionView v) {
    std::array<unsigned char, 64> bytes{};
    for (auto address : {v.address - 64U, v.address + v.bytes}) {
        check(hipMemcpy(bytes.data(), reinterpret_cast<void *>(address), 64, hipMemcpyDeviceToHost),
              "read projection guard");
        for (auto b : bytes)
            require(b == 0xa5, "projection owner guard modified");
    }
}
float half(unsigned bits) {
    const unsigned e = (bits >> 10U) & 31U, m = bits & 1023U;
    const float x =
        e == 0U ? std::ldexp(static_cast<float>(m), -24)
                : std::ldexp(1.0F + static_cast<float>(m) / 1024.0F, static_cast<int>(e) - 15);
    return bits & 0x8000U ? -x : x;
}
unsigned round_bf16(float x) {
    auto bits = std::bit_cast<unsigned>(x);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return bits & 0xffff0000U;
}
std::vector<unsigned> oracle(const std::vector<std::uint16_t> &input,
                             const std::vector<unsigned char> &weights, unsigned outputs,
                             bool wrong_association = false) {
    const auto width = static_cast<unsigned>(input.size());
    std::vector<unsigned> result(outputs);
    for (unsigned row = 0; row < outputs; ++row) {
        std::array<float, 256> partial{};
        for (unsigned leaf = 0; leaf < 256; ++leaf)
            for (unsigned feature = leaf; feature < width; feature += 256U) {
                const auto offset =
                    (static_cast<std::uint64_t>(row) * (width / 32U) + feature / 32U) * 34U;
                const float scale =
                    half(weights[offset] | static_cast<unsigned>(weights[offset + 1U]) << 8U);
                const int code = std::bit_cast<signed char>(weights[offset + 2U + feature % 32U]);
                const float activation =
                    std::bit_cast<float>(static_cast<unsigned>(input[feature]) << 16U);
                if (wrong_association) {
                    volatile float first = activation * scale;
                    partial[leaf] = std::fma(first, static_cast<float>(code), partial[leaf]);
                } else {
                    volatile float decoded = scale * static_cast<float>(code);
                    partial[leaf] = std::fma(activation, decoded, partial[leaf]);
                }
            }
        for (unsigned stride = 128; stride; stride /= 2)
            for (unsigned leaf = 0; leaf < stride; ++leaf) {
                volatile float sum = partial[leaf] + partial[leaf + stride];
                partial[leaf] = sum;
            }
        result[row] = round_bf16(partial[0]);
    }
    return result;
}
void mismatch(const std::vector<unsigned>& actual, const std::vector<unsigned>& expected,
              unsigned width, unsigned outputs, unsigned flavor, unsigned replay,
              unsigned mode, const char* reference) {
    std::cerr << "QSA_BITS_MISMATCH width=" << width << " outputs=" << outputs
              << " input=BF16 output=FP32-rounded-BF16 flavor=" << flavor
              << " replay=" << replay << " path=" << (replay ? "graph" : "eager")
              << " mode=" << mode << " reference=" << reference << '\n';
    unsigned count=0;
    for(unsigned row=0;row<outputs;++row)if(actual[row]!=expected[row]){
        if(count<16U)std::cerr << " output=" << row << " old=0x" << std::hex
                              << expected[row] << " new=0x" << actual[row] << std::dec << '\n';
        ++count;
    }
    std::cerr << " differing_output_words=" << count << '\n';
}
void family(unsigned width, unsigned outputs, unsigned flavor) {
    Owner owner;
    const auto bytes = qsa_input_prefetch::required_weight_bytes(width, outputs);
    const auto input = owner.allocate(width * 2ULL), weights = owner.allocate(bytes),
               output = owner.allocate(outputs * 4ULL);
    std::vector<std::uint16_t> hidden(width);
    std::vector<unsigned char> packed(bytes);
    unsigned seed = width * 65537U + outputs;
    for (auto &value : hidden) {
        seed = 1664525U * seed + 1013904223U;
        value = static_cast<unsigned short>(0x3e00U | (seed & 0x81ffU));
    }
    constexpr std::array<unsigned short, 7> scales{0x0000U, 0x8000U, 0x0001U, 0x03ffU,
                                                   0x0400U, 0x7bffU, 0xfbffU};
    constexpr std::array<unsigned short, 8> nonfinite_scales{
        0x7e11U,0xfe29U,0x7f31U,0xff47U,0x7c00U,0xfc00U,0x3c00U,0xbc00U};
    constexpr std::array<signed char, 5> codes{-128, -1, 0, 1, 127};
    constexpr std::array<unsigned short, 8> activations{0,       0x8000U, 1,       0x8001U,
                                                        0x0080U, 0x8080U, 0x3f80U, 0xbf80U};
    for (unsigned row = 0; row < outputs; ++row)
        for (unsigned group = 0; group < width / 32U; ++group) {
            const auto offset = (static_cast<std::uint64_t>(row) * (width / 32U) + group) * 34U;
            seed = 1664525U * seed + 1013904223U;
            const unsigned scale = flavor == 5 ? 0x3c00U
                : flavor == 4 ? nonfinite_scales[(row+group)%nonfinite_scales.size()]
                : flavor == 1 ? scales[row % scales.size()]
                                               : (flavor == 2 ? 1U : 0x3000U | (seed & 0x83ffU));
            packed[offset] = static_cast<unsigned char>(scale);
            packed[offset + 1U] = static_cast<unsigned char>(scale >> 8U);
            for (unsigned j = 0; j < 32; ++j) {
                seed = 1664525U * seed + 1013904223U;
                packed[offset + 2U + j] = flavor == 5 ? 1U : flavor == 2 ? 127U
                                          : flavor == 1
                                              ? static_cast<unsigned char>(codes[j % codes.size()])
                                              : static_cast<unsigned char>(seed >> 24U);
            }
        }
    if (flavor == 1)
        for (unsigned i = 0; i < width; ++i)
            hidden[i] = activations[i % activations.size()];
    if (flavor == 2)
        hidden.assign(width, 0x0080U);
    if(flavor==4){
        constexpr std::array<std::uint16_t,8> payloads{
            0x7fc1U,0xffc2U,0x7fc3U,0xffc4U,0x7f80U,0xff80U,0x3f80U,0xbf80U};
        for(unsigned i=0;i<width;++i)hidden[i]=payloads[i%payloads.size()];
    }
    if(flavor==5){
        for(unsigned i=0;i<width;++i){
            const unsigned id=(5U*i+(i>>7U))&127U;
            hidden[i]=static_cast<std::uint16_t>(((id&64U)<<9U)|0x7fc0U|(id&63U));
        }
        for(unsigned stride=128;stride;stride/=2U)
            for(unsigned leaf=0;leaf<stride;++leaf)
                require(hidden[leaf]!=hidden[leaf+stride],
                        "NaN fixture must distinguish every reduction-stride pair");
    }
    upload(weights, packed.data());
    const std::array modes{HipQsaInputProjectionMode::legacy256,
                           HipQsaInputProjectionMode::virtual256,
                           HipQsaInputProjectionMode::prefetch8};
    for (auto mode : modes) {
        hipGraph_t graph{};
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal),
              "capture projection");
        try {
            launch_hip_qsa_input_projection_probe(mode, input, weights, output, width, outputs,
                                                  owner.stream);
        } catch (...) {
            (void)hipStreamEndCapture(owner.stream, &graph);
            if (graph)
                (void)hipGraphDestroy(graph);
            throw;
        }
        check(hipStreamEndCapture(owner.stream, &graph), "end projection capture");
        owner.graphs.push_back(graph);
        hipGraphExec_t execution{};
        check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0),
              "instantiate projection");
        owner.executions.push_back(execution);
        std::size_t count = 0;
        check(hipGraphGetNodes(graph, nullptr, &count), "count projection graph");
        require(count == 1, "projection graph must contain1kernel");
        hipGraphNode_t node{};
        check(hipGraphGetNodes(graph, &node, &count), "get projection node");
        hipKernelNodeParams params{};
        check(hipGraphKernelNodeGetParams(node, &params), "projection parameters");
        require(params.blockDim.x == (mode == HipQsaInputProjectionMode::legacy256 ? 256U : 32U) &&
                    params.gridDim.x == outputs,
                "projection launch geometry differs");
    }
    for (unsigned replay = 0; replay < 4; ++replay) {
        if (replay)
            for (auto &value : hidden)
                value ^= 0x8000U;
        upload(input, hidden.data());
        // CPU NaN operand priority is not a device reference. Nonfinite cases
        // compare every output word against the same device legacy kernel.
        const auto expected = flavor>=4 ? std::vector<unsigned>{} : oracle(hidden, packed, outputs);
        if (flavor == 2) {
            require(expected[0] == (replay % 2U ? 0x80020000U : 0x00020000U),
                    "decode-first control bits must survive BF16");
            require(expected != oracle(hidden, packed, outputs, true),
                    "association mutant must be detected");
        }
        std::vector<unsigned> first;
        for (unsigned mode = 0; mode < 3; ++mode) {
            check(hipMemsetAsync(reinterpret_cast<void *>(output.address), 0xff, output.bytes,
                                 owner.stream),
                  "poison entire semantic output");
            if (replay == 0)
                launch_hip_qsa_input_projection_probe(modes[mode], input, weights, output, width,
                                                      outputs, owner.stream);
            else
                check(hipGraphLaunch(owner.executions[mode], owner.stream), "replay projection");
            check(hipStreamSynchronize(owner.stream), "drain projection");
            std::vector<unsigned> actual(outputs);
            check(hipMemcpy(actual.data(), reinterpret_cast<void *>(output.address), output.bytes,
                            hipMemcpyDeviceToHost),
                  "read projection bits");
            if(flavor<4 && actual!=expected){
                mismatch(actual,expected,width,outputs,flavor,replay,mode,"host-fixed256");
                throw std::runtime_error("projection differs from independent fixed256 FMA/BF16 oracle");
            }
            if (mode == 0)
                first = actual;
            else if(actual!=first){
                mismatch(actual,first,width,outputs,flavor,replay,mode,"device-legacy256");
                throw std::runtime_error("legacy/virtual/prefetch bits differ");
            }
            guard(input);
            guard(weights);
            guard(output);
        }
    }
    // Invalid calls must reject before enqueue, preserving every poisoned output.
    check(hipMemset(reinterpret_cast<void *>(output.address), 0xff, output.bytes),
          "poison rejection destination");
    for (unsigned kind = 0; kind < 5; ++kind) {
        auto bad_input = input, bad_weights = weights, bad_output = output;
        unsigned w = width;
        if (kind == 0)
            --bad_weights.bytes;
        if (kind == 1)
            bad_weights.address += 2U;
        if (kind == 2)
            bad_output.address = bad_input.address;
        if (kind == 3)
            bad_input.address += 2U;
        if (kind == 4)
            w = 33;
        bool rejected = false;
        try {
            launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode::prefetch8, bad_input,
                                                  bad_weights, bad_output, w, outputs,
                                                  owner.stream);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid projection was not rejected");
    }
    check(hipStreamSynchronize(owner.stream), "drain rejected projection");
    std::vector<unsigned> unchanged(outputs);
    check(hipMemcpy(unchanged.data(), reinterpret_cast<void *>(output.address), output.bytes,
                    hipMemcpyDeviceToHost),
          "read rejected projection");
    for (auto value : unchanged)
        require(value == 0xffffffffU, "invalid projection changed destination");
    guard(input);
    guard(weights);
    guard(output);
    std::cout << "QSA prefetch fixed256 family width=" << width << " outputs=" << outputs
              << " flavor=" << flavor << " PASS\n";
}
} // namespace
int main(int argc,char** argv) try {
    if(argc>1){
        require(argc==2 && std::string(argv[1])=="--nonfinite","expected no args or --nonfinite");
        for(unsigned flavor:{4U,5U})family(640,9,flavor);
        std::cout << "QSA input nonfinite exact diagnostic fixtures PASS\n";
        return 0;
    }
    for (auto shape : {std::pair{32U, 1U},
                       {64U, 7U},
                       {96U, 31U},
                       {288U, 33U},
                       {640U, 255U},
                       {2560U, 257U},
                       {2560U, 12288U},
                       {2592U, 33U}})
        family(shape.first, shape.second, 0);
    family(2560, 7, 1);
    family(2560, 7, 2);
    for(unsigned flavor:{4U,5U})family(640,9,flavor);
    std::cout << "QSA input prefetch exact diagnostic fixtures PASS\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
