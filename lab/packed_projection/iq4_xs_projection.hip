#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "strix/projection_reduction.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STRIX_WAVE_SIZE
#define STRIX_WAVE_SIZE 32
#endif

namespace {

constexpr unsigned kThreads = 256;
constexpr std::uint32_t kBlockElements = 256;
constexpr std::uint32_t kSubblockElements = 32;
constexpr std::uint32_t kSubblocks = 8;

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__ std::int8_t kCodebookDevice[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

constexpr std::array<std::int8_t, 16> kCodebookHost{
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
    }
}

__global__ void report_wave_size_kernel(std::uint32_t* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = static_cast<std::uint32_t>(warpSize);
    }
}

__host__ __device__ std::uint8_t local_scale_code(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const std::uint8_t low_packed =
        block.scales_l[subblock / 2U];
    const unsigned low_shift =
        4U * (subblock % 2U);
    const std::uint8_t low =
        static_cast<std::uint8_t>(
            (low_packed >> low_shift) & 0x0fU);
    const std::uint8_t high =
        static_cast<std::uint8_t>(
            (block.scales_h >> (2U * subblock)) & 0x03U);
    return static_cast<std::uint8_t>(
        low | static_cast<std::uint8_t>(high << 4U));
}

__device__ float dequantized_weight(
    const IQ4XSBlock& block,
    std::uint32_t within_block) {
    const auto subblock =
        within_block / kSubblockElements;
    const auto within =
        within_block % kSubblockElements;
    const auto packed =
        block.qs[subblock * 16U + (within % 16U)];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(packed & 0x0fU)
            : static_cast<std::uint8_t>(packed >> 4U);
    const int local =
        static_cast<int>(
            local_scale_code(block, subblock)) - 32;
    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(kCodebookDevice[code]);
}

float dequantized_weight_host(
    const IQ4XSBlock& block,
    std::uint32_t within_block) {
    const auto subblock =
        within_block / kSubblockElements;
    const auto within =
        within_block % kSubblockElements;
    const auto packed =
        block.qs[subblock * 16U + (within % 16U)];
    const auto code =
        within < 16U
            ? static_cast<std::uint8_t>(packed & 0x0fU)
            : static_cast<std::uint8_t>(packed >> 4U);
    const int local =
        static_cast<int>(
            local_scale_code(block, subblock)) - 32;
    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(kCodebookHost[code]);
}

template <unsigned Rows, bool ShuffleTail = false>
__global__ void iq4_xs_projection_kernel(
    const float* activations,
    const IQ4XSBlock* weights,
    float* output,
    std::uint32_t input_features,
    std::uint32_t output_features) {
    extern __shared__ float shared[];

    const auto output_index = blockIdx.x;
    if (output_index >= output_features) {
        return;
    }

    const auto blocks_per_row =
        input_features / kBlockElements;
    const auto* weight_row =
        weights +
        static_cast<std::uint64_t>(output_index) *
            blocks_per_row;

    float accumulators[Rows] = {};

    for (std::uint32_t input_index = threadIdx.x;
         input_index < input_features;
         input_index += blockDim.x) {
        const auto block_index =
            input_index / kBlockElements;
        const auto within_block =
            input_index % kBlockElements;
        const float weight =
            dequantized_weight(
                weight_row[block_index],
                within_block);

#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            accumulators[row] = fmaf(
                activations[
                    static_cast<std::uint64_t>(row) *
                        input_features +
                    input_index], weight, accumulators[row]);
        }
    }

#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        shared[
            static_cast<std::size_t>(row) *
                blockDim.x +
            threadIdx.x] =
            accumulators[row];
    }
    __syncthreads();

    for (unsigned offset = blockDim.x / 2U;
         offset >= (ShuffleTail ? 32U : 1U);
         offset /= 2U) {
        if (threadIdx.x < offset) {
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row) {
                const auto base =
                    static_cast<std::size_t>(row) *
                    blockDim.x;
                shared[base + threadIdx.x] +=
                    shared[base + threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if constexpr (ShuffleTail) {
        if (threadIdx.x < 32U) {
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row) {
                float value = shared[static_cast<std::size_t>(row) * blockDim.x + threadIdx.x];
                for (unsigned offset = 16; offset != 0; offset /= 2) {
                    const float next = __shfl_down(value, offset, 32);
                    if (threadIdx.x < offset) value = value + next;
                }
                if (threadIdx.x == 0) output[static_cast<std::uint64_t>(row) * output_features + output_index] = value;
            }
        }
    } else if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            output[
                static_cast<std::uint64_t>(row) *
                    output_features +
                output_index] =
                shared[
                    static_cast<std::size_t>(row) *
                    blockDim.x];
        }
    }
}

template <typename T>
struct DeviceBuffer {
    T* pointer = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        check(
            hipMalloc(
                reinterpret_cast<void**>(&pointer),
                count * sizeof(T)),
            "hipMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            static_cast<void>(hipFree(pointer));
        }
    }
};

template <typename T>
void copy_to_device(
    DeviceBuffer<T>& destination,
    const std::vector<T>& source) {
    check(
        hipMemcpy(
            destination.pointer,
            source.data(),
            source.size() * sizeof(T),
            hipMemcpyHostToDevice),
        "hipMemcpy H2D");
}

template <typename T>
std::vector<T> copy_from_device(
    const DeviceBuffer<T>& source,
    std::size_t count) {
    std::vector<T> output(count);
    check(
        hipMemcpy(
            output.data(),
            source.pointer,
            count * sizeof(T),
            hipMemcpyDeviceToHost),
        "hipMemcpy D2H");
    return output;
}

void set_local_scale(
    IQ4XSBlock& block,
    std::uint32_t subblock,
    std::uint8_t value) {
    const auto low =
        static_cast<std::uint8_t>(value & 0x0fU);
    const auto high =
        static_cast<std::uint16_t>(
            (value >> 4U) & 0x03U);

    auto& low_byte =
        block.scales_l[subblock / 2U];
    if ((subblock % 2U) == 0) {
        low_byte =
            static_cast<std::uint8_t>(
                (low_byte & 0xf0U) | low);
    } else {
        low_byte =
            static_cast<std::uint8_t>(
                (low_byte & 0x0fU) |
                static_cast<std::uint8_t>(low << 4U));
    }

    const auto shift = 2U * subblock;
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h &
            static_cast<std::uint16_t>(
                ~(0x03U << shift)));
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h |
            static_cast<std::uint16_t>(
                high << shift));
}

std::vector<IQ4XSBlock> make_weights(
    std::uint32_t input_features,
    std::uint32_t output_features) {
    const auto blocks_per_row =
        input_features / kBlockElements;
    std::vector<IQ4XSBlock> weights(
        static_cast<std::size_t>(output_features) *
        blocks_per_row);

    for (std::size_t block_index = 0;
         block_index < weights.size();
         ++block_index) {
        auto& block = weights[block_index];
        block.d = __float2half(
            0.00025F *
            static_cast<float>(
                (block_index % 7U) + 1U));
        block.scales_h = 0;
        for (auto& value : block.scales_l) {
            value = 0;
        }

        for (std::uint32_t subblock = 0;
             subblock < kSubblocks;
             ++subblock) {
            const auto scale =
                static_cast<std::uint8_t>(
                    24U +
                    ((block_index + subblock * 3U) % 17U));
            set_local_scale(
                block, subblock, scale);

            for (std::uint32_t byte = 0;
                 byte < 16U;
                 ++byte) {
                const auto low =
                    static_cast<std::uint8_t>(
                        (block_index +
                         subblock +
                         byte * 3U) %
                        16U);
                const auto high =
                    static_cast<std::uint8_t>(
                        (block_index * 5U +
                         subblock * 7U +
                         byte) %
                        16U);
                block.qs[
                    subblock * 16U + byte] =
                    static_cast<std::uint8_t>(
                        low |
                        static_cast<std::uint8_t>(
                            high << 4U));
            }
        }
    }
    return weights;
}

std::vector<float> cpu_reference(
    const std::vector<float>& activations,
    const std::vector<IQ4XSBlock>& weights,
    std::uint32_t rows,
    std::uint32_t input_features,
    std::uint32_t output_features) {
    const auto blocks_per_row =
        input_features / kBlockElements;
    std::vector<float> output(
        static_cast<std::size_t>(rows) *
        output_features);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        for (std::uint32_t out = 0;
             out < output_features;
             ++out) {
            double sum = 0.0;
            const auto* weight_row =
                weights.data() +
                static_cast<std::uint64_t>(out) *
                    blocks_per_row;
            for (std::uint32_t in = 0;
                 in < input_features;
                 ++in) {
                const auto block_index =
                    in / kBlockElements;
                const auto within_block =
                    in % kBlockElements;
                const double weight =
                    static_cast<double>(
                        dequantized_weight_host(
                            weight_row[block_index],
                            within_block));
                sum +=
                    static_cast<double>(
                        activations[
                            static_cast<std::uint64_t>(row) *
                                input_features +
                            in]) *
                    weight;
            }
            output[
                static_cast<std::uint64_t>(row) *
                    output_features +
                out] =
                static_cast<float>(sum);
        }
    }
    return output;
}

void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const char* name) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            std::string(name) + ": size mismatch");
    }
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            throw std::runtime_error(std::string(name) + ": nonfinite output");
        }
        const float difference =
            std::fabs(actual[index] - expected[index]);
        const float scale =
            std::max(
                1.0F,
                std::fabs(expected[index]));
        if (difference > tolerance * scale) {
            throw std::runtime_error(
                std::string(name) +
                ": mismatch at index " +
                std::to_string(index));
        }
    }
}

template <unsigned Rows, bool ShuffleTail = false>
void launch_projection(
    const DeviceBuffer<float>& activations,
    const DeviceBuffer<IQ4XSBlock>& weights,
    DeviceBuffer<float>& output,
    std::uint32_t input_features,
    std::uint32_t output_features,
    hipStream_t stream = nullptr) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            iq4_xs_projection_kernel<Rows, ShuffleTail>),
        dim3(output_features),
        dim3(kThreads),
        Rows * kThreads * sizeof(float),
        stream,
        activations.pointer,
        weights.pointer,
        output.pointer,
        input_features,
        output_features);
}

template <unsigned Rows, bool ShuffleTail>
class ProjectionGraph {
public:
    ProjectionGraph(const DeviceBuffer<float>& input, const DeviceBuffer<IQ4XSBlock>& weights,
                    DeviceBuffer<float>& output, std::uint32_t k, std::uint32_t n) {
        try {
            check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "graph stream create");
            check(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal), "graph capture begin");
            capturing_ = true;
            launch_projection<Rows, ShuffleTail>(input, weights, output, k, n, stream_);
            check(hipGetLastError(), "graph projection capture launch");
            const auto end = hipStreamEndCapture(stream_, &graph_);
            capturing_ = false;
            check(end, "graph capture end");
            check(hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0), "graph instantiate");
        } catch (...) { cleanup(); throw; }
    }
    ProjectionGraph(const ProjectionGraph&) = delete;
    ProjectionGraph& operator=(const ProjectionGraph&) = delete;
    ~ProjectionGraph() { cleanup(); }
    void launch() { check(hipGraphLaunch(exec_, stream_), "projection graph replay"); }
    void synchronize() { check(hipStreamSynchronize(stream_), "projection graph completion"); }
    hipStream_t stream() const { return stream_; }
private:
    void cleanup() noexcept {
        if (capturing_) {
            hipGraph_t discarded{};
            static_cast<void>(hipStreamEndCapture(stream_, &discarded));
            if (discarded) static_cast<void>(hipGraphDestroy(discarded));
        }
        if (stream_) static_cast<void>(hipStreamSynchronize(stream_));
        if (exec_) static_cast<void>(hipGraphExecDestroy(exec_));
        if (graph_) static_cast<void>(hipGraphDestroy(graph_));
        if (stream_) static_cast<void>(hipStreamDestroy(stream_));
        exec_ = nullptr; graph_ = nullptr; stream_ = nullptr; capturing_ = false;
    }
    hipStream_t stream_{};
    hipGraph_t graph_{};
    hipGraphExec_t exec_{};
    bool capturing_ = false;
};

template <unsigned Rows>
void correctness_fixture() {
    constexpr std::uint32_t input_features = 512;
    constexpr std::uint32_t output_features = 9;

    std::vector<float> activations(
        static_cast<std::size_t>(Rows) *
        input_features);
    for (std::size_t index = 0;
         index < activations.size();
         ++index) {
        activations[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 23U) -
                11) /
            17.0F;
    }

    const auto weights =
        make_weights(
            input_features,
            output_features);

    DeviceBuffer<float> d_activations(
        activations.size());
    DeviceBuffer<IQ4XSBlock> d_weights(
        weights.size());
    DeviceBuffer<float> d_output(
        static_cast<std::size_t>(Rows) *
        output_features);

    copy_to_device(
        d_activations, activations);
    copy_to_device(
        d_weights, weights);

    launch_projection<Rows>(
        d_activations,
        d_weights,
        d_output,
        input_features,
        output_features);
    check(
        hipGetLastError(),
        "IQ4_XS correctness launch");
    check(
        hipDeviceSynchronize(),
        "IQ4_XS correctness sync");

    const auto actual =
        copy_from_device(
            d_output,
            static_cast<std::size_t>(Rows) *
            output_features);
    const auto expected =
        cpu_reference(
            activations,
            weights,
            Rows,
            input_features,
            output_features);
    expect_close(
        actual,
        expected,
        5e-5F,
        "IQ4_XS projection");
}

void row_invariance_fixture() {
    constexpr std::uint32_t input_features = 512;
    constexpr std::uint32_t output_features = 7;

    std::vector<float> batch(
        2U * input_features);
    for (std::size_t index = 0;
         index < batch.size();
         ++index) {
        batch[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 29U) -
                14) /
            19.0F;
    }
    const std::vector<float> solo(
        batch.begin() + input_features,
        batch.end());
    const auto weights =
        make_weights(
            input_features,
            output_features);

    DeviceBuffer<float> d_batch(batch.size());
    DeviceBuffer<float> d_solo(solo.size());
    DeviceBuffer<IQ4XSBlock> d_weights(
        weights.size());
    DeviceBuffer<float> d_batch_output(
        2U * output_features);
    DeviceBuffer<float> d_solo_output(
        output_features);

    copy_to_device(d_batch, batch);
    copy_to_device(d_solo, solo);
    copy_to_device(d_weights, weights);

    launch_projection<2>(
        d_batch,
        d_weights,
        d_batch_output,
        input_features,
        output_features);
    launch_projection<1>(
        d_solo,
        d_weights,
        d_solo_output,
        input_features,
        output_features);
    check(
        hipGetLastError(),
        "IQ4_XS row invariance launch");
    check(
        hipDeviceSynchronize(),
        "IQ4_XS row invariance sync");

    const auto batch_output =
        copy_from_device(
            d_batch_output,
            2U * output_features);
    const auto solo_output =
        copy_from_device(
            d_solo_output,
            output_features);
    const std::vector<float> second(
        batch_output.begin() + output_features,
        batch_output.end());

    expect_close(
        second,
        solo_output,
        1e-7F,
        "IQ4_XS row invariance");
}

template <unsigned Rows>
void exact_tree_fixture(std::uint32_t input_features) {
    constexpr std::uint32_t outputs = 7;
    constexpr unsigned guard = 5;
    const auto weights = make_weights(input_features, outputs);
    std::vector<float> input(static_cast<std::size_t>(Rows) * input_features);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int code = static_cast<int>((i * 17U + i / input_features) % 61U) - 30;
        input[i] = std::ldexp(static_cast<float>(code), static_cast<int>(i % 13U) - 6);
    }
    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<IQ4XSBlock> d_weights(weights.size());
    DeviceBuffer<float> d_baseline(static_cast<std::size_t>(Rows) * outputs + guard);
    DeviceBuffer<float> d_candidate(static_cast<std::size_t>(Rows) * outputs + guard);
    std::vector<float> poison(static_cast<std::size_t>(Rows) * outputs + guard,
                              std::bit_cast<float>(0x7fc12345U));
    copy_to_device(d_input, input);
    copy_to_device(d_weights, weights);
    copy_to_device(d_baseline, poison);
    copy_to_device(d_candidate, poison);
    launch_projection<Rows, false>(d_input, d_weights, d_baseline, input_features, outputs);
    launch_projection<Rows, true>(d_input, d_weights, d_candidate, input_features, outputs);
    check(hipGetLastError(), "exact reduction fixture launch");
    check(hipDeviceSynchronize(), "exact reduction fixture sync");
    const auto baseline = copy_from_device(d_baseline, poison.size());
    const auto candidate = copy_from_device(d_candidate, poison.size());
    // Capture only this fixed-shape projection. Input values remain device data;
    // whole-model token/position scalar launchers are not made graph-safe here.
    const auto original_input = input;
    ProjectionGraph<Rows, true> graph(d_input, d_weights, d_candidate, input_features, outputs);
    for (float& value : input) value = -value;
    copy_to_device(d_input, input);
    graph.launch();
    graph.synchronize();
    const auto replay = copy_from_device(d_candidate, poison.size());
    const auto blocks = input_features / kBlockElements;
    std::vector<float> decoded(input_features);
    bool changed_output = false;
    for (std::uint32_t out = 0; out < outputs; ++out) {
        for (std::uint32_t k = 0; k < input_features; ++k) {
            decoded[k] = dequantized_weight_host(weights[out * blocks + k / kBlockElements], k % kBlockElements);
        }
        for (unsigned row = 0; row < Rows; ++row) {
            const float expected = strix::projection_dot_reference(decoded,
                std::span<const float>(original_input).subspan(static_cast<std::size_t>(row) * input_features, input_features));
            const float expected_replay = strix::projection_dot_reference(decoded,
                std::span<const float>(input).subspan(static_cast<std::size_t>(row) * input_features, input_features));
            const auto index = row * outputs + out;
            changed_output = changed_output || expected != expected_replay;
            if (std::bit_cast<std::uint32_t>(baseline[index]) != std::bit_cast<std::uint32_t>(expected) ||
                std::bit_cast<std::uint32_t>(candidate[index]) != std::bit_cast<std::uint32_t>(expected) ||
                std::bit_cast<std::uint32_t>(replay[index]) != std::bit_cast<std::uint32_t>(expected_replay)) {
                throw std::runtime_error("projection exact arithmetic gate failed");
            }
        }
    }
    if (!changed_output) throw std::runtime_error("graph input-change fixture lacks a sensitive output");
    for (std::size_t i = static_cast<std::size_t>(Rows) * outputs; i < poison.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(baseline[i]) != 0x7fc12345U ||
            std::bit_cast<std::uint32_t>(candidate[i]) != 0x7fc12345U ||
            std::bit_cast<std::uint32_t>(replay[i]) != 0x7fc12345U) {
            throw std::runtime_error("projection wrote outside active output rows");
        }
    }
}

struct ScopedHipEvent {
    hipEvent_t value{};
    ScopedHipEvent() { check(hipEventCreate(&value), "benchmark event create"); }
    ScopedHipEvent(const ScopedHipEvent&) = delete;
    ScopedHipEvent& operator=(const ScopedHipEvent&) = delete;
    ~ScopedHipEvent() { if (value) static_cast<void>(hipEventDestroy(value)); }
};

template <typename Launch>
double benchmark_us(
    Launch&& launch,
    unsigned warmup = 10,
    unsigned iterations = 100,
    hipStream_t stream = nullptr) {
    for (unsigned index = 0;
         index < warmup;
         ++index) {
        launch();
    }
    check(
        hipGetLastError(),
        "IQ4_XS benchmark warmup launch");
    check(
        hipDeviceSynchronize(),
        "IQ4_XS benchmark warmup sync");

    ScopedHipEvent start;
    ScopedHipEvent stop;
    check(
        hipEventRecord(start.value, stream),
        "hipEventRecord start");

    for (unsigned index = 0;
         index < iterations;
         ++index) {
        launch();
    }

    check(
        hipGetLastError(),
        "IQ4_XS benchmark measured launch");
    check(
        hipEventRecord(stop.value, stream),
        "hipEventRecord stop");
    check(
        hipEventSynchronize(stop.value),
        "hipEventSynchronize stop");

    float milliseconds = 0.0F;
    check(
        hipEventElapsedTime(
            &milliseconds,
            start.value,
            stop.value),
        "hipEventElapsedTime");
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0F) {
        throw std::runtime_error("invalid measured projection event interval");
    }

    return
        static_cast<double>(milliseconds) *
        1000.0 /
        static_cast<double>(iterations);
}

struct KernelAttributes {
    int registers = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
};

template <unsigned Rows, bool ShuffleTail = false>
KernelAttributes attributes() {
    hipFuncAttributes value{};
    check(
        hipFuncGetAttributes(
            &value,
            reinterpret_cast<const void*>(
                iq4_xs_projection_kernel<Rows, ShuffleTail>)),
        "hipFuncGetAttributes IQ4_XS");
    return KernelAttributes{
        value.numRegs,
        value.sharedSizeBytes,
        value.localSizeBytes,
    };
}

struct Measurement {
    bool shuffle_tail = false;
    bool graph_replay = false;
    double capture_us = 0.0;
    unsigned rows = 0;
    double microseconds = 0.0;
    std::uint64_t weight_bytes = 0;
    double physical_weight_gib_s = 0.0;
    double serial_equivalent_weight_gib_s = 0.0;
    KernelAttributes resources;
};

template <unsigned Rows, bool ShuffleTail = false, bool GraphReplay = false>
Measurement benchmark_row_class() {
    constexpr std::uint32_t input_features = 2560;
    constexpr std::uint32_t output_features = 2560;
    constexpr auto blocks_per_row =
        input_features / kBlockElements;
    constexpr std::uint64_t weight_bytes =
        static_cast<std::uint64_t>(
            output_features) *
        blocks_per_row *
        sizeof(IQ4XSBlock);

    std::vector<float> activations(
        static_cast<std::size_t>(Rows) *
        input_features,
        0.125F);
    const auto weights =
        make_weights(
            input_features,
            output_features);

    DeviceBuffer<float> d_activations(
        activations.size());
    DeviceBuffer<IQ4XSBlock> d_weights(
        weights.size());
    DeviceBuffer<float> d_output(
        static_cast<std::size_t>(Rows) *
        output_features);

    copy_to_device(
        d_activations, activations);
    copy_to_device(
        d_weights, weights);

    std::unique_ptr<ProjectionGraph<Rows, ShuffleTail>> graph;
    const auto capture_start = std::chrono::steady_clock::now();
    if constexpr (GraphReplay) {
        graph = std::make_unique<ProjectionGraph<Rows, ShuffleTail>>(
            d_activations, d_weights, d_output, input_features, output_features);
    }
    const double capture_us = GraphReplay ? std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - capture_start).count() : 0.0;
    const auto microseconds =
        benchmark_us([&] {
            if constexpr (GraphReplay) { graph->launch(); }
            else launch_projection<Rows, ShuffleTail>(
                d_activations,
                d_weights,
                d_output,
                input_features,
                output_features);
        }, 10, 100, GraphReplay ? graph->stream() : nullptr);

    const double seconds =
        microseconds * 1e-6;
    const double gib =
        static_cast<double>(weight_bytes) /
        (1024.0 * 1024.0 * 1024.0);

    Measurement result;
    result.shuffle_tail = ShuffleTail;
    result.graph_replay = GraphReplay;
    result.capture_us = capture_us;
    result.rows = Rows;
    result.microseconds = microseconds;
    result.weight_bytes = weight_bytes;
    result.physical_weight_gib_s =
        seconds > 0.0
            ? gib / seconds
            : 0.0;
    result.serial_equivalent_weight_gib_s =
        seconds > 0.0
            ? gib *
                  static_cast<double>(Rows) /
                  seconds
            : 0.0;
    result.resources = attributes<Rows, ShuffleTail>();
    return result;
}

void print_measurement(
    const Measurement& value,
    bool last) {
    std::cout
        << "    {\"rows\":" << value.rows
        << ",\"reduction\":\"" << (value.shuffle_tail ? "shuffle_tail" : "lds_tree") << '"'
        << ",\"graph_replay\":" << (value.graph_replay ? "true" : "false")
        << ",\"capture_us\":" << value.capture_us
        << ",\"graph_memory_bytes\":null"
        << ",\"microseconds\":"
        << value.microseconds
        << ",\"weight_bytes\":"
        << value.weight_bytes
        << ",\"physical_weight_gib_s\":"
        << value.physical_weight_gib_s
        << ",\"serial_equivalent_weight_gib_s\":"
        << value.serial_equivalent_weight_gib_s
        << ",\"registers\":"
        << value.resources.registers
        << ",\"static_shared_bytes\":"
        << value.resources.static_shared_bytes
        << ",\"dynamic_shared_bytes\":"
        << value.rows *
               kThreads *
               sizeof(float)
        << ",\"local_bytes\":"
        << value.resources.local_bytes
        << '}'
        << (last ? '\n' : ',')
        << '\n';
}

}  // namespace

int main() try {
    check(
        hipSetDevice(0),
        "hipSetDevice");

    hipDeviceProp_t properties{};
    check(
        hipGetDeviceProperties(
            &properties, 0),
        "hipGetDeviceProperties");

    const std::string architecture =
        properties.gcnArchName;
    if (architecture.rfind(
            "gfx1151", 0) != 0) {
        throw std::runtime_error(
            "IQ4_XS projection lab requires gfx1151; got " +
            architecture);
    }

    DeviceBuffer<std::uint32_t> d_wave(1);
    hipLaunchKernelGGL(
        report_wave_size_kernel,
        dim3(1),
        dim3(1),
        0,
        0,
        d_wave.pointer);
    check(
        hipGetLastError(),
        "wave report launch");
    check(
        hipDeviceSynchronize(),
        "wave report sync");

    const auto runtime_wave =
        copy_from_device(d_wave, 1);
    if (runtime_wave[0] != STRIX_WAVE_SIZE) {
        throw std::runtime_error(
            "compiled wave-size mismatch");
    }

    correctness_fixture<1>();
    correctness_fixture<2>();
    correctness_fixture<4>();
    correctness_fixture<8>();
    row_invariance_fixture();
    for (std::uint32_t k : {256U, 512U, 2560U, 4096U}) {
        exact_tree_fixture<1>(k);
        exact_tree_fixture<2>(k);
        exact_tree_fixture<4>(k);
        exact_tree_fixture<8>(k);
    }

    const auto c1 =
        benchmark_row_class<1>();
    const auto r2 =
        benchmark_row_class<2>();
    const auto r4 =
        benchmark_row_class<4>();
    const auto r8 =
        benchmark_row_class<8>();

    std::cout
        << "{\n"
        << "  \"status\":\"pass\",\n"
        << "  \"format\":\"IQ4_XS\",\n"
        << "  \"gpu_arch\":\""
        << architecture
        << "\",\n"
        << "  \"wave_size\":"
        << STRIX_WAVE_SIZE
        << ",\n"
        << "  \"row_invariance_fixture\":true,\n"
        << "  \"exact_tree_fixture\":true,\n"
        << "  \"arithmetic\":\"iq4_xs-f32-fma256-tree-v1\",\n"
        << "  \"qualified\":false,\n"
        << "  \"measurements\":[\n";

    print_measurement(c1, false);
    print_measurement(r2, false);
    print_measurement(r4, false);
    print_measurement(r8, false);
    print_measurement(benchmark_row_class<1, true>(), false);
    print_measurement(benchmark_row_class<2, true>(), false);
    print_measurement(benchmark_row_class<4, true>(), false);
    print_measurement(benchmark_row_class<8, true>(), false);
    print_measurement(benchmark_row_class<1, true, true>(), false);
    print_measurement(benchmark_row_class<2, true, true>(), false);
    print_measurement(benchmark_row_class<4, true, true>(), false);
    print_measurement(benchmark_row_class<8, true, true>(), true);

    std::cout
        << "  ]\n"
        << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr
        << "strix-iq4-xs-projection: "
        << error.what()
        << '\n';
    return 1;
}
