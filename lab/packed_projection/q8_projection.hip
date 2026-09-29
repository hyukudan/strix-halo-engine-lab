#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STRIX_WAVE_SIZE
#define STRIX_WAVE_SIZE 32
#endif

namespace {

constexpr unsigned kThreads = 256;
constexpr std::uint32_t kBlockElements = 32;

struct Q8Block {
    __half scale;
    std::int8_t values[kBlockElements];
};

static_assert(sizeof(Q8Block) == 34);

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

template <unsigned Rows>
__global__ void q8_projection_kernel(const float* activations,
                                     const Q8Block* weights,
                                     float* output,
                                     std::uint32_t input_features,
                                     std::uint32_t output_features) {
    extern __shared__ float shared[];
    const auto output_index = blockIdx.x;
    if (output_index >= output_features) {
        return;
    }

    const auto blocks_per_row = input_features / kBlockElements;
    const auto* weight_row =
        weights + static_cast<std::uint64_t>(output_index) * blocks_per_row;

    float accumulators[Rows] = {};

    for (std::uint32_t input_index = threadIdx.x;
         input_index < input_features;
         input_index += blockDim.x) {
        const auto block_index = input_index / kBlockElements;
        const auto block_offset = input_index % kBlockElements;
        const auto& block = weight_row[block_index];
        const float weight =
            __half2float(block.scale) *
            static_cast<float>(block.values[block_offset]);

#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            accumulators[row] +=
                activations[static_cast<std::uint64_t>(row) * input_features +
                            input_index] *
                weight;
        }
    }

#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        shared[static_cast<std::size_t>(row) * blockDim.x + threadIdx.x] =
            accumulators[row];
    }
    __syncthreads();

    for (unsigned offset = blockDim.x / 2U;
         offset > 0;
         offset /= 2U) {
        if (threadIdx.x < offset) {
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row) {
                const auto base =
                    static_cast<std::size_t>(row) * blockDim.x;
                shared[base + threadIdx.x] +=
                    shared[base + threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            output[static_cast<std::uint64_t>(row) * output_features +
                   output_index] =
                shared[static_cast<std::size_t>(row) * blockDim.x];
        }
    }
}

template <typename T>
struct DeviceBuffer {
    T* pointer = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        check(hipMalloc(&pointer, count * sizeof(T)), "hipMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            hipFree(pointer);
        }
    }
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination,
                    const std::vector<T>& source) {
    check(hipMemcpy(destination.pointer,
                    source.data(),
                    source.size() * sizeof(T),
                    hipMemcpyHostToDevice),
          "hipMemcpy H2D");
}

template <typename T>
std::vector<T> copy_from_device(const DeviceBuffer<T>& source,
                                std::size_t count) {
    std::vector<T> output(count);
    check(hipMemcpy(output.data(),
                    source.pointer,
                    count * sizeof(T),
                    hipMemcpyDeviceToHost),
          "hipMemcpy D2H");
    return output;
}

std::vector<float> cpu_reference(const std::vector<float>& activations,
                                 const std::vector<Q8Block>& weights,
                                 std::uint32_t rows,
                                 std::uint32_t input_features,
                                 std::uint32_t output_features) {
    const auto blocks_per_row = input_features / kBlockElements;
    std::vector<float> output(
        static_cast<std::size_t>(rows) * output_features);

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t out = 0; out < output_features; ++out) {
            double sum = 0.0;
            const auto* weight_row =
                weights.data() +
                static_cast<std::uint64_t>(out) * blocks_per_row;
            for (std::uint32_t in = 0; in < input_features; ++in) {
                const auto block_index = in / kBlockElements;
                const auto block_offset = in % kBlockElements;
                const auto& block = weight_row[block_index];
                const double weight =
                    static_cast<double>(__half2float(block.scale)) *
                    static_cast<double>(block.values[block_offset]);
                sum +=
                    static_cast<double>(
                        activations[
                            static_cast<std::uint64_t>(row) * input_features +
                            in]) *
                    weight;
            }
            output[static_cast<std::uint64_t>(row) * output_features + out] =
                static_cast<float>(sum);
        }
    }
    return output;
}

void expect_close(const std::vector<float>& actual,
                  const std::vector<float>& expected,
                  float tolerance,
                  const char* name) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + ": size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float scale = std::max(1.0F, std::fabs(expected[index]));
        if (difference > tolerance * scale) {
            throw std::runtime_error(
                std::string(name) + ": mismatch at " +
                std::to_string(index));
        }
    }
}

template <unsigned Rows>
void launch_projection(const DeviceBuffer<float>& activations,
                       const DeviceBuffer<Q8Block>& weights,
                       DeviceBuffer<float>& output,
                       std::uint32_t input_features,
                       std::uint32_t output_features) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(q8_projection_kernel<Rows>),
        dim3(output_features),
        dim3(kThreads),
        Rows * kThreads * sizeof(float),
        0,
        activations.pointer,
        weights.pointer,
        output.pointer,
        input_features,
        output_features);
}

template <unsigned Rows>
void correctness_fixture() {
    constexpr std::uint32_t input_features = 64;
    constexpr std::uint32_t output_features = 9;
    constexpr auto blocks_per_row = input_features / kBlockElements;

    std::vector<float> activations(
        static_cast<std::size_t>(Rows) * input_features);
    for (std::size_t index = 0; index < activations.size(); ++index) {
        activations[index] =
            static_cast<float>((static_cast<int>(index % 17U) - 8)) /
            11.0F;
    }

    std::vector<Q8Block> weights(
        static_cast<std::size_t>(output_features) * blocks_per_row);
    for (std::size_t block_index = 0;
         block_index < weights.size();
         ++block_index) {
        weights[block_index].scale =
            __float2half(0.01F * static_cast<float>((block_index % 7U) + 1U));
        for (std::uint32_t index = 0; index < kBlockElements; ++index) {
            weights[block_index].values[index] =
                static_cast<std::int8_t>(
                    static_cast<int>((block_index * 13U + index * 7U) % 31U) -
                    15);
        }
    }

    DeviceBuffer<float> d_activations(activations.size());
    DeviceBuffer<Q8Block> d_weights(weights.size());
    DeviceBuffer<float> d_output(
        static_cast<std::size_t>(Rows) * output_features);

    copy_to_device(d_activations, activations);
    copy_to_device(d_weights, weights);

    launch_projection<Rows>(
        d_activations, d_weights, d_output,
        input_features, output_features);
    check(hipGetLastError(), "q8 projection correctness launch");
    check(hipDeviceSynchronize(), "q8 projection correctness sync");

    const auto actual = copy_from_device(
        d_output,
        static_cast<std::size_t>(Rows) * output_features);
    const auto expected = cpu_reference(
        activations, weights, Rows, input_features, output_features);
    expect_close(actual, expected, 2e-5F, "q8 projection");
}

void row_invariance_fixture() {
    constexpr std::uint32_t rows = 2;
    constexpr std::uint32_t input_features = 64;
    constexpr std::uint32_t output_features = 7;
    constexpr auto blocks_per_row = input_features / kBlockElements;

    std::vector<float> activations(rows * input_features);
    for (std::size_t index = 0; index < activations.size(); ++index) {
        activations[index] =
            static_cast<float>((static_cast<int>(index % 19U) - 9)) /
            13.0F;
    }
    std::vector<float> solo(
        activations.begin() + input_features,
        activations.end());

    std::vector<Q8Block> weights(output_features * blocks_per_row);
    for (std::size_t block_index = 0;
         block_index < weights.size();
         ++block_index) {
        weights[block_index].scale = __float2half(0.03125F);
        for (std::uint32_t index = 0; index < kBlockElements; ++index) {
            weights[block_index].values[index] =
                static_cast<std::int8_t>(
                    static_cast<int>((block_index + index * 3U) % 23U) -
                    11);
        }
    }

    DeviceBuffer<float> d_batch(activations.size());
    DeviceBuffer<float> d_solo(solo.size());
    DeviceBuffer<Q8Block> d_weights(weights.size());
    DeviceBuffer<float> d_batch_output(rows * output_features);
    DeviceBuffer<float> d_solo_output(output_features);
    copy_to_device(d_batch, activations);
    copy_to_device(d_solo, solo);
    copy_to_device(d_weights, weights);

    launch_projection<2>(
        d_batch, d_weights, d_batch_output,
        input_features, output_features);
    launch_projection<1>(
        d_solo, d_weights, d_solo_output,
        input_features, output_features);
    check(hipGetLastError(), "q8 row invariance launch");
    check(hipDeviceSynchronize(), "q8 row invariance sync");

    const auto batch = copy_from_device(
        d_batch_output, rows * output_features);
    const auto single = copy_from_device(
        d_solo_output, output_features);
    const std::vector<float> second(
        batch.begin() + output_features,
        batch.end());
    expect_close(second, single, 1e-7F, "q8 row invariance");
}

template <typename Launch>
double benchmark_us(Launch&& launch,
                    unsigned warmup = 10,
                    unsigned iterations = 100) {
    for (unsigned index = 0; index < warmup; ++index) {
        launch();
    }
    check(hipGetLastError(), "q8 benchmark warmup launch");
    check(hipDeviceSynchronize(), "q8 benchmark warmup sync");

    hipEvent_t start{};
    hipEvent_t stop{};
    check(hipEventCreate(&start), "hipEventCreate start");
    check(hipEventCreate(&stop), "hipEventCreate stop");
    check(hipEventRecord(start), "hipEventRecord start");
    for (unsigned index = 0; index < iterations; ++index) {
        launch();
    }
    check(hipGetLastError(), "q8 benchmark measured launch");
    check(hipEventRecord(stop), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize stop");

    float milliseconds = 0.0F;
    check(hipEventElapsedTime(&milliseconds, start, stop),
          "hipEventElapsedTime");
    check(hipEventDestroy(start), "hipEventDestroy start");
    check(hipEventDestroy(stop), "hipEventDestroy stop");
    return static_cast<double>(milliseconds) * 1000.0 /
           static_cast<double>(iterations);
}

struct KernelAttributes {
    int registers = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
};

template <unsigned Rows>
KernelAttributes attributes() {
    hipFuncAttributes value{};
    check(
        hipFuncGetAttributes(
            &value,
            reinterpret_cast<const void*>(
                q8_projection_kernel<Rows>)),
        "hipFuncGetAttributes q8 projection");
    return KernelAttributes{
        value.numRegs,
        value.sharedSizeBytes,
        value.localSizeBytes,
    };
}

struct Measurement {
    unsigned rows = 0;
    double microseconds = 0.0;
    std::uint64_t weight_bytes = 0;
    double physical_weight_gib_s = 0.0;
    double serial_equivalent_weight_gib_s = 0.0;
    KernelAttributes resources;
};

template <unsigned Rows>
Measurement benchmark_row_class() {
    constexpr std::uint32_t input_features = 2560;
    constexpr std::uint32_t output_features = 2560;
    constexpr auto blocks_per_row = input_features / kBlockElements;
    constexpr std::uint64_t weight_bytes =
        static_cast<std::uint64_t>(output_features) *
        blocks_per_row * sizeof(Q8Block);

    std::vector<float> activations(
        static_cast<std::size_t>(Rows) * input_features,
        0.125F);
    std::vector<Q8Block> weights(
        static_cast<std::size_t>(output_features) * blocks_per_row);
    for (auto& block : weights) {
        block.scale = __float2half(0.015625F);
        for (std::uint32_t index = 0; index < kBlockElements; ++index) {
            block.values[index] =
                static_cast<std::int8_t>(
                    static_cast<int>(index) - 16);
        }
    }

    DeviceBuffer<float> d_activations(activations.size());
    DeviceBuffer<Q8Block> d_weights(weights.size());
    DeviceBuffer<float> d_output(
        static_cast<std::size_t>(Rows) * output_features);
    copy_to_device(d_activations, activations);
    copy_to_device(d_weights, weights);

    const auto microseconds = benchmark_us([&] {
        launch_projection<Rows>(
            d_activations, d_weights, d_output,
            input_features, output_features);
    });

    const double seconds = microseconds * 1e-6;
    const double gib = static_cast<double>(weight_bytes) /
                       (1024.0 * 1024.0 * 1024.0);

    Measurement result;
    result.rows = Rows;
    result.microseconds = microseconds;
    result.weight_bytes = weight_bytes;
    result.physical_weight_gib_s =
        seconds > 0.0 ? gib / seconds : 0.0;
    result.serial_equivalent_weight_gib_s =
        seconds > 0.0 ? gib * static_cast<double>(Rows) / seconds : 0.0;
    result.resources = attributes<Rows>();
    return result;
}

void print_measurement(const Measurement& value, bool last) {
    std::cout
        << "    {\"rows\":" << value.rows
        << ",\"microseconds\":" << value.microseconds
        << ",\"weight_bytes\":" << value.weight_bytes
        << ",\"physical_weight_gib_s\":"
        << value.physical_weight_gib_s
        << ",\"serial_equivalent_weight_gib_s\":"
        << value.serial_equivalent_weight_gib_s
        << ",\"registers\":" << value.resources.registers
        << ",\"static_shared_bytes\":"
        << value.resources.static_shared_bytes
        << ",\"dynamic_shared_bytes\":"
        << value.rows * kThreads * sizeof(float)
        << ",\"local_bytes\":" << value.resources.local_bytes
        << '}' << (last ? '\n' : ',') << '\n';
}

}  // namespace

int main() try {
    check(hipSetDevice(0), "hipSetDevice");

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0),
          "hipGetDeviceProperties");
    const std::string architecture = properties.gcnArchName;
    if (architecture.rfind("gfx1151", 0) != 0) {
        throw std::runtime_error(
            "packed projection lab requires gfx1151; got " +
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
    check(hipGetLastError(), "wave report launch");
    check(hipDeviceSynchronize(), "wave report sync");
    const auto runtime_wave = copy_from_device(d_wave, 1);
    if (runtime_wave[0] != STRIX_WAVE_SIZE) {
        throw std::runtime_error("compiled wave-size mismatch");
    }

    correctness_fixture<1>();
    correctness_fixture<2>();
    correctness_fixture<4>();
    correctness_fixture<8>();
    row_invariance_fixture();

    const auto c1 = benchmark_row_class<1>();
    const auto r2 = benchmark_row_class<2>();
    const auto r4 = benchmark_row_class<4>();
    const auto r8 = benchmark_row_class<8>();

    std::cout << "{\n"
              << "  \"status\":\"pass\",\n"
              << "  \"format\":\"Q8_0\",\n"
              << "  \"gpu_arch\":\"" << architecture << "\",\n"
              << "  \"wave_size\":" << STRIX_WAVE_SIZE << ",\n"
              << "  \"row_invariance_fixture\":true,\n"
              << "  \"measurements\":[\n";
    print_measurement(c1, false);
    print_measurement(r2, false);
    print_measurement(r4, false);
    print_measurement(r8, true);
    std::cout << "  ]\n"
              << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-q8-projection: " << error.what() << '\n';
    return 1;
}
