#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
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

template <unsigned Rows>
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
            accumulators[row] +=
                activations[
                    static_cast<std::uint64_t>(row) *
                        input_features +
                    input_index] *
                weight;
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
         offset > 0;
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

    if (threadIdx.x == 0) {
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

template <unsigned Rows>
void launch_projection(
    const DeviceBuffer<float>& activations,
    const DeviceBuffer<IQ4XSBlock>& weights,
    DeviceBuffer<float>& output,
    std::uint32_t input_features,
    std::uint32_t output_features) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            iq4_xs_projection_kernel<Rows>),
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

template <typename Launch>
double benchmark_us(
    Launch&& launch,
    unsigned warmup = 10,
    unsigned iterations = 100) {
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

    hipEvent_t start{};
    hipEvent_t stop{};
    check(
        hipEventCreate(&start),
        "hipEventCreate start");
    check(
        hipEventCreate(&stop),
        "hipEventCreate stop");
    check(
        hipEventRecord(start),
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
        hipEventRecord(stop),
        "hipEventRecord stop");
    check(
        hipEventSynchronize(stop),
        "hipEventSynchronize stop");

    float milliseconds = 0.0F;
    check(
        hipEventElapsedTime(
            &milliseconds,
            start,
            stop),
        "hipEventElapsedTime");
    check(
        hipEventDestroy(start),
        "hipEventDestroy start");
    check(
        hipEventDestroy(stop),
        "hipEventDestroy stop");

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

template <unsigned Rows>
KernelAttributes attributes() {
    hipFuncAttributes value{};
    check(
        hipFuncGetAttributes(
            &value,
            reinterpret_cast<const void*>(
                iq4_xs_projection_kernel<Rows>)),
        "hipFuncGetAttributes IQ4_XS");
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

    const auto microseconds =
        benchmark_us([&] {
            launch_projection<Rows>(
                d_activations,
                d_weights,
                d_output,
                input_features,
                output_features);
        });

    const double seconds =
        microseconds * 1e-6;
    const double gib =
        static_cast<double>(weight_bytes) /
        (1024.0 * 1024.0 * 1024.0);

    Measurement result;
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
    result.resources = attributes<Rows>();
    return result;
}

void print_measurement(
    const Measurement& value,
    bool last) {
    std::cout
        << "    {\"rows\":" << value.rows
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
        << "  \"measurements\":[\n";

    print_measurement(c1, false);
    print_measurement(r2, false);
    print_measurement(r4, false);
    print_measurement(r8, true);

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
