#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STRIX_WAVE_SIZE
#define STRIX_WAVE_SIZE 32
#endif

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kMaximumTopK = 16;

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

__device__ bool better(float left_value,
                       std::uint32_t left_index,
                       float right_value,
                       std::uint32_t right_index) {
    return left_value > right_value ||
           (left_value == right_value && left_index < right_index);
}

__global__ void residual_add_kernel(const float* left,
                                    const float* right,
                                    float* output,
                                    std::uint64_t count) {
    const auto stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += stride) {
        output[index] = left[index] + right[index];
    }
}

__global__ void partial_rope_kernel(const float* input,
                                    const float* cosine,
                                    const float* sine,
                                    float* output,
                                    std::uint32_t rows,
                                    std::uint32_t hidden_size,
                                    std::uint32_t rotary_dim) {
    const auto row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const auto row_offset =
        static_cast<std::uint64_t>(row) * hidden_size;

    for (std::uint32_t index = threadIdx.x;
         index < hidden_size;
         index += blockDim.x) {
        if (index < rotary_dim) {
            const auto half = rotary_dim / 2U;
            const auto partner =
                index < half ? index + half : index - half;
            const float rotated =
                index < half ? -input[row_offset + partner]
                             : input[row_offset + partner];
            output[row_offset + index] =
                input[row_offset + index] * cosine[index] +
                rotated * sine[index];
        } else {
            output[row_offset + index] = input[row_offset + index];
        }
    }
}

__global__ void rms_norm_kernel(const float* input,
                                const float* weight,
                                float* output,
                                std::uint32_t rows,
                                std::uint32_t hidden_size,
                                float epsilon) {
    extern __shared__ float shared[];
    const auto row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const auto row_offset =
        static_cast<std::uint64_t>(row) * hidden_size;
    float local_sum = 0.0F;
    for (std::uint32_t index = threadIdx.x;
         index < hidden_size;
         index += blockDim.x) {
        const float value = input[row_offset + index];
        local_sum += value * value;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned offset = blockDim.x / 2U;
         offset > 0;
         offset /= 2U) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const float inverse_rms =
        rsqrtf(shared[0] / static_cast<float>(hidden_size) + epsilon);

    for (std::uint32_t index = threadIdx.x;
         index < hidden_size;
         index += blockDim.x) {
        const float value = input[row_offset + index];
        output[row_offset + index] =
            value * inverse_rms * (1.0F + weight[index]);
    }
}

struct Candidate {
    float value;
    std::uint32_t index;
};

__device__ Candidate reduce_best(Candidate local,
                                 Candidate* shared_values) {
    shared_values[threadIdx.x] = local;
    __syncthreads();

    for (unsigned offset = blockDim.x / 2U;
         offset > 0;
         offset /= 2U) {
        if (threadIdx.x < offset) {
            const auto other = shared_values[threadIdx.x + offset];
            if (better(other.value, other.index,
                       shared_values[threadIdx.x].value,
                       shared_values[threadIdx.x].index)) {
                shared_values[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    return shared_values[0];
}

__global__ void argmax_kernel(const float* input,
                              std::uint32_t* output,
                              std::uint32_t rows,
                              std::uint32_t width) {
    extern __shared__ Candidate shared_values[];
    const auto row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    Candidate local{
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<std::uint32_t>::max(),
    };
    const auto row_offset = static_cast<std::uint64_t>(row) * width;
    for (std::uint32_t index = threadIdx.x;
         index < width;
         index += blockDim.x) {
        const float value = input[row_offset + index];
        if (better(value, index, local.value, local.index)) {
            local = Candidate{value, index};
        }
    }

    const auto winner = reduce_best(local, shared_values);
    if (threadIdx.x == 0) {
        output[row] = winner.index;
    }
}

__global__ void top_k_kernel(const float* input,
                             std::uint32_t* output,
                             std::uint32_t rows,
                             std::uint32_t width,
                             std::uint32_t k) {
    extern __shared__ Candidate shared_values[];
    __shared__ std::uint32_t selected[kMaximumTopK];

    const auto row = blockIdx.x;
    if (row >= rows || k == 0 || k > kMaximumTopK) {
        return;
    }
    const auto row_offset = static_cast<std::uint64_t>(row) * width;

    for (std::uint32_t rank = 0; rank < k; ++rank) {
        Candidate local{
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<std::uint32_t>::max(),
        };

        for (std::uint32_t index = threadIdx.x;
             index < width;
             index += blockDim.x) {
            bool already_selected = false;
            for (std::uint32_t previous = 0;
                 previous < rank;
                 ++previous) {
                if (selected[previous] == index) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }

            const float value = input[row_offset + index];
            if (better(value, index, local.value, local.index)) {
                local = Candidate{value, index};
            }
        }

        const auto winner = reduce_best(local, shared_values);
        if (threadIdx.x == 0) {
            selected[rank] = winner.index;
            output[static_cast<std::uint64_t>(row) * k + rank] =
                winner.index;
        }
        __syncthreads();
    }
}

std::vector<float> cpu_rms_norm(const std::vector<float>& input,
                                const std::vector<float>& weight,
                                std::uint32_t rows,
                                std::uint32_t hidden_size,
                                float epsilon) {
    std::vector<float> output(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto offset =
            static_cast<std::uint64_t>(row) * hidden_size;
        double sum = 0.0;
        for (std::uint32_t index = 0; index < hidden_size; ++index) {
            const double value = input[offset + index];
            sum += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(sum / hidden_size + epsilon);
        for (std::uint32_t index = 0; index < hidden_size; ++index) {
            output[offset + index] =
                static_cast<float>(
                    static_cast<double>(input[offset + index]) *
                    inverse_rms *
                    (1.0 + static_cast<double>(weight[index])));
        }
    }
    return output;
}

std::vector<float> cpu_partial_rope(const std::vector<float>& input,
                                    const std::vector<float>& cosine,
                                    const std::vector<float>& sine,
                                    std::uint32_t rows,
                                    std::uint32_t hidden_size,
                                    std::uint32_t rotary_dim) {
    std::vector<float> output(input.size());
    const auto half = rotary_dim / 2U;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto offset =
            static_cast<std::uint64_t>(row) * hidden_size;
        for (std::uint32_t index = 0; index < hidden_size; ++index) {
            if (index < rotary_dim) {
                const auto partner =
                    index < half ? index + half : index - half;
                const float rotated =
                    index < half ? -input[offset + partner]
                                 : input[offset + partner];
                output[offset + index] =
                    input[offset + index] * cosine[index] +
                    rotated * sine[index];
            } else {
                output[offset + index] = input[offset + index];
            }
        }
    }
    return output;
}

std::vector<std::uint32_t> cpu_top_k(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t width,
    std::uint32_t k) {
    std::vector<std::uint32_t> output(
        static_cast<std::size_t>(rows) * k);
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::vector<std::uint32_t> indices(width);
        for (std::uint32_t index = 0; index < width; ++index) {
            indices[index] = index;
        }
        const auto offset = static_cast<std::uint64_t>(row) * width;
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<std::ptrdiff_t>(k),
            indices.end(),
            [&](std::uint32_t left, std::uint32_t right) {
                const float left_value = input[offset + left];
                const float right_value = input[offset + right];
                return left_value > right_value ||
                       (left_value == right_value && left < right);
            });
        for (std::uint32_t rank = 0; rank < k; ++rank) {
            output[static_cast<std::uint64_t>(row) * k + rank] =
                indices[rank];
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
        const float scale =
            std::max(1.0F, std::fabs(expected[index]));
        if (difference > tolerance * scale) {
            throw std::runtime_error(
                std::string(name) + ": mismatch at index " +
                std::to_string(index));
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
            static_cast<void>(
                hipFree(pointer));
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

void run_fixture(std::uint32_t rows) {
    constexpr std::uint32_t hidden_size = 16;
    constexpr std::uint32_t rotary_dim = 8;
    constexpr std::uint32_t top_k = 3;

    std::vector<float> input(
        static_cast<std::size_t>(rows) * hidden_size);
    std::vector<float> residual(input.size());
    std::vector<float> weight(hidden_size);
    std::vector<float> cosine(rotary_dim);
    std::vector<float> sine(rotary_dim);

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            static_cast<float>((static_cast<int>(index % 11U) - 5)) /
            7.0F;
        residual[index] =
            static_cast<float>((index % 5U) + 1U) / 13.0F;
    }
    for (std::uint32_t index = 0; index < hidden_size; ++index) {
        weight[index] =
            static_cast<float>((static_cast<int>(index % 7U) - 3)) /
            50.0F;
    }
    for (std::uint32_t index = 0; index < rotary_dim; ++index) {
        const float angle = 0.05F * static_cast<float>(index + 1U);
        cosine[index] = std::cos(angle);
        sine[index] = std::sin(angle);
    }

    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_residual(residual.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_cosine(cosine.size());
    DeviceBuffer<float> d_sine(sine.size());
    DeviceBuffer<float> d_output(input.size());
    DeviceBuffer<std::uint32_t> d_indices(
        static_cast<std::size_t>(rows) * top_k);

    copy_to_device(d_input, input);
    copy_to_device(d_residual, residual);
    copy_to_device(d_weight, weight);
    copy_to_device(d_cosine, cosine);
    copy_to_device(d_sine, sine);

    const auto blocks =
        static_cast<unsigned>((input.size() + kThreads - 1U) / kThreads);
    hipLaunchKernelGGL(
        residual_add_kernel,
        dim3(blocks),
        dim3(kThreads),
        0,
        0,
        d_input.pointer,
        d_residual.pointer,
        d_output.pointer,
        static_cast<std::uint64_t>(input.size()));
    check(hipGetLastError(), "residual_add_kernel launch");
    check(hipDeviceSynchronize(), "residual_add_kernel sync");

    auto gpu = copy_from_device(d_output, input.size());
    std::vector<float> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = input[index] + residual[index];
    }
    expect_close(gpu, expected, 1e-6F, "residual");

    hipLaunchKernelGGL(
        partial_rope_kernel,
        dim3(rows),
        dim3(kThreads),
        0,
        0,
        d_input.pointer,
        d_cosine.pointer,
        d_sine.pointer,
        d_output.pointer,
        rows,
        hidden_size,
        rotary_dim);
    check(hipGetLastError(), "partial_rope_kernel launch");
    check(hipDeviceSynchronize(), "partial_rope_kernel sync");

    gpu = copy_from_device(d_output, input.size());
    expected = cpu_partial_rope(
        input, cosine, sine, rows, hidden_size, rotary_dim);
    expect_close(gpu, expected, 1e-6F, "partial_rope");

    hipLaunchKernelGGL(
        rms_norm_kernel,
        dim3(rows),
        dim3(kThreads),
        kThreads * sizeof(float),
        0,
        d_input.pointer,
        d_weight.pointer,
        d_output.pointer,
        rows,
        hidden_size,
        1e-6F);
    check(hipGetLastError(), "rms_norm_kernel launch");
    check(hipDeviceSynchronize(), "rms_norm_kernel sync");

    gpu = copy_from_device(d_output, input.size());
    expected = cpu_rms_norm(
        input, weight, rows, hidden_size, 1e-6F);
    expect_close(gpu, expected, 2e-5F, "rms_norm");

    hipLaunchKernelGGL(
        top_k_kernel,
        dim3(rows),
        dim3(kThreads),
        kThreads * sizeof(Candidate),
        0,
        d_input.pointer,
        d_indices.pointer,
        rows,
        hidden_size,
        top_k);
    check(hipGetLastError(), "top_k_kernel launch");
    check(hipDeviceSynchronize(), "top_k_kernel sync");

    const auto gpu_indices = copy_from_device(
        d_indices,
        static_cast<std::size_t>(rows) * top_k);
    const auto expected_indices =
        cpu_top_k(input, rows, hidden_size, top_k);
    if (gpu_indices != expected_indices) {
        throw std::runtime_error("top-k mismatch");
    }

    DeviceBuffer<std::uint32_t> d_argmax(rows);
    hipLaunchKernelGGL(
        argmax_kernel,
        dim3(rows),
        dim3(kThreads),
        kThreads * sizeof(Candidate),
        0,
        d_input.pointer,
        d_argmax.pointer,
        rows,
        hidden_size);
    check(hipGetLastError(), "argmax_kernel launch");
    check(hipDeviceSynchronize(), "argmax_kernel sync");

    const auto gpu_argmax = copy_from_device(d_argmax, rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        if (gpu_argmax[row] !=
            expected_indices[static_cast<std::uint64_t>(row) * top_k]) {
            throw std::runtime_error("argmax mismatch");
        }
    }
}



struct FunctionStats {
    int max_threads_per_block = 0;
    int num_registers = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
};

FunctionStats function_stats(const void* function) {
    hipFuncAttributes attributes{};
    check(hipFuncGetAttributes(&attributes, function),
          "hipFuncGetAttributes");
    return FunctionStats{
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        attributes.sharedSizeBytes,
        attributes.localSizeBytes,
    };
}

template <typename Launch>
double benchmark_kernel_us(Launch&& launch,
                           unsigned warmup = 20,
                           unsigned iterations = 200) {
    for (unsigned iteration = 0; iteration < warmup; ++iteration) {
        launch();
    }
    check(hipGetLastError(), "benchmark warmup launch");
    check(hipDeviceSynchronize(), "benchmark warmup sync");

    hipEvent_t start{};
    hipEvent_t stop{};
    check(hipEventCreate(&start), "hipEventCreate start");
    check(hipEventCreate(&stop), "hipEventCreate stop");

    check(hipEventRecord(start), "hipEventRecord start");
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        launch();
    }
    check(hipGetLastError(), "benchmark measured launch");
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

struct BenchmarkResults {
    double rms_norm_us = 0.0;
    double residual_add_us = 0.0;
    double partial_rope_us = 0.0;
    double argmax_us = 0.0;
    double top_k_us = 0.0;
};

BenchmarkResults run_benchmarks() {
    constexpr std::uint32_t hidden_size = 2560;
    constexpr std::uint32_t attention_heads = 24;
    constexpr std::uint32_t head_dim = 256;
    constexpr std::uint32_t rotary_dim = 64;
    constexpr std::uint32_t vocab_size = 248320;
    constexpr std::uint32_t experts = 512;
    constexpr std::uint32_t expert_top_k = 10;

    std::vector<float> hidden(hidden_size, 0.25F);
    std::vector<float> residual(hidden_size, 0.125F);
    std::vector<float> weight(hidden_size, 0.0F);
    std::vector<float> heads(
        static_cast<std::size_t>(attention_heads) * head_dim,
        0.2F);
    std::vector<float> cosine(rotary_dim, 1.0F);
    std::vector<float> sine(rotary_dim, 0.0F);
    std::vector<float> logits(vocab_size);
    std::vector<float> router(experts);

    for (std::uint32_t index = 0; index < vocab_size; ++index) {
        logits[index] =
            static_cast<float>((index * 17U) % 1009U) / 1009.0F;
    }
    for (std::uint32_t index = 0; index < experts; ++index) {
        router[index] =
            static_cast<float>((index * 29U) % 521U) / 521.0F;
    }

    DeviceBuffer<float> d_hidden(hidden.size());
    DeviceBuffer<float> d_residual(residual.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_heads(heads.size());
    DeviceBuffer<float> d_cosine(cosine.size());
    DeviceBuffer<float> d_sine(sine.size());
    DeviceBuffer<float> d_hidden_output(hidden.size());
    DeviceBuffer<float> d_heads_output(heads.size());
    DeviceBuffer<float> d_logits(logits.size());
    DeviceBuffer<float> d_router(router.size());
    DeviceBuffer<std::uint32_t> d_argmax(1);
    DeviceBuffer<std::uint32_t> d_top_k(expert_top_k);

    copy_to_device(d_hidden, hidden);
    copy_to_device(d_residual, residual);
    copy_to_device(d_weight, weight);
    copy_to_device(d_heads, heads);
    copy_to_device(d_cosine, cosine);
    copy_to_device(d_sine, sine);
    copy_to_device(d_logits, logits);
    copy_to_device(d_router, router);

    BenchmarkResults results;

    results.rms_norm_us = benchmark_kernel_us([&] {
        hipLaunchKernelGGL(
            rms_norm_kernel,
            dim3(1),
            dim3(kThreads),
            kThreads * sizeof(float),
            0,
            d_hidden.pointer,
            d_weight.pointer,
            d_hidden_output.pointer,
            1,
            hidden_size,
            1e-6F);
    });

    results.residual_add_us = benchmark_kernel_us([&] {
        hipLaunchKernelGGL(
            residual_add_kernel,
            dim3((hidden_size + kThreads - 1U) / kThreads),
            dim3(kThreads),
            0,
            0,
            d_hidden.pointer,
            d_residual.pointer,
            d_hidden_output.pointer,
            static_cast<std::uint64_t>(hidden_size));
    });

    results.partial_rope_us = benchmark_kernel_us([&] {
        hipLaunchKernelGGL(
            partial_rope_kernel,
            dim3(attention_heads),
            dim3(kThreads),
            0,
            0,
            d_heads.pointer,
            d_cosine.pointer,
            d_sine.pointer,
            d_heads_output.pointer,
            attention_heads,
            head_dim,
            rotary_dim);
    });

    results.argmax_us = benchmark_kernel_us([&] {
        hipLaunchKernelGGL(
            argmax_kernel,
            dim3(1),
            dim3(kThreads),
            kThreads * sizeof(Candidate),
            0,
            d_logits.pointer,
            d_argmax.pointer,
            1,
            vocab_size);
    });

    results.top_k_us = benchmark_kernel_us([&] {
        hipLaunchKernelGGL(
            top_k_kernel,
            dim3(1),
            dim3(kThreads),
            kThreads * sizeof(Candidate),
            0,
            d_router.pointer,
            d_top_k.pointer,
            1,
            experts,
            expert_top_k);
    });

    return results;
}

void print_function_stats(const char* name, const FunctionStats& stats) {
    std::cout << "    \"" << name << "\": {"
              << "\"max_threads_per_block\":"
              << stats.max_threads_per_block << ','
              << "\"num_registers\":" << stats.num_registers << ','
              << "\"static_shared_bytes\":"
              << stats.static_shared_bytes << ','
              << "\"local_bytes\":" << stats.local_bytes
              << '}';
}

void run_row_invariance_fixture() {
    constexpr std::uint32_t rows = 2;
    constexpr std::uint32_t hidden_size = 16;
    constexpr std::uint32_t rotary_dim = 8;
    constexpr std::uint32_t top_k = 3;

    std::vector<float> input(
        static_cast<std::size_t>(rows) * hidden_size);
    std::vector<float> weight(hidden_size);
    std::vector<float> cosine(rotary_dim);
    std::vector<float> sine(rotary_dim);

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            static_cast<float>((static_cast<int>(index % 13U) - 6)) /
            9.0F;
    }
    for (std::uint32_t index = 0; index < hidden_size; ++index) {
        weight[index] =
            static_cast<float>((static_cast<int>(index % 5U) - 2)) /
            40.0F;
    }
    for (std::uint32_t index = 0; index < rotary_dim; ++index) {
        const float angle = 0.07F * static_cast<float>(index + 1U);
        cosine[index] = std::cos(angle);
        sine[index] = std::sin(angle);
    }

    std::vector<float> solo_input(
        input.begin() + hidden_size,
        input.begin() + 2U * hidden_size);

    DeviceBuffer<float> d_batch(input.size());
    DeviceBuffer<float> d_solo(solo_input.size());
    DeviceBuffer<float> d_weight(weight.size());
    DeviceBuffer<float> d_cosine(cosine.size());
    DeviceBuffer<float> d_sine(sine.size());
    DeviceBuffer<float> d_batch_output(input.size());
    DeviceBuffer<float> d_solo_output(solo_input.size());
    DeviceBuffer<std::uint32_t> d_batch_indices(rows * top_k);
    DeviceBuffer<std::uint32_t> d_solo_indices(top_k);

    copy_to_device(d_batch, input);
    copy_to_device(d_solo, solo_input);
    copy_to_device(d_weight, weight);
    copy_to_device(d_cosine, cosine);
    copy_to_device(d_sine, sine);

    hipLaunchKernelGGL(
        rms_norm_kernel,
        dim3(rows),
        dim3(kThreads),
        kThreads * sizeof(float),
        0,
        d_batch.pointer,
        d_weight.pointer,
        d_batch_output.pointer,
        rows,
        hidden_size,
        1e-6F);
    check(hipGetLastError(), "row-invariant batch RMSNorm launch");

    hipLaunchKernelGGL(
        rms_norm_kernel,
        dim3(1),
        dim3(kThreads),
        kThreads * sizeof(float),
        0,
        d_solo.pointer,
        d_weight.pointer,
        d_solo_output.pointer,
        1,
        hidden_size,
        1e-6F);
    check(hipGetLastError(), "row-invariant solo RMSNorm launch");
    check(hipDeviceSynchronize(), "row-invariant RMSNorm sync");

    const auto batch_rms = copy_from_device(d_batch_output, input.size());
    const auto solo_rms = copy_from_device(d_solo_output, solo_input.size());
    const std::vector<float> batch_second_rms(
        batch_rms.begin() + hidden_size,
        batch_rms.begin() + 2U * hidden_size);
    expect_close(
        batch_second_rms,
        solo_rms,
        1e-7F,
        "RMSNorm row invariance");

    hipLaunchKernelGGL(
        partial_rope_kernel,
        dim3(rows),
        dim3(kThreads),
        0,
        0,
        d_batch.pointer,
        d_cosine.pointer,
        d_sine.pointer,
        d_batch_output.pointer,
        rows,
        hidden_size,
        rotary_dim);
    check(hipGetLastError(), "row-invariant batch RoPE launch");

    hipLaunchKernelGGL(
        partial_rope_kernel,
        dim3(1),
        dim3(kThreads),
        0,
        0,
        d_solo.pointer,
        d_cosine.pointer,
        d_sine.pointer,
        d_solo_output.pointer,
        1,
        hidden_size,
        rotary_dim);
    check(hipGetLastError(), "row-invariant solo RoPE launch");
    check(hipDeviceSynchronize(), "row-invariant RoPE sync");

    const auto batch_rope = copy_from_device(d_batch_output, input.size());
    const auto solo_rope = copy_from_device(d_solo_output, solo_input.size());
    const std::vector<float> batch_second_rope(
        batch_rope.begin() + hidden_size,
        batch_rope.begin() + 2U * hidden_size);
    expect_close(
        batch_second_rope,
        solo_rope,
        0.0F,
        "RoPE row invariance");

    hipLaunchKernelGGL(
        top_k_kernel,
        dim3(rows),
        dim3(kThreads),
        kThreads * sizeof(Candidate),
        0,
        d_batch.pointer,
        d_batch_indices.pointer,
        rows,
        hidden_size,
        top_k);
    check(hipGetLastError(), "row-invariant batch top-k launch");

    hipLaunchKernelGGL(
        top_k_kernel,
        dim3(1),
        dim3(kThreads),
        kThreads * sizeof(Candidate),
        0,
        d_solo.pointer,
        d_solo_indices.pointer,
        1,
        hidden_size,
        top_k);
    check(hipGetLastError(), "row-invariant solo top-k launch");
    check(hipDeviceSynchronize(), "row-invariant top-k sync");

    const auto batch_topk = copy_from_device(d_batch_indices, rows * top_k);
    const auto solo_topk = copy_from_device(d_solo_indices, top_k);
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
        if (batch_topk[top_k + rank] != solo_topk[rank]) {
            throw std::runtime_error("top-k row invariance mismatch");
        }
    }
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
            "basic kernel lab requires a gfx1151 device; got " +
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
    check(hipGetLastError(), "report_wave_size_kernel launch");
    check(hipDeviceSynchronize(), "report_wave_size_kernel sync");
    const auto runtime_wave = copy_from_device(d_wave, 1);
    if (runtime_wave[0] != STRIX_WAVE_SIZE) {
        throw std::runtime_error(
            "compiled wave size mismatch: expected " +
            std::to_string(STRIX_WAVE_SIZE) + ", device kernel reported " +
            std::to_string(runtime_wave[0]));
    }

    run_fixture(1);
    run_fixture(2);
    run_row_invariance_fixture();

    const auto benchmark_results = run_benchmarks();

    const auto rms_stats = function_stats(
        reinterpret_cast<const void*>(rms_norm_kernel));
    const auto residual_stats = function_stats(
        reinterpret_cast<const void*>(residual_add_kernel));
    const auto rope_stats = function_stats(
        reinterpret_cast<const void*>(partial_rope_kernel));
    const auto argmax_stats = function_stats(
        reinterpret_cast<const void*>(argmax_kernel));
    const auto top_k_stats = function_stats(
        reinterpret_cast<const void*>(top_k_kernel));

    std::cout << "{\n"
              << "  \"status\": \"pass\",\n"
              << "  \"gpu_arch\": \"" << architecture << "\",\n"
              << "  \"wave_size\": " << STRIX_WAVE_SIZE << ",\n"
              << "  \"threads\": " << kThreads << ",\n"
              << "  \"precision\": \"fp32\",\n"
              << "  \"row_invariance_fixture\": true,\n"
              << "  \"benchmarks_us\": {\n"
              << "    \"rms_norm_c1_h2560\": "
              << benchmark_results.rms_norm_us << ",\n"
              << "    \"residual_c1_h2560\": "
              << benchmark_results.residual_add_us << ",\n"
              << "    \"partial_rope_24x256_r64\": "
              << benchmark_results.partial_rope_us << ",\n"
              << "    \"argmax_vocab248320\": "
              << benchmark_results.argmax_us << ",\n"
              << "    \"top10_experts512\": "
              << benchmark_results.top_k_us << "\n"
              << "  },\n"
              << "  \"function_attributes\": {\n";
    print_function_stats("rms_norm", rms_stats);
    std::cout << ",\n";
    print_function_stats("residual_add", residual_stats);
    std::cout << ",\n";
    print_function_stats("partial_rope", rope_stats);
    std::cout << ",\n";
    print_function_stats("argmax", argmax_stats);
    std::cout << ",\n";
    print_function_stats("top_k", top_k_stats);
    std::cout << "\n  }\n"
              << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-basic-kernels: " << error.what() << '\n';
    return 1;
}
