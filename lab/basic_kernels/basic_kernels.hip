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
constexpr unsigned kMaximumTopK = 8;

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
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

    run_fixture(1);
    run_fixture(2);

    std::cout << "{\n"
              << "  \"status\": \"pass\",\n"
              << "  \"gpu_arch\": \"" << architecture << "\",\n"
              << "  \"wave_size\": " << STRIX_WAVE_SIZE << ",\n"
              << "  \"threads\": " << kThreads << ",\n"
              << "  \"precision\": \"fp32\",\n"
              << "  \"row_invariance_fixture\": true\n"
              << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-basic-kernels: " << error.what() << '\n';
    return 1;
}
