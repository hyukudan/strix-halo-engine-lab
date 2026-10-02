#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr std::uint32_t kBlockElements = 32;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kIntermediate = 640;
constexpr std::uint32_t kGateUpOutputs = 2U * kIntermediate;
constexpr std::uint32_t kMaxRowsPerBatch = 8;
constexpr std::string_view kFixtureId = "q8-target-dims-overlap-v1";

struct Q8Block {
    __half scale;
    std::int8_t values[kBlockElements];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(kHidden % kBlockElements == 0);
static_assert(kIntermediate % kBlockElements == 0);

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
    }
}

struct Options {
    int rows = 2;
    int experts = 16;
    int top_k = 10;
    int iterations = 20;
};

int parse_positive(std::string_view text, const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(std::string(name) + " is empty");
    }
    int value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(std::string(name) + " must be integer");
        }
        if (value > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
            throw std::overflow_error(std::string(name) + " overflows");
        }
        value = value * 10 + (c - '0');
    }
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto next = [&](const char* name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--rows") {
            options.rows = parse_positive(next("--rows"), "--rows");
        } else if (arg == "--experts") {
            options.experts = parse_positive(next("--experts"), "--experts");
        } else if (arg == "--top-k") {
            options.top_k = parse_positive(next("--top-k"), "--top-k");
        } else if (arg == "--iterations") {
            options.iterations = parse_positive(next("--iterations"), "--iterations");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (options.rows != 1 && options.rows != 2 &&
        options.rows != 4 && options.rows != 8) {
        throw std::invalid_argument("--rows must be 1, 2, 4 or 8");
    }
    if (options.experts <= 0 || options.experts > 512) {
        throw std::invalid_argument("--experts must be within 1..512");
    }
    if (options.top_k <= 0 || options.top_k > options.experts ||
        options.top_k > 16) {
        throw std::invalid_argument("--top-k is out of range");
    }
    return options;
}

__host__ __device__ float dequantized_weight(
    const Q8Block& block,
    std::uint32_t within_block) {
    return __half2float(block.scale) *
           static_cast<float>(block.values[within_block]);
}

float dequantized_weight_host(
    const Q8Block& block,
    std::uint32_t within_block) {
    return dequantized_weight(block, within_block);
}

std::vector<Q8Block> make_expert_weights(
    int experts,
    std::uint32_t input_features,
    std::uint32_t output_features,
    std::uint32_t salt) {
    const auto blocks_per_row = input_features / kBlockElements;
    const auto blocks_per_expert =
        static_cast<std::uint64_t>(output_features) * blocks_per_row;
    std::vector<Q8Block> weights(
        static_cast<std::size_t>(experts) * blocks_per_expert);

    for (std::size_t block_index = 0;
         block_index < weights.size(); ++block_index) {
        auto& block = weights[block_index];
        block.scale = __float2half(
            0.00008F *
            static_cast<float>(((block_index + salt) % 7U) + 1U));
        for (std::uint32_t item = 0; item < kBlockElements; ++item) {
            const auto raw =
                static_cast<int>(
                    (block_index * 13U +
                     static_cast<std::size_t>(item) * 17U +
                     static_cast<std::size_t>(salt) * 19U) % 255U) -
                127;
            block.values[item] = static_cast<std::int8_t>(raw);
        }
    }
    return weights;
}

template <unsigned Rows>
__global__ void expert_projection_kernel(
    const float* __restrict__ activations,
    const std::uint32_t* __restrict__ row_indices,
    const Q8Block* __restrict__ weights,
    std::uint32_t expert,
    float* __restrict__ output,
    std::uint32_t input_features,
    std::uint32_t output_features) {
    extern __shared__ float shared[];
    const auto output_index = blockIdx.x;
    if (output_index >= output_features) {
        return;
    }

    const auto blocks_per_row = input_features / kBlockElements;
    const auto blocks_per_expert =
        static_cast<std::uint64_t>(output_features) * blocks_per_row;
    const auto* weight_row =
        weights +
        static_cast<std::uint64_t>(expert) * blocks_per_expert +
        static_cast<std::uint64_t>(output_index) * blocks_per_row;

    float accumulators[Rows] = {};

    for (std::uint32_t input_index = threadIdx.x;
         input_index < input_features;
         input_index += blockDim.x) {
        const auto block_index = input_index / kBlockElements;
        const auto within_block = input_index % kBlockElements;
        const float weight =
            dequantized_weight(weight_row[block_index], within_block);

#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            const auto source_row = row_indices[row];
            accumulators[row] +=
                activations[
                    static_cast<std::uint64_t>(source_row) * input_features +
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

    for (unsigned offset = blockDim.x / 2U; offset > 0; offset /= 2U) {
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
            output[
                static_cast<std::uint64_t>(row) * output_features +
                output_index] =
                shared[static_cast<std::size_t>(row) * blockDim.x];
        }
    }
}

template <unsigned Rows>
__global__ void silu_gate_kernel(
    const float* __restrict__ gate_up,
    float* __restrict__ intermediate) {
    const auto index =
        static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = Rows * kIntermediate;
    if (index >= total) {
        return;
    }
    const auto row = index / kIntermediate;
    const auto inner = index % kIntermediate;
    const auto base =
        static_cast<std::uint64_t>(row) * kGateUpOutputs;
    const float gate = gate_up[base + inner];
    const float up = gate_up[base + kIntermediate + inner];
    intermediate[
        static_cast<std::uint64_t>(row) * kIntermediate + inner] =
        (gate / (1.0F + expf(-gate))) * up;
}

template <unsigned Rows>
__global__ void expert_down_scatter_kernel(
    const float* __restrict__ intermediate,
    const std::uint32_t* __restrict__ row_indices,
    const float* __restrict__ route_weights,
    const Q8Block* __restrict__ weights,
    std::uint32_t expert,
    float* __restrict__ output) {
    extern __shared__ float shared[];
    const auto output_index = blockIdx.x;
    if (output_index >= kHidden) {
        return;
    }

    constexpr auto blocks_per_row = kIntermediate / kBlockElements;
    constexpr auto blocks_per_expert =
        static_cast<std::uint64_t>(kHidden) * blocks_per_row;
    const auto* weight_row =
        weights +
        static_cast<std::uint64_t>(expert) * blocks_per_expert +
        static_cast<std::uint64_t>(output_index) * blocks_per_row;

    float accumulators[Rows] = {};
    for (std::uint32_t input_index = threadIdx.x;
         input_index < kIntermediate;
         input_index += blockDim.x) {
        const auto within_block = input_index % kBlockElements;
        const float weight =
            dequantized_weight(weight_row[0], within_block);
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            accumulators[row] +=
                intermediate[
                    static_cast<std::uint64_t>(row) * kIntermediate +
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

    for (unsigned offset = blockDim.x / 2U; offset > 0; offset /= 2U) {
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
            const auto destination_row = row_indices[row];
            atomicAdd(
                &output[
                    static_cast<std::uint64_t>(destination_row) * kHidden +
                    output_index],
                shared[static_cast<std::size_t>(row) * blockDim.x] *
                    route_weights[row]);
        }
    }
}

struct ExpertBatch {
    std::uint32_t expert = 0;
    std::size_t offset = 0;
    std::uint32_t rows = 0;
};

std::uint32_t largest_chunk(std::size_t remaining) {
    if (remaining >= 8U) return 8;
    if (remaining >= 4U) return 4;
    if (remaining >= 2U) return 2;
    if (remaining >= 1U) return 1;
    return 0;
}

void build_grouping(
    int rows,
    int top_k,
    int experts,
    const std::vector<std::uint32_t>& selected,
    const std::vector<float>& weights,
    std::vector<std::uint32_t>& grouped_rows,
    std::vector<float>& grouped_weights,
    std::vector<ExpertBatch>& batches) {
    std::vector<std::vector<std::size_t>> per_expert(
        static_cast<std::size_t>(experts));

    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * top_k + slot;
            per_expert[selected[assignment]].push_back(assignment);
        }
    }

    for (int expert = 0; expert < experts; ++expert) {
        const auto& list = per_expert[static_cast<std::size_t>(expert)];
        std::size_t cursor = 0;
        while (cursor < list.size()) {
            const auto count =
                largest_chunk(list.size() - cursor);
            const auto output_offset = grouped_rows.size();
            batches.push_back(
                ExpertBatch{
                    static_cast<std::uint32_t>(expert),
                    output_offset,
                    count,
                });
            for (std::uint32_t item = 0; item < count; ++item) {
                const auto assignment = list[cursor + item];
                grouped_rows.push_back(
                    static_cast<std::uint32_t>(
                        assignment / static_cast<std::size_t>(top_k)));
                grouped_weights.push_back(weights[assignment]);
            }
            cursor += count;
        }
    }
}

float host_projection_value(
    const float* activation,
    const Q8Block* weight_row,
    std::uint32_t input_features) {
    double sum = 0.0;
    for (std::uint32_t input = 0; input < input_features; ++input) {
        const auto block_index = input / kBlockElements;
        const auto within_block = input % kBlockElements;
        sum += static_cast<double>(activation[input]) *
               static_cast<double>(
                   dequantized_weight_host(
                       weight_row[block_index], within_block));
    }
    return static_cast<float>(sum);
}

void cpu_reference(
    int rows,
    int top_k,
    const std::vector<float>& hidden,
    const std::vector<std::uint32_t>& selected,
    const std::vector<float>& route_weights,
    const std::vector<Q8Block>& gate_up,
    const std::vector<Q8Block>& down,
    std::vector<float>& output) {
    std::fill(output.begin(), output.end(), 0.0F);
    constexpr auto gate_blocks_per_row = kHidden / kBlockElements;
    constexpr auto gate_blocks_per_expert =
        static_cast<std::uint64_t>(kGateUpOutputs) * gate_blocks_per_row;
    constexpr auto down_blocks_per_row = kIntermediate / kBlockElements;
    constexpr auto down_blocks_per_expert =
        static_cast<std::uint64_t>(kHidden) * down_blocks_per_row;

    for (int row = 0; row < rows; ++row) {
        const auto* activation =
            hidden.data() + static_cast<std::size_t>(row) * kHidden;
        for (int slot = 0; slot < top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * top_k + slot;
            const auto expert = selected[assignment];

            std::array<float, kIntermediate> intermediate{};
            for (std::uint32_t inner = 0; inner < kIntermediate; ++inner) {
                const auto* gate_row =
                    gate_up.data() +
                    static_cast<std::uint64_t>(expert) * gate_blocks_per_expert +
                    static_cast<std::uint64_t>(inner) * gate_blocks_per_row;
                const auto* up_row =
                    gate_up.data() +
                    static_cast<std::uint64_t>(expert) * gate_blocks_per_expert +
                    static_cast<std::uint64_t>(kIntermediate + inner) *
                        gate_blocks_per_row;

                const float gate =
                    host_projection_value(activation, gate_row, kHidden);
                const float up =
                    host_projection_value(activation, up_row, kHidden);
                intermediate[inner] =
                    (gate / (1.0F + std::exp(-gate))) * up;
            }

            for (std::uint32_t feature = 0; feature < kHidden; ++feature) {
                const auto* down_row =
                    down.data() +
                    static_cast<std::uint64_t>(expert) * down_blocks_per_expert +
                    static_cast<std::uint64_t>(feature) * down_blocks_per_row;
                output[
                    static_cast<std::uint64_t>(row) * kHidden + feature] +=
                    host_projection_value(
                        intermediate.data(), down_row, kIntermediate) *
                    route_weights[assignment];
            }
        }
    }
}

template <unsigned Rows>
void launch_batch(
    const float* hidden,
    const std::uint32_t* row_indices,
    const float* route_weights,
    const Q8Block* gate_up,
    const Q8Block* down,
    std::uint32_t expert,
    float* gate_up_temp,
    float* intermediate_temp,
    float* output,
    hipStream_t stream) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(expert_projection_kernel<Rows>),
        dim3(kGateUpOutputs),
        dim3(kThreads),
        Rows * kThreads * sizeof(float),
        stream,
        hidden,
        row_indices,
        gate_up,
        expert,
        gate_up_temp,
        kHidden,
        kGateUpOutputs);

    constexpr unsigned activation_threads = 256;
    const unsigned total = Rows * kIntermediate;
    const unsigned blocks =
        (total + activation_threads - 1U) / activation_threads;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(silu_gate_kernel<Rows>),
        dim3(blocks),
        dim3(activation_threads),
        0,
        stream,
        gate_up_temp,
        intermediate_temp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(expert_down_scatter_kernel<Rows>),
        dim3(kHidden),
        dim3(kThreads),
        Rows * kThreads * sizeof(float),
        stream,
        intermediate_temp,
        row_indices,
        route_weights,
        down,
        expert,
        output);
}

void dispatch_batch(
    const ExpertBatch& batch,
    const float* hidden,
    const std::uint32_t* rows,
    const float* weights,
    const Q8Block* gate_up,
    const Q8Block* down,
    float* gate_up_temp,
    float* intermediate_temp,
    float* output,
    hipStream_t stream) {
    const auto* batch_rows = rows + batch.offset;
    const auto* batch_weights = weights + batch.offset;
    switch (batch.rows) {
    case 1:
        launch_batch<1>(hidden, batch_rows, batch_weights, gate_up, down,
                        batch.expert, gate_up_temp, intermediate_temp, output, stream);
        break;
    case 2:
        launch_batch<2>(hidden, batch_rows, batch_weights, gate_up, down,
                        batch.expert, gate_up_temp, intermediate_temp, output, stream);
        break;
    case 4:
        launch_batch<4>(hidden, batch_rows, batch_weights, gate_up, down,
                        batch.expert, gate_up_temp, intermediate_temp, output, stream);
        break;
    case 8:
        launch_batch<8>(hidden, batch_rows, batch_weights, gate_up, down,
                        batch.expert, gate_up_temp, intermediate_temp, output, stream);
        break;
    default:
        throw std::logic_error("unsupported expert batch row class");
    }
}

float max_abs_error(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

double benchmark_batches(
    int iterations,
    const std::vector<ExpertBatch>& batches,
    const float* hidden,
    const std::uint32_t* rows,
    const float* route_weights,
    const Q8Block* gate_up,
    const Q8Block* down,
    float* gate_up_temp,
    float* intermediate_temp,
    float* output,
    std::size_t output_bytes) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");

    check(hipEventRecord(begin), "hipEventRecord begin");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check(hipMemset(output, 0, output_bytes), "hipMemset output");
        for (const auto& batch : batches) {
            dispatch_batch(
                batch, hidden, rows, route_weights, gate_up, down,
                gate_up_temp, intermediate_temp, output, nullptr);
        }
        check(hipGetLastError(), "packed MoE batch launch");
    }
    check(hipEventRecord(end), "hipEventRecord end");
    check(hipEventSynchronize(end), "hipEventSynchronize end");

    float milliseconds = 0.0F;
    check(hipEventElapsedTime(&milliseconds, begin, end), "hipEventElapsedTime");
    hipEventDestroy(begin);
    hipEventDestroy(end);
    return static_cast<double>(milliseconds) /
           static_cast<double>(iterations);
}

}  // namespace

int main(int argc, char** argv) try {
    const auto options = parse_args(argc, argv);
    const auto assignments =
        static_cast<std::size_t>(options.rows) * options.top_k;

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    if (properties.warpSize != 32) {
        throw std::runtime_error("target Q8 MoE lab requires Wave32");
    }
    if (std::string_view(properties.gcnArchName).substr(0, 7) != "gfx1151") {
        throw std::runtime_error("target Q8 MoE lab requires gfx1151");
    }

    std::vector<float> hidden(
        static_cast<std::size_t>(options.rows) * kHidden);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] =
            static_cast<float>(static_cast<int>(index % 31U) - 15) / 23.0F;
    }

    std::vector<std::uint32_t> selected(assignments);
    std::vector<float> route_weights(assignments);
    for (int row = 0; row < options.rows; ++row) {
        float sum = 0.0F;
        for (int slot = 0; slot < options.top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * options.top_k + slot;
            selected[assignment] =
                static_cast<std::uint32_t>(
                    (slot + row / 2) % options.experts);
            route_weights[assignment] =
                1.0F + static_cast<float>(options.top_k - slot);
            sum += route_weights[assignment];
        }
        for (int slot = 0; slot < options.top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * options.top_k + slot;
            route_weights[assignment] /= sum;
        }
    }

    const auto gate_up =
        make_expert_weights(options.experts, kHidden, kGateUpOutputs, 3U);
    const auto down =
        make_expert_weights(options.experts, kIntermediate, kHidden, 11U);

    std::vector<float> reference(
        static_cast<std::size_t>(options.rows) * kHidden);
    cpu_reference(
        options.rows, options.top_k, hidden, selected, route_weights,
        gate_up, down, reference);

    std::vector<std::uint32_t> direct_rows(assignments);
    std::vector<float> direct_weights = route_weights;
    std::vector<ExpertBatch> direct_batches;
    direct_batches.reserve(assignments);
    for (int row = 0; row < options.rows; ++row) {
        for (int slot = 0; slot < options.top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * options.top_k + slot;
            direct_rows[assignment] = static_cast<std::uint32_t>(row);
            direct_batches.push_back(
                ExpertBatch{
                    selected[assignment],
                    assignment,
                    1U,
                });
        }
    }

    std::vector<std::uint32_t> grouped_rows;
    std::vector<float> grouped_weights;
    std::vector<ExpertBatch> grouped_batches;
    grouped_rows.reserve(assignments);
    grouped_weights.reserve(assignments);
    build_grouping(
        options.rows, options.top_k, options.experts,
        selected, route_weights,
        grouped_rows, grouped_weights, grouped_batches);

    float* d_hidden = nullptr;
    Q8Block *d_gate_up = nullptr, *d_down = nullptr;
    std::uint32_t* d_rows = nullptr;
    float *d_route_weights = nullptr, *d_gate_up_temp = nullptr;
    float *d_intermediate_temp = nullptr, *d_output = nullptr;

    check(hipMalloc(&d_hidden, hidden.size() * sizeof(float)), "hipMalloc hidden");
    check(hipMalloc(&d_gate_up, gate_up.size() * sizeof(Q8Block)),
          "hipMalloc gate_up");
    check(hipMalloc(&d_down, down.size() * sizeof(Q8Block)),
          "hipMalloc down");
    check(hipMalloc(&d_rows, assignments * sizeof(std::uint32_t)),
          "hipMalloc rows");
    check(hipMalloc(&d_route_weights, assignments * sizeof(float)),
          "hipMalloc route weights");
    check(hipMalloc(&d_gate_up_temp,
                    kMaxRowsPerBatch * kGateUpOutputs * sizeof(float)),
          "hipMalloc gate_up temp");
    check(hipMalloc(&d_intermediate_temp,
                    kMaxRowsPerBatch * kIntermediate * sizeof(float)),
          "hipMalloc intermediate temp");
    check(hipMalloc(&d_output, reference.size() * sizeof(float)),
          "hipMalloc output");

    check(hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy hidden");
    check(hipMemcpy(d_gate_up, gate_up.data(),
                    gate_up.size() * sizeof(Q8Block),
                    hipMemcpyHostToDevice), "copy gate_up");
    check(hipMemcpy(d_down, down.data(),
                    down.size() * sizeof(Q8Block),
                    hipMemcpyHostToDevice), "copy down");

    const auto output_bytes = reference.size() * sizeof(float);

    check(hipMemcpy(d_rows, direct_rows.data(),
                    assignments * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice), "copy direct rows");
    check(hipMemcpy(d_route_weights, direct_weights.data(),
                    assignments * sizeof(float),
                    hipMemcpyHostToDevice), "copy direct weights");
    check(hipMemset(d_output, 0, output_bytes), "clear direct output");
    for (const auto& batch : direct_batches) {
        dispatch_batch(
            batch, d_hidden, d_rows, d_route_weights,
            d_gate_up, d_down, d_gate_up_temp, d_intermediate_temp,
            d_output, nullptr);
    }
    check(hipGetLastError(), "direct packed MoE launch");
    check(hipDeviceSynchronize(), "direct sync");

    std::vector<float> direct_output(reference.size());
    check(hipMemcpy(direct_output.data(), d_output, output_bytes,
                    hipMemcpyDeviceToHost), "copy direct output");

    const double direct_ms =
        benchmark_batches(
            options.iterations, direct_batches,
            d_hidden, d_rows, d_route_weights,
            d_gate_up, d_down, d_gate_up_temp, d_intermediate_temp,
            d_output, output_bytes);

    check(hipMemcpy(d_rows, grouped_rows.data(),
                    assignments * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice), "copy grouped rows");
    check(hipMemcpy(d_route_weights, grouped_weights.data(),
                    assignments * sizeof(float),
                    hipMemcpyHostToDevice), "copy grouped weights");
    check(hipMemset(d_output, 0, output_bytes), "clear grouped output");
    for (const auto& batch : grouped_batches) {
        dispatch_batch(
            batch, d_hidden, d_rows, d_route_weights,
            d_gate_up, d_down, d_gate_up_temp, d_intermediate_temp,
            d_output, nullptr);
    }
    check(hipGetLastError(), "grouped packed MoE launch");
    check(hipDeviceSynchronize(), "grouped sync");

    std::vector<float> grouped_output(reference.size());
    check(hipMemcpy(grouped_output.data(), d_output, output_bytes,
                    hipMemcpyDeviceToHost), "copy grouped output");

    const double grouped_ms =
        benchmark_batches(
            options.iterations, grouped_batches,
            d_hidden, d_rows, d_route_weights,
            d_gate_up, d_down, d_gate_up_temp, d_intermediate_temp,
            d_output, output_bytes);

    const float direct_error =
        max_abs_error(direct_output, reference);
    const float grouped_error =
        max_abs_error(grouped_output, reference);
    const float path_error =
        max_abs_error(direct_output, grouped_output);

    const auto gate_weight_bytes_per_expert =
        static_cast<std::uint64_t>(kGateUpOutputs) *
        (kHidden / kBlockElements) * sizeof(Q8Block);
    const auto down_weight_bytes_per_expert =
        static_cast<std::uint64_t>(kHidden) *
        (kIntermediate / kBlockElements) * sizeof(Q8Block);
    const auto expert_weight_bytes =
        gate_weight_bytes_per_expert + down_weight_bytes_per_expert;

    const auto direct_weight_bytes =
        static_cast<std::uint64_t>(direct_batches.size()) *
        expert_weight_bytes;
    const auto grouped_weight_bytes =
        static_cast<std::uint64_t>(grouped_batches.size()) *
        expert_weight_bytes;

    std::cout << "{\n"
              << "  \"schema\": \"strix.moe-q8-reuse.v1\",\n"
              << "  \"fixture\": \"" << kFixtureId << "\",\n"
              << "  \"device\": \"" << properties.name << "\",\n"
              << "  \"gfx_arch\": \"" << properties.gcnArchName << "\",\n"
              << "  \"warp_size\": " << properties.warpSize << ",\n"
              << "  \"hidden\": " << kHidden << ",\n"
              << "  \"intermediate\": " << kIntermediate << ",\n"
              << "  \"rows\": " << options.rows << ",\n"
              << "  \"experts\": " << options.experts << ",\n"
              << "  \"top_k\": " << options.top_k << ",\n"
              << "  \"iterations\": " << options.iterations << ",\n"
              << "  \"assignments\": " << assignments << ",\n"
              << "  \"direct_batches\": " << direct_batches.size() << ",\n"
              << "  \"grouped_batches\": " << grouped_batches.size() << ",\n"
              << "  \"expert_weight_bytes\": " << expert_weight_bytes << ",\n"
              << "  \"direct_modeled_weight_bytes\": "
              << direct_weight_bytes << ",\n"
              << "  \"grouped_modeled_weight_bytes\": "
              << grouped_weight_bytes << ",\n"
              << "  \"modeled_weight_bytes_saved\": "
              << (direct_weight_bytes - grouped_weight_bytes) << ",\n"
              << "  \"grouped_assignments_per_weight_traversal\": "
              << (grouped_batches.empty() ? 0.0 : static_cast<double>(assignments) / static_cast<double>(grouped_batches.size())) << ",\n"
              << "  \"direct_ms\": " << direct_ms << ",\n"
              << "  \"grouped_ms\": " << grouped_ms << ",\n"
              << "  \"speedup\": "
              << (grouped_ms > 0.0 ? direct_ms / grouped_ms : 0.0) << ",\n"
              << "  \"max_direct_reference_error\": "
              << direct_error << ",\n"
              << "  \"max_grouped_reference_error\": "
              << grouped_error << ",\n"
              << "  \"max_direct_grouped_error\": "
              << path_error << "\n"
              << "}\n";

    hipFree(d_output);
    hipFree(d_intermediate_temp);
    hipFree(d_gate_up_temp);
    hipFree(d_route_weights);
    hipFree(d_rows);
    hipFree(d_down);
    hipFree(d_gate_up);
    hipFree(d_hidden);

    return direct_error <= 5e-3F &&
                   grouped_error <= 5e-3F &&
                   path_error <= 5e-3F
               ? 0
               : 2;
} catch (const std::exception& error) {
    std::cerr << "strix-moe-q8-0: " << error.what() << '\n';
    return 1;
}
