#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kHidden = 64;
constexpr int kIntermediate = 32;
constexpr int kThreads = 64;
constexpr int kMaxExperts = 32;
constexpr int kMaxTopK = 8;

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(error));
    }
}

struct Options {
    int rows = 4;
    int experts = 16;
    int top_k = 4;
    int iterations = 200;
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
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto next = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
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
    if (options.experts <= 0 || options.experts > kMaxExperts) {
        throw std::invalid_argument("--experts is out of range");
    }
    if (options.top_k <= 0 || options.top_k > options.experts ||
        options.top_k > kMaxTopK) {
        throw std::invalid_argument("--top-k is out of range");
    }
    return options;
}

__device__ float device_silu(float value) {
    return value / (1.0F + expf(-value));
}

float host_silu(float value) {
    return value / (1.0F + std::exp(-value));
}

__global__ __launch_bounds__(kThreads)
void expert_assignment_kernel(
    const float* hidden,
    const float* gate_up,
    const float* down,
    const int* assignment_rows,
    const int* assignment_experts,
    const float* assignment_weights,
    int assignments,
    float* output) {
    const int assignment = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (assignment >= assignments) {
        return;
    }

    const int row = assignment_rows[assignment];
    const int expert = assignment_experts[assignment];
    const float route_weight = assignment_weights[assignment];
    __shared__ float intermediate[kIntermediate];

    if (lane < kIntermediate) {
        float gate = 0.0F;
        float up = 0.0F;
        const auto hidden_base = static_cast<std::size_t>(row) * kHidden;
        const auto expert_base =
            static_cast<std::size_t>(expert) * 2U * kIntermediate * kHidden;
        const auto gate_base =
            expert_base + static_cast<std::size_t>(lane) * kHidden;
        const auto up_base =
            expert_base + static_cast<std::size_t>(kIntermediate + lane) * kHidden;

        for (int feature = 0; feature < kHidden; ++feature) {
            const float x = hidden[hidden_base + feature];
            gate += gate_up[gate_base + feature] * x;
            up += gate_up[up_base + feature] * x;
        }
        intermediate[lane] = device_silu(gate) * up;
    }
    __syncthreads();

    if (lane < kHidden) {
        float result = 0.0F;
        const auto down_base =
            (static_cast<std::size_t>(expert) * kHidden + lane) * kIntermediate;
        for (int inner = 0; inner < kIntermediate; ++inner) {
            result += down[down_base + inner] * intermediate[inner];
        }
        atomicAdd(&output[static_cast<std::size_t>(row) * kHidden + lane],
                  result * route_weight);
    }
}

void cpu_reference(
    int rows,
    int top_k,
    const std::vector<float>& hidden,
    const std::vector<float>& gate_up,
    const std::vector<float>& down,
    const std::vector<int>& selected,
    const std::vector<float>& route_weights,
    std::vector<float>& output) {
    std::fill(output.begin(), output.end(), 0.0F);
    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * top_k + slot;
            const int expert = selected[assignment];
            std::array<float, kIntermediate> intermediate{};

            for (int inner = 0; inner < kIntermediate; ++inner) {
                float gate = 0.0F;
                float up = 0.0F;
                const auto expert_base =
                    static_cast<std::size_t>(expert) * 2U *
                    kIntermediate * kHidden;
                const auto gate_base =
                    expert_base + static_cast<std::size_t>(inner) * kHidden;
                const auto up_base =
                    expert_base +
                    static_cast<std::size_t>(kIntermediate + inner) * kHidden;
                for (int feature = 0; feature < kHidden; ++feature) {
                    const float x =
                        hidden[static_cast<std::size_t>(row) * kHidden + feature];
                    gate += gate_up[gate_base + feature] * x;
                    up += gate_up[up_base + feature] * x;
                }
                intermediate[inner] = host_silu(gate) * up;
            }

            for (int feature = 0; feature < kHidden; ++feature) {
                float result = 0.0F;
                const auto down_base =
                    (static_cast<std::size_t>(expert) * kHidden + feature) *
                    kIntermediate;
                for (int inner = 0; inner < kIntermediate; ++inner) {
                    result += down[down_base + inner] * intermediate[inner];
                }
                output[static_cast<std::size_t>(row) * kHidden + feature] +=
                    result * route_weights[assignment];
            }
        }
    }
}

void build_grouped(
    int rows,
    int top_k,
    int experts,
    const std::vector<int>& selected,
    const std::vector<float>& weights,
    std::vector<int>& grouped_rows,
    std::vector<int>& grouped_experts,
    std::vector<float>& grouped_weights,
    int& distinct_experts) {
    std::vector<std::vector<std::size_t>> per_expert(
        static_cast<std::size_t>(experts));
    for (int row = 0; row < rows; ++row) {
        for (int slot = 0; slot < top_k; ++slot) {
            const auto assignment =
                static_cast<std::size_t>(row) * top_k + slot;
            per_expert[static_cast<std::size_t>(selected[assignment])]
                .push_back(assignment);
        }
    }

    distinct_experts = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const auto& list = per_expert[static_cast<std::size_t>(expert)];
        if (list.empty()) {
            continue;
        }
        ++distinct_experts;
        for (const auto assignment : list) {
            grouped_rows.push_back(
                static_cast<int>(assignment / static_cast<std::size_t>(top_k)));
            grouped_experts.push_back(expert);
            grouped_weights.push_back(weights[assignment]);
        }
    }
}

float max_abs_error(const std::vector<float>& a,
                    const std::vector<float>& b) {
    float result = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

double benchmark(
    int iterations,
    int assignments,
    const float* hidden,
    const float* gate_up,
    const float* down,
    const int* rows,
    const int* experts,
    const float* weights,
    float* output,
    std::size_t output_bytes) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");

    check(hipEventRecord(begin), "hipEventRecord begin");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check(hipMemset(output, 0, output_bytes), "hipMemset output");
        expert_assignment_kernel<<<assignments, kThreads>>>(
            hidden, gate_up, down, rows, experts, weights, assignments, output);
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
    const int assignments = options.rows * options.top_k;

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");

    std::vector<float> hidden(
        static_cast<std::size_t>(options.rows) * kHidden);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] =
            static_cast<float>(static_cast<int>(i % 23U) - 11) * 0.01F;
    }

    std::vector<float> gate_up(
        static_cast<std::size_t>(options.experts) * 2U *
        kIntermediate * kHidden);
    std::vector<float> down(
        static_cast<std::size_t>(options.experts) *
        kHidden * kIntermediate);
    for (std::size_t i = 0; i < gate_up.size(); ++i) {
        gate_up[i] =
            static_cast<float>(static_cast<int>(i % 19U) - 9) * 0.002F;
    }
    for (std::size_t i = 0; i < down.size(); ++i) {
        down[i] =
            static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.002F;
    }

    std::vector<int> selected(static_cast<std::size_t>(assignments));
    std::vector<float> weights(static_cast<std::size_t>(assignments));
    for (int row = 0; row < options.rows; ++row) {
        float sum = 0.0F;
        for (int slot = 0; slot < options.top_k; ++slot) {
            const auto index =
                static_cast<std::size_t>(row) * options.top_k + slot;
            selected[index] = (row * 3 + slot * 5) % options.experts;
            weights[index] = 1.0F + static_cast<float>(slot);
            sum += weights[index];
        }
        for (int slot = 0; slot < options.top_k; ++slot) {
            const auto index =
                static_cast<std::size_t>(row) * options.top_k + slot;
            weights[index] /= sum;
        }
    }

    std::vector<float> reference(
        static_cast<std::size_t>(options.rows) * kHidden);
    cpu_reference(options.rows, options.top_k, hidden, gate_up, down,
                  selected, weights, reference);

    std::vector<int> direct_rows(static_cast<std::size_t>(assignments));
    for (int row = 0; row < options.rows; ++row) {
        for (int slot = 0; slot < options.top_k; ++slot) {
            direct_rows[static_cast<std::size_t>(row) * options.top_k + slot] =
                row;
        }
    }

    std::vector<int> grouped_rows;
    std::vector<int> grouped_experts;
    std::vector<float> grouped_weights;
    grouped_rows.reserve(static_cast<std::size_t>(assignments));
    grouped_experts.reserve(static_cast<std::size_t>(assignments));
    grouped_weights.reserve(static_cast<std::size_t>(assignments));
    int distinct_experts = 0;
    build_grouped(options.rows, options.top_k, options.experts,
                  selected, weights,
                  grouped_rows, grouped_experts, grouped_weights,
                  distinct_experts);

    float *d_hidden{}, *d_gate_up{}, *d_down{}, *d_weights{}, *d_output{};
    int *d_rows{}, *d_experts{};
    check(hipMalloc(&d_hidden, hidden.size() * sizeof(float)), "hipMalloc hidden");
    check(hipMalloc(&d_gate_up, gate_up.size() * sizeof(float)), "hipMalloc gate_up");
    check(hipMalloc(&d_down, down.size() * sizeof(float)), "hipMalloc down");
    check(hipMalloc(&d_rows, static_cast<std::size_t>(assignments) * sizeof(int)),
          "hipMalloc rows");
    check(hipMalloc(&d_experts, static_cast<std::size_t>(assignments) * sizeof(int)),
          "hipMalloc experts");
    check(hipMalloc(&d_weights, static_cast<std::size_t>(assignments) * sizeof(float)),
          "hipMalloc weights");
    check(hipMalloc(&d_output, reference.size() * sizeof(float)), "hipMalloc output");

    check(hipMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy hidden");
    check(hipMemcpy(d_gate_up, gate_up.data(), gate_up.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy gate_up");
    check(hipMemcpy(d_down, down.data(), down.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy down");

    const auto output_bytes = reference.size() * sizeof(float);
    std::vector<float> direct_output(reference.size());
    std::vector<float> grouped_output(reference.size());

    check(hipMemcpy(d_rows, direct_rows.data(),
                    static_cast<std::size_t>(assignments) * sizeof(int),
                    hipMemcpyHostToDevice), "copy direct rows");
    check(hipMemcpy(d_experts, selected.data(),
                    static_cast<std::size_t>(assignments) * sizeof(int),
                    hipMemcpyHostToDevice), "copy direct experts");
    check(hipMemcpy(d_weights, weights.data(),
                    static_cast<std::size_t>(assignments) * sizeof(float),
                    hipMemcpyHostToDevice), "copy direct weights");
    check(hipMemset(d_output, 0, output_bytes), "clear direct output");
    expert_assignment_kernel<<<assignments, kThreads>>>(
        d_hidden, d_gate_up, d_down,
        d_rows, d_experts, d_weights, assignments, d_output);
    check(hipDeviceSynchronize(), "direct sync");
    check(hipMemcpy(direct_output.data(), d_output, output_bytes,
                    hipMemcpyDeviceToHost), "copy direct output");
    const double direct_ms = benchmark(
        options.iterations, assignments, d_hidden, d_gate_up, d_down,
        d_rows, d_experts, d_weights, d_output, output_bytes);

    check(hipMemcpy(d_rows, grouped_rows.data(),
                    static_cast<std::size_t>(assignments) * sizeof(int),
                    hipMemcpyHostToDevice), "copy grouped rows");
    check(hipMemcpy(d_experts, grouped_experts.data(),
                    static_cast<std::size_t>(assignments) * sizeof(int),
                    hipMemcpyHostToDevice), "copy grouped experts");
    check(hipMemcpy(d_weights, grouped_weights.data(),
                    static_cast<std::size_t>(assignments) * sizeof(float),
                    hipMemcpyHostToDevice), "copy grouped weights");
    check(hipMemset(d_output, 0, output_bytes), "clear grouped output");
    expert_assignment_kernel<<<assignments, kThreads>>>(
        d_hidden, d_gate_up, d_down,
        d_rows, d_experts, d_weights, assignments, d_output);
    check(hipDeviceSynchronize(), "grouped sync");
    check(hipMemcpy(grouped_output.data(), d_output, output_bytes,
                    hipMemcpyDeviceToHost), "copy grouped output");
    const double grouped_ms = benchmark(
        options.iterations, assignments, d_hidden, d_gate_up, d_down,
        d_rows, d_experts, d_weights, d_output, output_bytes);

    const float direct_error = max_abs_error(direct_output, reference);
    const float grouped_error = max_abs_error(grouped_output, reference);
    const float path_error = max_abs_error(direct_output, grouped_output);

    hipFuncAttributes attributes{};
    check(hipFuncGetAttributes(
              &attributes,
              reinterpret_cast<const void*>(expert_assignment_kernel)),
          "hipFuncGetAttributes");

    std::cout << "{\n"
              << "  \"device\": \"" << properties.name << "\",\n"
              << "  \"gfx_arch\": \"" << properties.gcnArchName << "\",\n"
              << "  \"warp_size\": " << properties.warpSize << ",\n"
              << "  \"rows\": " << options.rows << ",\n"
              << "  \"experts\": " << options.experts << ",\n"
              << "  \"top_k\": " << options.top_k << ",\n"
              << "  \"assignments\": " << assignments << ",\n"
              << "  \"distinct_experts\": " << distinct_experts << ",\n"
              << "  \"direct_ms\": " << direct_ms << ",\n"
              << "  \"grouped_ms\": " << grouped_ms << ",\n"
              << "  \"max_direct_reference_error\": " << direct_error << ",\n"
              << "  \"max_grouped_reference_error\": " << grouped_error << ",\n"
              << "  \"max_direct_grouped_error\": " << path_error << ",\n"
              << "  \"kernel_shared_bytes\": " << attributes.sharedSizeBytes << ",\n"
              << "  \"kernel_registers\": " << attributes.numRegs << "\n"
              << "}\n";

    hipFree(d_output);
    hipFree(d_weights);
    hipFree(d_experts);
    hipFree(d_rows);
    hipFree(d_down);
    hipFree(d_gate_up);
    hipFree(d_hidden);

    return direct_error <= 1e-4F &&
                   grouped_error <= 1e-4F &&
                   path_error <= 1e-4F
               ? 0
               : 2;
} catch (const std::exception& error) {
    std::cerr << "strix-moe-experts: " << error.what() << '\n';
    return 1;
}
