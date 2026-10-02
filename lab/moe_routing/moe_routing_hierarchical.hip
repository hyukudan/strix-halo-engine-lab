#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kMaxExperts = 512;
constexpr int kThreads = 512;
constexpr int kWaveWidth = 32;
constexpr int kMaxGroups = kMaxExperts / kWaveWidth;
constexpr int kMaxTopK = 16;
constexpr std::string_view kFixtureId = "deterministic-cutoff-tie-v1";

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
    }
}

struct Options {
    int rows = 4;
    int experts = 512;
    int top_k = 10;
    int iterations = 500;
    bool normalize_selected = true;
};

int parse_positive(std::string_view text, const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(std::string(name) + " is empty");
    }
    int value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(std::string(name) + " must be an integer");
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
        } else if (arg == "--no-normalize-selected") {
            options.normalize_selected = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.rows != 1 && options.rows != 2 &&
        options.rows != 4 && options.rows != 8) {
        throw std::invalid_argument("--rows must be 1, 2, 4 or 8");
    }
    if (options.experts <= 0 || options.experts > kMaxExperts) {
        throw std::invalid_argument("--experts must be within 1..512");
    }
    if (options.top_k <= 0 || options.top_k > options.experts ||
        options.top_k > kMaxTopK) {
        throw std::invalid_argument("--top-k is out of supported range");
    }
    return options;
}

__device__ bool better_pair(
    float left_value, int left_index,
    float right_value, int right_index) {
    return left_value > right_value ||
           (left_value == right_value && left_index < right_index);
}

__global__ void route_hierarchical_wave32_kernel(
    const float* __restrict__ logits,
    int experts,
    int top_k,
    bool normalize_selected,
    int* __restrict__ selected_indices,
    float* __restrict__ selected_weights) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int lane_in_group = lane % kWaveWidth;
    const int group = lane / kWaveWidth;
    const int expert = lane;

    __shared__ float scratch[kThreads];
    __shared__ float probabilities[kMaxExperts];
    __shared__ float group_values[kMaxGroups];
    __shared__ int group_indices[kMaxGroups];
    __shared__ float selected_values[kMaxTopK];

    float local_max = -INFINITY;
    if (expert < experts) {
        local_max = logits[static_cast<std::size_t>(row) * experts + expert];
    }
    scratch[lane] = local_max;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];

    float local_sum = 0.0F;
    if (expert < experts) {
        const float probability =
            expf(logits[static_cast<std::size_t>(row) * experts + expert] - maximum);
        probabilities[expert] = probability;
        local_sum = probability;
    }
    scratch[lane] = local_sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    const float denominator = scratch[0];

    if (expert < experts) {
        probabilities[expert] /= denominator;
    }
    __syncthreads();

    const int groups = (experts + kWaveWidth - 1) / kWaveWidth;
    for (int slot = 0; slot < top_k; ++slot) {
        float group_value =
            expert < experts ? probabilities[expert] : -INFINITY;
        int group_index = expert < experts ? expert : experts;

        for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down(group_value, offset, kWaveWidth);
            const int other_index =
                __shfl_down(group_index, offset, kWaveWidth);
            if (lane_in_group + offset < kWaveWidth &&
                better_pair(other_value, other_index, group_value, group_index)) {
                group_value = other_value;
                group_index = other_index;
            }
        }

        if (lane_in_group == 0) {
            group_values[group] = group_value;
            group_indices[group] = group_index;
        }
        __syncthreads();

        if (group == 0) {
            float winner_value =
                lane_in_group < groups ? group_values[lane_in_group] : -INFINITY;
            int winner_index =
                lane_in_group < groups ? group_indices[lane_in_group] : experts;

            for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
                const float other_value =
                    __shfl_down(winner_value, offset, kWaveWidth);
                const int other_index =
                    __shfl_down(winner_index, offset, kWaveWidth);
                if (lane_in_group + offset < kWaveWidth &&
                    better_pair(
                        other_value, other_index, winner_value, winner_index)) {
                    winner_value = other_value;
                    winner_index = other_index;
                }
            }

            if (lane_in_group == 0) {
                const auto output =
                    static_cast<std::size_t>(row) * top_k + slot;
                selected_indices[output] = winner_index;
                selected_values[slot] = winner_value;
                probabilities[winner_index] = -INFINITY;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        float selected_sum = 1.0F;
        if (normalize_selected) {
            selected_sum = 0.0F;
            for (int slot = 0; slot < top_k; ++slot) {
                selected_sum += selected_values[slot];
            }
        }
        for (int slot = 0; slot < top_k; ++slot) {
            const auto output =
                static_cast<std::size_t>(row) * top_k + slot;
            selected_weights[output] = selected_values[slot] / selected_sum;
        }
    }
}

void cpu_reference(
    int rows,
    int experts,
    int top_k,
    bool normalize_selected,
    const std::vector<float>& logits,
    std::vector<int>& indices,
    std::vector<float>& weights) {
    for (int row = 0; row < rows; ++row) {
        const auto base = static_cast<std::size_t>(row) * experts;
        const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(base);
        const auto end = begin + experts;
        const float maximum = *std::max_element(begin, end);

        std::vector<float> probabilities(static_cast<std::size_t>(experts));
        double denominator = 0.0;
        for (int expert = 0; expert < experts; ++expert) {
            const float value = std::exp(logits[base + expert] - maximum);
            probabilities[static_cast<std::size_t>(expert)] = value;
            denominator += value;
        }
        for (auto& probability : probabilities) {
            probability = static_cast<float>(
                static_cast<double>(probability) / denominator);
        }

        std::vector<int> order(static_cast<std::size_t>(experts));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + top_k, order.end(),
            [&](int left, int right) {
                const float left_value = probabilities[static_cast<std::size_t>(left)];
                const float right_value = probabilities[static_cast<std::size_t>(right)];
                return left_value != right_value ? left_value > right_value : left < right;
            });

        double selected_sum = normalize_selected ? 0.0 : 1.0;
        if (normalize_selected) {
            for (int slot = 0; slot < top_k; ++slot) {
                selected_sum += probabilities[
                    static_cast<std::size_t>(order[static_cast<std::size_t>(slot)])];
            }
        }
        for (int slot = 0; slot < top_k; ++slot) {
            const auto output = static_cast<std::size_t>(row) * top_k + slot;
            const int expert = order[static_cast<std::size_t>(slot)];
            indices[output] = expert;
            weights[output] = static_cast<float>(
                static_cast<double>(probabilities[static_cast<std::size_t>(expert)]) /
                selected_sum);
        }
    }
}

float max_abs_error(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    float result = 0.0F;
    for (std::size_t i = 0; i < left.size(); ++i) {
        result = std::max(result, std::abs(left[i] - right[i]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) try {
    const auto options = parse_args(argc, argv);

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    if (properties.warpSize != kWaveWidth) {
        throw std::runtime_error("hierarchical routing candidate requires Wave32");
    }
    if (properties.maxThreadsPerBlock < kThreads) {
        throw std::runtime_error("device cannot launch 512-thread routing blocks");
    }

    const auto logit_count =
        static_cast<std::size_t>(options.rows) * options.experts;
    const auto selected_count =
        static_cast<std::size_t>(options.rows) * options.top_k;

    std::vector<float> logits(logit_count);
    for (int row = 0; row < options.rows; ++row) {
        for (int expert = 0; expert < options.experts; ++expert) {
            const float wave =
                std::sin(static_cast<float>((row + 1) * (expert + 3)) * 0.017F);
            logits[static_cast<std::size_t>(row) * options.experts + expert] =
                wave + static_cast<float>(expert) * 1e-5F +
                static_cast<float>(row) * 1e-4F;
        }
    }
    if (options.experts > options.top_k) {
        for (int expert = 0; expert < options.experts; ++expert) {
            logits[static_cast<std::size_t>(expert)] =
                -static_cast<float>(expert);
        }
        logits[static_cast<std::size_t>(options.top_k)] =
            logits[static_cast<std::size_t>(options.top_k - 1)];
    }

    std::vector<int> reference_indices(selected_count);
    std::vector<float> reference_weights(selected_count);
    cpu_reference(
        options.rows, options.experts, options.top_k,
        options.normalize_selected, logits,
        reference_indices, reference_weights);

    float* d_logits = nullptr;
    int* d_indices = nullptr;
    float* d_weights = nullptr;
    check(hipMalloc(&d_logits, logits.size() * sizeof(float)), "hipMalloc logits");
    check(hipMalloc(&d_indices, selected_count * sizeof(int)), "hipMalloc indices");
    check(hipMalloc(&d_weights, selected_count * sizeof(float)), "hipMalloc weights");
    check(
        hipMemcpy(
            d_logits, logits.data(), logits.size() * sizeof(float),
            hipMemcpyHostToDevice),
        "copy logits");

    route_hierarchical_wave32_kernel<<<options.rows, kThreads>>>(
        d_logits, options.experts, options.top_k,
        options.normalize_selected, d_indices, d_weights);
    check(hipGetLastError(), "hierarchical route launch");
    check(hipDeviceSynchronize(), "hierarchical route sync");

    std::vector<int> gpu_indices(selected_count);
    std::vector<float> gpu_weights(selected_count);
    check(
        hipMemcpy(
            gpu_indices.data(), d_indices, selected_count * sizeof(int),
            hipMemcpyDeviceToHost),
        "copy indices");
    check(
        hipMemcpy(
            gpu_weights.data(), d_weights, selected_count * sizeof(float),
            hipMemcpyDeviceToHost),
        "copy weights");

    const bool exact_indices = gpu_indices == reference_indices;
    const float weight_error =
        max_abs_error(gpu_weights, reference_weights);

    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");
    check(hipEventRecord(begin), "hipEventRecord begin");
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        route_hierarchical_wave32_kernel<<<options.rows, kThreads>>>(
            d_logits, options.experts, options.top_k,
            options.normalize_selected, d_indices, d_weights);
    }
    check(hipGetLastError(), "hierarchical timing launches");
    check(hipEventRecord(end), "hipEventRecord end");
    check(hipEventSynchronize(end), "hipEventSynchronize end");
    float total_ms = 0.0F;
    check(hipEventElapsedTime(&total_ms, begin, end), "hipEventElapsedTime");

    hipFuncAttributes attributes{};
    check(
        hipFuncGetAttributes(
            &attributes,
            reinterpret_cast<const void*>(route_hierarchical_wave32_kernel)),
        "hipFuncGetAttributes");

    std::cout << "{\n"
              << "  \"device\": \"" << properties.name << "\",\n"
              << "  \"gfx_arch\": \"" << properties.gcnArchName << "\",\n"
              << "  \"fixture\": \"" << kFixtureId << "\",\n"
              << "  \"warp_size\": " << properties.warpSize << ",\n"
              << "  \"rows\": " << options.rows << ",\n"
              << "  \"experts\": " << options.experts << ",\n"
              << "  \"top_k\": " << options.top_k << ",\n"
              << "  \"iterations\": " << options.iterations << ",\n"
              << "  \"normalize_selected\": "
              << (options.normalize_selected ? "true" : "false") << ",\n"
              << "  \"group_width\": " << kWaveWidth << ",\n"
              << "  \"groups\": "
              << ((options.experts + kWaveWidth - 1) / kWaveWidth) << ",\n"
              << "  \"local_candidate_scans\": "
              << (options.experts * options.top_k) << ",\n"
              << "  \"group_reduction_candidates\": "
              << (((options.experts + kWaveWidth - 1) / kWaveWidth) *
                  options.top_k) << ",\n"
              << "  \"average_us\": "
              << (static_cast<double>(total_ms) * 1000.0 /
                  static_cast<double>(options.iterations)) << ",\n"
              << "  \"exact_indices\": "
              << (exact_indices ? "true" : "false") << ",\n"
              << "  \"max_weight_error\": " << weight_error << ",\n"
              << "  \"kernel_local_bytes\": " << attributes.localSizeBytes << ",\n"
              << "  \"kernel_shared_bytes\": " << attributes.sharedSizeBytes << ",\n"
              << "  \"kernel_registers\": " << attributes.numRegs << ",\n"
              << "  \"max_threads_per_block\": "
              << attributes.maxThreadsPerBlock << "\n"
              << "}\n";

    hipEventDestroy(begin);
    hipEventDestroy(end);
    hipFree(d_weights);
    hipFree(d_indices);
    hipFree(d_logits);

    return exact_indices && weight_error <= 5e-6F ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "strix-moe-routing-hierarchical: " << error.what() << '\n';
    return 1;
}
