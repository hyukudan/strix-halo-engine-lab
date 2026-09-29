#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;
constexpr int kValueTile = 32;
constexpr int kTiles = kValueDim / kValueTile;
constexpr int kThreads = 128;

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(error));
    }
}

struct Options {
    int tokens = 4;
    int iterations = 100;
    int heads = 4;
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
        if (arg == "--tokens") {
            options.tokens = parse_positive(next("--tokens"), "--tokens");
        } else if (arg == "--iterations") {
            options.iterations = parse_positive(next("--iterations"), "--iterations");
        } else if (arg == "--heads") {
            options.heads = parse_positive(next("--heads"), "--heads");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.tokens != 1 && options.tokens != 2 &&
        options.tokens != 4 && options.tokens != 8) {
        throw std::invalid_argument("--tokens must be 1, 2, 4 or 8");
    }
    return options;
}

template <int Tokens>
__global__ __launch_bounds__(kThreads)
void gdn_window_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    int heads,
    float* __restrict__ state,
    float* __restrict__ output) {
    const int lane = static_cast<int>(threadIdx.x);
    const int tile = static_cast<int>(blockIdx.x) % kTiles;
    const int head = static_cast<int>(blockIdx.x) / kTiles;
    const int value_base = tile * kValueTile;

    // 128 lanes x 32 value columns = 4096 floats = 16 KiB LDS.
    __shared__ float reduction[kThreads * kValueTile];

    float retained[kValueTile];
    #pragma unroll
    for (int column = 0; column < kValueTile; ++column) {
        const int value_index = value_base + column;
        const std::size_t state_index =
            (static_cast<std::size_t>(head) * kKeyDim + lane) * kValueDim +
            value_index;
        retained[column] = state[state_index];
    }

    #pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float d = decay[static_cast<std::size_t>(token) *
                              heads + head];
        const float b = beta[static_cast<std::size_t>(token) *
                             heads + head];
        const float k = key[(static_cast<std::size_t>(token) *
                             heads + head) * kKeyDim + lane];
        const float q = query[(static_cast<std::size_t>(token) *
                               heads + head) * kKeyDim + lane];

        #pragma unroll
        for (int column = 0; column < kValueTile; ++column) {
            retained[column] *= d;
            reduction[lane * kValueTile + column] = retained[column] * k;
        }
        __syncthreads();

        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                #pragma unroll
                for (int column = 0; column < kValueTile; ++column) {
                    reduction[lane * kValueTile + column] +=
                        reduction[(lane + stride) * kValueTile + column];
                }
            }
            __syncthreads();
        }

        float delta[kValueTile];
        if (lane == 0) {
            #pragma unroll
            for (int column = 0; column < kValueTile; ++column) {
                const int value_index = value_base + column;
                const float target =
                    value[(static_cast<std::size_t>(token) *
                           heads + head) * kValueDim + value_index];
                delta[column] =
                    (target - reduction[column]) * b;
                reduction[column] = delta[column];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int column = 0; column < kValueTile; ++column) {
            const float token_delta = reduction[column];
            retained[column] += k * token_delta;
            reduction[lane * kValueTile + column] = retained[column] * q;
        }
        __syncthreads();

        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                #pragma unroll
                for (int column = 0; column < kValueTile; ++column) {
                    reduction[lane * kValueTile + column] +=
                        reduction[(lane + stride) * kValueTile + column];
                }
            }
            __syncthreads();
        }

        if (lane == 0) {
            #pragma unroll
            for (int column = 0; column < kValueTile; ++column) {
                const int value_index = value_base + column;
                output[(static_cast<std::size_t>(token) *
                        heads + head) * kValueDim + value_index] =
                    reduction[column];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int column = 0; column < kValueTile; ++column) {
        const int value_index = value_base + column;
        const std::size_t state_index =
            (static_cast<std::size_t>(head) * kKeyDim + lane) * kValueDim +
            value_index;
        state[state_index] = retained[column];
    }
}

void cpu_reference(
    int tokens,
    int heads,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& decay,
    const std::vector<float>& beta,
    std::vector<float>& state,
    std::vector<float>& output) {
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            const auto th = static_cast<std::size_t>(token) * heads + head;
            const float d = decay[th];
            const float b = beta[th];

            for (int value_index = 0; value_index < kValueDim; ++value_index) {
                float remembered = 0.0F;
                for (int key_index = 0; key_index < kKeyDim; ++key_index) {
                    const auto state_index =
                        (static_cast<std::size_t>(head) * kKeyDim + key_index) *
                            kValueDim +
                        value_index;
                    state[state_index] *= d;
                    remembered += state[state_index] *
                        key[th * kKeyDim + key_index];
                }

                const float delta =
                    (value[th * kValueDim + value_index] - remembered) * b;

                float result = 0.0F;
                for (int key_index = 0; key_index < kKeyDim; ++key_index) {
                    const auto state_index =
                        (static_cast<std::size_t>(head) * kKeyDim + key_index) *
                            kValueDim +
                        value_index;
                    state[state_index] +=
                        key[th * kKeyDim + key_index] * delta;
                    result += state[state_index] *
                        query[th * kKeyDim + key_index];
                }
                output[th * kValueDim + value_index] = result;
            }
        }
    }
}

template <int Tokens>
void launch_window(
    int heads,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    float* state,
    float* output,
    hipStream_t stream) {
    const dim3 launch_grid(static_cast<unsigned>(heads * kTiles), 1U, 1U);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gdn_window_kernel<Tokens>),
        launch_grid,
        dim3(kThreads),
        0,
        stream,
        query, key, value, decay, beta, heads, state, output);
}

template <int Tokens>
double benchmark_window(
    int iterations,
    int heads,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    float* state,
    float* output,
    const std::vector<float>& initial_state) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate(begin)");
    check(hipEventCreate(&end), "hipEventCreate(end)");

    double total_ms = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check(hipMemcpy(state, initial_state.data(),
                        initial_state.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "reset state");
        check(hipEventRecord(begin), "hipEventRecord(begin)");
        launch_window<Tokens>(
            heads, query, key, value, decay, beta, state, output, nullptr);
        check(hipGetLastError(), "gdn window launch");
        check(hipEventRecord(end), "hipEventRecord(end)");
        check(hipEventSynchronize(end), "hipEventSynchronize(end)");
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, begin, end), "hipEventElapsedTime");
        total_ms += static_cast<double>(ms);
    }

    hipEventDestroy(begin);
    hipEventDestroy(end);
    return total_ms / static_cast<double>(iterations);
}

double benchmark_serial(
    int tokens,
    int iterations,
    int heads,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    float* state,
    float* output,
    const std::vector<float>& initial_state) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate(begin)");
    check(hipEventCreate(&end), "hipEventCreate(end)");

    double total_ms = 0.0;
    const std::size_t q_stride = static_cast<std::size_t>(heads) * kKeyDim;
    const std::size_t v_stride = static_cast<std::size_t>(heads) * kValueDim;
    const std::size_t scalar_stride = static_cast<std::size_t>(heads);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        check(hipMemcpy(state, initial_state.data(),
                        initial_state.size() * sizeof(float),
                        hipMemcpyHostToDevice),
              "reset state");
        check(hipEventRecord(begin), "hipEventRecord(begin)");
        for (int token = 0; token < tokens; ++token) {
            launch_window<1>(
                heads,
                query + static_cast<std::size_t>(token) * q_stride,
                key + static_cast<std::size_t>(token) * q_stride,
                value + static_cast<std::size_t>(token) * v_stride,
                decay + static_cast<std::size_t>(token) * scalar_stride,
                beta + static_cast<std::size_t>(token) * scalar_stride,
                state,
                output + static_cast<std::size_t>(token) * v_stride,
                nullptr);
            check(hipGetLastError(), "serial gdn launch");
        }
        check(hipEventRecord(end), "hipEventRecord(end)");
        check(hipEventSynchronize(end), "hipEventSynchronize(end)");
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, begin, end), "hipEventElapsedTime");
        total_ms += static_cast<double>(ms);
    }

    hipEventDestroy(begin);
    hipEventDestroy(end);
    return total_ms / static_cast<double>(iterations);
}

template <int Tokens>
void dispatch_window(
    int heads,
    const float* query,
    const float* key,
    const float* value,
    const float* decay,
    const float* beta,
    float* state,
    float* output) {
    launch_window<Tokens>(
        heads, query, key, value, decay, beta, state, output, nullptr);
    check(hipGetLastError(), "validation window launch");
    check(hipDeviceSynchronize(), "validation window sync");
}

float max_abs_error(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("error vectors differ in size");
    }
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

    const auto state_elements =
        static_cast<std::size_t>(options.heads) * kKeyDim * kValueDim;
    const auto q_elements =
        static_cast<std::size_t>(options.tokens) * options.heads * kKeyDim;
    const auto v_elements =
        static_cast<std::size_t>(options.tokens) * options.heads * kValueDim;
    const auto scalar_elements =
        static_cast<std::size_t>(options.tokens) * options.heads;

    std::vector<float> initial_state(state_elements);
    std::vector<float> query(q_elements);
    std::vector<float> key(q_elements);
    std::vector<float> value(v_elements);
    std::vector<float> decay(scalar_elements);
    std::vector<float> beta(scalar_elements);

    for (std::size_t i = 0; i < initial_state.size(); ++i) {
        initial_state[i] =
            static_cast<float>(static_cast<int>(i % 37U) - 18) * 0.0005F;
    }
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] =
            static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.01F;
        key[i] =
            static_cast<float>(static_cast<int>(i % 13U) - 6) * 0.01F;
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] =
            static_cast<float>(static_cast<int>(i % 29U) - 14) * 0.02F;
    }
    for (std::size_t i = 0; i < scalar_elements; ++i) {
        decay[i] = 0.97F - static_cast<float>(i % 3U) * 0.01F;
        beta[i] = 0.2F + static_cast<float>(i % 5U) * 0.03F;
    }

    // Normalize q/k on the host so this lab isolates state update traffic.
    for (int token = 0; token < options.tokens; ++token) {
        for (int head = 0; head < options.heads; ++head) {
            const auto base =
                (static_cast<std::size_t>(token) * options.heads + head) * kKeyDim;
            double q_norm = 1e-6;
            double k_norm = 1e-6;
            for (int i = 0; i < kKeyDim; ++i) {
                q_norm += static_cast<double>(query[base + i]) * query[base + i];
                k_norm += static_cast<double>(key[base + i]) * key[base + i];
            }
            q_norm = std::sqrt(q_norm);
            k_norm = std::sqrt(k_norm);
            const float q_scale =
                static_cast<float>(1.0 / (q_norm * std::sqrt(kKeyDim)));
            const float k_scale = static_cast<float>(1.0 / k_norm);
            for (int i = 0; i < kKeyDim; ++i) {
                query[base + i] *= q_scale;
                key[base + i] *= k_scale;
            }
        }
    }

    std::vector<float> reference_state = initial_state;
    std::vector<float> reference_output(v_elements);
    cpu_reference(
        options.tokens, options.heads,
        query, key, value, decay, beta,
        reference_state, reference_output);

    float *d_query{}, *d_key{}, *d_value{}, *d_decay{}, *d_beta{};
    float *d_state{}, *d_output{};
    check(hipMalloc(&d_query, query.size() * sizeof(float)), "hipMalloc query");
    check(hipMalloc(&d_key, key.size() * sizeof(float)), "hipMalloc key");
    check(hipMalloc(&d_value, value.size() * sizeof(float)), "hipMalloc value");
    check(hipMalloc(&d_decay, decay.size() * sizeof(float)), "hipMalloc decay");
    check(hipMalloc(&d_beta, beta.size() * sizeof(float)), "hipMalloc beta");
    check(hipMalloc(&d_state, initial_state.size() * sizeof(float)), "hipMalloc state");
    check(hipMalloc(&d_output, reference_output.size() * sizeof(float)), "hipMalloc output");

    check(hipMemcpy(d_query, query.data(), query.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy query");
    check(hipMemcpy(d_key, key.data(), key.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy key");
    check(hipMemcpy(d_value, value.data(), value.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy value");
    check(hipMemcpy(d_decay, decay.data(), decay.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy decay");
    check(hipMemcpy(d_beta, beta.data(), beta.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy beta");
    check(hipMemcpy(d_state, initial_state.data(),
                    initial_state.size() * sizeof(float),
                    hipMemcpyHostToDevice), "copy initial state");

    switch (options.tokens) {
    case 1:
        dispatch_window<1>(options.heads, d_query, d_key, d_value,
                           d_decay, d_beta, d_state, d_output);
        break;
    case 2:
        dispatch_window<2>(options.heads, d_query, d_key, d_value,
                           d_decay, d_beta, d_state, d_output);
        break;
    case 4:
        dispatch_window<4>(options.heads, d_query, d_key, d_value,
                           d_decay, d_beta, d_state, d_output);
        break;
    case 8:
        dispatch_window<8>(options.heads, d_query, d_key, d_value,
                           d_decay, d_beta, d_state, d_output);
        break;
    default:
        throw std::logic_error("unsupported token count");
    }

    std::vector<float> gpu_state(state_elements);
    std::vector<float> gpu_output(v_elements);
    check(hipMemcpy(gpu_state.data(), d_state,
                    gpu_state.size() * sizeof(float),
                    hipMemcpyDeviceToHost), "copy state back");
    check(hipMemcpy(gpu_output.data(), d_output,
                    gpu_output.size() * sizeof(float),
                    hipMemcpyDeviceToHost), "copy output back");

    const float state_error = max_abs_error(gpu_state, reference_state);
    const float output_error = max_abs_error(gpu_output, reference_output);

    const double serial_ms = benchmark_serial(
        options.tokens, options.iterations, options.heads,
        d_query, d_key, d_value, d_decay, d_beta,
        d_state, d_output, initial_state);

    double window_ms = 0.0;
    switch (options.tokens) {
    case 1:
        window_ms = benchmark_window<1>(
            options.iterations, options.heads, d_query, d_key, d_value,
            d_decay, d_beta, d_state, d_output, initial_state);
        break;
    case 2:
        window_ms = benchmark_window<2>(
            options.iterations, options.heads, d_query, d_key, d_value,
            d_decay, d_beta, d_state, d_output, initial_state);
        break;
    case 4:
        window_ms = benchmark_window<4>(
            options.iterations, options.heads, d_query, d_key, d_value,
            d_decay, d_beta, d_state, d_output, initial_state);
        break;
    case 8:
        window_ms = benchmark_window<8>(
            options.iterations, options.heads, d_query, d_key, d_value,
            d_decay, d_beta, d_state, d_output, initial_state);
        break;
    }

    const std::uint64_t state_bytes =
        static_cast<std::uint64_t>(state_elements * sizeof(float));
    const std::uint64_t serial_state_bytes =
        state_bytes * static_cast<std::uint64_t>(options.tokens) * 2ULL;
    const std::uint64_t retained_state_bytes = state_bytes * 2ULL;

    hipFuncAttributes attributes{};
    switch (options.tokens) {
    case 1:
        check(hipFuncGetAttributes(
                  &attributes,
                  reinterpret_cast<const void*>(gdn_window_kernel<1>)),
              "hipFuncGetAttributes");
        break;
    case 2:
        check(hipFuncGetAttributes(
                  &attributes,
                  reinterpret_cast<const void*>(gdn_window_kernel<2>)),
              "hipFuncGetAttributes");
        break;
    case 4:
        check(hipFuncGetAttributes(
                  &attributes,
                  reinterpret_cast<const void*>(gdn_window_kernel<4>)),
              "hipFuncGetAttributes");
        break;
    case 8:
        check(hipFuncGetAttributes(
                  &attributes,
                  reinterpret_cast<const void*>(gdn_window_kernel<8>)),
              "hipFuncGetAttributes");
        break;
    }

    std::cout << "{
"
              << "  \"device\": \"" << properties.name << "\",
"
              << "  \"gfx_arch\": \"" << properties.gcnArchName << "\",
"
              << "  \"warp_size\": " << properties.warpSize << ",
"
              << "  \"heads\": " << options.heads << ",
"
              << "  \"tokens\": " << options.tokens << ",
"
              << "  \"key_dim\": " << kKeyDim << ",
"
              << "  \"value_dim\": " << kValueDim << ",
"
              << "  \"value_tile\": " << kValueTile << ",
"
              << "  \"serial_ms\": " << serial_ms << ",
"
              << "  \"retained_window_ms\": " << window_ms << ",
"
              << "  \"speedup\": "
              << (window_ms > 0.0 ? serial_ms / window_ms : 0.0) << ",
"
              << "  \"serial_state_bytes\": " << serial_state_bytes << ",
"
              << "  \"retained_state_bytes\": " << retained_state_bytes << ",
"
              << "  \"state_bytes_avoided\": "
              << (serial_state_bytes - retained_state_bytes) << ",
"
              << "  \"max_state_error\": " << state_error << ",
"
              << "  \"max_output_error\": " << output_error << ",
"
              << "  \"kernel_local_bytes\": " << attributes.localSizeBytes << ",
"
              << "  \"kernel_shared_bytes\": " << attributes.sharedSizeBytes << ",
"
              << "  \"kernel_registers\": " << attributes.numRegs << ",
"
              << "  \"max_threads_per_block\": " << attributes.maxThreadsPerBlock
              << "
}
";

    hipFree(d_output);
    hipFree(d_state);
    hipFree(d_beta);
    hipFree(d_decay);
    hipFree(d_value);
    hipFree(d_key);
    hipFree(d_query);

    return (state_error <= 5e-4F && output_error <= 5e-4F) ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "strix-recurrent-gdn: " << error.what() << '\n';
    return 1;
}
