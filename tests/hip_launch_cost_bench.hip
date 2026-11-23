// No-model bench of the per-launch cost on the HIP device (FABLE-ARCH package 9).
//
// It separates four costs that decide what fewer launches (fusion) and run-ahead / bound launch arrays can save:
//   stream issue gap  : dependent small kernels back to back on one stream,
//   graph replay gap  : the same chain captured once and replayed,
//   host issue rate   : host time to enqueue a chain (is the stream host bound?),
//   host sync trip    : launch + stream synchronize / event synchronize per step.
// The fused_* cases run the same total work inside one launch, so (chain - fused) / n is the cost of a boundary.
// Prints one `launch_cost case=... us_per=...` line per case (median of the repetitions) and `launch_cost: DONE`.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define LC_CHECK(expr)                                                                                     \
    do {                                                                                                   \
        const hipError_t lc_status = (expr);                                                               \
        if (lc_status != hipSuccess) {                                                                     \
            std::fprintf(stderr, "launch_cost: HIP error %s at %s:%d\n", hipGetErrorString(lc_status),     \
                         __FILE__, __LINE__);                                                              \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

constexpr int kTinyFloats = 1024;   // 4 KiB row: one norm / round / mix leaf
constexpr int kWideBlocks = 40;     // one block per compute unit on the reference APU class
constexpr int kWideFloats = 16384;  // 64 KiB per launch across the wide grid

__global__ void empty_kernel() {}

// One block of 256 threads: read, scale and write a 4 KiB row (dependent chain through the same buffer).
__global__ void tiny_kernel(float* row) {
    for (int i = static_cast<int>(threadIdx.x); i < kTinyFloats; i += static_cast<int>(blockDim.x)) {
        row[i] = row[i] * 0.999f + 1.0e-3f;
    }
}

__global__ void tiny_fused_kernel(float* row, int steps) {
    for (int s = 0; s < steps; ++s) {
        for (int i = static_cast<int>(threadIdx.x); i < kTinyFloats; i += static_cast<int>(blockDim.x)) {
            row[i] = row[i] * 0.999f + 1.0e-3f;
        }
        __syncthreads();
    }
}

// Wide grid: every compute unit busy for a short leaf (GEMV-tail sized).
__global__ void wide_kernel(float* data) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    for (int j = i; j < kWideFloats; j += static_cast<int>(gridDim.x * blockDim.x)) {
        data[j] = data[j] * 0.999f + 1.0e-3f;
    }
}

struct Stats {
    double median;
    double lo;
    double hi;
};

Stats summarise(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return Stats{v[v.size() / 2], v.front(), v.back()};
}

void print(const char* name, int n, const std::vector<double>& per_launch_us) {
    const Stats s = summarise(per_launch_us);
    std::printf("launch_cost case=%s n=%d reps=%zu us_per=%.3f min=%.3f max=%.3f\n", name, n, per_launch_us.size(),
                s.median, s.lo, s.hi);
}

template <class Enqueue>
double device_us(hipStream_t stream, hipEvent_t a, hipEvent_t b, Enqueue enqueue) {
    LC_CHECK(hipEventRecord(a, stream));
    enqueue();
    LC_CHECK(hipEventRecord(b, stream));
    LC_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    LC_CHECK(hipEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms) * 1000.0;
}

double host_us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 7;
    if (n < 16 || reps < 3) {
        std::fprintf(stderr, "usage: strix-hip-launch-cost-bench [launches>=16] [reps>=3]\n");
        return 1;
    }
    hipDeviceProp_t prop{};
    LC_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("launch_cost device=%s cus=%d\n", prop.gcnArchName, prop.multiProcessorCount);

    hipStream_t stream{};
    LC_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipEvent_t a{}, b{};
    LC_CHECK(hipEventCreate(&a));
    LC_CHECK(hipEventCreate(&b));
    float* row = nullptr;
    float* wide = nullptr;
    LC_CHECK(hipMalloc(&row, sizeof(float) * kTinyFloats));
    LC_CHECK(hipMalloc(&wide, sizeof(float) * kWideFloats));
    LC_CHECK(hipMemset(row, 0, sizeof(float) * kTinyFloats));
    LC_CHECK(hipMemset(wide, 0, sizeof(float) * kWideFloats));

    auto chain_empty = [&] { for (int i = 0; i < n; ++i) empty_kernel<<<1, 32, 0, stream>>>(); };
    auto chain_tiny = [&] { for (int i = 0; i < n; ++i) tiny_kernel<<<1, 256, 0, stream>>>(row); };
    auto chain_wide = [&] { for (int i = 0; i < n; ++i) wide_kernel<<<kWideBlocks, 256, 0, stream>>>(wide); };
    auto fused_tiny = [&] { tiny_fused_kernel<<<1, 256, 0, stream>>>(row, n); };

    // Warm up code objects and clocks.
    for (int w = 0; w < 3; ++w) {
        (void)device_us(stream, a, b, chain_empty);
        (void)device_us(stream, a, b, chain_tiny);
        (void)device_us(stream, a, b, chain_wide);
        (void)device_us(stream, a, b, fused_tiny);
    }

    const double dn = static_cast<double>(n);
    std::vector<double> v_empty, v_tiny, v_wide, v_fused, v_issue, v_g_empty, v_g_tiny, v_g_wide, v_sync, v_event;
    for (int r = 0; r < reps; ++r) {
        v_empty.push_back(device_us(stream, a, b, chain_empty) / dn);
        v_tiny.push_back(device_us(stream, a, b, chain_tiny) / dn);
        v_wide.push_back(device_us(stream, a, b, chain_wide) / dn);
        v_fused.push_back(device_us(stream, a, b, fused_tiny) / dn);
        // Host issue rate: enqueue only (the queue drains afterwards, outside the timed region).
        const auto t0 = std::chrono::steady_clock::now();
        chain_tiny();
        v_issue.push_back(host_us_since(t0) / dn);
        LC_CHECK(hipStreamSynchronize(stream));
    }
    print("stream_empty", n, v_empty);
    print("stream_tiny", n, v_tiny);
    print("stream_wide40", n, v_wide);
    print("fused_tiny", n, v_fused);
    print("host_issue_tiny", n, v_issue);

    // Graph replay of the same chains.
    auto replay = [&](auto chain, std::vector<double>& out) {
        hipGraph_t graph{};
        hipGraphExec_t exec{};
        LC_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        chain();
        LC_CHECK(hipStreamEndCapture(stream, &graph));
        LC_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int w = 0; w < 2; ++w) (void)device_us(stream, a, b, [&] { LC_CHECK(hipGraphLaunch(exec, stream)); });
        for (int r = 0; r < reps; ++r) {
            out.push_back(device_us(stream, a, b, [&] { LC_CHECK(hipGraphLaunch(exec, stream)); }) / dn);
        }
        LC_CHECK(hipGraphExecDestroy(exec));
        LC_CHECK(hipGraphDestroy(graph));
    };
    replay(chain_empty, v_g_empty);
    replay(chain_tiny, v_g_tiny);
    replay(chain_wide, v_g_wide);
    print("graph_empty", n, v_g_empty);
    print("graph_tiny", n, v_g_tiny);
    print("graph_wide40", n, v_g_wide);

    // Host round trips: one tiny launch then a blocking wait, as a step with a host sync would do.
    const int trips = std::max(16, n / 8);
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < trips; ++i) {
            tiny_kernel<<<1, 256, 0, stream>>>(row);
            LC_CHECK(hipStreamSynchronize(stream));
        }
        v_sync.push_back(host_us_since(t0) / static_cast<double>(trips));
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < trips; ++i) {
            tiny_kernel<<<1, 256, 0, stream>>>(row);
            LC_CHECK(hipEventRecord(b, stream));
            LC_CHECK(hipEventSynchronize(b));
        }
        v_event.push_back(host_us_since(t0) / static_cast<double>(trips));
    }
    print("sync_trip_tiny", trips, v_sync);
    print("event_trip_tiny", trips, v_event);

    const double gap_stream = summarise(v_tiny).median - summarise(v_fused).median;
    const double gap_graph = summarise(v_g_tiny).median - summarise(v_fused).median;
    std::printf("launch_cost boundary_us stream=%.3f graph=%.3f sync_extra=%.3f\n", gap_stream, gap_graph,
                summarise(v_sync).median - summarise(v_tiny).median);

    LC_CHECK(hipFree(row));
    LC_CHECK(hipFree(wide));
    LC_CHECK(hipEventDestroy(a));
    LC_CHECK(hipEventDestroy(b));
    LC_CHECK(hipStreamDestroy(stream));
    std::printf("launch_cost: DONE\n");
    return 0;
}
