#include "cuda/core/dense/rows_bf16_narrow.cuh"
#include "cuda/core/mix_cuda_latency.cuh"
#include "strix/device/runtime.hpp"
#include "strix/exec/catalog.hpp"

#include <algorithm>
#include <bit>
#include <initializer_list>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace mix = strix::cuda::mix;
namespace dense = strix::cuda_core;
namespace residual = strix::core::residual_mix;
namespace native = strix::device::native;
using native::check;

constexpr float guard = -1234.25F;
constexpr unsigned padding = 4U;
constexpr auto old_bank_id = "cuda-sm120-residual-bank-softmax-v1";
constexpr auto resident_bank_id = "cuda-sm120-residual-bank-mix-r1-v1";
constexpr auto new_bank_id = "cuda-sm120-residual-bank-softmax-tiled-r1-v1";
constexpr auto old_dense_id = "cuda-sm120-bf16-r1-mma3-group-round-v1";
constexpr auto new_dense_id = "cuda-sm120-bf16-r1-mma3-output-split-round-v1";

// A pageable cudaMemcpy may return once the source is staged, before its DMA lands, and the legacy stream it uses
// does not order against the fixture's non-blocking stream. Every seed therefore completes before any arm reads or
// writes the image (f1: the batch arm's last-seeded canary image landed after the kernel stored its rows).
template <class T>
void upload(const native::Buffer<T>& to, const std::vector<T>& values) {
    check(cudaMemcpy(to.data(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "seed operands");
    check(cudaDeviceSynchronize(), "seeded image resident");
}

std::vector<float> read(const native::Buffer<float>& from) {
    std::vector<float> values(from.span().size());
    check(cudaMemcpy(values.data(), from.data(), values.size() * sizeof(float), cudaMemcpyDeviceToHost), "read fixture output");
    return values;
}

bool same(float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

void compare(const native::Buffer<float>& a, const native::Buffer<float>& b, unsigned width, unsigned images) {
    const auto x = read(a), y = read(b);
    for (unsigned q = 0; q < images; ++q) {
        const auto base = std::size_t{q} * (width + 2U * padding);
        for (unsigned i = 0; i < width + 2U * padding; ++i) {
            const bool canary = i < padding || i >= width + padding;
            if (!same(x[base + i], y[base + i]) || same(x[base + i], guard) != canary) {
                std::fprintf(stderr, "image %u slot %u: %08x != %08x or canary changed\n", q, i,
                             std::bit_cast<std::uint32_t>(x[base + i]), std::bit_cast<std::uint32_t>(y[base + i]));
                throw std::runtime_error("MIXLAT bit comparison");
            }
        }
    }
}

std::uint32_t next(std::uint32_t& seed) { return seed = seed * 1664525U + 1013904223U; }

std::vector<float> samples(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    for (auto& value : values)
        value = static_cast<float>(static_cast<int>(next(seed) >> 9U) - 4194304) * 0x1.38b7a6p-22F;
    return values;
}

struct BankCase {
    BankCase(unsigned width, unsigned banks, unsigned queries, bool normalized, cudaStream_t stream)
        : sources(std::size_t{width} * banks), learned(std::size_t{width} * queries) {
        upload(sources, samples(sources.span().size(), 1U));
        upload(learned, samples(learned.span().size(), 2U));
        args = {.queries = learned.data(), .rows = 1, .width = width, .bank_count = banks,
                .query_count = queries, .epsilon = 1e-6F, .normalized = normalized};
        for (unsigned b = 0; b < banks; ++b) {
            auto* row = sources.data() + std::size_t{b} * width;
            args.banks[b] = row;
            if (normalized)
                check(residual::launch_normalize({row, row, 1, width, args.epsilon}, stream), "normalize test bank");
        }
    }

    void enqueue(std::string_view identity, float* output, cudaStream_t stream) const {
        auto call = args;
        for (unsigned q = 0; q < args.query_count; ++q)
            call.output[q] = output + std::size_t{q} * (args.width + 2U * padding) + padding;
        cudaError_t status = cudaErrorInvalidValue;
        if (identity == old_bank_id) status = mix::enqueue(call, stream);
        if (identity == resident_bank_id) status = residual::launch_mix(call, stream);
        if (identity == new_bank_id) status = mix::enqueue_tiled(call, stream);
        check(status, "enqueue bank test arm");
    }

    native::Buffer<float> sources, learned;
    residual::MixArgs args{};
};

struct DenseCase {
    DenseCase(unsigned in, unsigned out, std::uint32_t seed)
        : weights(std::size_t{in} * out), input(in),
          weight{weights.data(), dense::WeightFormat::bf16, out, in} {
        std::vector<std::uint16_t> values(weights.span().size());
        for (auto& value : values) {
            const auto bits = next(seed);
            value = static_cast<std::uint16_t>((bits >> 16U & 0x8000U) | (0x3d00U + (bits >> 24U)));
        }
        upload(weights, values);
        upload(input, samples(in, seed));
    }

    void enqueue(std::string_view identity, float* output, cudaStream_t stream) const {
        auto* y = output + padding;
        if (identity == old_dense_id)
            check(dense::dense_bf16_mma3_rows_round(&weight, &y, 1, input.data(), 1U, stream), "enqueue incumbent MMA3");
        else if (identity == new_dense_id)
            check(dense::dense_bf16_mma3_r1_output_split(weight, y, input.data(), stream), "enqueue split-output MMA3");
        else throw std::invalid_argument("unknown dense bench identity");
    }

    native::Buffer<std::uint16_t> weights;
    native::Buffer<float> input;
    dense::DenseWeight weight;
};

void compare_bank(unsigned width, unsigned banks, unsigned queries, bool normalized, cudaStream_t stream) {
    const BankCase test(width, banks, queries, normalized, stream);
    const auto size = std::size_t{queries} * (width + 2U * padding);
    native::Buffer<float> old(size), resident(size), tiled(size);
    const std::vector<float> canaries(size, guard);
    upload(old, canaries);
    upload(resident, canaries);
    upload(tiled, canaries);
    test.enqueue(old_bank_id, old.data(), stream);
    test.enqueue(resident_bank_id, resident.data(), stream);
    test.enqueue(new_bank_id, tiled.data(), stream);
    native::synchronize(stream);
    compare(old, tiled, width, queries);
    compare(resident, tiled, width, queries);
}

// Residual bank batch: every job (set, index) against the single-query incumbent K8c readout and the single-query
// tiled readout of the same query row. Set outputs are padded images, so the canaries catch any stray store.
void compare_batch(unsigned width, unsigned banks, std::initializer_list<unsigned> counts, bool normalized,
                   cudaStream_t stream) {
    unsigned jobs = 0;
    for (const auto count : counts) jobs += count;
    const BankCase test(width, banks, jobs, normalized, stream);
    const auto image = std::size_t{width} + 2U * padding;
    const auto size = std::size_t{jobs} * image;
    native::Buffer<float> batch(size), old(size), tiled(size);
    const std::vector<float> canaries(size, guard);
    upload(batch, canaries);
    upload(old, canaries);
    upload(tiled, canaries);
    residual::BankBatchArgs args{.width = width, .bank_count = banks, .epsilon = test.args.epsilon,
                                 .normalized = normalized};
    for (unsigned b = 0; b < banks; ++b) args.banks[b] = test.args.banks[b];
    // Set s writes rows [first_s, first_s + count_s) of one [jobs][width] image; misplaced rows differ from the
    // single-query arms after the per-job gather below.
    native::Buffer<float> rows(std::size_t{jobs} * width);
    upload(rows, std::vector<float>(rows.span().size(), guard));
    unsigned first = 0;
    for (const auto count : counts) {
        const auto offset = std::size_t{first} * width;
        args.sets[args.set_count++] = {test.learned.data() + offset, rows.data() + offset, count};
        first += count;
    }
    // The bank normalization above runs on the fixture stream: it completes before the arms run.
    check(cudaDeviceSynchronize(), "seeded images resident");
    check(mix::enqueue_batch(args, stream), "enqueue bank batch");
    for (unsigned job = 0; job < jobs; ++job) {
        auto single = residual::job_args(args, job);
        single.output[0] = old.data() + job * image + padding;
        check(mix::enqueue(single, stream), "enqueue single-query K8c");
        single.output[0] = tiled.data() + job * image + padding;
        check(mix::enqueue_tiled(single, stream), "enqueue single-query tiled");
        check(cudaMemcpyAsync(batch.data() + job * image + padding, residual::job_args(args, job).output[0],
                              std::size_t{width} * sizeof(float), cudaMemcpyDeviceToDevice, stream), "gather batch row");
    }
    native::synchronize(stream);
    compare(old, batch, width, jobs);
    compare(tiled, batch, width, jobs);
}

void compare_dense(unsigned in, unsigned out, std::uint32_t seed, cudaStream_t stream) {
    const DenseCase test(in, out, seed);
    const auto size = out + 2U * padding;
    native::Buffer<float> old(size), split(size);
    const std::vector<float> canaries(size, guard);
    upload(old, canaries);
    upload(split, canaries);
    test.enqueue(old_dense_id, old.data(), stream);
    test.enqueue(new_dense_id, split.data(), stream);
    native::synchronize(stream);
    compare(old, split, out, 1U);
}

// Entry cases execute independently of X6's RMS-only dispatch. The same
// operand builders and arm launchers are used for timing and conformance.
struct Replay {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaEvent_t begin{}, end{};
    ~Replay() {
        if (executable) (void)cudaGraphExecDestroy(executable);
        if (graph) (void)cudaGraphDestroy(graph);
        if (begin) (void)cudaEventDestroy(begin);
        if (end) (void)cudaEventDestroy(end);
    }
};

void bench_case(std::string_view id, const char* shape, unsigned elements, const auto& operands,
                cudaStream_t stream, const cudaDeviceProp& device, const char* power) {
    constexpr unsigned nodes = 128U;
    native::Buffer<float> output(elements);
    Replay replay{};
    check(cudaEventCreate(&replay.begin), "create begin event");
    check(cudaEventCreate(&replay.end), "create end event");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture bench entry");
    check(cudaEventRecord(replay.begin, stream), "record begin");
    for (unsigned i = 0; i < nodes; ++i) operands.enqueue(id, output.data(), stream);
    check(cudaEventRecord(replay.end, stream), "record end");
    check(cudaStreamEndCapture(stream, &replay.graph), "end capture");
    check(cudaGraphInstantiate(&replay.executable, replay.graph, 0ULL), "instantiate bench graph");
    check(cudaGraphLaunch(replay.executable, stream), "warm up");
    native::synchronize(stream);
    float best = std::numeric_limits<float>::infinity();
    for (unsigned round = 0; round < 16U; ++round) {
        check(cudaGraphLaunch(replay.executable, stream), "replay bench entry");
        native::synchronize(stream);
        float ms{};
        check(cudaEventElapsedTime(&ms, replay.begin, replay.end), "read graph interval");
        best = std::min(best, ms);
    }
    std::printf("BASE %.*s device=\"%s\" power=\"%s\" shape=%s graph_nodes=%u min_graph_us_per_node=%.3f\n",
                static_cast<int>(id.size()), id.data(), device.name, power, shape, nodes,
                static_cast<double>(best) * 1000.0 / nodes);
}

void bench_entries(std::string_view id, const char* power, cudaStream_t stream) {
    using namespace strix::exec;
    int ordinal{};
    cudaDeviceProp properties{};
    check(cudaGetDevice(&ordinal), "read device ordinal");
    check(cudaGetDeviceProperties(&properties, ordinal), "read device description");
    if (id != resident_bank_id &&
        std::ranges::none_of(Catalog::builtin_entries(Backend::cuda), [&](const auto& e) { return e.identity == id; }))
        throw std::invalid_argument("unknown CUDA catalogue identity");
    if (id == old_dense_id || id == new_dense_id) {
        const DenseCase test(4608, 64, 9U);
        bench_case(id, "M1-K4608-N64", 64 + 2 * padding, test, stream, properties, power);
    } else if (id == old_bank_id || id == resident_bank_id || id == new_bank_id) {
        const BankCase normalized(2304, 2, 1, true, stream);
        bench_case(id, "R1-W2304-B2-Q1-normalized", 2304 + 2 * padding, normalized, stream, properties, power);
        const BankCase raw(2304, 5, 3, false, stream);
        bench_case(id, "R1-W2304-B5-Q3-raw", 3 * (2304 + 2 * padding), raw, stream, properties, power);
    } else throw std::invalid_argument("no MIXLAT bench case for identity");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && (argc != 6 || std::string_view(argv[1]) != "entry" ||
                         std::string_view(argv[2]) != "--entry" || std::string_view(argv[4]) != "--power-state"))
            throw std::invalid_argument("usage: strix-mixlat-cuda-fixture [entry --entry ID --power-state STATE]");
        native::StreamOwner stream;
        if (argc != 1) {
            bench_entries(argv[3], argv[5], stream.get());
            return 0;
        }
        for (unsigned banks = 1; banks <= 5; ++banks) {
            compare_bank(2304, banks, 1, false, stream.get());
            compare_bank(2304, banks, 3, false, stream.get());
        }
        compare_bank(2304, 2, 1, true, stream.get());
        compare_bank(513, 8, 3, false, stream.get());
        compare_bank(4096, 8, 8, true, stream.get());
        std::puts("PASS MIXLAT bank tiled bitwise vs K8c and resident CUDA");
        compare_batch(2304, 2, {24, 24}, true, stream.get());
        compare_batch(2304, 2, {1}, true, stream.get());
        compare_batch(2304, 5, {3, 1, 7}, false, stream.get());
        compare_batch(513, 8, {2, 2, 2, 2}, false, stream.get());
        compare_batch(4096, 8, {9}, true, stream.get());
        std::puts("PASS MIXLAT bank batch bitwise vs single-query K8c and tiled");
        for (std::uint32_t seed = 1; seed <= 5; ++seed)
            compare_dense(4608, 64, seed, stream.get());
        compare_dense(32, 16, 6U, stream.get());
        compare_dense(160, 48, 7U, stream.get());
        std::puts("PASS MIXLAT BF16 MMA3 output split bitwise vs group_round");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MIXLAT %s\n", error.what());
        return 1;
    }
}
