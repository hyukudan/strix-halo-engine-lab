// Native operator fixture for sm_120a. No model files are required.
#include "k1bind_cases.hpp"
#include "cuda_core/dense/bf16_rows.hpp"

#include <bit>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace cc = strix::cuda_core;

namespace {

constexpr unsigned kFixtureRows = 8;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
using Buffer = cc::UniqueHandle<T*, cudaFree>;

template <class T>
Buffer<T> allocate(std::size_t count) {
    Buffer<T> buffer;
    check(cudaMalloc(buffer.put(), count * sizeof(T)));
    return buffer;
}

template <class T>
Buffer<T> upload(const std::vector<T>& host) {
    auto buffer = allocate<T>(host.size());
    check(cudaMemcpy(buffer.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return buffer;
}

template <class Submit>
void capture_and_run(cudaStream_t stream, std::size_t expected_launches, const Submit& submit) {
    cc::UniqueHandle<cudaGraph_t, cudaGraphDestroy> graph;
    cc::UniqueHandle<cudaGraphExec_t, cudaGraphExecDestroy> exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const auto submitted = submit();
    const auto captured = cudaStreamEndCapture(stream, graph.put());
    check(submitted);
    check(captured);
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph.get(), nullptr, &count));
    if (count != expected_launches) throw std::runtime_error("unexpected graph node count");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph.get(), nodes.data(), &count));
    for (const auto node : nodes) {
        cudaGraphNodeType type{};
        check(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) throw std::runtime_error("unexpected non-kernel graph node");
    }
    check(cudaGraphInstantiate(exec.put(), graph.get(), 0));
    check(cudaGraphLaunch(exec.get(), stream));
    check(cudaStreamSynchronize(stream));
}

struct ProjectionBuffers {
    Buffer<std::uint16_t> weights;
    Buffer<float> incumbent, planned;
};

void compare(const ProjectionBuffers& job, std::size_t count, unsigned member) {
    std::vector<float> incumbent(count + 1), planned(count + 1);
    const auto bytes = incumbent.size() * sizeof(float);
    check(cudaMemcpy(incumbent.data(), job.incumbent.get(), bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(planned.data(), job.planned.get(), bytes, cudaMemcpyDeviceToHost));
    const auto a = std::bit_cast<std::uint32_t>(incumbent.back());
    const auto b = std::bit_cast<std::uint32_t>(planned.back());
    if (a != 0x7f7f7f7fU || b != 0x80808080U) throw std::runtime_error("output guard modified");
    for (std::size_t i = 0; i < count; ++i) {
        const auto expected = std::bit_cast<std::uint32_t>(incumbent[i]);
        const auto actual = std::bit_cast<std::uint32_t>(planned[i]);
        if (expected == actual) continue;
        std::fprintf(stderr, "job=%u element=%zu incumbent=%08x planned=%08x\n", member, i, expected, actual);
        throw std::runtime_error("output bits differ");
    }
}

void run(const k1bind_test::Case& test, const strix::device::Profile& profile, cudaStream_t stream) {
    constexpr auto capacity = strix::core::rows::kMaxJobs;
    using Projection = strix::core::Bf16RowProjection;
    std::mt19937 rng(0x5be198a1U);
    std::array<Buffer<float>, 3> inputs;
    std::array<ProjectionBuffers, capacity> buffers;
    std::array<Projection, capacity> incumbent{}, planned{};
    std::array<cc::Bf16DenseRowGroup, 3> incumbent_groups{}, planned_groups{};
    unsigned next = 0;
    for (std::size_t i = 0; i < test.groups.size(); ++i) {
        const auto& group = test.groups[i];
        auto& input = inputs[group.input];
        if (!input.get()) {
            std::vector<float> host(std::size_t{kFixtureRows} * group.width);
            for (auto& value : host)
                value = static_cast<float>(static_cast<int>(rng() % 16000001U) - 8000000) * 0x1p-24f;
            input = upload(host);
        }
        const auto first = next;
        for (const auto& projection : group.projections) {
            auto& job = buffers[next];
            std::vector<std::uint16_t> host(std::size_t{projection.out} * group.width);
            for (auto& value : host) {
                const auto sign = rng() & 0x8000U;
                value = static_cast<std::uint16_t>(sign | (0x3a00U + rng() % 1536U));
            }
            job.weights = upload(host);
            const auto size = std::size_t{kFixtureRows} * projection.out + 1;
            job.incumbent = allocate<float>(size);
            job.planned = allocate<float>(size);
            incumbent[next] = {job.weights.get(), projection.out, job.incumbent.get(), projection.round_bf16};
            planned[next] = {job.weights.get(), projection.out, job.planned.get(), projection.round_bf16};
            ++next;
        }
        incumbent_groups[i] = {input.get(), group.width, std::span(incumbent).subspan(first, next - first)};
        planned_groups[i] = {input.get(), group.width, std::span(planned).subspan(first, next - first)};
    }

    for (unsigned rows = 1; rows <= kFixtureRows; ++rows) {
        for (unsigned j = 0; j < next; ++j) {
            const auto size = (std::size_t{rows} * incumbent[j].outputs + 1) * sizeof(float);
            check(cudaMemsetAsync(buffers[j].incumbent.get(), 0x7f, size, stream));
            check(cudaMemsetAsync(buffers[j].planned.get(), 0x80, size, stream));
        }
        check(cudaStreamSynchronize(stream));
        capture_and_run(stream, test.incumbent_launches, [&] {
            for (const auto& group : std::span(incumbent_groups).first(test.groups.size())) {
                const auto status = cc::enqueue_bf16_dense_rows(group.input, group.input_width, rows, group.projections,
                                                                stream, cc::DenseArith::mma3, nullptr);
                if (status != cudaSuccess) return status;
            }
            return cudaSuccess;
        });
        capture_and_run(stream, test.planned_launches, [&] {
            if (test.groups.size() != 1)
                return cc::enqueue_bf16_dense_row_groups(std::span(planned_groups).first(test.groups.size()),
                                                         rows, profile, stream);
            const auto& group = planned_groups.front();
            return cc::enqueue_bf16_dense_rows(group.input, group.input_width, rows, group.projections,
                                               stream, cc::DenseArith::mma3, nullptr, &profile);
        });
        for (unsigned j = 0; j < next; ++j) compare(buffers[j], std::size_t{rows} * incumbent[j].outputs, j);
        std::printf("k1bind %s-r%u PASS launches=%zu->%zu\n", test.name, rows,
                    test.incumbent_launches, test.planned_launches);
    }
}

}  // namespace

int main() {
    try {
        int device = 0, units = 0;
        check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&units, cudaDevAttrMultiProcessorCount, device));
        strix::device::Profile profile{};
        profile.units = static_cast<unsigned>(units);
        cc::UniqueHandle<cudaStream_t, cudaStreamDestroy> stream;
        check(cudaStreamCreateWithFlags(stream.put(), cudaStreamNonBlocking));
        for (const auto& test : k1bind_test::cases) run(test, profile, stream.get());
    } catch (const std::exception& err) {
        std::fprintf(stderr, "k1bind FAIL: %s\n", err.what());
        return 1;
    }
}
