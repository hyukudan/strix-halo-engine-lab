// Native operator fixture for sm_120a. No model files are required.
#include "k1bind_cases.hpp"
#include "cuda/core/dense/bf16_rows.hpp"
#ifdef STRIX_TEST_K1_RECIPE
#include "exec/catalog_cuda_decode.hpp"
#include "exec/cuda_program_recipe.hpp"
#include "cuda/core/dense/dense_mma.hpp"
#include <cstring>
#include <string_view>
#endif

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

#ifdef STRIX_TEST_K1_RECIPE
void host_recipe() {
    namespace cd = strix::exec::cuda_decode;
    using strix::core::Bf16RowProjection;
    alignas(16) float inputs[2][32]{}, outputs[3][16]{};
    alignas(16) std::uint16_t w[32 * 16]{};
    strix::device::Profile profile{};
    profile.units = 1;
    strix::exec::CudaProgramRecipe recipe;
    {
        strix::core::Bf16ProjectionBatch batch;
        const std::array first{Bf16RowProjection{w, 16, outputs[0]}, Bf16RowProjection{w, 7, outputs[1]}};
        const std::array second{Bf16RowProjection{w, 16, outputs[2]}};
        if (!batch.append(inputs[0], 32, first) || !batch.append(inputs[1], 32, second) ||
            !batch.try_round_store(outputs[0], 16)) throw std::runtime_error("host projection collection");
        for (auto& leaf : cd::bind_projection_batch(batch, profile)) recipe.kernel(std::move(leaf), true);
        const Bf16RowProjection overlap{w, 16, outputs[0] + 4};
        if (!batch.append(inputs[0], 32, std::span(&overlap, 1))) throw std::runtime_error("host overlap setup");
        bool refused = false;
        try { (void)cd::bind_projection_batch(batch, profile); }
        catch (const std::invalid_argument&) { refused = true; }
        if (!refused) throw std::runtime_error("accepted overlapping outputs");
    }
    const auto compiled = recipe.compile({}, strix::exec::CudaReadback::resident);
    const auto& p = compiled.program;
    if (p.launches.size() != 2 || p.predicted.launches != 2 || compiled.binding.nodes.size() != 2 ||
        p.launches[0].entry != "cuda-sm120-bf16-r1-fp32-group-v1" ||
        p.launches[1].entry != "cuda-sm120-bf16-r1-mma3-independent-v1" ||
        !p.launches[1].chain_identity.ends_with("rf") || p.launches[1].after != 0 ||
        p.launches[1].args_bytes != sizeof(cc::MmaGroupTable)) throw std::runtime_error("incorrect compiled leaves");
    cc::MmaGroupTable args{};
    std::memcpy(&args, p.args.data() + p.launches[1].args_offset, sizeof(args));
    if (args.tiles.count != 2 || args.tiles.total() != 2 ||
        args.jobs[0].x != inputs[0] || args.jobs[1].x != inputs[1] ||
        args.jobs[0].y != outputs[0] || args.jobs[1].y != outputs[2] ||
        !args.jobs[0].round_bf16 || args.jobs[1].round_bf16)
        throw std::runtime_error("incorrect owned independent MMA arguments");
    std::puts("k1 recipe host PASS owned inputs, output aliases, mixed rounds, actual compiled leaves");
}

void recipe(std::span<const cc::Bf16DenseRowGroup> groups, std::span<const ProjectionBuffers> buffers,
            const strix::device::Profile& profile, const k1bind_test::Case& test, cudaStream_t stream) {
    namespace cd = strix::exec::cuda_decode;
    strix::exec::CudaProgramRecipe recipe;
    std::vector<unsigned> widths;
    {
        strix::core::Bf16ProjectionBatch batch;
        for (const auto& group : groups) {
            if (!batch.append(group.input, group.input_width, group.projections)) throw std::runtime_error("append");
        }
        for (const auto& group : groups) {
            for (const auto& p : group.projections) {
                const auto j = widths.size();
                widths.push_back(p.outputs);
                // Round selected MMA results; compare to a separate reference round.
                if (j % 2 == 0 && batch.try_round_store(p.output, p.outputs)) {
                    const auto tail = cd::bind(cd::Bf16Round{buffers[j].incumbent.get(), p.outputs});
                    tail.fn(tail.args, tail.grid, stream);
                }
                check(cudaMemsetAsync(p.output, 0x80, (std::size_t{p.outputs} + 1) * sizeof(float), stream));
            }
        }
        for (auto& leaf : cd::bind_projection_batch(batch, profile)) recipe.kernel(std::move(leaf), true);
    } // The recipe must have copied every launch argument before this point.
    const auto compiled = recipe.compile({}, strix::exec::CudaReadback::resident);
    const auto& program = compiled.program;
    if (program.launches.size() != test.planned_launches || program.predicted.launches != test.planned_launches ||
        compiled.binding.nodes.size() != test.planned_launches) throw std::runtime_error("unexpected recipe leaf count");
    check(cudaStreamSynchronize(stream));
    capture_and_run(stream, program.launches.size(), [&] {
        for (const auto& leaf : program.launches) {
            if (leaf.kernels != 1) throw std::runtime_error("incorrect leaf launch receipt");
            leaf.fn(std::span(program.args).subspan(leaf.args_offset, leaf.args_bytes), leaf.grid, stream);
        }
        return cudaSuccess;
    });
    for (unsigned j = 0; j < widths.size(); ++j) compare(buffers[j], widths[j], j);
    std::printf("k1 recipe %s PASS native-leaves=%zu\n", test.name, program.launches.size());
}
#endif

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
#ifdef STRIX_TEST_K1_RECIPE
        if (rows == 1) recipe(std::span(planned_groups).first(test.groups.size()), std::span(buffers).first(next),
                             profile, test, stream);
#endif
        std::printf("k1bind %s-r%u PASS launches=%zu->%zu\n", test.name, rows,
                    test.incumbent_launches, test.planned_launches);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef STRIX_TEST_K1_RECIPE
        if (argc == 2 && std::string_view(argv[1]) == "--host-only") { host_recipe(); return 0; }
#else
        (void)argv;
#endif
        if (argc != 1) throw std::invalid_argument("unexpected fixture arguments");
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
