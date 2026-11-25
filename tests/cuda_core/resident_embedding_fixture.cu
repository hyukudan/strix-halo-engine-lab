// Shared activation regression; no model, private kernel, or performance measurement.
#include "cuda/core/dense/bf16_rows.hpp"
#include "core/lookup_catalog.hpp"
#include "exec/catalog_cuda_decode.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
namespace cc = strix::cuda_core;
namespace cd = strix::exec::cuda_decode;
namespace sf = strix::weight::scalar_float;
constexpr unsigned vocabulary = 11, guard = 4;
constexpr std::uint32_t sentinel = 0x7f7f7f7fU;
constexpr float epsilon = 0x1p-10F;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class T> struct Buffer {
    explicit Buffer(std::size_t n) : count(n) { check(cudaMalloc(storage.put(), n * sizeof(T))); }
    T* data() const { return storage.get(); }
    void upload(const std::vector<T>& host) {
        require(host.size() == count, "upload size");
        check(cudaMemcpy(data(), host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> result(count);
        check(cudaMemcpy(result.data(), data(), count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    cc::UniqueHandle<T*, cudaFree> storage;
    std::size_t count;
};
void canaries(const std::vector<float>& values) {
    for (unsigned i = 0; i < guard; ++i)
        require(std::bit_cast<std::uint32_t>(values[i]) == sentinel &&
                std::bit_cast<std::uint32_t>(values[values.size() - guard + i]) == sentinel,
                "embedding output guard changed");
}
template <class Submit> void replay(cudaStream_t stream, unsigned expected, const Submit& submit) {
    cc::UniqueHandle<cudaGraph_t, cudaGraphDestroy> graph;
    cc::UniqueHandle<cudaGraphExec_t, cudaGraphExecDestroy> executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    submit();
    check(cudaStreamEndCapture(stream, graph.put()));
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph.get(), nullptr, &count));
    require(count == expected, "embedding activation kernel count");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph.get(), nodes.data(), &count));
    for (auto node : nodes) {
        cudaGraphNodeType kind{};
        check(cudaGraphNodeGetType(node, &kind));
        require(kind == cudaGraphNodeTypeKernel, "unexpected embedding graph node");
    }
    check(cudaGraphInstantiate(executable.put(), graph.get(), 0));
    check(cudaGraphLaunch(executable.get(), stream));
    check(cudaStreamSynchronize(stream));
}
void run(unsigned rows, unsigned width, cudaStream_t stream) {
    std::vector<std::uint16_t> table(std::size_t{vocabulary} * width), gain(width);
    for (std::size_t i = 0; i < table.size(); ++i)
        table[i] = sf::bf16_encode(static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.25F);
    for (unsigned i = 0; i < width; ++i) gain[i] = sf::bf16_encode(1.0F + static_cast<float>(i % 4U) * 0.125F);
    std::vector<std::uint32_t> ids(rows);
    for (unsigned r = 0; r < rows; ++r) ids[r] = (7U * r + 3U) % vocabulary;
    const auto count = std::size_t{rows} * width;
    const std::vector<float> blank(count + 2U * guard, std::bit_cast<float>(sentinel));
    Buffer<std::uint16_t> weights(table.size()), gains(gain.size());
    Buffer<std::uint32_t> tokens(rows);
    Buffer<float> old_output(blank.size()), new_output(blank.size()), catalogue(width + 2U * guard);
    weights.upload(table); gains.upload(gain); tokens.upload(ids);
    old_output.upload(blank); new_output.upload(blank);
    catalogue.upload(std::vector<float>(width + 2U * guard, std::bit_cast<float>(sentinel)));
    float* old = old_output.data() + guard;
    float* current = new_output.data() + guard;
    const cc::DenseWeight embedding{weights.data(), cc::WeightFormat::bf16, vocabulary, width};
    check(cc::embedding_rows(embedding, tokens.data(), old, rows, stream));
    check(cudaStreamSynchronize(stream));
    const auto widened = old_output.read();
    canaries(widened);
    for (std::size_t i = 0; i < count; ++i)
        require(std::bit_cast<std::uint32_t>(widened[guard + i]) ==
                (std::uint32_t{table[std::size_t{ids[i / width]} * width + i % width]} << 16U), "CPU BF16 widening");
    const cc::GroupedView shape{rows, 1U, width};
    replay(stream, 2, [&] {
        check(cc::embedding_rows(embedding, tokens.data(), old, rows, stream));
        check(cc::rmsnorm_bf16(old, shape, gains.data(), old, shape, epsilon, stream));
    });
    replay(stream, 2, [&] {
        check(cc::enqueue_bf16_embeddings_rmsnorm(weights.data(), vocabulary, width, gains.data(), epsilon,
                                                tokens.data(), current, rows, stream));
    });
    const auto reference = old_output.read(), actual = new_output.read();
    canaries(reference); canaries(actual);
    for (unsigned r = 0; r < rows; ++r) {
        double sum = 0;
        for (unsigned c = 0; c < width; ++c) {
            const float x = sf::u2f(std::uint32_t{table[std::size_t{ids[r]} * width + c]} << 16U);
            sum += static_cast<double>(x) * x;
        }
        const double inverse = 1.0 / std::sqrt(sum / width + epsilon);
        for (unsigned c = 0; c < width; ++c) {
            const auto i = guard + std::size_t{r} * width + c;
            require(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(reference[i]), "old/new RMS bits");
            const float x = sf::u2f(std::uint32_t{table[std::size_t{ids[r]} * width + c]} << 16U);
            const float g = sf::u2f(std::uint32_t{gain[c]} << 16U);
            const float cpu = sf::bf16_round(static_cast<float>(static_cast<double>(x) * inverse * g));
            require(std::isfinite(actual[i]) && (std::bit_cast<std::uint32_t>(actual[i]) & 0xffffU) == 0,
                    "norm output is not widened BF16");
            // Independent high precision scalar oracle: permit one BF16 rounding step.
            require(cpu == 0.0F ? actual[i] == 0.0F : std::fabs(actual[i] - cpu) <= std::fabs(cpu) / 128.0F,
                    "CPU scalar RMS reference");
        }
    }
    const auto binding = cd::bind(cd::Embedding{weights.data(), tokens.data(), catalogue.data() + guard, vocabulary, width});
    require(binding.identity == strix::cuda::core::lookup::detail::kEntries.front().identity &&
            binding.predicted.launches == 1U, "catalogue does not bind shared lookup");
    replay(stream, 1, [&] { binding.fn(binding.args, binding.grid, stream); });
    const auto gathered = catalogue.read();
    canaries(gathered);
    for (unsigned c = 0; c < width; ++c)
        require(std::bit_cast<std::uint32_t>(gathered[guard + c]) ==
                (std::uint32_t{table[std::size_t{ids[0]} * width + c]} << 16U), "catalogue CPU widening");
    std::printf("resident embedding R%u D%u PASS helper-exact cpu-widen cpu-norm canaries catalogue-r1\n", rows, width);
}
} // namespace
int main() try {
    cc::UniqueHandle<cudaStream_t, cudaStreamDestroy> stream;
    check(cudaStreamCreate(stream.put()));
    run(1, 32, stream.get());
    run(1, 2304, stream.get());
    run(4, 2304, stream.get());
    run(129, 2304, stream.get());
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "resident embedding FAIL: %s\n", error.what());
    return 1;
}
