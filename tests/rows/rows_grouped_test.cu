// Run on a CUDA device with BF16 MMA and programmatic dependent launch support.
#include "cuda_core/dense/dense_mma.hpp"
#include "strix/core/rows.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace cc = strix::cuda_core;
namespace rows = strix::core::rows;
namespace weight = strix::weight;

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
using DeviceBuffer = cc::UniqueHandle<T*, cudaFree>;

template <class T>
DeviceBuffer<T> upload(const std::vector<T>& values) {
    DeviceBuffer<T> result;
    const auto bytes = values.size() * sizeof(T);
    check_cuda(cudaMalloc(result.put(), bytes));
    check_cuda(cudaMemcpy(result.get(), values.data(), bytes, cudaMemcpyHostToDevice));
    return result;
}

struct Buffers {
    DeviceBuffer<unsigned char> w;
    DeviceBuffer<float> x, y, reference, residual;
};

std::vector<unsigned char> weights(cc::WeightFormat format, std::uint32_t out, std::uint32_t in, std::mt19937& rng) {
    if (format == cc::WeightFormat::bf16) {
        std::vector<unsigned char> result(static_cast<std::size_t>(out) * in * 2);
        for (std::size_t i = 0; i < result.size(); i += 2) {
            const auto bits = (rng() & 0x8000U) | (0x3980U + (rng() % 640U));
            result[i] = static_cast<unsigned char>(bits & 0xffU);
            result[i + 1] = static_cast<unsigned char>(bits >> 8);
        }
        return result;
    }
    std::vector<unsigned char> result(static_cast<std::size_t>(out) * (in / 32) * 34);
    for (std::size_t i = 0; i < result.size(); i += 34) {
        const auto scale = i % (17 * 34) == 0 ? 0U : (rng() & 0x8000U) | (0x1800U + rng() % 3072U);
        result[i] = static_cast<unsigned char>(scale & 0xffU);
        result[i + 1] = static_cast<unsigned char>(scale >> 8);
        for (unsigned c = 2; c < 34; ++c) result[i + c] = static_cast<unsigned char>(rng() & 0xffU);
    }
    return result;
}

std::vector<float> activations(std::size_t count, std::mt19937& rng) {
    std::vector<float> result(count);
    for (auto& value : result) {
        const auto mantissa = static_cast<std::int32_t>(rng() % 20000001U) - 10000000;
        value = static_cast<float>(mantissa) * 0x1p-23f;
    }
    return result;
}

void run(cc::WeightFormat format, std::uint32_t row_count, const strix::device::Profile& device, cudaStream_t stream) {
    const bool bf16 = format == cc::WeightFormat::bf16;
    const auto storage = bf16 ? weight::kBf16WeightStorage : weight::kQ8WeightStorage;
    constexpr std::array<std::uint32_t, rows::kMaxJobs> inputs{32, 256, 288, 512, 1056, 2304, 4096, 4352};
    std::array<Buffers, rows::kMaxJobs> buffers;
    std::array<rows::Job, rows::kMaxJobs> jobs{};
    std::mt19937 rng(0xd35e1234U);
    for (unsigned j = 0; j < rows::kMaxJobs; ++j) {
        auto& buf = buffers[j];
        const auto in = inputs[j];
        const auto out = (j % 4 + 1) * 16;
        const auto count = static_cast<std::size_t>(row_count) * out;
        buf.w = upload(weights(format, out, in, rng));
        buf.x = upload(activations(static_cast<std::size_t>(row_count) * in, rng));
        const bool in_place = j == 5;
        buf.y = upload(in_place ? activations(count, rng) : std::vector<float>(count, 0x1p20f));
        buf.reference = upload(std::vector<float>(count, -0x1p20f));
        rows::ChainId epilogue = 0;
        const float* residual = nullptr;
        if (j % 3 == 1 || in_place) {
            epilogue = cc::kRowsAddResidual;
            if (!in_place) buf.residual = upload(activations(count, rng));
            residual = in_place ? buf.y.get() : buf.residual.get();
        } else if (bf16 && j % 3 == 2) {
            epilogue = cc::kRowsRoundBf16;
        }
        jobs[j] = {{buf.w.get(), storage}, out, in, {buf.x.get()}, {buf.y.get()}, epilogue, {{residual}, {}}};
    }

    for (unsigned j = 0; j < rows::kMaxJobs; ++j) {
        const auto& job = jobs[j];
        auto* reference = buffers[j].reference.get();
        if (job.epilogue == cc::kRowsRoundBf16) {
            const cc::DenseWeight w{job.w.data, format, job.out, job.in};
            check_cuda(cc::dense_bf16_mma3_rows_round(&w, &reference, 1, buffers[j].x.get(), row_count, stream));
        } else {
            cc::MmaCall single{};
            single.w[0] = job.w.data;
            single.y[0] = reference;
            single.out[0] = job.out;
            single.n = 1;
            single.format = format;
            single.in = job.in;
            single.rows = row_count;
            single.x = buffers[j].x.get();
            single.residual = static_cast<const float*>(job.aux[0].data);
            check_cuda(cc::dense_mma_launch(single, stream));
        }
    }

    const auto plan = rows::plan({jobs, row_count}, device);
    if (plan.launches().size() != 1 || plan.launches()[0].tiles.count != rows::kMaxJobs)
        throw std::runtime_error("fixture requires one eight-job launch");
    check_cuda(cc::dense_mma3_rows_grouped(plan, stream));
    check_cuda(cudaStreamSynchronize(stream));

    for (unsigned j = 0; j < rows::kMaxJobs; ++j) {
        const auto count = static_cast<std::size_t>(row_count) * jobs[j].out;
        const auto bytes = count * sizeof(float);
        std::vector<float> actual(count), reference(count);
        check_cuda(cudaMemcpy(actual.data(), buffers[j].y.get(), bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(reference.data(), buffers[j].reference.get(), bytes, cudaMemcpyDeviceToHost));
        for (std::size_t k = 0; k < count; ++k) {
            const auto a = std::bit_cast<std::uint32_t>(actual[k]);
            const auto b = std::bit_cast<std::uint32_t>(reference[k]);
            if (std::isfinite(reference[k]) && a == b) continue;
            std::fprintf(stderr, "FAIL %s rows=%u job=%u element=%zu grouped=%08x separate=%08x\n",
                         bf16 ? "BF16" : "Q8_0", row_count, j, k, a, b);
            throw std::runtime_error("grouped output differs");
        }
    }
    std::printf("PASS %s rows=%u jobs=%u grouped == separate (bitwise)\n",
                bf16 ? "BF16" : "Q8_0", row_count, rows::kMaxJobs);
}

}  // namespace

int main() {
    try {
        int ordinal = 0, units = 0;
        check_cuda(cudaGetDevice(&ordinal));
        check_cuda(cudaDeviceGetAttribute(&units, cudaDevAttrMultiProcessorCount, ordinal));
        strix::device::Profile device{};
        device.units = static_cast<std::uint32_t>(units);
        cc::UniqueHandle<cudaStream_t, cudaStreamDestroy> stream;
        check_cuda(cudaStreamCreateWithFlags(stream.put(), cudaStreamNonBlocking));
        for (auto format : {cc::WeightFormat::bf16, cc::WeightFormat::q8_0})
            for (auto row_count : {1U, 2U, 4U, 8U, 16U, 24U, 32U}) run(format, row_count, device, stream.get());
        std::puts("PASS CUDA grouped MMA3 rows");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA grouped MMA3 rows: %s\n", e.what());
        return 1;
    }
}
