// Operator fixture for sm_120a. No models are required.
#include "cuda/core/dense/dense_mma.hpp"
#include "cuda/core/dense/rows_fp4.cuh"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/weight/formats/scalar_float.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <bit>
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
using Buffer = cc::UniqueHandle<T*, cudaFree>;

template <class T>
Buffer<T> upload(const std::vector<T>& host) {
    Buffer<T> device;
    const auto bytes = host.size() * sizeof(T);
    check_cuda(cudaMalloc(device.put(), bytes));
    check_cuda(cudaMemcpy(device.get(), host.data(), bytes, cudaMemcpyHostToDevice));
    return device;
}

std::vector<float> download(const float* device, std::size_t count) {
    std::vector<float> host(count);
    check_cuda(cudaMemcpy(host.data(), device, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

struct JobBuffers {
    Buffer<unsigned char> weights, scales;
    Buffer<float> x, y, reference, residual;
};

// Separate rounded host operations for the MXFP8 arithmetic oracle.
float mul(float a, float b) { const volatile float v = a * b; return v; }
float add(float a, float b) { const volatile float v = a + b; return v; }

std::vector<float> mxfp8_reference(const rows::Job& job, std::uint32_t count, std::uint32_t splits,
                                   const std::vector<unsigned char>& codes, const std::vector<unsigned char>& scales,
                                   const std::vector<float>& x, const std::vector<float>& residual) {
    std::vector<float> y(static_cast<std::size_t>(count) * job.out);
    const auto blocks = job.in / 32;
    for (std::uint32_t r = 0; r < count; ++r) {
        for (std::uint32_t o = 0; o < job.out; ++o) {
            float sum = 0.0f;
            for (unsigned s = 0; s < splits; ++s) {
                float partial = 0.0f;
                for (auto g = s * blocks / splits; g < (s + 1) * blocks / splits; ++g) {
                    float dot = 0.0f;
                    for (unsigned k = g * 32; k < (g + 1) * 32; ++k) {
                        const auto w = weight::scalar_float::e4m3(codes[static_cast<std::size_t>(o) * job.in + k]);
                        dot = add(dot, mul(w, x[static_cast<std::size_t>(r) * job.in + k]));
                    }
                    const auto scale = weight::scalar_float::e8m0(scales[static_cast<std::size_t>(o) * blocks + g]);
                    partial = add(partial, mul(dot, scale));
                }
                sum = s == 0 ? partial : add(sum, partial);
            }
            const auto i = static_cast<std::size_t>(r) * job.out + o;
            y[i] = add(residual[i], sum);
        }
    }
    return y;
}

void check_bits(const std::vector<float>& actual, const std::vector<float>& expected,
                unsigned row_count, unsigned job) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto a = std::bit_cast<std::uint32_t>(actual[i]);
        const auto b = std::bit_cast<std::uint32_t>(expected[i]);
        if (a == b) continue;
        std::fprintf(stderr, "FAIL rows=%u job=%u element=%zu actual=%08x expected=%08x\n", row_count, job, i, a, b);
        throw std::runtime_error("output bits differ");
    }
}

void run(std::uint32_t count, const strix::device::Profile& profile, cudaStream_t stream) {
    constexpr std::array<std::uint32_t, 6> outs{16, 1536, 80, 16, 48, 32};
    constexpr std::array<std::uint32_t, 6> ins{128, 8192, 32, 1056, 256, 288};
    std::array<JobBuffers, outs.size()> buffers;
    std::array<rows::Job, outs.size()> jobs{};
    std::array<std::vector<float>, 2> expected_mxfp8;
    std::array<std::vector<unsigned char>, 2> mxfp8_codes, mxfp8_scales;
    std::array<std::vector<float>, outs.size()> host_x, host_resid;
    std::mt19937 rng(0x98243476U);

    for (unsigned j = 0; j < jobs.size(); ++j) {
        auto& buf = buffers[j];
        const auto out = outs[j], in = ins[j];
        host_x[j].resize(static_cast<std::size_t>(count) * in);
        host_resid[j].resize(static_cast<std::size_t>(count) * out);
        for (float& x : host_x[j]) x = static_cast<float>(static_cast<int>(rng() % 200001U) - 100000) * 0x1p-18f;
        for (float& r : host_resid[j]) r = static_cast<float>(static_cast<int>(rng() % 401U) - 200) * 0x1p-8f;
        buf.x = upload(host_x[j]);
        buf.residual = upload(host_resid[j]);
        buf.y = upload(std::vector<float>(host_resid[j].size(), 0x1p20f));
        buf.reference = upload(std::vector<float>(host_resid[j].size(), -0x1p20f));
        auto& job = jobs[j];

        if (j < 2) {
            const float outer = j == 0 ? 0.37f : 0.0825f;
            std::vector<unsigned char> packed(cc::nvfp4_bytes(out, in));
            for (std::size_t r = 0; r < packed.size(); r += cc::kNvfp4RecordBytes) {
                for (unsigned k = 0; k < 1024; ++k) packed[r + k] = static_cast<unsigned char>(rng());
                for (unsigned k = 1024; k < cc::kNvfp4RecordBytes; ++k)
                    packed[r + k] = static_cast<unsigned char>(rng() % 127U);
            }
            buf.weights = upload(packed);
            job.w = {buf.weights.get(), rows::kNvfp4Records, nullptr, outer};
            const cc::DenseWeight legacy{buf.weights.get(), cc::WeightFormat::nvfp4, out, in, outer};
            check_cuda(cc::nvfp4_dense_rows(legacy, buf.x.get(), buf.reference.get(), count, buf.residual.get(), stream));
        } else if (j < 4) {
            auto& codes = mxfp8_codes[j - 2];
            auto& scales = mxfp8_scales[j - 2];
            codes.resize(static_cast<std::size_t>(out) * in);
            scales.resize(static_cast<std::size_t>(out) * (in / 32));
            for (auto& c : codes) c = static_cast<unsigned char>((rng() % 127U) | (rng() & 128U));
            for (auto& s : scales) s = static_cast<unsigned char>(121 + rng() % 12U);
            buf.weights = upload(codes);
            buf.scales = upload(scales);
            job.w = {buf.weights.get(), rows::kMxfp8Rows, buf.scales.get()};
        } else {
            const bool bf16 = j == 4;
            const auto blocks = static_cast<std::size_t>(out) * (in / 32);
            std::vector<unsigned char> packed(bf16 ? static_cast<std::size_t>(out) * in * 2 : blocks * 34, 1);
            const auto stride = bf16 ? 2U : 34U;
            for (std::size_t k = 0; k < packed.size(); k += stride) {
                packed[k] = 0;
                packed[k + 1] = bf16 ? 0x3f : 0x3c;
            }
            buf.weights = upload(packed);
            job.w = {buf.weights.get(), bf16 ? weight::kBf16WeightStorage : weight::kQ8WeightStorage};
            cc::MmaCall legacy{};
            legacy.w[0] = buf.weights.get();
            legacy.y[0] = buf.reference.get();
            legacy.out[0] = out;
            legacy.n = 1;
            legacy.format = bf16 ? cc::WeightFormat::bf16 : cc::WeightFormat::q8_0;
            legacy.in = in;
            legacy.rows = count;
            legacy.x = buf.x.get();
            legacy.residual = buf.residual.get();
            check_cuda(cc::dense_mma_launch(legacy, stream));
        }
        job.out = out;
        job.in = in;
        job.x = {buf.x.get()};
        job.y = {buf.y.get()};
        job.epilogue = cc::kRowsAddResidual;
        job.aux[0] = {buf.residual.get()};
    }

    const auto plan = rows::plan({jobs, count}, profile);
    for (const auto& launch : plan.launches()) {
        if (launch.form != rows::Form::mxfp8_split_k) continue;
        for (unsigned q = 0; q < launch.tiles.count; ++q) {
            const unsigned j = launch.jobs[q].in == ins[2] ? 2 : 3;
            expected_mxfp8[j - 2] = mxfp8_reference(jobs[j], count, launch.splits[q], mxfp8_codes[j - 2],
                                                   mxfp8_scales[j - 2], host_x[j], host_resid[j]);
        }
    }
    const auto bytes = cc::rows_split_k_workspace(plan);
    Buffer<float> scratch;
    check_cuda(cudaMalloc(scratch.put(), bytes));
    if (cc::cuda_rows_grouped(plan, scratch.get(), bytes - 1, stream) != cudaErrorInvalidValue)
        throw std::runtime_error("short workspace accepted");

    cc::UniqueHandle<cudaGraph_t, cudaGraphDestroy> graph;
    cc::UniqueHandle<cudaGraphExec_t, cudaGraphExecDestroy> exec;
    cc::nvfp4_set_readahead(1);
    (void)cc::nvfp4_take_readahead_bytes();
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check_cuda(cc::cuda_rows_grouped(plan, scratch.get(), bytes, stream));
    check_cuda(cudaStreamEndCapture(stream, graph.put()));
    const auto prefetched = cc::nvfp4_take_readahead_bytes();
    if ((prefetched != 0) != (count <= cc::kNvfp4ReadaheadMaxRows))
        throw std::runtime_error("unexpected capture-side readahead state");
    check_cuda(cudaGraphInstantiate(exec.put(), graph.get(), 0));
    check_cuda(cudaGraphLaunch(exec.get(), stream));
    check_cuda(cudaStreamSynchronize(stream));

    for (unsigned j = 0; j < jobs.size(); ++j) {
        const auto elements = static_cast<std::size_t>(count) * jobs[j].out;
        check_bits(download(buffers[j].y.get(), elements),
                   j == 2 || j == 3 ? expected_mxfp8[j - 2] : download(buffers[j].reference.get(), elements), count, j);
    }
    std::printf("PASS CUDA rows formats M=%u NVFP4=bitwise MXFP8=bitwise BF16=bitwise Q8_0=bitwise\n", count);
}

}  // namespace

int main() {
    try {
        cc::nvfp4_set_split_target(0);
        check_cuda(cc::nvfp4_set_split_profile(false));
        strix::device::Profile profile{};
        profile.units = cc::nvfp4_split_device().units;
        cc::UniqueHandle<cudaStream_t, cudaStreamDestroy> stream;
        check_cuda(cudaStreamCreateWithFlags(stream.put(), cudaStreamNonBlocking));
        for (const auto count : {1U, 2U, 4U, 8U, 16U, 32U}) run(count, profile, stream.get());
        cc::nvfp4_release_workspace(stream.get());
        std::puts("PASS CUDA rows formats");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA rows formats: %s\n", e.what());
        return 1;
    }
}
