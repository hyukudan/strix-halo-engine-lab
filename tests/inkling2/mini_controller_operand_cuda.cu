// Exercise the CUDA resident controller dispatch with the shared controller's normalization and latent producer.
#include "inkling2/inkling2_mini_controller_device.hpp"
#include "inkling2/inkling2_mini_controller_operands_cuda.hpp"
#include "inkling2/inkling2_mini_dense_cuda.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

namespace {

namespace cc = strix::cuda_core;
namespace core = strix::core;
namespace mini = strix::inkling2;

constexpr unsigned kRows = 2048;
constexpr auto kController = mini::kMiniController;
constexpr auto kWidth = kController.width;
constexpr auto kRank = kController.rank;
constexpr auto kElements = std::size_t{kRows} * kWidth;
constexpr auto kScratch = kController.scratch_elements(kRows);

void require(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct Buffer {
    explicit Buffer(std::size_t count) : size(count) { require(cudaMallocManaged(&data, count * sizeof(T))); }
    ~Buffer() { (void)cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::span<T> span() { return {data, size}; }
    T* data = nullptr;
    std::size_t size;
};

struct ControllerOps {
    cc::BigCtx* context;
    const mini::MiniControllerCertificate* certificate;
    core::Bf16Operand forced = core::Bf16Operand::fp32;
    std::array<core::Bf16Operand, 2> operands{};
    std::size_t calls{};

    cudaStream_t stream() const { return nullptr; }
    void check(cudaError_t status) const { require(status); }
    void rms(const float* input, const std::uint16_t* gain, float* output, unsigned rows) const {
        check(mini::enqueue_mini_cuda_rmsnorm(input, gain, output, rows, stream()));
    }
    void dense(const float* input, unsigned width, unsigned rows, std::span<const core::Bf16RowProjection> projections,
               core::Bf16Operand declared) {
        if (certificate) {
            const auto launch = mini::enqueue_controller_dense(*certificate, input, width, rows, projections,
                                                               stream(), context, declared);
            check(launch.status);
            operands.at(calls++) = launch.operand;
        } else {
            check(cc::enqueue_bf16_dense_rows(input, width, rows, projections, stream(),
                                             cc::DenseArith::mma3, context, nullptr, forced));
        }
    }
};

bool equal(std::span<const float> a, std::span<const float> b) {
    return std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

bool run_case(std::string_view name, cc::BigCtx* context, Buffer<std::uint16_t>& down, Buffer<std::uint16_t>& forget,
              Buffer<std::uint16_t>& gain, std::span<const float> raw, std::span<const float> message,
              const std::uint16_t* norm, core::Bf16Operand expected, bool require_difference = false) {
    mini::MiniControllerCertificate certificate;
    certificate.add_down(down.span());
    certificate.add_gate(forget.span());
    certificate.add_gate(gain.span());

    const core::residual_mix::ControllerWeights weights{{}, down.data, forget.data, gain.data, norm};
    Buffer<float> scratch(kScratch), baseline_scratch(kScratch), output(kElements), baseline_output(kElements);
    ControllerOps ops{context, &certificate}, baseline{context, nullptr};
    const auto run = [&](ControllerOps& operations, Buffer<float>& result, Buffer<float>& work) {
        core::residual_mix::enqueue_low_rank_controller_tail(kController, weights, kRows, raw, message,
                                                           result.span(), work.span(), operations);
    };
    run(ops, output, scratch);
    run(baseline, baseline_output, baseline_scratch);
    require(cudaDeviceSynchronize());

    // The controller leaves the forget/gain logits and the rounded latent in its scratch.
    const auto gate_bits_equal = [&] {
        return equal(scratch.span().subspan(kElements, 2U * kElements),
                     baseline_scratch.span().subspan(kElements, 2U * kElements));
    };
    const bool identical = gate_bits_equal() && equal(scratch.span().subspan(4U * kElements),
        baseline_scratch.span().subspan(4U * kElements)) && equal(output.span(), baseline_output.span());
    bool sensitive = true;
    if (require_difference) {
        ControllerOps forced{context, nullptr, core::Bf16Operand::bf16_exact};
        run(forced, output, scratch);
        require(cudaDeviceSynchronize());
        sensitive = !gate_bits_equal();
    }
    const bool passed = ops.calls == 2U && ops.operands == std::array{expected, expected} && identical && sensitive;
    std::cout << (passed ? "PASS " : "FAIL ") << name << ' '
              << (ops.operands[0] == core::Bf16Operand::bf16_exact ? 1 : 3)
              << " planes, latent/gates/output bitwise match " << identical << ", refusal sensitive " << sensitive << '\n';
    return passed;
}

}  // namespace

int main() {
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> context(cc::dense_big_ctx_create(),
                                                                            cc::dense_big_ctx_destroy);
    if (!context) return 1;
    Buffer<float> raw(kElements), message(kElements);
    for (std::size_t i = 0; i < kElements; ++i) {
        raw.data[i] = static_cast<float>(1U + i % 17U);
        message.data[i] = static_cast<float>(1U + i % 13U);
    }
    Buffer<std::uint16_t> down(std::size_t{kRank} * kController.down_operand_elements());
    Buffer<std::uint16_t> forget(std::size_t{kWidth} * kRank), gain(forget.size), norm(kWidth);
    std::ranges::fill(down.span(), 0x3C00U);
    std::ranges::fill(forget.span(), 0x3C00U);
    std::ranges::fill(gain.span(), 0x3C00U);
    std::ranges::fill(norm.span(), 0x3F80U);

    const auto test = [&](std::string_view name, core::Bf16Operand expected, bool require_difference = false) {
        return run_case(name, context.get(), down, forget, gain, raw.span(), message.span(), norm.data,
                        expected, require_difference);
    };
    const bool finite = test("finite", core::Bf16Operand::bf16_exact);
    forget.data[0] = 0x7F80U;
    const bool infinite = test("infinite forget", core::Bf16Operand::fp32, true);
    forget.data[0] = 0x3C00U;
    std::ranges::fill(down.span(), 0x7F00U);
    const bool oversized = test("oversized finite down", core::Bf16Operand::fp32, true);
    return finite && infinite && oversized ? 0 : 1;
}
