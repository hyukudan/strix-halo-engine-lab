#include "mini_dense_cuda_support.hpp"
#include "inkling2/inkling2_mini_resident_weights.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace mini = strix::inkling2;
namespace cc = strix::cuda_core;
namespace core = strix::core;
using Role = mini::MiniAttentionProjection;

namespace {

// The resident operand class end to end on staged device tensors: the shared staging scan
// (certify_mini_attention + core::Bf16StagedScan, the code the decoder runs after the upload) certifies device-resident
// attention tensors, the resident decoder's own backend factory (mini_attention_backend, the one resident_backend.hpp
// builds its attention backend with) turns the certificate into the declared class, and the planes the shared dense
// dispatch selects are captured there (planes_report) and compared with the three-plane output. A finite checkpoint
// dispatches ONE plane; an Inf or oversized finite coefficient in the projection, or an Inf gain, dispatches THREE and
// gives the three-plane bits (an operand hard-coded to FP32 reports 3 planes on the finite case, one hard-coded to
// bf16_exact reports 1 on the refused ones). The forced one-plane control with an Inf weight must differ from the
// three-plane output (the zero planes meet 0 x inf), which proves the refused case is discriminating.
bool check_resident_operand_class(unsigned rows, bool disable_one_plane) {
    using namespace mini;
    using namespace mini_dense_fixture;
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!ctx) return false;
    std::uint32_t seed = 0x0DDBA11U;
    Buffer<float> raw(std::size_t{rows} * kHiddenWidth, 0.0F), normalised(std::size_t{rows} * kHiddenWidth);
    for (std::size_t i = 0; i < raw.size; ++i) {
        seed = 1664525U * seed + 1013904223U;
        raw.data[i] = static_cast<float>(static_cast<int>(seed >> 8U) - 8388608) * 0x1p-21F;
    }
    Buffer<std::uint16_t> gain(kHiddenWidth, 0x3F80U);
    require(enqueue_mini_cuda_rmsnorm(raw.data, gain.data, normalised.data, rows, nullptr));
    require(cudaDeviceSynchronize());
    Buffer<std::uint16_t> weight(std::size_t{kQueryWidth} * kHiddenWidth), ordinary(4096U, 0x3F80U), bad_gain(4096U, 0x3F80U);
    fill_bf16_weights(weight, seed);
    bad_gain.data[7] = 0xFF80U;  // -inf layer norm gain
    const std::string query = std::string(mini_attention_projection_spec(Role::query).tensor_suffix);
    const std::uint16_t good_value = weight.data[0], inf_bits = 0xFF80U, huge_bits = 0x7F00U;
    Buffer<float> reference(std::size_t{rows} * kQueryWidth), result(std::size_t{rows} * kQueryWidth),
        forced(std::size_t{rows} * kQueryWidth);

    struct Outcome {
        bool certified;
        int planes;
        bool equals_three_planes;
    };
    const auto run = [&](std::uint16_t coefficient, bool poison_gain) {
        weight.data[0] = coefficient;
        require(cudaDeviceSynchronize());
        core::Bf16StagedScan scan;
        const auto certificate = certify_mini_attention(
            scan,
            [&](std::string_view name) -> std::span<const std::uint16_t> {
                if (name.ends_with(query)) return {weight.data, weight.size};
                if (poison_gain && name.ends_with("attn_norm.weight")) return {bad_gain.data, bad_gain.size};
                return {ordinary.data, ordinary.size};
            },
            [](std::span<const std::uint16_t> staged, std::span<std::uint16_t> host) {
                require(cudaMemcpy(host.data(), staged.data(), staged.size_bytes(), cudaMemcpyDeviceToHost));
            });
        int planes = 0;
        auto native = detail::mini_attention_backend(nullptr, ctx.get(), certificate, &planes);
        // Seam: with --fp32-operand the one-plane dispatch is disabled (the operand class forced to FP32); the fixture must FAIL.
        if (disable_one_plane) native.operand = core::Bf16Operand::fp32;
        const std::array view{MiniAttentionLinear{weight.data, kQueryWidth, result.data, false}};
        require(native.linear(normalised.data, kHiddenWidth, rows, view));
        require(cudaDeviceSynchronize());
        three_plane_reference(ctx.get(), weight.data, kQueryWidth, kHiddenWidth, rows, normalised.data, reference.data);
        return Outcome{certificate.route(core::Bf16Operand::bf16_exact) == core::Bf16Operand::bf16_exact, planes,
                       same_floats(result.data, reference.data, result.size)};
    };
    const auto finite = run(good_value, false);
    const auto infinite = run(inf_bits, false);
    // Forced one plane with the Inf weight still in place: it must not equal the three-plane output.
    {
        const std::array view{MiniAttentionLinear{weight.data, kQueryWidth, forced.data, false}};
        int forced_planes = 0;
        require(enqueue_mini_cuda_dense_rows(normalised.data, kHiddenWidth, rows, view, nullptr, cc::DenseArith::mma3,
                                             ctx.get(), core::Bf16Operand::bf16_exact, &forced_planes));
        require(cudaDeviceSynchronize());
        if (forced_planes != 1) return false;
    }
    const bool control_differs = !same_floats(forced.data, reference.data, forced.size);
    const auto oversized = run(huge_bits, false);
    const auto gain = run(good_value, true);
    const bool ok = finite.certified && finite.planes == 1 && finite.equals_three_planes && !infinite.certified &&
                    infinite.planes == 3 && infinite.equals_three_planes && control_differs && !oversized.certified &&
                    oversized.planes == 3 && oversized.equals_three_planes && !gain.certified && gain.planes == 3 &&
                    gain.equals_three_planes;
    // One flushed line per case: the expected values are in the notes of this fixture.
    std::cout << "CUDA Mini operand class, rows " << rows << ": finite planes " << finite.planes << " equal "
              << finite.equals_three_planes << "; inf planes " << infinite.planes << " equal "
              << infinite.equals_three_planes << " forced-one-plane differs " << control_differs << "; oversized planes "
              << oversized.planes << " equal " << oversized.equals_three_planes << "; inf gain planes " << gain.planes
              << " equal " << gain.equals_three_planes << std::endl;
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const bool disable_one_plane = argc > 1 && std::string_view(argv[1]) == "--fp32-operand";
    try {
        if (!check_resident_operand_class(2048U, disable_one_plane)) {
            std::cerr << "CUDA Mini operand class differs from the reference fixture\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "CUDA Mini operand class: exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << "CUDA Mini operand class: PASSED" << std::endl;
    return 0;
}
