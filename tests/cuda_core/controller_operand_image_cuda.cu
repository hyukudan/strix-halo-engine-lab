// The low-rank controller's BF16 operand image on CUDA (device fixture; run by the operator, outside CTest).
//
// Route A (FP32 operands): normalize_pair stores the rounded FP32 operand pair [rows][2 x width], then the down
// projection splits one plane (Bf16Operand::bf16_exact). Route B (image): normalize_pair_image stores only the BF16
// image of the same values and enqueue_bf16_dense_rows_plane reads it. The latent must be bitwise equal, and equal to
// the three-plane FP32 route (the one-plane contract). The gate shape (K = rank, unrounded sums) is checked on a host
// encoded plane the same way. Pad rows of the image start as NaN garbage: the entry zero-fills them as the split does.
// Refusals: short rows, a misaligned plane, a pitched weight and a missing context launch nothing.
#include "core/residual_mix.hpp"
#include "cuda/core/dense/bf16_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cc = strix::cuda_core;
namespace core = strix::core;
namespace rm = strix::core::residual_mix;
namespace sf = strix::weight::scalar_float;

namespace {

constexpr unsigned kWidth = 2304;
constexpr unsigned kStride = 2U * kWidth;
constexpr unsigned kRank = 64;

bool ok_status(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::printf("FAIL %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

template <class T>
struct Device {
    T* data = nullptr;
    explicit Device(std::size_t count) { (void)cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

template <class T>
std::vector<T> download(const T* source, std::size_t count) {
    std::vector<T> host(count);
    (void)cudaMemcpy(host.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<std::uint16_t> bf16_weights(std::size_t count, std::mt19937& rng, float scale) {
    std::normal_distribution<float> normal(0.0F, scale);
    std::vector<std::uint16_t> out(count);
    for (auto& w : out) w = sf::bf16_encode(normal(rng));
    return out;
}

// Down projection over the normalized pair: image route vs FP32 operand route vs three planes.
bool check_down(unsigned rows, cudaStream_t stream, cc::BigCtx* context, std::mt19937& rng) {
    const std::size_t elements = std::size_t{rows} * kWidth;
    std::vector<float> raw(elements), message(elements);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (std::size_t i = 0; i < elements; ++i) {
        raw[i] = normal(rng) * (i % 97U == 0U ? 64.0F : 1.0F);
        message[i] = normal(rng) + 1.0F + 0x1p-12F;  // residuals below BF16 before the rounded normalization
    }
    const auto down = bf16_weights(std::size_t{kRank} * kStride, rng, 0.02F);

    Device<float> d_raw(elements), d_message(elements), d_operands(2U * elements);
    Device<float> latent_a(std::size_t{rows} * kRank), latent_b(std::size_t{rows} * kRank),
        latent_c(std::size_t{rows} * kRank);
    Device<std::uint16_t> d_down(down.size());
    cudaMemcpy(d_raw.data, raw.data(), elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_message.data, message.data(), elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down.data, down.data(), down.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    const rm::NormalizeArgs first{d_raw.data, d_operands.data, rows, kWidth, 1e-6F, kWidth, kStride, true};
    const rm::NormalizeArgs second{d_message.data, d_operands.data + kWidth, rows, kWidth, 1e-6F, kWidth, kStride, true};
    const dim3 grid(rows < rm::detail::kGridLimit ? rows : rm::detail::kGridLimit, 2U);

    // A and C: FP32 operands, one plane declared / three planes.
    rm::detail::normalize_pair<<<grid, rm::detail::kThreads, 0, stream>>>(first, second);
    if (!ok_status(cudaGetLastError(), "normalize_pair")) return false;
    const core::Bf16RowProjection pa[] = {{d_down.data, kRank, latent_a.data, true}};
    const core::Bf16RowProjection pc[] = {{d_down.data, kRank, latent_c.data, true}};
    if (!ok_status(cc::enqueue_bf16_dense_rows(d_operands.data, kStride, rows, pa, stream, cc::DenseArith::mma3, context,
                                               nullptr, core::Bf16Operand::bf16_exact), "down one plane") ||
        !ok_status(cc::enqueue_bf16_dense_rows(d_operands.data, kStride, rows, pc, stream, cc::DenseArith::mma3, context,
                                               nullptr, core::Bf16Operand::fp32), "down three planes"))
        return false;

    // B: the image at the base of the (refilled with NaN bytes) FP32 operand rows.
    cudaMemsetAsync(d_operands.data, 0xFF, 2U * elements * sizeof(float), stream);
    auto* const image = reinterpret_cast<std::uint16_t*>(d_operands.data);
    rm::detail::launch_normalize_pair_image(grid, first, second, image, image + kWidth, stream);
    if (!ok_status(cudaGetLastError(), "normalize_pair_image")) return false;
    const core::Bf16RowProjection pb[] = {{d_down.data, kRank, latent_b.data, true}};
    const bool admitted = cc::bf16_plane_rows_admitted(kStride, rows, pb, context);
    if (!admitted) {
        std::printf("FAIL down rows %u: not admitted\n", rows);
        return false;
    }
    if (!ok_status(cc::enqueue_bf16_dense_rows_plane(image, kStride, rows, pb, stream, context), "down image")) return false;
    if (!ok_status(cudaStreamSynchronize(stream), "sync down")) return false;

    const auto a = download(latent_a.data, std::size_t{rows} * kRank);
    const auto b = download(latent_b.data, std::size_t{rows} * kRank);
    const auto c = download(latent_c.data, std::size_t{rows} * kRank);
    const bool ab = same_bits(a, b), ac = same_bits(a, c);
    std::printf("%s down rows %u (pad %u): image == FP32 one plane %d, one plane == three planes %d\n",
                ab && ac ? "PASS" : "FAIL", rows, cc::bf16_plane_rows_pad(rows) - rows, ab, ac);
    return ab && ac;
}

// Gate pair shape: K = rank on a host-encoded plane of a rounded latent, unrounded sums.
bool check_gates(unsigned rows, cudaStream_t stream, cc::BigCtx* context, std::mt19937& rng) {
    const std::size_t count = std::size_t{rows} * kRank;
    const unsigned pad = cc::bf16_plane_rows_pad(rows);
    std::vector<float> latent(count);
    std::vector<std::uint16_t> plane(std::size_t{pad} * kRank, 0xFFFFU);
    std::normal_distribution<float> normal(0.0F, 2.0F);
    for (std::size_t i = 0; i < count; ++i) {
        plane[i] = sf::bf16_encode(normal(rng));
        const std::uint32_t bits = std::uint32_t{plane[i]} << 16U;
        std::memcpy(&latent[i], &bits, sizeof bits);
    }
    const auto forget = bf16_weights(std::size_t{kWidth} * kRank, rng, 0.1F);
    const auto gain = bf16_weights(std::size_t{kWidth} * kRank, rng, 0.1F);
    Device<float> d_latent(count), out_a(std::size_t{rows} * kWidth * 2U), out_b(std::size_t{rows} * kWidth * 2U);
    Device<std::uint16_t> d_plane(plane.size()), d_forget(forget.size()), d_gain(gain.size());
    cudaMemcpy(d_latent.data, latent.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plane.data, plane.data(), plane.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forget.data, forget.data(), forget.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gain.data, gain.data(), gain.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    const std::size_t half = std::size_t{rows} * kWidth;
    const core::Bf16RowProjection pa[] = {{d_forget.data, kWidth, out_a.data}, {d_gain.data, kWidth, out_a.data + half}};
    const core::Bf16RowProjection pb[] = {{d_forget.data, kWidth, out_b.data}, {d_gain.data, kWidth, out_b.data + half}};
    if (!ok_status(cc::enqueue_bf16_dense_rows(d_latent.data, kRank, rows, pa, stream, cc::DenseArith::mma3, context,
                                               nullptr, core::Bf16Operand::bf16_exact), "gates one plane") ||
        !ok_status(cc::enqueue_bf16_dense_rows_plane(d_plane.data, kRank, rows, pb, stream, context), "gates image") ||
        !ok_status(cudaStreamSynchronize(stream), "sync gates"))
        return false;
    const bool equal = same_bits(download(out_a.data, 2U * half), download(out_b.data, 2U * half));
    std::printf("%s gates rows %u: image == FP32 one plane %d\n", equal ? "PASS" : "FAIL", rows, equal);
    return equal;
}

bool check_refusals(cudaStream_t stream, cc::BigCtx* context) {
    Device<std::uint16_t> plane(std::size_t{256} * kStride + 8U), weights(std::size_t{kRank} * kStride);
    Device<float> out(std::size_t{256} * kRank);
    const core::Bf16RowProjection p[] = {{weights.data, kRank, out.data, true}};
    core::Bf16RowProjection pitched[] = {{weights.data, kRank, out.data, true}};
    pitched[0].row_bytes = std::size_t{kStride} * 2U + 64U;
    cudaMemsetAsync(out.data, 0x7F, std::size_t{256} * kRank * sizeof(float), stream);
    const bool short_rows = !cc::bf16_plane_rows_admitted(kStride, 1U, p, context) &&
                            cc::enqueue_bf16_dense_rows_plane(plane.data, kStride, 1U, p, stream, context) ==
                                cudaErrorInvalidValue;
    const bool misaligned = cc::enqueue_bf16_dense_rows_plane(plane.data + 1, kStride, 256U, p, stream, context) ==
                            cudaErrorInvalidValue;
    const bool pitch = !cc::bf16_plane_rows_admitted(kStride, 256U, pitched, context);
    const bool no_context = !cc::bf16_plane_rows_admitted(kStride, 256U, p, nullptr);
    core::Bf16RowProjection separate[] = {{weights.data, kRank, out.data, true}};
    separate[0].round_separate = true;
    const bool separate_round = !cc::bf16_plane_rows_admitted(kStride, 256U, separate, context);
    cudaStreamSynchronize(stream);
    const auto o = download(out.data, std::size_t{256} * kRank);
    std::uint32_t first = 0;
    std::memcpy(&first, o.data(), sizeof first);
    const bool untouched = first == 0x7F7F7F7FU;
    const bool ok = short_rows && misaligned && pitch && no_context && separate_round && untouched;
    std::printf("%s refusals: short rows %d, misaligned %d, pitched %d, no context %d, separate round %d, output untouched %d\n",
                ok ? "PASS" : "FAIL", short_rows, misaligned, pitch, no_context, separate_round, untouched);
    return ok;
}

}  // namespace

int main() {
    cudaStream_t stream{};
    if (!ok_status(cudaStreamCreate(&stream), "stream")) return 1;
    cc::BigCtx* context = cc::dense_big_ctx_create();
    if (!context) {
        std::printf("FAIL dense context\n");
        return 1;
    }
    std::mt19937 rng(20261124U);
    bool ok = true;
    for (const unsigned rows : {129U, 1000U, 2048U, 8192U}) ok = check_down(rows, stream, context, rng) && ok;
    for (const unsigned rows : {129U, 2048U}) ok = check_gates(rows, stream, context, rng) && ok;
    ok = check_refusals(stream, context) && ok;
    cc::dense_big_ctx_destroy(context);
    cudaStreamDestroy(stream);
    std::printf(ok ? "ALL PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
