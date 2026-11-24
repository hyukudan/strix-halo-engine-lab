// Low-rank controller with the down operands as a BF16 image: enqueue_low_rank_controller stores the same output and
// latent bits whether the backend's down projection reads the FP32 operand rows (normalize_pair + enqueue_bf16_rows)
// or their BF16 image (normalize_pair_image + enqueue_bf16_image_rows), and keeps the FP32 form for decode rows.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/fma_bf16_rows.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#include "core/residual_mix.hpp"
#pragma clang diagnostic pop

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace rows_ns = strix::core::fma_rows;
namespace mix = strix::core::residual_mix;
using strix::core::Bf16RowProjection;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

std::vector<float> values(std::size_t count, std::mt19937& random, float range, bool specials) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(count);
    for (auto& x : out) x = value(random);
    if (specials) {
        const std::uint32_t bits[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
        for (std::size_t i = 7; i < out.size(); i += 104729U) std::memcpy(&out[i], &bits[random() % 6U], sizeof(float));
    }
    return out;
}

std::vector<std::uint16_t> bf16_weights(std::size_t count, std::mt19937& random, float range) {
    std::vector<std::uint16_t> out(count);
    const auto source = values(count, random, range, false);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &source[i], sizeof bits);
        out[i] = static_cast<std::uint16_t>(bits >> 16U);
    }
    return out;
}

// The backend operations of a resident model (dense = shared BF16 rows, gates with the controller epilogue).
struct Operations {
    unsigned width;
    unsigned dense_launches = 0;
    hipStream_t stream() const { return nullptr; }
    void check(hipError_t status) const {
        if (!ok(status)) throw std::runtime_error("controller-image: launch failed");
    }
    void dense(const float* input, unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections) {
        ++dense_launches;
        check(rows_ns::enqueue_bf16_rows(input, input_width, rows, projections, 1U << 20U, nullptr));
    }
    bool dense_controller(const float* input, unsigned input_width, unsigned rows,
                          std::span<const Bf16RowProjection> gates, const mix::ControllerArgs& control) {
        const auto status = rows_ns::enqueue_bf16_controller_rows(input, input_width, rows, gates, control.current,
                                                                  control.message, control.output, nullptr);
        if (status == hipErrorNotSupported) return false;
        check(status);
        return true;
    }
    void rms(const float* input, const std::uint16_t*, float* output, unsigned rows) {
        check(mix::launch_normalize({input, output, rows, width, 1e-6F, 0U, 0U, true}, nullptr));
    }
};

struct ImageOperations : Operations {
    unsigned image_launches = 0;
    static bool admits_dense_bf16_image(unsigned input_width, unsigned rows,
                                        std::span<const Bf16RowProjection> projections) {
        return rows_ns::admits_bf16_image_projections(input_width, rows, projections, 1U << 20U);
    }
    void dense_bf16_image(const std::uint16_t* input, unsigned input_width, unsigned rows,
                          std::span<const Bf16RowProjection> projections) {
        ++image_launches;
        check(rows_ns::enqueue_bf16_image_rows(input, input_width, rows, projections, 1U << 20U, nullptr));
    }
};

// 1 = equal (image form taken iff expected), 0 = differ / error.
// down_shift: the down weights start that many BF16 elements past an aligned allocation (1 = not 16-byte aligned:
// the image launch cannot read them, so the operands stay FP32 and nothing throws).
int run(unsigned rows, unsigned width, unsigned rank, unsigned down_shift, bool expect_image, std::mt19937& random,
        std::size_t& compared) {
    const mix::LowRankController plan{width, rank, 2U, 1e-6F, true, true};
    const std::size_t elements = std::size_t{rows} * width, scratch_elements = plan.scratch_elements(rows);
    auto* raw = upload(values(elements, random, 3.0F, true));
    auto* bank0 = upload(values(elements, random, 1.5F, false));
    auto* bank1 = upload(values(elements, random, 1.5F, true));
    auto* query = upload(values(width, random, 0.2F, false));
    auto* down_base = upload(bf16_weights(std::size_t{rank} * 2U * width + down_shift, random, 0.05F));
    auto* down = down_base ? down_base + down_shift : nullptr;
    auto* forget = upload(bf16_weights(std::size_t{width} * rank, random, 0.2F));
    auto* gain = upload(bf16_weights(std::size_t{width} * rank, random, 0.2F));
    auto* norm = upload(bf16_weights(width, random, 1.0F));
    float *out_a = nullptr, *out_b = nullptr, *scratch_a = nullptr, *scratch_b = nullptr;
    int result = 0;
    if (raw && bank0 && bank1 && query && down && forget && gain && norm &&
        ok(hipMalloc(&out_a, elements * sizeof(float))) && ok(hipMalloc(&out_b, elements * sizeof(float))) &&
        ok(hipMalloc(&scratch_a, scratch_elements * sizeof(float))) &&
        ok(hipMalloc(&scratch_b, scratch_elements * sizeof(float))) &&
        ok(hipMemset(out_b, 0x5A, elements * sizeof(float))) &&
        ok(hipMemset(scratch_b, 0x5A, scratch_elements * sizeof(float)))) {
        const mix::ConstRows banks[] = {{bank0, elements}, {bank1, elements}};
        const mix::ControllerWeights weights{{query, width}, down, forget, gain, norm};
        Operations base{width};
        ImageOperations image{{width}};
        try {
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw, elements}, banks, {out_a, elements},
                                             {scratch_a, scratch_elements}, base);
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw, elements}, banks, {out_b, elements},
                                             {scratch_b, scratch_elements}, image);
        } catch (const std::exception&) {
            return 0;
        }
        const bool took_image = image.image_launches == 1U && image.dense_launches + 1U == base.dense_launches;
        const bool kept_fp32 = image.image_launches == 0U && image.dense_launches == base.dense_launches;
        const std::size_t latent_offset = 4U * elements, latent_elements = std::size_t{rows} * rank;
        std::vector<float> a(elements), b(elements), la(latent_elements), lb(latent_elements);
        if (ok(hipDeviceSynchronize()) && (expect_image ? took_image : kept_fp32) &&
            ok(hipMemcpy(a.data(), out_a, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            ok(hipMemcpy(b.data(), out_b, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            ok(hipMemcpy(la.data(), scratch_a + latent_offset, latent_elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            ok(hipMemcpy(lb.data(), scratch_b + latent_offset, latent_elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            std::memcmp(a.data(), b.data(), elements * sizeof(float)) == 0 &&
            std::memcmp(la.data(), lb.data(), latent_elements * sizeof(float)) == 0) {
            result = 1;
            compared += elements + latent_elements;
        }
    }
    for (void* p : {static_cast<void*>(raw), static_cast<void*>(bank0), static_cast<void*>(bank1),
                    static_cast<void*>(query), static_cast<void*>(down_base), static_cast<void*>(forget),
                    static_cast<void*>(gain), static_cast<void*>(norm), static_cast<void*>(out_a),
                    static_cast<void*>(out_b), static_cast<void*>(scratch_a), static_cast<void*>(scratch_b)})
        if (p) (void)hipFree(p);
    return result;
}

}  // namespace

int main() {
    std::mt19937 random(20261124U);
    std::size_t compared = 0;
    // (rows, width, rank, image expected): the Inkling controller (width 2304, rank 64) with full and partial
    // 128-row blocks, a narrow width with a partial rank panel, decode rows that keep the FP32 operands, and prefill
    // rows whose down weights are not 16-byte aligned (admission refuses the image before it is stored).
    const struct { unsigned rows, width, rank, down_shift; bool image; } cases[] = {
        {1000U, 2304U, 64U, 0U, true}, {300U, 2304U, 64U, 0U, true}, {17U, 2304U, 64U, 0U, true},
        {129U, 256U, 16U, 0U, true},   {4U, 2304U, 64U, 0U, false},  {16U, 256U, 64U, 0U, false},
        {300U, 2304U, 64U, 1U, false},
    };
    for (const auto& c : cases) {
        if (run(c.rows, c.width, c.rank, c.down_shift, c.image, random, compared) != 1) {
            std::printf("controller-image: FAIL rows %u width %u rank %u\n", c.rows, c.width, c.rank);
            return 1;
        }
    }
    std::printf("controller-image: EXACT PASS cases=%zu values=%zu (output + latent bitwise, BF16 operand image vs FP32 "
                "operands)\n", sizeof(cases) / sizeof(cases[0]), compared);
    return 0;
}
