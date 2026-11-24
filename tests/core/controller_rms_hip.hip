// Row-owner controller: enqueue_low_rank_controller stores the same output bits whether the backend runs the gate pair
// with the controller epilogue followed by the BF16-rows norm (bf16_controller_kernel + rmsnorm_strided_kernel) or the
// row-owner launch (controller_rms_rows: gates, controller and norm in one launch, conditioned rows normalized in
// place), in place (output = raw) and apart, and keeps the two launches for decode rows.
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
#include <initializer_list>
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

// The backend operations of a resident model: dense = shared BF16 rows, gates with the controller epilogue, norm =
// the shared BF16-rows norm (parallel form) with a BF16 scale.
struct Operations {
    unsigned width;
    unsigned rms_launches = 0;
    static constexpr float kEpsilon = 1e-6F;
    hipStream_t stream() const { return nullptr; }
    void check(hipError_t status) const {
        if (!ok(status)) throw std::runtime_error("controller-rms: launch failed");
    }
    void dense(const float* input, unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections) {
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
    void rms(const float* input, const std::uint16_t* scale, float* output, unsigned rows) {
        ++rms_launches;
        check(strix::ops::norm::enqueue_rms_bf16_rows({input, scale, output, rows, width, width, width}, kEpsilon,
                                                      strix::ops::norm::RmsRowsReduction::parallel, nullptr));
    }
};

struct RowOwnerOperations : Operations {
    unsigned owned_launches = 0;
    bool dense_controller_rms(const float* input, unsigned input_width, unsigned rows,
                              std::span<const Bf16RowProjection> gates, const mix::ControllerArgs& control,
                              const std::uint16_t* scale) {
        const auto status = rows_ns::enqueue_bf16_controller_rms_rows(input, input_width, rows, gates, control.current,
                                                                      control.message, scale, kEpsilon, control.output,
                                                                      nullptr);
        if (status == hipErrorNotSupported) return false;
        check(status);
        ++owned_launches;
        return true;
    }
};

void release(std::initializer_list<void*> pointers) {
    for (void* p : pointers)
        if (p) (void)hipFree(p);
}

// 1 = equal (row-owner form taken iff expected), 0 = differ / error. in_place: output is the raw rows.
int run(unsigned rows, unsigned width, unsigned rank, bool in_place, bool expect_owned, std::mt19937& random,
        std::size_t& compared) {
    const mix::LowRankController plan{width, rank, 2U, 1e-6F, true, true};
    const std::size_t elements = std::size_t{rows} * width, scratch_elements = plan.scratch_elements(rows);
    const auto raw_host = values(elements, random, 3.0F, true);
    auto* raw_a = upload(raw_host);
    auto* raw_b = upload(raw_host);
    auto* bank0 = upload(values(elements, random, 1.5F, false));
    auto* bank1 = upload(values(elements, random, 1.5F, true));
    auto* query = upload(values(width, random, 0.2F, false));
    auto* down = upload(bf16_weights(std::size_t{rank} * 2U * width, random, 0.05F));
    auto* forget = upload(bf16_weights(std::size_t{width} * rank, random, 0.2F));
    auto* gain = upload(bf16_weights(std::size_t{width} * rank, random, 0.2F));
    auto* norm = upload(bf16_weights(width, random, 1.0F));
    float *out_a = nullptr, *out_b = nullptr, *scratch_a = nullptr, *scratch_b = nullptr;
    int result = 0;
    if (raw_a && raw_b && bank0 && bank1 && query && down && forget && gain && norm &&
        ok(hipMalloc(&out_a, elements * sizeof(float))) && ok(hipMalloc(&out_b, elements * sizeof(float))) &&
        ok(hipMalloc(&scratch_a, scratch_elements * sizeof(float))) &&
        ok(hipMalloc(&scratch_b, scratch_elements * sizeof(float))) &&
        ok(hipMemset(out_b, 0x5A, elements * sizeof(float))) &&
        ok(hipMemset(scratch_b, 0x5A, scratch_elements * sizeof(float)))) {
        const mix::ConstRows banks[] = {{bank0, elements}, {bank1, elements}};
        const mix::ControllerWeights weights{{query, width}, down, forget, gain, norm};
        float* const target_a = in_place ? raw_a : out_a;
        float* const target_b = in_place ? raw_b : out_b;
        Operations base{width};
        RowOwnerOperations owned{{width}};
        bool launched = true;
        try {
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw_a, elements}, banks, {target_a, elements},
                                             {scratch_a, scratch_elements}, base);
            mix::enqueue_low_rank_controller(plan, weights, rows, {raw_b, elements}, banks, {target_b, elements},
                                             {scratch_b, scratch_elements}, owned);
        } catch (const std::exception&) {
            launched = false;
        }
        const bool took_owned = owned.owned_launches == 1U && owned.rms_launches == 0U && base.rms_launches == 1U;
        const bool kept_two = owned.owned_launches == 0U && owned.rms_launches == 1U;
        std::vector<float> a(elements), b(elements);
        if (launched && ok(hipDeviceSynchronize()) && (expect_owned ? took_owned : kept_two) &&
            ok(hipMemcpy(a.data(), target_a, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            ok(hipMemcpy(b.data(), target_b, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
            std::memcmp(a.data(), b.data(), elements * sizeof(float)) == 0) {
            result = 1;
            compared += elements;
        }
    }
    release({raw_a, raw_b, bank0, bank1, query, down, forget, gain, norm, out_a, out_b, scratch_a, scratch_b});
    return result;
}

}  // namespace

// Host-only admission of the row-owner launch (no device access: every refused call returns before a launch).
static_assert(strix::core::controller_rms_rows::grid_blocks(0U) == 0U);
static_assert(strix::core::controller_rms_rows::grid_blocks(16U) == 1U);
static_assert(strix::core::controller_rms_rows::grid_blocks(17U) == 2U);
static_assert(strix::core::controller_rms_rows::grid_blocks(0xFFFFFFFFU) == 0x10000000U);

static int admission_refusals() {
    namespace crr = strix::core::controller_rms_rows;
    alignas(16) static unsigned char arena[256U * 1024U];
    const auto at = [](std::size_t offset) { return static_cast<void*>(arena + offset); };
    // width 272, rank 16, 32 rows: latent 2 KiB, panels 8.5 KiB, row blocks 34 KiB, scale 544 B, all disjoint.
    crr::Args base{static_cast<const float*>(at(0U)),           static_cast<const std::uint16_t*>(at(4096U)),
                   static_cast<const std::uint16_t*>(at(16384U)), static_cast<const float*>(at(32768U)),
                   static_cast<const float*>(at(73728U)),        static_cast<const std::uint16_t*>(at(155648U)),
                   static_cast<float*>(at(114688U)),             32U, 272U, 16U, 1.0e-6F};
    const auto output_bytes = [&](unsigned offset) { return static_cast<const std::uint16_t*>(at(114688U + offset)); };
    int failures = 0;
    const auto expect = [&](const char* name, bool got, bool want) {
        if (got != want) {
            std::printf("controller-rms: FAIL admission %s admitted %d expected %d\n", name, got ? 1 : 0, want ? 1 : 0);
            ++failures;
        }
    };
    expect("disjoint", crr::admits(base, 16U), true);
    auto a = base;
    a.scale = nullptr;
    expect("unscaled", crr::admits(a, 16U), true);
    a = base;
    a.scale = output_bytes(0U);  // the scale's 544 bytes inside the first output row
    expect("output_holds_scale", crr::admits(a, 16U), false);
    a = base;
    a.forget = output_bytes(34816U - 4096U);  // forget panel overlaps the last output rows
    expect("output_holds_forget", crr::admits(a, 16U), false);
    a = base;
    a.gain = output_bytes(1024U);
    expect("output_holds_gain", crr::admits(a, 16U), false);
    a = base;
    a.rank = 20U;  // not a K16 multiple: the raw launch refuses instead of running the rank-16 specialization
    expect("rank20", crr::admits(a, 16U), false);
    expect("rank20_raw_launch", crr::enqueue(a, nullptr) == hipSuccess, false);
    a = base;
    a.scale = output_bytes(0U);
    expect("alias_raw_launch", crr::enqueue(a, nullptr) == hipSuccess, false);
    return failures == 0 ? 1 : 0;
}

int main() {
    if (admission_refusals() != 1) return 1;
    std::mt19937 random(20261124U);
    std::size_t compared = 0;
    // (rows, width, rank, in place, row-owner expected): the Inkling controller (width 2304, rank 64) with full and
    // partial 16-row blocks, in place, narrow widths with rank 16 and 48 (a K16 tail in the incumbent tile), rows of
    // 256 values (the separate norm's one-wave form: two launches) and decode rows that keep the two launches.
    const struct { unsigned rows, width, rank; bool in_place, owned; } cases[] = {
        {1000U, 2304U, 64U, false, true}, {300U, 2304U, 64U, true, true},  {17U, 2304U, 64U, false, true},
        {129U, 384U, 16U, false, true},   {77U, 384U, 48U, true, true},    {77U, 256U, 48U, true, false},
        {129U, 256U, 16U, false, false},  {4U, 2304U, 64U, false, false},  {16U, 256U, 64U, true, false},
    };
    for (const auto& c : cases) {
        if (run(c.rows, c.width, c.rank, c.in_place, c.owned, random, compared) != 1) {
            std::printf("controller-rms: FAIL rows %u width %u rank %u in_place %d\n", c.rows, c.width, c.rank,
                        c.in_place ? 1 : 0);
            return 1;
        }
    }
    std::printf("controller-rms: EXACT PASS cases=%zu values=%zu (output bitwise, row-owner gates + controller + norm vs "
                "controller tile + BF16-rows norm)\n", sizeof(cases) / sizeof(cases[0]), compared);
    return 0;
}
