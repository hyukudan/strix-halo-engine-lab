// Mini HIP prefill attention with the relative bias projected inside the flash (the default prefill route) against
// the same forward with the [row][head][distance] image and its producer (project_bias = false): attended, gated and
// output rows bit-identical, the image untouched on the projected route, and the same failure word, including a
// non-finite logit only at a distance no score reads (exactly the bias producer's bit) and subnormal BF16 relative
// weights. Built with the production HIP floating-point flags (IEEE denormals, correctly rounded divide/sqrt).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_attention_prepare_hip.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

namespace {

using namespace strix::inkling2;
namespace bf16 = strix::weight::scalar_float;
constexpr unsigned rows = 80;  // more than the key-split bound and more than one 64-position work-group
constexpr unsigned max_projection_elements = kHiddenWidth * kQueryWidth;
constexpr float sentinel = 7.0F;

struct Arm {
    alignas(16) std::array<float, rows * kQueryWidth> attended{}, gated{};
    alignas(16) std::array<float, rows * kHiddenWidth> output{};
    unsigned failure{};
};

struct Fixture {
    alignas(16) std::array<std::uint16_t, max_projection_elements> matrix{};
    // Separate gain arrays of the sizes the clips read (attn.q_l2_clip.scale, attn.rel_l2_clip.scale).
    alignas(16) std::array<std::uint16_t, kQueryWidth> query_gains{};
    alignas(16) std::array<std::uint16_t, kQueryHeads * kRelativeWidth> relative_gains{};
    alignas(16) std::array<std::uint16_t, kRelativeWidth * kRelativeExtent> relative_weight{};
    alignas(16) std::array<float, rows * kHiddenWidth> input{};
    alignas(16) std::array<float, rows * kQueryWidth> query{}, gate_logits{};
    alignas(16) std::array<float, rows * kKvWidth> raw_keys{}, raw_values{}, keys{}, values{};
    alignas(16) std::array<float, rows * kKvHeads * kConvTaps> dynamic_keys{}, dynamic_values{};
    alignas(16) std::array<float, rows * kQueryHeads * kRelativeWidth> relative_features{};
    alignas(16) std::array<float, rows * kQueryHeads * kRelativeExtent> relative_logits{};
    alignas(16) std::array<float, rows * kQueryHeads> sinks{};
    alignas(16) std::array<float, kConvTaps * kHiddenWidth> lags{};
    alignas(16) std::array<float, (kConvTaps - 1U) * kKvWidth> key_history{}, value_history{};
    alignas(16) std::array<std::uint16_t, rows * kKvWidth> key_cache{}, value_cache{};
    alignas(16) std::array<std::uint16_t, kHeadWidth> padding{};
    Arm arms[2];
};

void require_hip(hipError_t status) {
    if (status == hipSuccess) return;
    std::cerr << hipGetErrorString(status) << '\n';
    std::exit(1);
}

struct Lcg {
    std::uint32_t state;
    float next(float scale) {
        state = state * 1664525U + 1013904223U;
        return scale * (static_cast<float>(state >> 8U) * 0x1p-23F - 1.0F);
    }
};

void run(Fixture& f, LayerVisit visit, bool project, hipStream_t stream) {
    f.key_history.fill(0.0F);
    f.value_history.fill(0.0F);
    f.key_cache.fill(0U);
    f.value_cache.fill(0U);
    f.relative_logits.fill(sentinel);
    Arm& arm = f.arms[project ? 1 : 0];
    arm.failure = 0U;
    const MiniProjectionInputs input{f.input.data(), f.input.data(), f.input.data()};
    MiniProjectionWeights weights;
    weights.matrix.fill(f.matrix.data());
    weights.matrix[static_cast<unsigned>(MiniAttentionProjection::relative_bias)] = f.relative_weight.data();
    weights.query_gain = f.query_gains.data();
    weights.relative_gain = f.relative_gains.data();
    weights.key_convolution = f.lags.data();
    weights.value_convolution = f.lags.data();
    const MiniProjectionWorkspace work{f.raw_keys.data(), f.raw_values.data(), f.dynamic_keys.data(),
        f.dynamic_values.data(), f.relative_features.data(), f.key_history.data(), f.value_history.data()};
    const MiniPreparedAttention prepared{f.query.data(), f.keys.data(), f.values.data(),
        f.relative_logits.data(), f.sinks.data(), f.gate_logits.data()};
    const MiniAttentionOutputRows output{arm.attended.data(), arm.gated.data(), arm.output.data()};
    const MiniHipAttentionCache cache{f.key_cache.data(), f.value_cache.data(), f.padding.data(), rows, false};
    require_hip(enqueue_mini_hip_attention_forward(visit, 0, rows, input, weights, work, prepared, output, cache,
                                                   &arm.failure, stream, {}, 1U, nullptr, project));
    require_hip(hipStreamSynchronize(stream));
}

bool same(const Arm& a, const Arm& b) {
    return a.failure == b.failure && std::memcmp(a.attended.data(), b.attended.data(), sizeof(a.attended)) == 0 &&
           std::memcmp(a.gated.data(), b.gated.data(), sizeof(a.gated)) == 0 &&
           std::memcmp(a.output.data(), b.output.data(), sizeof(a.output)) == 0;
}

bool image_untouched(const Fixture& f) {
    for (const float v : f.relative_logits)
        if (std::memcmp(&v, &sentinel, sizeof(v)) != 0) return false;
    return true;
}

bool compare(Fixture& f, LayerVisit visit, unsigned expected_failure, hipStream_t stream) {
    run(f, visit, false, stream);
    const bool image_written = !image_untouched(f);
    run(f, visit, true, stream);
    // Bit 1 is the relative-bias producer's flag. The whole word must match: a flag of a stage both routes share (the
    // query clip, the flash half-range check) would hide the bias-only check behind a shared failure.
    const bool pass = image_written && image_untouched(f) && same(f.arms[0], f.arms[1]) &&
                      f.arms[0].failure == expected_failure;
    std::cerr << "layer " << visit.physical_layer << " failure " << f.arms[0].failure << '/' << f.arms[1].failure
                  << " image written " << image_written << " outputs equal " << same(f.arms[0], f.arms[1]) << '\n';
    return pass;
}

}  // namespace

int main() {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* f = std::construct_at(static_cast<Fixture*>(storage));
    Lcg lcg{12345U};
    for (auto& w : f->matrix) w = bf16::bf16_encode(lcg.next(0.01F));
    for (auto& w : f->query_gains) w = bf16::bf16_encode(1.0F + lcg.next(0.5F));
    for (auto& w : f->relative_gains) w = bf16::bf16_encode(1.0F + lcg.next(0.5F));
    for (auto& w : f->relative_weight) w = bf16::bf16_encode(lcg.next(0.5F));
    for (auto& x : f->input) x = lcg.next(0.5F);
    for (auto& x : f->lags) x = lcg.next(0.5F);
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    constexpr auto schedule = mini_schedule();
    bool pass = true;
    for (const LayerVisit visit : {schedule[0], schedule[5]}) pass = compare(*f, visit, 0U, stream) && pass;
    // A non-finite weight at the last distance, which no score of these rows reads: both routes flag it.
    const auto kept = f->relative_weight[kRelativeExtent - 1U];
    f->relative_weight[kRelativeExtent - 1U] = bf16::bf16_encode(std::numeric_limits<float>::infinity());
    pass = compare(*f, schedule[0], 1U, stream) && pass;
    f->relative_weight[kRelativeExtent - 1U] = kept;
    // Subnormal BF16 relative weights (exponent 0, nonzero mantissa, both signs) in every third element: the projected
    // and image routes must keep them alike under the production denormal mode.
    const auto normal = f->relative_weight;
    for (std::size_t i = 0; i < f->relative_weight.size(); i += 3U)
        f->relative_weight[i] = static_cast<std::uint16_t>((i % 2U != 0U ? 0x8000U : 0U) | (1U + (i * 37U) % 127U));
    for (const LayerVisit visit : {schedule[0], schedule[5]}) pass = compare(*f, visit, 0U, stream) && pass;
    f->relative_weight = normal;
    if (!pass) std::cerr << "Mini HIP projected-bias prefill differs from the image route\n";
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(f);
    require_hip(hipFree(f));
    return pass ? 0 : 1;
}
