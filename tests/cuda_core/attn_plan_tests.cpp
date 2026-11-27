#include "attn_core/decode_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
namespace attn = strix::cuda::core::attn;
constexpr unsigned kNativeSplitHeadDim = 256;

void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "k4c plan FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}

void check_fp8_tail() {
    namespace fp8 = strix::attn::fp8;
    constexpr unsigned tile = fp8::capacity_alignment;
    const auto oracle = [](unsigned first, unsigned end, unsigned causal_end, const auto& ks,
                           const auto& vs, const auto& v, unsigned stride) {
        const auto loaded = fp8::live_tile_keys(first, end);
        const auto visible = fp8::live_tile_keys(first, std::min(end, causal_end));
        std::array<float, 2> sum{};
        float vmax = 0.0F;
        for (unsigned k = 0; k < loaded; ++k) {
            // K and the two scales are indexed by token, V by feature then token.
            const float key_scale = ks.at(k);
            const float value_scale = vs.at(k);
            const std::array payload{v.at(k), v.at(stride + k)};
            if (k >= visible) continue;
            vmax = std::max(vmax, value_scale);
            for (std::size_t d = 0; d < sum.size(); ++d) sum[d] += payload[d] * value_scale * key_scale;
        }
        return std::array{std::bit_cast<std::uint32_t>(sum[0]), std::bit_cast<std::uint32_t>(sum[1]),
                          std::bit_cast<std::uint32_t>(vmax)};
    };

    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    for (const unsigned first : {0U, 64U, fp8::split_keys}) {
        for (const unsigned count : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U}) {
            std::vector<float> ks(count, 2.0F), vs(count, 0.5F), values(2 * count, 1.0F);
            std::fill(values.begin() + count, values.end(), -2.0F);
            for (const unsigned shown : {0U, count / 2U, count}) {
                const auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };
                const auto visible = static_cast<float>(shown);
                const auto expected = std::array{bits(visible), bits(shown ? -2.0F * visible : 0.0F),
                                                 bits(shown ? 0.5F : 0.0F)};
                check(oracle(first, first + count, first + shown, ks, vs, values, count) == expected,
                      "live inputs match the oracle with no allocated padding");
                for (const auto poison : {std::array{1.0e30F, 448.0F}, std::array{infinity, nan}}) {
                    std::vector<float> pad_ks(tile, poison[0]), pad_vs(tile, poison[0]), pad_v(2 * tile, poison[1]);
                    std::copy_n(ks.begin(), shown, pad_ks.begin());
                    std::copy_n(vs.begin(), shown, pad_vs.begin());
                    std::copy_n(values.begin(), shown, pad_v.begin());
                    std::copy_n(values.begin() + count, shown, pad_v.begin() + tile);
                    check(oracle(first, first + count, first + shown, pad_ks, pad_vs, pad_v, tile) == expected,
                          "poisoned storage and causal suffixes cannot affect either V component or its scale");
                }
            }
        }
    }
    check(fp8::live_tile_keys(0U, 65U) == 64U && fp8::live_tile_keys(128U, 65U) == 0U &&
              fp8::live_tile_keys(0xfffffffeU, 0xffffffffU) == 1U,
          "live lane count handles whole, empty and maximum-index tiles");
}
}  // namespace

int main() {
    check_fp8_tail();
    attn::DecodeSpec prepared{128, 24, 8, 1, 8191, 1024, 1024, 1024, 0.08838834764831844F,
        attn::DecodeOrder::online_bf16, attn::KvLayout::ring, attn::OutputGate::sigmoid_bf16, true};
    auto plan = attn::make_decode_plan(prepared, kNativeSplitHeadDim);
    check(plan && plan->dim_class == 128 && plan->group == 3 && plan->spec.heads == 24 &&
          plan->spec.window == 1024 && plan->spec.relative_extent == 1024 && plan->spec.sinks &&
          plan->spec.layout == attn::KvLayout::ring && plan->splits == 1 &&
          plan->score_floats == 0 && plan->stat_floats == 0 && plan->scratch_floats == 0,
          "prepared ring preserves the single online walk");

    attn::DecodeSpec dense{256, 24, 4, 1, 8191, 8191};
    dense.gate = attn::OutputGate::sigmoid_interleaved;
    plan = attn::make_decode_plan(dense, kNativeSplitHeadDim);
    check(plan && plan->dim_class == 256 && plan->group == 6 && plan->spec.heads == 24 &&
          plan->spec.window == 0 && plan->spec.gate == attn::OutputGate::sigmoid_interleaved &&
          plan->splits == 32 && plan->score_floats == 24U * 8191U && plan->stat_floats == 24U * 32U &&
          plan->scratch_floats == 24U * (8191U + 32U * 258U),
          "dense plan maps a strided GQA group and separate split workspace");

    namespace fp8 = strix::attn::fp8;
    const auto shadow = fp8::make_plan(256, 24, 4, 8193);
    check(shadow && shadow->padded_capacity == 8256 && shadow->splits == 5 &&
          shadow->shadow_bytes() == 4U * 8256U * 520U &&
          shadow->covers(8192, 1) && !shadow->covers(8192, 2),
          "FP8 tail padding, shadow bytes and range checks");
    check(!fp8::make_plan(256, 24, 0, 8193) && !fp8::make_plan(128, 24, 4, 8193) &&
          !fp8::make_plan(256, 24, 4, std::numeric_limits<std::uint32_t>::max()),
          "FP8 rejects unsupported shape and capacity overflow");
    auto e4m3 = dense;
    e4m3.order = attn::DecodeOrder::online_e4m3;
    e4m3.rows = 8;
    plan = attn::make_decode_plan(e4m3, kNativeSplitHeadDim);
    check(plan && plan->splits == 4U && plan->scratch_floats == 8U * 24U * 4U * 258U,
          "FP8 decode and its native shadow use the same split plan");
    e4m3.sinks = true;
    check(!attn::make_decode_plan(e4m3, kNativeSplitHeadDim), "FP8 declines sinks outside the native contract");

    attn::DecodeSpec native{128, 12, 3, 1, 4097, 4097};
    native.gate = attn::OutputGate::sigmoid_strided;
    check(!attn::make_decode_plan(native, kNativeSplitHeadDim), "strided gate requires the native head dimension");
    check(attn::make_decode_plan(native, 128).has_value(), "strided gate uses the caller's native head dimension");
    native.order = attn::DecodeOrder::online_e4m3;
    check(!attn::make_decode_plan(native, 128), "the FP8 shadow plan declines a head dimension it does not cover");
    check(!attn::make_decode_plan(native, kNativeSplitHeadDim), "E4M3 requires the native head dimension");
    check(!attn::make_decode_plan(dense, 0) && !attn::make_decode_plan(dense, 513),
          "reject invalid native split head dimensions");

    attn::DecodeSpec other{96, 12, 3, 3, 513, 39, 37};
    other.layout = attn::KvLayout::ring;
    plan = attn::make_decode_plan(other, kNativeSplitHeadDim);
    check(plan && plan->dim_class == 96 && plan->group == 4 && plan->spec.heads == 12 &&
          plan->spec.window == 37 && plan->splits == 3 &&
          plan->scratch_floats == 36U * (513U + 3U * 98U),
          "independent head dimension, group, window and row count");
    other.head_dim = 80;
    plan = attn::make_decode_plan(other, kNativeSplitHeadDim);
    check(plan && plan->dim_class == 0, "runtime scalar dimension");
    other.order = attn::DecodeOrder::online_bf16;
    check(!attn::make_decode_plan(other, kNativeSplitHeadDim), "unsupported MMA dimension");

    prepared.cache_slots = 1023;
    check(!attn::make_decode_plan(prepared, kNativeSplitHeadDim), "short ring");
    prepared.cache_slots = 1024;
    prepared.rows = 2;
    check(!attn::make_decode_plan(prepared, kNativeSplitHeadDim), "ring lacks room for the earlier live tail");
    prepared.rows = 1;
    prepared.window = 0;
    check(!attn::make_decode_plan(prepared, kNativeSplitHeadDim), "full-prefix ring");

    dense.kv_heads = 5;
    check(!attn::make_decode_plan(dense, kNativeSplitHeadDim), "partial GQA group");
    dense.kv_heads = 4;
    dense.sinks = true;
    check(!attn::make_decode_plan(dense, kNativeSplitHeadDim), "unsupported global-max sink hook");
    dense.sinks = false;
    dense.scale = std::numeric_limits<float>::quiet_NaN();
    check(!attn::make_decode_plan(dense, kNativeSplitHeadDim), "non-finite score scale");
    std::puts("k4c plan PASS");
}
