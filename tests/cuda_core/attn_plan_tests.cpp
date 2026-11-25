#include "attn_core/decode_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
namespace attn = strix::cuda::core::attn;
constexpr unsigned kNativeSplitHeadDim = 256;

void check(bool ok, const char* message) {
    if (ok) return;
    std::fprintf(stderr, "k4c plan FAIL: %s\n", message);
    std::exit(EXIT_FAILURE);
}
}  // namespace

int main() {
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
