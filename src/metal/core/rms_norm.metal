#include <metal_stdlib>
using namespace metal;

struct RmsNormParameters {
    uint width;
    uint rows;
    uint use_scale;
    float epsilon;
};

kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant RmsNormParameters& params [[buffer(3)]],
                     uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) return;
    uint base = row * params.width;
    float sum = 0.0f;
    for (uint i = 0; i < params.width; ++i) sum += x[base + i] * x[base + i];
    float mean = sum / float(params.width) + params.epsilon;
    float inv = isfinite(mean) ? 1.0f / sqrt(mean) : as_type<float>(0x7fc00000u);
    for (uint i = 0; i < params.width; ++i)
        y[base + i] = (x[base + i] * inv) * (params.use_scale != 0u ? w[i] : 1.0f);
}
