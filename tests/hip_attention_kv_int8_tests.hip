#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// kv-int8-tok-v1: the decode and window KV writers store the host model's bytes, and the decode,
// window and prefill-group score/value kernels read them with identical results (bitwise), close to
// an FP64 attention over the dequantized cache.
namespace {
using namespace strix;
constexpr std::uint32_t heads = 24, kv_heads = 2, dim = 256, kv_width = kv_heads * dim, width = heads * dim;
constexpr std::uint32_t capacity = 64, rows_per_call = 8, count = 37, maximum = 64;
constexpr std::uint64_t row_bytes = 2ULL * kv_width + 8ULL * kv_heads;
int failures = 0;
void expect(bool ok, const std::string& text) { if (!ok) { std::cerr << "FAIL: " << text << '\n'; ++failures; } }
void check(hipError_t e, const char* text) { if (e != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(e)); }
template <class T> T* device(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }
template <class T> std::vector<T> read(const T* p, std::size_t n) {
    std::vector<T> v(n); check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return v;
}

struct Rng { std::uint64_t s; std::uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<std::uint32_t>(s >> 11); }
    float unit() { return static_cast<float>(next() >> 8) * (1.0F / 16777216.0F) * 2.0F - 1.0F; } };

float round_bf16(float x) {
    auto bits = std::bit_cast<std::uint32_t>(x);
    bits = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    return std::bit_cast<float>(bits);
}

// Host model of one KV row: codes rint(x / s) in [-127, 127] of the BF16-rounded values, s = max|x| / 127.
std::vector<std::uint8_t> quantize_row(const float* k, const float* v) {
    std::vector<std::uint8_t> row(row_bytes);
    for (std::uint32_t half = 0; half < 2U; ++half)
        for (std::uint32_t h = 0; h < kv_heads; ++h) {
            const float* x = (half ? v : k) + h * dim;
            float a = 0.0F;
            for (std::uint32_t f = 0; f < dim; ++f) a = std::max(a, std::fabs(round_bf16(x[f])));
            const float s = a / 127.0F;
            for (std::uint32_t f = 0; f < dim; ++f) {
                const float code = s > 0.0F ? std::clamp(std::nearbyint(round_bf16(x[f]) / s), -127.0F, 127.0F) : 0.0F;
                row[half * kv_width + h * dim + f] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
            }
            std::memcpy(row.data() + 2U * kv_width + 4U * (half * kv_heads + h), &s, 4U);
        }
    return row;
}
float dequantize(const std::vector<std::uint8_t>& state, std::uint32_t token, std::uint32_t half, std::uint32_t h, std::uint32_t f) {
    const auto* row = state.data() + token * row_bytes;
    float s = 0.0F;
    std::memcpy(&s, row + 2U * kv_width + 4U * (half * kv_heads + h), 4U);
    return static_cast<float>(static_cast<std::int8_t>(row[half * kv_width + h * dim + f])) * s;
}
} // namespace

int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) { std::cout << "SKIP: no HIP device\n"; return 77; }

    // Contract sizes: 12 layers x (2 x 512 codes + 4 FP32 scales) = 12480 bytes per token.
    auto model = ModelContract::qwen38_flash_next();
    expect(base_kv_bytes_per_token(model) == 24ULL * 1024ULL, "BF16 KV bytes per token");
    model.attention_kv_storage = AttentionKvStorage::int8_token;
    expect(model.validate().empty(), "int8 KV contract validates");
    expect(base_kv_bytes_per_token(model) == 12ULL * row_bytes && row_bytes == 1040U, "int8 KV bytes per token");
    const auto layout = build_indexed_attention_state_layout(model);
    expect(layout.kv_row_bytes == row_bytes && layout.kv_value_offset == kv_width && layout.kv_scale_offset == 2U * kv_width &&
        layout.kv_scale_bytes == 16U, "int8 KV row layout");

    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    Rng rng{0x51a7c0ffeeULL};
    std::vector<float> k(static_cast<std::size_t>(capacity) * kv_width), v(k.size());
    for (std::uint32_t t = 0; t < capacity; ++t)
        for (std::uint32_t i = 0; i < kv_width; ++i) {
            const auto h = i / dim;
            float scale = 1.0F + static_cast<float>(t % 7U) * 3.0F;
            if (t == 3U && h == 1U) scale = 0.0F;                   // all-zero head
            k[t * kv_width + i] = rng.unit() * scale;
            v[t * kv_width + i] = rng.unit() * scale * 0.25F;
            if (t == 5U && i % dim == 17U) k[t * kv_width + i] = 300.0F; // outlier dominates the head scale
            if (t == 9U && i % dim < 4U) v[t * kv_width + i] = (i % 2U ? 0.5F : -0.5F) * 127.0F; // exact ties of x / s
        }
    auto* dk = device<float>(k.size());
    auto* dv = device<float>(v.size());
    check(hipMemcpy(dk, k.data(), k.size() * 4U, hipMemcpyHostToDevice), "upload k");
    check(hipMemcpy(dv, v.data(), v.size() * 4U, hipMemcpyHostToDevice), "upload v");
    auto* decode_state = device<std::uint8_t>(capacity * row_bytes);
    auto* rows_state = device<std::uint8_t>(capacity * row_bytes);
    check(hipMemset(decode_state, 0x5a, capacity * row_bytes), "fill");
    check(hipMemset(rows_state, 0x5a, capacity * row_bytes), "fill");

    std::vector<float> q(static_cast<std::size_t>(heads) * dim * 2U);
    for (auto& x : q) x = round_bf16(rng.unit() * 2.0F);
    std::vector<std::uint32_t> selected(count);
    for (std::uint32_t i = 0; i < count; ++i) selected[i] = (i * 29U + 11U) % capacity;
    auto* dq = device<float>(q.size());
    auto* dsel = device<std::uint32_t>(count);
    check(hipMemcpy(dq, q.data(), q.size() * 4U, hipMemcpyHostToDevice), "upload q");
    check(hipMemcpy(dsel, selected.data(), count * 4U, hipMemcpyHostToDevice), "upload selected");
    const std::size_t score_elements = static_cast<std::size_t>(heads) * maximum;
    auto* scores_decode = device<float>(score_elements);
    auto* out_decode = device<float>(width);
    auto* scores_rows = device<float>(8U * score_elements);
    auto* out_rows = device<float>(8U * width);
    auto* scores_group = device<float>(16U * score_elements);
    auto* out_group = device<float>(16U * width);

    // Positions in a scattered order; the last call's attention reads the complete cache.
    for (std::uint32_t call = 0; call < capacity / rows_per_call; ++call) {
        HipKvInt8Probe p;
        p.attention_heads = heads; p.kv_heads = kv_heads; p.head_dim = dim; p.count = count; p.maximum_selected = maximum;
        for (std::uint32_t r = 0; r < rows_per_call; ++r) {
            const auto token = (call + r * 8U) % capacity;
            p.k[r] = dk + token * kv_width; p.v[r] = dv + token * kv_width; p.position[r] = token;
        }
        p.decode_state = decode_state; p.rows_state = rows_state; p.q_gate = dq; p.selected = dsel;
        p.scores_decode = scores_decode; p.out_decode = out_decode; p.scores_rows = scores_rows; p.out_rows = out_rows;
        p.scores_group = scores_group; p.out_group = out_group;
        launch_hip_attention_kv_int8_probe(p, stream);
    }
    check(hipStreamSynchronize(stream), "probe");

    const auto decode_bytes = read(decode_state, capacity * row_bytes);
    const auto rows_bytes = read(rows_state, capacity * row_bytes);
    expect(decode_bytes == rows_bytes, "decode and window writers store identical bytes");
    std::uint32_t mismatched_rows = 0;
    double worst = 0.0;
    for (std::uint32_t t = 0; t < capacity; ++t) {
        const auto host = quantize_row(&k[t * kv_width], &v[t * kv_width]);
        if (!std::equal(host.begin(), host.end(), decode_bytes.begin() + t * row_bytes)) ++mismatched_rows;
        for (std::uint32_t half = 0; half < 2U; ++half)
            for (std::uint32_t i = 0; i < kv_width; ++i) {
                const float x = round_bf16((half ? v : k)[t * kv_width + i]);
                float s = 0.0F;
                std::memcpy(&s, host.data() + 2U * kv_width + 4U * (half * kv_heads + i / dim), 4U);
                const double error = std::fabs(static_cast<double>(dequantize(decode_bytes, t, half, i / dim, i % dim)) - x);
                worst = std::max(worst, s > 0.0F ? error / s : error);
            }
    }
    expect(mismatched_rows == 0, "device rows equal the host quantizer (" + std::to_string(mismatched_rows) + " differ)");
    // Half a step plus the FP32 roundings of x / s (|x / s| <= 127, ulp 2^-17) and of code * s.
    expect(worst <= 0.5 + 1e-5, "dequantization error is at most half a step (" + std::to_string(worst) + ")");

    const auto sd = read(scores_decode, score_elements);
    const auto od = read(out_decode, width);
    const auto sr = read(scores_rows, 8U * score_elements);
    const auto orows = read(out_rows, 8U * width);
    const auto sg = read(scores_group, 16U * score_elements);
    const auto og = read(out_group, 16U * width);
    const auto same_scores = [&](const std::vector<float>& all, std::uint32_t r) {
        for (std::uint32_t h = 0; h < heads; ++h)
            for (std::uint32_t i = 0; i < count; ++i)
                if (std::bit_cast<std::uint32_t>(all[r * score_elements + h * maximum + i]) !=
                    std::bit_cast<std::uint32_t>(sd[h * maximum + i])) return false;
        return true;
    };
    const auto same_output = [&](const std::vector<float>& all, std::uint32_t r) {
        return std::memcmp(all.data() + static_cast<std::size_t>(r) * width, od.data(), width * 4U) == 0;
    };
    for (std::uint32_t r = 0; r < 8U; ++r) {
        expect(same_scores(sr, r), "window row " + std::to_string(r) + " probabilities equal decode");
        expect(same_output(orows, r), "window row " + std::to_string(r) + " output equals decode");
    }
    for (std::uint32_t r = 0; r < 16U; ++r) {
        expect(same_scores(sg, r), "prefill-group row " + std::to_string(r) + " probabilities equal decode");
        expect(same_output(og, r), "prefill-group row " + std::to_string(r) + " output equals decode");
    }

    // FP64 attention over the dequantized cache (BF16 probabilities, output and gate as the kernels).
    double max_error = 0.0, max_value = 0.0;
    for (std::uint32_t h = 0; h < heads; ++h) {
        const auto kvh = h / (heads / kv_heads);
        std::vector<double> p(count);
        double m = -1e300;
        for (std::uint32_t i = 0; i < count; ++i) {
            double dot = 0.0;
            for (std::uint32_t f = 0; f < dim; ++f) dot += static_cast<double>(q[h * dim * 2U + f]) * dequantize(decode_bytes, selected[i], 0U, kvh, f);
            p[i] = dot / 16.0; m = std::max(m, p[i]);
        }
        double sum = 0.0;
        for (auto& x : p) { x = std::exp(x - m); sum += x; }
        for (std::uint32_t f = 0; f < dim; ++f) {
            double attn = 0.0;
            for (std::uint32_t i = 0; i < count; ++i) attn += round_bf16(static_cast<float>(p[i] / sum)) * dequantize(decode_bytes, selected[i], 1U, kvh, f);
            const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(q[h * dim * 2U + dim + f])));
            const double expected = attn * gate;
            max_error = std::max(max_error, std::fabs(expected - od[h * dim + f]));
            max_value = std::max(max_value, std::fabs(expected));
        }
    }
    expect(max_error <= 0.02 * max_value, "decode output follows the FP64 dequantized attention (error " +
        std::to_string(max_error) + " of " + std::to_string(max_value) + ")");

    for (void* p : std::initializer_list<void*>{dk, dv, decode_state, rows_state, dq, dsel, scores_decode, out_decode, scores_rows,
             out_rows, scores_group, out_group}) check(hipFree(p), "free");
    check(hipStreamDestroy(stream), "stream destroy");
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "PASS: kv-int8-tok-v1 writers/readers (worst step error " << worst << ", output error " << max_error << ")\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
