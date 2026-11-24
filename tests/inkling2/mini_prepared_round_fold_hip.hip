// Isolated identity of the HIP prepared attention's round fold (no model): the stage with the fold off (every plain
// BF16 round its own launch) and on (the K/V convolutions round their reads and stores, the output projection rounds
// each completed sum in its store at prefill heights) stores the same bits, for the base, the operand-image, the
// projected-bias and the image + projected-bias backends, on local and full attention visits, at decode / short /
// prefill heights, over consecutive chunks of one stream (the convolution histories and positions carry across the
// chunk boundaries) and with NaN / Inf operands next to finite ones: prepared outputs, both convolution histories,
// the failure word, the gated rows and the attention output of the output stage, and the post convolution's output
// and history. The fold arm must also launch fewer kernels than the unfolded one (captured graph nodes), so the case
// cannot pass with the fold not reaching the stage.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "inkling2/inkling2_mini_attention_prepare_hip.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace {

using namespace strix::inkling2;
namespace sf = strix::weight::scalar_float;
namespace gqa = strix::attn_core;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL mini-prepared-round-fold %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

struct Rng {
    std::uint64_t s;
    std::uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<std::uint32_t>(s >> 11); }
    float unit() { return static_cast<float>(next() & 0xFFFFFFU) / 16777216.0F * 2.0F - 1.0F; }
};

template <class T> T* device_copy(const std::vector<T>& host) {
    T* d = nullptr;
    ok(hipMalloc(&d, host.size() * sizeof(T) + 16U), "malloc");
    ok(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
template <class T> T* device_zeros(std::size_t n) { return device_copy(std::vector<T>(n)); }

std::vector<std::uint16_t> weights_bf16(std::size_t n, float scale, Rng& rng, bool nan) {
    std::vector<std::uint16_t> w(n);
    for (auto& v : w) v = sf::bf16_encode(rng.unit() * scale);
    if (nan && n > 4096U) {
        constexpr std::uint16_t patterns[] = {0x7FC1U, 0xFFC1U, 0x7F81U, 0x7F80U, 0xFF80U};
        for (unsigned k = 0; k < 5U; ++k) w[(977U * (k + 1U)) % n] = patterns[k];
    }
    return w;
}

struct Arm {
    std::vector<std::uint32_t> bits;  // every compared output, in order
    std::size_t nodes = 0;            // kernel launches of the prepare + output stages (captured graph nodes)
};

void append(Arm& arm, const float* device, std::size_t n) {
    std::vector<float> host(n);
    ok(hipMemcpy(host.data(), device, n * sizeof(float), hipMemcpyDeviceToHost), "download");
    for (const float v : host) arm.bits.push_back(sf::f2u(v));
}

std::size_t graph_nodes(hipStream_t stream, const auto& body) {
    ok(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed), "begin capture");
    body();
    hipGraph_t graph{};
    ok(hipStreamEndCapture(stream, &graph), "end capture");
    std::size_t count = 0;
    ok(hipGraphGetNodes(graph, nullptr, &count), "graph nodes");
    ok(hipGraphDestroy(graph), "graph destroy");
    return count;
}

struct Operands {
    std::array<std::uint16_t*, MiniProjectionWeights::role_count> matrix{};
    std::uint16_t *query_gain{}, *relative_gain{};
    float *key_lags{}, *value_lags{}, *post_lags{};
};

Operands make_operands(Rng& rng, bool nan) {
    Operands o;
    using Role = MiniAttentionProjection;
    for (unsigned i = 0; i < o.matrix.size(); ++i) {
        const auto spec = mini_attention_projection_spec(static_cast<Role>(i));
        o.matrix[i] = device_copy(weights_bf16(std::size_t{spec.input_width} * spec.output_width, 0.04F, rng,
                                               nan && (static_cast<Role>(i) == Role::value ||
                                                       static_cast<Role>(i) == Role::output)));
    }
    std::vector<std::uint16_t> gain(kQueryWidth), rgain(kQueryHeads * kRelativeWidth);
    for (auto& g : gain) g = sf::bf16_encode(0.5F + 0.5F * std::abs(rng.unit()));
    for (auto& g : rgain) g = sf::bf16_encode(0.5F + 0.5F * std::abs(rng.unit()));
    o.query_gain = device_copy(gain);
    o.relative_gain = device_copy(rgain);
    const auto lags = [&](std::size_t channels) {
        std::vector<float> l(channels * kConvTaps);
        for (auto& v : l) v = rng.unit() * 0.5F;
        return device_copy(l);
    };
    o.key_lags = lags(kKvWidth);
    o.value_lags = lags(kKvWidth);
    o.post_lags = lags(kHiddenWidth);
    return o;
}

void release(Operands& o) {
    for (auto* m : o.matrix) (void)hipFree(m);
    for (void* p : {static_cast<void*>(o.query_gain), static_cast<void*>(o.relative_gain), static_cast<void*>(o.key_lags),
                    static_cast<void*>(o.value_lags), static_cast<void*>(o.post_lags)})
        (void)hipFree(p);
}

// One arm: the chunks of one stream through `make_backend(fold)`.
template <class MakeBackend>
Arm run_arm(const Operands& o, bool full, std::span<const unsigned> chunks, const std::vector<float>& input_all,
            const std::vector<float>& attended_all, bool fold, MakeBackend make_backend, bool projected,
            hipStream_t stream) {
    using Role = MiniAttentionProjection;
    const unsigned max_rows = *std::max_element(chunks.begin(), chunks.end());
    float* x = device_zeros<float>(std::size_t{max_rows} * kHiddenWidth);
    MiniProjectionWeights weights;
    for (unsigned i = 0; i < o.matrix.size(); ++i) weights.matrix[i] = o.matrix[i];
    weights.query_gain = o.query_gain;
    weights.relative_gain = o.relative_gain;
    weights.key_convolution = o.key_lags;
    weights.value_convolution = o.value_lags;
    float* raw_keys = device_zeros<float>(std::size_t{max_rows} * kKvWidth);
    float* raw_values = device_zeros<float>(std::size_t{max_rows} * kKvWidth);
    float* dyn_keys = device_zeros<float>(std::size_t{max_rows} * kKvHeads * kConvTaps);
    float* dyn_values = device_zeros<float>(std::size_t{max_rows} * kKvHeads * kConvTaps);
    float* features = device_zeros<float>(std::size_t{max_rows} * kQueryHeads * kRelativeWidth);
    float* key_history = device_zeros<float>((kConvTaps - 1U) * kKvWidth);
    float* value_history = device_zeros<float>((kConvTaps - 1U) * kKvWidth);
    float* post_history = device_zeros<float>((kConvTaps - 1U) * kHiddenWidth);
    const MiniProjectionWorkspace work{raw_keys, raw_values, dyn_keys, dyn_values, features, key_history, value_history};
    float* query = device_zeros<float>(std::size_t{max_rows} * kQueryWidth);
    float* key = device_zeros<float>(std::size_t{max_rows} * kKvWidth);
    float* value = device_zeros<float>(std::size_t{max_rows} * kKvWidth);
    float* logits = device_zeros<float>(std::size_t{max_rows} * kQueryHeads * kRelativeExtent);
    float* sinks = device_zeros<float>(std::size_t{max_rows} * kQueryHeads);
    float* gate_logits = device_zeros<float>(std::size_t{max_rows} * kQueryWidth);
    const MiniPreparedAttention prepared{query, key, value, logits, sinks, gate_logits};
    float* attended = device_zeros<float>(std::size_t{max_rows} * kQueryWidth);
    float* gated = device_zeros<float>(std::size_t{max_rows} * kQueryWidth);
    float* output = device_zeros<float>(std::size_t{max_rows} * kHiddenWidth);
    float* post = device_zeros<float>(std::size_t{max_rows} * kHiddenWidth);
    unsigned* failure = device_zeros<unsigned>(1);
    std::uint16_t* image = device_zeros<std::uint16_t>(std::size_t{max_rows} * std::max(kHiddenWidth, kQueryWidth) + 8U);
    const auto backend = make_backend(fold, std::span<std::uint16_t>(image, std::size_t{max_rows} * std::max(kHiddenWidth, kQueryWidth)));
    Arm arm;
    unsigned position = 0, offset = 0;
    for (const unsigned rows : chunks) {
        ok(hipMemcpy(x, input_all.data() + std::size_t{offset} * kHiddenWidth, std::size_t{rows} * kHiddenWidth * 4U,
                     hipMemcpyHostToDevice), "x");
        ok(hipMemcpy(attended, attended_all.data() + std::size_t{offset} * kQueryWidth,
                     std::size_t{rows} * kQueryWidth * 4U, hipMemcpyHostToDevice), "attended");
        const MiniProjectionInputs inputs{x, x, x};
        const auto stages = [&] {
            ok(gqa::enqueue_prepared_gqa<MiniPreparedGqaShape>(backend, full, position, rows, inputs, weights, work,
                                                               prepared, failure), "prepare");
            ok(gqa::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(backend, rows, o.matrix[static_cast<unsigned>(Role::output)],
                                                                      attended, gate_logits, gated, output), "output");
        };
        arm.nodes += graph_nodes(stream, stages);  // a capture does not run: the launches follow
        stages();
        ok(gqa::enqueue_prepared_gqa_post_convolution<MiniPreparedGqaShape>(backend, rows, o.post_lags, output, post_history,
                                                                            post), "post convolution");
        ok(hipStreamSynchronize(stream), "sync");
        append(arm, query, std::size_t{rows} * kQueryWidth);
        append(arm, key, std::size_t{rows} * kKvWidth);
        append(arm, value, std::size_t{rows} * kKvWidth);
        if (!projected) append(arm, logits, std::size_t{rows} * kQueryHeads * kRelativeExtent);
        append(arm, sinks, std::size_t{rows} * kQueryHeads);
        append(arm, gate_logits, std::size_t{rows} * kQueryWidth);
        append(arm, gated, std::size_t{rows} * kQueryWidth);
        append(arm, output, std::size_t{rows} * kHiddenWidth);
        append(arm, post, std::size_t{rows} * kHiddenWidth);
        append(arm, key_history, (kConvTaps - 1U) * kKvWidth);
        append(arm, value_history, (kConvTaps - 1U) * kKvWidth);
        append(arm, post_history, (kConvTaps - 1U) * kHiddenWidth);
        unsigned word = 0;
        ok(hipMemcpy(&word, failure, 4, hipMemcpyDeviceToHost), "failure");
        arm.bits.push_back(word);
        position += rows;
        offset += rows;
    }
    for (void* p : {static_cast<void*>(x), static_cast<void*>(raw_keys), static_cast<void*>(raw_values),
                    static_cast<void*>(dyn_keys), static_cast<void*>(dyn_values), static_cast<void*>(features),
                    static_cast<void*>(key_history), static_cast<void*>(value_history), static_cast<void*>(post_history),
                    static_cast<void*>(query), static_cast<void*>(key), static_cast<void*>(value), static_cast<void*>(logits),
                    static_cast<void*>(sinks), static_cast<void*>(gate_logits), static_cast<void*>(attended),
                    static_cast<void*>(gated), static_cast<void*>(output), static_cast<void*>(post),
                    static_cast<void*>(failure), static_cast<void*>(image)})
        (void)hipFree(p);
    return arm;
}

template <class MakeBackend>
bool compare(const char* variant, const Operands& o, bool full, std::span<const unsigned> chunks, bool nan, Rng& rng,
             MakeBackend make_backend, bool projected, hipStream_t stream) {
    unsigned total = 0;
    for (const auto c : chunks) total += c;
    std::vector<float> input(std::size_t{total} * kHiddenWidth), attended(std::size_t{total} * kQueryWidth);
    for (auto& v : input) v = rng.unit();
    for (auto& v : attended) v = rng.unit() * 2.0F;
    if (nan) {
        constexpr std::uint32_t patterns[] = {0x7FC12345U, 0xFFC12345U, 0x7F800001U, 0x7F800000U, 0xFF800000U};
        for (unsigned k = 0; k < 5U; ++k) {
            input[(std::size_t{37U + 211U * k} % total) * kHiddenWidth + 11U * k] = sf::u2f(patterns[k]);
            attended[(std::size_t{19U + 97U * k} % total) * kQueryWidth + 5U * k] = sf::u2f(patterns[(k + 1U) % 5U]);
        }
    }
    const auto off = run_arm(o, full, chunks, input, attended, false, make_backend, projected, stream);
    const auto on = run_arm(o, full, chunks, input, attended, true, make_backend, projected, stream);
    const bool same = off.bits == on.bits;
    const bool fewer = on.nodes < off.nodes;
    std::size_t differing = 0, nans = 0, payload_nans = 0;
    for (std::size_t i = 0; i < std::min(off.bits.size(), on.bits.size()); ++i) differing += off.bits[i] != on.bits[i];
    for (const auto b : on.bits) {
        const bool is_nan = (b & 0x7fffffffU) > 0x7f800000U;
        nans += is_nan;
        payload_nans += is_nan && (b & 0xffffU) != 0U;
    }
    // A NaN case that stored no NaN tested nothing about NaNs: it does not pass.
    const bool reached = !nan || nans != 0U;
    const bool pass = same && fewer && reached;
    std::printf("%s round-fold-prepared %s %s chunks=%zu first=%u nan=%d outputs=%zu differing=%zu nan_outputs=%zu "
                "low_payload_nan_outputs=%zu launches off=%zu on=%zu\n",
                pass ? "PASS" : "FAIL", variant, full ? "full" : "local", chunks.size(), chunks[0], nan ? 1 : 0,
                off.bits.size(), differing, nans, payload_nans, off.nodes, on.nodes);
    return pass;
}

}  // namespace

int main() {
    hipStream_t stream{};
    ok(hipStreamCreate(&stream), "stream");
    Rng rng{0x9E3779B97F4A7C15ULL};
    bool all = true;
    namespace detail_ = strix::inkling2::detail;
    const auto base = [&](bool fold, std::span<std::uint16_t>) { return detail_::MiniPreparedHipBackend{{stream}, fold}; };
    const auto projected_backend = [&](bool fold, std::span<std::uint16_t>) {
        return detail_::MiniPreparedHipProjectedBackend{{{stream}, fold}};
    };
    const auto image_backend = [&](bool fold, std::span<std::uint16_t> image) {
        return detail_::MiniPreparedHipImageBackend{{{stream}, fold}, image};
    };
    const auto image_projected = [&](bool fold, std::span<std::uint16_t> image) {
        return detail_::MiniPreparedHipImageProjectedBackend{{{{stream}, fold}, image}};
    };
    // Heights: decode (1, 2), a verify window (16), the tight-row bound + 1 (17), short prefill (64, 300), prefill
    // (4096, 4111), and chunk boundaries (rows continue across calls).
    const std::vector<std::vector<unsigned>> shapes = {{1U}, {1U, 1U, 2U}, {16U, 17U}, {64U, 1U, 300U}, {4096U, 15U}};
    for (const bool nan : {false, true}) {
        auto operands = make_operands(rng, nan);
        for (const bool full : {false, true}) {
            for (const auto& chunks : shapes) {
                if (nan && chunks.size() == 1U && chunks[0] == 1U) continue;
                all = compare("base", operands, full, chunks, nan, rng, base, false, stream) && all;
                all = compare("image", operands, full, chunks, nan, rng, image_backend, false, stream) && all;
                all = compare("projected-bias", operands, full, chunks, nan, rng, projected_backend, true, stream) && all;
                all = compare("image+projected-bias", operands, full, chunks, nan, rng, image_projected, true, stream) && all;
            }
        }
        release(operands);
    }
    std::puts(all ? "mini_prepared_round_fold_hip: PASS" : "mini_prepared_round_fold_hip: FAIL");
    ok(hipStreamDestroy(stream), "destroy");
    return all ? 0 : 1;
}
