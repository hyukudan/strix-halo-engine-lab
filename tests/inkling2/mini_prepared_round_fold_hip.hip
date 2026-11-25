// Isolated identity of the HIP prepared attention's round fold (no model): the stage with the fold off (every plain
// BF16 round its own launch) and on (the K/V convolutions round their reads and stores, the output projection rounds
// each completed sum in its store at prefill heights) stores the same bits, for the base, the operand-image, the
// projected-bias and the image + projected-bias backends, on local and full attention visits, at decode / short /
// prefill heights, over consecutive chunks of one stream (the convolution histories and positions carry across the
// chunk boundaries) and with NaN / Inf operands next to finite ones: prepared outputs, both convolution histories,
// the failure word, the gated rows and the attention output of the output stage, and the post convolution's output
// and history. The launches of the two stages are counted apart (captured graph nodes of the prepare stage and of the
// output stage): the prepare stage must launch fewer kernels folded (the K/V convolutions), and the output stage must
// launch exactly one kernel less per chunk whose rows exceed the tight-row bound (the output projection's store rounds,
// the separate round pass is gone) and the same count at tight rows, so a forced separate output round fails the case
// whatever the K/V folds save. On the image backends the route of the output stage is observed (the gate stores the
// BF16 operand image the projection reads, instead of the FP32 gated rows), the 4096-row chunks must take it, and the
// image bytes are compared with the BF16 encoding of the FP32 gate (the gated scratch is bypassed on that route).
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
    std::size_t prepare_nodes = 0;    // kernel launches of the prepare stage (captured graph nodes)
    std::size_t output_nodes = 0;     // kernel launches of the output stage (captured graph nodes)
    std::size_t output_rounds = 0;    // separate output round passes the output stage enqueued
    std::size_t image_chunks = 0;     // chunks whose output stage took the gate-image route
    std::size_t big_image_chunks = 0; // ... of which at 4096 rows or more
    std::size_t image_words = 0;      // gate image words compared with the FP32 gate reference
    std::size_t image_mismatch = 0;   // ... that differ from its BF16 encoding
    std::size_t route_faults = 0;     // chunks whose gate / gate-image launches contradict the admitted route
};

// What the output stage asked of the backend while its launches were captured.
struct StageCounts {
    bool on = false;
    unsigned gate = 0, gate_image = 0, round = 0, plans = 0, admitted = 0;
};

// The backend as the stage sees it, counting the output stage's calls (the production backend's own methods run).
template <class Base>
struct Probe : Base {
    StageCounts* counts{};
    auto gate(const float* input, const float* logits, float* output, std::size_t elements) const {
        if (counts->on) ++counts->gate;
        return Base::gate(input, logits, output, elements);
    }
    auto round(float* row, std::size_t elements, unsigned* failure = nullptr) const {
        if (counts->on) ++counts->round;
        return Base::round(row, elements, failure);
    }
    auto output_image_plan(unsigned width, unsigned rows, std::span<const MiniAttentionLinear> views, const float* input,
                           const float* logits) const requires requires { &Base::output_image_plan; } {
        const auto plan = Base::output_image_plan(width, rows, views, input, logits);
        if (counts->on) { ++counts->plans; counts->admitted += plan.admitted() ? 1U : 0U; }
        return plan;
    }
    auto gate_image(const float* input, const float* logits, std::size_t elements) const
        requires requires { &Base::gate_image; } {
        if (counts->on) ++counts->gate_image;
        return Base::gate_image(input, logits, elements);
    }
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
// The fold's own bound for the output store: a chunk of more rows than the tight-row decode kernels take.
constexpr bool folds_output_store(unsigned rows) { return rows > strix::core::gemv_rows::max_rows; }

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
    float* gate_reference = device_zeros<float>(std::size_t{max_rows} * kQueryWidth);
    StageCounts counts;
    const auto backend = make_backend(fold, std::span<std::uint16_t>(image, std::size_t{max_rows} * std::max(kHiddenWidth, kQueryWidth)),
                                      &counts);
    Arm arm;
    unsigned position = 0, offset = 0;
    for (const unsigned rows : chunks) {
        ok(hipMemcpy(x, input_all.data() + std::size_t{offset} * kHiddenWidth, std::size_t{rows} * kHiddenWidth * 4U,
                     hipMemcpyHostToDevice), "x");
        ok(hipMemcpy(attended, attended_all.data() + std::size_t{offset} * kQueryWidth,
                     std::size_t{rows} * kQueryWidth * 4U, hipMemcpyHostToDevice), "attended");
        const MiniProjectionInputs inputs{x, x, x};
        const auto prepare_stage = [&] {
            ok(gqa::enqueue_prepared_gqa<MiniPreparedGqaShape>(backend, full, position, rows, inputs, weights, work,
                                                               prepared, failure), "prepare");
        };
        const auto output_stage = [&] {
            ok(gqa::enqueue_prepared_gqa_output<MiniPreparedGqaShape>(backend, rows, o.matrix[static_cast<unsigned>(Role::output)],
                                                                      attended, gate_logits, gated, output), "output");
        };
        // A capture does not run: the launches of each stage are counted apart, then the stages run.
        arm.prepare_nodes += graph_nodes(stream, prepare_stage);
        counts = {};
        counts.on = true;
        arm.output_nodes += graph_nodes(stream, output_stage);
        counts.on = false;
        arm.output_rounds += counts.round;
        const bool image_route = counts.admitted != 0U;
        // The route the stage took must be the one its launches show: gate_image alone when admitted, gate alone otherwise.
        arm.route_faults += (image_route ? counts.gate_image != 1U || counts.gate != 0U
                                         : counts.gate_image != 0U || counts.gate != 1U) ? 1U : 0U;
        arm.image_chunks += image_route ? 1U : 0U;
        arm.big_image_chunks += image_route && rows >= 4096U ? 1U : 0U;
        prepare_stage();
        output_stage();
        ok(gqa::enqueue_prepared_gqa_post_convolution<MiniPreparedGqaShape>(backend, rows, o.post_lags, output, post_history,
                                                                            post), "post convolution");
        ok(hipStreamSynchronize(stream), "sync");
        append(arm, query, std::size_t{rows} * kQueryWidth);
        append(arm, key, std::size_t{rows} * kKvWidth);
        append(arm, value, std::size_t{rows} * kKvWidth);
        if (!projected) append(arm, logits, std::size_t{rows} * kQueryHeads * kRelativeExtent);
        append(arm, sinks, std::size_t{rows} * kQueryHeads);
        append(arm, gate_logits, std::size_t{rows} * kQueryWidth);
        if (image_route) {
            // The gate stored the BF16 operand image instead of the FP32 gated rows (the scratch stays as it was): the
            // image bytes are the BF16 encoding of the FP32 gate, which the separate gate pass computes for the reference.
            const std::size_t words = std::size_t{rows} * kQueryWidth;
            ok(gqa::PreparedGqaRowsDevice<MiniPreparedGqaShape, hipStream_t>{stream}.gate(attended, gate_logits, gate_reference,
                                                                                           words), "gate reference");
            ok(hipStreamSynchronize(stream), "sync reference");
            std::vector<float> reference(words);
            std::vector<std::uint16_t> stored(words);
            ok(hipMemcpy(reference.data(), gate_reference, words * 4U, hipMemcpyDeviceToHost), "download reference");
            ok(hipMemcpy(stored.data(), image, words * 2U, hipMemcpyDeviceToHost), "download image");
            for (std::size_t i = 0; i < words; ++i) {
                arm.image_mismatch += stored[i] != sf::bf16_encode(reference[i]) ? 1U : 0U;
                arm.bits.push_back(stored[i]);
            }
            arm.image_words += words;
        } else {
            append(arm, gated, std::size_t{rows} * kQueryWidth);
        }
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
                    static_cast<void*>(failure), static_cast<void*>(image),
                    static_cast<void*>(gate_reference)})
        (void)hipFree(p);
    return arm;
}

template <class MakeBackend>
bool compare(const char* variant, const Operands& o, bool full, std::span<const unsigned> chunks, bool nan, Rng& rng,
             MakeBackend make_backend, bool projected, bool image_variant, hipStream_t stream) {
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
    // Prepare stage: the K/V convolution folds remove launches. Output stage: exactly the separate round pass of every
    // chunk the output projection stores rounded (rows above the tight-row bound) is gone, nothing else changes.
    std::size_t folded_chunks = 0, big_chunks = 0;
    for (const auto c : chunks) {
        folded_chunks += folds_output_store(c) ? 1U : 0U;
        big_chunks += c >= 4096U ? 1U : 0U;
    }
    const bool fewer = on.prepare_nodes < off.prepare_nodes;
    const bool store_folded = off.output_nodes == on.output_nodes + folded_chunks &&
                              off.output_rounds == chunks.size() && on.output_rounds == chunks.size() - folded_chunks;
    // The gate-image route is taken by the image backends at 4096 rows and by no shorter chunk of this list (measured on
    // gfx1151; the same chunks in both arms), never by the others, and every image word is the BF16 encoding of the FP32 gate.
    const std::size_t want_big = image_variant ? big_chunks : 0U;
    const bool route = off.route_faults == 0U && on.route_faults == 0U && off.big_image_chunks == want_big &&
                       on.big_image_chunks == want_big && off.image_chunks == want_big && on.image_chunks == want_big &&
                       off.image_mismatch == 0U && on.image_mismatch == 0U &&
                       off.image_words == on.image_words && (want_big == 0U || on.image_words != 0U);
    std::size_t differing = 0, nans = 0, payload_nans = 0;
    for (std::size_t i = 0; i < std::min(off.bits.size(), on.bits.size()); ++i) differing += off.bits[i] != on.bits[i];
    for (const auto b : on.bits) {
        const bool is_nan = (b & 0x7fffffffU) > 0x7f800000U;
        nans += is_nan;
        payload_nans += is_nan && (b & 0xffffU) != 0U;
    }
    // A NaN case that stored no NaN tested nothing about NaNs: it does not pass.
    const bool reached = !nan || nans != 0U;
    const bool pass = same && fewer && store_folded && route && reached;
    std::printf("%s round-fold-prepared %s %s chunks=%zu first=%u nan=%d outputs=%zu differing=%zu nan_outputs=%zu "
                "low_payload_nan_outputs=%zu prepare_launches off=%zu on=%zu output_launches off=%zu on=%zu "
                "output_rounds off=%zu on=%zu store_folded=%d image_chunks=%zu image_words=%zu image_mismatch=%zu "
                "route_faults=%zu\n",
                pass ? "PASS" : "FAIL", variant, full ? "full" : "local", chunks.size(), chunks[0], nan ? 1 : 0,
                off.bits.size(), differing, nans, payload_nans, off.prepare_nodes, on.prepare_nodes, off.output_nodes,
                on.output_nodes, off.output_rounds, on.output_rounds, store_folded ? 1 : 0, on.image_chunks, on.image_words,
                off.image_mismatch + on.image_mismatch, off.route_faults + on.route_faults);
    return pass;
}

}  // namespace

int main() {
    hipStream_t stream{};
    ok(hipStreamCreate(&stream), "stream");
    Rng rng{0x9E3779B97F4A7C15ULL};
    bool all = true;
    namespace detail_ = strix::inkling2::detail;
    const auto base = [&](bool fold, std::span<std::uint16_t>, StageCounts* counts) {
        return Probe<detail_::MiniPreparedHipBackend>{{{stream}, fold}, counts};
    };
    const auto projected_backend = [&](bool fold, std::span<std::uint16_t>, StageCounts* counts) {
        return Probe<detail_::MiniPreparedHipProjectedBackend>{{{{stream}, fold}}, counts};
    };
    const auto image_backend = [&](bool fold, std::span<std::uint16_t> image, StageCounts* counts) {
        return Probe<detail_::MiniPreparedHipImageBackend>{{{{stream}, fold}, image}, counts};
    };
    const auto image_projected = [&](bool fold, std::span<std::uint16_t> image, StageCounts* counts) {
        return Probe<detail_::MiniPreparedHipImageProjectedBackend>{{{{{stream}, fold}, image}}, counts};
    };
    // Heights: decode (1, 2), a verify window (16), the tight-row bound + 1 (17), short prefill (64, 300), prefill
    // (4096, 4111), and chunk boundaries (rows continue across calls).
    const std::vector<std::vector<unsigned>> shapes = {{1U}, {1U, 1U, 2U}, {16U, 17U}, {64U, 1U, 300U}, {4096U, 15U}};
    for (const bool nan : {false, true}) {
        auto operands = make_operands(rng, nan);
        for (const bool full : {false, true}) {
            for (const auto& chunks : shapes) {
                if (nan && chunks.size() == 1U && chunks[0] == 1U) continue;
                all = compare("base", operands, full, chunks, nan, rng, base, false, false, stream) && all;
                all = compare("image", operands, full, chunks, nan, rng, image_backend, false, true, stream) && all;
                all = compare("projected-bias", operands, full, chunks, nan, rng, projected_backend, true, false, stream) && all;
                all = compare("image+projected-bias", operands, full, chunks, nan, rng, image_projected, true, true, stream) && all;
            }
        }
        release(operands);
    }
    std::puts(all ? "mini_prepared_round_fold_hip: PASS" : "mini_prepared_round_fold_hip: FAIL");
    ok(hipStreamDestroy(stream), "destroy");
    return all ? 0 : 1;
}
