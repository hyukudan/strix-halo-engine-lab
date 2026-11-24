// Projected relative bias of the prepared prompt flash (PreparedBf16Projection) against the image route: the image is
// produced on the device by the production producer (core::row_transforms::ShortBf16Projection<16>, four columns per
// element), then append_attention_prepared_bf16 runs once on the image and once with the projection. Output rows,
// K/V cache bytes and the failure flag must be equal byte for byte; with a non-finite logit the flag must be raised by
// both routes. Covers full and windowed masks, a continuation chunk on a filled cache, a wrapped cache, partial query
// tiles, both pipes, a short extent, features whose bound fails while every logit is finite, and non-finite logits.
#include "core/row_transforms.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

namespace strix::cuda_core {
void set_projected_optin_limit_for_test(int bytes);  // attn_flash.cu test seam
}

namespace {
namespace cc = strix::cuda_core;
namespace prep = strix::core::row_transforms;

int failures = 0;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "flash-projected %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) { check(cudaMalloc(&ptr_, (count ? count : 1) * sizeof(T)), "cudaMalloc"); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) {
        check(cudaMemcpy(ptr_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    [[nodiscard]] T* data() const { return ptr_; }
    void fill(const std::vector<T>& data) const {
        check(cudaMemcpy(ptr_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "fill");
    }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> data(count_);
        check(cudaMemcpy(data.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return data;
    }

private:
    std::size_t count_;
    T* ptr_ = nullptr;
};

class Values {
public:
    explicit Values(unsigned seed) : state_(seed) {}
    float next(float scale) {
        state_ = state_ * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>(state_ >> 16) - 32768) * (scale / 32768.0F);
    }
    std::uint16_t bf16(float scale) {
        const float value = next(scale);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        return static_cast<std::uint16_t>(bits >> 16U);
    }

private:
    unsigned state_;
};

enum class Poison { none, nan_feature, inf_weight, large_finite, overflow };

struct Case {
    const char* name;
    unsigned rows, first, window, extent, slots;
    bool wrapped;
    cc::FlashPipe pipe;
    Poison poison;
};

template <typename T>
bool same_bytes(const char* name, const char* what, const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return true;
    std::size_t at = 0;
    while (at < a.size() && std::memcmp(&a[at], &b[at], sizeof(T)) == 0) ++at;
    std::fprintf(stderr, "FAIL %s: %s differs (first index %zu of %zu)\n", name, what, at, a.size());
    ++failures;
    return false;
}

void run(const Case& c) {
    constexpr unsigned heads = 24, kv_heads = 8, dim = 128, width = cc::kPreparedProjectedBiasWidth;
    const cc::AttentionDims dims{heads, kv_heads, dim, 0.08838834764831845F, c.slots};
    Values values(0x9E3779B9U ^ (c.rows * 131U + c.first * 7U + c.window + c.extent));
    std::vector<float> q(std::size_t{c.rows} * heads * dim), k(std::size_t{c.rows} * kv_heads * dim),
        v(std::size_t{c.rows} * kv_heads * dim), sinks(std::size_t{c.rows} * heads),
        features(std::size_t{c.rows} * heads * width);
    for (auto& x : q) x = values.next(2.0F);
    for (auto& x : k) x = values.next(2.0F);
    for (auto& x : v) x = values.next(1.0F);
    for (auto& x : sinks) x = values.next(3.0F);
    for (auto& x : features) x = values.next(1.5F);
    std::vector<std::uint16_t> weights(std::size_t{width} * c.extent);
    for (auto& w : weights) w = values.bf16(2.0F);
    // A cache already holding earlier positions (continuation chunks read them).
    std::vector<std::uint16_t> cache(std::size_t{c.slots} * kv_heads * dim);
    for (auto& x : cache) x = values.bf16(1.0F);
    const unsigned hit = (c.rows / 2U) * heads + 5U;  // the poisoned (row, head)
    switch (c.poison) {
    case Poison::none: break;
    case Poison::nan_feature: features[std::size_t{hit} * width + 3U] = std::nanf(""); break;
    case Poison::inf_weight: weights[std::size_t{7} * c.extent + c.extent - 1U] = 0x7F80U; break;
    case Poison::large_finite:  // sum |x| w_max >= 2^126 but every chain stays finite
        for (unsigned i = 0; i < width; ++i) features[std::size_t{hit} * width + i] = i == 0 ? 6.0e37F : 0.0F;
        break;
    case Poison::overflow:
        for (unsigned i = 0; i < width; ++i) features[std::size_t{hit} * width + i] = 3.0e38F;
        for (unsigned d = 0; d < c.extent; ++d)
            for (unsigned i = 0; i < width; ++i) weights[std::size_t{i} * c.extent + d] = 0x4000U;  // 2.0
        break;
    }
    if (c.poison == Poison::large_finite) weights[0] = 0x4000U;  // w_max >= 2: the bound is >= 1.2e38 > 2^126

    Device<float> dq(q), dk(k), dv(v), ds(sinks), df(features);
    Device<std::uint16_t> dw(weights);
    Device<float> image(std::size_t{c.rows} * heads * c.extent);
    using Wide = prep::ShortBf16Projection<width, false, 4U>;
    if (!Wide::fits(df.data(), dw.data(), image.data(), c.extent)) { std::fprintf(stderr, "FAIL %s: producer layout\n", c.name); ++failures; return; }
    check(prep::elements(Wide::launch_size(std::size_t{c.rows} * heads, c.extent),
                         Wide{df.data(), dw.data(), image.data(), c.extent}, cudaStream_t{}), "image producer");

    Device<__nv_bfloat16> ka(cache.size()), va(cache.size()), kb(cache.size()), vb(cache.size());
    for (const auto* dst : {&ka, &va, &kb, &vb})
        check(cudaMemcpy(dst->data(), cache.data(), cache.size() * 2U, cudaMemcpyHostToDevice), "cache");
    Device<float> oa(q.size()), ob(q.size());
    Device<unsigned> fa(1), fb(1);
    check(cudaMemset(fa.data(), 0, sizeof(unsigned)), "flag");
    check(cudaMemset(fb.data(), 0, sizeof(unsigned)), "flag");

    const cc::PreparedBf16Flash on_image{image.data(), ds.data(), c.extent, c.window, c.wrapped};
    const cc::PreparedBf16Flash on_projection{nullptr, ds.data(), c.extent, c.window, c.wrapped};
    const cc::PreparedBf16Projection projection{df.data(), dw.data()};
    if (!cc::can_launch_prepared_bf16_projected(dims, on_projection, projection, dq.data(), kb.data(), vb.data(),
                                                c.first, c.rows, ob.data(), c.pipe)) {
        std::fprintf(stderr, "FAIL %s: projected route refused\n", c.name);
        ++failures;
        return;
    }
    check(cc::append_attention_prepared_bf16(dims, on_image, dq.data(), dk.data(), dv.data(), ka.data(), va.data(),
                                             c.first, c.rows, oa.data(), nullptr, c.pipe, fa.data()), "image route");
    check(cc::append_attention_prepared_bf16(dims, on_projection, projection, dq.data(), dk.data(), dv.data(),
                                             kb.data(), vb.data(), c.first, c.rows, ob.data(), nullptr, c.pipe, fb.data()),
          "projected route");
    check(cudaDeviceSynchronize(), "sync");

    const auto flag_a = fa.download(), flag_b = fb.download();
    const bool flagged = c.poison == Poison::nan_feature || c.poison == Poison::inf_weight || c.poison == Poison::overflow;
    bool ok = same_bytes(c.name, "failure flag", flag_a, flag_b);
    if (flag_a[0] != (flagged ? 1U : 0U)) {
        std::fprintf(stderr, "FAIL %s: image route flag %u, expected %u\n", c.name, flag_a[0], flagged ? 1U : 0U);
        ++failures;
        ok = false;
    }
    ok = same_bytes(c.name, "K cache", ka.download(), kb.download()) && ok;
    ok = same_bytes(c.name, "V cache", va.download(), vb.download()) && ok;
    if (!flagged) ok = same_bytes(c.name, "output", oa.download(), ob.download()) && ok;
    std::printf("%s %s rows %u first %u window %u extent %u flag %u\n", ok ? "PASS" : "FAIL", c.name, c.rows, c.first,
                c.window, c.extent, flag_b[0]);
}

void refusals() {
    constexpr unsigned heads = 24, kv_heads = 8, dim = 128, width = cc::kPreparedProjectedBiasWidth, rows = 64, slots = 64;
    const cc::AttentionDims dims{heads, kv_heads, dim, 0.0884F, slots};
    Device<float> q(1024), f(std::size_t{rows} * heads * width), o(std::size_t{rows} * heads * dim);
    Device<std::uint16_t> w(width * 1024);
    Device<__nv_bfloat16> kc(std::size_t{slots} * kv_heads * dim), vc(std::size_t{slots} * kv_heads * dim);
    const cc::PreparedBf16Projection projection{f.data(), w.data()};
    const auto admitted = [&](cc::PreparedBf16Flash prepared, cc::PreparedBf16Projection p) {
        return cc::can_launch_prepared_bf16_projected(dims, prepared, p, q.data(), kc.data(), vc.data(), 0, rows, o.data(),
                                                      cc::FlashPipe::pad2);
    };
    const cc::PreparedBf16Flash base{nullptr, nullptr, 1024, 0, false};
    bool ok = admitted(base, projection);
    auto image = base; image.relative_bias = q.data();
    auto none = base; none.relative_extent = 0;
    auto wide = base; wide.relative_extent = 1032;
    auto odd = base; odd.relative_extent = 1020;
    auto both = base; both.bidirectional = true;
    ok = ok && !admitted(image, projection) && !admitted(none, projection) && !admitted(wide, projection) &&
         !admitted(odd, projection) && !admitted(both, projection) && !admitted(base, {nullptr, w.data()}) &&
         !admitted(base, {f.data(), nullptr}) && !admitted(base, {f.data() + 1, w.data()});
    if (!ok) ++failures;
    std::printf("%s admission (image pointer, extent 0 / > 1024 / not a multiple of 8, bidirectional, missing or "
                "unaligned operands refused)\n", ok ? "PASS" : "FAIL");
}

// Features / weights inside the output image or inside a whole K / V cache are refused before any enqueue; the
// append leaves the cache untouched.
void overlaps() {
    constexpr unsigned heads = 1, kv_heads = 1, dim = 128, width = cc::kPreparedProjectedBiasWidth, rows = 256, slots = 512,
                       extent = 64;
    const cc::AttentionDims dims{heads, kv_heads, dim, 0.0884F, slots};
    const std::size_t out_floats = std::size_t{rows} * heads * dim, cache_elems = std::size_t{slots} * kv_heads * dim;
    Device<float> q(out_floats), k(std::size_t{rows} * kv_heads * dim), v(std::size_t{rows} * kv_heads * dim);
    Device<float> o(out_floats + std::size_t{rows} * heads * width), f(std::size_t{rows} * heads * width);
    Device<std::uint16_t> w(width * extent);
    Device<__nv_bfloat16> kc(cache_elems), vc(cache_elems);
    const std::vector<std::uint16_t> sentinel(cache_elems, 0x5AA5U);
    check(cudaMemcpy(kc.data(), sentinel.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "sentinel");
    check(cudaMemcpy(vc.data(), sentinel.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "sentinel");
    const cc::PreparedBf16Flash prepared{nullptr, nullptr, extent, 0, false};
    // {admitted by can_launch, append status}; the append runs on the same operands.
    const auto status = [&](const float* features, const std::uint16_t* weights) {
        const cc::PreparedBf16Projection p{features, weights};
        return std::pair{cc::can_launch_prepared_bf16_projected(dims, prepared, p, q.data(), kc.data(), vc.data(), 0, rows,
                                                                 o.data(), cc::FlashPipe::pad2),
                         cc::append_attention_prepared_bf16(dims, prepared, p, q.data(), k.data(), v.data(), kc.data(),
                                                            vc.data(), 0, rows, o.data(), nullptr)};
    };
    const auto refused = [](const std::pair<bool, cudaError_t>& r) { return !r.first && r.second == cudaErrorInvalidValue; };
    // The review's case: feature row 128 starts at float 2048 of the image, where output row 16 writes.
    const bool in_out = refused(status(o.data() + 2048, w.data()));
    const bool at_end = refused(status(o.data() + out_floats - 4, w.data()));
    const bool weights_in_out = refused(status(f.data(), reinterpret_cast<const std::uint16_t*>(o.data() + 16)));
    const bool features_in_k = refused(status(reinterpret_cast<const float*>(kc.data() + 256), w.data()));
    const bool weights_in_v = refused(status(f.data(), reinterpret_cast<const std::uint16_t*>(vc.data() + 512)));
    check(cudaDeviceSynchronize(), "sync");
    const auto cache_k = kc.download(), cache_v = vc.download();
    const bool untouched = std::memcmp(cache_k.data(), sentinel.data(), cache_elems * 2U) == 0 &&
                           std::memcmp(cache_v.data(), sentinel.data(), cache_elems * 2U) == 0;
    // Directly after the image (the output buffer carries room for it) the operands are disjoint: admitted and run.
    const auto touching = status(o.data() + out_floats, w.data());
    check(cudaDeviceSynchronize(), "sync");
    const bool touching_ok = touching.first && touching.second == cudaSuccess;
    const bool ok = in_out && at_end && weights_in_out && features_in_k && weights_in_v && untouched && touching_ok;
    if (!ok) ++failures;
    std::printf("%s overlap (features in output image / at its end, weights in output image, features in K cache, weights "
                "in V cache refused with the cache untouched; features directly after the image admitted)\n",
                ok ? "PASS" : "FAIL");
}

// A refused device setup (opt-in limit below the extent's shared memory) returns an error with the K/V cache and the
// output untouched; the same call then runs once the limit is real again.
void setup_refusal() {
    constexpr unsigned heads = 24, kv_heads = 8, dim = 128, width = cc::kPreparedProjectedBiasWidth, rows = 130, slots = 256,
                       extent = 1024;
    const cc::AttentionDims dims{heads, kv_heads, dim, 0.0884F, slots};
    Values values(0x1234567U);
    std::vector<float> q(std::size_t{rows} * heads * dim), k(std::size_t{rows} * kv_heads * dim), v(k.size()),
        features(std::size_t{rows} * heads * width);
    for (auto& x : q) x = values.next(1.0F);
    for (auto& x : k) x = values.next(1.0F);
    for (auto& x : v) x = values.next(1.0F);
    for (auto& x : features) x = values.next(1.0F);
    std::vector<std::uint16_t> weights(std::size_t{width} * extent);
    for (auto& x : weights) x = values.bf16(1.0F);
    Device<float> dq(q), dk(k), dv(v), df(features), out(q.size());
    Device<std::uint16_t> dw(weights);
    const std::size_t cache_elems = std::size_t{slots} * kv_heads * dim;
    const std::vector<std::uint16_t> sentinel(cache_elems, 0xC3C3U);
    Device<__nv_bfloat16> kc(cache_elems), vc(cache_elems);
    check(cudaMemcpy(kc.data(), sentinel.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "sentinel");
    check(cudaMemcpy(vc.data(), sentinel.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "sentinel");
    check(cudaMemset(out.data(), 0xAB, q.size() * sizeof(float)), "out sentinel");
    const std::vector<float> out_sentinel = out.download();
    const cc::PreparedBf16Flash prepared{nullptr, nullptr, extent, 0, false};
    const cc::PreparedBf16Projection projection{df.data(), dw.data()};
    bool ok = true;
    for (const auto pipe : {cc::FlashPipe::pad2, cc::FlashPipe::swz3}) {
        cc::set_projected_optin_limit_for_test(48 * 1024);  // below the pipe's K/V stages plus the 32 KB weight table
        const auto refused = cc::append_attention_prepared_bf16(dims, prepared, projection, dq.data(), dk.data(), dv.data(),
                                                                kc.data(), vc.data(), 0, rows, out.data(), nullptr, pipe);
        cc::set_projected_optin_limit_for_test(0);
        check(cudaDeviceSynchronize(), "sync");
        const auto ck = kc.download(), cv = vc.download();
        const auto co = out.download();
        const bool untouched = std::memcmp(ck.data(), sentinel.data(), cache_elems * 2U) == 0 &&
                               std::memcmp(cv.data(), sentinel.data(), cache_elems * 2U) == 0 &&
                               std::memcmp(co.data(), out_sentinel.data(), co.size() * sizeof(float)) == 0;
        ok = ok && refused == cudaErrorInvalidValue && untouched;
    }
    const auto accepted = cc::append_attention_prepared_bf16(dims, prepared, projection, dq.data(), dk.data(), dv.data(),
                                                             kc.data(), vc.data(), 0, rows, out.data(), nullptr);
    check(cudaDeviceSynchronize(), "sync");
    const auto after = kc.download();
    ok = ok && accepted == cudaSuccess && std::memcmp(after.data(), sentinel.data(), cache_elems * 2U) != 0;
    if (!ok) ++failures;
    std::printf("%s setup refusal (both pipes: error returned, K/V sentinel bytes and output unchanged; accepted once the "
                "limit is real)\n", ok ? "PASS" : "FAIL");
}

// Two host threads, two streams, alternating extents 64 / 1024 on the same specialization: every launch must run and
// match the single-threaded reference of its extent byte for byte.
void concurrent_extents() {
    constexpr unsigned heads = 24, kv_heads = 8, dim = 128, width = cc::kPreparedProjectedBiasWidth, rows = 256, slots = 256,
                       iterations = 200;
    constexpr unsigned extents[2] = {64, 1024};
    const cc::AttentionDims dims{heads, kv_heads, dim, 0.0884F, slots};
    Values values(0xBEEF123U);
    std::vector<float> q(std::size_t{rows} * heads * dim), features(std::size_t{rows} * heads * width);
    for (auto& x : q) x = values.next(2.0F);
    for (auto& x : features) x = values.next(1.5F);
    std::vector<std::uint16_t> weights(std::size_t{width} * 1024);
    for (auto& x : weights) x = values.bf16(2.0F);
    const std::size_t cache_elems = std::size_t{slots} * kv_heads * dim;
    std::vector<std::uint16_t> cache(cache_elems);
    for (auto& x : cache) x = values.bf16(1.0F);
    Device<float> dq(q), df(features);
    Device<std::uint16_t> dw(weights);
    Device<__nv_bfloat16> kc(cache_elems), vc(cache_elems);
    check(cudaMemcpy(kc.data(), cache.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "cache");
    check(cudaMemcpy(vc.data(), cache.data(), cache_elems * 2U, cudaMemcpyHostToDevice), "cache");
    const cc::PreparedBf16Projection projection{df.data(), dw.data()};
    const auto launch = [&](unsigned extent, float* out, cudaStream_t stream) {
        const cc::PreparedBf16Flash prepared{nullptr, nullptr, extent, 0, false};
        return cc::attention_prefill_prepared_bf16(dims, prepared, projection, dq.data(), kc.data(), vc.data(), 0, rows, out,
                                                   stream, cc::FlashPipe::pad2);
    };
    std::vector<float> reference[2];
    for (unsigned e = 0; e < 2; ++e) {
        Device<float> out(q.size());
        check(launch(extents[e], out.data(), nullptr), "reference launch");
        check(cudaDeviceSynchronize(), "sync");
        reference[e] = out.download();
    }
    unsigned bad[2] = {0, 0}, errors[2] = {0, 0};
    const auto worker = [&](unsigned id) {
        check(cudaSetDevice(0), "set device");
        cudaStream_t stream{};
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        Device<float> out(q.size());
        for (unsigned i = 0; i < iterations; ++i) {
            const unsigned e = (i + id) % 2U;  // the two threads always hold different extents
            if (launch(extents[e], out.data(), stream) != cudaSuccess) { ++errors[id]; continue; }
            check(cudaStreamSynchronize(stream), "stream sync");
            const auto got = out.download();
            if (std::memcmp(got.data(), reference[e].data(), got.size() * sizeof(float)) != 0) ++bad[id];
        }
        check(cudaStreamDestroy(stream), "stream destroy");
    };
    std::thread a(worker, 0U), b(worker, 1U);
    a.join();
    b.join();
    const bool ok = bad[0] + bad[1] + errors[0] + errors[1] == 0;
    if (!ok) ++failures;
    std::printf("%s concurrent extents (2 threads x 2 streams x %u launches alternating 64 / 1024: %u launch errors, %u "
                "output mismatches)\n", ok ? "PASS" : "FAIL", iterations, errors[0] + errors[1], bad[0] + bad[1]);
}

}  // namespace

int main() {
    using cc::FlashPipe;
    const Case cases[] = {
        {"window-1024", 1000, 0, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::none},
        {"full", 777, 0, 0, 1024, 4096, false, FlashPipe::pad2, Poison::none},
        {"full-continuation", 300, 1500, 0, 1024, 4096, false, FlashPipe::pad2, Poison::none},
        {"window-continuation", 257, 2100, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::none},
        {"window-wrapped", 200, 3000, 1024, 1024, 1536, true, FlashPipe::pad2, Poison::none},
        {"window-swz3", 513, 64, 1024, 1024, 4096, false, FlashPipe::swz3, Poison::none},
        {"short-extent", 384, 40, 0, 64, 4096, false, FlashPipe::pad2, Poison::none},
        {"one-row", 1, 900, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::none},
        {"bound-fails-finite", 260, 0, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::large_finite},
        {"nan-feature", 260, 0, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::nan_feature},
        {"inf-weight", 130, 10, 0, 1024, 4096, false, FlashPipe::pad2, Poison::inf_weight},
        {"overflow", 140, 0, 1024, 1024, 4096, false, FlashPipe::pad2, Poison::overflow},
    };
    for (const auto& c : cases) run(c);
    refusals();
    overlaps();
    setup_refusal();
    concurrent_extents();
    std::printf("%s flash projected bias: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
