// The Inkling 2 Mini projected-bias attention route through its binding (enqueue_mini_cuda_attention_projected and its
// preflight) against the incumbent image route (enqueue_mini_cuda_attention on the image the production producer makes
// from the same features and weights). K/V cache bytes, output bytes and the failure flag must be equal. Cases: two rows,
// a partial 128-row tile, continuation positions on a filled cache, a wrapped sliding window, and non-finite bias values
// (a NaN feature and an infinite weight at a masked distance: the image's finite check flags them, so must the flash).
// Then the transition: with the device's dynamic shared-memory opt-in below the 1024-column projection's need (pad2:
// 68,624 bytes) the preflight refuses with cudaErrorInvalidValue, the route decision selects the image route (no error)
// and that route runs; any other preflight status stays an error.
#include "core/row_transforms.hpp"
#include "inkling2/inkling2_mini_attention_cuda.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strix::cuda_core {
void set_projected_optin_limit_for_test(int bytes);  // attn_flash.cu test seam
}

namespace {
namespace cc = strix::cuda_core;
namespace mini = strix::inkling2;
namespace prep = strix::core::row_transforms;

constexpr unsigned kHeads = mini::kQueryHeads, kKv = mini::kKvHeads, kDim = mini::kHeadWidth,
                   kExtent = mini::kRelativeExtent, kWidth = cc::kPreparedProjectedBiasWidth;
int failures = 0;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "mini-projected-bias %s: %s\n", operation, cudaGetErrorString(status));
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

enum class Poison { none, nan_feature, inf_weight_masked };

struct Case {
    const char* name;
    bool full_attention;
    unsigned rows, first, slots;
    bool wrapped;
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

mini::LayerVisit visit_of(bool full) {
    for (const auto visit : mini::mini_schedule())
        if (mini::is_full_attention(visit.physical_layer) == full) return visit;
    std::fprintf(stderr, "mini-projected-bias: no %s attention visit\n", full ? "full" : "sliding");
    std::exit(EXIT_FAILURE);
}

// One case's operands, both routes' caches and the image the producer made; `route` runs one of the two enqueues.
struct Fixture {
    explicit Fixture(const Case& c)
        : c(c), visit(visit_of(c.full_attention)), q(std::size_t{c.rows} * kHeads * kDim),
          k(std::size_t{c.rows} * kKv * kDim), v(k.size()), sinks(std::size_t{c.rows} * kHeads),
          features(std::size_t{c.rows} * kHeads * kWidth), weights(std::size_t{kWidth} * kExtent),
          cache(std::size_t{c.slots} * kKv * kDim) {
        Values values(0xA511E9B3U ^ (c.rows * 131U + c.first * 7U + c.slots));
        for (auto& x : q) x = values.next(2.0F);
        for (auto& x : k) x = values.next(2.0F);
        for (auto& x : v) x = values.next(1.0F);
        for (auto& x : sinks) x = values.next(3.0F);
        for (auto& x : features) x = values.next(1.5F);
        for (auto& w : weights) w = values.bf16(2.0F);
        for (auto& x : cache) x = values.bf16(1.0F);
        const unsigned hit = (c.rows / 2U) * kHeads + 5U;
        if (c.poison == Poison::nan_feature) features[std::size_t{hit} * kWidth + 3U] = std::nanf("");
        // The last distance (kExtent - 1) is masked for every row below it: the infinity sits where no key is live.
        if (c.poison == Poison::inf_weight_masked) weights[std::size_t{7} * kExtent + kExtent - 1U] = 0x7F80U;
    }

    const Case& c;
    mini::LayerVisit visit;
    std::vector<float> q, k, v, sinks, features;
    std::vector<std::uint16_t> weights, cache;
};

struct Result {
    std::vector<std::uint16_t> k_cache, v_cache;
    std::vector<float> output;
    unsigned flag = 0;
};

enum class Route { image, projected };

Result run_route(const Fixture& f, Route route) {
    const auto& c = f.c;
    Device<float> dq(f.q), dk(f.k), dv(f.v), ds(f.sinks), df(f.features);
    Device<std::uint16_t> dw(f.weights), kc(f.cache), vc(f.cache);
    Device<float> image(std::size_t{c.rows} * kHeads * kExtent), out(f.q.size());
    Device<unsigned> flag(1);
    check(cudaMemset(flag.data(), 0, sizeof(unsigned)), "flag");
    mini::MiniCudaAttentionRows rows{dq.data(), dk.data(), dv.data(), nullptr, ds.data(), out.data(), c.first, c.rows};
    const mini::MiniVisitKvCache cache{kc.data(), vc.data(), c.slots, c.wrapped};
    if (route == Route::image) {
        using Wide = prep::ShortBf16Projection<kWidth, false, 4U>;
        if (!Wide::fits(df.data(), dw.data(), image.data(), kExtent)) {
            std::fprintf(stderr, "FAIL %s: producer layout\n", c.name);
            std::exit(EXIT_FAILURE);
        }
        check(prep::elements(Wide::launch_size(std::size_t{c.rows} * kHeads, kExtent),
                             Wide{df.data(), dw.data(), image.data(), kExtent}, cudaStream_t{}), "image producer");
        rows.relative_logits = image.data();
        check(mini::enqueue_mini_cuda_attention(f.visit, rows, cache, nullptr, cc::FlashPipe::pad2, flag.data()), "image route");
    } else {
        check(mini::enqueue_mini_cuda_attention_projected(f.visit, rows, cache, df.data(), dw.data(), nullptr, flag.data()),
              "projected route");
    }
    check(cudaDeviceSynchronize(), "sync");
    return {kc.download(), vc.download(), out.download(), flag.download()[0]};
}

void run(const Case& c) {
    const Fixture f(c);
    const auto image = run_route(f, Route::image);
    const auto projected = run_route(f, Route::projected);
    const bool flagged = c.poison != Poison::none;
    bool ok = image.flag == projected.flag;
    if (image.flag != (flagged ? 1U : 0U)) {
        std::fprintf(stderr, "FAIL %s: image route flag %u, expected %u\n", c.name, image.flag, flagged ? 1U : 0U);
        ++failures;
        ok = false;
    }
    ok = same_bytes(c.name, "K cache", image.k_cache, projected.k_cache) && ok;
    ok = same_bytes(c.name, "V cache", image.v_cache, projected.v_cache) && ok;
    if (!flagged) ok = same_bytes(c.name, "output", image.output, projected.output) && ok;
    if (!ok) ++failures;
    std::printf("%s %s rows %u first %u %s slots %u%s flag %u\n", ok ? "PASS" : "FAIL", c.name, c.rows, c.first,
                c.full_attention ? "full" : "sliding", c.slots, c.wrapped ? " wrapped" : "", projected.flag);
}

// The transition: the preflight refuses below the opt-in the projection needs, the decision keeps the image route (not an
// error) and that route's bytes equal the projected route's at the real limit; a real error is not absorbed.
void transition() {
    const Case c{"transition", true, 130, 300, 4096, false, Poison::none};
    const Fixture f(c);
    const auto projected = run_route(f, Route::projected);
    Device<float> dq(f.q), dk(f.k), dv(f.v), ds(f.sinks), df(f.features), out(f.q.size());
    Device<std::uint16_t> dw(f.weights), kc(f.cache), vc(f.cache);
    const mini::MiniCudaAttentionRows rows{dq.data(), dk.data(), dv.data(), nullptr, ds.data(), out.data(), c.first, c.rows};
    const mini::MiniVisitKvCache cache{kc.data(), vc.data(), c.slots, c.wrapped};
    bool selected = false;
    const auto admitted = mini::mini_cuda_projected_decision(
        mini::preflight_mini_cuda_attention_projected(f.visit, rows, cache, df.data(), dw.data()), selected);
    cc::set_projected_optin_limit_for_test(48 * 1024);
    bool refused_selected = true;
    const auto refused_status = mini::preflight_mini_cuda_attention_projected(f.visit, rows, cache, df.data(), dw.data());
    const auto refused = mini::mini_cuda_projected_decision(refused_status, refused_selected);
    // The append itself refuses at that limit BEFORE any write: the K/V cache keeps its initial bytes.
    const auto append_status = mini::enqueue_mini_cuda_attention_projected(f.visit, rows, cache, df.data(), dw.data(), nullptr);
    check(cudaDeviceSynchronize(), "sync");
    const bool untouched = append_status == cudaErrorInvalidValue && kc.download() == f.cache && vc.download() == f.cache;
    cc::set_projected_optin_limit_for_test(0);
    bool error_selected = true;
    const auto error = mini::mini_cuda_projected_decision(cudaErrorMemoryAllocation, error_selected);
    const auto image = run_route(f, Route::image);
    bool ok = admitted == cudaSuccess && selected && refused_status == cudaErrorInvalidValue && refused == cudaSuccess &&
              !refused_selected && error == cudaErrorMemoryAllocation && !error_selected && untouched;
    ok = same_bytes(c.name, "K cache", image.k_cache, projected.k_cache) && ok;
    ok = same_bytes(c.name, "V cache", image.v_cache, projected.v_cache) && ok;
    ok = same_bytes(c.name, "output", image.output, projected.output) && ok;
    if (!ok) ++failures;
    std::printf("%s transition (real limit: projected route; 48 KiB opt-in: preflight refuses, decision keeps the image route "
                "without an error, its bytes equal the projected route's; the refused append leaves the K/V cache untouched; another "
                "status stays an error)\n", ok ? "PASS" : "FAIL");
}
}  // namespace

int main() {
    const Case cases[] = {
        {"two-rows-full", true, 2, 0, 4096, false, Poison::none},
        {"two-rows-sliding", false, 2, 0, 4096, false, Poison::none},
        {"two-rows-continuation", true, 2, 1500, 4096, false, Poison::none},
        {"partial-tile-130", true, 130, 0, 4096, false, Poison::none},
        {"partial-tile-sliding-continuation", false, 257, 2100, 4096, false, Poison::none},
        {"sliding-wrapped", false, 200, 3000, 1536, true, Poison::none},
        {"nan-feature", true, 130, 10, 4096, false, Poison::nan_feature},
        {"inf-weight-masked-distance", false, 2, 0, 4096, false, Poison::inf_weight_masked},
    };
    for (const auto& c : cases) run(c);
    transition();
    std::printf("%s mini projected bias (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
