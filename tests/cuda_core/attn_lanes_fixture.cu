// Lane-batched prepared attention identity (EXACT): append_attention_prepared_bf16_lanes must write the
// same cache bytes and the same output bits as append_attention_prepared_bf16 called once per lane, for
// full and sliding (wrapped) caches, both pipes, lane counts across one and
// two launch blocks, one sequence's consecutive rows on one cache (a verify window: same bits as the serial
// row-by-row appends), and refusal before any store.
#include "strix/cuda_core/attention.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
namespace cc = strix::cuda_core;
namespace lanes_ns = strix::attn::lanes;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "attn-lanes %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(const std::vector<T>& data) : count_(data.size()) {
        check(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
        check(cudaMemcpy(ptr_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { if (ptr_) (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    [[nodiscard]] T* data() const { return ptr_; }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> out(count_);
        check(cudaMemcpy(out.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return out;
    }

private:
    std::size_t count_;
    T* ptr_ = nullptr;
};

class Values {
public:
    explicit Values(std::uint32_t seed) : state_(seed) {}
    float next(float scale) {
        state_ = state_ * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>(state_ >> 16) - 32768) * (scale / 32768.0F);
    }
    template <typename T>
    std::vector<T> make(std::size_t count, float scale) {
        std::vector<T> out(count);
        for (auto& v : out) v = T(next(scale));
        return out;
    }

private:
    std::uint32_t state_;
};

constexpr unsigned kHeads = 24, kKvHeads = 8, kDim = 128, kExtent = 1024, kWindow = 1024;

template <typename T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// One case: lane i is packed row i at position pos[i] with its own caches.
bool run_case(const char* name, bool sliding, cc::FlashPipe pipe, const std::vector<unsigned>& pos,
              std::uint32_t seed) {
    const auto lanes = pos.size();
    const unsigned slots = sliding ? kWindow + 64U : 4096U;
    const std::size_t cache = std::size_t{slots} * kKvHeads * kDim;
    Values values(seed);
    const auto q = values.make<float>(lanes * kHeads * kDim, 2.0F);
    const auto k = values.make<float>(lanes * kKvHeads * kDim, 2.0F);
    const auto v = values.make<float>(lanes * kKvHeads * kDim, 1.0F);
    const auto bias = values.make<float>(lanes * kHeads * kExtent, 0.5F);
    const auto sinks = values.make<float>(lanes * kHeads, 1.0F);
    std::vector<std::vector<__nv_bfloat16>> prefix(lanes);
    for (auto& p : prefix) p = values.make<__nv_bfloat16>(cache, 1.0F);

    const Device<float> dq(q), dk(k), dv(v), db(bias), ds(sinks);
    const Device<float> out_ref(std::vector<float>(lanes * kHeads * kDim, NAN));
    const Device<float> out_lanes(std::vector<float>(lanes * kHeads * kDim, -NAN));
    const Device<unsigned> fail_ref(std::vector<unsigned>{0}), fail_lanes(std::vector<unsigned>{0});
    std::vector<Device<__nv_bfloat16>*> kr, vr, kl, vl;
    for (std::size_t i = 0; i < lanes; ++i) {
        kr.push_back(new Device<__nv_bfloat16>(prefix[i]));
        vr.push_back(new Device<__nv_bfloat16>(prefix[(i + 1) % lanes]));
        kl.push_back(new Device<__nv_bfloat16>(prefix[i]));
        vl.push_back(new Device<__nv_bfloat16>(prefix[(i + 1) % lanes]));
    }
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), slots};
    const cc::PreparedBf16Flash shape{nullptr, nullptr, kExtent, sliding ? kWindow : 0U, sliding};
    const cc::PreparedBf16LaneImages images{dq.data(), dk.data(), dv.data(), db.data(), ds.data(), out_lanes.data()};
    std::vector<lanes_ns::PreparedBf16Lane> desc;
    for (std::size_t i = 0; i < lanes; ++i) {
        // Per-lane reference: the incumbent one-lane append on its own caches.
        cc::PreparedBf16Flash prepared = shape;
        prepared.relative_bias = db.data() + i * kHeads * kExtent;
        prepared.sinks = ds.data() + i * kHeads;
        check(cc::append_attention_prepared_bf16(dims, prepared, dq.data() + i * kHeads * kDim,
                  dk.data() + i * kKvHeads * kDim, dv.data() + i * kKvHeads * kDim, kr[i]->data(), vr[i]->data(),
                  pos[i], 1U, out_ref.data() + i * kHeads * kDim, nullptr, pipe, fail_ref.data()), "per-lane append");
        desc.push_back({reinterpret_cast<std::uint16_t*>(kl[i]->data()), reinterpret_cast<std::uint16_t*>(vl[i]->data()),
                        pos[i], slots, sliding ? 1U : 0U});
    }
    check(cc::append_attention_prepared_bf16_lanes(dims, shape, images, desc, nullptr, pipe, fail_lanes.data()),
          "lanes append");
    check(cudaDeviceSynchronize(), "sync");

    bool ok = same_bytes(out_ref.download(), out_lanes.download());
    ok = ok && same_bytes(fail_ref.download(), fail_lanes.download()) && fail_lanes.download()[0] == 0U;
    for (std::size_t i = 0; i < lanes; ++i) {
        ok = ok && same_bytes(kr[i]->download(), kl[i]->download()) && same_bytes(vr[i]->download(), vl[i]->download());
        delete kr[i];
        delete vr[i];
        delete kl[i];
        delete vl[i];
    }
    std::printf("attn-lanes %s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// Admission refuses a bad lane (past capacity) and shared caches before any store: caches stay untouched.
bool refuses_before_store() {
    Values values(91);
    const auto prefix = values.make<__nv_bfloat16>(std::size_t{64} * kKvHeads * kDim, 1.0F);
    const Device<__nv_bfloat16> k0(prefix), v0(prefix), k1(prefix), v1(prefix);
    const Device<float> rows(values.make<float>(std::size_t{2} * kHeads * kExtent, 1.0F));
    const Device<float> out(std::vector<float>(std::size_t{2} * kHeads * kDim, 0.0F));
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 0.088F, 64};
    const cc::PreparedBf16Flash shape{nullptr, nullptr, kExtent, 0U, false};
    const cc::PreparedBf16LaneImages images{rows.data(), rows.data(), rows.data(), rows.data(), rows.data(), out.data()};
    const auto u16 = [](const Device<__nv_bfloat16>& d) { return reinterpret_cast<std::uint16_t*>(d.data()); };
    std::vector<lanes_ns::PreparedBf16Lane> past{{u16(k0), u16(v0), 3U, 64U, 0U}, {u16(k1), u16(v1), 64U, 64U, 0U}};
    std::vector<lanes_ns::PreparedBf16Lane> shared{{u16(k0), u16(v0), 3U, 64U, 0U}, {u16(k0), u16(v1), 5U, 64U, 0U}};
    const bool refused = cc::append_attention_prepared_bf16_lanes(dims, shape, images, past, nullptr) == cudaErrorInvalidValue &&
                         cc::append_attention_prepared_bf16_lanes(dims, shape, images, shared, nullptr) == cudaErrorInvalidValue;
    check(cudaDeviceSynchronize(), "sync");
    const bool untouched = same_bytes(prefix, k0.download()) && same_bytes(prefix, v1.download());
    std::printf("attn-lanes refuse-before-store %s\n", refused && untouched ? "PASS" : "FAIL");
    return refused && untouched;
}

// One sequence's verify window: `run` consecutive rows from `first` on ONE cache (lane i at first + i). Reference =
// the one-lane append row by row on a copy of that cache (the serial decode order); the lanes call must write the
// same cache bytes and output bits. A full cache and a wrapped ring of window + 64 slots (>= window - 1 + run).
bool run_sequence_case(const char* name, bool sliding, cc::FlashPipe pipe, unsigned first, unsigned run,
                       std::uint32_t seed) {
    const unsigned slots = sliding ? kWindow + 64U : 4096U;
    const std::size_t cache = std::size_t{slots} * kKvHeads * kDim;
    Values values(seed);
    const auto q = values.make<float>(std::size_t{run} * kHeads * kDim, 2.0F);
    const auto k = values.make<float>(std::size_t{run} * kKvHeads * kDim, 2.0F);
    const auto v = values.make<float>(std::size_t{run} * kKvHeads * kDim, 1.0F);
    const auto bias = values.make<float>(std::size_t{run} * kHeads * kExtent, 0.5F);
    const auto sinks = values.make<float>(std::size_t{run} * kHeads, 1.0F);
    const auto kprefix = values.make<__nv_bfloat16>(cache, 1.0F);
    const auto vprefix = values.make<__nv_bfloat16>(cache, 1.0F);
    const Device<float> dq(q), dk(k), dv(v), db(bias), ds(sinks);
    const Device<float> out_ref(std::vector<float>(std::size_t{run} * kHeads * kDim, NAN));
    const Device<float> out_lanes(std::vector<float>(std::size_t{run} * kHeads * kDim, -NAN));
    const Device<unsigned> fail_ref(std::vector<unsigned>{0}), fail_lanes(std::vector<unsigned>{0});
    const Device<__nv_bfloat16> kr(kprefix), vr(vprefix), kl(kprefix), vl(vprefix);
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), slots};
    const cc::PreparedBf16Flash shape{nullptr, nullptr, kExtent, sliding ? kWindow : 0U, sliding};
    const cc::PreparedBf16LaneImages images{dq.data(), dk.data(), dv.data(), db.data(), ds.data(), out_lanes.data()};
    std::vector<lanes_ns::PreparedBf16Lane> desc;
    for (unsigned i = 0; i < run; ++i) {
        cc::PreparedBf16Flash prepared = shape;
        prepared.relative_bias = db.data() + std::size_t{i} * kHeads * kExtent;
        prepared.sinks = ds.data() + std::size_t{i} * kHeads;
        check(cc::append_attention_prepared_bf16(dims, prepared, dq.data() + std::size_t{i} * kHeads * kDim,
                  dk.data() + std::size_t{i} * kKvHeads * kDim, dv.data() + std::size_t{i} * kKvHeads * kDim,
                  kr.data(), vr.data(), first + i, 1U, out_ref.data() + std::size_t{i} * kHeads * kDim, nullptr, pipe,
                  fail_ref.data()), "serial row append");
        desc.push_back({reinterpret_cast<std::uint16_t*>(kl.data()), reinterpret_cast<std::uint16_t*>(vl.data()),
                        first + i, slots, sliding ? 1U : 0U});
    }
    const auto status = cc::append_attention_prepared_bf16_lanes(dims, shape, images, desc, nullptr, pipe,
                                                                 fail_lanes.data());
    check(cudaDeviceSynchronize(), "sync");
    const bool ok = status == cudaSuccess && same_bytes(out_ref.download(), out_lanes.download()) &&
                    same_bytes(fail_ref.download(), fail_lanes.download()) && fail_lanes.download()[0] == 0U &&
                    same_bytes(kr.download(), kl.download()) && same_bytes(vr.download(), vl.download());
    std::printf("attn-lanes %s %s%s\n", name, ok ? "PASS" : "FAIL", status == cudaSuccess ? "" : " (refused)");
    return ok;
}

// One cache, refused before any store: a gap in the positions, and a ring that cannot hold window - 1 + run rows.
bool sequence_refusals() {
    Values values(93);
    constexpr unsigned slots = kWindow + 1U;  // a three-row run needs window + 2
    const auto prefix = values.make<__nv_bfloat16>(std::size_t{slots} * kKvHeads * kDim, 1.0F);
    const Device<__nv_bfloat16> k0(prefix), v0(prefix);
    const Device<float> rows(values.make<float>(std::size_t{3} * kHeads * kExtent, 1.0F));
    const Device<float> out(std::vector<float>(std::size_t{3} * kHeads * kDim, 0.0F));
    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 0.088F, slots};
    const cc::PreparedBf16LaneImages images{rows.data(), rows.data(), rows.data(), rows.data(), rows.data(), out.data()};
    auto* k = reinterpret_cast<std::uint16_t*>(k0.data());
    auto* v = reinterpret_cast<std::uint16_t*>(v0.data());
    const std::vector<lanes_ns::PreparedBf16Lane> gap{{k, v, 7U, slots, 0U}, {k, v, 9U, slots, 0U}};
    const std::vector<lanes_ns::PreparedBf16Lane> ring{{k, v, 2000U, slots, 1U}, {k, v, 2001U, slots, 1U},
                                                       {k, v, 2002U, slots, 1U}};
    const cc::PreparedBf16Flash full{nullptr, nullptr, kExtent, 0U, false};
    const cc::PreparedBf16Flash sliding{nullptr, nullptr, kExtent, kWindow, true};
    const bool refused = cc::append_attention_prepared_bf16_lanes(dims, full, images, gap, nullptr) == cudaErrorInvalidValue &&
                         cc::append_attention_prepared_bf16_lanes(dims, sliding, images, ring, nullptr) == cudaErrorInvalidValue;
    check(cudaDeviceSynchronize(), "sync");
    const bool untouched = same_bytes(prefix, k0.download()) && same_bytes(prefix, v0.download());
    std::printf("attn-lanes sequence-refusals %s\n", refused && untouched ? "PASS" : "FAIL");
    return refused && untouched;
}

}  // namespace

int main() {
    int cases = 0, failed = 0;
    const auto run = [&](bool ok) { ++cases; failed += ok ? 0 : 1; };
    std::vector<unsigned> p1{311}, p3{0, 1500, 77}, p8, p17;
    for (unsigned i = 0; i < 8; ++i) p8.push_back(5U + 211U * i);
    for (unsigned i = 0; i < 17; ++i) p17.push_back(3U + 173U * i);
    for (const auto pipe : {cc::FlashPipe::pad2, cc::FlashPipe::swz3}) {
        const bool pad = pipe == cc::FlashPipe::pad2;
        run(run_case(pad ? "full-1-pad2" : "full-1-swz3", false, pipe, p1, 11));
        run(run_case(pad ? "full-3-pad2" : "full-3-swz3", false, pipe, p3, 12));
        run(run_case(pad ? "full-8-pad2" : "full-8-swz3", false, pipe, p8, 13));
        run(run_case(pad ? "full-17-pad2" : "full-17-swz3", false, pipe, p17, 16));
        run(run_case(pad ? "sliding-8-pad2" : "sliding-8-swz3", true, pipe, p8, 14));
        run(run_case(pad ? "sliding-17-pad2" : "sliding-17-swz3", true, pipe, p17, 15));
        // One sequence's verify window on one cache (rows 2..17; 17 crosses a launch block).
        run(run_sequence_case(pad ? "sequence-full-3-pad2" : "sequence-full-3-swz3", false, pipe, 700U, 3U, 21));
        run(run_sequence_case(pad ? "sequence-full-17-pad2" : "sequence-full-17-swz3", false, pipe, 1023U, 17U, 22));
        run(run_sequence_case(pad ? "sequence-sliding-8-pad2" : "sequence-sliding-8-swz3", true, pipe, 2500U, 8U, 23));
        run(run_sequence_case(pad ? "sequence-sliding-17-pad2" : "sequence-sliding-17-swz3", true, pipe, 3000U, 17U, 24));
    }
    run(refuses_before_store());
    run(sequence_refusals());
    if (failed != 0) {
        std::printf("attn-lanes EXACT FAIL failed=%d cases=%d\n", failed, cases);
        return 1;
    }
    std::printf("attn-lanes EXACT PASS cases=%d\n", cases);
    return 0;
}
