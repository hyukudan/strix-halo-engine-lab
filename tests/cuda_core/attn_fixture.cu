#include "cuda/core/attn_decode.cuh"

#include "core/row_transforms.hpp"
#include "strix/cuda_core/attention.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
namespace attn = strix::cuda::core::attn;
namespace incumbent = strix::cuda_core;
namespace transform = strix::core::row_transforms;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "k4c %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        if (count != 0) check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
    }
    explicit Device(const std::vector<T>& data) : Device(data.size()) {
        check(cudaMemcpy(ptr_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { if (ptr_) (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] T* data() const { return ptr_; }
    void poison() const { check(cudaMemset(ptr_, 0xff, count_ * sizeof(T)), "poison output"); }
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
    template <typename T>
    std::vector<T> make(std::size_t count, float scale) {
        std::vector<T> result(count);
        for (auto& v : result) v = T(next(scale));
        return result;
    }

private:
    std::uint32_t state_;
};

void compare(const char* name, unsigned context, const Device<float>& reference,
             const Device<float>& candidate) {
    const auto want = reference.download(), got = candidate.download();
    for (std::size_t i = 0; i < want.size(); ++i) {
        const auto expected = std::bit_cast<std::uint32_t>(want[i]);
        const auto actual = std::bit_cast<std::uint32_t>(got[i]);
        if (std::isfinite(want[i]) && actual == expected) continue;
        std::fprintf(stderr, "k4c %s-%u FAIL element %zu: %08x vs %08x\n", name, context, i, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}

attn::DecodePlan plan_for(const attn::DecodeSpec& spec) {
    const auto plan = attn::make_decode_plan(spec);
    if (!plan) {
        std::fputs("k4c fixture plan rejected\n", stderr);
        std::exit(EXIT_FAILURE);
    }
    return *plan;
}

void dense(unsigned context) {
    attn::DecodeSpec spec{256, 24, 4, 1, context, context};
    spec.gate = attn::OutputGate::sigmoid_interleaved;
    spec.scale = 0.0625F;
    const auto plan = plan_for(spec);
    const auto n = std::size_t{spec.heads} * spec.head_dim;
    const auto cache = std::size_t{spec.cache_slots} * spec.kv_heads * spec.head_dim;
    Values values(context);
    Device<float> qg(values.make<float>(2 * n, 2.75F));
    Device<__nv_bfloat16> k(values.make<__nv_bfloat16>(cache, 2.0F));
    Device<__nv_bfloat16> v(values.make<__nv_bfloat16>(cache, 0.75F));
    Device<std::uint32_t> position(std::vector<std::uint32_t>{context - 1U});
    const incumbent::AttentionDims dims{spec.heads, spec.kv_heads, spec.head_dim, spec.scale, spec.cache_slots};
    const incumbent::GroupedView view{1, spec.heads, spec.head_dim, spec.heads * spec.head_dim * 2U, spec.head_dim * 2U};
    Device<float> ref(n), got(n), ref_scratch(incumbent::attention_scratch_floats(dims, 1)), scratch(plan.scratch_floats);
    ref.poison();
    got.poison();
    check(incumbent::attention_decode(dims, qg.data(), view, qg.data() + spec.head_dim, view, k.data(), v.data(),
                                      position.data(), 1, ref.data(), ref_scratch.data(), nullptr), "dense incumbent");
    attn::DecodeInput input{qg.data(), k.data(), v.data()};
    input.device_position = position.data();
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "dense family");
    compare("qwen27b", context, ref, got);
    std::printf("k4c qwen27b-%u PASS\n", context);
}

void prepared(const char* name, unsigned context, unsigned heads, unsigned kv_heads,
              unsigned window, unsigned rows, incumbent::FlashPipe pipe) {
    const unsigned slots = window == 0 ? context : window;
    const unsigned first = context - rows;
    attn::DecodeSpec spec{128, heads, kv_heads, rows, context, slots, window, 1024, 0.08838834764831844F,
        attn::DecodeOrder::online_bf16, window ? attn::KvLayout::ring : attn::KvLayout::slot_major,
        attn::OutputGate::none, true};
    const auto n = std::size_t{rows} * heads * spec.head_dim;
    const auto cache = std::size_t{slots} * kv_heads * spec.head_dim;
    Values values(context + heads);
    Device<float> q(values.make<float>(n, 2.75F)), g(values.make<float>(n, 4.0F));
    Device<float> rel(values.make<float>(std::size_t{rows} * heads * spec.relative_extent, 1.25F));
    Device<float> sinks(values.make<float>(std::size_t{rows} * heads, 3.0F));
    Device<__nv_bfloat16> k(values.make<__nv_bfloat16>(cache, 1.75F));
    Device<__nv_bfloat16> v(values.make<__nv_bfloat16>(cache, 0.75F));
    Device<std::uint32_t> position(std::vector<std::uint32_t>{first});
    const auto* device_position = rows == 1 ? position.data() : nullptr;
    const incumbent::AttentionDims dims{heads, kv_heads, spec.head_dim, spec.scale, slots};
    const incumbent::PreparedBf16Flash logits{rel.data(), sinks.data(), spec.relative_extent, window, window != 0};
    Device<float> ref(n), gated(n), got(n);
    ref.poison();
    got.poison();
    check(incumbent::attention_prefill_prepared_bf16(dims, logits, q.data(), k.data(), v.data(), first,
        rows, ref.data(), nullptr, pipe, device_position), "prepared incumbent");
    const attn::DecodeInput input{q.data(), k.data(), v.data(), rel.data(), sinks.data(),
        g.data(), first, device_position};
    check(attn::decode(plan_for(spec), input, got.data(), nullptr, 0, nullptr), "prepared family");
    compare(name, context, ref, got);

    gated.poison();
    got.poison();
    check(transform::elements(n, transform::Bf16SigmoidProduct{ref.data(), g.data(), gated.data()}, nullptr),
          "prepared incumbent gate");
    spec.gate = attn::OutputGate::sigmoid_bf16;
    check(attn::decode(plan_for(spec), input, got.data(), nullptr, 0, nullptr), "prepared gated family");
    compare(name, context, gated, got);
    std::printf("k4c %s-%u PASS\n", name, context);
}
}  // namespace

int main() {
    for (unsigned context : {1U, 17U, 4096U, 8191U}) {
        prepared("inkling2-window", context, 24, 8, 1024, 1, incumbent::FlashPipe::pad2);
        prepared("prepared-full", context, 12, 3, 0, context < 3U ? context : 3U, incumbent::FlashPipe::swz3);
        dense(context);
    }
}
