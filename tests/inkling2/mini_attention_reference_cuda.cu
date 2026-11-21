#include "strix/cuda_core/attention.hpp"

#include "attn_core/causal_gqa_reference.hpp"
#include "inkling2/inkling2_mini_attention_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace cc = strix::cuda_core;
constexpr unsigned kHeads = 6, kKvHeads = 2, kDim = 128, kRelativeExtent = 13;
constexpr unsigned kQWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;

void require(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct DeviceBuffer {
    explicit DeviceBuffer(std::size_t size) {
        require(cudaMallocManaged(&data, size * sizeof(T)));
    }
    ~DeviceBuffer() { require(cudaFree(data)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

float value(std::size_t i, unsigned seed) {
    return static_cast<float>(static_cast<int>((i * 19U + i / 13U + seed) % 37U) - 18) * 0.03125F;
}

bool check(unsigned first, unsigned rows, unsigned window, unsigned capacity, bool use_bias, bool use_sink) {
    const std::size_t q_size = std::size_t{rows} * kQWidth;
    const std::size_t kv_size = std::size_t{capacity} * kKvWidth;
    DeviceBuffer<float> q(q_size), output(q_size);
    DeviceBuffer<float> bias(std::size_t{rows} * kHeads * kRelativeExtent), sinks(std::size_t{rows} * kHeads);
    DeviceBuffer<__nv_bfloat16> k(kv_size), v(kv_size);
    std::vector<float> ref_k(kv_size), ref_v(kv_size), expected(q_size);
    std::vector<float> input_k(std::size_t{rows} * kKvWidth), input_v(input_k.size());
    DeviceBuffer<float> append_k(input_k.size()), append_v(input_v.size());

    for (std::size_t i = 0; i < q_size; ++i) q.data[i] = value(i, 1U);
    for (std::size_t i = 0; i < std::size_t{rows} * kHeads * kRelativeExtent; ++i) bias.data[i] = 3.0F * value(i, 2U);
    for (std::size_t i = 0; i < std::size_t{rows} * kHeads; ++i) sinks.data[i] = 3.0F + 2.0F * value(i, 3U);
    const unsigned first_live = strix::attn_core::mask::window_lo(first, window);
    for (unsigned p = first_live; p < first + rows; ++p) {
        const std::size_t slot = std::size_t{window ? p % capacity : p} * kKvWidth;
        for (unsigned i = 0; i < kKvWidth; ++i) {
            const auto index = std::size_t{p} * kKvWidth + i;
            const float key = value(index, 7U), val = value(index, 11U);
            if (p >= first) {
                const std::size_t current = std::size_t{p - first} * kKvWidth + i;
                input_k[current] = key;
                input_v[current] = val;
            } else {
                k.data[slot + i] = __float2bfloat16_rn(key);
                v.data[slot + i] = __float2bfloat16_rn(val);
            }
        }
    }

    // The host reference streams rows itself and needs only a window-sized ring.
    for (unsigned p = first_live; p < first; ++p) {
        const std::size_t slot = std::size_t{window ? p % window : p} * kKvWidth;
        for (unsigned i = 0; i < kKvWidth; ++i) {
            const auto index = std::size_t{p} * kKvWidth + i;
            ref_k[slot + i] = value(index, 7U);
            ref_v[slot + i] = value(index, 11U);
        }
    }

    const auto relative = use_bias ? std::span<const float>(bias.data, std::size_t{rows} * kHeads * kRelativeExtent)
                                   : std::span<const float>();
    const auto sink = use_sink ? std::span<const float>(sinks.data, std::size_t{rows} * kHeads)
                               : std::span<const float>();
    strix::attn_core::causal_gqa_reference(
        {kHeads, kKvHeads, kDim, window},
        {first, {q.data, q_size}, input_k, input_v, use_bias ? kRelativeExtent : 0U, relative, sink},
        {ref_k, ref_v}, expected);

    const cc::AttentionDims dims{kHeads, kKvHeads, kDim, 1.0F / std::sqrt(static_cast<float>(kDim)), capacity};
    const cc::PreparedBf16Flash prepared{relative.data(), sink.data(), use_bias ? kRelativeExtent : 0U, window, window != 0};
    std::ranges::copy(input_k, append_k.data);
    std::ranges::copy(input_v, append_v.data);
    require(cc::kv_store_prepared_bf16(dims, append_k.data, append_v.data, k.data, v.data,
                                      first, rows, window != 0, nullptr));
    require(cc::attention_prefill_prepared_bf16(dims, prepared, q.data, k.data, v.data, first, rows, output.data, nullptr));
    require(cudaDeviceSynchronize());
    const std::vector<float> padded(output.data, output.data + q_size);
    require(cc::attention_prefill_prepared_bf16(dims, prepared, q.data, k.data, v.data, first, rows,
                                               output.data, nullptr, cc::FlashPipe::swz3));
    require(cudaDeviceSynchronize());

    for (unsigned row = 0; row < rows; ++row) {
        const unsigned pos = first + row;
        const std::size_t slot = std::size_t{window ? pos % capacity : pos} * kKvWidth;
        for (unsigned i = 0; i < kKvWidth; ++i) {
            const auto source = std::size_t{row} * kKvWidth + i;
            if (__bfloat162float(k.data[slot + i]) != input_k[source] ||
                __bfloat162float(v.data[slot + i]) != input_v[source]) return false;
        }
    }

    float max_error = 0.0F;
    for (std::size_t i = 0; i < q_size; ++i) {
        if (!std::isfinite(output.data[i]) || output.data[i] != padded[i]) return false;
        max_error = std::max(max_error, std::abs(output.data[i] - expected[i]));
    }
    if (max_error > 0.006F) std::cerr << "CUDA relative/sink attention max error " << max_error << '\n';
    return max_error <= 0.006F;
}

bool check_mini_visit_slices() {
    namespace mini = strix::inkling2;
    constexpr auto schedule = mini::mini_schedule();
    constexpr auto initial = schedule[6], repeat = schedule[18];
    constexpr auto session = mini::MiniSessionStatePlan(2, 1);
    DeviceBuffer<std::uint16_t> k(session.kv_elements_per_plane()), v(session.kv_elements_per_plane());
    const mini::MiniSessionKvPlanes planes{{k.data, session.kv_elements_per_plane()},
                                           {v.data, session.kv_elements_per_plane()}};
    std::ranges::fill(planes.keys, std::uint16_t{0});
    std::ranges::fill(planes.values, std::uint16_t{0});
    DeviceBuffer<float> q(mini::kQueryWidth), kin(mini::kKvWidth), vin(mini::kKvWidth), output(mini::kQueryWidth);
    DeviceBuffer<float> relative(mini::kQueryHeads * mini::kRelativeExtent), sinks(mini::kQueryHeads);
    std::fill_n(q.data, mini::kQueryWidth, 0.0F);
    std::fill_n(kin.data, mini::kKvWidth, 0.0F);
    std::fill_n(vin.data, mini::kKvWidth, 2.0F);
    std::fill_n(relative.data, mini::kQueryHeads * mini::kRelativeExtent, 0.0F);
    std::fill_n(sinks.data, mini::kQueryHeads, 0.0F);
    const mini::MiniCudaAttentionRows rows{q.data, kin.data, vin.data, relative.data, sinks.data, output.data, 0, 1};
    const auto first_cache = mini::mini_visit_kv_cache(session, initial, planes);
    const auto repeat_cache = mini::mini_visit_kv_cache(session, repeat, planes);
    const auto invalid_pipe = static_cast<cc::FlashPipe>(255);
    if (mini::enqueue_mini_cuda_attention(initial, rows, session, planes, nullptr, invalid_pipe) !=
        cudaErrorInvalidValue) return false;
    auto misaligned = rows;
    ++misaligned.queries;
    if (mini::enqueue_mini_cuda_attention(initial, misaligned, session, planes, nullptr) !=
        cudaErrorInvalidValue) return false;
    require(cudaDeviceSynchronize());
    if (first_cache.values[0] != 0U) return false;

    DeviceBuffer<unsigned> failure(1U);
    *failure.data = 0U;
    const auto enqueue = [&] {
        require(mini::enqueue_mini_cuda_attention(initial, rows, first_cache, nullptr,
                                                  cc::FlashPipe::pad2, failure.data));
        require(cudaDeviceSynchronize());
    };
    enqueue();
    for (unsigned i = 0; i < mini::kKvWidth; ++i) {
        if (first_cache.values[i] != 0x4000U || repeat_cache.values[i] != 0U) return false;
    }
    if (*failure.data != 0U || !std::ranges::all_of(std::span<const float>(output.data, mini::kQueryWidth),
                                                   [](float element) { return element == 1.0F; })) return false;

    constexpr auto inf = std::numeric_limits<float>::infinity();
    for (float* operand : {q.data, kin.data, vin.data, relative.data, sinks.data}) {
        const float saved = *operand;
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), inf, -inf}) {
            *operand = bad;
            *failure.data = 0U;
            enqueue();
            if (*failure.data == 0U) return false;
        }
        *operand = saved;
    }
    return true;
}

}  // namespace

int main() {
    const cc::AttentionDims invalid{kHeads, kKvHeads, kDim, 0.1F, 8};
    alignas(16) float q_dummy = 0.0F;
    alignas(16) float out_dummy = 0.0F;
    alignas(16) __nv_bfloat16 k_dummy{}, v_dummy{};
    if (cc::attention_prefill_prepared_bf16(invalid, {nullptr, nullptr, 0, 8, true},
                                         &q_dummy, &k_dummy, &v_dummy, 65, 3, &out_dummy, nullptr)
        != cudaErrorInvalidValue) return 1;
    return check(91, 23, 17, 39, true, true) && check(91, 23, 17, 39, true, false) &&
           check(35, 131, 0, 170, true, true) &&
           check(5, 23, 0, 28, false, false) && check_mini_visit_slices() ? 0 : 1;
}
