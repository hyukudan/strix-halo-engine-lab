#include "cuda/core/chain/prep.cuh"
#include "mmprep_minimax_frozen.cuh"
#include "select_fixture_common.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace ch = strix::cuda::core::chain;
namespace prep = strix::ops::chain;
namespace frozen = mmprep_frozen;
using k5c_fixture::Device;
using k5c_fixture::compare;
using k5c_fixture::values;

constexpr unsigned width = 128, q_heads = 64, kv_heads = 4, iq_heads = 4;
constexpr unsigned rotary_half = 32, rotary_positions = 21;
constexpr unsigned layers = 3, guard = 17;
constexpr std::size_t index_stride = 18U * width + 19U;
constexpr std::uint16_t bf16_guard = 0x6a55;
constexpr float fp32_guard = -9421.25F;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

std::vector<float> input(unsigned rows, unsigned heads, unsigned seed) {
    auto result = values(std::size_t{rows} * heads * width + 2U * guard, seed);
    std::fill_n(result.begin(), guard, fp32_guard);
    std::fill_n(result.rbegin(), guard, fp32_guard);
    return result;
}

void check_cache(const char* name, const Device<std::uint16_t>& reference, const Device<std::uint16_t>& shared,
                 std::size_t begin, std::size_t count) {
    const auto bytes = shared.read();
    compare(name, reference.read(), bytes);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const bool live = i >= begin && i < begin + count;
        require((bytes[i] != bf16_guard) == live, "cache write outside live range or missed live element");
    }
}

void check_queries(const char* name, const Device<float>& reference, const Device<float>& shared) {
    const auto image = shared.read();
    compare(name, reference.read(), image);
    for (unsigned i = 0; i < guard; ++i)
        require(image[i] == fp32_guard && image[image.size() - 1U - i] == fp32_guard, "FP32 guard modified");
}

struct CacheImages {
    Device<std::uint16_t> key, value, index_key;

    explicit CacheImages(unsigned capacity)
        : key(layers * std::size_t{capacity} * kv_heads * width + 2U * guard, bf16_guard),
          value(layers * std::size_t{capacity} * kv_heads * width + 2U * guard, bf16_guard),
          index_key(layers * index_stride + 2U * guard, bf16_guard) {}
};

void run_case(std::span<const prep::PrepSequence> seqs, unsigned rows, unsigned layer) {
    const auto plan = prep::plan_prep({rows, q_heads, kv_heads, iq_heads, 1, rotary_half, rotary_positions},
                                     seqs, layer, index_stride);
    require(plan.has_value(), "fixture plan rejected");
    require(ch::launch_prep(*plan, {}, nullptr) == cudaErrorInvalidValue, "missing live bindings accepted");
    auto q = input(rows, q_heads, 13);
    auto iq = input(rows, iq_heads, 23);
    const auto k = input(rows, kv_heads, 37);
    auto v = input(rows, kv_heads, 51);
    const auto ik = input(rows, 1, 61);
    // BF16 exact halfway cases, signed zero and subnormal input on the raw V arm.
    constexpr std::array special{0x3f808000U, 0x3f818000U, 0xbf808000U, 0xbf818000U, 0x80000000U, 0x00008000U};
    for (std::size_t i = 0; i < special.size(); ++i) v[guard + i] = std::bit_cast<float>(special[i]);
    // The rotated zero head exercises the epsilon and the sign of a post-gain zero.
    std::fill_n(iq.begin() + guard, width, 0.0F);

    Device<float> q_ref(q), q_shared(q), iq_ref(iq), iq_shared(iq), k_d(k), v_d(v), ik_d(ik);
    Device<float> q_gain(values(width, 71)), k_gain(values(width, 83));
    Device<float> iq_gain(values(width, 97)), ik_gain(values(width, 113));
    std::vector<frozen::RopeRow> rope_ref(rotary_positions);
    std::vector<float> rope_shared(rotary_positions * 2U * rotary_half);
    for (unsigned p = 0; p < rotary_positions; ++p) {
        for (unsigned i = 0; i < rotary_half; ++i) {
            const float angle = static_cast<float>(p * (i + 1U)) / 193.0F;
            rope_shared[p * 2U * rotary_half + i] = rope_ref[p].c[i] = std::cos(angle);
            rope_shared[p * 2U * rotary_half + rotary_half + i] = rope_ref[p].s[i] = std::sin(angle);
        }
    }
    Device<frozen::RopeRow> rope_ref_d(rope_ref);
    Device<float> rope_shared_d(rope_shared);

    std::vector<std::unique_ptr<CacheImages>> ref, shared;
    std::vector<frozen::SeqView> ref_views;
    std::vector<ch::PrepSequenceView> shared_views;
    std::vector<std::uint32_t> row_seq(rows);
    unsigned seq_id = 0;
    for (const auto& seq : seqs) {
        const auto& a = *ref.emplace_back(std::make_unique<CacheImages>(seq.capacity));
        const auto& b = *shared.emplace_back(std::make_unique<CacheImages>(seq.capacity));
        ref_views.push_back({seq.first_row, seq.rows, seq.capacity,
                             a.key.data + guard, a.value.data + guard, a.index_key.data + guard, seq.first_position});
        shared_views.push_back({seq, b.key.data + guard, b.value.data + guard, b.index_key.data + guard});
        std::fill_n(row_seq.begin() + seq.first_row, seq.rows, seq_id);
        ++seq_id;
    }
    Device<frozen::SeqView> ref_views_d(ref_views);
    Device<ch::PrepSequenceView> shared_views_d(shared_views);
    Device<std::uint32_t> row_seq_d(row_seq);
    const frozen::MsaLayer ref_layer{layer, q_gain.data, k_gain.data, iq_gain.data, ik_gain.data, index_stride};
    frozen::msa_prep_kernel<<<dim3(rows, 73), 32>>>(ref_views_d.data, row_seq_d.data, ref_layer,
        q_ref.data + guard, k_d.data + guard, v_d.data + guard, iq_ref.data + guard, ik_d.data + guard, rope_ref_d.data);
    check(cudaGetLastError());
    const ch::PrepBindings bindings{shared_views_d.data, row_seq_d.data, rope_shared_d.data,
        q_shared.data + guard, k_d.data + guard, v_d.data + guard, iq_shared.data + guard, ik_d.data + guard,
        q_gain.data, k_gain.data, iq_gain.data, ik_gain.data};
    check(ch::launch_prep(*plan, bindings, nullptr));
    check(cudaDeviceSynchronize());

    check_queries("MMPREP Q", q_ref, q_shared);
    check_queries("MMPREP IQ", iq_ref, iq_shared);
    compare("MMPREP read-only K input", k, k_d.read());
    compare("MMPREP read-only V input", v, v_d.read());
    compare("MMPREP read-only IK input", ik, ik_d.read());
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        const auto& seq = seqs[i];
        const auto kv_begin = guard + (std::size_t{layer} * seq.capacity + seq.first_position) * kv_heads * width;
        const auto kv_count = std::size_t{seq.rows} * kv_heads * width;
        const auto index_begin = guard + layer * index_stride + std::size_t{seq.first_position} * width;
        check_cache("MMPREP K", ref[i]->key, shared[i]->key, kv_begin, kv_count);
        check_cache("MMPREP V", ref[i]->value, shared[i]->value, kv_begin, kv_count);
        check_cache("MMPREP IK", ref[i]->index_key, shared[i]->index_key, index_begin, std::size_t{seq.rows} * width);
    }
    const auto first_v = shared.front()->value.read();
    const auto v_begin = guard + (std::size_t{layer} * seqs.front().capacity + seqs.front().first_position) * kv_heads * width;
    constexpr std::array<std::uint16_t, special.size()> rounded{0x3f80, 0x3f82, 0xbf80, 0xbf82, 0x8000, 0};
    require(std::equal(rounded.begin(), rounded.end(), first_v.begin() + static_cast<std::ptrdiff_t>(v_begin)),
            "raw V BF16 halfway rounding or signed zero differs");
}

}  // namespace

int main() {
    try {
        const auto empty = prep::plan_prep({}, {}, 0, 0);
        require(empty && ch::launch_prep(*empty, {}, nullptr) == cudaSuccess, "empty plan must be a no-op");
        constexpr std::array multi{prep::PrepSequence{0, 3, 2, 9}, prep::PrepSequence{3, 1, 6, 7},
                                   prep::PrepSequence{4, 3, 11, 19}};
        constexpr std::array single{prep::PrepSequence{0, 1, 15, 17}};
        run_case(multi, 7, 1);
        run_case(single, 1, 0);
        std::puts("MMPREP CUDA: 2 cases, complete Q/IQ FP32 and K/V/IK BF16 images EXACT, guards PASS");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MMPREP CUDA FAIL: %s\n", error.what());
        return 1;
    }
}
