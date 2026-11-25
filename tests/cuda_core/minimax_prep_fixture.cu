#include "cuda/models/minimax/mm_rtx_kernels.cuh"
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

namespace mm = strix::minimax::rtxpf::cuda;
namespace frozen = mmprep_frozen;
using k5c_fixture::check;
using k5c_fixture::compare;
using k5c_fixture::Device;
using k5c_fixture::values;

constexpr unsigned kWidth = 128, kRotaryHalf = 32, kPositions = 79, kLayers = 3, kGuard = 17;
constexpr std::size_t kIndexStride = kPositions * kWidth + 13U;
constexpr std::uint16_t kCacheGuard = 0x6a55;
constexpr float kQueryGuard = -9421.25F;

struct Sequence {
    unsigned rows, pos, capacity;
};

struct CacheImages {
    Device<std::uint16_t> k, v, ik;

    explicit CacheImages(unsigned capacity)
        : k(kLayers * std::size_t{capacity} * 4U * kWidth + 2U * kGuard, kCacheGuard),
          v(kLayers * std::size_t{capacity} * 4U * kWidth + 2U * kGuard, kCacheGuard),
          ik(kLayers * kIndexStride + 2U * kGuard, kCacheGuard) {}
};

struct CachePair {
    CacheImages reference, actual;
    explicit CachePair(unsigned capacity) : reference(capacity), actual(capacity) {}
};

std::vector<float> input(unsigned rows, unsigned heads, unsigned seed) {
    auto data = values(std::size_t{rows} * heads * kWidth + 2U * kGuard, seed);
    std::fill_n(data.begin(), kGuard, kQueryGuard);
    std::fill_n(data.rbegin(), kGuard, kQueryGuard);
    return data;
}

void compare_query(const char* plane, const Device<float>& reference, const Device<float>& actual) {
    const auto image = actual.read();
    compare(plane, reference.read(), image);
    for (unsigned i = 0; i < kGuard; ++i)
        if (image[i] != kQueryGuard || image[image.size() - 1U - i] != kQueryGuard)
            throw std::runtime_error("query guard changed");
}

void compare_cache(const char* plane, const Device<std::uint16_t>& reference, const Device<std::uint16_t>& actual,
                   std::size_t start, std::size_t count) {
    const auto image = actual.read();
    compare(plane, reference.read(), image);
    for (std::size_t i = 0; i < image.size(); ++i) {
        const bool live = i >= start && i < start + count;
        if ((image[i] != kCacheGuard) != live) throw std::runtime_error("unexpected cache write extent");
    }
}

void run_case(mm::MsaPrep& prep, std::span<const Sequence> seqs, unsigned layer) {
    std::vector<std::unique_ptr<CachePair>> images;
    std::vector<mm::SeqView> actual_views;
    std::vector<frozen::SeqView> ref_views;
    std::vector<std::uint32_t> row_map;
    unsigned rows = 0;
    for (const auto& seq : seqs) {
        const auto seq_id = static_cast<std::uint32_t>(actual_views.size());
        const auto& pair = *images.emplace_back(std::make_unique<CachePair>(seq.capacity));
        actual_views.push_back({rows, seq.rows, seq.capacity, pair.actual.k.data + kGuard,
                                pair.actual.v.data + kGuard, pair.actual.ik.data + kGuard, seq.pos});
        ref_views.push_back({rows, seq.rows, seq.capacity, pair.reference.k.data + kGuard,
                             pair.reference.v.data + kGuard, pair.reference.ik.data + kGuard, seq.pos});
        row_map.insert(row_map.end(), seq.rows, seq_id);
        rows += seq.rows;
    }
    Device<std::uint32_t> row_map_d(row_map);
    Device<frozen::SeqView> ref_views_d(ref_views);

    const auto q = input(rows, 64, 13);
    const auto k = input(rows, 4, 37);
    auto v = input(rows, 4, 51);
    auto iq = input(rows, 4, 23);
    const auto ik = input(rows, 1, 61);
    constexpr std::array ties{0x3f808000U, 0x3f818000U, 0xbf808000U, 0xbf818000U, 0x80000000U, 0x00008000U};
    for (std::size_t i = 0; i < ties.size(); ++i) v[kGuard + i] = std::bit_cast<float>(ties[i]);
    std::fill_n(iq.begin() + kGuard, kWidth, 0.0F);
    Device<float> q_ref(q), q_actual(q), iq_ref(iq), iq_actual(iq), k_d(k), v_d(v), ik_d(ik);
    Device<float> q_gain(values(kWidth, 71)), k_gain(values(kWidth, 83));
    Device<float> iq_gain(values(kWidth, 97)), ik_gain(values(kWidth, 113));

    std::vector<frozen::RopeRow> ref_rope(kPositions);
    std::vector<float> actual_rope(kPositions * 2U * kRotaryHalf);
    for (unsigned p = 0; p < kPositions; ++p) {
        for (unsigned i = 0; i < kRotaryHalf; ++i) {
            const float angle = static_cast<float>(p * (i + 1U)) / 193.0F;
            actual_rope[p * 2U * kRotaryHalf + i] = ref_rope[p].c[i] = std::cos(angle);
            actual_rope[p * 2U * kRotaryHalf + kRotaryHalf + i] = ref_rope[p].s[i] = std::sin(angle);
        }
    }
    Device<frozen::RopeRow> ref_rope_d(ref_rope);
    Device<float> actual_rope_d(actual_rope);
    const frozen::MsaLayer ref_layer{layer, q_gain.data, k_gain.data, iq_gain.data, ik_gain.data, kIndexStride};
    frozen::msa_prep_kernel<<<dim3(rows, 73), 32>>>(ref_views_d.data, row_map_d.data, ref_layer,
        q_ref.data + kGuard, k_d.data + kGuard, v_d.data + kGuard, iq_ref.data + kGuard,
        ik_d.data + kGuard, ref_rope_d.data);
    check(cudaGetLastError(), "launch frozen MiniMax prep");
    prep.prepare(actual_views, row_map_d.data, rows, kLayers, kIndexStride, actual_rope_d.data, kPositions);
    prep.launch(layer, {q_gain.data, k_gain.data, iq_gain.data, ik_gain.data}, q_actual.data + kGuard,
                k_d.data + kGuard, v_d.data + kGuard, iq_actual.data + kGuard, ik_d.data + kGuard, nullptr);
    check(cudaDeviceSynchronize(), "complete MiniMax prep");

    compare_query("MiniMax Q", q_ref, q_actual);
    compare_query("MiniMax IQ", iq_ref, iq_actual);
    compare("MiniMax read-only K", k, k_d.read());
    compare("MiniMax read-only V", v, v_d.read());
    compare("MiniMax read-only IK", ik, ik_d.read());
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        const auto& seq = seqs[i];
        const auto& pair = *images[i];
        const auto kv_start = kGuard + (std::size_t{layer} * seq.capacity + seq.pos) * 4U * kWidth;
        const auto ik_start = kGuard + layer * kIndexStride + std::size_t{seq.pos} * kWidth;
        compare_cache("MiniMax K", pair.reference.k, pair.actual.k, kv_start, std::size_t{seq.rows} * 4U * kWidth);
        compare_cache("MiniMax V", pair.reference.v, pair.actual.v, kv_start, std::size_t{seq.rows} * 4U * kWidth);
        compare_cache("MiniMax IK", pair.reference.ik, pair.actual.ik, ik_start, std::size_t{seq.rows} * kWidth);
    }
}

}  // namespace

int main() {
    try {
        mm::MsaPrep prep;
        prep.prepare({}, nullptr, 0, kLayers, 0, nullptr, 0);
        prep.launch(0, {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        constexpr std::array multi{Sequence{33, 3, 40}, Sequence{0, 0, 1}, Sequence{3, 72, 76}};
        constexpr std::array cut{Sequence{32, 31, kPositions}};
        constexpr std::array tail{Sequence{1, kPositions - 1, kPositions}};
        run_case(prep, multi, 1);
        run_case(prep, cut, 0);
        prep.reset();
        run_case(prep, tail, kLayers - 1);
        std::puts("PASS MiniMax RTX MMPREP prep images bytes");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MiniMax RTX MMPREP: %s\n", error.what());
        return 1;
    }
}
