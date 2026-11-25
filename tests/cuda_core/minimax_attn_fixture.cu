#include "cuda/models/minimax/mm_rtx_kernels.cuh"
#include "mmattn_minimax_frozen.cuh"
#include "select_fixture_common.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

namespace mm = strix::minimax::rtxpf::cuda;
namespace frozen = mmattn_frozen;
using k5c_fixture::check;
using k5c_fixture::compare;
using k5c_fixture::Device;
using k5c_fixture::values;

constexpr unsigned kLayer = 1, kRows = 7, kGuard = 17;
constexpr unsigned kRowWidth = 64 * 128, kCacheWidth = 4 * 128;
constexpr unsigned kOutputs = kRows * kRowWidth;
constexpr float kCanary = -9317.0F;

std::vector<std::uint16_t> bf16_values(std::size_t size, unsigned seed) {
    std::vector<std::uint16_t> image;
    image.reserve(size);
    for (float x : values(size, seed))
        image.push_back(static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x) >> 16U));
    return image;
}

void run() {
    // These two sequences cross block boundaries at different absolute positions.
    // The row map keeps the empty sequence's slot, just as the producer does.
    std::array sequences{mm::SeqView{0, 4, 263, nullptr, nullptr, nullptr, 254},
                         mm::SeqView{4, 0, 0, nullptr, nullptr, nullptr},
                         mm::SeqView{4, 3, 2057, nullptr, nullptr, nullptr, 2047}};
    constexpr unsigned cache_rows = 263 + 2057;
    constexpr auto cache_size = (kLayer + 1U) * cache_rows * kCacheWidth;
    Device<std::uint16_t> k(bf16_values(cache_size, 19)), v(bf16_values(cache_size, 37));
    auto query = values(kOutputs, 23);
    std::fill_n(query.begin(), 128, -0.0F);
    query[128] = std::bit_cast<float>(0x7fc12000U);
    Device<float> query_d(query);

    std::vector<strix::attn::SelectedBlockSequence> actual_views;
    std::vector<frozen::SeqView> reference_views;
    std::vector<std::uint32_t> row_map(kRows);
    std::vector<std::uint32_t> blocks(kRows * 4U * frozen::kTopK, 0xffffffffU);
    std::vector<std::uint32_t> counts(kRows * 4U);
    std::size_t offset = 0;
    for (unsigned i = 0; i < sequences.size(); ++i) {
        auto& seq = sequences[i];
        if (seq.rows != 0) {
            seq.k = k.data + offset;
            seq.v = v.data + offset;
        }
        offset += std::size_t{kLayer + 1U} * seq.cap * kCacheWidth;
        actual_views.push_back(mm::msa_attention_sequence(seq));
        reference_views.push_back({seq.row0, seq.rows, seq.cap, seq.k, seq.v, nullptr, seq.pos0});
        for (unsigned r = 0; r < seq.rows; ++r) {
            const auto row = seq.row0 + r;
            const auto visible_blocks = (seq.pos0 + r) / frozen::kBlock + 1U;
            row_map[row] = i;
            for (unsigned g = 0; g < 4; ++g) {
                if (row == 0 && g == 3) continue;  // Empty list for one group only.
                const auto item = row * 4U + g;
                const auto count = std::min(visible_blocks, frozen::kTopK - g);
                counts[item] = count;
                for (unsigned j = 0; j < count; ++j)
                    blocks[item * frozen::kTopK + j] = (visible_blocks + g - j) % visible_blocks;
            }
        }
    }
    Device<strix::attn::SelectedBlockSequence> actual_d(actual_views);
    Device<frozen::SeqView> reference_d(reference_views);
    Device<std::uint32_t> row_map_d(row_map), blocks_d(blocks), counts_d(counts);

    constexpr unsigned frozen_shared = 128 * frozen::kKStride * 2 + 128 * 128 * 2 + 16 * 128 * 4;
    check(cudaFuncSetAttribute(frozen::attend_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, frozen_shared),
          "frozen MiniMax staged shared memory");
    mm::prepare_msa_attention();
    for (const bool legacy : {true, false}) {
        Device<float> expected(kOutputs + 2U * kGuard, kCanary), actual(kOutputs + 2U * kGuard, kCanary);
        if (legacy) {
            frozen::attend_kernel<<<kRows * 8U, 256>>>(reference_d.data, row_map_d.data, kRows, kLayer, query_d.data,
                blocks_d.data, counts_d.data, expected.data + kGuard);
        } else {
            frozen::attend_shared_kernel<<<dim3(kRows, 4), 512, frozen_shared>>>(reference_d.data, row_map_d.data,
                kLayer, query_d.data, blocks_d.data, counts_d.data, expected.data + kGuard, 0);
        }
        check(cudaGetLastError(), "frozen MiniMax attention launch");
        mm::launch_msa_attention(actual_d.data, row_map_d.data, kRows, kLayer, query_d.data, blocks_d.data,
                                 counts_d.data, actual.data + kGuard, nullptr, legacy);
        check(cudaDeviceSynchronize(), "complete MiniMax attention");
        const auto image = actual.read();
        compare(legacy ? "MiniMax warp attention" : "MiniMax staged attention", expected.read(), image);
        for (std::size_t i = 0; i < image.size(); ++i) {
            const bool guard = i < kGuard || i >= kGuard + kOutputs;
            if (guard != (image[i] == kCanary)) throw std::runtime_error("unexpected attention write extent");
        }
    }
}

}  // namespace

int main() {
    try {
        run();
        std::puts("PASS MiniMax RTX MMATTN attention bytes (warp, staged)");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MiniMax RTX MMATTN: %s\n", error.what());
        return 1;
    }
}
