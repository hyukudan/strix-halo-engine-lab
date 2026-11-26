#include "cuda/models/minimax/mm_rtx_index.hpp"
#include "cuda/models/minimax/mm_rtx_attn.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>

int main() {
    namespace mm = strix::minimax::rtxpf::cuda;
    constexpr unsigned rows = 5, max_blocks = 19, items = rows * mm::kIdxHeads;
    const std::array seqs{
        mm::SeqView{0, 2, 2, nullptr, nullptr, nullptr},
        mm::SeqView{2, 3, 2050, nullptr, nullptr, nullptr, 2047},
    };
    const std::array<unsigned, rows> expected_groups{1, 1, 16, 17, 17};
    std::array<float, items * max_blocks> scores{};
    std::array<std::uint32_t, items * mm::kTopK> selected{};
    std::array<std::uint32_t, items> counts{};
    const auto picks = mm::msa_pick_rows(seqs, rows, max_blocks, scores.data(), selected.data(), counts.data());
    static_assert(mm::kMsaPick.top_k == 16 && mm::kMsaPick.expansion == 1 && mm::kMsaPick.force_newest &&
                  mm::kMsaPick.order == mm::select::KeyOrder::canonical_float);
    if (picks.size() != items) return 1;
    for (unsigned i = 0; i < items; ++i) {
        const auto& pick = picks[i];
        if (pick.groups != expected_groups[i / mm::kIdxHeads] || pick.tail != 0 ||
            pick.scores != scores.data() + i * max_blocks || pick.selected != selected.data() + i * mm::kTopK ||
            pick.count != counts.data() + i) {
            std::fprintf(stderr, "MiniMax RTX pick descriptor mismatch at item %u\n", i);
            return 1;
        }
    }
    std::puts("PASS MiniMax RTX MSA pick descriptors");

    constexpr unsigned cap = 2178, layers = 2, width = 128;
    constexpr std::array geometry{
        mm::SeqView{0, 35, 2081, nullptr, nullptr, nullptr, 2046},
        mm::SeqView{35, 9, cap, nullptr, nullptr, nullptr, 2169},
        mm::SeqView{44, 2, 7, nullptr, nullptr, nullptr, 5},
    };
    constexpr auto cache = mm::msa_index_cache(nullptr, cap, layers);
    static_assert(cache.row_stride == width && cache.layer_stride == cap * width &&
                  cache.sequence_stride == layers * cap * width);
    static_assert(cache.offset(1, 1, 2169) == (3U * cap + 2169U) * width &&
                  cache.offset(2, 0, 6) == (4U * cap + 6U) * width);
    static_assert(mm::kMsaScore.top_k == 16 && mm::kMsaScore.heads == 4 &&
                  mm::kMsaScore.dim == width && mm::kMsaScore.block_width == width);
    const auto origins = mm::msa_score_sequences(geometry);
    if (origins.size() != geometry.size()) return 1;
    for (std::size_t i = 0; i < geometry.size(); ++i) {
        if (origins[i].first_row != geometry[i].row0 || origins[i].first_position != geometry[i].pos0) return 1;
    }
    const auto tiles = mm::msa_score_tiles(geometry);
    constexpr std::array expected_tiles{
        mm::select::ScoreTile{0, 32}, mm::select::ScoreTile{32, 3}, mm::select::ScoreTile{35, 9},
        mm::select::ScoreTile{44, 2},
    };
    if (tiles.size() != expected_tiles.size()) return 1;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].first_row != expected_tiles[i].first_row || tiles[i].rows != expected_tiles[i].rows) return 1;
    }
    std::puts("PASS MiniMax RTX MSA strided cache and geometry");

    namespace rtx = strix::minimax::rtxpf;
    const rtx::host::TensorInfo query{"BF16", {512, 6144}, 0, 0, 512U * 6144U * 2U};
    const rtx::host::TensorInfo key{"F32", {128, 6144}, 0, 0, 128U * 6144U * 4U};
    auto bad_query = query;
    bad_query.shape = {6144, 512};
    auto bad_key = key;
    --bad_key.bytes;
    auto unsupported = key;
    unsupported.dtype = "I32";
    const rtx::IndexProjection bad_stride{"bad_stride", 513, 512};
    if (!rtx::kIndexQuery.admits(query) || !rtx::kIndexKey.admits(key) ||
        rtx::kIndexQuery.admits(bad_query) || rtx::kIndexKey.admits(bad_key) ||
        rtx::kIndexKey.admits(unsupported) || bad_stride.admits(query)) {
        std::fputs("MiniMax RTX MMIDX weight admission mismatch\n", stderr);
        return 1;
    }
    std::puts("PASS MiniMax RTX MMIDX weight admission");

    const std::array prep_seqs{
        mm::msa_prep_sequence(geometry[0]), mm::msa_prep_sequence(geometry[1]), mm::msa_prep_sequence(geometry[2]),
    };
    const auto prep = mm::plan_msa_prep(46, cap, prep_seqs, 1, layers, cache.layer_stride);
    if (prep.heads != 73 || prep.shape.rotary_half != 32 || prep.layer != 1 ||
        prep.position(prep_seqs[1], 35) != 2169 ||
        prep.cache_head<true>(prep_seqs[1], 2169, 0) != cache.layer_stride + 2169U * width ||
        prep.cache_head<false>(prep_seqs[1], 2169, 3) != ((cap + 2169U) * 4U + 3U) * width ||
        mm::plan_msa_prep(0, 0, {}, 0, layers, 0).heads != 0) {
        std::fputs("MiniMax RTX MMPREP plan mismatch\n", stderr);
        return 1;
    }

    const auto refused = [](auto&& plan) {
        try {
            (void)plan();
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).starts_with("MMPREP refused MiniMax RTX");
        }
        return false;
    };
    if (!refused([&] { return mm::plan_msa_prep(45, cap, prep_seqs, 1, layers, cache.layer_stride); }) ||
        !refused([&] { return mm::plan_msa_prep(46, cap - 1, prep_seqs, 1, layers, cache.layer_stride); }) ||
        !refused([&] { return mm::plan_msa_prep(46, cap, prep_seqs, layers, layers, cache.layer_stride); }) ||
        !refused([&] { return mm::plan_msa_prep(46, cap, prep_seqs, 1, layers, cache.layer_stride - 1); })) {
        std::fputs("MiniMax RTX MMPREP accepted an invalid plan or omitted the refusal name\n", stderr);
        return 1;
    }
    static_assert(mm::admits_msa_prep_gain("F32", 512) && mm::admits_msa_prep_gain("BF16", 256) &&
                  mm::admits_msa_prep_gain("F16", 256) && !mm::admits_msa_prep_gain("F32", 513) &&
                  !mm::admits_msa_prep_gain("BF16", 254) && !mm::admits_msa_prep_gain("I16", 256));
    std::puts("PASS MiniMax RTX MMPREP plan and gain admission");

    std::uint16_t k = 0, v = 0, ik = 0;
    const auto attention_seq = mm::msa_attention_sequence({35, 9, cap, &k, &v, &ik, 2169});
    constexpr auto warp = mm::plan_msa_attention(46, true);
    constexpr auto staged = mm::plan_msa_attention(46, false);
    static_assert(mm::msa_attention_schedule(true) == strix::attn::SelectedBlockSchedule::warp &&
                  mm::msa_attention_schedule(false) == strix::attn::SelectedBlockSchedule::staged);
    static_assert(warp.grid_x == 46U * 8U && warp.grid_y == 1 && warp.threads == 256 && warp.shared_bytes == 0);
    static_assert(staged.grid_x == 46 && staged.grid_y == 4 && staged.threads == 512 && staged.shared_bytes == 74752);
    static_assert(mm::plan_msa_attention(0, true).grid_x == 0 && mm::plan_msa_attention(0, false).grid_x == 0);
    if (attention_seq.k != &k || attention_seq.v != &v || attention_seq.row0 != 35 ||
        attention_seq.pos0 != 2169 || attention_seq.capacity != cap) {
        std::fputs("MiniMax RTX MMATTN sequence mapping mismatch\n", stderr);
        return 1;
    }
    for (const bool legacy : {true, false}) {
        try {
            constexpr auto overflow = std::numeric_limits<unsigned>::max() / mm::kKvHeads + 1U;
            (void)mm::plan_msa_attention(overflow, legacy);
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()) == "MMATTN refused MiniMax RTX row extent") continue;
        }
        std::fputs("MiniMax RTX MMATTN did not refuse row overflow by name\n", stderr);
        return 1;
    }
    constexpr unsigned staged_only_rows = 0x20000000U;
    static_assert(mm::plan_msa_attention(staged_only_rows, false).grid_x == staged_only_rows);
    try {
        (void)mm::plan_msa_attention(staged_only_rows, true);
        std::fputs("MiniMax RTX MMATTN accepted a warp grid overflow\n", stderr);
        return 1;
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()) != "MMATTN refused MiniMax RTX launch geometry") return 1;
    }
    std::puts("PASS MiniMax RTX MMATTN plan and admission");
}
