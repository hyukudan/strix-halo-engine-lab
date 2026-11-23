#include "cuda/core/select_rows.cuh"
#include "select_fixture_common.cuh"
#include "select_qwen_baseline.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace {
namespace sel = strix::cuda::core::select;
namespace exec = strix::exec;
namespace qwen = select_qwen_baseline;
using k5c_fixture::check;
using k5c_fixture::compare;
using k5c_fixture::Device;
using k5c_fixture::poison;
using k5c_fixture::score_poison;
using k5c_fixture::values;

constexpr sel::RowStrides strides{520, 547, 2063};
constexpr unsigned score_offset = 2, pick_offset = 3;
constexpr sel::ScoreSpec score_spec{qwen::kTopK, qwen::kIndexHeads, qwen::kIndexDim, qwen::kRatio};
constexpr sel::PickSpec pick_spec{qwen::kTopK, qwen::kRatio, sel::KeyOrder::eligible_float};

template <class T>
std::vector<T> repack(const std::vector<T>& packed, std::size_t rows, std::size_t width,
                      std::size_t stride, std::size_t offset, T sentinel) {
    std::vector<T> result((rows + 1U) * stride, sentinel);
    for (std::size_t row = 0; row < rows; ++row)
        std::copy_n(packed.data() + row * width, width, result.data() + row * stride + offset);
    return result;
}

struct Batch {
    const unsigned rows, first;
    const std::vector<float> packed_query;
    Device<float> query, keys, scores;
    Device<unsigned> selected;

    Batch(unsigned count, unsigned position, unsigned seed)
        : rows(count), first(position), packed_query(values(rows * qwen::kIndexQWidth, seed)),
          query(repack(packed_query, rows, qwen::kIndexQWidth, strides.query, 4, score_poison)),
          keys(values((first + rows) / qwen::kRatio * qwen::kIndexDim, seed + 1U)),
          scores((rows + 1U) * strides.scores, score_poison),
          selected((rows + 1U) * strides.selected, poison) {}

    unsigned groups() const { return (first + rows) / qwen::kRatio; }

    sel::RowBases bases() const {
        return {query.data + 4, keys.data, scores.data + score_offset, selected.data + pick_offset};
    }

    void compare_scores_and_seed_pick() const {
        const Device<float> ref_query(packed_query);
        const Device<float> shuffle((rows + 1U) * strides.scores, score_poison);
        const Device<float> tree((rows + 1U) * strides.scores, score_poison);
        const unsigned waves = ((rows + 3U) / 4U) * ((groups() + 31U) / 32U);
        const auto score_stride = static_cast<unsigned>(strides.scores);
        qwen::block_score_kernel<<<(waves + 7U) / 8U, 256>>>(ref_query.data, rows, first, keys.data, groups(),
            shuffle.data + score_offset, score_stride);
        check(cudaGetLastError(), "frozen shuffle enqueue");
        qwen::block_score_tree_kernel<<<dim3((rows + 63U) / 64U, (groups() + 255U) / 256U), 256>>>(
            ref_query.data, rows, first, keys.data, groups(), tree.data + score_offset, score_stride);
        check(cudaGetLastError(), "frozen tree enqueue");
        compare("K5A shuffle score bytes and sentinels", shuffle, scores);
        compare("K5A tree score bytes and sentinels", tree, scores);

        auto input = scores.read();
        constexpr float minus_inf = -std::numeric_limits<float>::infinity();
        for (unsigned row = rows - 3U; row < rows; ++row) {
            auto* score = input.data() + row * strides.scores + score_offset;
            const unsigned visible = (first + row + 1U) / qwen::kRatio;
            if (row == rows - 3U) {
                for (unsigned i = 0; i < visible; ++i) score[i] = i % 2U == 0 ? -0.0F : 0.0F;
            } else {
                std::fill_n(score, visible, minus_inf);
                score[0] = std::bit_cast<float>(0x7fc12345U);
                if (row == rows - 1U) {
                    score[1] = -0.0F;
                    score[2] = 0.0F;
                    score[3] = std::numeric_limits<float>::infinity();
                }
            }
        }
        check(cudaMemcpy(scores.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "seed K5A pick edges");
    }

    void freeze_picks(Device<float>& ref_scores, Device<unsigned>& ref_ids) const {
        qwen::select_kernel<<<rows, qwen::kSelectThreads>>>(ref_scores.data + score_offset,
            static_cast<unsigned>(strides.scores), first, ref_ids.data);
        check(cudaGetLastError(), "frozen pick enqueue");
    }

    void compare_picks(const Device<float>& ref_scores, const Device<unsigned>& ref_ids) const {
        compare("K5A aliased score mask and sentinels", ref_scores, scores);
        compare("K5A selected IDs, tails and sentinels",
                repack(ref_ids.read(), rows, qwen::kMaxSelected, strides.selected, pick_offset, poison), selected.read());
    }
};

void affine_case() {
    Batch batch(71, 2046, 57);
    const sel::AffineRows rows{batch.bases(), strides, batch.first};
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree>(rows, batch.rows, batch.groups(), score_spec, nullptr),
          "affine score enqueue");
    batch.compare_scores_and_seed_pick();
    Device<float> mask(batch.scores.read());
    Device<unsigned> ids(batch.rows * qwen::kMaxSelected, poison);
    batch.freeze_picks(mask, ids);
    check(sel::enqueue_pick(rows, batch.rows, pick_spec, nullptr), "affine pick enqueue");
    batch.compare_picks(mask, ids);
    std::puts("K5A affine shuffle/tree scores, picks, masks and sentinels EXACT");
}

void slot_case() {
    Batch a(8, 2049, 11), b(7, 2070, 91);
    // Reversed slot order and interleaved X3 rows. The missing mask bit, missing
    // row flag and out-of-width row each point at an otherwise writable row.
    const Device table(std::vector{sel::SlotRowBase{b.bases(), b.first}, sel::SlotRowBase{a.bases(), a.first}});
    exec::StepBlock step{};
    const unsigned active = a.rows + b.rows;
    step.h.rows = active + 2U;
    step.h.valid_mask = (1U << active) - 1U;
    for (unsigned i = 0; i < active; ++i) {
        auto& row = step.row[i];
        const bool in_a = i % 2U == 0;
        row.position = (in_a ? a.first : b.first) + i / 2U;
        row.slot = static_cast<unsigned>(in_a);
        row.flags = exec::step_valid;
    }
    for (unsigned i = active; i < active + 3U; ++i) {
        step.row[i].position = a.first + a.rows;
        step.row[i].slot = 1;
    }
    step.row[active].flags = step.row[active + 2U].flags = exec::step_valid;
    step.h.valid_mask |= (1U << (active + 1U)) | (1U << (active + 2U));
    const Device descriptor(std::vector{step});
    const sel::SlotRows rows{descriptor.data, table.data, 2, strides};
    check(sel::enqueue_scores<sel::ScoreForm::relu_tree>(rows, active + 3U, std::max(a.groups(), b.groups()),
          score_spec, nullptr), "slot score enqueue");
    a.compare_scores_and_seed_pick();
    b.compare_scores_and_seed_pick();
    Device<float> a_mask(a.scores.read()), b_mask(b.scores.read());
    Device<unsigned> a_ids(a.rows * qwen::kMaxSelected, poison), b_ids(b.rows * qwen::kMaxSelected, poison);
    a.freeze_picks(a_mask, a_ids);
    b.freeze_picks(b_mask, b_ids);
    check(sel::enqueue_pick(rows, active + 3U, pick_spec, nullptr), "slot pick enqueue");
    a.compare_picks(a_mask, a_ids);
    b.compare_picks(b_mask, b_ids);
    std::puts("K5A X3 slot-indirect shuffle/tree scores, picks, masks and sentinels EXACT");
}
} // namespace

int main() {
    affine_case();
    slot_case();
}
